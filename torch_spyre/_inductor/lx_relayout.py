# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan LX relayouts from producer and consumer per-core views.

The regular LX planner handles same-core scratchpad persistence. This module
finds edges where a producer and consumer use different per-core views of the
same tensor. Accepted edges are materialized as an explicit, frontend-allocated
S1 -> SHUFFLE -> S2 sequence.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise
from torch_spyre._C import SpyreTensorLayout

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    iteration_space_from_op,
    op_read_writes,
    try_device_coordinates,
)
from torch_spyre._inductor.scratchpad.utils import _op_num_cores, _op_short_name

LX_RELAYOUT_ATTR = "_spyre_lx_relayout_inputs"
LX_RELAYOUT_SHUFFLE_ATTR = "_spyre_lx_relayout_shuffle"
LX_RELAYOUT_DESTINATION_PREFIX = "__spyre_lx_relayout_destination__"

logger = get_inductor_logger("lx_relayout")


def make_lx_relayout_destination_name(source_name: str, consumer_name: str) -> str:
    return f"{LX_RELAYOUT_DESTINATION_PREFIX}:{source_name}:{consumer_name}"


def is_lx_relayout_destination(name: str) -> bool:
    return name.startswith(f"{LX_RELAYOUT_DESTINATION_PREFIX}:")


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    """Names and exact per-core geometry for one LX shuffle."""

    source_name: str
    consumer_name: str
    source_core_id_to_device_slice: dict[str, dict[str, int]]
    destination_core_id_to_device_slice: dict[str, dict[str, int]]
    source_device_dim_splits: dict[str, int]
    destination_device_dim_splits: dict[str, int]
    shuffle_iteration_symbols: tuple[sympy.Symbol, ...]
    device_dim_to_iteration_symbol: dict[str, sympy.Symbol]
    consumer_is_matmul: bool = False
    source_lx_address: int | None = None
    destination_lx_address: int | None = None
    materialized_destination_name: str | None = None

    @property
    def destination_name(self) -> str:
        return make_lx_relayout_destination_name(self.source_name, self.consumer_name)

    @property
    def edge_key(self) -> tuple[str, str]:
        return self.source_name, self.consumer_name


def _intervals_overlap(
    lhs_slot: int, lhs_split: int, rhs_slot: int, rhs_split: int
) -> bool:
    return lhs_slot * rhs_split < (rhs_slot + 1) * lhs_split and (
        rhs_slot * lhs_split < (lhs_slot + 1) * rhs_split
    )


def _is_equal_footprint_geometry(
    producer_map: dict[str, dict[str, int]],
    producer_splits: dict[str, int],
    consumer_map: dict[str, dict[str, int]],
    consumer_splits: dict[str, int],
) -> bool:
    """Whether two complete per-core partitions admit a bounded shuffle."""

    if producer_map.keys() != consumer_map.keys():
        return False

    def slice_count(core_map: dict[str, dict[str, int]]) -> int:
        return len(
            {
                tuple(sorted((dim, int(slot)) for dim, slot in per_core.items()))
                for per_core in core_map.values()
            }
        )

    fanout = {core: 0 for core in producer_map}
    fanin = {core: 0 for core in consumer_map}
    transfer_count = 0
    dims = set(producer_splits) | set(consumer_splits)
    for producer_core, producer_slice in producer_map.items():
        for consumer_core, consumer_slice in consumer_map.items():
            if all(
                _intervals_overlap(
                    int(producer_slice.get(dim, 0)),
                    int(producer_splits.get(dim, 1)),
                    int(consumer_slice.get(dim, 0)),
                    int(consumer_splits.get(dim, 1)),
                )
                for dim in dims
            ):
                fanout[producer_core] += 1
                fanin[consumer_core] += 1
                transfer_count += 1

    fanout_values = set(fanout.values())
    fanin_values = set(fanin.values())
    uniform = len(fanout_values) == 1 and len(fanin_values) == 1
    covers_all_cores = (
        min(fanout_values, default=0) > 0 and min(fanin_values, default=0) > 0
    )
    producer_is_partitioned = slice_count(producer_map) == len(
        producer_map
    ) and math.prod(producer_splits.values()) == len(producer_map)
    consumer_is_partitioned = slice_count(consumer_map) == len(
        consumer_map
    ) and math.prod(consumer_splits.values()) == len(consumer_map)
    return (
        transfer_count > 0
        and covers_all_cores
        and uniform
        and producer_is_partitioned
        and consumer_is_partitioned
    )


def _single_write_dep(op: ComputedBuffer, buf_name: str) -> MemoryDep | None:
    matches = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and dep.name == buf_name
    ]
    return matches[0] if len(matches) == 1 else None


def _operations_by_name(graph: GraphLowering) -> dict[str, Operation]:
    """Index the current graph, including operations inserted by graph editing."""

    return {op.get_name(): op for op in graph.operations}


def _restickify_reads_computed_input(
    operations: dict[str, Operation], op: Operation
) -> bool:
    if _op_short_name(op) != "restickify":
        return False
    return any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def _matmul_operand_source_good_for_lx_relayout(
    operations: dict[str, Operation], op: Operation
) -> bool:
    """Exclude graph-input and weight restickifies from activation relayout."""

    return _op_short_name(op) != "restickify" or _restickify_reads_computed_input(
        operations, op
    )


def _core_id_to_device_slice(
    view: PerCoreView,
    core_count: int,
) -> dict[str, dict[str, int]] | None:
    """Return ownership as ``core -> device-dim -> slice-index``."""

    core_id = sympy.Symbol("core_id")
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    split_dims = {int(dim): int(split) for dim, split in view.work_slice_dims}
    result: dict[str, dict[str, int]] = {}

    for core in range(core_count):
        per_core: dict[str, int] = {}
        for dim, split in split_dims.items():
            expr = sympy.sympify(expr_by_dim.get(dim, 0))
            slot = sympy.simplify(expr.subs(core_id, core))
            if getattr(slot, "free_symbols", None):
                return None
            try:
                slot_int = int(slot)
            except TypeError:
                return None
            if slot_int < 0 or slot_int >= split:
                return None
            per_core[str(dim)] = slot_int
        result[str(core)] = per_core

    return result


def _work_slice_dims(view: PerCoreView) -> dict[str, int]:
    return {str(int(dim)): int(split) for dim, split in view.work_slice_dims}


def _materializable_iteration_geometry(
    device_coordinates: list[sympy.Expr],
    iteration_symbols: tuple[sympy.Symbol, ...],
    source_device_dim_splits: dict[str, int],
    destination_device_dim_splits: dict[str, int],
) -> tuple[tuple[sympy.Symbol, ...], dict[str, sympy.Symbol]] | None:
    """Resolve every non-unit allocation dimension before LX is committed.

    The allocator cannot safely reserve S1/S2 from device-dimension indices
    alone: codegen ultimately needs iteration symbols to emit the SHUFFLE.
    Return that complete symbol contract here, while fallback to HBM is still
    possible.  Unit dimensions may be absent from an access; their only legal
    slot is zero and the SDSC mapper already treats them as implicit.
    """

    iteration_symbol_set = set(iteration_symbols)
    used_symbols = tuple(
        symbol
        for symbol in iteration_symbols
        if any(symbol in coordinate.free_symbols for coordinate in device_coordinates)
    )
    if not used_symbols:
        return None

    device_dim_to_symbol: dict[str, sympy.Symbol] = {}
    symbol_splits: dict[tuple[str, sympy.Symbol], int] = {}
    device_dims = set(source_device_dim_splits) | set(destination_device_dim_splits)
    for device_dim in device_dims:
        try:
            index = int(device_dim)
        except ValueError:
            return None
        source_split = int(source_device_dim_splits.get(device_dim, 1))
        destination_split = int(destination_device_dim_splits.get(device_dim, 1))
        requires_symbol = source_split != 1 or destination_split != 1
        if index < 0 or index >= len(device_coordinates):
            if requires_symbol:
                return None
            continue
        symbols = [
            symbol
            for symbol in device_coordinates[index].free_symbols
            if symbol in iteration_symbol_set
        ]
        if len(symbols) != 1:
            if requires_symbol:
                return None
            continue
        symbol = symbols[0]
        device_dim_to_symbol[device_dim] = symbol
        for side, split in (
            ("source", source_split),
            ("destination", destination_split),
        ):
            key = (side, symbol)
            previous = symbol_splits.get(key)
            if previous is not None and previous != split:
                return None
            symbol_splits[key] = split

    if any(
        int(split) != 1 and device_dim not in device_dim_to_symbol
        for splits in (source_device_dim_splits, destination_device_dim_splits)
        for device_dim, split in splits.items()
    ):
        return None
    return used_symbols, device_dim_to_symbol


def _dense_view(
    core_map: dict[str, dict[str, int]],
    splits: dict[str, int],
    dims: set[str],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    ordered_dims = sorted(dims, key=int)
    dense_map = {
        core: {dim: int(per_core.get(dim, 0)) for dim in ordered_dims}
        for core, per_core in core_map.items()
    }
    dense_splits = {dim: int(splits.get(dim, 1)) for dim in ordered_dims}
    return dense_map, dense_splits


def per_core_view_matches_lx_relayout_side(
    op: Operation,
    view: PerCoreView,
    plan: LXRelayoutPlan,
    *,
    destination: bool,
) -> bool:
    """Check a final scheduled access against one side of a relayout plan.

    Planning records physical device-dimension ownership before the scheduler
    exists.  Post-fusion validation must compare that contract with the final
    positional core mapping, not merely with the split-factor product.
    """

    expected_map = (
        plan.destination_core_id_to_device_slice
        if destination
        else plan.source_core_id_to_device_slice
    )
    expected_splits = (
        plan.destination_device_dim_splits
        if destination
        else plan.source_device_dim_splits
    )
    core_count = _op_num_cores(op)
    if core_count != len(expected_map):
        return False
    current_map = _core_id_to_device_slice(view, core_count)
    if current_map is None:
        return False
    current_map, current_splits = _dense_view(
        current_map,
        _work_slice_dims(view),
        set(expected_splits),
    )
    return current_map == expected_map and current_splits == expected_splits


def rebind_lx_relayout_iteration_geometry(
    plan: LXRelayoutPlan,
    stl: SpyreTensorLayout,
    dep: MemoryDep,
    iteration_symbols: tuple[sympy.Symbol, ...],
) -> LXRelayoutPlan | None:
    """Rebind a pre-scheduling plan to the scheduler's final loop symbols.

    The allocation-facing ownership is expressed in stable device dimensions,
    while Inductor is free to rename loop symbols during scheduling.  Rebinding
    happens in the post-fusion safety pass; failure there demotes the source to
    HBM before codegen, so symbol drift cannot become a late assertion.
    """

    device_coordinates = try_device_coordinates(stl, dep, None)
    if device_coordinates is None:
        return None
    geometry = _materializable_iteration_geometry(
        device_coordinates,
        iteration_symbols,
        plan.source_device_dim_splits,
        plan.destination_device_dim_splits,
    )
    if geometry is None:
        return None
    shuffle_iteration_symbols, device_dim_to_iteration_symbol = geometry
    return dataclasses.replace(
        plan,
        shuffle_iteration_symbols=shuffle_iteration_symbols,
        device_dim_to_iteration_symbol=device_dim_to_iteration_symbol,
    )


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_ATTR):
            delattr(op, LX_RELAYOUT_ATTR)
        if hasattr(op, LX_RELAYOUT_SHUFFLE_ATTR):
            delattr(op, LX_RELAYOUT_SHUFFLE_ATTR)


def collect_lx_relayout_plans(
    graph: GraphLowering, cache: dict | None = None
) -> list[LXRelayoutPlan]:
    """Plan bounded LX relayouts from producer and consumer coordinates.

    Only records movement for single-writer intermediates whose producer output
    is final (not K-split partials). Every read must either have the producer's
    exact PerCoreView or receive its own materializable relayout destination;
    otherwise all plans for that source are rejected so LX eligibility cannot
    hide an uncovered core-division mismatch.
    """

    if not config.lx_planner_relayout:
        return []

    operations = _operations_by_name(graph)
    reads_by_source: dict[str, list[tuple[Operation, MemoryDep, int]]] = {}
    indirect_consumers: set[str] = set()
    for consumer in graph.operations:
        memory_reads = [
            dep for dep in op_read_writes(consumer).reads if isinstance(dep, MemoryDep)
        ]
        if any(dep.is_indirect() for dep in memory_reads):
            indirect_consumers.add(consumer.get_name())
        for read_index, dep in enumerate(memory_reads):
            reads_by_source.setdefault(dep.name, []).append((consumer, dep, read_index))
    plans: list[LXRelayoutPlan] = []

    for source_name, read_edges in reads_by_source.items():
        producer = operations.get(source_name)
        if not isinstance(producer, ComputedBuffer):
            continue
        write_dep = _single_write_dep(producer, source_name)
        if write_dep is None:
            continue

        producer_view, producer_has_partial, producer_representable = (
            _per_core_view_on_buf(producer, write_dep, source_name, cache)
        )
        if producer_has_partial or not producer_representable:
            continue
        producer_core_count = _op_num_cores(producer)
        producer_core_slices = _core_id_to_device_slice(
            producer_view, producer_core_count
        )
        if producer_core_slices is None:
            continue
        producer_work_slice_dims = _work_slice_dims(producer_view)

        source_plans: dict[tuple[str, str], LXRelayoutPlan] = {}
        direct_consumers: set[str] = set()
        source_is_covered = True
        for consumer, dep, read_index in read_edges:
            if not isinstance(consumer, ComputedBuffer) or producer is consumer:
                source_is_covered = False
                break
            # Indirect indices are separate TensorArgs in codegen and cannot be
            # rebound to a relayout destination without also rewriting the
            # gather/scatter index contract.  Keep the whole consumer on its
            # normal fallback path instead of stamping a plan that codegen
            # cannot faithfully materialize.
            if consumer.get_name() in indirect_consumers:
                source_is_covered = False
                break
            consumer_view, consumer_has_partial, consumer_representable = (
                _per_core_view_on_buf(consumer, dep, source_name, cache)
            )
            if consumer_has_partial or not consumer_representable:
                source_is_covered = False
                break
            consumer_core_count = _op_num_cores(consumer)
            if (
                producer_view == consumer_view
                and producer_core_count == consumer_core_count
            ):
                consumer_name = consumer.get_name()
                if (source_name, consumer_name) in source_plans:
                    source_is_covered = False
                    break
                direct_consumers.add(consumer_name)
                continue

            is_matmul_consumer = _is_matmul_op(consumer)
            if not is_matmul_consumer and not isinstance(consumer.data, Pointwise):
                source_is_covered = False
                break
            if is_matmul_consumer and (
                not _matmul_operand_source_good_for_lx_relayout(operations, producer)
                or read_index not in (0, 1)
            ):
                source_is_covered = False
                break

            consumer_work_slice_dims = _work_slice_dims(consumer_view)
            consumer_core_slices = _core_id_to_device_slice(
                consumer_view, consumer_core_count
            )
            if consumer_core_slices is None:
                source_is_covered = False
                break

            relayout_dims = set(producer_work_slice_dims) | set(
                consumer_work_slice_dims
            )
            dense_producer_core_slices, dense_producer_work_slice_dims = _dense_view(
                producer_core_slices, producer_work_slice_dims, relayout_dims
            )
            dense_consumer_core_slices, dense_consumer_work_slice_dims = _dense_view(
                consumer_core_slices, consumer_work_slice_dims, relayout_dims
            )
            if not _is_equal_footprint_geometry(
                dense_producer_core_slices,
                dense_producer_work_slice_dims,
                dense_consumer_core_slices,
                dense_consumer_work_slice_dims,
            ):
                source_is_covered = False
                break

            device_coordinates = try_device_coordinates(
                producer.layout.device_layout,
                dep,
                None,
            )
            if device_coordinates is None:
                source_is_covered = False
                break
            iteration_geometry = _materializable_iteration_geometry(
                device_coordinates,
                tuple(iteration_space_from_op(consumer)),
                dense_producer_work_slice_dims,
                dense_consumer_work_slice_dims,
            )
            if iteration_geometry is None:
                source_is_covered = False
                break
            shuffle_iteration_symbols, device_dim_to_iteration_symbol = (
                iteration_geometry
            )

            plan = LXRelayoutPlan(
                source_name=source_name,
                consumer_name=consumer.get_name(),
                source_core_id_to_device_slice=dense_producer_core_slices,
                destination_core_id_to_device_slice=dense_consumer_core_slices,
                source_device_dim_splits=dense_producer_work_slice_dims,
                destination_device_dim_splits=dense_consumer_work_slice_dims,
                shuffle_iteration_symbols=shuffle_iteration_symbols,
                device_dim_to_iteration_symbol=device_dim_to_iteration_symbol,
                consumer_is_matmul=is_matmul_consumer,
            )
            if plan.consumer_name in direct_consumers:
                source_is_covered = False
                break
            previous = source_plans.get(plan.edge_key)
            if previous is not None and previous != plan:
                source_is_covered = False
                break
            source_plans[plan.edge_key] = plan

        if source_is_covered:
            plans.extend(source_plans.values())

    if logger.isEnabledFor(logging.DEBUG):
        if plans:
            logger.debug(
                "final LX relayout plan:\n%s",
                "\n".join(
                    "  %s -> %s: source_splits=%s destination_splits=%s"
                    % (
                        plan.source_name,
                        plan.consumer_name,
                        plan.source_device_dim_splits,
                        plan.destination_device_dim_splits,
                    )
                    for plan in plans
                ),
            )
        else:
            logger.debug("final LX relayout plan: (none)")

    return plans


def record_lx_relayout_plan(graph: GraphLowering, plan: LXRelayoutPlan) -> None:
    """Stamp a relayout plan after the source buffer is placed in LX."""

    consumer = _operations_by_name(graph)[plan.consumer_name]
    plans = dict(getattr(consumer, LX_RELAYOUT_ATTR, {}))
    plans[plan.source_name] = plan
    setattr(consumer, LX_RELAYOUT_ATTR, plans)


def materialize_lx_relayout_operations(graph: GraphLowering) -> None:
    """Insert accepted LX relayouts as real graph operations.

    Scratchpad planning first proves that both S1 and S2 fit atomically and
    records their addresses on the consumer.  This pass turns each accepted
    edge into a private identity-copy node before scheduling; codegen recognizes
    the node's relayout metadata and lowers it to SHUFFLE/STCDPOpLx.
    """

    from torch_spyre._inductor.ir import FixedTiledLayout
    from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor

    pending = [
        plan
        for consumer in graph.operations
        for plan in getattr(consumer, LX_RELAYOUT_ATTR, {}).values()
    ]
    if not pending:
        return

    editor = GraphEditor(graph)
    for plan in pending:
        source = graph.try_get_buffer(plan.source_name)
        consumer = graph.try_get_buffer(plan.consumer_name)
        assert isinstance(source, ComputedBuffer)
        assert isinstance(consumer, ComputedBuffer)
        assert plan.source_lx_address is not None
        assert plan.destination_lx_address is not None

        # GraphLowering's pre-scheduling pipeline can be entered more than once
        # by nested Inductor compilation. Re-plan the already-materialized copy
        # in place instead of growing another clone in front of it.
        if getattr(consumer, "operation_name", None) == "lx_relayout_shuffle":
            reads = [
                dep
                for dep in op_read_writes(consumer).reads
                if isinstance(dep, MemoryDep)
            ]
            assert any(dep.name == plan.source_name for dep in reads)
            destination_layout = consumer.get_layout()
            assert isinstance(destination_layout, FixedTiledLayout)
            destination_layout.allocation["lx"] = plan.destination_lx_address
            materialized_plan = dataclasses.replace(
                plan,
                materialized_destination_name=consumer.get_name(),
            )
            setattr(consumer, LX_RELAYOUT_SHUFFLE_ATTR, materialized_plan)
            logger.debug(
                "updated LX relayout %s -> %s at %#x -> %#x",
                plan.source_name,
                consumer.get_name(),
                plan.source_lx_address,
                plan.destination_lx_address,
            )
            continue

        source_layout = source.get_layout()
        assert isinstance(source_layout, FixedTiledLayout)
        destination_layout = FixedTiledLayout(
            source_layout.device,
            source_layout.dtype,
            source_layout.size,
            source_layout.stride,
            source_layout.device_layout,
        )
        destination_layout.allocation["lx"] = plan.destination_lx_address

        shuffle = editor.insert_clone_before_consumer(
            source,
            consumer,
            destination_layout,
        )
        shuffle.operation_name = "lx_relayout_shuffle"
        materialized_plan = dataclasses.replace(
            plan,
            materialized_destination_name=shuffle.get_name(),
        )
        setattr(shuffle, LX_RELAYOUT_SHUFFLE_ATTR, materialized_plan)

        logger.debug(
            "materialized LX relayout %s -> %s -> %s at %#x -> %#x",
            plan.source_name,
            shuffle.get_name(),
            plan.consumer_name,
            plan.source_lx_address,
            plan.destination_lx_address,
        )

    # The consumer metadata was a deferred insertion queue.  Once real graph
    # nodes exist it must not reach codegen as a second, late materialization.
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_ATTR):
            delattr(op, LX_RELAYOUT_ATTR)
