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
same tensor. Accepted edges are materialized as ordinary pointwise copies;
SDSC codegen selects the LX transport from the two tensor work divisions.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise
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

LX_RELAYOUT_DESTINATION_PREFIX = "__spyre_lx_relayout_destination__"

logger = get_inductor_logger("lx_relayout")


def make_lx_relayout_destination_name(source_name: str, consumer_name: str) -> str:
    return f"{LX_RELAYOUT_DESTINATION_PREFIX}:{source_name}:{consumer_name}"


def is_lx_relayout_destination(name: str) -> bool:
    return name.startswith(f"{LX_RELAYOUT_DESTINATION_PREFIX}:")


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    """Names and exact per-core geometry for one LX copy."""

    source_name: str
    consumer_name: str
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    destination_is_kernel_operand: bool = False
    source_lx_address: int | None = None
    destination_lx_address: int | None = None

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
    producer: PerCoreView,
    consumer: PerCoreView,
    num_cores: int,
) -> bool:
    """Whether two complete per-core partitions admit a bounded peer copy."""

    producer_core_map = _core_id_to_device_slice(producer, num_cores)
    consumer_core_map = _core_id_to_device_slice(consumer, num_cores)
    if producer_core_map is None or consumer_core_map is None:
        return False

    def slice_count(core_map: dict[str, dict[int, int]]) -> int:
        return len(
            {
                tuple(sorted((dim, int(slot)) for dim, slot in per_core.items()))
                for per_core in core_map.values()
            }
        )

    producer_splits = _work_slice_dims(producer)
    consumer_splits = _work_slice_dims(consumer)
    fanout = {core: 0 for core in producer_core_map}
    fanin = {core: 0 for core in consumer_core_map}
    transfer_count = 0
    dims = set(producer_splits) | set(consumer_splits)
    for producer_core, producer_slice in producer_core_map.items():
        for consumer_core, consumer_slice in consumer_core_map.items():
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
    producer_is_partitioned = slice_count(producer_core_map) == num_cores and (
        math.prod(producer_splits.values()) == num_cores
    )
    consumer_is_partitioned = slice_count(consumer_core_map) == num_cores and (
        math.prod(consumer_splits.values()) == num_cores
    )
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


def _is_activation_source(operations: dict[str, Operation], op: Operation) -> bool:
    """Exclude restickified graph inputs and weights from activation relayout."""

    return _op_short_name(op) != "restickify" or any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def _core_id_to_device_slice(
    view: PerCoreView,
    core_count: int,
) -> dict[str, dict[int, int]] | None:
    """Return ownership as ``core -> device-dim -> slice-index``."""

    core_id = sympy.Symbol("core_id")
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    split_dims = {int(dim): int(split) for dim, split in view.work_slice_dims}
    result: dict[str, dict[int, int]] = {}

    for core in range(core_count):
        per_core: dict[int, int] = {}
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
            per_core[dim] = slot_int
        result[str(core)] = per_core

    return result


def _work_slice_dims(view: PerCoreView) -> dict[int, int]:
    return {int(dim): int(split) for dim, split in view.work_slice_dims}


def _has_materializable_iteration_geometry(
    device_coordinates: list[sympy.Expr],
    iteration_symbols: tuple[sympy.Symbol, ...],
    source: PerCoreView,
    destination: PerCoreView,
) -> bool:
    """Check every non-unit allocation dimension before LX is committed.

    The allocator cannot safely reserve S1/S2 from device-dimension indices
    alone: codegen needs those dimensions to map to iteration symbols. Unit
    dimensions may be absent; their only legal slot is zero.
    """

    iteration_symbol_set = set(iteration_symbols)
    if not any(
        coordinate.free_symbols & iteration_symbol_set
        for coordinate in device_coordinates
    ):
        return False

    symbol_splits: dict[tuple[str, sympy.Symbol], int] = {}
    source_splits = _work_slice_dims(source)
    destination_splits = _work_slice_dims(destination)
    device_dims = set(source_splits) | set(destination_splits)
    for device_dim in device_dims:
        index = int(device_dim)
        source_split = int(source_splits.get(device_dim, 1))
        destination_split = int(destination_splits.get(device_dim, 1))
        requires_symbol = source_split != 1 or destination_split != 1
        if index < 0 or index >= len(device_coordinates):
            if requires_symbol:
                return False
            continue
        symbols = [
            symbol
            for symbol in device_coordinates[index].free_symbols
            if symbol in iteration_symbol_set
        ]
        if len(symbols) != 1:
            if requires_symbol:
                return False
            continue
        symbol = symbols[0]
        for side, split in (
            ("source", source_split),
            ("destination", destination_split),
        ):
            key = (side, symbol)
            previous = symbol_splits.get(key)
            if previous is not None and previous != split:
                return False
            symbol_splits[key] = split

    return True


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        layout = getattr(op, "layout", None)
        if layout is not None and hasattr(layout, "lx_view"):
            layout.lx_view = None
            layout.lx_is_kernel_operand = False


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

    operations = {op.get_name(): op for op in graph.operations}
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
        if _core_id_to_device_slice(producer_view, producer_core_count) is None:
            continue

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
                not _is_activation_source(operations, producer)
                or read_index not in (0, 1)
            ):
                source_is_covered = False
                break

            if (
                consumer_core_count != producer_core_count
                or _core_id_to_device_slice(consumer_view, consumer_core_count) is None
            ):
                source_is_covered = False
                break

            if not _is_equal_footprint_geometry(
                producer_view,
                consumer_view,
                producer_core_count,
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
            if not _has_materializable_iteration_geometry(
                device_coordinates,
                tuple(iteration_space_from_op(consumer)),
                producer_view,
                consumer_view,
            ):
                source_is_covered = False
                break

            plan = LXRelayoutPlan(
                source_name=source_name,
                consumer_name=consumer.get_name(),
                source_view=producer_view,
                destination_view=consumer_view,
                num_cores=producer_core_count,
                destination_is_kernel_operand=is_matmul_consumer,
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
                        plan.source_view.work_slice_dims,
                        plan.destination_view.work_slice_dims,
                    )
                    for plan in plans
                ),
            )
        else:
            logger.debug("final LX relayout plan: (none)")

    return plans


def materialize_lx_relayouts(graph: GraphLowering, plans: list[LXRelayoutPlan]) -> None:
    """Insert accepted LX relayouts as real graph operations.

    Scratchpad planning first proves that both S1 and S2 fit atomically. This
    pass turns each accepted edge into a private identity-copy node before
    scheduling. SDSC codegen selects SHUFFLE from the copy's tensor work
    divisions.
    """

    from torch_spyre._inductor.ir import FixedTiledLayout
    from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor

    if not plans:
        return

    editor = GraphEditor(graph)
    for plan in plans:
        source = graph.try_get_buffer(plan.source_name)
        consumer = graph.try_get_buffer(plan.consumer_name)
        assert isinstance(source, ComputedBuffer)
        assert isinstance(consumer, ComputedBuffer)
        assert plan.source_lx_address is not None
        assert plan.destination_lx_address is not None

        # GraphLowering's pre-scheduling pipeline can be entered more than once
        # by nested Inductor compilation. Re-plan the already-materialized copy
        # in place instead of growing another clone in front of it.
        if getattr(consumer, "operation_name", None) == "lx_relayout_copy":
            reads = [
                dep
                for dep in op_read_writes(consumer).reads
                if isinstance(dep, MemoryDep)
            ]
            assert any(dep.name == plan.source_name for dep in reads)
            destination_layout = consumer.get_layout()
            assert isinstance(destination_layout, FixedTiledLayout)
            destination_layout.allocation["lx"] = plan.destination_lx_address
            source_layout = source.get_layout()
            assert isinstance(source_layout, FixedTiledLayout)
            source_layout.lx_view = plan.source_view
            destination_layout.lx_view = plan.destination_view
            destination_layout.lx_is_kernel_operand = plan.destination_is_kernel_operand
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
        source_layout.lx_view = plan.source_view
        destination_layout.lx_view = plan.destination_view
        destination_layout.lx_is_kernel_operand = plan.destination_is_kernel_operand

        copy = editor.insert_clone_before_consumer(
            source,
            consumer,
            destination_layout,
        )
        copy.operation_name = "lx_relayout_copy"

        logger.debug(
            "materialized LX relayout %s -> %s -> %s at %#x -> %#x",
            plan.source_name,
            copy.get_name(),
            plan.consumer_name,
            plan.source_lx_address,
            plan.destination_lx_address,
        )
