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
from torch_spyre._inductor.scratchpad.utils import _op_num_cores

LX_RELAYOUT_DESTINATION_PREFIX = "__spyre_lx_relayout_destination__"
_MATERIALIZED_COPIES_ATTR = "_spyre_lx_relayout_copies"
logger = get_inductor_logger("lx_relayout")


def _op_short_name(op: Operation) -> str:
    for fx_node in (getattr(op, "origin_node", None), *getattr(op, "origins", ())):
        target = getattr(fx_node, "target", None)
        for attr in ("_opname", "__name__", "name"):
            if name := getattr(target, attr, None):
                return str(name)
    return "None"


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
    consumer_is_matmul: bool = False
    destination_buffer_name: str | None = None
    source_lx_address: int | None = None
    destination_lx_address: int | None = None

    @property
    def destination_name(self) -> str:
        return self.destination_buffer_name or make_lx_relayout_destination_name(
            self.source_name, self.consumer_name
        )

    @property
    def edge_key(self) -> tuple[str, str]:
        return self.source_name, self.consumer_name


def materialized_lx_relayout_copies(
    graph: GraphLowering,
) -> dict[tuple[str, str], tuple[str, LXRelayoutPlan]]:
    """Return the exact private copies materialized for this graph."""

    return getattr(graph, _MATERIALIZED_COPIES_ATTR, {})


def discard_materialized_lx_relayouts(
    graph: GraphLowering, source_name: str
) -> set[str]:
    """Forget every materialized relayout destination connected to a source."""

    copies = materialized_lx_relayout_copies(graph)
    removed = set()
    for edge, (copy_name, _) in list(copies.items()):
        if edge[0] == source_name:
            removed.add(copy_name)
            del copies[edge]
    return removed


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

    producer_splits = _work_slice_dims(producer)
    consumer_splits = _work_slice_dims(consumer)
    dims = set(producer_splits) | set(consumer_splits)
    overlaps = {
        (producer_core, consumer_core)
        for producer_core, producer_slice in producer_core_map.items()
        for consumer_core, consumer_slice in consumer_core_map.items()
        if all(
            _intervals_overlap(
                int(producer_slice.get(dim, 0)),
                int(producer_splits.get(dim, 1)),
                int(consumer_slice.get(dim, 0)),
                int(consumer_splits.get(dim, 1)),
            )
            for dim in dims
        )
    }
    fanout = [
        sum(source == core for source, _ in overlaps) for core in range(num_cores)
    ]
    fanin = [sum(target == core for _, target in overlaps) for core in range(num_cores)]

    def unique_slices(core_map):
        return len({tuple(sorted(row.items())) for row in core_map.values()})

    return all(
        (
            overlaps,
            min(fanout, default=0) > 0,
            min(fanin, default=0) > 0,
            len(set(fanout)) == 1,
            len(set(fanin)) == 1,
            unique_slices(producer_core_map) == num_cores,
            unique_slices(consumer_core_map) == num_cores,
            math.prod(producer_splits.values()) == num_cores,
            math.prod(consumer_splits.values()) == num_cores,
        )
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
) -> dict[int, dict[int, int]] | None:
    """Return ownership as ``core -> device-dim -> slice-index``."""

    core_id = sympy.Symbol("core_id")
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    split_dims = {int(dim): int(split) for dim, split in view.work_slice_dims}
    result: dict[int, dict[int, int]] = {}

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
        result[core] = per_core

    return result


def _work_slice_dims(view: PerCoreView) -> dict[int, int]:
    return {int(dim): int(split) for dim, split in view.work_slice_dims}


def _has_materializable_iteration_geometry(
    device_coordinates: list[sympy.Expr],
    iteration_symbols: tuple[sympy.Symbol, ...],
    source: PerCoreView,
    destination: PerCoreView,
) -> bool:
    """Require every split device dimension to map to one loop symbol."""
    symbols = set(iteration_symbols)
    seen: dict[tuple[int, sympy.Symbol], int] = {}
    for side, view in enumerate((source, destination)):
        for device_dim, split in view.work_slice_dims:
            if split == 1:
                continue
            if device_dim < 0 or device_dim >= len(device_coordinates):
                return False
            coordinate_symbols = device_coordinates[device_dim].free_symbols & symbols
            if len(coordinate_symbols) != 1:
                return False
            key = (side, next(iter(coordinate_symbols)))
            previous = seen.setdefault(key, int(split))
            if previous != split:
                return False
    return True


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        layout = getattr(op, "layout", None)
        if layout is not None and hasattr(layout, "lx_view"):
            layout.lx_view = None
            layout.lx_consumer_is_matmul = False


_DIRECT_EDGE = object()


def _plan_consumer_edge(
    operations: dict[str, Operation],
    producer: ComputedBuffer,
    source_name: str,
    source_view: PerCoreView,
    num_cores: int,
    consumer: Operation,
    dep: MemoryDep,
    read_index: int,
    cache: dict | None,
) -> LXRelayoutPlan | object | None:
    if not isinstance(consumer, ComputedBuffer) or consumer is producer:
        return None
    reads = [d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)]
    if any(d.is_indirect() for d in reads):
        return None

    view, has_partial, representable = _per_core_view_on_buf(
        consumer, dep, source_name, cache
    )
    consumer_cores = _op_num_cores(consumer)
    if has_partial or not representable or consumer_cores != num_cores:
        return None
    if view == source_view:
        return _DIRECT_EDGE

    is_matmul = _is_matmul_op(consumer)
    if not is_matmul and not isinstance(consumer.data, Pointwise):
        return None
    if is_matmul and (
        not _is_activation_source(operations, producer) or read_index not in (0, 1)
    ):
        return None
    if not _is_equal_footprint_geometry(source_view, view, num_cores):
        return None

    coordinates = try_device_coordinates(producer.layout.device_layout, dep, None)
    if coordinates is None or not _has_materializable_iteration_geometry(
        coordinates,
        tuple(iteration_space_from_op(consumer)),
        source_view,
        view,
    ):
        return None
    return LXRelayoutPlan(
        source_name,
        consumer.get_name(),
        source_view,
        view,
        num_cores,
        is_matmul,
    )


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

    materialized = materialized_lx_relayout_copies(graph)
    if materialized:
        restored = []
        for (source_name, _), (copy_name, original) in materialized.items():
            source = graph.try_get_buffer(source_name)
            copy = graph.try_get_buffer(copy_name)
            source_layout = getattr(source, "layout", None)
            copy_layout = getattr(copy, "layout", None)
            if source_layout is None or copy_layout is None:
                return []
            source_layout.lx_view = original.source_view
            copy_layout.lx_view = original.destination_view
            copy_layout.lx_consumer_is_matmul = original.consumer_is_matmul
            restored.append(
                dataclasses.replace(
                    original,
                    consumer_name=copy_name,
                    destination_buffer_name=copy_name,
                    source_lx_address=None,
                    destination_lx_address=None,
                )
            )
        return restored

    operations = {op.get_name(): op for op in graph.operations}
    reads_by_source: dict[str, list[tuple[Operation, MemoryDep, int]]] = {}
    for consumer in graph.operations:
        memory_reads = [
            dep for dep in op_read_writes(consumer).reads if isinstance(dep, MemoryDep)
        ]
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

        source_plans: list[LXRelayoutPlan] = []
        consumers: set[str] = set()
        for consumer, dep, read_index in read_edges:
            consumer_name = consumer.get_name()
            result = _plan_consumer_edge(
                operations,
                producer,
                source_name,
                producer_view,
                producer_core_count,
                consumer,
                dep,
                read_index,
                cache,
            )
            if result is None or consumer_name in consumers:
                break
            consumers.add(consumer_name)
            if isinstance(result, LXRelayoutPlan):
                source_plans.append(result)
        else:
            plans.extend(source_plans)

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
    copies = materialized_lx_relayout_copies(graph)
    setattr(graph, _MATERIALIZED_COPIES_ATTR, copies)
    for plan in plans:
        source = graph.try_get_buffer(plan.source_name)
        assert isinstance(source, ComputedBuffer)
        assert plan.source_lx_address is not None
        assert plan.destination_lx_address is not None

        copy_name = plan.destination_buffer_name
        if copy_name is not None:
            copy = graph.try_get_buffer(copy_name)
            assert isinstance(copy, ComputedBuffer)
            destination_layout = copy.get_layout()
            assert isinstance(destination_layout, FixedTiledLayout)
        else:
            consumer = graph.try_get_buffer(plan.consumer_name)
            assert isinstance(consumer, ComputedBuffer)
            template = source.get_layout()
            assert isinstance(template, FixedTiledLayout)
            destination_layout = FixedTiledLayout(
                template.device,
                template.dtype,
                list(template.size),
                list(template.stride),
                template.device_layout,
            )
            copy = editor.insert_clone_before_consumer(
                source, consumer, destination_layout
            )
            copies[plan.edge_key] = (copy.get_name(), plan)

        source_layout = source.get_layout()
        assert isinstance(source_layout, FixedTiledLayout)
        destination_layout.allocation["lx"] = plan.destination_lx_address
        source_layout.lx_view = plan.source_view
        destination_layout.lx_view = plan.destination_view
        destination_layout.lx_consumer_is_matmul = plan.consumer_is_matmul
        logger.debug(
            "accepted LX relayout %s -> %s: source_view=%s source_lx=%d "
            "destination_view=%s destination_lx=%d",
            plan.source_name,
            copy.get_name(),
            plan.source_view,
            plan.source_lx_address,
            plan.destination_view,
            plan.destination_lx_address,
        )
