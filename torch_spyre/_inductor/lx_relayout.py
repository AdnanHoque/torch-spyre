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

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from typing import cast

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
    Pointwise,
)

from . import config
from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger
from .op_spec import TensorWorkDivision
from .pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    iteration_space_from_op,
    op_read_writes,
    try_device_coordinates,
)
from .scratchpad.utils import _op_num_cores

logger = get_inductor_logger("lx_relayout")
_DESTINATION_PREFIX = "__spyre_lx_relayout__"
_REGISTRY = "_spyre_lx_relayout_copies"


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    consumer_name: str
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    consumer_is_matmul: bool = False
    destination_buffer_name: str | None = None
    source_address: int | None = None
    destination_address: int | None = None

    @property
    def destination_name(self) -> str:
        return self.destination_buffer_name or (
            f"{_DESTINATION_PREFIX}:{self.source_name}:{self.consumer_name}"
        )

    @property
    def edge(self) -> tuple[str, str]:
        return self.source_name, self.consumer_name


def work_division_from_view(
    view: PerCoreView | None,
    device_coordinates: Sequence[sympy.Expr],
    iteration_symbols: Sequence[sympy.Symbol],
) -> TensorWorkDivision | None:
    """Project physical per-core ownership into operation-loop symbols."""

    if view is None:
        return None
    loop_symbols = set(iteration_symbols)
    splits: dict[sympy.Symbol, int] = {}
    core_map: dict[sympy.Symbol, sympy.Expr] = {}
    for device_dim, split in view.work_slice_dims:
        if device_dim >= len(device_coordinates):
            raise ValueError(f"missing device coordinate {device_dim}")
        matches = device_coordinates[device_dim].free_symbols & loop_symbols
        if len(matches) != 1:
            raise ValueError(f"cannot map device dimension {device_dim} to one loop")
        dim = next(iter(matches))
        slot = sympy.sympify(dict(view.core_to_slot).get(device_dim, 0))
        if dim in splits and (splits[dim], core_map[dim]) != (split, slot):
            raise ValueError(f"conflicting ownership for loop {dim}")
        splits[dim] = split
        core_map[dim] = slot
    return TensorWorkDivision(splits, core_map)


def materialized_lx_relayouts(
    graph: GraphLowering,
) -> dict[tuple[str, str], tuple[str, LXRelayoutPlan]]:
    return getattr(graph, _REGISTRY, {})


def discard_lx_relayout_group(graph: GraphLowering, source_name: str) -> set[str]:
    copies = materialized_lx_relayouts(graph)
    removed = set()
    for edge, (copy_name, _) in list(copies.items()):
        if edge[0] == source_name:
            removed.add(copy_name)
            del copies[edge]
    return removed


def _core_slices(view: PerCoreView, num_cores: int) -> dict[int, dict[int, int]] | None:
    core_id = sympy.Symbol("core_id")
    splits = dict(view.work_slice_dims)
    slots = dict(view.core_to_slot)
    result = {}
    for core in range(num_cores):
        row = {}
        for dim, split in splits.items():
            value = sympy.sympify(slots.get(dim, 0)).subs(core_id, core)
            if value.free_symbols:
                return None
            slot = int(value)
            if slot < 0 or slot >= split:
                return None
            row[dim] = slot
        result[core] = row
    return result


def _overlap(a: int, an: int, b: int, bn: int) -> bool:
    return a * bn < (b + 1) * an and b * an < (a + 1) * bn


def _compatible_partitions(
    source: PerCoreView, destination: PerCoreView, num_cores: int
) -> bool:
    source_map = _core_slices(source, num_cores)
    destination_map = _core_slices(destination, num_cores)
    if source_map is None or destination_map is None:
        return False
    source_splits = dict(source.work_slice_dims)
    destination_splits = dict(destination.work_slice_dims)
    dims = set(source_splits) | set(destination_splits)
    edges = {
        (s_core, d_core)
        for s_core, s_slice in source_map.items()
        for d_core, d_slice in destination_map.items()
        if all(
            _overlap(
                s_slice.get(dim, 0),
                source_splits.get(dim, 1),
                d_slice.get(dim, 0),
                destination_splits.get(dim, 1),
            )
            for dim in dims
        )
    }
    fanout = [sum(src == core for src, _ in edges) for core in range(num_cores)]
    fanin = [sum(dst == core for _, dst in edges) for core in range(num_cores)]
    return bool(edges) and all(
        (
            min(fanout) > 0,
            min(fanin) > 0,
            len(set(fanout)) == 1,
            len(set(fanin)) == 1,
            len({tuple(sorted(row.items())) for row in source_map.values()})
            == num_cores,
            len({tuple(sorted(row.items())) for row in destination_map.values()})
            == num_cores,
            math.prod(source_splits.values()) == num_cores,
            math.prod(destination_splits.values()) == num_cores,
        )
    )


def _single_write(op: ComputedBuffer, name: str) -> MemoryDep | None:
    writes = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and dep.name == name
    ]
    return writes[0] if len(writes) == 1 and not writes[0].is_indirect() else None


def _op_short_name(op: Operation) -> str:
    for fx_node in (getattr(op, "origin_node", None), *getattr(op, "origins", ())):
        target = getattr(fx_node, "target", None)
        for attr in ("_opname", "__name__", "name"):
            if name := getattr(target, attr, None):
                return str(name)
    return "None"


def _is_activation_source(operations: dict[str, Operation], op: Operation) -> bool:
    """Exclude restickified graph inputs and weights from activation relayout."""

    return _op_short_name(op) != "restickify" or any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def collect_lx_relayout_plans(
    graph: GraphLowering, cache: dict | None = None
) -> list[LXRelayoutPlan]:
    if not config.lx_planner_relayout or getattr(config, "ktir_emitter", False):
        return []
    existing = materialized_lx_relayouts(graph)
    if existing:
        result = []
        for (source_name, _), (copy_name, plan) in existing.items():
            source = graph.try_get_buffer(source_name)
            copy = graph.try_get_buffer(copy_name)
            if source is None or copy is None:
                existing.clear()
                return []
            getattr(source, "layout").lx_view = plan.source_view
            getattr(copy, "layout").lx_view = plan.destination_view
            getattr(copy, "layout").lx_consumer_is_matmul = plan.consumer_is_matmul
            result.append(
                dataclasses.replace(
                    plan,
                    consumer_name=copy_name,
                    destination_buffer_name=copy_name,
                    source_address=None,
                    destination_address=None,
                )
            )
        return result

    operations = {op.get_name(): op for op in graph.operations}
    reads: dict[str, list[tuple[Operation, MemoryDep, int]]] = {}
    for consumer in graph.operations:
        deps = [d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)]
        for index, dep in enumerate(deps):
            reads.setdefault(dep.name, []).append((consumer, dep, index))

    result = []
    for source_name, consumer_reads in reads.items():
        producer = operations.get(source_name)
        if (
            not isinstance(producer, ComputedBuffer)
            or not isinstance(producer.layout, FixedTiledLayout)
            or (write := _single_write(producer, source_name)) is None
        ):
            continue
        source_view, partial, representable = _per_core_view_on_buf(
            producer, write, source_name, cache
        )
        num_cores = _op_num_cores(producer)
        if partial or not representable or _core_slices(source_view, num_cores) is None:
            continue

        source_plans = []
        seen_consumers = set()
        for consumer, dep, read_index in consumer_reads:
            consumer_name = consumer.get_name()
            if (
                consumer_name in seen_consumers
                or not isinstance(consumer, ComputedBuffer)
                or isinstance(consumer.layout, MutationLayoutSHOULDREMOVE)
            ):
                break
            seen_consumers.add(consumer_name)
            deps = [
                d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)
            ]
            if any(d.is_indirect() for d in deps):
                break
            view, consumer_partial, representable = _per_core_view_on_buf(
                consumer, dep, source_name, cache
            )
            if (
                consumer_partial
                or not representable
                or _op_num_cores(consumer) != num_cores
            ):
                break
            if view == source_view:
                continue
            is_matmul = _is_matmul_op(consumer)
            if (not is_matmul and not isinstance(consumer.data, Pointwise)) or (
                is_matmul
                and (
                    not _is_activation_source(operations, producer)
                    or read_index not in (0, 1)
                )
            ):
                break
            consumer_coordinates = try_device_coordinates(
                producer.layout.device_layout, dep, None
            )
            producer_coordinates = try_device_coordinates(
                producer.layout.device_layout, write, None
            )
            if not _compatible_partitions(source_view, view, num_cores):
                break
            if consumer_coordinates is None or producer_coordinates is None:
                break
            try:
                consumer_symbols = tuple(iteration_space_from_op(consumer))
                work_division_from_view(
                    source_view, consumer_coordinates, consumer_symbols
                )
                work_division_from_view(view, consumer_coordinates, consumer_symbols)
                work_division_from_view(
                    source_view,
                    producer_coordinates,
                    tuple(iteration_space_from_op(producer)),
                )
            except ValueError:
                break
            source_plans.append(
                LXRelayoutPlan(
                    source_name,
                    consumer_name,
                    source_view,
                    view,
                    num_cores,
                    is_matmul,
                )
            )
        else:
            result.extend(source_plans)
    return result


def materialize_lx_relayouts(graph: GraphLowering, plans: list[LXRelayoutPlan]) -> None:
    for op in graph.operations:
        if isinstance(layout := getattr(op, "layout", None), FixedTiledLayout):
            layout.lx_view = None
            layout.lx_consumer_is_matmul = False
    copies = materialized_lx_relayouts(graph)
    if not plans:
        copies.clear()
        return
    from .scratchpad.graph_editor import GraphEditor

    editor = GraphEditor(graph)
    setattr(graph, _REGISTRY, copies)
    for plan in plans:
        source = cast(ComputedBuffer, graph.get_buffer(plan.source_name))
        if plan.destination_buffer_name:
            copy = cast(ComputedBuffer, graph.get_buffer(plan.destination_buffer_name))
        else:
            consumer = cast(ComputedBuffer, graph.get_buffer(plan.consumer_name))
            copy = editor.insert_clone_before_consumer(source, consumer)
            copies[plan.edge] = (copy.get_name(), plan)

        assert plan.source_address is not None and plan.destination_address is not None
        getattr(source, "layout").allocation["lx"] = plan.source_address
        getattr(copy, "layout").allocation["lx"] = plan.destination_address
        getattr(source, "layout").lx_view = plan.source_view
        getattr(copy, "layout").lx_view = plan.destination_view
        getattr(copy, "layout").lx_consumer_is_matmul = plan.consumer_is_matmul
        logger.debug(
            "accepted LX relayout %s -> %s: source=%s@%d destination=%s@%d",
            source.get_name(),
            copy.get_name(),
            plan.source_view,
            plan.source_address,
            plan.destination_view,
            plan.destination_address,
        )
