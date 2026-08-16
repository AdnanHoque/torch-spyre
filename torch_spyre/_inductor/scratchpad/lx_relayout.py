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

from .. import config
from ..constants import (
    COARSE_TILE_HOISTED_LOOP_GROUP_ATTR,
    CORE_MAPPING_CONTIGUOUS_DIM_ATTR,
)
from ..core_mapping import core_to_slice_mapping
from ..ir import FixedTiledLayout
from ..logging_utils import get_inductor_logger
from ..op_spec import TensorWorkDivision
from ..pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    iteration_space_from_op,
    op_short_name,
    op_read_writes,
    try_device_coordinates,
)
from .utils import _op_num_cores

logger = get_inductor_logger("lx_relayout")
_DESTINATION_PREFIX = "__spyre_lx_relayout__"
_REGISTRY = "_spyre_lx_relayout_copies"


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    consumer_names: tuple[str, ...]
    source_view: PerCoreView
    destination_view: PerCoreView
    num_cores: int
    source_address: int | None = None
    destination_address: int | None = None

    @property
    def destination_name(self) -> str:
        return f"{_DESTINATION_PREFIX}:{self.source_name}:{self.consumer_names[0]}"

    @property
    def edge(self) -> tuple[str, str]:
        return self.source_name, self.destination_name


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
    slots = dict(view.core_to_slot)
    for device_dim, split in view.work_slice_dims:
        if device_dim >= len(device_coordinates):
            raise ValueError(f"missing device coordinate {device_dim}")
        matches = device_coordinates[device_dim].free_symbols & loop_symbols
        if len(matches) != 1:
            raise ValueError(f"cannot map device dimension {device_dim} to one loop")
        dim = next(iter(matches))
        slot = sympy.sympify(slots[device_dim])
        if dim in splits and (splits[dim], core_map[dim]) != (split, slot):
            raise ValueError(f"conflicting ownership for loop {dim}")
        splits[dim] = split
        core_map[dim] = slot
    return TensorWorkDivision(splits, core_map)


def materialized_lx_relayouts(
    graph: GraphLowering,
) -> dict[tuple[str, str], tuple[str, LXRelayoutPlan]]:
    return getattr(graph, _REGISTRY, {})


def _discard_lx_relayout_group(graph: GraphLowering, source_name: str) -> set[str]:
    copies = materialized_lx_relayouts(graph)
    removed = set()
    for edge, (copy_name, _) in list(copies.items()):
        if edge[0] == source_name:
            removed.add(copy_name)
            del copies[edge]
    return removed


def _clear_lx_state(layout: FixedTiledLayout) -> None:
    """Clear an LX buffer's placement and physical ownership."""

    layout.allocation.pop("lx", None)
    layout.lx_view = None


def demote_lx_relayout_group(
    graph: GraphLowering, source_name: str, reason: str
) -> None:
    """Remove one relayout group from LX and its materialization registry."""

    names = {source_name, *_discard_lx_relayout_group(graph, source_name)}
    for name in names:
        buffer = graph.try_get_buffer(name)
        if buffer is None:
            continue
        layout = buffer.get_layout()
        if isinstance(layout, FixedTiledLayout):
            _clear_lx_state(layout)
    logger.info("demoted %s out of LX: %s", ", ".join(sorted(names)), reason)


def _core_slices(view: PerCoreView, num_cores: int) -> dict[int, dict[int, int]]:
    core_id = sympy.Symbol("core_id")
    splits = dict(view.work_slice_dims)
    slots = dict(view.core_to_slot)
    result = {}
    for core in range(num_cores):
        row = {}
        for dim, split in splits.items():
            value = sympy.sympify(slots[dim]).subs(core_id, core)
            assert not value.free_symbols, f"non-concrete owner slot {value}"
            slot = int(value)
            assert 0 <= slot < split, f"owner slot {slot} outside split {split}"
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


def _matching_core_mapping_contiguous_dim(
    division: TensorWorkDivision,
    iteration_symbols: Sequence[sympy.Symbol],
    num_cores: int,
) -> tuple[bool, int | None]:
    """Find the core-order choice that realizes ``division`` exactly.

    ``work_division_from_view`` recovers both split factors and the physical
    core-to-slice assignment.  Applying only the split factors is insufficient:
    an identity defaults to its first split dimension varying fastest, whereas
    a matmul may deliberately make K vary fastest.  Try the ordinary order and
    each actually-split dimension, accepting only a mapping that agrees on
    every physical core.  The boolean distinguishes a matching ordinary order
    (``None``) from no representable order.
    """

    iteration_symbols = tuple(iteration_symbols)
    if any(dim not in iteration_symbols for dim in division.work_slices):
        return False, None
    dim_splits = tuple(
        int(division.work_slices.get(dim, 1)) for dim in iteration_symbols
    )
    if math.prod(dim_splits) != num_cores:
        return False, None

    core_id = sympy.Symbol("core_id")
    candidates: list[int | None] = [None]
    candidates.extend(i for i, split in enumerate(dim_splits) if split > 1)
    for contiguous_dim in candidates:
        mapping = core_to_slice_mapping(
            iteration_symbols,
            dim_splits,
            num_cores,
            contiguous_dim=contiguous_dim,
        )
        if all(
            int(mapping[dim].subs(core_id, core))
            == int(sympy.sympify(target).subs(core_id, core))
            for dim, target in division.core_id_to_work_slice.items()
            for core in range(num_cores)
        ):
            return True, contiguous_dim
    return False, None


def align_hoisted_invariant_copy_work_divisions(
    graph: GraphLowering,
) -> list[str]:
    """Match an invariant copy to one common matmul-consumer ownership.

    Coarse tiling synthesizes an activation copy before a temporal expert loop.
    Work distribution subsequently optimizes that identity independently from
    the matmuls in the loop.  When all matmul consumers read the copy with one
    identical full-core view, retaining the identity's independent split would
    require an avoidable LX shuffle (or spill the activation to HBM).  Project
    the common consumer view back onto the copy instead.

    This is deliberately fail closed.  Only marked coarse-tile read copies with
    at least two matmul consumers are eligible; every consumer must use the same
    core count and exact :class:`PerCoreView`, and that view must use every core
    exactly once.  An ordinary copy, a second incompatible view, a non-matmul
    user, partial ownership, or a projection mismatch leaves the original work
    division untouched.
    """

    if not config.lx_planning:
        return []

    operations = {op.get_name(): op for op in graph.operations}
    reads: dict[str, list[tuple[Operation, MemoryDep]]] = {}
    for consumer in graph.operations:
        for dep in op_read_writes(consumer).reads:
            if isinstance(dep, MemoryDep):
                reads.setdefault(dep.name, []).append((consumer, dep))

    aligned = []
    for source_name, consumer_reads in reads.items():
        producer = operations.get(source_name)
        if (
            not isinstance(producer, ComputedBuffer)
            or not isinstance(producer.layout, FixedTiledLayout)
            or not source_name.startswith("coarse_tile_read_copy_")
            or not getattr(producer, COARSE_TILE_HOISTED_LOOP_GROUP_ATTR, ())
            or (write := _single_write(producer, source_name)) is None
        ):
            continue

        num_cores = _op_num_cores(producer)
        common_view: PerCoreView | None = None
        seen_consumers = set()
        rejected = False
        cache: dict = {}
        for consumer, dep in consumer_reads:
            consumer_name = consumer.get_name()
            if (
                consumer_name in seen_consumers
                or not isinstance(consumer, ComputedBuffer)
                or isinstance(consumer.layout, MutationLayoutSHOULDREMOVE)
                or not _is_matmul_op(consumer)
                or _op_num_cores(consumer) != num_cores
            ):
                rejected = True
                break
            seen_consumers.add(consumer_name)
            deps = [
                item
                for item in op_read_writes(consumer).reads
                if isinstance(item, MemoryDep) and item.name == source_name
            ]
            if len(deps) != 1:
                rejected = True
                break
            view, _partial_reduction, representable = _per_core_view_on_buf(
                consumer, dep, source_name, cache
            )
            # ``has_partial_reduction`` is an op-level fact.  A gate/up BMM
            # split on K therefore reports it even though this edge is the
            # ordinary X read whose M/K ownership is exactly what we need to
            # match.  It is meaningful for the producer's write check below,
            # not as a veto on this read edge.
            if not representable:
                rejected = True
                break
            if common_view is None:
                common_view = view
            elif view != common_view:
                rejected = True
                break

        if rejected or common_view is None or len(seen_consumers) < 2:
            continue

        core_slices = _core_slices(common_view, num_cores)
        unique_slices = {
            tuple(sorted(per_core.items())) for per_core in core_slices.values()
        }
        if (
            len(unique_slices) != num_cores
            or math.prod(dict(common_view.work_slice_dims).values()) != num_cores
        ):
            continue

        producer_coordinates = try_device_coordinates(
            producer.layout.device_layout, write, None
        )
        if producer_coordinates is None:
            continue
        try:
            division = work_division_from_view(
                common_view,
                producer_coordinates,
                tuple(iteration_space_from_op(producer)),
            )
        except ValueError:
            continue
        if division is None or math.prod(division.work_slices.values()) != num_cores:
            continue

        mapping_matches, contiguous_dim = _matching_core_mapping_contiguous_dim(
            division,
            tuple(iteration_space_from_op(producer)),
            num_cores,
        )
        if not mapping_matches:
            continue

        from ..work_division import TensorDep, apply_splits

        had_splits = hasattr(producer, "op_it_space_splits")
        old_splits = getattr(producer, "op_it_space_splits", ({}, {}))
        had_mapping = hasattr(producer, CORE_MAPPING_CONTIGUOUS_DIM_ATTR)
        old_mapping = getattr(producer, CORE_MAPPING_CONTIGUOUS_DIM_ATTR, None)
        if contiguous_dim is None:
            if had_mapping:
                delattr(producer, CORE_MAPPING_CONTIGUOUS_DIM_ATTR)
        else:
            setattr(producer, CORE_MAPPING_CONTIGUOUS_DIM_ATTR, contiguous_dim)
        apply_splits(
            producer,
            dict(division.work_slices),
            TensorDep(write, producer.layout),
        )
        projected, partial, representable = _per_core_view_on_buf(
            producer, write, source_name, {}
        )
        if partial or not representable or projected != common_view:
            if had_splits:
                producer.op_it_space_splits = old_splits
            else:
                delattr(producer, "op_it_space_splits")
            if had_mapping:
                setattr(producer, CORE_MAPPING_CONTIGUOUS_DIM_ATTR, old_mapping)
            elif hasattr(producer, CORE_MAPPING_CONTIGUOUS_DIM_ATTR):
                delattr(producer, CORE_MAPPING_CONTIGUOUS_DIM_ATTR)
            continue

        aligned.append(source_name)
        logger.debug(
            "aligned hoisted invariant copy %s to common consumers %s: %s",
            source_name,
            sorted(seen_consumers),
            common_view,
        )
    return aligned


def align_activation_stationary_loop_work_divisions(
    graph: GraphLowering,
) -> list[str]:
    """Use one row-only ownership for a temporal shared-LHS expert loop.

    The activation-stationary control deliberately keeps one full token-row
    shard live while the expert loop advances its weights.  Independently
    optimizing the leaves can otherwise choose M16xK2 for gate/up, M32 for
    pointwise operations, and M8xN4 for down.  Those choices need ownership
    redistribution between every leaf and defeat the transport-free control.

    This pass is intentionally narrow and fail closed.  It requires exactly
    two marked shared-LHS matmuls in one coarse-tile loop, both reading the same
    marked invariant preheader, and a unique common row extent divisible by all
    configured cores.  Only operations in that exact loop with one output
    iteration dimension of that extent are changed.  Expert-weight copies have
    no row dimension and remain untouched.
    """

    if not config.lx_planning or config.sencores <= 1:
        return []

    operations = {op.get_name(): op for op in graph.operations}
    reads: dict[str, list[tuple[ComputedBuffer, MemoryDep]]] = {}
    for consumer in graph.operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        for dep in op_read_writes(consumer).reads:
            if isinstance(dep, MemoryDep):
                reads.setdefault(dep.name, []).append((consumer, dep))

    decisions: list[tuple[ComputedBuffer, sympy.Symbol, MemoryDep]] = []
    for source_name, consumer_reads in reads.items():
        producer = operations.get(source_name)
        owner = getattr(producer, COARSE_TILE_HOISTED_LOOP_GROUP_ATTR, ())
        if (
            not isinstance(producer, ComputedBuffer)
            or not source_name.startswith("coarse_tile_read_copy_")
            or not owner
        ):
            continue

        shared_lhs = []
        for consumer, dep in consumer_reads:
            op_info = getattr(consumer.data, "op_info", None) or {}
            if (
                _is_matmul_op(consumer)
                and "activation_stationary_shared_lhs_mm" in op_info
                and getattr(getattr(consumer, "loop_info", None), "loop_group_id", ())
                == owner
            ):
                shared_lhs.append((consumer, dep))
        if len(shared_lhs) != 2:
            continue

        row_extents = []
        for consumer, x_dep in shared_lhs:
            write = _single_write(consumer, consumer.get_name())
            if write is None:
                break
            row_symbols = x_dep.index.free_symbols & write.index.free_symbols
            if len(row_symbols) != 1:
                break
            row = next(iter(row_symbols))
            extent = iteration_space_from_op(consumer).get(row)
            if extent is None:
                break
            row_extents.append(extent)
        if len(row_extents) != 2 or row_extents[0] != row_extents[1]:
            continue
        row_extent = sympy.sympify(row_extents[0])
        if not row_extent.is_number or int(row_extent) % config.sencores:
            continue

        group_decisions: list[tuple[ComputedBuffer, sympy.Symbol, MemoryDep]] = []
        for op in graph.operations:
            if (
                not isinstance(op, ComputedBuffer)
                or isinstance(op.layout, MutationLayoutSHOULDREMOVE)
                or getattr(getattr(op, "loop_info", None), "loop_group_id", ())
                != owner
            ):
                continue
            write = _single_write(op, op.get_name())
            if write is None:
                continue
            it_space = iteration_space_from_op(op)
            rows = [
                symbol
                for symbol, extent in it_space.items()
                if symbol in write.index.free_symbols and extent == row_extent
            ]
            if len(rows) == 1:
                group_decisions.append((op, rows[0], write))

        # Both shared-LHS matmuls and the down matmul must be part of the
        # aligned group.  Otherwise this is not the connected FFN pattern.
        group_ops = [op for op, _, _ in group_decisions]
        matmuls = [op for op in group_ops if _is_matmul_op(op)]
        if len(matmuls) != 3 or any(
            all(op is not group_op for group_op in group_ops) for op, _ in shared_lhs
        ):
            continue
        decisions.extend(group_decisions)

    if not decisions:
        return []

    from ..work_division import TensorDep, apply_splits

    aligned = []
    for op, row, write in decisions:
        apply_splits(op, {row: config.sencores}, TensorDep(write, op.layout))
        aligned.append(op.get_name())
    return aligned


def _single_write(op: ComputedBuffer, name: str) -> MemoryDep | None:
    writes = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and dep.name == name
    ]
    if len(writes) != 1 or writes[0].is_indirect():
        return None
    return writes[0]


def _is_activation_source(operations: dict[str, Operation], op: Operation) -> bool:
    """Exclude restickified graph inputs and weights from activation relayout."""

    return op_short_name(op) != "restickify" or any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def collect_lx_relayout_plans(graph: GraphLowering) -> list[LXRelayoutPlan]:
    if not config.lx_planner_relayout or config.ktir_emitter:
        return []
    assert not materialized_lx_relayouts(graph), (
        "LX relayout planning requires an unmaterialized graph"
    )

    cache: dict = {}
    operations = {op.get_name(): op for op in graph.operations}
    reads: dict[str, list[tuple[Operation, MemoryDep]]] = {}
    for consumer in graph.operations:
        deps = [d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)]
        for dep in deps:
            reads.setdefault(dep.name, []).append((consumer, dep))

    result: list[LXRelayoutPlan] = []
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
        if partial or not representable:
            continue

        # Activation eligibility belongs to the producer, not to an individual
        # edge. Never relayout a restickified graph input or weight.
        if not _is_activation_source(operations, producer):
            continue

        producer_coordinates = try_device_coordinates(
            producer.layout.device_layout, write, None
        )
        if producer_coordinates is None:
            continue
        try:
            work_division_from_view(
                source_view,
                producer_coordinates,
                tuple(iteration_space_from_op(producer)),
            )
        except ValueError:
            continue

        consumers_by_view: dict[PerCoreView, list[str]] = {}
        seen_consumers = set()
        rejection_reason = None
        for consumer, dep in consumer_reads:
            consumer_name = consumer.get_name()
            if consumer_name in seen_consumers:
                rejection_reason = "consumer reads the source more than once"
                break
            if not isinstance(consumer, ComputedBuffer) or isinstance(
                consumer.layout, MutationLayoutSHOULDREMOVE
            ):
                rejection_reason = "consumer is not a supported computed buffer"
                break
            seen_consumers.add(consumer_name)
            deps = [
                d for d in op_read_writes(consumer).reads if isinstance(d, MemoryDep)
            ]
            if any(d.is_indirect() for d in deps):
                rejection_reason = "consumer uses indirect access"
                break
            view, consumer_partial, representable = _per_core_view_on_buf(
                consumer, dep, source_name, cache
            )
            if (
                consumer_partial
                or not representable
                or _op_num_cores(consumer) != num_cores
            ):
                rejection_reason = (
                    "consumer ownership is partial, unrepresentable, or uses a "
                    "different core count"
                )
                break
            consumer_coordinates = try_device_coordinates(
                producer.layout.device_layout, dep, None
            )
            if consumer_coordinates is None:
                rejection_reason = "consumer coordinates are unavailable"
                break
            consumer_symbols = tuple(iteration_space_from_op(consumer))
            try:
                work_division_from_view(
                    source_view, consumer_coordinates, consumer_symbols
                )
            except ValueError:
                rejection_reason = "source ownership cannot be projected to consumer"
                break
            if view == source_view:
                continue
            is_matmul = _is_matmul_op(consumer)
            if is_matmul and len(deps) != 2:
                rejection_reason = "matmul consumer does not have two inputs"
                break
            if not is_matmul and not isinstance(consumer.data, Pointwise):
                rejection_reason = "consumer is neither pointwise nor matmul"
                break
            if not _compatible_partitions(source_view, view, num_cores):
                rejection_reason = "source and destination partitions are incompatible"
                break
            try:
                work_division_from_view(view, consumer_coordinates, consumer_symbols)
            except ValueError:
                rejection_reason = (
                    "destination ownership cannot be projected to consumer"
                )
                break
            consumers_by_view.setdefault(view, []).append(consumer_name)
        else:
            result.extend(
                LXRelayoutPlan(
                    source_name,
                    tuple(consumer_names),
                    source_view,
                    destination_view,
                    num_cores,
                )
                for destination_view, consumer_names in consumers_by_view.items()
            )
        if rejection_reason is not None:
            logger.debug(
                "rejected LX relayout candidate source=%s consumer=%s: %s",
                source_name,
                consumer_name,
                rejection_reason,
            )
    return result


def materialize_lx_relayouts(graph: GraphLowering, plans: list[LXRelayoutPlan]) -> None:
    if not plans:
        assert not materialized_lx_relayouts(graph)
        return
    from .graph_editor import GraphEditor

    copies = materialized_lx_relayouts(graph)
    assert not copies, "LX relayouts were already materialized"
    editor = GraphEditor(graph)
    setattr(graph, _REGISTRY, copies)
    for plan in plans:
        source = cast(ComputedBuffer, graph.get_buffer(plan.source_name))
        consumers = [
            cast(ComputedBuffer, graph.get_buffer(name)) for name in plan.consumer_names
        ]
        copy = editor.insert_clone_before_consumers(source, consumers)
        copies[plan.edge] = (copy.get_name(), plan)

        assert plan.source_address is not None and plan.destination_address is not None
        assert plan.source_view != plan.destination_view
        source_layout = cast(FixedTiledLayout, source.layout)
        copy_layout = cast(FixedTiledLayout, copy.layout)
        source_layout.allocation["lx"] = plan.source_address
        copy_layout.allocation["lx"] = plan.destination_address
        source_layout.lx_view = plan.source_view
        copy_layout.lx_view = plan.destination_view
        logger.debug(
            "accepted LX relayout %s -> %s: source=%s@%d destination=%s@%d",
            source.get_name(),
            copy.get_name(),
            plan.source_view,
            plan.source_address,
            plan.destination_view,
            plan.destination_address,
        )
