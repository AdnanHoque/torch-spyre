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

from collections import Counter
import dataclasses
import math
from typing import Sequence

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.virtualized import V

from torch_spyre._inductor import config
from torch_spyre._inductor.constants import SHARED_WEIGHT_UNIT_BMM_INFO_KEY
from torch_spyre._inductor.core_mapping import (
    materialize_core_mapping,
    resolve_core_mapping,
)
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    apply_splits_from_index_coeff,
    iteration_space_from_op,
    op_read_writes,
    try_device_coordinates,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.scratchpad.utils import _op_num_cores, _op_short_name
from torch_spyre._inductor.views import align_tensors

LX_RELAYOUT_ATTR = "_spyre_lx_relayout_inputs"
LX_RELAYOUT_DESTINATION_PREFIX = "__spyre_lx_relayout_destination__"


def make_lx_relayout_destination_name(source_name: str) -> str:
    return f"{LX_RELAYOUT_DESTINATION_PREFIX}:{source_name}"


def is_lx_relayout_destination(name: str) -> bool:
    return name.startswith(f"{LX_RELAYOUT_DESTINATION_PREFIX}:")


@dataclasses.dataclass(frozen=True)
class LXShuffleGeometry:
    """Materialization-ready geometry expressed as consumer-axis ordinals."""

    iteration_axes: tuple[int, ...]
    consumer_rank: int
    consumer_splits: tuple[int, ...]
    destination_splits: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.iteration_axes
            or self.consumer_rank <= 0
            or len(set(self.iteration_axes)) != len(self.iteration_axes)
            or any(
                axis < 0 or axis >= self.consumer_rank for axis in self.iteration_axes
            )
            or len(self.consumer_splits) != self.consumer_rank
            or any(split <= 0 for split in self.consumer_splits)
            or len(self.destination_splits) != len(self.iteration_axes)
            or any(split <= 0 for split in self.destination_splits)
        ):
            raise ValueError("invalid LX shuffle geometry")


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    """Names and exact per-core geometry for one LX shuffle."""

    source_name: str
    consumer_name: str
    # Allocation ownership is keyed by consumer iteration-axis ordinal rather
    # than physical device-dimension ordinal. ``align_tensors`` may normalize
    # or reorder physical dimensions before SuperDSC generation, while these
    # logical axes remain stable.
    source_core_id_to_axis_slice: dict[str, dict[str, int]]
    destination_core_id_to_axis_slice: dict[str, dict[str, int]]
    source_axis_splits: dict[str, int]
    destination_axis_splits: dict[str, int]
    destination_size_ratio: int
    shuffle_geometry: LXShuffleGeometry
    destination_lx_address: int | None = None

    @property
    def destination_name(self) -> str:
        return make_lx_relayout_destination_name(self.source_name)


def _intervals_overlap(
    lhs_slot: int, lhs_split: int, rhs_slot: int, rhs_split: int
) -> bool:
    return lhs_slot * rhs_split < (rhs_slot + 1) * lhs_split and (
        rhs_slot * lhs_split < (lhs_slot + 1) * rhs_split
    )


def _destination_size_ratio(
    producer_map: dict[str, dict[str, int]],
    producer_splits: dict[str, int],
    consumer_map: dict[str, dict[str, int]],
    consumer_splits: dict[str, int],
) -> int | None:
    """Return S2:S1 per-core size ratio for exact uniform shuffle geometry."""

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
    max_fanout = max(fanout_values, default=0)
    max_fanin = max(fanin_values, default=0)
    covers_all_cores = (
        min(fanout_values, default=0) > 0 and min(fanin_values, default=0) > 0
    )
    producer_is_partitioned = slice_count(producer_map) == len(producer_map)
    consumer_is_partitioned = slice_count(consumer_map) == len(consumer_map)
    if transfer_count == 0 or not covers_all_cores or not uniform:
        return None
    if (
        producer_is_partitioned
        and consumer_is_partitioned
        and max_fanout <= 1
        and max_fanin <= 1
    ):
        return 1
    if (
        producer_is_partitioned
        and not consumer_is_partitioned
        and max_fanout > 1
        and max_fanin > 1
    ):
        return max_fanin
    return None


def _single_write_dep(op: ComputedBuffer, buf_name: str) -> MemoryDep | None:
    matches = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and not dep.is_indirect() and dep.name == buf_name
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


def _has_shared_weight_unit_bmm_info(op_info: object) -> bool:
    if not isinstance(op_info, dict):
        return False
    info = op_info.get(SHARED_WEIGHT_UNIT_BMM_INFO_KEY)
    return isinstance(info, dict) and info.get("batch_dim") == 0


def _core_id_to_device_slice(
    view: PerCoreView,
    core_count: int,
) -> dict[str, dict[str, int]] | None:
    """Return ownership as ``core -> device-dim -> slice-index``."""

    core_id = sympy.Symbol("core_id")
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    split_dims = {int(dim): int(split) for dim, split in view.work_slice_dims}
    if not split_dims.keys() <= expr_by_dim.keys():
        return None
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


def _uniform_full_grid_coverage(
    core_map: dict[str, dict[str, int]], splits: dict[str, int]
) -> bool:
    """Return whether every physical slice has the same nonzero core count."""

    if not core_map or not splits or any(split <= 0 for split in splits.values()):
        return False
    dims = tuple(sorted(splits, key=int))
    slots: Counter[tuple[int, ...]] = Counter()
    for per_core in core_map.values():
        if set(per_core) != set(dims):
            return False
        slot = tuple(int(per_core[dim]) for dim in dims)
        if any(value < 0 or value >= splits[dim] for value, dim in zip(slot, dims)):
            return False
        slots[slot] += 1
    return len(slots) == math.prod(splits.values()) and len(set(slots.values())) == 1


def _distribution_by_consumer_axis(
    device_coordinates: Sequence[sympy.Expr],
    consumer_symbols: Sequence[sympy.Symbol],
    core_map: dict[str, dict[str, int]],
    device_dim_splits: dict[str, int],
) -> tuple[dict[str, dict[str, int]], dict[str, int]] | None:
    """Convert physical ownership to alignment-stable consumer-axis ordinals."""

    symbol_to_axis = {symbol: axis for axis, symbol in enumerate(consumer_symbols)}
    consumer_symbol_set = set(consumer_symbols)
    axis_splits: dict[str, int] = {}
    split_device_dim_by_axis: dict[str, str] = {}

    def is_axis_ownership_coordinate(
        coordinate: sympy.Expr, symbol: sympy.Symbol
    ) -> bool:
        if sympy.simplify(coordinate - symbol) == 0:
            return True
        if coordinate.func is not sympy.floor or len(coordinate.args) != 1:
            return False
        coefficient, term = coordinate.args[0].as_coeff_Mul()
        return (
            term == symbol
            and coefficient.is_Rational
            and coefficient.p == 1
            and coefficient.q > 0
        )

    for device_dim, split in device_dim_splits.items():
        try:
            index = int(device_dim)
            split = int(split)
        except (TypeError, ValueError):
            return None
        if split <= 0 or index < 0 or index >= len(device_coordinates):
            return None
        mapped_symbols = [
            symbol
            for symbol in device_coordinates[index].free_symbols
            if symbol in consumer_symbol_set
        ]
        if len(mapped_symbols) != 1:
            if split == 1:
                continue
            return None
        symbol = mapped_symbols[0]
        # A physical dimension such as ``floor(k / stick_size)`` still owns a
        # slice of the logical ``k`` axis.  Other transformed coordinates do
        # not provide the contiguous logical-axis contract required here.
        if split > 1 and not is_axis_ownership_coordinate(
            device_coordinates[index], symbol
        ):
            return None
        axis = str(symbol_to_axis[symbol])
        if axis in axis_splits and axis_splits[axis] != split:
            return None
        if split > 1 and axis in split_device_dim_by_axis:
            return None
        axis_splits[axis] = split
        if split > 1:
            split_device_dim_by_axis[axis] = str(device_dim)

    if not axis_splits:
        return None

    result: dict[str, dict[str, int]] = {}
    for core, per_device_dim in core_map.items():
        for device_dim, slot in per_device_dim.items():
            split = int(device_dim_splits.get(str(device_dim), 1))
            if split == 1 and int(slot) != 0:
                return None
        per_axis = {axis: 0 for axis in axis_splits}
        for axis, device_dim in split_device_dim_by_axis.items():
            slot = int(per_device_dim.get(device_dim, 0))
            if slot < 0 or slot >= axis_splits[axis]:
                return None
            per_axis[axis] = slot
        result[str(core)] = per_axis

    return result, axis_splits


def _shuffle_alignment_is_stable(
    graph: GraphLowering,
    device_size: Sequence[sympy.Expr],
    device_coordinates: Sequence[sympy.Expr],
    consumer_iteration_space: dict[sympy.Symbol, sympy.Expr],
    geometry: LXShuffleGeometry,
) -> bool:
    """Preflight the exact SHUFFLE normalization without mutating graph state."""

    if V.graph is not graph:
        return False
    consumer_symbols = tuple(consumer_iteration_space)
    shuffle_symbols = tuple(consumer_symbols[axis] for axis in geometry.iteration_axes)
    if len(shuffle_symbols) != len(geometry.destination_splits):
        return False
    proposed_iteration_space = {
        symbol: (consumer_iteration_space[symbol], split)
        for symbol, split in zip(shuffle_symbols, geometry.destination_splits)
    }
    tensors = [
        {"size": list(device_size), "coordinates": list(device_coordinates)}
        for _ in range(2)
    ]

    had_repeat_info = hasattr(graph, "_repeat_info")
    repeat_info = getattr(graph, "_repeat_info", None)
    if had_repeat_info and not isinstance(repeat_info, dict):
        return False
    saved_repeat_info = dict(repeat_info) if repeat_info is not None else None
    try:
        aligned_iteration_space, _ = align_tensors(
            proposed_iteration_space,
            tensors,
        )
        if tuple(aligned_iteration_space) != shuffle_symbols:
            return False
        for symbol, (extent, split) in proposed_iteration_space.items():
            aligned_extent, aligned_split = aligned_iteration_space[symbol]
            if sympy.simplify(aligned_extent - extent) != 0 or int(
                aligned_split
            ) != int(split):
                return False
        return True
    except (
        Unsupported,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ):
        return False
    finally:
        if had_repeat_info:
            current_repeat_info = getattr(graph, "_repeat_info", None)
            if isinstance(current_repeat_info, dict):
                current_repeat_info.clear()
                current_repeat_info.update(saved_repeat_info or {})
            else:
                setattr(graph, "_repeat_info", dict(saved_repeat_info or {}))
        elif hasattr(graph, "_repeat_info"):
            delattr(graph, "_repeat_info")


def _build_lx_shuffle_geometry(
    device_coordinates: Sequence[sympy.Expr],
    consumer_symbols: Sequence[sympy.Symbol],
    consumer_axis_splits: Sequence[int],
    producer_map: dict[str, dict[str, int]],
    producer_splits: dict[str, int],
    consumer_map: dict[str, dict[str, int]],
    consumer_splits: dict[str, int],
) -> LXShuffleGeometry | None:
    """Build the exact symbol geometry codegen will consume, or reject it."""

    participant_count = len(consumer_map)
    expected_cores = {str(core) for core in range(participant_count)}
    if (
        set(producer_map) != set(consumer_map)
        or set(producer_map) != expected_cores
        or not _uniform_full_grid_coverage(producer_map, producer_splits)
        or not _uniform_full_grid_coverage(consumer_map, consumer_splits)
    ):
        return None

    if (
        len(set(consumer_symbols)) != len(consumer_symbols)
        or len(consumer_axis_splits) != len(consumer_symbols)
        or any(split <= 0 for split in consumer_axis_splits)
        or math.prod(consumer_axis_splits) != participant_count
    ):
        return None
    consumer_symbol_set = set(consumer_symbols)
    consumer_symbol_to_axis = {
        symbol: axis for axis, symbol in enumerate(consumer_symbols)
    }
    used_symbols = {
        symbol
        for coordinate in device_coordinates
        for symbol in coordinate.free_symbols
        if symbol in consumer_symbol_set
    }
    symbols = tuple(symbol for symbol in consumer_symbols if symbol in used_symbols)
    if not symbols:
        return None

    def splits_by_symbol(
        device_dim_splits: dict[str, int],
    ) -> dict[sympy.Symbol, int] | None:
        result: dict[sympy.Symbol, int] = {}
        for device_dim, split in device_dim_splits.items():
            try:
                index = int(device_dim)
            except (TypeError, ValueError):
                return None
            if split <= 0 or index < 0 or index >= len(device_coordinates):
                return None
            mapped_symbols = [
                symbol
                for symbol in device_coordinates[index].free_symbols
                if symbol in consumer_symbol_set
            ]
            if len(mapped_symbols) != 1:
                if split == 1:
                    continue
                return None
            symbol = mapped_symbols[0]
            if split > 1 and result.get(symbol, 1) > 1:
                return None
            if symbol in result and result[symbol] != split:
                return None
            result[symbol] = split
        return result

    source_by_symbol = splits_by_symbol(producer_splits)
    destination_by_symbol = splits_by_symbol(consumer_splits)
    if source_by_symbol is None or destination_by_symbol is None:
        return None

    source_symbol_splits = tuple(source_by_symbol.get(symbol, 1) for symbol in symbols)
    destination_symbol_splits = tuple(
        destination_by_symbol.get(symbol, 1) for symbol in symbols
    )
    iteration_axes = tuple(consumer_symbol_to_axis[symbol] for symbol in symbols)
    if (
        participant_count % math.prod(source_symbol_splits) != 0
        or participant_count % math.prod(destination_symbol_splits) != 0
        or destination_symbol_splits
        != tuple(consumer_axis_splits[axis] for axis in iteration_axes)
    ):
        return None

    # The SHUFFLE runs over only the tensor-retained consumer axes. SuperDSC
    # therefore resolves a fresh natural mapping for that reduced space, with
    # omitted axes represented as replicas. Admit the plan only when that
    # mapping is exactly the destination allocation ownership. Otherwise the
    # SHUFFLE corelet and S2 tensor would disagree about which slice a core owns.
    core_id = sympy.Symbol("core_id")
    shuffle_mapping = resolve_core_mapping(
        destination_symbol_splits,
        participant_count,
    )
    shuffle_coords = materialize_core_mapping(
        shuffle_mapping,
        symbols,
        destination_symbol_splits,
        participant_count,
        core_id=core_id,
    )
    for core, per_device_dim in consumer_map.items():
        core_number = int(core)
        for device_dim, slot in per_device_dim.items():
            coordinate = device_coordinates[int(device_dim)]
            mapped_symbols = [
                symbol
                for symbol in coordinate.free_symbols
                if symbol in consumer_symbol_set
            ]
            expected_slot = 0
            if len(mapped_symbols) == 1:
                expr = shuffle_coords.get(str(mapped_symbols[0]))
                if expr is not None:
                    expected_slot = int(expr.subs(core_id, core_number))
            if int(slot) != expected_slot:
                return None

    return LXShuffleGeometry(
        iteration_axes=iteration_axes,
        consumer_rank=len(consumer_symbols),
        consumer_splits=tuple(int(split) for split in consumer_axis_splits),
        destination_splits=destination_symbol_splits,
    )


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_ATTR):
            delattr(op, LX_RELAYOUT_ATTR)


def collect_lx_relayout_plans(
    graph: GraphLowering, cache: dict | None = None
) -> list[LXRelayoutPlan]:
    """Plan bounded LX relayouts from producer and consumer coordinates.

    V1 only records movement into matmul consumers for single-writer
    intermediate tensors whose producer output is final (not K-split partials)
    and whose producer and consumer PerCoreViews differ. Same-view edges remain
    owned by the existing LX planner.
    """

    if not config.lx_planner_relayout or config.co_optimizing_lx_planning:
        return []

    operations = _operations_by_name(graph)
    read_counts: dict[str, int] = {}
    for op in graph.operations:
        for dep in op_read_writes(op).reads:
            if isinstance(dep, MemoryDep):
                read_counts[dep.name] = read_counts.get(dep.name, 0) + 1
    planned: dict[str, list[LXRelayoutPlan]] = {}

    for consumer in graph.operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        if not _is_matmul_op(consumer):
            continue
        if _has_shared_weight_unit_bmm_info(getattr(consumer.data, "op_info", None)):
            continue
        reads = list(op_read_writes(consumer).reads)
        if any(not isinstance(dep, MemoryDep) or dep.is_indirect() for dep in reads):
            continue
        if len(reads) != 2:
            continue
        consumer_iteration_space = iteration_space_from_op(consumer)
        consumer_symbols = tuple(consumer_iteration_space)
        consumer_work_division: dict[sympy.Symbol, int] = {}
        if hasattr(consumer, "op_it_space_splits"):
            writes = list(op_read_writes(consumer).writes)
            if (
                len(writes) != 1
                or not isinstance(writes[0], MemoryDep)
                or writes[0].is_indirect()
            ):
                continue
            consumer_work_division = apply_splits_from_index_coeff(
                consumer.op_it_space_splits,
                writes[0].index,
                reads[0].index,
                consumer_iteration_space,
            )
        consumer_axis_splits = tuple(
            int(consumer_work_division.get(symbol, 1)) for symbol in consumer_symbols
        )
        for dep in reads:
            producer = operations.get(dep.name)
            if not isinstance(producer, ComputedBuffer) or producer is consumer:
                continue

            write_dep = _single_write_dep(producer, dep.name)
            if write_dep is None:
                continue

            producer_view, producer_has_partial, producer_representable = (
                _per_core_view_on_buf(producer, write_dep, dep.name, cache)
            )
            if producer_has_partial or not producer_representable:
                continue

            consumer_view, consumer_has_partial, consumer_representable = (
                _per_core_view_on_buf(consumer, dep, dep.name, cache)
            )
            if consumer_has_partial or not consumer_representable:
                continue
            if producer_view == consumer_view:
                continue

            producer_core_count = _op_num_cores(producer)
            consumer_core_count = _op_num_cores(consumer)
            producer_core_slices = _core_id_to_device_slice(
                producer_view, producer_core_count
            )
            if producer_core_slices is None:
                continue

            consumer_work_slice_dims = _work_slice_dims(consumer_view)
            consumer_core_slices = _core_id_to_device_slice(
                consumer_view, consumer_core_count
            )
            if consumer_core_slices is None:
                continue

            producer_work_slice_dims = _work_slice_dims(producer_view)
            relayout_dims = set(producer_work_slice_dims) | set(
                consumer_work_slice_dims
            )
            producer_core_slices, producer_work_slice_dims = _dense_view(
                producer_core_slices, producer_work_slice_dims, relayout_dims
            )
            consumer_core_slices, consumer_work_slice_dims = _dense_view(
                consumer_core_slices, consumer_work_slice_dims, relayout_dims
            )
            destination_size_ratio = _destination_size_ratio(
                producer_core_slices,
                producer_work_slice_dims,
                consumer_core_slices,
                consumer_work_slice_dims,
            )
            if destination_size_ratio is None:
                continue

            coordinates = try_device_coordinates(
                producer.get_layout().device_layout, dep, None
            )
            if coordinates is None:
                continue
            shuffle_geometry = _build_lx_shuffle_geometry(
                coordinates,
                consumer_symbols,
                consumer_axis_splits,
                producer_core_slices,
                producer_work_slice_dims,
                consumer_core_slices,
                consumer_work_slice_dims,
            )
            if shuffle_geometry is None:
                continue
            if not _shuffle_alignment_is_stable(
                graph,
                producer.get_layout().device_layout.device_size,
                coordinates,
                consumer_iteration_space,
                shuffle_geometry,
            ):
                continue

            source_distribution = _distribution_by_consumer_axis(
                coordinates,
                consumer_symbols,
                producer_core_slices,
                producer_work_slice_dims,
            )
            destination_distribution = _distribution_by_consumer_axis(
                coordinates,
                consumer_symbols,
                consumer_core_slices,
                consumer_work_slice_dims,
            )
            if source_distribution is None or destination_distribution is None:
                continue
            source_axis_map, source_axis_splits = source_distribution
            destination_axis_map, destination_axis_splits = destination_distribution

            def axis_splits_are_lowerable(axis_splits: dict[str, int]) -> bool:
                for axis, split in axis_splits.items():
                    try:
                        extent = int(
                            consumer_iteration_space[consumer_symbols[int(axis)]]
                        )
                        split = int(split)
                    except (IndexError, KeyError, TypeError, ValueError):
                        return False
                    if split <= 0 or extent % split != 0:
                        return False
                return True

            if (
                set(source_axis_splits) != set(destination_axis_splits)
                or any(
                    destination_axis_splits[axis] != consumer_axis_splits[int(axis)]
                    for axis in destination_axis_splits
                )
                or not (
                    axis_splits_are_lowerable(source_axis_splits)
                    and axis_splits_are_lowerable(destination_axis_splits)
                )
            ):
                continue

            if not _matmul_operand_source_good_for_lx_relayout(operations, producer):
                continue

            plan = LXRelayoutPlan(
                source_name=dep.name,
                consumer_name=consumer.get_name(),
                source_core_id_to_axis_slice=source_axis_map,
                destination_core_id_to_axis_slice=destination_axis_map,
                source_axis_splits=source_axis_splits,
                destination_axis_splits=destination_axis_splits,
                destination_size_ratio=destination_size_ratio,
                shuffle_geometry=shuffle_geometry,
            )
            planned.setdefault(plan.source_name, []).append(plan)

    # V1 deliberately materializes one consumer view per source. Sharing one
    # S2 allocation across independently scheduled consumers requires a wider
    # lifetime and scheduling contract.
    candidates = [
        plans[0]
        for source_name, plans in planned.items()
        if len(plans) == 1 and read_counts.get(source_name, 0) == 1
    ]
    # V1 does not model a synthetic S2 as the producer input of a later
    # relayout. Keep root edges and reject downstream links in a chain.
    relayout_consumers = {plan.consumer_name for plan in candidates}
    return [plan for plan in candidates if plan.source_name not in relayout_consumers]


def record_lx_relayout_plan(graph: GraphLowering, plan: LXRelayoutPlan) -> None:
    """Stamp a relayout plan after the source buffer is placed in LX."""

    consumer = _operations_by_name(graph).get(plan.consumer_name)
    if consumer is None:
        return
    plans = getattr(consumer, LX_RELAYOUT_ATTR, None)
    if not isinstance(plans, dict):
        plans = {}
        setattr(consumer, LX_RELAYOUT_ATTR, plans)
    plans[plan.source_name] = plan
