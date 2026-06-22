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

"""Experimental cross-core LX-to-LX relayout planner.

The regular LX planner already handles same-core scratchpad persistence.  This
module only records edges where producer and consumer slice the same buffer with
different per-core ownership, which requires explicit LX relayout.
"""

from __future__ import annotations

import dataclasses
import copy
import itertools
import math
from typing import Any

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation

from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _per_core_view_on_buf,
    device_coordinates,
)

logger = get_inductor_logger("lx_relayout")

LX_RELAYOUT_ATTR = "_spyre_lx_relayout_plan"
LX_RELAYOUT_OP_INFO_KEY = "lx_relayout"


@dataclasses.dataclass(frozen=True)
class LXRelayoutCell:
    cell_index: int
    source_core: int
    dest_core: int
    dim_starts: dict[str, int]
    dim_sizes: dict[str, int]
    bytes: int
    source_offset_bytes: int
    dest_offset_bytes: int


@dataclasses.dataclass(frozen=True)
class LXRelayoutSubview:
    starts: list[int]
    sizes: list[int]


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    source_name: str
    producer_name: str
    consumer_name: str
    device_sizes: list[int]
    device_stride_map: list[int]
    element_bytes: int
    producer_core_count: int
    consumer_core_count: int
    producer_region_bytes: int
    consumer_region_bytes: int
    cells: list[LXRelayoutCell]
    movement_subview: LXRelayoutSubview | None = None

    @property
    def bytes_moved(self) -> int:
        return sum(cell.bytes for cell in self.cells)


def _op_num_cores(op: Operation) -> int:
    splits: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    factors = [int(factor) for per_dim in splits for factor in per_dim.values()]
    return math.prod(factors) if factors else 1


def _single_write_dep(op: ComputedBuffer, buf_name: str) -> MemoryDep | None:
    matches = [
        dep
        for dep in op.get_read_writes().writes
        if isinstance(dep, MemoryDep) and dep.name == buf_name
    ]
    return matches[0] if len(matches) == 1 else None


def _device_layout_and_element_bytes(buf: Any) -> tuple[list[int], list[int], int]:
    layout = getattr(buf, "layout", None)
    dev_layout = getattr(layout, "device_layout", None)
    if dev_layout is None:
        raise ValueError("buffer-has-no-device-layout")
    device_sizes = [int(size) for size in dev_layout.device_size]
    device_stride_map = [int(stride) for stride in dev_layout.stride_map]
    element_bytes = num_bytes(dev_layout.device_dtype)
    return device_sizes, device_stride_map, element_bytes


def _normalize_view_splits(view: PerCoreView) -> dict[int, int]:
    return {int(dim): int(split) for dim, split in view.work_slice_dims}


def _owner_lookup(
    view: PerCoreView,
    core_count: int,
) -> tuple[dict[tuple[int, ...], int], str | None]:
    dims = tuple(int(dim) for dim, _split in view.work_slice_dims)
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    core_id = sympy.Symbol("core_id")
    owners: dict[tuple[int, ...], int] = {}
    for core in range(core_count):
        key = tuple(
            int(sympy.sympify(expr_by_dim.get(dim, 0)).subs(core_id, core))
            for dim in dims
        )
        if key in owners:
            return {}, "duplicate-owner"
        owners[key] = core
    return owners, None


def _owner_key(
    common_index: dict[int, int],
    owner_dims: tuple[int, ...],
    *,
    side_splits: dict[int, int],
    common_splits: dict[int, int],
) -> tuple[int, ...]:
    return tuple(
        int(common_index.get(dim, 0))
        * int(side_splits.get(dim, 1))
        // int(common_splits.get(dim, 1))
        for dim in owner_dims
    )


def _side_slice_geometry(
    *,
    view: PerCoreView,
    owner_key: tuple[int, ...],
    device_sizes: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return per-device-dim start/size for one side's owning core slice."""

    splits = _normalize_view_splits(view)
    owner_dims = tuple(int(dim) for dim, _split in view.work_slice_dims)
    slot_by_dim = dict(zip(owner_dims, owner_key))
    starts: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for dim, dim_size in enumerate(device_sizes):
        split = int(splits.get(dim, 1))
        chunk = int(dim_size) // split
        slot = int(slot_by_dim.get(dim, 0))
        starts[dim] = slot * chunk
        sizes[dim] = chunk
    return starts, sizes


def _default_subview(device_sizes: list[int]) -> LXRelayoutSubview:
    return LXRelayoutSubview(
        starts=[0 for _size in device_sizes],
        sizes=[int(size) for size in device_sizes],
    )


def _is_full_subview(
    subview: LXRelayoutSubview,
    *,
    device_sizes: list[int],
) -> bool:
    return (
        len(subview.starts) == len(device_sizes)
        and len(subview.sizes) == len(device_sizes)
        and all(int(start) == 0 for start in subview.starts)
        and [int(size) for size in subview.sizes]
        == [int(size) for size in device_sizes]
    )


def _validate_subview(
    subview: LXRelayoutSubview,
    *,
    device_sizes: list[int],
) -> str | None:
    if len(subview.starts) != len(device_sizes) or len(subview.sizes) != len(
        device_sizes
    ):
        return "subview-rank-mismatch"
    for start, size, device_size in zip(subview.starts, subview.sizes, device_sizes):
        start = int(start)
        size = int(size)
        device_size = int(device_size)
        if start < 0 or size <= 0 or start + size > device_size:
            return "subview-out-of-bounds"
    return None


def _subview_ranges_for_common_splits(
    *,
    subview: LXRelayoutSubview,
    device_sizes: list[int],
    common_splits: dict[int, int],
    moved_dims: tuple[int, ...],
) -> tuple[list[range], str | None]:
    """Return common-refinement index ranges clipped to a logical subview."""

    reason = _validate_subview(subview, device_sizes=device_sizes)
    if reason is not None:
        return [], reason

    restricted_dims = {
        dim
        for dim, (start, size, device_size) in enumerate(
            zip(subview.starts, subview.sizes, device_sizes)
        )
        if int(start) != 0 or int(size) != int(device_size)
    }
    missing_restricted_dims = restricted_dims - set(common_splits)
    if missing_restricted_dims:
        return [], "subview-requires-unsplit-device-dim"

    ranges: list[range] = []
    for dim in moved_dims:
        split = int(common_splits[dim])
        chunk = int(device_sizes[dim]) // split
        start = int(subview.starts[dim])
        end = start + int(subview.sizes[dim])
        if start % chunk != 0 or end % chunk != 0:
            return [], "subview-not-aligned-to-common-cell"
        ranges.append(range(start // chunk, end // chunk))
    return ranges, None


def _local_offset_bytes(
    *,
    cell_starts: dict[str, int],
    cell_sizes: dict[str, int],
    slice_starts: dict[int, int],
    slice_sizes: dict[int, int],
    element_bytes: int,
    device_stride_map: list[int] | None = None,
    dim_order: list[int] | None = None,
) -> tuple[int, str | None]:
    """Return the packed per-core byte offset for a cell.

    When a device stride map is available, preserve its fastest-to-slowest
    physical order while recomputing packed strides inside the per-core slice.
    Without a stride map, retain the legacy d0_, d1_, ... packed order.
    """

    dim_count = len(slice_sizes)
    if dim_order is not None:
        if sorted(dim_order) != list(range(dim_count)):
            return 0, "invalid-local-dim-order"
    elif device_stride_map is not None and len(device_stride_map) == dim_count:
        dim_order = sorted(
            range(dim_count), key=lambda dim: (device_stride_map[dim], dim)
        )
    else:
        dim_order = list(range(dim_count))

    stride_by_dim: dict[int, int] = {}
    stride = 1
    for dim in dim_order:
        stride_by_dim[dim] = stride
        stride *= int(slice_sizes[dim])

    offset_elements = 0
    for dim in range(dim_count):
        start = int(cell_starts[f"d{dim}_"])
        size = int(cell_sizes[f"d{dim}_"])
        slice_start = int(slice_starts[dim])
        slice_size = int(slice_sizes[dim])
        delta = start - slice_start
        if delta < 0 or delta + size > slice_size:
            return 0, "cell-outside-side-slice"
        offset_elements += delta * stride_by_dim[dim]
    return offset_elements * int(element_bytes), None


def _lx_relayout_v1_stride_map(
    *,
    device_sizes: list[int],
    device_stride_map: list[int],
    element_bytes: int,
) -> tuple[list[int], str | None]:
    """Return a physical device-dim stride map for whole-stick data move planning.

    Some BMM-shaped fixed-tile layouts carry an extra trailing host stride or a
    collapsed size-one dimension for the in-stick coordinate, e.g.
    ``device_size=[M, out_sticks, 1, 64]`` with
    ``stride_map=[512, 64, -1, 1]``.  For movement planning the final device
    dim is the physical stick element dim and nonpositive sentinel strides
    should not be considered faster than that stick dim.
    """

    if len(device_stride_map) == len(device_sizes) + 1:
        if int(device_stride_map[-1]) != 1:
            return [], "lx-relayout-v1-requires-device-stride-map"
        strides = [int(stride) for stride in device_stride_map[:-1]]
    elif len(device_stride_map) == len(device_sizes):
        strides = [int(stride) for stride in device_stride_map]
    else:
        return [], "lx-relayout-v1-requires-device-stride-map"

    if (
        strides
        and int(device_sizes[-1]) * int(element_bytes) == 128
        and int(strides[-1]) != 1
    ):
        strides[-1] = 1
    if any(int(stride) <= 0 for stride in strides):
        positive_strides = [int(stride) for stride in strides if int(stride) > 0]
        if not positive_strides:
            return [], "lx-relayout-v1-requires-device-stride-map"
        sentinel_stride = max(positive_strides) * max(math.prod(device_sizes), 1)
        strides = [
            int(stride) if int(stride) > 0 else int(sentinel_stride)
            for stride in strides
        ]
    return strides, None


def _lx_relayout_v1_lx_dim_order(
    *,
    device_sizes: list[int],
    device_stride_map: list[int],
    element_bytes: int,
) -> tuple[list[int], str | None]:
    """Return Deeptools' local LX order for fixed-tiled whole-stick moves.

    Fixed-tiled matrix tensors use one unit-stride stick-element dimension and
    one outer stick dimension.  DCC lowers LX views with stick elements fastest,
    non-stick logical dimensions next, and the outer stick dimension slowest.
    The global device stride map orders the outer stick before the row/M dim,
    which is correct for logical tensor indexing but wrong for per-core LX
    addresses consumed by data ops.
    """

    device_stride_map, reason = _lx_relayout_v1_stride_map(
        device_sizes=device_sizes,
        device_stride_map=device_stride_map,
        element_bytes=element_bytes,
    )
    if reason is not None:
        return [], reason
    fastest_dim = min(
        range(len(device_sizes)),
        key=lambda dim: (int(device_stride_map[dim]), dim),
    )
    if int(device_stride_map[fastest_dim]) != 1:
        return [], "lx-relayout-v1-requires-unit-stride-stick-dim"

    stick_elems = int(device_sizes[fastest_dim])
    stick_outer_dims = [
        dim
        for dim, stride in enumerate(device_stride_map)
        if dim != fastest_dim and int(stride) == stick_elems
    ]
    if len(stick_outer_dims) > 1:
        return [], "lx-relayout-v1-ambiguous-stick-outer-dim"

    stick_outer = set(stick_outer_dims)
    non_stick_dims = [
        dim
        for dim in range(len(device_sizes))
        if dim != fastest_dim and dim not in stick_outer
    ]
    non_stick_dims.sort(key=lambda dim: (int(device_stride_map[dim]), dim))
    stick_outer_dims.sort(key=lambda dim: (int(device_stride_map[dim]), dim))
    return [fastest_dim, *non_stick_dims, *stick_outer_dims], None


def _lx_relayout_v1_refined_splits(
    *,
    common_splits: dict[int, int],
    device_sizes: list[int],
    device_stride_map: list[int],
    element_bytes: int,
) -> tuple[dict[int, int], str | None]:
    """Refine common splits to physical whole-stick movements for v1 lowering."""

    device_stride_map, reason = _lx_relayout_v1_stride_map(
        device_sizes=device_sizes,
        device_stride_map=device_stride_map,
        element_bytes=element_bytes,
    )
    if reason is not None:
        return {}, reason
    fastest_dim = min(
        range(len(device_sizes)),
        key=lambda dim: (int(device_stride_map[dim]), dim),
    )
    if int(device_stride_map[fastest_dim]) != 1:
        return {}, "lx-relayout-v1-requires-unit-stride-stick-dim"
    if int(device_sizes[fastest_dim]) * int(element_bytes) != 128:
        return {}, "lx-relayout-v1-requires-128-byte-stick-dim"
    if int(common_splits.get(fastest_dim, 1)) != 1:
        return {}, "lx-relayout-v1-cannot-data move-split-stick-dim"

    refined = dict(common_splits)
    for dim, dim_size in enumerate(device_sizes):
        if dim == fastest_dim:
            continue
        refined[dim] = max(int(refined.get(dim, 1)), int(dim_size))
    return refined, None


def _view_region_bytes(
    view: PerCoreView,
    *,
    device_sizes: list[int],
    element_bytes: int,
) -> int:
    splits = _normalize_view_splits(view)
    elements = 1
    for dim, dim_size in enumerate(device_sizes):
        elements *= int(dim_size) // int(splits.get(dim, 1))
    return elements * int(element_bytes)


def _static_int(expr: Any) -> int | None:
    try:
        sym_expr = sympy.sympify(expr)
    except Exception:  # noqa: BLE001
        return None
    if getattr(sym_expr, "free_symbols", None):
        return None
    try:
        return int(sym_expr)
    except Exception:  # noqa: BLE001
        return None


def _range_size_by_symbol(ranges: dict[Any, Any]) -> dict[sympy.Symbol, int]:
    result: dict[sympy.Symbol, int] = {}
    for sym, value in ranges.items():
        size = _static_int(value)
        if size is None or size <= 0:
            continue
        key = sym if isinstance(sym, sympy.Symbol) else sympy.Symbol(str(sym))
        result[key] = size
    return result


def _single_free_symbol(expr: sympy.Expr) -> sympy.Symbol | None:
    free_symbols = list(getattr(expr, "free_symbols", set()))
    if len(free_symbols) != 1:
        return None
    return next(iter(free_symbols))


def _as_symbol_floor_div(expr: sympy.Expr) -> tuple[sympy.Symbol, int] | None:
    expr = sympy.simplify(expr)
    if expr.func.__name__ == "FloorDiv" and len(expr.args) == 2:
        sym = _single_free_symbol(sympy.sympify(expr.args[0]))
        divisor = _static_int(expr.args[1])
        if sym is not None and divisor is not None and divisor > 0:
            return sym, divisor

    if expr.func == sympy.floor and len(expr.args) == 1:
        arg = sympy.simplify(expr.args[0])
    else:
        arg = expr
    sym = _single_free_symbol(arg)
    if sym is None:
        return None
    coeff = sympy.simplify(arg.coeff(sym))
    if coeff.is_Rational and coeff.p == 1 and coeff.q > 0:
        return sym, int(coeff.q)
    return None


def _as_symbol_mod(expr: sympy.Expr) -> tuple[sympy.Symbol, int] | None:
    expr = sympy.simplify(expr)
    if expr.func != sympy.Mod or len(expr.args) != 2:
        return None
    sym = _single_free_symbol(sympy.sympify(expr.args[0]))
    modulus = _static_int(expr.args[1])
    if sym is not None and modulus is not None and modulus > 0:
        return sym, modulus
    return None


def _coord_subview_range(
    coord: sympy.Expr,
    *,
    ranges: dict[sympy.Symbol, int],
    device_size: int,
) -> tuple[int, int] | None:
    coord = sympy.simplify(coord)
    free_symbols: set[sympy.Symbol] = set(coord.free_symbols)
    if not free_symbols:
        start = _static_int(coord)
        if start is None or start < 0 or start >= int(device_size):
            return None
        return start, 1

    zero_subs = {sym: 0 for sym in free_symbols}
    const = _static_int(coord.subs(zero_subs))
    if const is None:
        return None
    residual = sympy.simplify(coord - const)

    sym = _single_free_symbol(residual)
    if sym is not None and sym in ranges and sympy.simplify(residual - sym) == 0:
        size = int(ranges[sym])
        if const < 0 or const + size > int(device_size):
            return None
        return const, size

    floor_div = _as_symbol_floor_div(residual)
    if floor_div is not None:
        sym, divisor = floor_div
        source_size = ranges.get(sym)
        if source_size is None or source_size % divisor != 0:
            return None
        size = source_size // divisor
        if const < 0 or const + size > int(device_size):
            return None
        return const, size

    mod = _as_symbol_mod(residual)
    if mod is not None:
        sym, modulus = mod
        source_size = ranges.get(sym)
        if source_size is None:
            return None
        if source_size <= modulus:
            size = source_size
        elif source_size % modulus == 0:
            size = modulus
        else:
            return None
        if const < 0 or const + size > int(device_size):
            return None
        return const, size

    return None


def _subview_from_device_coordinates(
    *,
    coordinates: list[sympy.Expr],
    ranges: dict[Any, Any],
    device_sizes: list[int],
) -> LXRelayoutSubview | None:
    range_sizes = _range_size_by_symbol(ranges)
    if len(coordinates) != len(device_sizes):
        return None

    starts: list[int] = []
    sizes: list[int] = []
    for coord, device_size in zip(coordinates, device_sizes):
        subrange = _coord_subview_range(
            sympy.sympify(coord),
            ranges=range_sizes,
            device_size=int(device_size),
        )
        if subrange is None:
            return None
        start, size = subrange
        starts.append(start)
        sizes.append(size)
    return LXRelayoutSubview(starts=starts, sizes=sizes)


def _movement_subview_from_read_dep(
    buf: Any,
    read_dep: MemoryDep,
    *,
    device_sizes: list[int],
) -> LXRelayoutSubview:
    layout = getattr(buf, "layout", None)
    dev_layout = getattr(layout, "device_layout", None)
    if dev_layout is None:
        return _default_subview(device_sizes)
    try:
        coords = device_coordinates(dev_layout, read_dep)
        subview = _subview_from_device_coordinates(
            coordinates=coords,
            ranges=getattr(read_dep, "ranges", {}),
            device_sizes=device_sizes,
        )
    except Exception:  # noqa: BLE001
        subview = None
    if subview is None:
        return _default_subview(device_sizes)
    return subview


def build_lx_relayout_cells(
    *,
    producer_view: PerCoreView,
    consumer_view: PerCoreView,
    device_sizes: list[int],
    device_stride_map: list[int] | None = None,
    element_bytes: int,
    producer_core_count: int,
    consumer_core_count: int,
    max_cells: int | None = None,
    lx_relayout_v1: bool = False,
    movement_subview: LXRelayoutSubview | None = None,
) -> tuple[list[LXRelayoutCell], str | None]:
    """Return common-refinement movement cells, or a skip reason."""

    subview = movement_subview or _default_subview(device_sizes)
    reason = _validate_subview(subview, device_sizes=device_sizes)
    if reason is not None:
        return [], reason

    producer_splits = _normalize_view_splits(producer_view)
    consumer_splits = _normalize_view_splits(consumer_view)
    moved_dims = tuple(sorted(set(producer_splits) | set(consumer_splits)))
    common_splits = {
        dim: math.lcm(producer_splits.get(dim, 1), consumer_splits.get(dim, 1))
        for dim in moved_dims
    }
    if lx_relayout_v1:
        common_splits, reason = _lx_relayout_v1_refined_splits(
            common_splits=common_splits,
            device_sizes=device_sizes,
            device_stride_map=device_stride_map or [],
            element_bytes=element_bytes,
        )
        if reason is not None:
            return [], reason
        moved_dims = tuple(sorted(common_splits))
        local_dim_order, reason = _lx_relayout_v1_lx_dim_order(
            device_sizes=device_sizes,
            device_stride_map=device_stride_map or [],
            element_bytes=element_bytes,
        )
        if reason is not None:
            return [], reason
    else:
        local_dim_order = None

    for dim, split in common_splits.items():
        if dim < 0 or dim >= len(device_sizes):
            return [], "view-dim-outside-device-layout"
        if int(device_sizes[dim]) % int(split) != 0:
            return [], "device-dim-not-divisible-by-common-split"

    producer_owners, reason = _owner_lookup(producer_view, producer_core_count)
    if reason is not None:
        return [], f"producer-{reason}"
    consumer_owners, reason = _owner_lookup(consumer_view, consumer_core_count)
    if reason is not None:
        return [], f"consumer-{reason}"

    producer_owner_dims = tuple(dim for dim, _split in producer_view.work_slice_dims)
    consumer_owner_dims = tuple(dim for dim, _split in consumer_view.work_slice_dims)
    cells: list[LXRelayoutCell] = []

    ranges, reason = _subview_ranges_for_common_splits(
        subview=subview,
        device_sizes=device_sizes,
        common_splits=common_splits,
        moved_dims=moved_dims,
    )
    if reason is not None:
        return [], reason
    cell_count = math.prod(len(r) for r in ranges) if ranges else 1
    if max_cells is not None and cell_count > max_cells:
        return [], "too-many-common-refinement-cells"

    iterator = math.prod(len(r) for r in ranges)
    if iterator == 0:
        return [], "empty-common-refinement"

    for cell_index, indices in enumerate(itertools.product(*ranges)):
        common_index = dict(zip(moved_dims, indices))
        producer_key = _owner_key(
            common_index,
            producer_owner_dims,
            side_splits=producer_splits,
            common_splits=common_splits,
        )
        consumer_key = _owner_key(
            common_index,
            consumer_owner_dims,
            side_splits=consumer_splits,
            common_splits=common_splits,
        )
        if producer_key not in producer_owners:
            return [], "producer-owner-map-incomplete"
        if consumer_key not in consumer_owners:
            return [], "consumer-owner-map-incomplete"

        starts: dict[str, int] = {}
        sizes: dict[str, int] = {}
        for dim, dim_size in enumerate(device_sizes):
            split = common_splits.get(dim, 1)
            chunk = int(dim_size) // int(split)
            starts[f"d{dim}_"] = int(common_index.get(dim, 0)) * chunk
            sizes[f"d{dim}_"] = chunk
        cell_bytes = math.prod(sizes.values()) * int(element_bytes)
        producer_starts, producer_sizes = _side_slice_geometry(
            view=producer_view,
            owner_key=producer_key,
            device_sizes=device_sizes,
        )
        consumer_starts, consumer_sizes = _side_slice_geometry(
            view=consumer_view,
            owner_key=consumer_key,
            device_sizes=device_sizes,
        )
        source_offset, reason = _local_offset_bytes(
            cell_starts=starts,
            cell_sizes=sizes,
            slice_starts=producer_starts,
            slice_sizes=producer_sizes,
            element_bytes=element_bytes,
            device_stride_map=device_stride_map,
            dim_order=local_dim_order,
        )
        if reason is not None:
            return [], f"producer-{reason}"
        dest_offset, reason = _local_offset_bytes(
            cell_starts=starts,
            cell_sizes=sizes,
            slice_starts=consumer_starts,
            slice_sizes=consumer_sizes,
            element_bytes=element_bytes,
            device_stride_map=device_stride_map,
            dim_order=local_dim_order,
        )
        if reason is not None:
            return [], f"consumer-{reason}"
        cells.append(
            LXRelayoutCell(
                cell_index=cell_index,
                source_core=producer_owners[producer_key],
                dest_core=consumer_owners[consumer_key],
                dim_starts=starts,
                dim_sizes=sizes,
                bytes=cell_bytes,
                source_offset_bytes=source_offset,
                dest_offset_bytes=dest_offset,
            )
        )

    return cells, None


def validate_lx_relayout_cell_coverage(
    cells: list[LXRelayoutCell],
    *,
    device_sizes: list[int],
    movement_subview: LXRelayoutSubview | None = None,
) -> str | None:
    """Check that movement cells tile the logical device rectangle exactly."""

    if not cells:
        return "no-cells"

    subview = movement_subview or _default_subview(device_sizes)
    reason = _validate_subview(subview, device_sizes=device_sizes)
    if reason is not None:
        return reason

    dims = len(device_sizes)
    boxes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    boundaries: list[set[int]] = [
        {int(start), int(start) + int(size)}
        for start, size in zip(subview.starts, subview.sizes)
    ]
    total_volume = 0
    for cell in cells:
        starts: list[int] = []
        ends: list[int] = []
        volume = 1
        for dim, _device_size in enumerate(device_sizes):
            key = f"d{dim}_"
            if key not in cell.dim_starts or key not in cell.dim_sizes:
                return "coverage-cell-missing-dim"
            start = int(cell.dim_starts[key])
            size = int(cell.dim_sizes[key])
            end = start + size
            subview_start = int(subview.starts[dim])
            subview_end = subview_start + int(subview.sizes[dim])
            if start < subview_start or size <= 0 or end > subview_end:
                return "coverage-cell-out-of-bounds"
            starts.append(start)
            ends.append(end)
            boundaries[dim].add(start)
            boundaries[dim].add(end)
            volume *= size
        boxes.append((tuple(starts), tuple(ends)))
        total_volume += volume

    expected_volume = math.prod(int(size) for size in subview.sizes)
    if total_volume != expected_volume:
        return "coverage-volume-mismatch"

    axes = [sorted(axis) for axis in boundaries]
    interval_counts = [max(len(axis) - 1, 0) for axis in axes]
    compressed_cell_count = math.prod(interval_counts)
    if compressed_cell_count > max(1_000_000, len(cells) * 16):
        return "coverage-validation-too-complex"

    axis_indices = [
        {boundary: index for index, boundary in enumerate(axis)} for axis in axes
    ]
    strides: list[int] = []
    stride = 1
    for count in reversed(interval_counts):
        strides.append(stride)
        stride *= count
    strides.reverse()

    occupied: set[int] = set()
    for box_starts, box_ends in boxes:
        cell_ranges = [
            range(
                axis_indices[dim][box_starts[dim]],
                axis_indices[dim][box_ends[dim]],
            )
            for dim in range(dims)
        ]
        for index_tuple in itertools.product(*cell_ranges):
            linear_index = sum(
                int(index) * strides[dim] for dim, index in enumerate(index_tuple)
            )
            if linear_index in occupied:
                return "coverage-cell-overlap"
            occupied.add(linear_index)
    return None


def _lx_relayout_v1_support_reason(cells: list[LXRelayoutCell]) -> str | None:
    """Return why the current Deeptools lx-relayout carrier cannot lower cells.

    The v1 Deeptools lowering emits whole-stick L3 load/store movements. It is
    only valid when each logical movement is also a non-overlapping contiguous
    destination byte range with 128-byte aligned source/destination addresses.
    """

    stick_bytes = 128
    destination_ranges_by_core: dict[int, list[tuple[int, int]]] = {}
    producer_base = int(config.lx_relayout_producer_lx_base)
    consumer_base = int(config.lx_relayout_consumer_lx_base)
    for cell in cells:
        if int(cell.bytes) <= 0 or int(cell.bytes) % stick_bytes != 0:
            return "lx-relayout-v1-requires-stick-sized-moves"
        source_lx_address = producer_base + int(cell.source_offset_bytes)
        dest_lx_address = consumer_base + int(cell.dest_offset_bytes)
        if source_lx_address % stick_bytes != 0:
            return "lx-relayout-v1-requires-stick-aligned-source-address"
        if dest_lx_address % stick_bytes != 0:
            return "lx-relayout-v1-requires-stick-aligned-destination-address"
        start = int(cell.dest_offset_bytes)
        end = start + int(cell.bytes)
        destination_ranges_by_core.setdefault(int(cell.dest_core), []).append(
            (start, end)
        )

    for ranges in destination_ranges_by_core.values():
        ranges.sort()
        previous_end: int | None = None
        for start, end in ranges:
            if previous_end is not None and start < previous_end:
                return "lx-relayout-v1-requires-contiguous-destination-cells"
            previous_end = end
    return None


def _slice_payload(starts: dict[str, int], sizes: dict[str, int]) -> dict[str, Any]:
    return {
        "starts": {dim: int(starts[dim]) for dim in sorted(starts)},
        "sizes": {dim: int(sizes[dim]) for dim in sorted(sizes)},
    }


def _byte_range_payload(start: int, size: int) -> dict[str, int]:
    return {"start": int(start), "end": int(start) + int(size)}


def _dataop_movement_payload(move: dict[str, Any]) -> dict[str, Any]:
    return {
        "moveIndex": int(move["move_index"]),
        "bytes": int(move["bytes"]),
        "source": {
            "core": int(move["source_core"]),
            "logicalSlice": move["source_slice"],
            "lxAddress": int(move["source_lx_address"]),
            "localByteRange": move["source_local_byte_range"],
            "lxByteRange": move["source_lx_byte_range"],
        },
        "destination": {
            "core": int(move["destination_core"]),
            "logicalSlice": move["destination_slice"],
            "lxAddress": int(move["destination_lx_address"]),
            "localByteRange": move["destination_local_byte_range"],
            "lxByteRange": move["destination_lx_byte_range"],
        },
    }


def _try_extend_logical_slice(
    current: dict[str, Any],
    next_slice: dict[str, Any],
) -> dict[str, Any] | None:
    current_starts = copy.deepcopy(current.get("starts", {}))
    current_sizes = copy.deepcopy(current.get("sizes", {}))
    next_starts = next_slice.get("starts", {})
    next_sizes = next_slice.get("sizes", {})
    if set(current_starts) != set(next_starts) or set(current_sizes) != set(next_sizes):
        return None

    extending_dim: str | None = None
    for dim in sorted(current_starts):
        current_start = int(current_starts[dim])
        current_size = int(current_sizes[dim])
        next_start = int(next_starts[dim])
        next_size = int(next_sizes[dim])
        if current_start == next_start and current_size == next_size:
            continue
        if next_start == current_start + current_size:
            if extending_dim is not None:
                return None
            extending_dim = dim
            current_sizes[dim] = current_size + next_size
            continue
        return None

    if extending_dim is None:
        return copy.deepcopy(current)
    return {"starts": current_starts, "sizes": current_sizes}


def _merge_logical_slice_for_dataop(
    current: dict[str, Any],
    next_slice: dict[str, Any],
) -> dict[str, Any]:
    merged = _try_extend_logical_slice(current, next_slice)
    if merged is not None:
        return merged
    # Deeptools lowers lx relayouts from byte ranges and core ids; the
    # logical slice is diagnostic metadata.  Keep it compact when adjacent byte
    # ranges do not form a single rectangular logical slice.
    return {"starts": {}, "sizes": {}, "coalesced": True}


def _coalesce_dataop_movements(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(movements) < 2:
        return movements

    sorted_movements = sorted(
        movements,
        key=lambda move: (
            int(move["source"]["core"]),
            int(move["destination"]["core"]),
            int(move["source"]["lxAddress"]),
            int(move["destination"]["lxAddress"]),
            int(move["moveIndex"]),
        ),
    )
    coalesced: list[dict[str, Any]] = []
    for movement in sorted_movements:
        movement = copy.deepcopy(movement)
        if not coalesced:
            coalesced.append(movement)
            continue

        previous = coalesced[-1]
        same_cores = int(previous["source"]["core"]) == int(
            movement["source"]["core"]
        ) and int(previous["destination"]["core"]) == int(
            movement["destination"]["core"]
        )
        previous_source = previous["source"]["lxByteRange"]
        movement_source = movement["source"]["lxByteRange"]
        previous_dest = previous["destination"]["lxByteRange"]
        movement_dest = movement["destination"]["lxByteRange"]
        contiguous = int(previous_source["end"]) == int(
            movement_source["start"]
        ) and int(previous_dest["end"]) == int(movement_dest["start"])
        if not same_cores or not contiguous:
            coalesced.append(movement)
            continue

        merged_source_slice = _merge_logical_slice_for_dataop(
            previous["source"]["logicalSlice"],
            movement["source"]["logicalSlice"],
        )
        merged_dest_slice = _merge_logical_slice_for_dataop(
            previous["destination"]["logicalSlice"],
            movement["destination"]["logicalSlice"],
        )
        previous["bytes"] = int(previous["bytes"]) + int(movement["bytes"])
        previous["source"]["localByteRange"]["end"] = movement["source"][
            "localByteRange"
        ]["end"]
        previous["source"]["lxByteRange"]["end"] = movement_source["end"]
        previous["source"]["logicalSlice"] = merged_source_slice
        previous["destination"]["localByteRange"]["end"] = movement["destination"][
            "localByteRange"
        ]["end"]
        previous["destination"]["lxByteRange"]["end"] = movement_dest["end"]
        previous["destination"]["logicalSlice"] = merged_dest_slice

    for index, movement in enumerate(coalesced):
        movement["moveIndex"] = index
    return coalesced


def _side_stride(
    previous: dict[str, Any], movement: dict[str, Any], side: str
) -> int | None:
    previous_side = previous[side]
    movement_side = movement[side]
    lx_stride = int(movement_side["lxAddress"]) - int(previous_side["lxAddress"])
    local_stride = int(movement_side["localByteRange"]["start"]) - int(
        previous_side["localByteRange"]["start"]
    )
    lx_range_stride = int(movement_side["lxByteRange"]["start"]) - int(
        previous_side["lxByteRange"]["start"]
    )
    if lx_stride != local_stride or lx_stride != lx_range_stride:
        return None
    if (
        int(movement_side["localByteRange"]["end"])
        - int(previous_side["localByteRange"]["end"])
        != lx_stride
    ):
        return None
    if (
        int(movement_side["lxByteRange"]["end"])
        - int(previous_side["lxByteRange"]["end"])
        != lx_stride
    ):
        return None
    return lx_stride


def _range_side_payload(side: dict[str, Any]) -> dict[str, Any]:
    return {
        "core": int(side["core"]),
        "logicalSlice": copy.deepcopy(side.get("logicalSlice", {})),
        "lxAddress": int(side["lxAddress"]),
        "localByteRange": copy.deepcopy(side["localByteRange"]),
        "lxByteRange": copy.deepcopy(side["lxByteRange"]),
    }


def _dataop_movement_ranges(movements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not movements:
        return []

    ranges: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    previous: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        current["rangeIndex"] = len(ranges)
        ranges.append(current)
        current = None

    for movement in movements:
        movement = copy.deepcopy(movement)
        if current is None:
            current = {
                "rangeIndex": -1,
                "moveIndex": int(movement["moveIndex"]),
                "count": 1,
                "bytesPerMove": int(movement["bytes"]),
                "sourceStrideBytes": 0,
                "destinationStrideBytes": 0,
                "source": _range_side_payload(movement["source"]),
                "destination": _range_side_payload(movement["destination"]),
                **(
                    {"relay": copy.deepcopy(movement["relay"])}
                    if "relay" in movement
                    else {}
                ),
            }
            previous = movement
            continue

        assert previous is not None
        source_stride = _side_stride(previous, movement, "source")
        destination_stride = _side_stride(previous, movement, "destination")
        expected_source_stride = int(current["sourceStrideBytes"])
        expected_destination_stride = int(current["destinationStrideBytes"])
        compatible = (
            int(movement["bytes"]) == int(current["bytesPerMove"])
            and int(movement["source"]["core"]) == int(current["source"]["core"])
            and int(movement["destination"]["core"])
            == int(current["destination"]["core"])
            and source_stride is not None
            and destination_stride is not None
            and int(movement["moveIndex"]) == int(previous["moveIndex"]) + 1
            and movement.get("relay") == current.get("relay")
            and (int(current["count"]) == 1 or source_stride == expected_source_stride)
            and (
                int(current["count"]) == 1
                or destination_stride == expected_destination_stride
            )
        )
        if not compatible:
            flush()
            current = {
                "rangeIndex": -1,
                "moveIndex": int(movement["moveIndex"]),
                "count": 1,
                "bytesPerMove": int(movement["bytes"]),
                "sourceStrideBytes": 0,
                "destinationStrideBytes": 0,
                "source": _range_side_payload(movement["source"]),
                "destination": _range_side_payload(movement["destination"]),
                **(
                    {"relay": copy.deepcopy(movement["relay"])}
                    if "relay" in movement
                    else {}
                ),
            }
            previous = movement
            continue

        assert source_stride is not None
        assert destination_stride is not None
        if int(current["count"]) == 1:
            current["sourceStrideBytes"] = int(source_stride)
            current["destinationStrideBytes"] = int(destination_stride)
        current["count"] = int(current["count"]) + 1
        previous = movement

    flush()
    return ranges


def _expand_dataop_movement_ranges(
    ranges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    movements: list[dict[str, Any]] = []
    for movement_range in ranges:
        count = int(movement_range.get("count") or 0)
        bytes_per_move = int(movement_range.get("bytesPerMove") or 0)
        source_stride = int(movement_range.get("sourceStrideBytes") or 0)
        destination_stride = int(movement_range.get("destinationStrideBytes") or 0)
        if count <= 0 or bytes_per_move <= 0:
            continue
        for offset in range(count):
            move = {
                "moveIndex": int(movement_range["moveIndex"]) + offset,
                "bytes": bytes_per_move,
                "source": copy.deepcopy(movement_range["source"]),
                "destination": copy.deepcopy(movement_range["destination"]),
            }
            if "relay" in movement_range:
                move["relay"] = copy.deepcopy(movement_range["relay"])
            for side_name, stride in (
                ("source", source_stride),
                ("destination", destination_stride),
            ):
                side = move[side_name]
                byte_offset = offset * stride
                side["lxAddress"] = int(side["lxAddress"]) + byte_offset
                side["localByteRange"]["start"] = (
                    int(side["localByteRange"]["start"]) + byte_offset
                )
                side["localByteRange"]["end"] = (
                    int(side["localByteRange"]["end"]) + byte_offset
                )
                side["lxByteRange"]["start"] = (
                    int(side["lxByteRange"]["start"]) + byte_offset
                )
                side["lxByteRange"]["end"] = (
                    int(side["lxByteRange"]["end"]) + byte_offset
                )
            movements.append(move)
    return movements


def _compact_lx_relayout_dataop_for_json(dataop: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(dataop)
    if config.lx_relayout_range_encoding and compact.get("movementRanges"):
        compact.pop("movements", None)
        lowering = compact.setdefault("lowering", {})
        lowering["rangeEncoded"] = True
    return compact


def _lx_relayout_dataop_payload(
    *,
    source_name: str,
    producer_name: str,
    consumer_name: str,
    producer_base: int,
    consumer_base: int,
    coverage: dict[str, Any],
    dependency_order: list[dict[str, Any]],
    movements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the backend-facing Deeptools data-op JSON form."""

    dataop_movements = _coalesce_dataop_movements(
        [_dataop_movement_payload(move) for move in movements]
    )
    return {
        "op": {"name": "STCDPOpLx"},
        "schemaVersion": 0,
        "sourceName": source_name,
        "producer": producer_name,
        "consumer": consumer_name,
        "producerLxBase": int(producer_base),
        "consumerLxBase": int(consumer_base),
        "coverage": coverage,
        "dependencyOrder": [
            {
                "order": int(row["order"]),
                "kind": row["kind"],
                **({"op": row["op"]} if "op" in row else {}),
                **({"sourceName": row["source_name"]} if "source_name" in row else {}),
                **({"primitive": row["primitive"]} if "primitive" in row else {}),
                **(
                    {"cellCount": int(row["cell_count"])} if "cell_count" in row else {}
                ),
            }
            for row in dependency_order
        ],
        "lowering": {
            "strategy": "explicit_lx_copy_via_l3",
            "addressUnits": "bytes",
            "coalescedMovements": len(dataop_movements),
            "sourceMovements": len(movements),
            "movementRanges": len(_dataop_movement_ranges(dataop_movements)),
            "rangeEncoded": False,
        },
        "movements": dataop_movements,
        "movementRanges": _dataop_movement_ranges(dataop_movements),
    }


def build_lx_relayout_metadata(plan: LXRelayoutPlan) -> dict[str, Any]:
    """Build torch-spyre-side metadata for a STCDPOpLx range data-op."""

    producer_base = int(config.lx_relayout_producer_lx_base)
    consumer_base = int(config.lx_relayout_consumer_lx_base)
    movements: list[dict[str, Any]] = []
    for cell in plan.cells:
        source_lx_address = producer_base + int(cell.source_offset_bytes)
        dest_lx_address = consumer_base + int(cell.dest_offset_bytes)
        logical_slice = _slice_payload(cell.dim_starts, cell.dim_sizes)
        movements.append(
            {
                "move_index": int(cell.cell_index),
                "bytes": int(cell.bytes),
                "source_core": int(cell.source_core),
                "source_slice": logical_slice,
                "source_lx_address": source_lx_address,
                "source_local_byte_range": _byte_range_payload(
                    cell.source_offset_bytes,
                    cell.bytes,
                ),
                "source_lx_byte_range": _byte_range_payload(
                    source_lx_address,
                    cell.bytes,
                ),
                "destination_core": int(cell.dest_core),
                "destination_slice": logical_slice,
                "destination_lx_address": dest_lx_address,
                "destination_local_byte_range": _byte_range_payload(
                    cell.dest_offset_bytes,
                    cell.bytes,
                ),
                "destination_lx_byte_range": _byte_range_payload(
                    dest_lx_address,
                    cell.bytes,
                ),
            }
        )

    coverage: dict[str, Any] = {
        "device_sizes": [int(size) for size in plan.device_sizes],
        "status": validate_lx_relayout_cell_coverage(
            plan.cells,
            device_sizes=plan.device_sizes,
            movement_subview=plan.movement_subview,
        )
        or "complete",
    }
    if plan.movement_subview is not None and not _is_full_subview(
        plan.movement_subview,
        device_sizes=plan.device_sizes,
    ):
        coverage["subview"] = {
            "starts": [int(start) for start in plan.movement_subview.starts],
            "sizes": [int(size) for size in plan.movement_subview.sizes],
        }
    dependency_order: list[dict[str, Any]] = [
        {
            "order": 0,
            "kind": "producer_lx_write_before_relayout",
            "op": plan.producer_name,
            "source_name": plan.source_name,
        },
        {
            "order": 1,
            "kind": "lx_relayout",
            "primitive": "lx_relayout_v0",
            "cell_count": len(plan.cells),
        },
        {
            "order": 2,
            "kind": "consumer_lx_read_after_relayout",
            "op": plan.consumer_name,
            "source_name": plan.source_name,
        },
    ]
    metadata = {
        "primitive": "lx_relayout_v0",
        "schema_version": 0,
        "source_name": plan.source_name,
        "producer": plan.producer_name,
        "consumer": plan.consumer_name,
        "producer_lx_base": producer_base,
        "consumer_lx_base": consumer_base,
        "coverage": coverage,
        "dependency_order": dependency_order,
        "deeptools_dataop": _lx_relayout_dataop_payload(
            source_name=plan.source_name,
            producer_name=plan.producer_name,
            consumer_name=plan.consumer_name,
            producer_base=producer_base,
            consumer_base=consumer_base,
            coverage=coverage,
            dependency_order=dependency_order,
            movements=movements,
        ),
    }
    if "subview" in coverage:
        metadata["logical_subview"] = copy.deepcopy(coverage["subview"])
    return metadata


def _planned_payload(plan: LXRelayoutPlan) -> dict[str, Any]:
    payload = {
        "source_name": plan.source_name,
        "producer": plan.producer_name,
        "consumer": plan.consumer_name,
        "device_sizes": plan.device_sizes,
        "device_stride_map": plan.device_stride_map,
        "element_bytes": plan.element_bytes,
        "producer_core_count": plan.producer_core_count,
        "consumer_core_count": plan.consumer_core_count,
        "producer_region_bytes": plan.producer_region_bytes,
        "consumer_region_bytes": plan.consumer_region_bytes,
        "cell_count": len(plan.cells),
        "bytes_moved": plan.bytes_moved,
    }
    if plan.movement_subview is not None and not _is_full_subview(
        plan.movement_subview,
        device_sizes=plan.device_sizes,
    ):
        payload["movement_subview"] = {
            "starts": [int(start) for start in plan.movement_subview.starts],
            "sizes": [int(size) for size in plan.movement_subview.sizes],
        }
    metadata = build_lx_relayout_metadata(plan)
    metadata["deeptools_dataop"] = _compact_lx_relayout_dataop_for_json(
        metadata["deeptools_dataop"]
    )
    payload["lx_relayout"] = metadata
    return payload


def plan_lx_relayout_edge(
    graph: GraphLowering,
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    read_dep: MemoryDep,
    *,
    cache: dict | None = None,
) -> LXRelayoutPlan | None:
    write_dep = _single_write_dep(producer, read_dep.name)
    if write_dep is None:
        return None

    producer_view, producer_partial = _per_core_view_on_buf(
        producer,
        write_dep,
        read_dep.name,
        cache=cache,
    )
    consumer_view, _consumer_partial = _per_core_view_on_buf(
        consumer,
        read_dep,
        read_dep.name,
        cache=cache,
    )
    if producer_partial:
        return None
    if producer_view == consumer_view:
        return None

    try:
        buf = graph.get_buffer(read_dep.name)
        device_sizes, device_stride_map, element_bytes = (
            _device_layout_and_element_bytes(buf)
        )
        movement_subview = _movement_subview_from_read_dep(
            buf,
            read_dep,
            device_sizes=device_sizes,
        )
        cells, reason = build_lx_relayout_cells(
            producer_view=producer_view,
            consumer_view=consumer_view,
            device_sizes=device_sizes,
            device_stride_map=device_stride_map,
            element_bytes=element_bytes,
            producer_core_count=_op_num_cores(producer),
            consumer_core_count=_op_num_cores(consumer),
            max_cells=config.lx_relayout_max_cells,
            lx_relayout_v1=True,
            movement_subview=movement_subview,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "lx_relayout skipped %s -> %s for %s: %s",
            producer.get_name(),
            consumer.get_name(),
            read_dep.name,
            type(exc).__name__,
        )
        return None
    if reason is not None:
        logger.debug(
            "lx_relayout skipped %s -> %s for %s: %s",
            producer.get_name(),
            consumer.get_name(),
            read_dep.name,
            reason,
        )
        return None
    reason = _lx_relayout_v1_support_reason(cells)
    if reason is not None:
        logger.debug(
            "lx_relayout skipped %s -> %s for %s: %s",
            producer.get_name(),
            consumer.get_name(),
            read_dep.name,
            reason,
        )
        return None

    return LXRelayoutPlan(
        source_name=read_dep.name,
        producer_name=producer.get_name(),
        consumer_name=consumer.get_name(),
        device_sizes=device_sizes,
        device_stride_map=device_stride_map,
        element_bytes=element_bytes,
        producer_core_count=_op_num_cores(producer),
        consumer_core_count=_op_num_cores(consumer),
        producer_region_bytes=_view_region_bytes(
            producer_view,
            device_sizes=device_sizes,
            element_bytes=element_bytes,
        ),
        consumer_region_bytes=_view_region_bytes(
            consumer_view,
            device_sizes=device_sizes,
            element_bytes=element_bytes,
        ),
        cells=cells,
        movement_subview=movement_subview,
    )


def _attach_plan_to_consumer(consumer: ComputedBuffer, plan: LXRelayoutPlan) -> None:
    plan_payload = _planned_payload(plan)
    existing_move_info = getattr(consumer, LX_RELAYOUT_ATTR, None)
    move_info = existing_move_info if isinstance(existing_move_info, dict) else {}
    move_info[plan.source_name] = plan_payload
    setattr(consumer, LX_RELAYOUT_ATTR, move_info)

    data = getattr(consumer, "data", None)
    op_info = getattr(data, "op_info", None)
    if isinstance(op_info, dict):
        op_move_info = op_info.setdefault(LX_RELAYOUT_OP_INFO_KEY, {})
        if not isinstance(op_move_info, dict):
            op_move_info = {}
            op_info[LX_RELAYOUT_OP_INFO_KEY] = op_move_info
        op_move_info[plan.source_name] = plan_payload


def plan_lx_relayouts(graph: GraphLowering) -> None:
    if not config.lx_planner_relayout:
        return

    name_to_op = {
        op.get_name(): op for op in graph.operations if isinstance(op, ComputedBuffer)
    }
    cache: dict = {}
    plans: list[LXRelayoutPlan] = []
    edge_count = 0
    for consumer in graph.operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        for dep in consumer.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            producer = name_to_op.get(dep.name)
            if producer is None:
                continue
            edge_count += 1
            plan = plan_lx_relayout_edge(
                graph,
                producer,
                consumer,
                dep,
                cache=cache,
            )
            if plan is None:
                continue
            plans.append(plan)
            _attach_plan_to_consumer(consumer, plan)

    if edge_count:
        logger.info(
            "lx_relayout summary edges=%d planned=%d bytes=%d realize=%s",
            edge_count,
            len(plans),
            sum(plan.bytes_moved for plan in plans),
            config.lx_planner_relayout_realize,
        )
