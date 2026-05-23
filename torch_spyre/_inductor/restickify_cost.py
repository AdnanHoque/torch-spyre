# Copyright 2025 The Torch-Spyre Authors.
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

"""Pure cost-model core for restickify ring locality and transfer planning.

This module is intentionally free of any ``torch``/``sympy``/package
dependencies so it can be imported and unit-tested in isolation. All inputs
are plain dicts/lists/ints/strs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def ring_distance(src_core: int, dst_core: int, ring_size: int) -> int:
    """Shortest distance between two physical cores on a bidirectional ring."""
    if ring_size <= 0:
        raise ValueError(f"ring_size must be positive, got {ring_size}")
    src = src_core % ring_size
    dst = dst_core % ring_size
    delta = abs(src - dst)
    return min(delta, ring_size - delta)


def materialize_default_core_mapping(
    dim_order: Sequence[Any],
    dim_splits: Mapping[Any, int],
    num_cores: int | None = None,
) -> dict[str, dict[str, int]]:
    """Materialize the default core_id -> work-slice map used by SuperDSC."""
    dims = [str(dim) for dim in dim_order]
    splits = {str(dim): int(split) for dim, split in dim_splits.items()}
    if num_cores is None:
        num_cores = math.prod(splits.get(dim, 1) for dim in dims)

    core_mapping: dict[str, dict[str, int]] = {}
    for core_id in range(num_cores):
        inner_product = 1
        per_dim: dict[str, int] = {}
        for dim in dims:
            split = splits.get(dim, 1)
            if split <= 0:
                raise ValueError(f"split for {dim} must be positive, got {split}")
            if split == 1:
                slice_idx = 0
            elif inner_product == 1:
                slice_idx = core_id % split
            else:
                slice_idx = (core_id // inner_product) % split
            per_dim[dim] = int(slice_idx)
            inner_product *= split
        core_mapping[str(core_id)] = per_dim
    return core_mapping


def normalize_core_mapping(
    raw: Mapping[Any, Mapping[Any, Any]],
) -> dict[str, dict[str, int]]:
    """Coerce a core mapping into ``{str: {str: int}}`` form."""
    return {
        str(core_id): {str(dim): int(slice_idx) for dim, slice_idx in per_dim.items()}
        for core_id, per_dim in raw.items()
    }


def _core_rectangles(
    iteration_sizes: Mapping[str, int],
    split_factors: Mapping[str, int],
    core_mapping: Mapping[str, Mapping[str, int]],
) -> dict[int, dict[str, tuple[int, int]]]:
    """Compute the per-core ``[start, end)`` rectangle for each dimension."""
    rectangles: dict[int, dict[str, tuple[int, int]]] = {}
    for core_id_str, per_dim in core_mapping.items():
        core_id = int(core_id_str)
        rect: dict[str, tuple[int, int]] = {}
        for sym, size in iteration_sizes.items():
            split = int(split_factors.get(sym, 1))
            if split <= 0:
                raise ValueError(f"split for {sym} must be positive, got {split}")
            if size % split != 0:
                raise ValueError(
                    f"size for {sym} ({size}) is not divisible by split {split}"
                )
            slice_idx = int(per_dim.get(sym, 0))
            if slice_idx < 0 or slice_idx >= split:
                raise ValueError(
                    f"slice {slice_idx} for {sym} outside split factor {split}"
                )
            chunk = size // split
            rect[sym] = (slice_idx * chunk, (slice_idx + 1) * chunk)
        rectangles[core_id] = rect
    return rectangles


def _intersection_volume(
    producer_rect: Mapping[str, tuple[int, int]],
    restickify_rect: Mapping[str, tuple[int, int]],
    restickify_to_producer: Mapping[str, str],
) -> int:
    """Element overlap between two rectangles via a symbol correspondence."""
    volume = 1
    for restickify_sym, (rest_start, rest_end) in restickify_rect.items():
        producer_sym = restickify_to_producer.get(restickify_sym)
        if producer_sym is None:
            continue
        prod_start, prod_end = producer_rect[producer_sym]
        overlap = max(0, min(prod_end, rest_end) - max(prod_start, rest_start))
        if overlap == 0:
            return 0
        volume *= overlap
    return volume


def _total_elements(iteration_sizes: Mapping[str, int]) -> int:
    """Product of all iteration-space dimension sizes."""
    return math.prod(iteration_sizes.values())


def estimate_byte_hops_from_mappings(
    producer_sizes: Mapping[str, int],
    restickify_sizes: Mapping[str, int],
    producer_splits: Mapping[str, int],
    restickify_splits: Mapping[str, int],
    producer_mapping: Mapping[str, Mapping[str, int]],
    restickify_mapping: Mapping[str, Mapping[str, int]],
    symbol_map: Mapping[str, str],
    elem_size_bytes: int,
    ring_size: int,
) -> tuple[int, int, int]:
    """Return ``(bytes_moved, byte_hops, max_hops)`` for two core mappings."""
    producer_rects = _core_rectangles(
        producer_sizes, producer_splits, normalize_core_mapping(producer_mapping)
    )
    restickify_rects = _core_rectangles(
        restickify_sizes,
        restickify_splits,
        normalize_core_mapping(restickify_mapping),
    )

    bytes_moved = _total_elements(restickify_sizes) * elem_size_bytes
    byte_hops = 0
    max_hops = 0
    for producer_core, producer_rect in producer_rects.items():
        for restickify_core, restickify_rect in restickify_rects.items():
            overlap_elements = _intersection_volume(
                producer_rect, restickify_rect, symbol_map
            )
            if overlap_elements == 0:
                continue
            hops = ring_distance(producer_core, restickify_core, ring_size)
            max_hops = max(max_hops, hops)
            byte_hops += overlap_elements * elem_size_bytes * hops
    return bytes_moved, byte_hops, max_hops


def producer_aligned_dim_order(
    restickify_dims: Sequence[Any],
    producer_splits: Mapping[str, int],
    symbol_map: Mapping[str, str],
) -> tuple[list[Any] | None, str | None]:
    """Prioritize the restickify dim mapped to the producer's dominant split."""
    scored_dims: list[tuple[Any, int]] = []
    for dim in restickify_dims:
        producer_sym = symbol_map.get(str(dim))
        split = producer_splits.get(producer_sym, 1) if producer_sym else 1
        if split > 1:
            scored_dims.append((dim, split))

    if not scored_dims:
        return None, "producer-has-no-mapped-split"

    max_split = max(split for _, split in scored_dims)
    dominant_dims = [dim for dim, split in scored_dims if split == max_split]
    if len(dominant_dims) != 1:
        return None, "ambiguous-producer-split"

    dominant = dominant_dims[0]
    return [dominant, *(dim for dim in restickify_dims if dim != dominant)], None


def build_transfer_plan(
    producer_sizes: Mapping[str, int],
    consumer_sizes: Mapping[str, int],
    producer_splits: Mapping[str, int],
    consumer_splits: Mapping[str, int],
    producer_mapping: Mapping[str, Mapping[str, int]],
    consumer_mapping: Mapping[str, Mapping[str, int]],
    symbol_map: Mapping[str, str],
    ring_size: int,
) -> tuple[list[dict[str, int]], dict[str, int]]:
    """Return the per-tile cross-core transfer plan for a SAME-LAYOUT re-partition.

    For each (producer_core, consumer_core) pair that share overlapping elements,
    emit a transfer ``{src_core, dst_core, elements, hops}``. Same layout => pure
    address remap, no transpose. Returns ``(transfers, summary)`` where ``summary``
    has total_transfers, local_elements (hops==0), remote_elements (hops>0),
    total_byte_hops (uses elem_size_bytes=1; caller scales) and max_hops.
    """
    producer_rects = _core_rectangles(
        producer_sizes, producer_splits, normalize_core_mapping(producer_mapping)
    )
    consumer_rects = _core_rectangles(
        consumer_sizes, consumer_splits, normalize_core_mapping(consumer_mapping)
    )

    transfers: list[dict[str, int]] = []
    local_elements = 0
    remote_elements = 0
    total_byte_hops = 0
    max_hops = 0
    for producer_core, producer_rect in producer_rects.items():
        for consumer_core, consumer_rect in consumer_rects.items():
            elements = _intersection_volume(
                producer_rect, consumer_rect, symbol_map
            )
            if elements == 0:
                continue
            hops = ring_distance(producer_core, consumer_core, ring_size)
            transfers.append(
                {
                    "src_core": producer_core,
                    "dst_core": consumer_core,
                    "elements": elements,
                    "hops": hops,
                }
            )
            if hops == 0:
                local_elements += elements
            else:
                remote_elements += elements
            total_byte_hops += elements * hops
            max_hops = max(max_hops, hops)

    summary = {
        "total_transfers": len(transfers),
        "local_elements": local_elements,
        "remote_elements": remote_elements,
        "total_byte_hops": total_byte_hops,
        "max_hops": max_hops,
    }
    return transfers, summary
