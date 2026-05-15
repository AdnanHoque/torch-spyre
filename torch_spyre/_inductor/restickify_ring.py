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

"""Helpers for conservative producer-aligned restickify core mappings."""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer

from .pass_utils import (
    apply_splits_from_index_coeff,
    concretize_expr,
    iteration_space_from_op,
)

CORE_MAPPING_OVERRIDE_ATTR = "_spyre_core_id_to_work_slice_override"
CORE_MAPPING_OVERRIDE_OP_INFO_KEY = "core_id_to_work_slice_override"


@dataclasses.dataclass(frozen=True)
class RestickifyRingEstimate:
    restickify_name: str
    producer_name: str
    consumer_names: list[str]
    bytes_moved: int
    byte_hops: int
    avg_hops: float
    max_hops: int
    producer_splits: dict[str, int]
    restickify_splits: dict[str, int]
    symbol_map: dict[str, str]
    skip_reason: str | None = None


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


def materialize_k_fast_core_mapping(
    iteration_sizes: Mapping[str, int],
    dim_splits: Mapping[str, int],
    num_cores: int | None = None,
) -> dict[str, dict[str, int]]:
    """Materialize SuperDSC's k-fast core mapping for matmul producers."""
    dims = list(iteration_sizes.keys())
    if len(dims) < 3:
        return materialize_default_core_mapping(dims, dim_splits, num_cores)
    return materialize_default_core_mapping(
        [dims[-1], *dims[:-1]],
        dim_splits,
        num_cores,
    )


def normalize_core_mapping(
    raw: Mapping[Any, Mapping[Any, Any]],
) -> dict[str, dict[str, int]]:
    return {
        str(core_id): {str(dim): int(slice_idx) for dim, slice_idx in per_dim.items()}
        for core_id, per_dim in raw.items()
    }


def is_restickify_op(op: ComputedBuffer) -> bool:
    """Return true for compiler-inserted ``spyre.restickify`` buffers."""
    origins = getattr(op, "origins", None)
    if not origins:
        return False
    for origin in origins:
        if isinstance(origin, torch.fx.Node) and (
            origin.target is torch.ops.spyre.restickify.default
        ):
            return True
    return False


def build_name_to_op_map(operations) -> dict[str, ComputedBuffer]:
    return {
        op.get_name(): op for op in operations if isinstance(op, ComputedBuffer)
    }


def build_consumers_of(operations) -> dict[str, list[ComputedBuffer]]:
    consumers: dict[str, list[ComputedBuffer]] = {}
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        for dep in op.get_read_writes().reads:
            if isinstance(dep, MemoryDep):
                consumers.setdefault(dep.name, []).append(op)
    return consumers


def producer_for_restickify(
    restickify_op: ComputedBuffer,
    name_to_op: Mapping[str, ComputedBuffer],
) -> tuple[tuple[ComputedBuffer, MemoryDep] | None, str | None]:
    reads = [
        dep
        for dep in restickify_op.get_read_writes().reads
        if isinstance(dep, MemoryDep)
    ]
    if len(reads) != 1:
        return None, "multi-producer-or-no-input"

    read_dep = reads[0]
    producer = name_to_op.get(read_dep.name)
    if producer is None:
        return None, "graph-input-or-missing-producer"
    return (producer, read_dep), None


def op_iteration_sizes(op: ComputedBuffer) -> dict[str, int]:
    return {
        str(sym): int(concretize_expr(size))
        for sym, size in iteration_space_from_op(op).items()
    }


def decode_op_splits(op: ComputedBuffer) -> dict[str, int]:
    """Decode coeff-keyed ``op_it_space_splits`` into scheduler-symbol splits."""
    it_space = iteration_space_from_op(op)
    splits: dict[Any, int] = {sym: 1 for sym in it_space}
    encoded = getattr(op, "op_it_space_splits", None)
    if encoded is not None:
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next(
            (dep.index for dep in rw.reads if isinstance(dep, MemoryDep)),
            write_index,
        )
        splits = apply_splits_from_index_coeff(
            encoded, write_index, read_index, it_space
        )
    return {str(sym): int(splits.get(sym, 1)) for sym in it_space}


def split_dims_only(splits: Mapping[str, int]) -> dict[str, int]:
    return {sym: split for sym, split in splits.items() if split > 1}


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


def extract_strides(index_expr, var_names) -> dict[str, int]:
    """Return per-symbol stride coefficients for a linear index expression."""
    if index_expr is None:
        return {}
    expr = sympy.sympify(index_expr)
    out: dict[str, int] = {}
    for var in var_names:
        try:
            coeff = expr.coeff(var)
            if coeff == 0:
                continue
            out[str(var)] = int(concretize_expr(coeff))
        except (TypeError, ValueError):
            continue
    return out


def build_symbol_correspondence(
    producer_strides: Mapping[str, int],
    consumer_strides: Mapping[str, int],
) -> tuple[dict[str, str], str | None]:
    """Map consumer symbols to producer symbols by matching buffer strides."""
    producer_counts = Counter(producer_strides.values())
    consumer_counts = Counter(consumer_strides.values())
    if any(count > 1 for count in producer_counts.values()):
        return {}, "ambiguous-producer-stride"
    if any(count > 1 for count in consumer_counts.values()):
        return {}, "ambiguous-consumer-stride"

    producer_sym_by_stride = {
        stride: sym for sym, stride in producer_strides.items()
    }
    mapping: dict[str, str] = {}
    for consumer_sym, consumer_stride in consumer_strides.items():
        producer_sym = producer_sym_by_stride.get(consumer_stride)
        if producer_sym is not None:
            mapping[consumer_sym] = producer_sym
    return mapping, None


def restickify_symbol_map(
    producer: ComputedBuffer,
    restickify_op: ComputedBuffer,
    read_dep: MemoryDep,
) -> tuple[dict[str, str], str | None]:
    producer_writes = [
        dep
        for dep in producer.get_read_writes().writes
        if isinstance(dep, MemoryDep)
    ]
    if len(producer_writes) != 1:
        return {}, "producer-write-unsupported"

    producer_write = producer_writes[0]
    producer_strides = extract_strides(producer_write.index, producer_write.var_names)
    restickify_strides = extract_strides(read_dep.index, read_dep.var_names)
    symbol_map, reason = build_symbol_correspondence(
        producer_strides, restickify_strides
    )
    if reason is not None:
        return {}, reason

    producer_sizes = op_iteration_sizes(producer)
    restickify_sizes = op_iteration_sizes(restickify_op)
    mapped_producer_symbols = set(symbol_map.values())
    missing_rest = [
        sym
        for sym, size in restickify_sizes.items()
        if size > 1 and sym not in symbol_map
    ]
    missing_prod = [
        sym
        for sym, size in producer_sizes.items()
        if size > 1 and sym not in mapped_producer_symbols
    ]
    if missing_rest or missing_prod:
        return {}, "incomplete-symbol-map"

    for restickify_sym, producer_sym in symbol_map.items():
        if restickify_sizes[restickify_sym] != producer_sizes[producer_sym]:
            return {}, "mismatched-symbol-size"
    return symbol_map, None


def _mapping_for_op(
    op: ComputedBuffer,
    iteration_sizes: Mapping[str, int],
    split_factors: Mapping[str, int],
    k_fast_ops: Sequence[Any] | None = None,
) -> dict[str, dict[str, int]]:
    override = getattr(op, CORE_MAPPING_OVERRIDE_ATTR, None)
    if override is not None:
        return normalize_core_mapping(override)
    if (
        k_fast_ops is not None
        and op in k_fast_ops
        and len(iteration_sizes) >= 3
        and split_factors.get(list(iteration_sizes)[-1], 1) > 1
    ):
        return materialize_k_fast_core_mapping(
            iteration_sizes,
            split_factors,
            math.prod(split_factors.values()),
        )
    return materialize_default_core_mapping(
        list(iteration_sizes.keys()),
        split_factors,
        math.prod(split_factors.values()),
    )


def _core_rectangles(
    iteration_sizes: Mapping[str, int],
    split_factors: Mapping[str, int],
    core_mapping: Mapping[str, Mapping[str, int]],
) -> dict[int, dict[str, tuple[int, int]]]:
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


def _element_size_bytes(op: ComputedBuffer) -> int:
    dtype = op.get_layout().dtype
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is not None:
        return int(itemsize)
    return int(torch.tensor([], dtype=dtype).element_size())


def _total_elements(iteration_sizes: Mapping[str, int]) -> int:
    return math.prod(iteration_sizes.values())


def _bytes_moved_or_zero(op: ComputedBuffer) -> int:
    try:
        return _total_elements(op_iteration_sizes(op)) * _element_size_bytes(op)
    except Exception:  # noqa: BLE001
        return 0


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


def estimate_restickify_ring_cost(
    restickify_op: ComputedBuffer,
    name_to_op: Mapping[str, ComputedBuffer],
    consumers_of: Mapping[str, list[ComputedBuffer]],
    ring_size: int,
    k_fast_ops: Sequence[Any] | None = None,
) -> RestickifyRingEstimate:
    restickify_name = restickify_op.get_name()
    consumer_names = [op.get_name() for op in consumers_of.get(restickify_name, [])]
    producer_info, reason = producer_for_restickify(restickify_op, name_to_op)
    bytes_moved = _bytes_moved_or_zero(restickify_op)
    if producer_info is None:
        return RestickifyRingEstimate(
            restickify_name=restickify_name,
            producer_name="<none>",
            consumer_names=consumer_names,
            bytes_moved=bytes_moved,
            byte_hops=0,
            avg_hops=0.0,
            max_hops=0,
            producer_splits={},
            restickify_splits={},
            symbol_map={},
            skip_reason=reason,
        )

    producer, read_dep = producer_info
    producer_name = producer.get_name()
    producer_splits = decode_op_splits(producer)
    restickify_splits = decode_op_splits(restickify_op)
    symbol_map, reason = restickify_symbol_map(producer, restickify_op, read_dep)
    if reason is not None:
        return RestickifyRingEstimate(
            restickify_name=restickify_name,
            producer_name=producer_name,
            consumer_names=consumer_names,
            bytes_moved=bytes_moved,
            byte_hops=0,
            avg_hops=0.0,
            max_hops=0,
            producer_splits=split_dims_only(producer_splits),
            restickify_splits=split_dims_only(restickify_splits),
            symbol_map={},
            skip_reason=reason,
        )

    try:
        producer_sizes = op_iteration_sizes(producer)
        restickify_sizes = op_iteration_sizes(restickify_op)
        elem_size = _element_size_bytes(restickify_op)
        producer_mapping = _mapping_for_op(
            producer,
            producer_sizes,
            producer_splits,
            k_fast_ops,
        )
        restickify_mapping = _mapping_for_op(
            restickify_op, restickify_sizes, restickify_splits
        )
        bytes_moved, byte_hops, max_hops = estimate_byte_hops_from_mappings(
            producer_sizes,
            restickify_sizes,
            producer_splits,
            restickify_splits,
            producer_mapping,
            restickify_mapping,
            symbol_map,
            elem_size,
            ring_size,
        )
    except Exception as exc:  # noqa: BLE001
        return RestickifyRingEstimate(
            restickify_name=restickify_name,
            producer_name=producer_name,
            consumer_names=consumer_names,
            bytes_moved=bytes_moved,
            byte_hops=0,
            avg_hops=0.0,
            max_hops=0,
            producer_splits=split_dims_only(producer_splits),
            restickify_splits=split_dims_only(restickify_splits),
            symbol_map=symbol_map,
            skip_reason=type(exc).__name__,
        )

    avg_hops = byte_hops / bytes_moved if bytes_moved else 0.0
    return RestickifyRingEstimate(
        restickify_name=restickify_name,
        producer_name=producer_name,
        consumer_names=consumer_names,
        bytes_moved=bytes_moved,
        byte_hops=byte_hops,
        avg_hops=avg_hops,
        max_hops=max_hops,
        producer_splits=split_dims_only(producer_splits),
        restickify_splits=split_dims_only(restickify_splits),
        symbol_map=symbol_map,
        skip_reason=None,
    )


def build_restickify_core_mapping_override(
    restickify_op: ComputedBuffer,
    name_to_op: Mapping[str, ComputedBuffer],
    k_fast_ops: Sequence[Any] | None = None,
) -> tuple[dict[str, dict[str, int]] | None, str | None]:
    """Build a producer-aligned core mapping for a restickify op if exact."""
    producer_info, reason = producer_for_restickify(restickify_op, name_to_op)
    if producer_info is None:
        return None, reason

    producer, read_dep = producer_info
    producer_splits = decode_op_splits(producer)
    restickify_splits = decode_op_splits(restickify_op)
    symbol_map, reason = restickify_symbol_map(producer, restickify_op, read_dep)
    if reason is not None:
        return None, reason

    producer_core_count = math.prod(producer_splits.values())
    restickify_core_count = math.prod(restickify_splits.values())
    if producer_core_count != restickify_core_count:
        return None, "different-core-count"

    reverse_symbol_map = {
        producer_sym: rest_sym for rest_sym, producer_sym in symbol_map.items()
    }
    for producer_sym, producer_split in producer_splits.items():
        restickify_sym = reverse_symbol_map.get(producer_sym)
        restickify_split = restickify_splits.get(restickify_sym, 1)
        if producer_split != restickify_split:
            return None, "different-split-factors"

    for restickify_sym, restickify_split in restickify_splits.items():
        producer_sym = symbol_map.get(restickify_sym)
        producer_split = producer_splits.get(producer_sym, 1)
        if restickify_split != producer_split:
            return None, "different-split-factors"

    producer_sizes = op_iteration_sizes(producer)
    restickify_sizes = op_iteration_sizes(restickify_op)
    producer_mapping = _mapping_for_op(
        producer,
        producer_sizes,
        producer_splits,
        k_fast_ops,
    )

    override: dict[str, dict[str, int]] = {}
    for core_id, producer_slices in producer_mapping.items():
        per_dim: dict[str, int] = {}
        for restickify_sym in restickify_sizes:
            producer_sym = symbol_map.get(restickify_sym)
            per_dim[restickify_sym] = (
                producer_slices.get(producer_sym, 0) if producer_sym is not None else 0
            )
        override[str(core_id)] = per_dim
    return override, None
