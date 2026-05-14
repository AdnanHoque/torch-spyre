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
) -> dict[str, dict[str, int]]:
    override = getattr(op, CORE_MAPPING_OVERRIDE_ATTR, None)
    if override is not None:
        return normalize_core_mapping(override)
    return materialize_default_core_mapping(
        list(iteration_sizes.keys()),
        split_factors,
        math.prod(split_factors.values()),
    )


def build_restickify_core_mapping_override(
    restickify_op: ComputedBuffer,
    name_to_op: Mapping[str, ComputedBuffer],
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
    producer_mapping = _mapping_for_op(producer, producer_sizes, producer_splits)

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
