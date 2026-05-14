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

"""Phase 1 ring-aware restickify: producer-aligned consumer split priorities.

Hook for work_distribution_pass: given a consumer op and a name->producer
map, return a re-ordered output_dims list that biases work_distribution
toward splits which physically match the producer's split dim. The intent
is that the intervening restickify ends up with zero inter-core traffic.

The mechanism mirrors the v2 telemetry's stride-matching: producer's write
index and consumer's read index over the shared buffer both encode each
iteration symbol's stride; symbols sharing the same stride physically
index the same buffer dim.
"""

from __future__ import annotations

import sympy
from sympy import Symbol
from torch._inductor.ir import ComputedBuffer

from .logging_utils import get_inductor_logger
from .pass_utils import (
    apply_splits_from_index_coeff,
    concretize_expr,
    iteration_space_from_op,
)

logger = get_inductor_logger("mapping_alignment")


def _decode_op_splits(op: ComputedBuffer) -> dict[str, int]:
    """Per-symbol splits for op, or {} when none set."""
    encoded = getattr(op, "op_it_space_splits", None)
    if encoded is None:
        return {}
    try:
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        it_space = iteration_space_from_op(op)
        per_dim = apply_splits_from_index_coeff(
            encoded, write_index, read_index, it_space
        )
        return {str(s): v for s, v in per_dim.items() if v > 1}
    except Exception:  # noqa: BLE001
        return {}


def _extract_strides(index_expr, var_names) -> dict[str, int]:
    """Per-symbol stride coefficients in a linear index expression."""
    if index_expr is None:
        return {}
    out: dict[str, int] = {}
    expr = sympy.sympify(index_expr)
    for v in var_names:
        try:
            coeff = expr.coeff(v)
            if coeff == 0:
                continue
            out[str(v)] = int(concretize_expr(coeff))
        except (TypeError, ValueError):
            continue
    return out


def _build_sym_correspondence(
    producer_strides: dict[str, int],
    consumer_strides: dict[str, int],
) -> dict[str, str]:
    """Map consumer symbols to producer symbols via shared buffer strides.

    Two symbols indexing a buffer with the same stride access the same
    physical tensor dim. Used to detect alignment opportunities even when
    iteration symbols differ across ops.
    """
    producer_sym_by_stride: dict[int, str] = {}
    for sym, stride in producer_strides.items():
        producer_sym_by_stride.setdefault(stride, sym)

    out: dict[str, str] = {}
    for c_sym, c_stride in consumer_strides.items():
        if c_stride in producer_sym_by_stride:
            out[c_sym] = producer_sym_by_stride[c_stride]
    return out


def reorder_output_dims_for_producer_alignment(
    op: ComputedBuffer,
    output_dims: list[Symbol],
    name_to_op: dict[str, ComputedBuffer],
) -> list[Symbol]:
    """Promote output_dims that match a producer's split to the front.

    For each producer of op:
      1. Decode producer's per-symbol splits.
      2. Stride-match consumer iteration symbols to producer's via the
         shared buffer's read/write index expressions.
      3. For each consumer symbol whose corresponding producer symbol has
         split > 1, promote it to the front of output_dims.

    Producer-aligned symbols come first (in producer-split-factor-descending
    order); the rest follow in their original priority order. Caller can
    safely ignore the result when the hint produces no aligned candidates.
    """
    rw = op.get_read_writes()
    promote: dict[Symbol, int] = {}  # sym -> producer's split factor

    for read_dep in rw.reads:
        producer = name_to_op.get(read_dep.name)
        if producer is None:
            continue
        producer_splits = _decode_op_splits(producer)
        if not producer_splits:
            continue

        producer_rw = producer.get_read_writes()
        producer_writes = list(producer_rw.writes)
        if not producer_writes:
            continue
        producer_write = producer_writes[0]

        p_strides = _extract_strides(producer_write.index, producer_write.var_names)
        c_strides = _extract_strides(read_dep.index, read_dep.var_names)
        cons_to_prod = _build_sym_correspondence(p_strides, c_strides)

        for c_sym_str, p_sym_str in cons_to_prod.items():
            split_factor = producer_splits.get(p_sym_str, 1)
            if split_factor <= 1:
                continue
            for sym in output_dims:
                if str(sym) == c_sym_str:
                    # Keep the largest producer-split factor we've seen.
                    promote[sym] = max(promote.get(sym, 0), split_factor)
                    break

    if not promote:
        return output_dims

    # Aligned dims sorted by producer's split factor descending; ties broken
    # by original position so the output is deterministic.
    aligned = sorted(
        promote.keys(),
        key=lambda s: (-promote[s], output_dims.index(s)),
    )
    rest = [d for d in output_dims if d not in promote]

    if logger.isEnabledFor(20):  # logging.INFO
        logger.info(
            "alignment hint for %s: %s (producer_split_factors=%s)",
            op.get_name(),
            [str(s) for s in aligned],
            {str(s): f for s, f in promote.items()},
        )

    return aligned + rest


def build_name_to_op_map(operations) -> dict[str, ComputedBuffer]:
    """Build a buffer-name -> producing ComputedBuffer index for fast lookup."""
    name_to_op: dict[str, ComputedBuffer] = {}
    for op in operations:
        if isinstance(op, ComputedBuffer):
            name_to_op[op.get_name()] = op
    return name_to_op


# ----------------------------------------------------------------------
# Precise hop cost: per-core slice enumeration and pairwise overlap.
# ----------------------------------------------------------------------


def _ring_distance(a: int, b: int, num_cores: int) -> int:
    """Shorter direction on a `num_cores`-bidirectional ring."""
    d = abs(a - b)
    return min(d, num_cores - d)


def _per_core_slice_indices(
    dim_order: list[str],
    splits: dict[str, int],
    num_cores: int,
) -> list[dict[str, tuple[int, int]]]:
    """For each core, return its slice index per split dim.

    Matches `_get_core_to_slice_mapping`'s convention: iteration_space order
    determines which split dim varies fastest along core_id. Returns a list
    of length num_cores; entry i is a dict {sym_str: (slice_idx, total_splits)}.

    Unsplit dims (factor 1) are omitted — they contribute the full extent
    on every core and don't affect overlap computation.
    """
    slices: list[dict[str, tuple[int, int]]] = [{} for _ in range(num_cores)]
    inner = 1
    for sym in dim_order:
        factor = splits.get(sym, 1)
        if factor <= 1:
            continue
        for c in range(num_cores):
            idx = (c // inner) % factor
            slices[c][sym] = (idx, factor)
        inner *= factor
    return slices


def compute_precise_hop_cost(
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    consumer_to_producer_sym: dict[str, str],
    num_cores: int,
) -> tuple[int, float] | None:
    """Compute exact pairwise ring cost for a producer-consumer edge.

    Returns (total_overlap_units, total_unit_hops) — both keyed off the
    iteration-space "units" (= bytes when scaled by dtype.itemsize and any
    stick-vs-elem factor; left dimensionless here so the caller can do the
    scaling). Returns None when the inputs don't permit a precise answer
    (e.g. producer/consumer split on dims with no stride correspondence).

    Algorithm:
      1. Compute each producer core's slice index along each split dim.
      2. Same for consumer, translated into producer's symbol space.
      3. For each (p_core, c_core) pair, compute overlap factor:
         product over shared dims of (1 if slice_indices match else 0),
         and the per-pair unit count = total / (factor product).
      4. Sum ring_dist(p, c) * overlap_units across all pairs.

    The total_overlap_units returned equals the iteration-space size when
    the splits perfectly partition (which they do for typical Spyre ops).
    """
    p_splits = _decode_op_splits(producer)
    c_splits = _decode_op_splits(consumer)
    if not p_splits:
        return None

    # Translate consumer splits into producer's symbol space via the
    # correspondence map. Drop splits we can't translate — they contribute
    # to neither producer's nor consumer's tracked dims.
    c_splits_in_p: dict[str, int] = {}
    for c_sym, factor in c_splits.items():
        p_sym = consumer_to_producer_sym.get(c_sym)
        if p_sym is not None:
            c_splits_in_p[p_sym] = factor

    p_it_space = iteration_space_from_op(producer)
    p_dim_order = [str(s) for s in p_it_space.keys()]

    c_it_space = iteration_space_from_op(consumer)
    c_dim_order_raw = [str(s) for s in c_it_space.keys()]
    # Translate consumer dim order to producer's symbols; preserve order so
    # the slice-index calculation matches what `_get_core_to_slice_mapping`
    # would produce on the consumer side.
    c_dim_order_in_p: list[str] = []
    for c_sym in c_dim_order_raw:
        p_sym = consumer_to_producer_sym.get(c_sym)
        if p_sym is not None:
            c_dim_order_in_p.append(p_sym)

    p_slices = _per_core_slice_indices(p_dim_order, p_splits, num_cores)
    c_slices = _per_core_slice_indices(c_dim_order_in_p, c_splits_in_p, num_cores)

    # The number of "iteration-space units" produced by the producer is
    # the product of its extents (we leave element/stick scaling to the
    # caller). Per-pair overlap = total_units / (cross product of split
    # factors), since splits partition the unit count.
    total_units = 1
    for sym in p_it_space:
        try:
            total_units *= int(concretize_expr(p_it_space[sym]))
        except (TypeError, ValueError):
            return None  # symbolic extent — can't quantify

    # Cross product of split factors across all dims that appear on either
    # side (taking max of producer and consumer factor per dim).
    all_dims = set(p_splits) | set(c_splits_in_p)
    pair_divisor = 1
    for sym in all_dims:
        pair_divisor *= max(p_splits.get(sym, 1), c_splits_in_p.get(sym, 1))

    # When a dim is split on both sides with the same factor, the slice
    # indices must MATCH for overlap to be non-zero. When split on only one
    # side, the other side has the full extent → overlap always non-zero.
    # When split with different factors, partial overlap (handled below).

    total_unit_hops = 0.0
    total_overlap_units = 0
    for p_core in range(num_cores):
        for c_core in range(num_cores):
            # Compute per-pair overlap as a fraction of the per-pair-divisor.
            overlap_factor = 1
            for sym in all_dims:
                pf = p_splits.get(sym, 1)
                cf = c_splits_in_p.get(sym, 1)
                pi = p_slices[p_core].get(sym, (0, 1))[0]
                ci = c_slices[c_core].get(sym, (0, 1))[0]
                # Map slice indices to ranges in [0, max_factor) units.
                max_f = max(pf, cf)
                p_lo = pi * (max_f // pf)
                p_hi = p_lo + (max_f // pf)
                c_lo = ci * (max_f // cf)
                c_hi = c_lo + (max_f // cf)
                ov = max(0, min(p_hi, c_hi) - max(p_lo, c_lo))
                overlap_factor *= ov
            if overlap_factor == 0:
                continue
            overlap_units = (total_units * overlap_factor) // pair_divisor
            total_overlap_units += overlap_units
            dist = _ring_distance(p_core, c_core, num_cores)
            total_unit_hops += dist * overlap_units

    return total_overlap_units, total_unit_hops
