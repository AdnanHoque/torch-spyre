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
