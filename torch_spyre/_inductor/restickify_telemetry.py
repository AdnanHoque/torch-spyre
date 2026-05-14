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

"""Phase 0 telemetry for ring-aware restickify (RFC draft).

Read-only diagnostic pass. For each restickify op, logs producer/consumer
identity and op_it_space_splits so we can see whether the two ends of the
restickify agree on a core mapping. Off by default; enable with
SPYRE_RESTICKIFY_TELEMETRY=1.

This is the MVP for Phase 0 of the Ring-Aware Restickify RFC. Hop-cost
math comes later once we can confirm the plumbing works end-to-end.
"""

from __future__ import annotations


import sympy
import torch
from torch._inductor.ir import ComputedBuffer, Operation

from . import config
from .logging_utils import get_inductor_logger
from .pass_utils import (
    apply_splits_from_index_coeff,
    concretize_expr,
    iteration_space_from_op,
)

logger = get_inductor_logger("restickify_telemetry")

# Spyre AIU 1.0 has 32 cores on the RIU ring. On a ring of N cores, the
# mean signed-distance between a uniformly random pair is ~N/4 hops (the
# average of min(|i-j|, N-|i-j|) over all i, j).
NUM_CORES_DEFAULT = 32


def _is_restickify(op: ComputedBuffer) -> bool:
    """True if op was inserted by the restickify pass.

    insert_restickify sets op.origins to the synthetic FX node it created
    with target torch.ops.spyre.restickify.default. That's the cleanest
    structural signal pre-codegen.
    """
    origins = getattr(op, "origins", None)
    if not origins:
        return False
    for o in origins:
        if not isinstance(o, torch.fx.Node):
            continue
        target = o.target
        # FX targets for spyre.restickify show up as the OpOverload itself.
        if target is torch.ops.spyre.restickify.default:
            return True
    return False


def _decode_splits(op: ComputedBuffer) -> dict[str, int] | None:
    """Decode op.op_it_space_splits to per-symbol splits.

    Returns None if the op has no splits assigned (e.g. graph inputs, or
    work_distribution_pass declined to split because cores_used <= 1).
    Returns {} for ops with op_it_space_splits set but all values == 1.
    """
    encoded = getattr(op, "op_it_space_splits", None)
    if encoded is None:
        return None
    try:
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        it_space = iteration_space_from_op(op)
        per_dim = apply_splits_from_index_coeff(
            encoded, write_index, read_index, it_space
        )
        # Drop unsplit dims for readability; sort for deterministic output.
        return {
            str(s): v
            for s, v in sorted(per_dim.items(), key=lambda kv: str(kv[0]))
            if v > 1
        }
    except Exception as e:  # noqa: BLE001
        return {"<decode-error>": type(e).__name__}  # type: ignore[dict-item]


def _restickify_bytes(op: ComputedBuffer) -> int | None:
    """Total bytes moved by this restickify (size × dtype itemsize).

    Returns None if we can't resolve a concrete shape/dtype (e.g. symbolic
    sizes that haven't been concretized).
    """
    try:
        layout = op.get_layout()
        # Pull the logical size of the output via the layout. The total bytes
        # moved by a restickify is the size of the output it produces.
        size = layout.size
        elems = 1
        for s in size:
            elems *= int(concretize_expr(s))
        return elems * layout.dtype.itemsize
    except Exception:  # noqa: BLE001
        return None


def _mean_random_hops(num_cores: int) -> float:
    """Average min-ring-distance for a uniformly random pair (i ≠ j).

    There are num_cores * (num_cores - 1) ordered pairs; for each non-zero
    distance d in {1, ..., num_cores-1}, num_cores pairs are at that
    distance. So the mean over all i≠j is Σ_{d=1}^{N-1} min(d, N-d) / (N-1).

    For 32 cores: ~8.26. For 16 cores: ~4.27.
    """
    total = 0
    for d in range(1, num_cores):
        total += min(d, num_cores - d)
    return total / (num_cores - 1)


def _estimate_hops_per_byte(
    producer_splits: dict | None,
    self_splits: dict | None,
    num_cores: int = NUM_CORES_DEFAULT,
) -> float:
    """Coarse hop estimate keyed off mapping alignment.

    Phase 0 first-order model. Actual hop math (using
    _get_core_to_slice_mapping to reconstruct per-stick source/dest cores)
    is Phase 0 follow-up if the coarse signal warrants it.

    Returns expected min-ring hops per byte traversed.
    """
    if producer_splits is None:
        # Graph input — bytes come from HBM, not from another core's local
        # store. RIU hops aren't the right model; return 0 so this row
        # doesn't dominate the "fix this" list.
        return 0.0
    if producer_splits == self_splits:
        # Same dim, same factor → consumer reads its own producer slice.
        return 0.0
    common = set(producer_splits or {}) & set(self_splits or {})
    if not common:
        # Orthogonal mapping → uniformly random pairing → mean half-ring.
        return _mean_random_hops(num_cores)
    # Partial match — approximate as half the orthogonal cost. Refine when
    # we add the proper _get_core_to_slice_mapping-based computation.
    return _mean_random_hops(num_cores) / 2


def _extract_strides(index_expr, var_names) -> dict[str, int]:
    """Pull per-symbol stride coefficients out of a linear index expression.

    Inductor's index expressions for buffer accesses are linear sums of
    iteration symbols weighted by their physical buffer strides
    (e.g. `2048*d0 + d1` for a (M, N=2048) tensor accessed as buf[d0*2048+d1]).
    The stride identifies which TENSOR DIM each iteration symbol indexes,
    which is what we need to match symbols across producer/consumer when
    the iteration spaces have different symbol names.

    Returns {sym_name: int_stride}. Symbols with zero coefficient are dropped.
    Strides that don't concretize to ints (symbolic shapes) are also dropped
    with a logged warning rather than blowing up the pass.
    """
    if index_expr is None:
        return {}
    out: dict[str, int] = {}
    for v in var_names:
        try:
            coeff = sympy.sympify(index_expr).coeff(v)
            if coeff == 0:
                continue
            out[str(v)] = int(concretize_expr(coeff))
        except (TypeError, ValueError):
            # Symbolic stride — skip rather than crash the pass.
            continue
    return out


def _build_sym_correspondence(
    producer_strides: dict[str, int],
    consumer_strides: dict[str, int],
) -> dict[str, str]:
    """Match consumer symbols to producer symbols via shared buffer strides.

    Two symbols indexing the buffer with the same stride access the same
    tensor dim. That's how we detect e.g. `consumer.d1 == producer.d0`
    after a transpose: both have stride H in the shared buffer's index.

    Returns dict {consumer_sym: producer_sym}. Symbols without a match are
    not present in the output.
    """
    # Invert producer's stride map. If two producer symbols share a stride
    # (rare but possible with broadcast), pick the first deterministically.
    producer_sym_by_stride: dict[int, str] = {}
    for sym, stride in producer_strides.items():
        producer_sym_by_stride.setdefault(stride, sym)

    matches: dict[str, str] = {}
    for c_sym, c_stride in consumer_strides.items():
        if c_stride in producer_sym_by_stride:
            matches[c_sym] = producer_sym_by_stride[c_stride]
    return matches


def _physical_alignment_status(
    producer_splits: dict | None,
    self_splits: dict | None,
    consumer_to_producer_sym: dict[str, str],
) -> str:
    """Classify alignment using the symbol-correspondence map.

    Differs from `_alignment_status` (which only compares symbol names) by
    first translating consumer symbols to producer's equivalent. This
    catches transposes and other index-permutation cases where the cores
    are physically aligned despite different symbol names.
    """
    if producer_splits is None:
        return "no-producer"
    # Translate consumer splits to producer's symbol space.
    translated: dict[str, int] = {}
    untranslatable = []
    for c_sym, factor in (self_splits or {}).items():
        p_sym = consumer_to_producer_sym.get(c_sym)
        if p_sym is None:
            untranslatable.append(c_sym)
        else:
            translated[p_sym] = factor

    if untranslatable:
        # We couldn't match these — fall back to symbol-only check on the
        # untranslated portion. Conservative: treat as mismatched.
        return "mismatched"
    if translated == producer_splits:
        return "aligned"
    common = set(translated) & set(producer_splits or {})
    if not common:
        return "mismatched"
    # Same physical dim split with different factor, or partial overlap.
    return "partial"


def _alignment_status(producer_splits: dict | None, self_splits: dict | None) -> str:
    """Classify producer/consumer mapping alignment.

    - 'no-producer': producer is a graph input (no op_it_space_splits)
    - 'aligned':     same dim symbols split the same way (zero-hop restickify)
    - 'mismatched':  different dim symbols → orthogonal mapping, worst case
    - 'partial':     some dims match, some don't
    """
    if producer_splits is None:
        return "no-producer"
    if not producer_splits and not self_splits:
        return "aligned"
    if producer_splits == self_splits:
        return "aligned"
    common = set(producer_splits or {}) & set(self_splits or {})
    if not common:
        return "mismatched"
    return "partial"


def restickify_telemetry(operations: list[Operation]) -> None:
    """Diagnostic pass: log each restickify with producer/consumer mappings.

    Runs after work_distribution (and k_fast_override), so op_it_space_splits
    is final on every ComputedBuffer.
    """
    if not config.restickify_telemetry:
        return

    # Map buffer name -> producing ComputedBuffer for fast producer lookup.
    name_to_op: dict[str, ComputedBuffer] = {}
    for op in operations:
        if isinstance(op, ComputedBuffer):
            name_to_op[op.get_name()] = op

    # Build consumer index: for each buffer name, which ops read it.
    consumers_of: dict[str, list[ComputedBuffer]] = {}
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        rw = op.get_read_writes()
        for dep in rw.reads:
            consumers_of.setdefault(dep.name, []).append(op)

    restick_count = 0
    total_ring_byte_hops = 0
    rows: list[tuple] = []  # (ring_byte_hops, name, status, producer, ...)
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if not _is_restickify(op):
            continue
        restick_count += 1

        my_name = op.get_name()
        rw = op.get_read_writes()
        producer_names = [d.name for d in rw.reads]
        producers = [name_to_op[n] for n in producer_names if n in name_to_op]
        consumers = consumers_of.get(my_name, [])

        self_splits = _decode_splits(op)
        if producers:
            producer = producers[0]
            producer_name = producer.get_name()
            producer_splits = _decode_splits(producer)
        else:
            producer = None
            producer_name = producer_names[0] if producer_names else "<no-input>"
            producer_splits = None

        # Symbol-name alignment (the v1 status, kept for comparison logging).
        status_sym = _alignment_status(producer_splits, self_splits)

        # Physical alignment via stride-matching of iteration symbols across
        # the producer's write and the restickify's read of the shared buffer.
        consumer_to_producer_sym: dict[str, str] = {}
        status_phys = status_sym  # fall-back when stride extraction fails
        if producer is not None and producer_splits is not None:
            # producer's write of its output buffer
            p_writes = list(producer.get_read_writes().writes)
            # restickify's read of producer's buffer
            r_reads = [d for d in rw.reads if d.name == producer.get_name()]
            if p_writes and r_reads:
                p_dep = p_writes[0]
                r_dep = r_reads[0]
                p_strides = _extract_strides(p_dep.index, p_dep.var_names)
                r_strides = _extract_strides(r_dep.index, r_dep.var_names)
                consumer_to_producer_sym = _build_sym_correspondence(
                    p_strides, r_strides
                )
                status_phys = _physical_alignment_status(
                    producer_splits, self_splits, consumer_to_producer_sym
                )

        bytes_moved = _restickify_bytes(op)
        # Hop estimate keyed off the PHYSICAL status, not the symbol status.
        # Physical-aligned restickifies have ~0 inter-core ring traffic; the
        # bytes still move on each core but locally.
        if status_phys == "aligned":
            hops_per_byte = 0.0
        elif status_phys == "no-producer":
            hops_per_byte = 0.0  # bytes come from HBM, not a peer core
        elif status_phys == "partial":
            hops_per_byte = _mean_random_hops(NUM_CORES_DEFAULT) / 2
        else:  # mismatched
            hops_per_byte = _mean_random_hops(NUM_CORES_DEFAULT)
        ring_byte_hops = (
            int(bytes_moved * hops_per_byte) if bytes_moved is not None else 0
        )
        total_ring_byte_hops += ring_byte_hops

        rows.append(
            (
                ring_byte_hops,
                my_name,
                status_phys,
                status_sym,
                producer_name,
                producer_splits,
                [c.get_name() for c in consumers],
                self_splits,
                bytes_moved,
                hops_per_byte,
                consumer_to_producer_sym,
            )
        )

    # Sort by ring cost descending so the worst offenders show first.
    rows.sort(key=lambda r: r[0], reverse=True)
    for (
        ring_byte_hops,
        my_name,
        status_phys,
        status_sym,
        producer_name,
        producer_splits,
        consumer_names,
        self_splits,
        bytes_moved,
        hops_per_byte,
        consumer_to_producer_sym,
    ) in rows:
        bytes_str = "<unknown>" if bytes_moved is None else f"{bytes_moved}"
        logger.info(
            "restickify=%s status=%s (sym=%s) ring_byte_hops=%d bytes=%s "
            "hops_per_byte=%.2f producer=%s producer_splits=%s consumers=%s "
            "self_splits=%s sym_map=%s",
            my_name,
            status_phys,
            status_sym,
            ring_byte_hops,
            bytes_str,
            hops_per_byte,
            producer_name,
            producer_splits if producer_splits is not None else "<none>",
            consumer_names,
            self_splits if self_splits is not None else "<none>",
            consumer_to_producer_sym or "{}",
        )

    if restick_count > 0:
        logger.info(
            "summary: total_restickifies=%d total_ring_byte_hops=%d (%.2f MB-hops)",
            restick_count,
            total_ring_byte_hops,
            total_ring_byte_hops / (1024 * 1024),
        )
