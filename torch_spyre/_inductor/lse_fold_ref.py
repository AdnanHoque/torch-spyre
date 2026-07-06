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

"""LSE-combine fold operator — the frontend-owned value oracle (G2 reduce lane).

This is the ARITHMETIC HALF of the LSE ring-fold REDUCE collective: the
frontend-owned fold operator that combines two flash-attention partials over
DISJOINT key sets.  The frontend owns the OPERATOR; the backend owns its
REALIZATION.  The eventual SFP ``lse_combine`` device primitive (deeptools
lowering, still gated) must be value-validated against this module.

This module is PURE PYTHON with an OPTIONAL numpy dependency.  It never imports
torch, so it can be exercised with the system ``python3`` (the package
``__init__`` imports torch and is not needed here).  ``numpy`` is only required
for the array-typed :class:`Partial` fold; :func:`lse_partial_nbytes` and the
LSE-combine identity are stdlib-only.

A flash partial over a key subset ``K_i`` is the triple::

    P_i = (m_i, l_i, A_i)

    m_i : running row-max of the pre-softmax scores over K_i    (scalar/head-row)
    l_i : denominator mass  = sum_{k in K_i} exp(s_k - m_i)      (scalar/head-row)
    A_i : unnormalized out  = sum_{k in K_i} exp(s_k - m_i) v_k  (d_v vec/head-row)

The fold operator ``lse_combine`` merges two partials into one partial over the
UNION ``K_1 u K_2`` (the key sets must be DISJOINT)::

    m  = max(m1, m2)
    a1 = exp(m1 - m)            # in (0, 1]; == 1 for the arg-max side
    a2 = exp(m2 - m)            # in (0, 1]
    A  = a1 * A1 + a2 * A2
    l  = a1 * l1 + a2 * l2

The normalized attention output is produced ONCE at the end of the fold::

    O  = A / l

Properties this oracle fixes (asserted in tests/tensor/test_lse_fold.py):
  * associativity            (fold order over disjoint sets is value-equivalent)
  * commutativity            (up to rounding)
  * run-to-run bit determinism under a STATIC ring order
  * fp32 carry vs fp16 drift (the carry dtype is load-bearing)
  * equivalence to a single-pass softmax over the UNION of key sets

COST NOTE (G3, informational): the payload folded per hop is (m, l, A) — a d_v
vector plus two scalars — which is L-INDEPENDENT (independent of the key-tile
count L).  Over P ring participants the fold is F-DOMINATED, so the winning
schedule is ``reduce_tree_fold`` (ceil(log2 P) hops), NOT ``reduce_plain_ring``
(P-1 hops).  This module does not price; it only fixes the VALUES the priced
schedule must reproduce, whatever hop order it uses.  See comm_cost.py.
"""

from __future__ import annotations

import dataclasses
import math

try:
    import numpy as np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover - numpy is present in CI
    np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False

# fp32 is the carry dtype.  This is not cosmetic: the rescale factors
# exp(m_i - m) and the accumulation of A/l MUST be done in fp32 or the tail
# partials (small exp(m_i - m)) lose all their bits.  See the fp32-vs-fp16 test.
if _HAVE_NUMPY:
    CARRY_DTYPE = np.float32
else:  # pragma: no cover
    CARRY_DTYPE = float

# Bytes of one fp32 carry scalar / element.  Used by lse_partial_nbytes to size
# the L-independent (m, l, A) payload for G3 pricing.
CARRY_ITEMSIZE = 4


# =====================================================================
# Payload sizing (stdlib only) — the estimated_tensor_bytes for the plan.
# =====================================================================
def lse_partial_nbytes(d_v: int, n_head_rows: int = 1) -> int:
    """Bytes of one flash partial triple (m, l, A) in the fp32 carry dtype.

    The folded payload is L-INDEPENDENT: two scalars (m, l) plus a ``d_v``
    output vector, per head-row.  This is the ``estimated_tensor_bytes`` the LX
    relayout reduce plan attaches so ``comm_edge_from_plan`` prices a nonzero
    per-hop payload.  (Even at ``operand_bytes=0`` the reduce is F-DOMINATED, so
    this only refines an already-correct F-cost.)

    Args:
      d_v: output feature width per head-row (length of A).
      n_head_rows: number of head-rows folded together (query rows x heads).

    Returns:
      (2 scalars + d_v vector) * fp32 itemsize * n_head_rows, in bytes.
    """
    per_row = (2 + int(d_v)) * CARRY_ITEMSIZE
    return int(per_row * max(1, int(n_head_rows)))


def reduce_tree_hops(p: int) -> int:
    """ceil(log2 P) — the tree-fold hop count G3 selects for the reduce lane."""
    return max(0, math.ceil(math.log2(p))) if p >= 2 else 0


# =====================================================================
# The partial triple and the fold operator (numpy-backed).
# =====================================================================
@dataclasses.dataclass(frozen=True)
class Partial:
    """A flash-attention partial over one (disjoint) key subset, one head-row.

    m : running max of pre-softmax scores over this key subset (scalar).
    l : denominator mass sum exp(s - m)                          (scalar).
    A : unnormalized output sum exp(s - m) * v                   (d_v vector).

    All fields are held in the fp32 CARRY_DTYPE.  ``A`` is a 1-D array of length
    ``d_v``; ``m`` and ``l`` are np scalars.  The IDENTITY partial (empty key
    set) is (m=-inf, l=0, A=0); folding it is an exact no-op.
    """

    m: "np.floating"
    l: "np.floating"
    A: "np.ndarray"

    @staticmethod
    def identity(d_v: int) -> "Partial":
        """The fold identity: an empty key set. m=-inf, l=0, A=0."""
        _require_numpy()
        return Partial(
            m=CARRY_DTYPE(-np.inf),
            l=CARRY_DTYPE(0.0),
            A=np.zeros(d_v, dtype=CARRY_DTYPE),
        )


def _require_numpy() -> None:
    if not _HAVE_NUMPY:  # pragma: no cover
        raise RuntimeError(
            "lse_fold_ref: numpy is required for the array-typed fold; only "
            "lse_partial_nbytes / reduce_tree_hops are stdlib-only."
        )


def lse_combine(p1: Partial, p2: Partial) -> Partial:
    """Fold two flash partials over DISJOINT key sets into one partial.

    m = max(m1, m2); rescale each side by exp(m_i - m) in fp32; add.  This is
    the fold-operator generalization of k_fast's '+' reduce (the frontend owns
    the operator; the backend owns realization).

    The (-inf, 0, 0) identity is handled cleanly: exp(-inf - m) = 0 and
    max(-inf, m2) = m2, so lse_combine(identity, p) == p exactly.  When both m
    are -inf (both empty) the factors are forced to 0 so A/l stay 0 (no NaN).
    """
    _require_numpy()
    m = np.maximum(p1.m, p2.m).astype(CARRY_DTYPE)
    with np.errstate(invalid="ignore"):
        a1 = np.exp((p1.m - m)).astype(CARRY_DTYPE)
        a2 = np.exp((p2.m - m)).astype(CARRY_DTYPE)
    if not np.isfinite(m):
        a1 = CARRY_DTYPE(0.0)
        a2 = CARRY_DTYPE(0.0)
    A = (a1 * p1.A.astype(CARRY_DTYPE) + a2 * p2.A.astype(CARRY_DTYPE)).astype(
        CARRY_DTYPE
    )
    l = (a1 * p1.l + a2 * p2.l).astype(CARRY_DTYPE)
    return Partial(m=m, l=l, A=A)


def normalize(p: Partial) -> "np.ndarray":
    """Final normalization, applied ONCE at the end of the fold: O = A / l."""
    _require_numpy()
    return (p.A.astype(CARRY_DTYPE) / p.l).astype(CARRY_DTYPE)


def fold(partials, order=None) -> Partial:
    """Reduce a list of partials with lse_combine in a STATIC order.

    ``order`` is an explicit index permutation (the static ring order).  The
    same ``order`` on the same inputs yields a bit-identical result run to run
    (this is what makes the on-device ring fold run-to-run deterministic).
    """
    _require_numpy()
    if order is None:
        order = range(len(partials))
    idx = list(order)
    d_v = partials[0].A.shape[0]
    acc = Partial.identity(d_v)
    for i in idx:
        acc = lse_combine(acc, partials[i])
    return acc


def tree_fold(partials) -> Partial:
    """Log2-depth pairwise tree fold (the G3 reduce_tree_fold value order).

    Associativity makes this value-equivalent to :func:`fold`; it exists so the
    oracle can pin the tree order the cost-optimal schedule actually uses.
    """
    _require_numpy()
    cur = list(partials)
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur) - 1, 2):
            nxt.append(lse_combine(cur[i], cur[i + 1]))
        if len(cur) % 2 == 1:
            nxt.append(cur[-1])
        cur = nxt
    return cur[0]


# =====================================================================
# Partial construction + the single-pass softmax reference (value spec).
# =====================================================================
def make_partial(scores: "np.ndarray", values: "np.ndarray") -> Partial:
    """Build a flash partial (m, l, A) from raw pre-softmax scores over one key
    subset and the corresponding value rows.

    scores : (n_k,)          pre-softmax scores s_k for one head-row.
    values : (n_k, d_v)      value vectors v_k.
    """
    _require_numpy()
    scores = scores.astype(CARRY_DTYPE)
    values = values.astype(CARRY_DTYPE)
    if scores.size == 0:
        return Partial.identity(values.shape[1])
    m = scores.max().astype(CARRY_DTYPE)
    w = np.exp(scores - m).astype(CARRY_DTYPE)  # (n_k,)
    l = w.sum().astype(CARRY_DTYPE)
    A = (w[:, None] * values).sum(axis=0).astype(CARRY_DTYPE)
    return Partial(m=m, l=l, A=A)


def single_pass_softmax_attention(
    scores: "np.ndarray", values: "np.ndarray"
) -> "np.ndarray":
    """Reference: standard softmax attention over the WHOLE (union) key set.

    scores : (N,)            all pre-softmax scores.
    values : (N, d_v)        all value vectors.
    Returns O = softmax(scores) @ values, computed in fp32.
    """
    _require_numpy()
    scores = scores.astype(CARRY_DTYPE)
    values = values.astype(CARRY_DTYPE)
    m = scores.max().astype(CARRY_DTYPE)
    w = np.exp(scores - m).astype(CARRY_DTYPE)
    return ((w[:, None] * values).sum(axis=0) / w.sum()).astype(CARRY_DTYPE)
