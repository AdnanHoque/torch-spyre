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

"""LSE-combine fold operator — pure-Python/numpy value oracle (A2, G2 reduce lane).

This is the ARITHMETIC HALF of the reduce lane: the frontend-owned fold operator
that combines two flash-attention partials over DISJOINT key sets. The backend
must eventually lower this to an SFP ``lse_combine`` primitive; THIS file is the
value oracle that primitive is validated against.

A flash partial over a key subset K_i is the triple::

    P_i = (m_i, l_i, A_i)

    m_i : running row-max of the pre-softmax scores over K_i           (scalar/head-row)
    l_i : denominator mass  = sum_{k in K_i} exp(s_k - m_i)            (scalar/head-row)
    A_i : unnormalized output = sum_{k in K_i} exp(s_k - m_i) * v_k    (d_v vector/head-row)

The fold operator ``lse_combine`` merges two partials into one partial over the
UNION K_1 u K_2 (the key sets must be disjoint):

    m  = max(m1, m2)
    a1 = exp(m1 - m)              # in [0, 1]; == 1 for the arg-max side
    a2 = exp(m2 - m)              # in [0, 1]
    A  = a1 * A1 + a2 * A2
    l  = a1 * l1 + a2 * l2

The final normalized attention output is produced ONCE at the end of the fold:

    O  = A / l

Key properties this oracle asserts:
  * associativity            (fold order over disjoint sets does not matter, up to rounding)
  * commutativity            (up to rounding)
  * run-to-run bit determinism under a STATIC ring order
  * fp32 carry vs fp16 drift (the carry dtype is load-bearing)
  * equivalence to a single-pass softmax over the UNION of key sets (the value spec)

COST NOTE (G3, informational): the payload folded per hop is (m, l, A) — a d_v
vector plus two scalars — which is L-INDEPENDENT (independent of the key-tile
count L). Over P ring participants the fold is F-DOMINATED, so the schedule that
wins is ``reduce_tree_fold`` (ceil(log2 P) hops), NOT ``reduce_plain_ring``
(P-1 hops). This file does not price; it only fixes the VALUES the priced
schedule must reproduce, whatever hop order it uses.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# fp32 is the carry dtype. This is not cosmetic: the rescale factors
# exp(m_i - m) and the accumulation of A/l must be done in fp32 or the tail
# partials (small exp(m_i - m)) lose all their bits. See test_fp32_carry_vs_fp16.
CARRY_DTYPE = np.float32


# =====================================================================
# The partial triple and the fold operator.
# =====================================================================
@dataclasses.dataclass(frozen=True)
class Partial:
    """A flash-attention partial over one (disjoint) key subset, one head-row.

    m : running max of pre-softmax scores over this key subset (scalar).
    l : denominator mass sum exp(s - m)                          (scalar).
    A : unnormalized output sum exp(s - m) * v                   (d_v vector).

    All fields are held in the fp32 CARRY_DTYPE. ``A`` is a 1-D array of length
    d_v; ``m`` and ``l`` are python/np scalars. The IDENTITY partial (empty key
    set) is (m=-inf, l=0, A=0); folding it is a no-op.
    """

    m: np.floating
    l: np.floating
    A: np.ndarray

    @staticmethod
    def identity(d_v: int) -> "Partial":
        """The fold identity: an empty key set. m=-inf, l=0, A=0."""
        return Partial(
            m=CARRY_DTYPE(-np.inf),
            l=CARRY_DTYPE(0.0),
            A=np.zeros(d_v, dtype=CARRY_DTYPE),
        )


def lse_combine(p1: Partial, p2: Partial) -> Partial:
    """Fold two flash partials over DISJOINT key sets into one partial.

    m = max(m1, m2); rescale each side by exp(m_i - m) in fp32; add. This is the
    fold-operator generalization of k_fast's '+' reduce (the frontend owns the
    operator; the backend owns realization).

    The (-inf, 0, 0) identity is handled by numpy: exp(-inf - m) = 0 cleanly,
    and max(-inf, m2) = m2, so lse_combine(identity, p) == p exactly.
    """
    m = np.maximum(p1.m, p2.m).astype(CARRY_DTYPE)
    # exp(-inf) -> 0.0; when both m are -inf, m is -inf and a1=a2=exp(0)=1 but
    # A1=A2=0 and l1=l2=0, so the result is the identity again (no NaN).
    with np.errstate(invalid="ignore"):
        a1 = np.exp((p1.m - m)).astype(CARRY_DTYPE)
        a2 = np.exp((p2.m - m)).astype(CARRY_DTYPE)
    # If m is -inf (both empty), force factors to 0 so A/l stay 0 (avoid inf-inf).
    if not np.isfinite(m):
        a1 = CARRY_DTYPE(0.0)
        a2 = CARRY_DTYPE(0.0)
    A = (a1 * p1.A.astype(CARRY_DTYPE) + a2 * p2.A.astype(CARRY_DTYPE)).astype(
        CARRY_DTYPE
    )
    l = (a1 * p1.l + a2 * p2.l).astype(CARRY_DTYPE)
    return Partial(m=m, l=l, A=A)


def normalize(p: Partial) -> np.ndarray:
    """Final normalization, applied ONCE at the end of the fold: O = A / l."""
    return (p.A.astype(CARRY_DTYPE) / p.l).astype(CARRY_DTYPE)


def fold(partials, order=None) -> Partial:
    """Reduce a list of partials with lse_combine in a STATIC order.

    ``order`` is an explicit index permutation (the static ring order). The same
    ``order`` on the same inputs yields a bit-identical result run to run (this
    is what makes the on-device ring fold run-to-run deterministic).
    """
    if order is None:
        order = range(len(partials))
    idx = list(order)
    d_v = partials[0].A.shape[0]
    acc = Partial.identity(d_v)
    for i in idx:
        acc = lse_combine(acc, partials[i])
    return acc


# =====================================================================
# Partial construction + the single-pass softmax reference (the value spec).
# =====================================================================
def make_partial(scores: np.ndarray, values: np.ndarray) -> Partial:
    """Build a flash partial (m, l, A) from raw pre-softmax scores over one key
    subset and the corresponding value rows.

    scores : (n_k,)          pre-softmax scores s_k for one head-row.
    values : (n_k, d_v)      value vectors v_k.
    """
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
    scores: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Reference: standard softmax attention over the WHOLE (union) key set.

    scores : (N,)            all pre-softmax scores.
    values : (N, d_v)        all value vectors.
    Returns O = softmax(scores) @ values, computed in fp32.
    """
    scores = scores.astype(CARRY_DTYPE)
    values = values.astype(CARRY_DTYPE)
    m = scores.max().astype(CARRY_DTYPE)
    w = np.exp(scores - m).astype(CARRY_DTYPE)
    return ((w[:, None] * values).sum(axis=0) / w.sum()).astype(CARRY_DTYPE)
