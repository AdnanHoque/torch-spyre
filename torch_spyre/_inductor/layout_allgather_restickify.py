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

"""Layout/relayout CONTRACT builders for LX collective realization.

A "contract" is the frontend-owned description of a collective edge that the
backend (Deeptools / SFP) lowers to physical movement + on-core compute.  The
frontend owns the OPERATOR + distribution metadata; the backend owns the
REALIZATION.  This module holds the contract builders; the LX relayout planner
(:mod:`lx_relayout`) records them on the consumer op and G3
(:mod:`comm_cost`) prices them.

Currently implemented:

  * :func:`make_lse_ring_fold_contract` — the G2 LSE ring-fold REDUCE collective:
    fold flash-attention partials over DISJOINT key sets with the LSE-combine
    operator (see :mod:`lse_fold_ref`).

The all-gather / operand-broadcast contract builder that gives this module its
name (``make_matmul_operand_allgather_contract``) is Codex's broadcast lane and
is realized inline in :mod:`lx_relayout` today; the sibling builder is a
follow-up (kept here so both collective families share one contract module).

The contracts are plain dicts (JSON-serializable) so they can be dumped next to
the other realized ``*_plan.json`` artifacts and diffed run-to-run.
"""

from __future__ import annotations

from typing import Any

# String tags — mirror lx_relayout so a contract is self-describing without a
# torch import (lx_relayout transitively imports torch; keep this module usable
# from pure-Python tests).  These MUST equal the lx_relayout / comm_cost values.
COMM_CLASS_REDUCE = "reduce"
COMM_CLASS_ALL_REDUCE = "all_reduce"
LSE_RING_FOLD = "lse_ring_fold"

# The frontend-owned LSE-combine operator identity the backend SFP lse_combine
# primitive must lower (see lse_fold_ref.lse_combine).  Carried as a tag string
# on the contract so a realized plan records exactly which fold semantics it
# assumes.
LSE_COMBINE_OPERATOR = (
    "m = max(m1, m2); "
    "a_i = exp(m_i - m); "
    "A = a1*A1 + a2*A2; "
    "l = a1*l1 + a2*l2; "
    "O = A / l   (normalize once at fold end)"
)

# fp32 carry itemsize; the (m, l, A) triple is carried in fp32 for determinism
# and tail-partial precision (see lse_fold_ref.CARRY_DTYPE / the fp32-vs-fp16
# oracle test).
_CARRY_ITEMSIZE = 4


def lse_partial_nbytes(d_v: int, n_head_rows: int = 1) -> int:
    """Bytes of one flash partial (m, l, A) triple in the fp32 carry dtype.

    Kept stdlib-only (duplicated from lse_fold_ref.lse_partial_nbytes) so this
    contract builder does not require numpy.  The payload is L-INDEPENDENT: two
    scalars (m, l) plus a ``d_v`` output vector, per head-row.
    """
    per_row = (2 + int(d_v)) * _CARRY_ITEMSIZE
    return int(per_row * max(1, int(n_head_rows)))


def make_lse_ring_fold_contract(
    *,
    producer_name: str,
    consumer_name: str,
    reduce_axis: str,
    scatter_axis: str | None,
    participant_count: int,
    d_v: int,
    n_head_rows: int = 1,
    all_reduce: bool = False,
    static_ring_order: bool = True,
    fp32_carry: bool = True,
) -> dict[str, Any]:
    """Build the LSE ring-fold REDUCE contract for a flash Lk-reduce edge.

    The fold reduces flash partials over the ``reduce_axis`` (the Lk / key-tile
    split, which contracts away) and — for the REDUCE (not ALL_REDUCE) case —
    lands the normalized result scattered over ``scatter_axis`` (heads).  The
    LAYOUT DIVIDEND: reduce-over-Lk-within-head + scatter-result-over-heads
    lands the output HEAD-SPLIT, matching the k_fast out-projection input layout
    => ZERO relayout at attention -> out-proj.

    Args:
      producer_name / consumer_name: the LX relayout edge endpoints.
      reduce_axis: the device-dim name that carries the Lk reduction split; it
        contracts away in the fold.
      scatter_axis: the device-dim the reduced result stays split over (heads),
        or None for an all-reduce (replicated) result.
      participant_count: P — the number of cores holding disjoint key partials
        (the SPLIT factor along ``reduce_axis``, NOT a full 32-core ring).  The
        fold spans ONLY these P cores.
      d_v: output feature width per head-row (length of A).
      n_head_rows: number of head-rows folded together.
      all_reduce: True => COMM_CLASS_ALL_REDUCE (result replicated to every
        consumer); False => COMM_CLASS_REDUCE (result scattered — the layout
        dividend).
      static_ring_order: fix the fold order for run-to-run bit determinism.
      fp32_carry: carry (m, l, A) in fp32 (load-bearing for tail precision).

    Returns:
      A JSON-serializable contract dict carrying the communication class, the
      LSE-combine operator identity, the payload triple shape/bytes, the
      participant count, and the determinism flags.  The tree-fold hop count
      ``ceil(log2 P)`` is recorded as the target schedule (G3 picks it because
      the reduce is F-DOMINATED).
    """
    if participant_count < 1:
        raise ValueError(f"participant_count must be >= 1, got {participant_count}")

    comm_class = COMM_CLASS_ALL_REDUCE if all_reduce else COMM_CLASS_REDUCE
    payload_bytes = lse_partial_nbytes(d_v, n_head_rows)
    # ceil(log2 P) — the tree-fold hop count (0 for P<=1).
    tree_hops = 0
    p = int(participant_count)
    if p >= 2:
        hops = 0
        v = 1
        while v < p:
            v <<= 1
            hops += 1
        tree_hops = hops

    return {
        "kind": LSE_RING_FOLD,
        "producer_name": producer_name,
        "consumer_name": consumer_name,
        "communication_class": comm_class,
        "communication_pattern": LSE_RING_FOLD,
        "realization_strategy": LSE_RING_FOLD,
        # --- fold operator (backend SFP lse_combine must match) ---
        "operator": LSE_COMBINE_OPERATOR,
        "reduce_op": "lse_combine",
        # --- distribution / layout geometry ---
        "reduce_axis": reduce_axis,
        "scatter_axis": scatter_axis,
        "participant_count": p,
        # --- payload triple (m, l, A) — L-INDEPENDENT ---
        "payload": {
            "m_shape": [n_head_rows],
            "l_shape": [n_head_rows],
            "A_shape": [n_head_rows, int(d_v)],
            "carry_dtype": "float32" if fp32_carry else "float16",
            "estimated_tensor_bytes": payload_bytes,
        },
        # --- determinism / correctness flags ---
        "fp32_carry": bool(fp32_carry),
        "static_ring_order": bool(static_ring_order),
        # --- cost target (G3 picks reduce_tree_fold, F-dominated) ---
        "target_schedule": "reduce_tree_fold",
        "tree_fold_hops": tree_hops,
        "is_reduction": True,
    }
