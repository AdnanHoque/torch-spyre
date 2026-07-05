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

"""LSE ring-fold: compose the KV-carousel decode merge from primitives.

Each core runs a local flash pass over its owned KV shards and emits an
UNNORMALIZED partial ``(A, l, m)``. Merging the P partials into one output is an
associative reduction under the log-sum-exp combine.

The AIU has no single-core fused move-then-reduce op: ``STCDPOpLx`` is pure
movement, and reduce/accumulate exists only in the multi-device collective
layer (cross-chip AllReduce/AllGather), not on the intra-AIU 32-core ring. So
the frontend COMPOSES the merge as SHUFFLE (an ``STCDPOpLx`` ring move) plus a
separate SFP ``lse_combine`` op per hop. This module prices that composition and
emits the hop schedule.

Frontend vs backend split:
  * frontend (this file): the ring order, the per-hop op pair, the numerically
    exact combine, and the cost that ranks the three merge topologies.
  * backend ask: fusing move+combine into one hop, and overlapping a hop's move
    with the previous hop's combine. Today each hop is a sequential
    move-then-combine barrier (the relayout is its own program step); overlap
    needs the deeptools input-fetch-overlap gate extended past its current
    Conv-only consumer list.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

# Ring fabric constants. These mirror comm_cost_model.py (the shared on-chip
# comm cost model on the fork) so move pricing stays consistent; kept local so
# this package builds without that branch. rho = ring GB/s per direction.
RING_GBPS = 166.0
SFP_GBPS = 166.0  # lse_combine streams the payload through the SFP once

# lambda: fixed per-hop launch/traversal latency, microseconds. The merge is on
# the critical path of every layer every decode step and is sequence-length
# independent, so lambda -- not bandwidth -- sets which topology wins. Strawman
# until measured on device; flagged as THE crossover number.
HOP_LATENCY_US = 0.5

# The three merge topologies. See ring_fold for the cost of each.
A_LINEAR = "A_linear"
B_REDUCE_SCATTER = "B_reduce_scatter"
C_HALVING = "C_halving"


def _us(num_bytes: float, gbps: float) -> float:
    """Microseconds to move num_bytes at gbps."""
    return num_bytes / (gbps * 1e3)


@dataclasses.dataclass
class FlashPartial:
    """One core's UNNORMALIZED attention partial over its owned KV shards.

    A: fp32 [H, d]  running weighted value sum   sum_j exp(s_j - m) * v_j
    l: fp32 [H]     running softmax mass         sum_j exp(s_j - m)
    m: fp32 [H]     running row max of scores    -inf until a block is seen

    H is the query-head count (GQA is already expanded to query heads at this
    stage), d the head dim. On device A/l/m are LX-resident fp32 tensors; here
    they are numpy arrays so the combine below is the exact fp32 reference the
    SFP op must match.
    """

    A: np.ndarray  # [H, d] fp32
    l: np.ndarray  # noqa: E741  softmax mass; standard flash (m, l, A) notation
    m: np.ndarray  # [H]    fp32

    @classmethod
    def empty(cls, H: int, d: int) -> "FlashPartial":
        """Identity element of the fold: zero mass, -inf running max."""
        return cls(
            A=np.zeros((H, d), dtype=np.float32),
            l=np.zeros((H,), dtype=np.float32),
            m=np.full((H,), -np.inf, dtype=np.float32),
        )

    @property
    def nbytes(self) -> int:
        """Payload one hop carries: the whole (A, l, m) triple."""
        return int(self.A.nbytes + self.l.nbytes + self.m.nbytes)


def lse_combine(p1: FlashPartial, p2: FlashPartial) -> FlashPartial:
    """Associative, fp32 log-sum-exp merge of two partials.

        m = max(m1, m2)
        a = exp(m1 - m), b = exp(m2 - m)     both in (0, 1] -> no overflow
        A = a * A1 + b * A2
        l = a * l1 + b * l2

    A head unseen on both sides has m1 = m2 = -inf; m stays -inf and the scales
    are forced to 0 (its mass is already 0), which avoids exp(nan). The combine
    is commutative up to fp rounding; a fixed ring order makes the fold
    deterministic.
    """
    m = np.maximum(p1.m, p2.m)
    dead = np.isneginf(m)  # head unseen on both sides
    with np.errstate(invalid="ignore"):  # -inf minus -inf on dead lanes
        a = np.where(dead, np.float32(0.0), np.exp(p1.m - m)).astype(np.float32)
        b = np.where(dead, np.float32(0.0), np.exp(p2.m - m)).astype(np.float32)
    A = (a[:, None] * p1.A + b[:, None] * p2.A).astype(np.float32)
    l = (a * p1.l + b * p2.l).astype(np.float32)  # noqa: E741  softmax mass
    return FlashPartial(A=A, l=l, m=m.astype(np.float32))


def normalize(p: FlashPartial) -> np.ndarray:
    """Finish the softmax: O[H, d] = A / l. On device this is one SFP realdiv."""
    return (p.A / p.l[:, None]).astype(np.float32)


@dataclasses.dataclass
class Hop:
    """One ring step of the merge: a move, optionally followed by a combine."""

    round: int
    distance: int  # ring hops between src and dst (1 = neighbor)
    payload_bytes: int
    combine: bool  # True if an lse_combine follows this move


@dataclasses.dataclass
class FoldSchedule:
    """The composed merge: ordered hops, cost breakdown, and endpoint layout."""

    topology: str
    hops: list[Hop]
    move_us: float
    combine_us: float
    latency_us: float
    total_us: float
    endpoint_layout: str  # where the result lands
    reduced: FlashPartial
    backend_ask: str


def _hops_for(topology: str, P: int, payload: int) -> tuple[list[Hop], str]:
    """Hop list and endpoint layout for a topology. Payload = per-core bytes."""
    if topology == A_LINEAR:
        # Fold every core's partial into one along the ring: P-1 sequential
        # neighbor hops, each carrying a full partial. Result on one core.
        hops = [Hop(r, 1, payload, True) for r in range(P - 1)]
        return hops, "single_core"

    if topology == B_REDUCE_SCATTER:
        # Reduce-scatter the heads (P-1 combining hops of payload/P), then
        # all-gather (P-1 move-only hops of payload/P). Result lands sharded by
        # head across the cores -- the head-split the O-projection wants, so the
        # O-proj hand-off is relayout-free.
        shard = max(1, payload // P)
        rs = [Hop(r, 1, shard, True) for r in range(P - 1)]
        ag = [Hop(P - 1 + r, 1, shard, False) for r in range(P - 1)]
        return rs + ag, "head_split"

    if topology == C_HALVING:
        # Recursive-doubling all-reduce: log2(P) rounds, round r exchanges with
        # the peer at ring distance 2**r. Fewer launches than A, but on a
        # physical ring a distance-2**r hop traverses 2**r links, so its latency
        # only collapses to A's if distance is flat (see ring_distance_flat).
        rounds = int(math.log2(P))
        hops = [Hop(r, 1 << r, payload, True) for r in range(rounds)]
        return hops, "all_replicated"

    raise ValueError(f"unknown fold topology: {topology}")


def ring_fold(
    partials: list[FlashPartial],
    topology: str,
    P: int,
    lambda_us: float = HOP_LATENCY_US,
    rho: float = RING_GBPS,
    ring_distance_flat: bool = False,
) -> FoldSchedule:
    """Merge P local partials into one, composing SHUFFLE + lse_combine per hop.

    Returns the hop schedule and its cost. The reduced partial is computed here
    by folding lse_combine in ring order -- that is the real frontend semantics,
    runnable now; on device the same numbers come out of the SFP.

    Cost model (per hop):
      move     bytes / rho
      combine  bytes / SFP        (only on combining hops)
      latency  distance * lambda  (distance collapses to 1 iff the ring can
                                    address a distance-d peer at flat cost)

    So the three topologies trade off exactly as the RFC frames them:
      A  (P-1) full-payload combining hops, distance 1   -> bandwidth-heavy
      B  2(P-1) hops of payload/P                          -> fewest bytes, most
                                                             launches, head-split
      C  log2(P) rounds                                    -> fewest launches,
                                                             degrades to A's
                                                             latency unless flat
    lambda is the number that sets the crossover; it is L-independent.
    """
    if not partials:
        raise ValueError("ring_fold needs at least one partial")
    payload = partials[0].nbytes

    hops, endpoint = _hops_for(topology, P, payload)

    move_us = sum(_us(h.payload_bytes, rho) for h in hops)
    combine_us = sum(_us(h.payload_bytes, SFP_GBPS) for h in hops if h.combine)
    latency_us = sum(
        (1 if ring_distance_flat else h.distance) * lambda_us for h in hops
    )
    total_us = move_us + combine_us + latency_us

    reduced = partials[0]
    for p in partials[1:]:
        reduced = lse_combine(reduced, p)

    return FoldSchedule(
        topology=topology,
        hops=hops,
        move_us=round(move_us, 3),
        combine_us=round(combine_us, 3),
        latency_us=round(latency_us, 3),
        total_us=round(total_us, 3),
        endpoint_layout=endpoint,
        reduced=reduced,
        backend_ask=(
            "fuse move+combine into one hop and overlap a hop's move with the "
            "prior hop's combine (deeptools input-fetch-overlap gate, today "
            "Conv-only, must accept matmul/SFP consumers); the composition here "
            "is frontend and buildable now"
        ),
    )


def _self_test() -> None:
    """lse_combine folds shard-wise partials back to a dense softmax attention,
    and the three topologies price out as the RFC predicts."""
    rng = np.random.default_rng(0)
    H, d, L, P = 4, 16, 96, 8
    q = rng.standard_normal((H, d)).astype(np.float32)
    k = rng.standard_normal((L, H, d)).astype(np.float32)
    v = rng.standard_normal((L, H, d)).astype(np.float32)

    # Dense reference: full softmax(q k^T) @ v per head.
    dense = np.empty((H, d), dtype=np.float32)
    for h in range(H):
        s = q[h] @ k[:, h, :].T
        w = np.exp(s - s.max())
        w /= w.sum()
        dense[h] = w @ v[:, h, :]

    # Shard L across P cores block-cyclically, build each partial, fold them.
    partials = []
    for core in range(P):
        idx = [j for j in range(L) if j % P == core]  # B_kv = 1
        p = FlashPartial.empty(H, d)
        for h in range(H):
            if not idx:
                continue
            s = q[h] @ k[idx, h, :].T
            mx = s.max()
            w = np.exp(s - mx)
            p.A[h] = w @ v[idx, h, :]
            p.l[h] = w.sum()
            p.m[h] = mx
        partials.append(p)

    for topo in (A_LINEAR, B_REDUCE_SCATTER, C_HALVING):
        sched = ring_fold(partials, topo, P)
        out = normalize(sched.reduced)
        err = float(np.abs(out - dense).max())
        assert err < 1e-4, (topo, err)
        print(f"{topo:<18} err={err:.2e}  {sched.total_us:.3f}us  "
              f"{sched.endpoint_layout}")


if __name__ == "__main__":
    _self_test()
