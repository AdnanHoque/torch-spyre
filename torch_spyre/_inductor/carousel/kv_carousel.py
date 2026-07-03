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

"""KV carousel frontend: sequence-sharded decode attention.

Decode attention at batch 1 is LPDDR-bound. A head-split KV cache activates only
H_kv of P LPDDR channels (~25% aggregate ceiling) and underfills each. This
carousel instead shards the KV cache along SEQUENCE across all P cores, so every
channel streams: the ceiling rises by up to P / H_kv. Placement is a compile-
time function (no runtime page table in v1), which is why this does NOT apply to
vLLM/paged-attention near term.

What is real vs stubbed here:
  * kv_placement: real, pure arithmetic, testable now.
  * FlashPartial / lse_combine / ring_fold: real (see fold.py).
  * local_flash_partial_pass: INTERFACE + stub. The online-softmax kernel it
    describes is a backend deliverable; the live SDPA lowering is decomposed
    (dense softmax, no flash), so this op does not exist yet.
  * decode_step_plan: a costing/plan composition, not an executor. It orders the
    ops and prices them; the local pass and the fused fold hop are backend asks.
"""

from __future__ import annotations

import dataclasses
import math

# FlashPartial is defined in fold.py (the fold's data currency) and re-exported
# here as the local pass's output type; defining it there breaks the import
# cycle (kv_carousel -> fold, not the reverse).
from .fold import (
    B_REDUCE_SCATTER,
    FlashPartial,
    FoldSchedule,
    HOP_LATENCY_US,
    RING_GBPS,
    SFP_GBPS,
    _us,
    ring_fold,
)

# Defaults are AIU/Granite constants; production reads P from config.sencores.
DEFAULT_P = 32  # cores == LPDDR channels
DEFAULT_H = 32  # query heads
DEFAULT_H_KV = 8  # stored KV heads (GQA group = H // H_kv)
DEFAULT_D = 128  # head dim
DEFAULT_B_KV = 64  # decode block-cyclic block size (strawman)

FP16_BYTES = 2
FP32_BYTES = 4
# Read-only LPDDR aggregate (the KV stream is a read); one core drives its share.
HBM_READ_GBPS = 143.0


@dataclasses.dataclass(frozen=True)
class KVSegment:
    """Piecewise cache layout at a decode step.

    prefill_len (S): positions [0, S) were laid down contiguously during
    prefill; positions >= S are the block-cyclic decode tail. S = 0 means a
    pure-decode cache (no prefill segment).
    """

    prefill_len: int


def kv_placement(pos: int, P: int, B_kv: int, segment: KVSegment) -> int:
    """Owning core for KV cache position ``pos``. Compile-time, no page table.

    Piecewise, so the prefill and decode carousels compose seam-free:
      * prefill [0, S): contiguous ``(pos * P) // S`` -- the same core the
        token-split (M-split) prefill matmul assigns to token pos, so the KV it
        writes lands local (zero relayout into the decode phase).
      * decode [S, .): block-cyclic in blocks of B_kv, ``(off // B_kv) % P``
        with ``off = pos - S`` -- spreads the growing cache across all P cores
        so every LPDDR channel streams.
    """
    S = segment.prefill_len
    if pos < S:
        return (pos * P) // S
    off = pos - S
    return (off // B_kv) % P


@dataclasses.dataclass(frozen=True)
class OwnedKVBlocks:
    """The KV shard one core owns for the current decode step.

    On device key/value are LX/HBM base pointers + block extents produced by the
    kv_placement partition; here they are opaque handles the kernel consumes.
    Each covers all H_kv stored heads over this core's owned positions.
    """

    key: object  # [n_owned, H_kv, d] handle
    value: object  # [n_owned, H_kv, d] handle
    n_owned: int  # positions this core owns this step


def local_flash_partial_pass(
    q: object,  # [H, d] query for this token, all heads, on-core
    kv: OwnedKVBlocks,
    H: int,
    H_kv: int,
    d: int,
    scale: float,
) -> FlashPartial:
    """One core's flash pass over its owned KV shard -> an UNNORMALIZED partial.

    KERNEL DELTA (backend deliverable, not composable from today's ops): the
    live SDPA lowering DECOMPOSES attention -- full QK^T matmul, a dense softmax
    over the whole Skv, then attn@V, with a dead logsumexp. This pass instead
    streams the core's owned KV blocks and keeps a running (A, l, m) via the
    online-softmax recurrence, per query head h:

        for each owned KV position j (score s = scale * <q[h], k[j]>):
            m_new = max(m[h], s)
            alpha = exp(m[h] - m_new)          # rescale the running accumulator
            A[h]  = alpha * A[h] + exp(s - m_new) * v[j]
            l[h]  = alpha * l[h] + exp(s - m_new)
            m[h]  = m_new

    GQA: each of the H_kv stored KV heads is streamed once and applied to its
    G = H // H_kv query heads; the partial is emitted at query-head granularity
    (H rows) so the fold and O-proj see full heads. m starts at -inf, A/l at 0
    (FlashPartial.empty), so a core owning no positions emits the fold identity.
    The output is UNNORMALIZED: division by l happens once after the ring fold.
    """
    raise NotImplementedError(
        "local_flash_partial_pass is the online-softmax kernel the backend must "
        "provide; the live SDPA path is decomposed (dense softmax, no flash). "
        "This frontend defines its contract and the (A, l, m) it emits."
    )


@dataclasses.dataclass
class Stage:
    """One step of the decode-attention pipeline, with cost and who owns it."""

    name: str
    est_us: float
    owner: str  # "frontend" (composable now) | "backend" (op/kernel ask)
    note: str


@dataclasses.dataclass
class DecodeStepPlan:
    """Ordered per-layer decode-attention plan with a cost breakdown."""

    stages: list[Stage]
    fold: FoldSchedule
    total_us: float
    layout_handoff: str
    ceiling_lift: float  # streaming speedup vs head-split (~ P / H_kv)


def decode_step_plan(
    L: int,
    topology: str = B_REDUCE_SCATTER,
    P: int = DEFAULT_P,
    H: int = DEFAULT_H,
    H_kv: int = DEFAULT_H_KV,
    d: int = DEFAULT_D,
    lambda_us: float = HOP_LATENCY_US,
    rho: float = RING_GBPS,
) -> DecodeStepPlan:
    """Compose one layer's decode attention as ordered, priced stages:

        Q-broadcast -> local flash pass -> ring fold -> normalize -> O-proj.

    L is the current cache length. Costs are first-order (bytes / bandwidth);
    the local pass and the fused fold hop are backend asks and marked as such.
    The local-pass cost is streaming-ceiling only and needs device measurement.
    """
    if H % H_kv:
        raise ValueError(f"H={H} not divisible by H_kv={H_kv}")

    # 1. Q broadcast: one token, all heads, fp16, multicast to all P cores.
    #    Multicast is one STCDPOpLx op (P1 live); it circulates the ring once.
    q_bytes = H * d * FP16_BYTES
    q_us = _us(q_bytes, rho) + (P - 1) * lambda_us

    # 2. Local flash pass: each core streams its owned KV (K and V, fp16). With
    #    sequence sharding all P channels stream, so wall time is per-core bytes
    #    over the per-core LPDDR share. Head-split would stream only H_kv
    #    channels over the full sequence -> ceiling_lift ~ P / H_kv.
    per_core_bw = HBM_READ_GBPS / P
    owned = math.ceil(L / P)
    kv_bytes_core = 2 * owned * H_kv * d * FP16_BYTES  # K + V
    local_us = _us(kv_bytes_core, per_core_bw)
    headsplit_bw = HBM_READ_GBPS / H_kv  # only H_kv channels active
    headsplit_us = _us(2 * L * d * FP16_BYTES, headsplit_bw)
    ceiling_lift = headsplit_us / local_us if local_us else float("inf")

    # 3. Ring fold of the P partials. Placeholder partials for byte accounting
    #    only (cost is shape-driven); the numeric merge happens on real partials
    #    from stage 2.
    placeholders = [FlashPartial.empty(H, d) for _ in range(P)]
    fold = ring_fold(placeholders, topology, P, lambda_us=lambda_us, rho=rho)

    # 4. Normalize O = A / l, one SFP realdiv over H*d fp32.
    norm_us = _us(H * d * FP32_BYTES, SFP_GBPS)

    # 5. O-proj hand-off. If the fold lands head-split (topology B) that IS the
    #    K-split the O-projection wants -> zero relayout. Otherwise O must be
    #    resharded to head-split first (one move of the fp16 output).
    if fold.endpoint_layout == "head_split":
        handoff_us = 0.0
        handoff = "zero relayout: fold endpoint head-split == O-proj K-split"
    else:
        handoff_us = _us(H * d * FP16_BYTES, rho) + (P - 1) * lambda_us
        handoff = (
            f"relayout: fold endpoint {fold.endpoint_layout} -> O-proj wants "
            "head-split; one reshard"
        )

    stages = [
        Stage("q_broadcast", round(q_us, 3), "frontend",
              "STCDPOpLx multicast, one token to P cores (P1 live)"),
        Stage("local_flash_pass", round(local_us, 3), "backend",
              f"online-softmax kernel delta; ceiling lift ~{ceiling_lift:.1f}x "
              "vs head-split; device-measured"),
        Stage("ring_fold", fold.total_us, "frontend", fold.backend_ask),
        Stage("normalize", round(norm_us, 3), "frontend", "SFP realdiv O=A/l"),
        Stage("o_proj_handoff", round(handoff_us, 3), "frontend", handoff),
    ]
    total_us = round(sum(s.est_us for s in stages), 3)

    return DecodeStepPlan(
        stages=stages,
        fold=fold,
        total_us=total_us,
        layout_handoff=handoff,
        ceiling_lift=round(ceiling_lift, 2),
    )
