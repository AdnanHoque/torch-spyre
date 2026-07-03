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

"""Analytical go/no-go for the weight and KV carousels.

Both carousels trade DRAM weight/KV bytes for on-chip ring traffic. Whether
that trade wins is a roofline question with two knobs the probes measure:
ring bandwidth per direction (rho) and per-hop latency (lambda). This module
turns (rho, lambda, C, u) into a verdict, before any device time is spent.

Weight carousel (prefill Y[S,N]=X[S,K]@W[K,N], M-split over P cores):
  Per step each core does one local matmul on its S/P rows and one N-tile,
  while the next N-tile rotates in. Transport hides under compute when
      rho >= u * C * b / (2 S)                              (the compute gate)
  Derivation: t_c = 2*(S/P)*K*(N/P)/C_core, t_x = K*(N/P)*b/rho; cancel the
  tile K*(N/P); C_core = u*C/P so P cancels -> the gate above. K and N drop
  out: the gate is set by tokens S and the array fill u alone.

  HONEST CAVEAT: prefill runs at u ~= 0.3 (wide-N array under-fill). At that u
  the gate is *easy* to pass for S >= ~1024 -- but only because the array is
  slow, which is the very thing the carousel exists to fix. So passing the
  gate means the ring is not the bottleneck; it does NOT mean an xP wall-clock
  win. The carousel's value is seam-transparency + array-fill, not DRAM bytes.

KV carousel (decode, BS=1, sequence-sharded over all P channels):
  Head-split leaves only H_kv of P channels streaming KV; the carousel lights
  all P -> the KV-stream term speeds up by ~P/H_kv. Cost is an L-independent
  per-step overhead: Q broadcast + the LSE ring-fold merge. Crossover:
      L_min = (T_bcast + T_merge) * BW_car / (2 * d * b * (P - H_kv))
  Merge is on the critical path every layer every step and is L-independent,
  so lambda sets the crossover. Merge topology is a frontend choice priced
  here for the measured lambda.

A3 caveat: 'all P channels stream' is realizable only via core-ownership of
the shards, not HBM channel placement (the compiler cannot pin a channel --
see probes.py P5). u_kv folds in the fill that core-ownership actually gets.
"""

from __future__ import annotations

import dataclasses
import math


@dataclasses.dataclass(frozen=True)
class HW:
    """Hardware + fabric parameters. Defaults are this stream's constants;
    rho and lam are placeholders until probes.py measures them on device."""

    P: int = 32  # cores == LPDDR channels
    b: int = 2  # bytes/element, fp16
    rho: float = 166.0e9  # ring B/s per direction (RIU BiRing, one dir)
    lam: float = 0.5e-6  # per-hop latency, s. UNMEASURED -> probes.py P5.
    C: float = 3.0e14  # array peak flop/s (~300 TOPS fp16, MAC=2 flop)
    u: float = 0.30  # prefill wide-N array fill fraction (measured under-fill)
    # KV / decode side.
    hbm_read: float = 143.0e9  # read-only LPDDR bus ceiling, B/s
    u_kv: float = 0.70  # per-channel fill the carousel realizes at long L
    H: int = 32  # query heads
    H_kv: int = 8  # KV heads (GQA group = H/H_kv)
    d: int = 128  # head dim


# ---- weight carousel ------------------------------------------------------

WEIGHT_S = (512, 1024, 2048, 4096)


def rho_min_weight(hw: HW, S: int, u: float | None = None) -> float:
    """Minimum ring B/s for transport to hide under compute at prefill S.

    rho_min = u * C * b / (2 S). Independent of K, N (they cancel with the
    per-core tile). Pass u=1.0 to see the peak-array (fully-filled) bar.
    """
    u = hw.u if u is None else u
    return u * hw.C * hw.b / (2.0 * S)


@dataclasses.dataclass(frozen=True)
class WeightVerdict:
    S: int
    rho_min_underfill: float  # at measured u
    rho_min_peak: float  # at u=1.0
    rho: float
    go: bool  # transport hidden at the measured u


def weight_gate(hw: HW) -> list[WeightVerdict]:
    out = []
    for S in WEIGHT_S:
        rmin = rho_min_weight(hw, S)
        out.append(
            WeightVerdict(
                S=S,
                rho_min_underfill=rmin,
                rho_min_peak=rho_min_weight(hw, S, u=1.0),
                rho=hw.rho,
                go=hw.rho >= rmin,
            )
        )
    return out


# ---- KV carousel ----------------------------------------------------------

# Merge fold over the per-head partials (A[h,d] fp32, l fp32, m fp32) for all
# heads. Payload is tiny (KB), so at 166 GB/s the per-hop transfer is < 0.1us
# and lambda dominates -> topology choice is a hop-count choice.
def _merge_payload_bytes(hw: HW) -> int:
    per_head = hw.d * 4 + 4 + 4  # A[d] fp32 + l + m
    return hw.H * per_head


def merge_cost(hw: HW, topology: str) -> float:
    """Seconds for one LSE ring-fold, by frontend topology.

      A linear        : P-1 hops, full payload each
      B rs+allgather  : 2(P-1) hops, payload/P each (endpoint head-split)
      C recursive-half: log2(P) rounds, ~payload/2 each (wins if lambda-bound)
    """
    pl = _merge_payload_bytes(hw)
    P = hw.P
    if topology == "A":
        hops = P - 1
        return hops * (pl / hw.rho + hw.lam)
    if topology == "B":
        hops = 2 * (P - 1)
        return hops * (pl / P / hw.rho + hw.lam)
    if topology == "C":
        rounds = int(math.log2(P))
        return rounds * (pl / 2 / hw.rho + hw.lam)
    raise ValueError(topology)


def bcast_cost(hw: HW) -> float:
    """Q broadcast: one token, all heads, over the ring + one hop of latency."""
    q_bytes = hw.H * hw.d * hw.b
    return q_bytes / hw.rho + hw.lam


def bw_car(hw: HW) -> float:
    """Carousel KV-stream bandwidth: all P channels at fill u_kv."""
    return hw.hbm_read * hw.u_kv


def bw_base(hw: HW) -> float:
    """Head-split baseline: only H_kv of P channels stream."""
    return bw_car(hw) * hw.H_kv / hw.P


def kv_crossover_L(hw: HW, topology: str) -> float:
    """Tokens L above which the carousel decode step beats head-split.

    L_min = (T_bcast + T_merge) * BW_car / (2 d b (P - H_kv)). The stream
    saving grows linearly in L; the overhead is L-independent.
    """
    overhead = bcast_cost(hw) + merge_cost(hw, topology)
    per_L_saving_rate = 2 * hw.d * hw.b * (hw.P - hw.H_kv) / bw_car(hw)
    return overhead / per_L_saving_rate


def kv_step_speedup(hw: HW, L: int, topology: str) -> float:
    """Decode-step KV-term speedup at sequence length L (carousel/baseline)."""
    kv_bytes = 2 * L * hw.H_kv * hw.d * hw.b  # K+V, owned positions
    t_base = kv_bytes / bw_base(hw)
    t_car = kv_bytes / bw_car(hw) + bcast_cost(hw) + merge_cost(hw, topology)
    return t_base / t_car


# ---- report ---------------------------------------------------------------

def _gbps(x: float) -> float:
    return x / 1e9


def _main() -> None:
    hw = HW()
    print("carousel roofline  (HW defaults; rho/lambda are probes.py inputs)")
    print(
        f"  P={hw.P}  rho={_gbps(hw.rho):.0f} GB/s  lambda={hw.lam*1e6:.2f} us  "
        f"C={hw.C:.1e} flop/s  u_prefill={hw.u}"
    )

    print("\n== WEIGHT carousel: compute-bound gate  rho >= u*C*b/(2S) ==")
    print(f"  {'S':>6}{'rho_min@u=.3':>14}{'rho_min@u=1':>13}{'rho':>8}{'verdict':>9}")
    for v in weight_gate(hw):
        print(
            f"  {v.S:>6}{_gbps(v.rho_min_underfill):>12.0f}GB"
            f"{_gbps(v.rho_min_peak):>11.0f}GB{_gbps(v.rho):>6.0f}GB"
            f"{('GO' if v.go else 'NO-GO'):>9}"
        )
    print("  CAVEAT: GO here means the ring hides transport *because the array")
    print("  is under-filled* (u~0.3) -- not an xP wall-clock win. The carousel")
    print("  pays off via seam-transparency + array-fill; DRAM-byte saving is a")
    print("  byte-count win, not a prefill wall-clock win at this u.")

    print(f"\n== KV carousel: crossover L_min (win ceiling ~ P/H_kv = {hw.P//hw.H_kv}x) ==")
    print(f"  {'topology':>18}{'T_merge(us)':>13}{'L_min':>8}")
    names = {"A": "A linear", "B": "B rs+allgather", "C": "C recursive-half"}
    for t in ("A", "B", "C"):
        print(
            f"  {names[t]:>18}{merge_cost(hw, t)*1e6:>13.2f}"
            f"{kv_crossover_L(hw, t):>8.0f}"
        )
    print("  lambda sets these; it is UNMEASURED -> probes.py P5. Sweep:")
    print(f"    {'lambda(us)':>12}" + "".join(f"{t:>8}" for t in ("A", "B", "C")))
    for lam_us in (0.25, 0.5, 1.0, 2.0, 4.0):
        h = dataclasses.replace(hw, lam=lam_us * 1e-6)
        row = "".join(f"{kv_crossover_L(h, t):>8.0f}" for t in ("A", "B", "C"))
        print(f"    {lam_us:>12.2f}{row}")

    print("\n  decode-step KV-term speedup vs head-split (topology C):")
    print(f"    {'L':>7}{'speedup':>9}")
    for L in (256, 512, 1024, 2048, 4096, 8192):
        print(f"    {L:>7}{kv_step_speedup(hw, L, 'C'):>8.2f}x")


if __name__ == "__main__":
    _main()
