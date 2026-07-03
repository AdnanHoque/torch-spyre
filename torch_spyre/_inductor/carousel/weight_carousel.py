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

"""Weight-carousel work-division plan for a prefill matmul Y[S,N]=X[S,K]@W[K,N].

Naive M-split gives each of P cores a full copy of W, so W is read from DRAM P
times. The carousel loads each core one N-tile of W, then rotates tiles around
the ring: over P steps every core sees every tile and computes its full-N output
row-block. DRAM weight bytes drop from P*|W| to |W|. Output lands under the SAME
M-split it entered with, so the op is a drop-in swap for naive M-split (see
`seam_layout`).

The group carousel exposes one knob r (replication radius, a divisor of P): the
ring is cut into r arcs of P'=P/r cores, each arc holds one full copy of W.
  r = P  -> P arcs of 1 core, no rotation  = today's naive full replication.
  r = 1  -> one arc of P cores, full rotation = maximum DRAM saving.
So old and new plans are the two endpoints of one cost-model family, priced by
`weight_carousel_cost`.

This module is PLANNING + COST ONLY. It changes no numerics and emits no graph.
The physical rotation (SHUFFLE emission + PSUM tile schedule) is a backend job;
`emit_rotation_step` is the interface stub naming the exact deeptools dependency.
"""

from __future__ import annotations

import dataclasses
import os

# Emission is gated on this flag; enumeration/costing stay callable for analysis
# and tests. In production this belongs in _inductor/config.py alongside the
# other TS_/env knobs; kept local here so the module imports with no torch.
TS_ENABLE_CAROUSEL = os.environ.get("TS_ENABLE_CAROUSEL", "0") == "1"


def carousel_enabled() -> bool:
    return TS_ENABLE_CAROUSEL


@dataclasses.dataclass(frozen=True)
class CarouselHW:
    """AIU constants the cost model reads. Defaults are the established values;
    every field is overridable so a device sweep can refit it.
    """

    cores: int = 32  # P: cores the M-split spans
    lx_bytes_per_core: int = 2 * 1024 * 1024  # per-core scratchpad
    ring_gbps: float = 166.0  # RIU ring bandwidth, per direction (rho)
    hbm_gbps: float = 143.0  # LPDDR read-only stream (weights/X are reads)
    macs_us_core: float = (98.304e12 / 2 / 32) / 1e6  # DL16 peak, MACs/us/core
    dtype_bytes: int = 2  # fp16
    pt_rows: int = 8  # PT pass row block
    target_pt_passes: int = 8  # per-core M passes that fill the PT pipeline
    # lambda: per-hop ring latency. STUB VALUE -- needs a device measurement; it
    # is the number that sets the win/no-win crossover, so treat it as unknown.
    hop_latency_us: float = 0.5
    # P2 overlap gate. False = today: the relayout SHUFFLE is a sequential
    # barrier (producer->move->consumer), so per step compute and transport add.
    # True = compute/transport overlap; NEEDS the backend edit that extends the
    # Conv-only input-fetch overlap gate to matmul consumers. See emit stub.
    overlap: bool = False


@dataclasses.dataclass(frozen=True)
class WeightCarouselPlan:
    """One member of the r-family. split/rot_axis are fixed by construction and
    kept explicit so the plan is self-describing to a reader and a validator.
    """

    S: int  # tokens (M dimension)
    K: int  # contraction
    N: int  # weight/output width (stick dimension)
    P: int  # cores across the M-split
    r: int  # replication radius, divides P; arcs = r, arc size P' = P/r
    K_t: int  # rotated K-slab depth; divides K (double-buffer granularity)
    split: str = "M"  # work is split along M (tokens)
    rot_axis: str = "N"  # N-tiles rotate (keeps each output tile's K-loop local)

    @property
    def arc_cores(self) -> int:
        """P' = P/r: cores per arc = number of carousel steps."""
        return self.P // self.r

    @property
    def tile_width(self) -> int:
        """Per-core N-tile width in elements = N / P'."""
        return self.N // self.arc_cores


def _divisors(n: int) -> list[int]:
    ds: set[int] = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
        i += 1
    return sorted(ds)


def _lx_highwater(plan: WeightCarouselPlan, hw: CarouselHW) -> int:
    """Peak per-core LX bytes during the loop: a double-buffered W K-slab, the
    matching X K-slab, and the fp32 PSUM accumulator for the output tile.
    A whole N-tile (K*tile_width) is ~1 MiB, too big to double-buffer, so the
    rotated unit is a K_t-deep slab (RFC transport granularity).
    """
    b = hw.dtype_bytes
    m_t = plan.S // plan.P  # per-core token rows
    w_slab = 2 * plan.K_t * plan.tile_width * b  # rotating W slab, double-buffered
    x_slab = m_t * plan.K_t * b  # X rows against the same K_t depth
    psum_tile = m_t * plan.tile_width * 4  # PSUM is fp32
    return w_slab + x_slab + psum_tile


def weight_carousel_cost(plan: WeightCarouselPlan, hw: CarouselHW) -> dict:
    """Per-resource cost of one plan. Returns dram_bytes, link_occupancy,
    lx_highwater, t_est (all required keys) plus the t_est breakdown.

    t_est = dram + ring + compute + (P'-1)*lambda, combined by the overlap
    model in `hw`. HONEST: for wide-N prefill the compute term dominates
    (array-under-filled ~30% PT-util), so cutting dram_bytes barely moves t_est
    -- the carousel's prefill value is seam-transparency + array-fill, not a
    wall-clock DRAM win. The model shows this directly.
    """
    b = hw.dtype_bytes
    S, K, N, P, r = plan.S, plan.K, plan.N, plan.P, plan.r
    Pp = plan.arc_cores

    W = K * N * b
    X = S * K * b
    Y = S * N * b

    # r arcs each load one full copy of W; X read once, Y written once (M-split).
    dram_bytes = r * W + X + Y

    # Feasibility: clean M-split, integer N-tiles, K_t divides K.
    feasible = (S % P == 0) and (N % Pp == 0) and (K % plan.K_t == 0)
    if not feasible:
        return {
            "dram_bytes": dram_bytes,
            "link_occupancy": 0,
            "lx_highwater": None,
            "t_est": float("inf"),
        }

    # Ring: over the rotation each of the P' tiles crosses P'-1 links; spread
    # over the arc's P' links that is (P'-1)/P'*|W| bytes per link. r-invariant
    # per link (arcs are disjoint ring segments). Zero when P'==1 (no rotation).
    link_occupancy = 0.0 if Pp == 1 else (Pp - 1) / Pp * W

    lx_highwater = _lx_highwater(plan, hw)
    if lx_highwater > hw.lx_bytes_per_core:
        return {
            "dram_bytes": dram_bytes,
            "link_occupancy": link_occupancy,
            "lx_highwater": lx_highwater,
            "t_est": float("inf"),
        }

    # Compute: total work S*K*N over P cores, r-invariant. Array-under-fill
    # derate keyed on per-core M=S/P -- the same knob the production matmul cost
    # model uses (pt_eff = sqrt(pt_passes/target)). Carousel keeps M-split=P, so
    # this term is identical to naive M-split: the carousel does not change
    # compute util, only DRAM bytes.
    m_t = S // P
    pt_passes = max(1.0, m_t / hw.pt_rows)
    pt_eff = min(1.0, (pt_passes / hw.target_pt_passes) ** 0.5)
    t_compute = (S * K * N / P) / (hw.macs_us_core * pt_eff)

    # Ring wall time: per-link bytes / rho (links run in parallel).
    t_ring = link_occupancy / (hw.ring_gbps * 1e3)
    # DRAM wall time for the whole plan's byte budget.
    t_dram = dram_bytes / (hw.hbm_gbps * 1e3)
    t_lambda = max(0, Pp - 1) * hw.hop_latency_us

    if hw.overlap:
        # Full overlap (needs the backend gate): critical resource sets the wall.
        t_est = max(t_dram, t_ring, t_compute) + t_lambda
    else:
        # Today: sequential barrier -> resources add.
        t_est = t_dram + t_ring + t_compute + t_lambda

    return {
        "dram_bytes": dram_bytes,
        "link_occupancy": link_occupancy,
        "lx_highwater": lx_highwater,
        "t_est": t_est,
        # breakdown (not required keys, cheap and load-bearing for honesty)
        "t_dram": t_dram,
        "t_ring": t_ring,
        "t_compute": t_compute,
        "t_lambda": t_lambda,
        "pt_eff": pt_eff,
    }


def enumerate_weight_carousel_plans(
    S: int, K: int, N: int, P: int, hw: CarouselHW
) -> list[WeightCarouselPlan]:
    """Yield the r-family: one plan per replication radius r in divisors(P) that
    admits a clean split. For each r pick the largest K_t (dividing K) whose LX
    high-water fits -- the coarsest feasible double-buffer. K_t is itself a
    sweepable knob; largest-that-fits keeps the family one plan per r.
    """
    plans: list[WeightCarouselPlan] = []
    for r in _divisors(P):
        Pp = P // r
        if S % P != 0 or N % Pp != 0:
            continue  # no clean M-split or integer N-tile at this r
        for K_t in reversed(_divisors(K)):  # largest first
            cand = WeightCarouselPlan(S=S, K=K, N=N, P=P, r=r, K_t=K_t)
            if _lx_highwater(cand, hw) <= hw.lx_bytes_per_core:
                plans.append(cand)
                break  # coarsest feasible K_t for this r
    return plans


def plan_weight_carousel(
    S: int, K: int, N: int, P: int, hw: CarouselHW | None = None
) -> WeightCarouselPlan | None:
    """Pick the lowest-t_est member of the r-family. Returns None when the
    TS_ENABLE_CAROUSEL flag is off (emission gate) or nothing is feasible.
    """
    if not carousel_enabled():
        return None
    hw = hw or CarouselHW(cores=P)
    plans = enumerate_weight_carousel_plans(S, K, N, P, hw)
    if not plans:
        return None
    return min(plans, key=lambda p: weight_carousel_cost(p, hw)["t_est"])


def seam_layout(plan: WeightCarouselPlan) -> dict:
    """Seam-transparent layout descriptor. The carousel consumes X and produces
    Y under the SAME M-split: phase p (core p) owns token rows
    [p*S/P, (p+1)*S/P) on BOTH input and output. Identical in/out layout is what
    makes the plan a drop-in swap for naive M-split -- no relayout at either
    seam. `seam_transparent` is the property a work-division validator checks.
    """
    block = plan.S // plan.P
    phase_rows = [(p * block, (p + 1) * block) for p in range(plan.P)]
    return {
        "split_axis": "M",
        "phase_rows_in": phase_rows,
        "phase_rows_out": phase_rows,  # identical -> no seam relayout
        "seam_transparent": True,
    }


def emit_rotation_step(*args, **kwargs):
    """STUB -- physical realization of one carousel step. NOT IMPLEMENTED.

    One step is: (1) each core does its full-K local matmul of its X rows against
    its current W K-slab into a PSUM tile; (2) rotate every core's W slab one hop
    around its arc. This is a pure work-division/movement plan (no new numerics),
    but the emission is a backend job with these exact deeptools dependencies:

    MOVEMENT (the rotate) -- lowers as an STCDPOpLx LX->LX reshard through the
      generic relayout path `Dxp::insertRelayoutSdsc` (SdscRelayoutInsertion.cpp,
      invoked from dxp.cpp runDsmRelayout). A delta-1 ring rotation is a subset
      of an arbitrary coreIdToWkSlice_ permutation, so it is already accepted;
      multicast (for r>1 group broadcast) is live via reqMulticast (stcdpOp.cpp).
      LX->LX with no HBM bounce holds only while the destination slab fits LX --
      that is exactly what K_t sizes (see `_lx_highwater`). [VERIFIED path.]

    OVERLAP (the perf gate, P2) -- CarouselHW.overlap=True is only real once the
      input-fetch overlap hook (dsmperf.cpp overlapInpFetchWithCompute), today
      hard-gated to Conv2D/SparseConv2D consumers, is extended to matmul/BMM
      consumers. A seam-transparent reshard keeps layoutDimOrder identical, so
      assignCanOverlapInpFetch stays eligible; the gate is the only blocker.
      Until then overlap=False (barrier) is the honest wall-clock. [BACKEND EDIT.]

    SCHEDULE (the PSUM accumulation across steps) -- a DSM concern, not
      expressible from the frontend. [BACKEND.]

    REWARD -- the production matmul cost model counts each weight operand once
      (bytes_total = (M*K+K*N+M*N)*b) and amortizes the broadcast up to cohort 8,
      so a DRAM-byte saving is invisible to it. Rewarding the carousel needs a
      new m-fold weight-replication HBM term in _matmul_split_cost.
      [COST-MODEL EDIT, load-bearing.]
    """
    raise NotImplementedError(
        "emit_rotation_step is a research stub; see docstring for the "
        "STCDPOpLx / overlap-gate / PSUM-schedule backend dependencies."
    )


def _main() -> None:
    """Print the r-family for a representative wide-N Granite prefill matmul.
    Shows the two honest facts: dram_bytes drops toward r=1, but under overlap
    the compute term (array-under-filled) caps the wall -- the prefill win is
    seam-transparency + array-fill, not DRAM.
    """
    S, K, N, P = 512, 4096, 14336, 32
    hw = CarouselHW(cores=P)
    _MiB = 1024 * 1024
    print(f"weight carousel r-family  S={S} K={K} N={N} P={P}  overlap={hw.overlap}")
    header = (
        f"{'r':>3}{'P_arc':>7}{'K_t':>6}{'dram_MiB':>10}"
        f"{'link_MiB':>10}{'lx_KiB':>9}{'t_est_us':>10}"
    )
    print(header)
    print("-" * len(header))
    plans = enumerate_weight_carousel_plans(S, K, N, P, hw)
    best = min(plans, key=lambda p: weight_carousel_cost(p, hw)["t_est"])
    for p in plans:
        c = weight_carousel_cost(p, hw)
        mark = "  <- min t_est" if p is best else ""
        print(
            f"{p.r:>3}{p.arc_cores:>7}{p.K_t:>6}"
            f"{c['dram_bytes'] / _MiB:>10.1f}{c['link_occupancy'] / _MiB:>10.1f}"
            f"{c['lx_highwater'] / 1024:>9.1f}{c['t_est']:>10.1f}{mark}"
        )


if __name__ == "__main__":
    _main()
