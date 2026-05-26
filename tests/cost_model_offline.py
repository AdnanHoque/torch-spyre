"""Offline cost-model validator for the matmul work-division planner.

NO torch / no compile / no device. Pure-Python exploration of whether a
cost function can reproduce every validated split decision (QO, KV, MLP,
MoE gate/up, MoE down, large/small-K bmm) before we wire anything in to
work_division.py. Iterate on the cost function until every shape's
``argmin cost`` matches the ``expected_split``; *only then* plumb it into
the planner.

Run:  python3 tests/cost_model_offline.py
"""

from __future__ import annotations

import dataclasses
import math
from sympy import divisors


# =============================================================================
# Hardware constants (AIU 1.0; see reference_aiu_architecture memory)
# =============================================================================
MAX_CORES = 32
PT_ROWS = 8                  # PT block rows per corelet
STICK_BYTES = 128            # 64 elems x fp16
DTYPE_BYTES = 2              # fp16
HBM_BW_GBS = 166             # aggregate HBM BW
LX_TOTAL_BYTES_PER_CORE = 2 * 1024 * 1024   # 2MB LX scratchpad per core (placeholder; refine)
# Peak MACs/us per core: 300 TOPS aggregate / 2 (ops per MAC) / 32 cores / 1e6 us/s
PEAK_MACS_PER_US_PER_CORE = (300e12 / 2 / MAX_CORES) / 1e6      # ~4.7M MACs/us/core
PSUM_PER_ELEM_US = 4e-4          # PSUM cost per output element per ring hop (fit from MoE: k=2 added ~3.4ms / 8.4M elems / 1 hop)
M_SPLIT_MIN = 4                  # smallest useful m-split (heuristic M_MIN; under it, PT array under-fed)
TARGET_M_PENALTY_US = 50.0       # per log2 step away from the shape-aware target m_split (fit from QO: 6% gap m=8 vs m=4)
TARGET_PT_PASSES = 8             # per-core PT passes for full pipeline utilisation
COHORT_BROADCAST_LIMIT = 8       # cohort sizes <= this broadcast cheaply; beyond, contention grows
BATCH_SPLIT_PENALTY = 0.6        # multiplicative penalty per batch-split step (empirical, large-K bmm fit)


# =============================================================================
# Cost function
# =============================================================================
@dataclasses.dataclass
class Hw:
    max_cores: int = MAX_CORES
    pt_rows: int = PT_ROWS
    dtype_bytes: int = DTYPE_BYTES
    hbm_bw_gbs: float = HBM_BW_GBS
    lx_per_core: int = LX_TOTAL_BYTES_PER_CORE
    peak_macs_us_core: float = PEAK_MACS_PER_US_PER_CORE
    psum_per_elem_us: float = PSUM_PER_ELEM_US
    target_pt_passes: int = TARGET_PT_PASSES
    cohort_broadcast_limit: int = COHORT_BROADCAST_LIMIT
    batch_split_penalty: float = BATCH_SPLIT_PENALTY
    m_split_min: int = M_SPLIT_MIN
    target_m_penalty_us: float = TARGET_M_PENALTY_US
    lx_frac: float = 0.8       # DXP_LX_FRAC_AVAIL


def split_cost(B, M, K, N, b, m, n, k, hw: Hw) -> float:
    """Estimate kernel cost (us) for a matmul ``[B,M,K]@[B,K,N]`` with the
    given (b, m, n, k) split across cores. Returns +inf if infeasible.

    v1 model (broadcast HBM + PT-pipeline penalty + PSUM):
      * HBM is broadcast within cohorts: each unique LHS/RHS/OUT byte is read
        once aggregate, regardless of cohort size (cohort cores share via LX
        broadcast). This is the *upper bound* on broadcast efficiency.
      * PT-pipeline efficiency: per-core M should give >= target_pt_passes
        PT passes (=8 at high LX) for the DDC to overlap memory and compute.
        Under-pipelined splits have a derated effective compute throughput.
      * PSUM cost: k-split reduction is non-trivial; model 50us per ring hop
        (empirically fit so pure-K loses to 2D for QO at LX=0.8).
      * Cost = compute + hbm + psum (no overlap; conservative). Refine to
        partial overlap when we have data to fit it.
    """
    cores_used = b * m * n * k
    if cores_used > hw.max_cores or cores_used == 0:
        return float("inf")

    # Per-core tile (elements)
    m_t = M // m
    n_t = N // n
    k_t = K // k

    # ---- Compute with PT-pipeline penalty ----
    # PT pipeline wants >= target_pt_passes per core to hide memory + setup.
    # Effective peak derates linearly when per-core M can't sustain it.
    target_pt_passes = hw.target_pt_passes
    pt_passes = max(1.0, m_t / hw.pt_rows)
    pt_efficiency = min(1.0, pt_passes / target_pt_passes)
    effective_peak = hw.peak_macs_us_core * pt_efficiency
    total_macs = B * M * N * K
    compute_us = (total_macs / cores_used) / effective_peak

    # ---- HBM with broadcast + cohort-contention penalty ----
    # LHS broadcast to n-cohort; RHS to m-cohort; OUT written once per (b,m,n).
    # k-cohort PSUMs on-chip, no HBM write for partials.
    # Empirically: broadcast within ~8 cores is cheap; beyond that contention
    # grows. Penalize hbm by (max(m, n) / cohort_limit) when cohort > limit.
    lhs_bytes = B * M * K * hw.dtype_bytes
    rhs_bytes = B * K * N * hw.dtype_bytes
    out_bytes = B * M * N * hw.dtype_bytes
    cohort = max(m, n)
    cohort_penalty = max(1.0, cohort / hw.cohort_broadcast_limit)
    hbm_us = (lhs_bytes + rhs_bytes + out_bytes) / (hw.hbm_bw_gbs * 1000)
    hbm_us *= cohort_penalty

    # ---- PSUM ring ----
    # Each k-step adds a ring-hop PSUM reduction over the *output elements*
    # (M x N per (b, m, n) tile). Fit from MoE data: k=2 added ~3.4ms over 8.4M
    # output elems with 1 hop -> ~4e-4 us per elem per hop. Assumes ring-adjacent
    # k_fast emission (1 hop per k-step); without it, hops scale as m * n.
    output_elems = (B // b) * (M // m) * (N // n) * b * m * n  # i.e. B*M*N total
    psum_us = max(0, k - 1) * output_elems * hw.psum_per_elem_us

    # ---- Batch-split penalty ----
    # Empirically (measured on bmm shapes): b_split > 1 regresses kernel time
    # multiplicatively -- roughly 1.2x per b-step on large-K bmm, less on small-K.
    # The default planner iterates batch within cores; we encode that as the
    # preferred mode here.
    batch_penalty = 1.0 + hw.batch_split_penalty * max(0, b - 1)

    # ---- Shape-aware m_split target (empirical) ----
    # The PT-array sweet spot for m_split depends on M and LX budget. Device data
    # (QO: m=8 vs m=4 is 6% faster; MoE: m=4 vs m=2 is 75% faster) shows the rule
    # ``target_m = clamp(M_MIN, max_cores//2, M / (TARGET_PT_PASSES * PT_ROWS))``
    # tracks the optimum. Penalize log2-distance from that target.
    target_m = max(
        hw.m_split_min,
        min(hw.max_cores // 2, max(1, M // (hw.target_pt_passes * hw.pt_rows))),
    )
    m_distance = abs(math.log2(max(1, m) / target_m))
    target_m_us = m_distance * hw.target_m_penalty_us

    return (compute_us + hbm_us + psum_us + target_m_us) * batch_penalty


# =============================================================================
# Feasible-split enumeration
# =============================================================================
def enumerate_splits(B, M, N, K, max_cores):
    """Yield (b, m, n, k) where each divides its dim and b*m*n*k <= max_cores.

    N and K should be in STICKS (planner's iteration space treats sticks as
    atomic units). The caller does the elem -> stick conversion.
    """
    b_divs = [int(d) for d in divisors(max(1, B))]
    m_divs = [int(d) for d in divisors(max(1, M))]
    n_divs = [int(d) for d in divisors(max(1, N))]
    k_divs = [int(d) for d in divisors(max(1, K))]
    for b in b_divs:
        for mm in m_divs:
            for nn in n_divs:
                for kk in k_divs:
                    if b * mm * nn * kk <= max_cores:
                        yield (b, mm, nn, kk)


# =============================================================================
# Validation set
# Each entry: name, (B, M, K, N) in elements, expected (b, m, n, k) split,
# kernel_ms measured (None if not available). N and K are converted to sticks
# (64 elems each) for the planner's view.
# =============================================================================
ELEMS_PER_STICK = 64

@dataclasses.dataclass
class Shape:
    name: str
    B: int
    M: int
    K: int          # elements
    N: int          # elements
    expected: tuple[int, int, int, int]   # (b, m, n, k) split, n & k in sticks-cohort
    measured_us: float | None = None

VALIDATED = [
    Shape("QO bs=1",   1,  512,  4096,  4096, (1, 8, 4, 1), measured_us=326),
    Shape("KV bs=1",   1,  512,  4096,  1024, (1, 8, 4, 1), measured_us=114),
    Shape("MLP bs=1",  1,  512,  4096, 12800, (1, 8, 4, 1), measured_us=1453),
    Shape("MoE gate/up", 8,  128, 2048,  8192, (1, 4, 8, 1), measured_us=2722),
    Shape("MoE down",    8,  128, 8192,  2048, (1, 4, 8, 1), measured_us=1986),
    Shape("bmm large-K", 8,  512, 4096,  512,  (1, 8, 4, 1), measured_us=827),
    Shape("bmm small-K", 8,  512,   64,  512,  (1, 8, 4, 1), measured_us=83),
]


# =============================================================================
# Main: argmin cost vs expected
# =============================================================================
def pick(shape: Shape, hw: Hw, top_k: int = 5):
    """Return the (cost, split) chosen plus the top-k candidates.

    The planner splits N and K in *sticks* (so divisibility uses stick counts),
    but the cost function reasons in *elements* (MACs and bytes).
    """
    n_sticks = shape.N // ELEMS_PER_STICK
    k_sticks = shape.K // ELEMS_PER_STICK
    feas = []
    for b, m, n, k in enumerate_splits(shape.B, shape.M, n_sticks, k_sticks, hw.max_cores):
        c = split_cost(shape.B, shape.M, shape.K, shape.N, b, m, n, k, hw)
        if math.isfinite(c):
            feas.append((c, (b, m, n, k)))
    feas.sort()
    return feas[:top_k]


def main():
    hw = Hw(lx_frac=0.8)
    print(f"hw: max_cores={hw.max_cores} pt_rows={hw.pt_rows} "
          f"hbm={hw.hbm_bw_gbs}GB/s lx_per_core={hw.lx_per_core//1024}KB "
          f"peak={hw.peak_macs_us_core:.2g}MACs/us/core lx_frac={hw.lx_frac}")
    print()
    print(f"{'shape':<22} {'expected':>20} {'chosen':>20} "
          f"{'cost_us':>10} {'measured_us':>12}  {'rank':>6}")
    print("-" * 100)
    n_top1 = n_tied = n_top5 = 0
    TOP_N = 5
    for s in VALIDATED:
        top = pick(s, hw, top_k=TOP_N)
        chosen_cost, chosen = top[0]
        # Find expected's rank in the top-N (1-indexed); None if not in top-N.
        rank = next((i + 1 for i, (_, sp) in enumerate(top) if sp == s.expected), None)
        # Tied-at-top: expected has the same cost as the chosen (just lost tiebreak)
        tied_at_top = rank is not None and any(
            sp == s.expected and abs(c - chosen_cost) < 0.01 for c, sp in top
        )
        if chosen == s.expected:
            tag, n_top1 = "TOP-1", n_top1 + 1
        elif tied_at_top:
            tag, n_tied = "TIED", n_tied + 1
        elif rank is not None:
            tag, n_top5 = f"#{rank}", n_top5 + 1
        else:
            tag = "MISS"
        print(f"{s.name:<22} {str(s.expected):>20} {str(chosen):>20} "
              f"{chosen_cost:>10.1f} {str(s.measured_us):>12}  {tag:>6}")
        if rank is None or rank > 1:
            for c, sp in top:
                t = "  <-expected" if sp == s.expected else ""
                print(f"     {str(sp):>20}  cost={c:>8.1f}{t}")
    print()
    print(f"TOP-1: {n_top1}/{len(VALIDATED)}  TIED (cost ≈ top): {n_tied}/{len(VALIDATED)}  "
          f"in TOP-{TOP_N}: {n_top1 + n_tied + n_top5}/{len(VALIDATED)}")


if __name__ == "__main__":
    main()
