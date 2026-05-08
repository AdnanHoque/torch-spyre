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

"""Per-op HMI cost model — Phase 0 of Project B.

Predicts wall time for a single matmul op given its shape, planner
split, dtype, and emission mode. Calibrated against measurements in
the diag-core-ordering branch; see diag_hmi_cost_model_calibrate.py
for the validation harness.

Model structure:

    t_compute  = per_core_macs / (PT_PEAK * achieved_frac * pt_util(M_per, N_per))
    t_hmi      = hmi_bytes(shape, split) / HMI_BW
    t_psum     = chain_hops × payload / SFP_BW              (only if k > 1)
    wall       = max(LAUNCH_FLOOR, max(t_compute, t_hmi) + t_psum)

The compute and HMI terms are assumed to overlap (kernel-template
prefetch); PSUM is on the critical path after compute. Launch floor
is a hard minimum.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---- hardware constants (32-core AIU, fp16 measurements) --------------

LAUNCH_FLOOR_MS = 3.0       # per-call overhead floor
# Effective HMI BW measured under pure-M with broadcast B accounting
# (diag_hmi_bw_pure_m.py): wall ≈ LF + bytes/40 GB/s for B in 128–272 MB.
# 67 GB/s is the spec headline; achieved under matmul kernel templates
# with cross-core ring sharing is ~40 GB/s.
HMI_BW_GBS = 40.0
SFP_BW_GBS = 32.0           # dedicated PSUM ring
PT_PEAK_TFLOPS_PER_CORE = 1.0
ACHIEVED_FRAC = 1.0         # achieved-fraction at peak PT utilisation
PT_ROWS = 8                 # vertical PT array dim — feeds M
PT_COLS_SIMD = 64           # horizontal PT × SIMD — feeds N


# ---- helpers ----------------------------------------------------------

_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "int8": 1}


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES[dtype]


def _pt_util(m_per: int, n_per: int) -> float:
    """PT-array fill fraction.

    The PT array is 8 rows × 8 cols × 8-way SIMD. M-per-core feeds
    rows; N-per-core feeds cols×SIMD. When per-core dim is below the
    array's natural width, only some lanes do useful work.
    """
    row = min(1.0, m_per / PT_ROWS)
    col = min(1.0, n_per / PT_COLS_SIMD)
    return row * col


def _hmi_bytes(M: int, N: int, K: int, split: tuple[int, int, int],
               dtype: str) -> int:
    """First-order HMI byte count for one matmul invocation.

    Under K-split (k > 1), each K-cluster fetches only its K-slice
    of A and B, so the per-cluster bytes formula

        (M·K + K·N) / k + M·N

    matches measurement (Project B Phase 0, Track 2 Phase 0). For
    k = 1 (pure-M or pure-N), this reduces to the broadcast form
    M·K + K·N + M·N.
    """
    db = _dtype_bytes(dtype)
    _, _, k = split
    return ((M * K + K * N) // k + M * N) * db


def _total_psum_ring_bytes(M: int, N: int, split: tuple[int, int, int],
                           k_fast: bool, psum_dtype: str = "fp32") -> int:
    """Total bytes traversing the SFP ring across ALL PSUM chains.

    There are m·n parallel chains (one per (m_slice, n_slice) cell).
    Each chain has k members and (k-1) sends. Each send traverses
    one ring hop in k_fast emission, m·n hops in the default
    (mixed-radix) emission.

    All chains share the SFP ring's bandwidth, so we count total ring
    bytes and divide by SFP_BW once.
    """
    m, n, k = split
    if k <= 1:
        return 0
    num_chains = m * n
    sends_per_chain = k - 1
    hops_per_send = 1 if k_fast else (m * n)
    per_chain_payload = (M // m) * (N // n) * _dtype_bytes(psum_dtype)
    return num_chains * sends_per_chain * hops_per_send * per_chain_payload


def _chain_hops(split: tuple[int, int, int], k_fast: bool) -> int:
    """Total ring positions traversed by one chain (for breakdown reporting)."""
    m, n, k = split
    if k <= 1:
        return 0
    return (k - 1) * (1 if k_fast else m * n)


# ---- LX overflow / streaming regime model -----------------------------
# Probe 3 (May 2026, DSv3 o_proj M=2048 (1, 8, 4)+kf): wall grows
# linearly with per-core PSUM accumulator overage past LX. Slope
# ~17 ms per overage factor (overage = c_psum / LX_BYTES_PER_CORE).
# Probe 6 (May 2026, three shapes): within the n=1 streaming-output
# fast path, three regimes by chain length. The chain=4 → chain=8
# boundary is universal across shapes.
#
# Calibration constants are a single-shape fit and need broader
# coverage for production use; the *form* (regime-routed, n=1 vs n>1
# distinguished) is empirically robust.

LX_BYTES_PER_CORE = 2 * 1024 * 1024
PSUM_OVERFLOW_MS_PER_FACTOR = 17.0          # Probe 3 calibration

# n=1 streaming-output regime additive costs (Probe 6 calibration)
N1_PIPELINE_REGIME_COST_MS = 3.0     # chain ≤ 4
N1_ALLREDUCE_REGIME_COST_MS = 14.0   # chain == 32


def _psum_dtype_bytes() -> int:
    return _dtype_bytes("fp32")


def _c_psum_per_core(M: int, N: int, split: tuple[int, int, int]) -> int:
    """PSUM accumulator residency per core in bytes (fp32)."""
    m, n, _ = split
    return (M // m) * (N // n) * _psum_dtype_bytes()


def _n1_sync_regime_cost_ms(M: int, N: int, split: tuple[int, int, int]) -> float:
    """Sync-regime cost for n=1 streaming path with 4 < k < 32.

    Probe 6 calibration on three shapes: cost ≈ 1.5 × payload_MB + 5,
    where payload_MB = M_per × N × dtype_psum_bytes / 1MB. This is a
    rough fit (3 shapes × 2 chain lengths each) and likely understates
    cost on shapes with bigger payload-per-head — needs more data.
    """
    m, _, _ = split
    payload_mb = (M // m) * N * _psum_dtype_bytes() / (1024 * 1024)
    return 1.5 * payload_mb + 5.0


def _psum_regime_cost_ms(
    M: int, N: int, K: int,
    split: tuple[int, int, int],
    k_fast: bool,
    psum_pipe_ms: float,
) -> float:
    """Total PSUM-related cost (ms) under the regime-routed model.

    This is the single source for what was historically just
    `t_psum_ms`. It returns the additive PSUM cost on the wall
    critical path: the SFP ring traversal time plus, separately, the
    LX-overflow / streaming-regime additive penalties measured by
    Probes 3-6.
    """
    m, n, k = split
    if k <= 1:
        return 0.0

    if n == 1:
        # n=1 streaming-output fast path (Probe 4-6).
        # PSUM ring time is small under kf; the dominant cost is
        # the regime-specific overhead identified in Probe 6.
        if k <= 4:
            return psum_pipe_ms + N1_PIPELINE_REGIME_COST_MS
        if k == 32:
            return psum_pipe_ms + N1_ALLREDUCE_REGIME_COST_MS
        # 4 < k < 32 — sync regime
        return psum_pipe_ms + _n1_sync_regime_cost_ms(M, N, split)

    # n > 1: the ring-traversal pipe term, plus a PSUM-overflow penalty
    # when the per-core accumulator exceeds LX *and* k_fast emission is
    # in use. The 17 ms/factor calibration is from Probe 3, which ran
    # at (1, 8, 4)+kf — the catastrophic A-re-fetch-per-N-tile mechanism
    # there is k_fast-conditional. Under identity emission with the same
    # overage, observed wall is K-dependent in a way the current model
    # doesn't capture; we don't apply the penalty there to avoid
    # over-predicting on small-K +id rows.
    c_psum = _c_psum_per_core(M, N, split)
    if k_fast and c_psum > LX_BYTES_PER_CORE:
        overage_factor = c_psum / LX_BYTES_PER_CORE
        return psum_pipe_ms + PSUM_OVERFLOW_MS_PER_FACTOR * (overage_factor - 1.0)
    return psum_pipe_ms


# ---- main API ---------------------------------------------------------

@dataclass
class CostBreakdown:
    """Per-component cost (ms) breakdown for one matmul op."""

    t_compute_ms: float
    t_hmi_ms: float
    t_psum_ms: float
    t_launch_floor_ms: float
    t_wall_ms: float
    pt_util: float
    hmi_bytes: int
    chain_hops: int


def predict(
    shape: tuple[int, int, int],
    split: tuple[int, int, int],
    dtype: str = "fp16",
    *,
    k_fast: bool = False,
    hmi_bw_gbs: float = HMI_BW_GBS,
    achieved_frac: float = ACHIEVED_FRAC,
    launch_floor_ms: float = LAUNCH_FLOOR_MS,
) -> CostBreakdown:
    """Predict wall time (ms) for one matmul invocation."""
    M, N, K = shape
    m, n, k = split
    if m * n * k != 32:
        raise ValueError(f"split {split} must multiply to 32 cores")

    M_per = M // m
    N_per = N // n
    K_per = K // k

    # Compute
    macs = M_per * N_per * K_per
    flops = 2 * macs
    util = _pt_util(M_per, N_per)
    if util > 0:
        peak_flops_per_s = PT_PEAK_TFLOPS_PER_CORE * 1e12 * achieved_frac * util
        t_compute_s = flops / peak_flops_per_s
    else:
        t_compute_s = float("inf")
    t_compute_ms = t_compute_s * 1e3

    # HMI
    hmi_bytes = _hmi_bytes(M, N, K, split, dtype)
    t_hmi_ms = hmi_bytes / (hmi_bw_gbs * 1e9) * 1e3

    # PSUM — regime-routed (see _psum_regime_cost_ms).
    chain_hops = _chain_hops(split, k_fast=k_fast)
    psum_bytes = _total_psum_ring_bytes(M, N, split, k_fast=k_fast)
    psum_pipe_ms = psum_bytes / (SFP_BW_GBS * 1e9) * 1e3
    t_psum_ms = _psum_regime_cost_ms(
        M, N, K, split, k_fast=k_fast, psum_pipe_ms=psum_pipe_ms,
    )

    # Launch floor stacks on top of HMI but overlaps with compute.
    # Probe diag_hmi_bw_pure_m.py: HMI-bound pure-M wall ≈ LF + bytes/BW
    # exactly — LF and HMI are serial (LF likely IS HMI activity for
    # binary/descriptor fetch). Compute, by contrast, runs concurrently
    # with both LF and HMI: compute-bound shapes measure compute alone,
    # not LF + compute.
    t_wall_ms = max(t_compute_ms, t_hmi_ms + launch_floor_ms) + t_psum_ms

    return CostBreakdown(
        t_compute_ms=t_compute_ms,
        t_hmi_ms=t_hmi_ms,
        t_psum_ms=t_psum_ms,
        t_launch_floor_ms=launch_floor_ms,
        t_wall_ms=t_wall_ms,
        pt_util=util,
        hmi_bytes=hmi_bytes,
        chain_hops=chain_hops,
    )


def is_compute_bound(breakdown: CostBreakdown) -> bool:
    return breakdown.t_compute_ms > breakdown.t_hmi_ms


def is_launch_floor_bound(breakdown: CostBreakdown) -> bool:
    """Launch-floor-bound when the LF term exceeds the work term.

    With the additive wall formula (wall = LF + max(compute, hmi) + psum),
    a shape is LF-bound when the work portion is small relative to LF —
    we use 50% as the cutoff for classification purposes only.
    """
    work = max(breakdown.t_compute_ms, breakdown.t_hmi_ms) + breakdown.t_psum_ms
    return work < breakdown.t_launch_floor_ms * 0.5


def label(breakdown: CostBreakdown) -> str:
    if is_launch_floor_bound(breakdown):
        return "launch-floor-bound"
    if is_compute_bound(breakdown):
        return "compute-bound"
    return "HMI-bound"
