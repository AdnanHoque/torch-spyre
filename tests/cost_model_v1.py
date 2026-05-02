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

"""Cost-model v1 for Spyre matmul planner — Phase 1.1.

Phase 1.0 found an average 10.3% / max 38.5% gap between the default
planner's `(m, n, k)` factorization and the empirical-best across 13
production shapes. This module is the first-cut empirical cost model
to predict per-split wall time, with the eventual goal of replacing
the greedy-by-priority planner with a cost-model-driven one.

Design principles for v1:

- **Empirically-derived terms only.** Each constant in the model is
  calibrated against measurements from our Phase 0 + Phase 1.0
  diagnostics. We don't import formulas from external papers (TL,
  CUTLASS, etc.) — we observe what Spyre actually does and fit terms
  to that.
- **Pure-function model.** No Spyre runtime dependency; can be called
  from tests, validation scripts, or eventually the planner.
- **Simple first, refine on data.** We start with the smallest model
  that could plausibly predict wall-time, validate it, then add
  complexity only where validation shows the simple model fails.

The v1 model has three components:

  T_predicted(M, N, K, m, n, k) = max(
      T_launch_floor,                  # constant ~3 ms (Phase 0b)
      max(T_compute, T_dma)            # whichever bottleneck dominates
  )

T_compute  = per-core compute work / effective per-core throughput
T_dma      = per-core DDR traffic / effective DDR bandwidth

The "max(launch_floor, max(...))" reflects two empirical facts:

1. Phase 0b: wall time has a ~3 ms floor regardless of work size.
2. Phase 0a: at large shapes, wall time scales with the slower of
   compute or DDR — pipelined load/compute/store overlap appears to
   happen on Spyre (else the floor would be `launch + load + compute
   + store`, not max).

Whether (2) is correct is part of what Phase 1.2 (predict-vs-measure
validation) will tell us.

CALIBRATION STATUS: constants below are initial guesses from earlier
phases. Phase 1.2 will refine them by fitting against measured data.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---- Calibrated constants ---------------------------------------------------
# All of these will be refined in Phase 1.2 by least-squares fit against the
# Phase 1.0 measurement set. Initial values are from earlier phases.

# Per-launch fixed overhead. Measured in Phase 0b across 3 shape regimes
# (130K to 1G FLOPs/call): wall time was ~3 ms regardless of compute size,
# meaning compute below this threshold is invisible. Treat as a hard floor.
LAUNCH_FLOOR_MS: float = 3.0

# Effective per-core fp16 compute throughput. Spyre's published peak is
# ~150 TFLOPs/s aggregate (so ~4.7 TFLOPs/s per core if perfectly utilized).
# Empirically we never hit anywhere near peak — Phase 0a SDPA showed 0.18
# TFLOPs/s aggregate, Phase 1 SplitK bench showed ~9 TFLOPs/s aggregate at
# best on prefill matmul. We start with a low estimate and let calibration
# adjust.
PER_CORE_TFLOPS: float = 0.5  # initial guess; will be calibrated

# Effective DDR bandwidth absorbed by cross-core sharing. DDR-traffic Phase 0
# observed eff BW from 77 to 671 GB/s under different splits at the same
# shape. The "effective" bandwidth a single core perceives depends on how
# much the split shares operands across cores. v1 uses a single average
# value; v2 may model split-dependent sharing factors.
EFFECTIVE_DDR_BW_GBS: float = 200.0  # initial — refine via calibration

NUM_CORES_DEFAULT: int = 32
DTYPE_BYTES_DEFAULT: int = 2  # fp16


@dataclass
class CostBreakdown:
    """Per-component costs in milliseconds. Useful for diagnosing where
    a prediction's cost comes from."""
    t_launch: float
    t_compute: float
    t_load: float
    t_store: float
    t_total: float

    def __repr__(self) -> str:
        return (
            f"CostBreakdown(launch={self.t_launch:.2f} ms, "
            f"compute={self.t_compute:.2f}, "
            f"load={self.t_load:.2f}, store={self.t_store:.2f}, "
            f"total={self.t_total:.2f})"
        )


def per_core_compute_flops(M: int, N: int, K: int, m: int, n: int, k: int) -> int:
    """Per-core matmul work, in FLOPs. fp16 mul+add counted as 2 ops."""
    M_per = M // m
    N_per = N // n
    K_per = K // k
    return 2 * M_per * N_per * K_per


def cross_core_traffic_bytes(
    M: int, N: int, K: int, m: int, n: int, k: int,
    dtype_bytes: int = DTYPE_BYTES_DEFAULT,
) -> tuple[int, int, int]:
    """Total bytes loaded/stored across all cores, NAIVE accounting (no
    sharing). DDR-traffic Phase 0 derived these formulas:

        A_load  = n × |A|   (each N-band's cores collectively read full A)
        B_load  = m × |B|   (each M-band's cores collectively read full B)
        C_store = k × |C|   (k partial outputs per element when k > 1)

    Spyre's cross-core sharing absorbs SOME of this redundancy in practice
    (eff BW > peak on (32, 1, 1) splits). v1 ignores the sharing — it
    treats every byte in this aggregate as a real DDR transit. v2 may
    introduce a split-dependent sharing factor.

    Returns (A_load, B_load, C_store) in bytes.
    """
    A = M * K * dtype_bytes
    B = K * N * dtype_bytes
    C = M * N * dtype_bytes
    return n * A, m * B, k * C


def predict_wall_ms(
    M: int, N: int, K: int, m: int, n: int, k: int,
    dtype_bytes: int = DTYPE_BYTES_DEFAULT,
    num_cores: int = NUM_CORES_DEFAULT,
    *,
    return_breakdown: bool = False,
) -> float | CostBreakdown:
    """Predict wall time in milliseconds for matmul `(M, N, K)` with split
    `(m, n, k)` on Spyre.

    Args:
        M, N, K: matmul dimensions (output rows × output cols × reduction).
        m, n, k: split factors. m·n·k should equal num_cores for a well-
            saturated kernel; we don't enforce this — the caller is
            responsible.
        return_breakdown: if True, return a CostBreakdown rather than scalar.
    """
    # Per-core compute time in ms.
    flops = per_core_compute_flops(M, N, K, m, n, k)
    t_compute = flops / (PER_CORE_TFLOPS * 1e12) * 1e3

    # Per-core DDR-transit time in ms. Total cross-core traffic divided by
    # num_cores gives the average per-core figure, then divided by per-core
    # effective bandwidth. v1 model assumes uniform bandwidth across cores.
    a_load, b_load, c_store = cross_core_traffic_bytes(M, N, K, m, n, k, dtype_bytes)
    per_core_load_bytes = (a_load + b_load) / num_cores
    per_core_store_bytes = c_store / num_cores
    per_core_bw = (EFFECTIVE_DDR_BW_GBS * 1e9) / num_cores  # bytes/sec/core
    t_load = per_core_load_bytes / per_core_bw * 1e3
    t_store = per_core_store_bytes / per_core_bw * 1e3

    # Combination rule v1: assume pipelined — load/compute/store overlap
    # so the bottleneck is the slowest of (load+store, compute). Floor by
    # launch overhead.
    t_kernel_work = max(t_compute, t_load + t_store)
    t_total = max(LAUNCH_FLOOR_MS, t_kernel_work)

    if return_breakdown:
        return CostBreakdown(
            t_launch=LAUNCH_FLOOR_MS,
            t_compute=t_compute,
            t_load=t_load,
            t_store=t_store,
            t_total=t_total,
        )
    return t_total


# ---- Validity checks (catch infeasible splits) ------------------------------

STICK_ELEMS_FP16 = 64


def is_feasible_split(
    M: int, N: int, K: int, m: int, n: int, k: int,
    *,
    stick_elems: int = STICK_ELEMS_FP16,
) -> tuple[bool, str]:
    """Stick-alignment + divisibility constraints. Same checks used in
    `tests/diag_split_gap.py`. Returns (feasible, reason_if_not).

    Backend-imposed constraints (per-core span limit, EAR overflow on big-
    K factorizations) are NOT modeled in v1 — they manifest as compile-
    time errors. Phase 1.2 may add empirical infeasibility prediction by
    learning from observed errors.
    """
    if M % m != 0:
        return False, f"M={M} not divisible by m={m}"
    if (N // n) < stick_elems or (N // n) % stick_elems != 0:
        return False, f"N/n={N // n} not stick-aligned (>= {stick_elems})"
    if (K // k) < stick_elems or (K // k) % stick_elems != 0:
        return False, f"K/k={K // k} not stick-aligned (>= {stick_elems})"
    return True, ""


def best_split_by_model(
    M: int, N: int, K: int,
    *,
    num_cores: int = NUM_CORES_DEFAULT,
    dtype_bytes: int = DTYPE_BYTES_DEFAULT,
) -> tuple[tuple[int, int, int], float, list[tuple[tuple[int, int, int], float]]]:
    """Among all `(m, n, k)` factorizations of `num_cores` satisfying
    `is_feasible_split`, return the one with minimum predicted wall time.

    Returns:
        (best_split, best_predicted_ms, ranked_list)
    where ranked_list is all feasible splits sorted by predicted wall time
    (ascending).
    """
    candidates: list[tuple[tuple[int, int, int], float]] = []
    for mm in range(1, num_cores + 1):
        if num_cores % mm != 0:
            continue
        rem = num_cores // mm
        for nn in range(1, rem + 1):
            if rem % nn != 0:
                continue
            kk = rem // nn
            ok, _ = is_feasible_split(M, N, K, mm, nn, kk)
            if not ok:
                continue
            t_pred = predict_wall_ms(M, N, K, mm, nn, kk,
                                    dtype_bytes=dtype_bytes,
                                    num_cores=num_cores)
            candidates.append(((mm, nn, kk), t_pred))
    candidates.sort(key=lambda x: x[1])
    if not candidates:
        raise ValueError(f"No feasible factorization of {num_cores} for shape "
                         f"({M}, {N}, {K}) at fp16 stick alignment.")
    return candidates[0][0], candidates[0][1], candidates


__all__ = [
    "LAUNCH_FLOOR_MS",
    "PER_CORE_TFLOPS",
    "EFFECTIVE_DDR_BW_GBS",
    "NUM_CORES_DEFAULT",
    "DTYPE_BYTES_DEFAULT",
    "CostBreakdown",
    "per_core_compute_flops",
    "cross_core_traffic_bytes",
    "predict_wall_ms",
    "is_feasible_split",
    "best_split_by_model",
]
