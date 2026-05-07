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

"""Track 2 Phase 1 — three-fix layered cost-model evaluation.

Phase 0 of Track 2 identified three independent residual mechanisms
in the Phase 0 cost model that per-cluster HMI bytes alone doesn't
fix:

  1. PSUM term over-predicts on (1, n, k>1) + identity emission
     because it treats the SFP ring as a single 32 GB/s pipe instead
     of 32 parallel links.
  2. HMI BW under-predicts at small M on wide-K wide-N pure-M
     shapes (DSv3 o_proj M=32: implied BW 128 GB/s).
  3. Catastrophic LX overflow on shapes where A_per_core ≫ 2 MB
     (DSv3 o_proj M=2048 +id at A_per = 32 MB).

This script lays four progressively-layered cost-model variants
against the 30-row validation set and reports per-row residuals so
we can see which fix moves which row:

  V0 = baseline (current hmi_cost_model.predict)
  V1 = V0 + per-cluster bytes for K-split  (Phase 0's fix)
  V2 = V1 + PSUM aggregate-link model
  V3 = V2 + LX-overflow re-fetch penalty

Mechanism (1), the PSUM aggregate-link model, replaces

    t_psum = total_bytes / link_BW

with

    t_psum = max(per_chain_latency, total_bytes / (ring_size × link_BW))

where per_chain_latency = sends × hops × payload / link_BW. This
captures both the latency-bound regime (one slow chain limits the
others) and the throughput-bound regime (chains saturate disjoint
ring links in parallel).

Mechanism (3), LX overflow penalty, multiplies HMI bytes by
overage_factor = max(1.0, A_per_core / LX_BYTES). Crude but matches
the "operand re-fetch per N-tile" mechanism Project B Phase 0
documented.

Mechanism (2) (small-M HMI BW) is NOT addressed here. It would
require either a regime-switched BW model or a calibration sweep on
hardware. Reported separately in V3 residuals as the leftover.

Usage:
    python tests/diag_kfast_residual_phase1.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import (  # noqa: E402
    ACHIEVED_FRAC,
    HMI_BW_GBS,
    LAUNCH_FLOOR_MS,
    PT_PEAK_TFLOPS_PER_CORE,
    SFP_BW_GBS,
    _pt_util,
)
from tests.lx_fit import LX_BYTES_PER_CORE  # noqa: E402
from tests.diag_hmi_cost_model_calibrate import VALIDATION  # noqa: E402


_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "int8": 1}
RING_SIZE = 32  # AIU 1.0 has 32 active cores in a 1D SFP ring


# ---- HMI byte models ------------------------------------------------

def _hmi_bytes_broadcast(M, N, K, split, dtype):
    return (M * K + K * N + M * N) * _DTYPE_BYTES[dtype]


def _hmi_bytes_per_cluster(M, N, K, split, dtype):
    _, _, k = split
    return ((M * K + K * N) // k + M * N) * _DTYPE_BYTES[dtype]


# ---- PSUM models ----------------------------------------------------

def _psum_pipe(M, N, split, k_fast):
    """Current model: total bytes / single SFP-ring pipe BW."""
    m, n, k = split
    if k <= 1:
        return 0.0
    num_chains = m * n
    sends = k - 1
    hops = 1 if k_fast else (m * n)
    payload = (M // m) * (N // n) * _DTYPE_BYTES["fp32"]
    total = num_chains * sends * hops * payload
    return total / (SFP_BW_GBS * 1e9) * 1e3


def _psum_aggregate(M, N, split, k_fast):
    """Aggregate-link model.

    Time = max(per-chain latency, total bytes / aggregate ring BW)

    per-chain latency = sends * hops * payload / link_BW
    aggregate BW = ring_size * link_BW
    """
    m, n, k = split
    if k <= 1:
        return 0.0
    num_chains = m * n
    sends = k - 1
    hops = 1 if k_fast else (m * n)
    payload = (M // m) * (N // n) * _DTYPE_BYTES["fp32"]
    per_chain_bytes = sends * hops * payload
    per_chain_latency_ms = per_chain_bytes / (SFP_BW_GBS * 1e9) * 1e3
    total_bytes = num_chains * per_chain_bytes
    aggregate_throughput_ms = total_bytes / (
        RING_SIZE * SFP_BW_GBS * 1e9) * 1e3
    return max(per_chain_latency_ms, aggregate_throughput_ms)


# ---- LX-overflow penalty -------------------------------------------

def _lx_overage_factor(M, N, K, split, dtype):
    m, _, k = split
    M_per = M // m
    K_per = K // k
    a_bytes = M_per * K_per * _DTYPE_BYTES[dtype]
    return max(1.0, a_bytes / LX_BYTES_PER_CORE)


# ---- variant predict -----------------------------------------------

@dataclass
class Pred:
    t_compute_ms: float
    t_hmi_ms: float
    t_psum_ms: float
    t_wall_ms: float


def _predict(shape, split, dtype, k_fast,
             *, bytes_fn, psum_fn, lx_penalty: bool) -> Pred:
    M, N, K = shape
    m, n, k = split
    M_per = M // m
    N_per = N // n
    K_per = K // k

    macs = M_per * N_per * K_per
    flops = 2 * macs
    util = _pt_util(M_per, N_per)
    if util > 0:
        peak = PT_PEAK_TFLOPS_PER_CORE * 1e12 * ACHIEVED_FRAC * util
        t_compute = flops / peak * 1e3
    else:
        t_compute = float("inf")

    hmi_bytes = bytes_fn(M, N, K, split, dtype)
    if lx_penalty:
        hmi_bytes = int(hmi_bytes * _lx_overage_factor(
            M, N, K, split, dtype))
    t_hmi = hmi_bytes / (HMI_BW_GBS * 1e9) * 1e3

    t_psum = psum_fn(M, N, split, k_fast)

    t_wall = max(t_compute, t_hmi + LAUNCH_FLOOR_MS) + t_psum
    return Pred(t_compute, t_hmi, t_psum, t_wall)


# ---- variant configs -----------------------------------------------

VARIANTS = [
    ("V0 baseline",
     dict(bytes_fn=_hmi_bytes_broadcast, psum_fn=_psum_pipe,
          lx_penalty=False)),
    ("V1 +per-cluster",
     dict(bytes_fn=_hmi_bytes_per_cluster, psum_fn=_psum_pipe,
          lx_penalty=False)),
    ("V2 +psum-agg",
     dict(bytes_fn=_hmi_bytes_per_cluster, psum_fn=_psum_aggregate,
          lx_penalty=False)),
    ("V3 +lx-penalty",
     dict(bytes_fn=_hmi_bytes_per_cluster, psum_fn=_psum_aggregate,
          lx_penalty=True)),
]


def _split_class(split, mode):
    m, n, k = split
    if k == 1 and m == 32:
        return "pure-M"
    if k > 1 and mode == "kf":
        return "K-split+kf"
    if k > 1:
        return "K-split+id"
    return "other"


# ---- main ----------------------------------------------------------

def main() -> int:
    print("# Track 2 Phase 1 — layered cost-model variants\n")
    print("V0 = baseline (broadcast bytes, pipe PSUM)")
    print("V1 = V0 + per-cluster bytes (Track 2 Phase 0 fix)")
    print("V2 = V1 + PSUM aggregate-link model")
    print("V3 = V2 + LX overflow re-fetch penalty\n")

    print("## Section A — Per-row residuals across V0..V3\n")
    print("| row | class | meas | V0 err | V1 err | V2 err | V3 err |")
    print("|---|---|---:|---:|---:|---:|---:|")

    rows_results = []
    for row_label, M, N, K, split, mode, measured in VALIDATION:
        kfst = (mode == "kf")
        klass = _split_class(split, mode)
        errs = {}
        preds = {}
        for name, kw in VARIANTS:
            p = _predict((M, N, K), split, "fp16", kfst, **kw)
            preds[name] = p
            errs[name] = (p.t_wall_ms - measured) / measured * 100
        print(f"| {row_label} | {klass} | {measured:.2f} | "
              f"{errs['V0 baseline']:+.1f}% | "
              f"{errs['V1 +per-cluster']:+.1f}% | "
              f"{errs['V2 +psum-agg']:+.1f}% | "
              f"{errs['V3 +lx-penalty']:+.1f}% |")
        rows_results.append(dict(
            label=row_label, klass=klass, measured=measured,
            split=split, mode=mode, shape=(M, N, K),
            errs=errs, preds=preds,
        ))

    print()
    print("## Section B — Aggregate fit (mean |error|)\n")
    print("| class | n | V0 | V1 | V2 | V3 |")
    print("|---|---:|---:|---:|---:|---:|")
    by_class = {}
    for r in rows_results:
        by_class.setdefault(r["klass"], []).append(r)
    for klass, rs in sorted(by_class.items()):
        means = []
        for vname, _ in VARIANTS:
            absent = [abs(r["errs"][vname]) for r in rs]
            means.append(statistics.mean(absent))
        print(f"| {klass} | {len(rs)} | "
              + " | ".join(f"{m:.1f}%" for m in means) + " |")
    means = []
    overs = []
    for vname, _ in VARIANTS:
        absent = [abs(r["errs"][vname]) for r in rows_results]
        means.append(statistics.mean(absent))
        overs.append(sum(1 for e in absent if e > 10))
    print(f"| **all** | {len(rows_results)} | "
          + " | ".join(f"**{m:.1f}%**" for m in means) + " |")
    print()
    print("Rows with |error| > 10%:")
    for (vname, _), o in zip(VARIANTS, overs):
        print(f"  {vname}: {o}/{len(rows_results)}")
    print()

    print("## Section C — Residual after V3 (combined fix)\n")
    leftovers = [
        r for r in rows_results if abs(r["errs"]["V3 +lx-penalty"]) > 10
    ]
    leftovers.sort(key=lambda r: -abs(r["errs"]["V3 +lx-penalty"]))
    if not leftovers:
        print("  (none — V3 closes everything within ±10%)")
    else:
        print("Rows where the three-fix V3 still misses by >10%:\n")
        print("| row | class | shape | meas | V3 pred | err |")
        print("|---|---|---|---:|---:|---:|")
        for r in leftovers:
            v3p = r["preds"]["V3 +lx-penalty"]
            print(f"| {r['label']} | {r['klass']} | {r['shape']} | "
                  f"{r['measured']:.2f} | {v3p.t_wall_ms:.2f} | "
                  f"{r['errs']['V3 +lx-penalty']:+.1f}% |")
    print()

    # Verdict
    v0 = means[0]
    v3 = means[-1]
    print("## Verdict\n")
    print(f"  V0 (baseline) mean |error|: {v0:.1f}%")
    print(f"  V3 (all three fixes)       : {v3:.1f}%")
    if v3 < 10:
        print(f"  V3 closes the model to within 10% mean — phase complete.")
    elif v3 < 15:
        print(f"  V3 lifts mean fit substantially. Remaining outliers "
              "are {min(len(leftovers), 5)}-ish rows; categorise them "
              "to see if a fourth lever is needed.")
    else:
        print(f"  V3 is a partial fix. Mean still high — leftovers above "
              "name the next mechanism to investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
