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

"""Track 2 Phase 0 — k_fast / k-split residual investigation.

Phase 0 of LX residency planner exposed that the cost model
over-predicts wall time by up to 2× on small-M k_fast / k-split
rows that fit LX. Worst case: DSv3 o_proj M=128 (1, 16, 2) + kf,
predicted 9.14 ms vs. measured 4.69 ms.

Hypothesis (per Project B Phase 0):
  the cost model's HMI byte accounting uses the *full-broadcast*
  form `M·K + K·N + M·N`, which is correct only for pure-M splits
  where every core needs the entire B operand. Under K-split (k>1)
  with a sane kernel template, each K-cluster only fetches its own
  K_per chunk of A and B, so per-cluster HMI bytes scale as

      (M·K + K·N) / k  +  M·N

  Project B's `hmi_cost_model_phase0_findings.md` measured this form
  fits to 2% on LX-fitting rows; the LX-fitting-and-still-wrong rows
  in the LX-Phase-0 residual table are the smoking gun.

This script tests the hypothesis by computing predicted wall under
both byte models against the 30-row validation set, and comparing
residual quality.

Output structure:
  Section A — per-row residuals under (a) full-broadcast (current
              model) and (b) per-cluster bytes form
  Section B — partition by split type (pure-M / K-split-id /
              K-split-kf): which rows does per-cluster fix? which
              rows does it not?
  Section C — what residual remains after per-cluster fix —
              points to next investigation thread

Usage:
    python tests/diag_kfast_residual_phase0.py
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
    HMI_BW_GBS,
    LAUNCH_FLOOR_MS,
    SFP_BW_GBS,
    _chain_hops,
    _pt_util,
    _total_psum_ring_bytes,
    PT_PEAK_TFLOPS_PER_CORE,
    ACHIEVED_FRAC,
)
from tests.diag_hmi_cost_model_calibrate import VALIDATION  # noqa: E402


_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "int8": 1}


# ---- alt HMI byte models ---------------------------------------------

def _hmi_bytes_full_broadcast(M, N, K, split, dtype):
    """Current cost-model form: every core sees full B (broadcast).

    HMI bytes per call = (M·K + K·N + M·N) · dtype_bytes
    """
    return (M * K + K * N + M * N) * _DTYPE_BYTES[dtype]


def _hmi_bytes_per_cluster(M, N, K, split, dtype):
    """Per-cluster form: K-split shares K-chunks across only k cores.

    HMI bytes per call = ((M·K + K·N) / k + M·N) · dtype_bytes

    For pure-M (k=1) this is identical to full-broadcast.
    For (1, n, k>1) it scales A and B by 1/k while leaving C
    unchanged (output is still M·N regardless of split structure).
    """
    _, _, k = split
    return ((M * K + K * N) // k + M * N) * _DTYPE_BYTES[dtype]


# ---- predict-with-alt-bytes ------------------------------------------

@dataclass
class Pred:
    t_compute_ms: float
    t_hmi_ms: float
    t_psum_ms: float
    t_wall_ms: float
    hmi_bytes: int


def _predict_alt(shape, split, dtype, k_fast, bytes_fn) -> Pred:
    """Re-run the wall computation with a swapped HMI byte function."""
    M, N, K = shape
    m, n, k = split
    M_per = M // m
    N_per = N // n
    K_per = K // k

    # Compute term (unchanged from hmi_cost_model.predict)
    macs = M_per * N_per * K_per
    flops = 2 * macs
    util = _pt_util(M_per, N_per)
    if util > 0:
        peak_flops_per_s = PT_PEAK_TFLOPS_PER_CORE * 1e12 * ACHIEVED_FRAC * util
        t_compute_ms = flops / peak_flops_per_s * 1e3
    else:
        t_compute_ms = float("inf")

    # HMI term — swapped
    hmi_bytes = bytes_fn(M, N, K, split, dtype)
    t_hmi_ms = hmi_bytes / (HMI_BW_GBS * 1e9) * 1e3

    # PSUM term (unchanged)
    psum_bytes = _total_psum_ring_bytes(M, N, split, k_fast=k_fast)
    t_psum_ms = psum_bytes / (SFP_BW_GBS * 1e9) * 1e3

    t_wall_ms = max(t_compute_ms, t_hmi_ms + LAUNCH_FLOOR_MS) + t_psum_ms

    return Pred(t_compute_ms=t_compute_ms, t_hmi_ms=t_hmi_ms,
                t_psum_ms=t_psum_ms, t_wall_ms=t_wall_ms,
                hmi_bytes=hmi_bytes)


def _split_class(split: tuple[int, int, int], mode: str) -> str:
    m, n, k = split
    if k == 1 and m == 32:
        return "pure-M"
    if k > 1 and mode == "kf":
        return "K-split+kf"
    if k > 1:
        return "K-split+id"
    return "other"


# ---- main ------------------------------------------------------------

def main() -> int:
    print("# Track 2 Phase 0 — per-cluster bytes vs. full-broadcast\n")
    print("Hypothesis: cost-model over-predicts on K-split rows because")
    print("it uses M·K + K·N + M·N (full broadcast) instead of")
    print("(M·K + K·N)/k + M·N (per-cluster, K-split shares).\n")

    print("## Section A — Per-row residuals under both models\n")
    print("| row | split | mode | class | meas ms | "
          "broadcast pred | bcast err | percluster pred | pcl err |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|")

    rows_with_results = []
    for row_label, M, N, K, split, mode, measured in VALIDATION:
        kfst = (mode == "kf")
        pb = _predict_alt((M, N, K), split, "fp16", kfst,
                          _hmi_bytes_full_broadcast)
        pc = _predict_alt((M, N, K), split, "fp16", kfst,
                          _hmi_bytes_per_cluster)
        bcast_err = (pb.t_wall_ms - measured) / measured * 100
        pcl_err = (pc.t_wall_ms - measured) / measured * 100
        klass = _split_class(split, mode)
        split_str = f"({split[0]},{split[1]},{split[2]})"
        print(f"| {row_label} | {split_str} | {mode} | {klass} | "
              f"{measured:.2f} | {pb.t_wall_ms:.2f} | "
              f"{bcast_err:+.1f}% | {pc.t_wall_ms:.2f} | {pcl_err:+.1f}% |")
        rows_with_results.append(dict(
            label=row_label, klass=klass, measured=measured,
            bcast_err=bcast_err, pcl_err=pcl_err,
            bcast_pred=pb.t_wall_ms, pcl_pred=pc.t_wall_ms,
            split=split, mode=mode, shape=(M, N, K),
        ))

    print()
    print("## Section B — Aggregate fit by split class\n")
    print("| class | n | broadcast: mean \\|err\\| | pcl: mean \\|err\\| | "
          "broadcast: max | pcl: max | broadcast over 10% | pcl over 10% |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    by_class: dict[str, list] = {}
    for r in rows_with_results:
        by_class.setdefault(r["klass"], []).append(r)
    overall = []
    for klass, rs in sorted(by_class.items()):
        bcast_abs = [abs(r["bcast_err"]) for r in rs]
        pcl_abs = [abs(r["pcl_err"]) for r in rs]
        bcast_over = sum(1 for e in bcast_abs if e > 10)
        pcl_over = sum(1 for e in pcl_abs if e > 10)
        print(f"| {klass} | {len(rs)} | {statistics.mean(bcast_abs):.1f}% | "
              f"{statistics.mean(pcl_abs):.1f}% | "
              f"{max(bcast_abs):.1f}% | {max(pcl_abs):.1f}% | "
              f"{bcast_over}/{len(rs)} | {pcl_over}/{len(rs)} |")
        overall.extend(rs)
    bcast_abs = [abs(r["bcast_err"]) for r in overall]
    pcl_abs = [abs(r["pcl_err"]) for r in overall]
    bcast_over = sum(1 for e in bcast_abs if e > 10)
    pcl_over = sum(1 for e in pcl_abs if e > 10)
    print(f"| **all** | {len(overall)} | "
          f"**{statistics.mean(bcast_abs):.1f}%** | "
          f"**{statistics.mean(pcl_abs):.1f}%** | "
          f"{max(bcast_abs):.1f}% | {max(pcl_abs):.1f}% | "
          f"{bcast_over}/{len(overall)} | {pcl_over}/{len(overall)} |")
    print()

    print("## Section C — Residual after per-cluster fix\n")
    print("Rows with |error| > 10% under per-cluster model — these are\n"
          "the residuals that per-cluster bytes does NOT explain:\n")
    leftovers = [r for r in rows_with_results if abs(r["pcl_err"]) > 10]
    leftovers.sort(key=lambda r: -abs(r["pcl_err"]))
    if not leftovers:
        print("  (none — per-cluster fix closes everything)")
    else:
        print("| row | class | shape | meas ms | pcl pred | err |")
        print("|---|---|---|---:|---:|---:|")
        for r in leftovers:
            print(f"| {r['label']} | {r['klass']} | {r['shape']} | "
                  f"{r['measured']:.2f} | {r['pcl_pred']:.2f} | "
                  f"{r['pcl_err']:+.1f}% |")

    print()
    print("## Verdict\n")
    bcast_mean = statistics.mean(abs(r["bcast_err"]) for r in rows_with_results)
    pcl_mean = statistics.mean(abs(r["pcl_err"]) for r in rows_with_results)
    delta = bcast_mean - pcl_mean
    if delta > 5:
        print(f"  Per-cluster bytes lifts cost-model accuracy "
              f"({bcast_mean:.1f}% → {pcl_mean:.1f}% mean error). "
              f"This is the dominant fix for the K-split residual.")
    elif delta > 1:
        print(f"  Per-cluster bytes is a partial fix "
              f"({bcast_mean:.1f}% → {pcl_mean:.1f}% mean error) — "
              "real but not dominant. Other residuals remain.")
    else:
        print(f"  Per-cluster bytes does not meaningfully improve fit "
              f"({bcast_mean:.1f}% → {pcl_mean:.1f}%). The K-split "
              "residual is somewhere else: PT util, LF interaction, "
              "or PSUM term.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
