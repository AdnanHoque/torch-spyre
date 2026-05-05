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

"""Calibration harness for hmi_cost_model.predict().

Feeds a fixed validation set of (shape, split, mode, measured_ms)
tuples through the cost model and reports per-row absolute and
relative errors. The validation set is harvested from the diag-branch
measurement files (M-sweep, popular-models sweep, planner-correctness
check) so the harness is self-contained — no hardware required.

Goal: predictions within ~10% of measured. Use this to tune the
constants in hmi_cost_model.py (LAUNCH_FLOOR_MS, HMI_BW_GBS,
ACHIEVED_FRAC, etc.) and the structural assumptions (PT util curve,
HMI byte accounting, etc.).

Usage:
    python tests/diag_hmi_cost_model_calibrate.py
"""

from __future__ import annotations

import statistics
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import predict, label  # noqa: E402


# ---- validation set ----------------------------------------------------
# Format: (label, M, N, K, split, mode, measured_ms)
# mode is "natural" (planner pure-M; identity emission), "id" (forced
# split, identity emission), or "kf" (forced split, k_fast emission).

VALIDATION = [
    # ── M-sweep on five real shapes (real_workloads_msweep) ────────────

    # L3-70B kv_proj: (M, 1024, 8192)  shape varies in M only
    ("L3-70B kv_proj M=32",   32,   1024,  8192, (32, 1, 1), "natural", 3.307),
    ("L3-70B kv_proj M=128",  128,  1024,  8192, (32, 1, 1), "natural", 3.371),
    ("L3-70B kv_proj M=512",  512,  1024,  8192, (32, 1, 1), "natural", 3.357),
    ("L3-70B kv_proj M=1024", 1024, 1024,  8192, (32, 1, 1), "natural", 3.433),
    ("L3-70B kv_proj M=2048", 2048, 1024,  8192, (32, 1, 1), "natural", 3.666),
    ("L3-70B kv_proj M=128 (1,16,2)+kf", 128, 1024, 8192, (1, 16, 2), "kf", 3.089),
    ("L3-70B kv_proj M=512 (1,16,2)+kf", 512, 1024, 8192, (1, 16, 2), "kf", 3.174),
    ("L3-70B kv_proj M=2048 (1,16,2)+kf", 2048, 1024, 8192, (1, 16, 2), "kf", 3.939),

    # Mixtral 8x7B kv_proj: (M, 1024, 4096)
    ("Mixtral kv_proj M=32",   32,  1024, 4096, (32, 1, 1), "natural", 3.106),
    ("Mixtral kv_proj M=128",  128, 1024, 4096, (32, 1, 1), "natural", 3.136),
    ("Mixtral kv_proj M=2048", 2048,1024, 4096, (32, 1, 1), "natural", 3.274),
    ("Mixtral kv_proj M=128 +kf", 128, 1024, 4096, (1, 16, 2), "kf", 3.009),

    # DSv3 o_proj: (M, 7168, 16384)
    ("DSv3 o_proj M=32",   32,   7168, 16384, (32, 1, 1), "natural", 4.839),
    ("DSv3 o_proj M=128",  128,  7168, 16384, (32, 1, 1), "natural", 9.135),
    ("DSv3 o_proj M=512",  512,  7168, 16384, (32, 1, 1), "natural", 8.473),
    ("DSv3 o_proj M=2048", 2048, 7168, 16384, (32, 1, 1), "natural", 13.278),
    ("DSv3 o_proj M=128 +kf", 128, 7168, 16384, (1, 16, 2), "kf", 4.691),
    ("DSv3 o_proj M=2048 +kf", 2048, 7168, 16384, (1, 16, 2), "kf", 31.227),

    # DSv3 down_proj: (M, 7168, 2048)
    ("DSv3 down_proj M=32",   32,   7168, 2048, (32, 1, 1), "natural", 3.167),
    ("DSv3 down_proj M=128",  128,  7168, 2048, (32, 1, 1), "natural", 3.726),
    ("DSv3 down_proj M=2048", 2048, 7168, 2048, (32, 1, 1), "natural", 4.437),
    ("DSv3 down_proj M=128 +kf", 128, 7168, 2048, (1, 16, 2), "kf", 3.156),

    # DSv3 q_a_proj: (M, 1536, 7168)
    ("DSv3 q_a_proj M=32",   32,  1536, 7168, (32, 1, 1), "natural", 3.467),
    ("DSv3 q_a_proj M=128",  128, 1536, 7168, (32, 1, 1), "natural", 3.516),
    ("DSv3 q_a_proj M=2048", 2048,1536, 7168, (32, 1, 1), "natural", 4.055),
    ("DSv3 q_a_proj M=128 +kf", 128, 1536, 7168, (1, 8, 4), "kf", 3.224),

    # ── Planner-correctness data (1, 16, 2) + identity emission (no kf) ─

    ("L3-70B kv_proj M=2048 (1,16,2)+id", 2048, 1024, 8192,  (1, 16, 2), "id", 10.930),
    ("Mixtral kv_proj M=2048 (1,16,2)+id", 2048, 1024, 4096, (1, 16, 2), "id", 6.943),
    ("DSv3 o_proj M=2048 (1,16,2)+id",    2048, 7168, 16384, (1, 16, 2), "id", 116.116),
    ("DSv3 down_proj M=2048 (1,16,2)+id", 2048, 7168, 2048,  (1, 16, 2), "id", 17.066),
]


# ---- runner -----------------------------------------------------------

def _predict_for_mode(M, N, K, split, mode):
    """Return predicted wall_ms for the (split, mode) combination."""
    k_fast = (mode == "kf")
    return predict((M, N, K), split, dtype="fp16", k_fast=k_fast)


def main() -> int:
    print("# HMI cost-model calibration\n")
    print(
        f"| {'shape':<42} | "
        f"{'mode':<8} | "
        f"{'measured ms':>11} | "
        f"{'pred ms':>8} | "
        f"{'err':>7} | "
        f"{'rel':>6} | "
        f"{'class':<18} |"
    )
    print(f"|{'-'*44}|{'-'*10}|{'-'*13}|{'-'*10}|{'-'*9}|{'-'*8}|{'-'*20}|")

    rel_errors = []
    abs_errors = []
    over_threshold = 0
    threshold_pct = 10.0

    for shape_label, M, N, K, split, mode, measured in VALIDATION:
        cb = _predict_for_mode(M, N, K, split, mode)
        pred = cb.t_wall_ms
        abs_err = pred - measured
        rel_err = abs(abs_err) / measured * 100
        rel_errors.append(rel_err)
        abs_errors.append(abs_err)
        if rel_err > threshold_pct:
            over_threshold += 1
        marker = "✗" if rel_err > threshold_pct else "✓"
        print(
            f"| {shape_label:<42} | "
            f"{mode:<8} | "
            f"{measured:>10.3f}  | "
            f"{pred:>7.3f}  | "
            f"{abs_err:>+6.3f}  | "
            f"{rel_err:>4.1f}% | "
            f"{label(cb):<16} {marker} |"
        )

    print()
    print("## Aggregate fit\n")
    print(f"  rows in validation set:  {len(VALIDATION)}")
    print(f"  rows over {threshold_pct:.0f}% rel error:  "
          f"{over_threshold} ({over_threshold / len(VALIDATION) * 100:.0f}%)")
    print(f"  median rel error:        {statistics.median(rel_errors):.1f}%")
    print(f"  mean rel error:          {statistics.mean(rel_errors):.1f}%")
    print(f"  max rel error:           {max(rel_errors):.1f}%")
    print(f"  signed mean abs error:   {statistics.mean(abs_errors):+.3f} ms")

    print()
    print("## Worst rows\n")
    rows = list(zip(VALIDATION, rel_errors, abs_errors))
    rows.sort(key=lambda r: -r[1])
    for (shape_label, M, N, K, split, mode, measured), rel, ab in rows[:8]:
        cb = _predict_for_mode(M, N, K, split, mode)
        print(f"  {shape_label:<42}  measured={measured:.2f} "
              f"pred={cb.t_wall_ms:.2f}  rel={rel:.1f}%  ({label(cb)})")
        print(f"     compute={cb.t_compute_ms:.2f}  hmi={cb.t_hmi_ms:.2f}  "
              f"psum={cb.t_psum_ms:.2f}  pt_util={cb.pt_util:.2f}  "
              f"hops={cb.chain_hops}")

    print()
    if max(rel_errors) <= threshold_pct:
        print(f"VERDICT: cost model fits within {threshold_pct:.0f}% on every row. "
              "Good enough to drive Phase 2 decisions.")
    elif over_threshold <= len(VALIDATION) // 4:
        print(f"VERDICT: most rows fit within {threshold_pct:.0f}%, but "
              f"{over_threshold} outliers need attention. Worst rows above "
              "tell you which model component (compute / HMI / PSUM / "
              "launch floor) is mis-modelled.")
    else:
        print(f"VERDICT: {over_threshold}/{len(VALIDATION)} rows outside "
              f"{threshold_pct:.0f}% bound. Cost model needs structural "
              "changes before Phase 1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
