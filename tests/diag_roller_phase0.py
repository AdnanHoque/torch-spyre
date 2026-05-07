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

"""Phase 0 driver — Roller-on-AIU constraint enumerator vs. measurements.

Question this script answers:

  Given the Phase-0 cost model (`tests/hmi_cost_model.py`) and the
  constraint enumerator (`tests/roller_constraint_enumerator.py`),
  how often does the cost-model-ranked top-1 candidate match the
  empirically best-measured (split, mode) on the 30-row Project B
  validation set?

Baseline expectation: low — Project B's residual writeup notes the
cost model under-predicts k-split benefit by ~95% on shapes with
large LX overflow risk. The portfolio doc target is "23% → 60%+"
after subsequent phases add structural fixes.

Phase 0 deliverable: this script + the residual map it produces. The
residuals are what later phases (LX-fit gate, split-aware bytes,
regime-switched PSUM) will calibrate against.

Usage:
    python tests/diag_roller_phase0.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import predict, label  # noqa: E402
from tests.roller_constraint_enumerator import (  # noqa: E402
    Candidate,
    enumerate_candidates,
)
from tests.diag_hmi_cost_model_calibrate import VALIDATION  # noqa: E402


# ---- group VALIDATION rows by shape ----------------------------------

@dataclass
class ShapeRow:
    """One measured (split, mode, ms) for a (M, N, K) shape."""

    split: tuple[int, int, int]
    mode: str                       # "natural" | "id" | "kf"
    measured_ms: float
    row_label: str                  # original label for traceability


def _group_by_shape(
    rows: list[tuple],
) -> dict[tuple[int, int, int], list[ShapeRow]]:
    by_shape: dict[tuple[int, int, int], list[ShapeRow]] = defaultdict(list)
    for row_label, M, N, K, split, mode, measured in rows:
        by_shape[(M, N, K)].append(
            ShapeRow(split=split, mode=mode, measured_ms=measured,
                     row_label=row_label)
        )
    return by_shape


# ---- ranking ---------------------------------------------------------

def _predict_ms(shape: tuple[int, int, int], split: tuple[int, int, int],
                mode: str) -> float:
    k_fast = (mode == "kf")
    return predict(shape, split, dtype="fp16", k_fast=k_fast).t_wall_ms


def _candidate_modes(split: tuple[int, int, int]) -> list[str]:
    """Modes a candidate split can run in.

    Pure-M (32, 1, 1) is identity-only (no PSUM ring traversal). Splits
    with k > 1 can run in id (default) or kf (k_fast emission).
    """
    _, _, k = split
    if k == 1:
        return ["natural"]      # equivalent to identity for pure-M
    return ["id", "kf"]


# ---- main ------------------------------------------------------------

def main() -> int:
    by_shape = _group_by_shape(VALIDATION)

    print("# Phase 0 — Roller enumerator vs. cost-model ranking\n")
    print(f"validation rows: {len(VALIDATION)}")
    print(f"unique shapes:   {len(by_shape)}\n")

    # Per-shape diagnostic table
    print("## Per-shape: cost-model top-1 vs. measured-best\n")
    print("| shape (M, N, K) | #valid | measured-best | predicted-best | "
          "match? | rank gap |")
    print("|---|---:|---|---|:-:|---:|")

    hits = 0
    eligible = 0
    miss_details: list[str] = []

    for shape, shape_rows in sorted(by_shape.items(), key=lambda kv: kv[0]):
        if len(shape_rows) < 2:
            # Can't rank without comparison rows — skip from accuracy stats.
            continue
        eligible += 1

        # Empirical winner among MEASURED options for this shape.
        measured_best = min(shape_rows, key=lambda r: r.measured_ms)

        # Cost-model prediction for each MEASURED option.
        scored: list[tuple[ShapeRow, float]] = [
            (r, _predict_ms(shape, r.split, r.mode)) for r in shape_rows
        ]
        scored.sort(key=lambda x: x[1])
        predicted_best, predicted_ms = scored[0]

        match = (predicted_best.split == measured_best.split
                 and predicted_best.mode == measured_best.mode)
        if match:
            hits += 1

        # Rank gap = position of measured-best in cost-model order.
        order_by_pred = [s for s, _ in scored]
        rank_gap = order_by_pred.index(measured_best)

        mb_str = (f"{measured_best.split} {measured_best.mode} "
                  f"{measured_best.measured_ms:.2f}ms")
        pb_str = (f"{predicted_best.split} {predicted_best.mode} "
                  f"pred {predicted_ms:.2f}ms "
                  f"meas {predicted_best.measured_ms:.2f}ms")
        print(f"| {shape} | {len(shape_rows)} | {mb_str} | {pb_str} | "
              f"{'✓' if match else '✗'} | {rank_gap} |")

        if not match:
            miss_details.append(
                f"  shape {shape}: measured-best {measured_best.split} "
                f"{measured_best.mode} ({measured_best.measured_ms:.2f}ms);  "
                f"cost-model picked {predicted_best.split} "
                f"{predicted_best.mode} (predicted {predicted_ms:.2f}ms, "
                f"actually {predicted_best.measured_ms:.2f}ms)."
            )

    print()
    print("## Top-1 accuracy on shapes with ≥2 measured options\n")
    pct = (hits / eligible * 100) if eligible else 0.0
    print(f"  eligible shapes: {eligible}")
    print(f"  hits:            {hits}")
    print(f"  accuracy:        {pct:.0f}%\n")

    # Constraint-enumeration coverage on each shape — independent of measurements.
    print("## Enumerator coverage (out of 21-triple unconstrained set)\n")
    print("| shape (M, N, K) | valid | pruned by div+LX+stick | "
          "PT-rows-fail | PT-cols-fail |")
    print("|---|---:|---:|---:|---:|")
    for shape in sorted(by_shape.keys()):
        cands = enumerate_candidates(*shape)
        rows_fail = sum(1 for c in cands if not c.pt_rows_filled)
        cols_fail = sum(1 for c in cands if not c.pt_cols_filled)
        print(f"| {shape} | {len(cands)} | {21 - len(cands)} | "
              f"{rows_fail} | {cols_fail} |")

    # Where does the cost model rank the measured-best, when it misses?
    if miss_details:
        print()
        print("## Cost-model misses — residuals to chase in later phases\n")
        for line in miss_details:
            print(line)

    # Headline metric for Phase 0 status.
    print()
    print("## Phase 0 status\n")
    if pct >= 60:
        print(f"  Top-1 accuracy {pct:.0f}% ≥ 60% target — enumerator + "
              "current cost model already match the portfolio-doc "
              "Phase 0 exit goal.")
    else:
        print(f"  Top-1 accuracy {pct:.0f}% < 60% target — residuals above "
              "are the work for Phase 1 (LX-fit gate, split-aware bytes, "
              "regime-switched PSUM).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
