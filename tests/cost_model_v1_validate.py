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

"""Cost-model v1 validation — Phase 1.2.

Loads Phase 1.0 split-gap measurements (`diag_split_gap_results.md`),
runs each `(shape, m, n, k)` row through `cost_model_v1.predict_wall_ms`,
and reports:

  1. **Wall-time MAPE** with default constants — how far off are the
     predictions in absolute ms terms.
  2. **Best-split top-K accuracy** — how often the cost model ranks the
     empirical-best split in its top-1 / top-3 / top-5. This is the
     metric that actually matters for the planner: we don't need
     accurate ms predictions, we need to *pick the right split*.
  3. **Calibration grid search** — sweep `PER_CORE_TFLOPS` and
     `EFFECTIVE_DDR_BW_GBS` over a coarse grid, find the (TFLOPS, BW)
     pair that minimizes MAPE. Reports both the calibrated MAPE and
     the calibrated top-K accuracy.

Run: python tests/cost_model_v1_validate.py
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass

import regex as re

import cost_model_v1 as cm

PHASE1_RESULTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "diag_split_gap_results.md",
)


# ---- Phase 1.0 markdown parser ------------------------------------------

@dataclass(frozen=True)
class _Sample:
    label: str       # shape display name
    M: int
    N: int
    K: int
    m: int
    n: int
    k: int
    measured_ms: float


_SHAPE_HEADER = re.compile(
    r"^##\s+(?P<label>.+?)\s+—\s+`\((?P<M>\d+),\s*(?P<N>\d+),\s*(?P<K>\d+)\)`",
    re.MULTILINE,
)
_ROW = re.compile(
    r"^\|\s*\((?P<m>\d+),\s*(?P<n>\d+),\s*(?P<k>\d+)\)\s*\|\s*"
    r"(?P<ms>[\d.]+|err)\s*\|",
    re.MULTILINE,
)


def _parse_phase1_results(path: str) -> list[_Sample]:
    """Extract `(label, M, N, K, m, n, k, measured_ms)` rows. Drops
    rows whose ms cell is `err`."""
    with open(path) as f:
        text = f.read()

    samples: list[_Sample] = []
    headers = list(_SHAPE_HEADER.finditer(text))
    for i, h in enumerate(headers):
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end]
        label = h.group("label").strip()
        M, N, K = int(h.group("M")), int(h.group("N")), int(h.group("K"))
        for row in _ROW.finditer(body):
            if row.group("ms") == "err":
                continue
            samples.append(_Sample(
                label=label, M=M, N=N, K=K,
                m=int(row.group("m")),
                n=int(row.group("n")),
                k=int(row.group("k")),
                measured_ms=float(row.group("ms")),
            ))
    return samples


# ---- error metrics ------------------------------------------------------

def _set_constants(per_core_tflops: float, ddr_bw_gbs: float, alpha: float) -> None:
    cm.PER_CORE_TFLOPS = per_core_tflops
    cm.EFFECTIVE_DDR_BW_GBS = ddr_bw_gbs
    cm.SHARING_FACTOR = alpha


def _mape(
    samples: list[_Sample], per_core_tflops: float, ddr_bw_gbs: float,
    alpha: float,
) -> float:
    """Mean absolute percentage error across all rows."""
    _set_constants(per_core_tflops, ddr_bw_gbs, alpha)
    abs_errs: list[float] = []
    for s in samples:
        pred = cm.predict_wall_ms(s.M, s.N, s.K, s.m, s.n, s.k)
        abs_errs.append(abs(pred - s.measured_ms) / s.measured_ms)
    return statistics.mean(abs_errs) * 100.0


def _top_k_accuracy(
    samples: list[_Sample],
    per_core_tflops: float,
    ddr_bw_gbs: float,
    alpha: float,
    k: int = 1,
) -> tuple[float, list[tuple[str, int, int]]]:
    """For each shape, find the empirical-best (m,n,k) and check whether
    the cost model ranks it within its top-k predictions. Returns
    (accuracy, per_shape_diagnostics)."""
    _set_constants(per_core_tflops, ddr_bw_gbs, alpha)

    by_shape: dict[tuple[str, int, int, int], list[_Sample]] = {}
    for s in samples:
        by_shape.setdefault((s.label, s.M, s.N, s.K), []).append(s)

    correct = 0
    diagnostics: list[tuple[str, int, int]] = []  # (label, best_pred_rank, total)
    for key, rows in by_shape.items():
        # empirical best
        best_meas = min(rows, key=lambda r: r.measured_ms)
        # rank by predicted ms
        ranked = sorted(
            rows,
            key=lambda r: cm.predict_wall_ms(r.M, r.N, r.K, r.m, r.n, r.k),
        )
        # find rank of empirical best
        rank = next(
            i for i, r in enumerate(ranked)
            if (r.m, r.n, r.k) == (best_meas.m, best_meas.n, best_meas.k)
        )
        if rank < k:
            correct += 1
        diagnostics.append((key[0], rank + 1, len(rows)))
    return correct / len(by_shape) * 100.0, diagnostics


# ---- calibration --------------------------------------------------------

def _calibrate(
    samples: list[_Sample],
    tflops_grid: list[float],
    bw_grid: list[float],
    alpha_grid: list[float],
    objective: str = "mape",
) -> tuple[float, float, float, float]:
    """Grid search over (TFLOPS, BW, alpha). `objective` ∈ {"mape",
    "regret"}: mape minimizes wall-time MAPE; regret minimizes the mean
    regret-ratio (planner-relevant). Returns (best_tflops, best_bw,
    best_alpha, best_objective_value)."""
    best = (None, None, None, float("inf"))
    for t in tflops_grid:
        for b in bw_grid:
            for a in alpha_grid:
                if objective == "mape":
                    val = _mape(samples, t, b, a)
                elif objective == "regret":
                    val, _ = _regret_ratio(samples, t, b, a)
                else:
                    raise ValueError(f"unknown objective {objective!r}")
                if val < best[3]:
                    best = (t, b, a, val)
    return best  # type: ignore[return-value]


# ---- report --------------------------------------------------------------

def _regret_ratio(
    samples: list[_Sample], per_core_tflops: float, ddr_bw_gbs: float,
    alpha: float,
) -> tuple[float, float]:
    """For each shape, compute `wall_at_model_pick / wall_at_empirical_best`.
    A ratio of 1.0 means the model picked something with the same measured
    wall time as the optimum (might or might not be the literal best split).
    A ratio of 1.5 means the model's pick is 50% slower. Returns
    (mean_regret, max_regret)."""
    _set_constants(per_core_tflops, ddr_bw_gbs, alpha)
    by_shape: dict[tuple[str, int, int, int], list[_Sample]] = {}
    for s in samples:
        by_shape.setdefault((s.label, s.M, s.N, s.K), []).append(s)

    ratios: list[float] = []
    for rows in by_shape.values():
        best_meas = min(rows, key=lambda r: r.measured_ms)
        model_pick = min(
            rows,
            key=lambda r: cm.predict_wall_ms(r.M, r.N, r.K, r.m, r.n, r.k),
        )
        ratios.append(model_pick.measured_ms / best_meas.measured_ms)
    return statistics.mean(ratios), max(ratios)


def _print_per_shape_breakdown(
    samples: list[_Sample],
    per_core_tflops: float,
    ddr_bw_gbs: float,
    alpha: float,
) -> None:
    _set_constants(per_core_tflops, ddr_bw_gbs, alpha)

    by_shape: dict[tuple[str, int, int, int], list[_Sample]] = {}
    for s in samples:
        by_shape.setdefault((s.label, s.M, s.N, s.K), []).append(s)

    print("\n## Per-shape breakdown\n")
    print("| shape | best measured | model picks | regret | rank | shape MAPE |")
    print("|---|---|---|---:|---:|---:|")
    for (label, M, N, K), rows in by_shape.items():
        best_meas = min(rows, key=lambda r: r.measured_ms)
        ranked_pred = sorted(
            rows,
            key=lambda r: cm.predict_wall_ms(r.M, r.N, r.K, r.m, r.n, r.k),
        )
        best_pred = ranked_pred[0]
        rank = next(
            i for i, r in enumerate(ranked_pred)
            if (r.m, r.n, r.k) == (best_meas.m, best_meas.n, best_meas.k)
        ) + 1
        regret = best_pred.measured_ms / best_meas.measured_ms
        shape_mape = statistics.mean(
            abs(cm.predict_wall_ms(r.M, r.N, r.K, r.m, r.n, r.k) - r.measured_ms)
            / r.measured_ms
            for r in rows
        ) * 100.0
        print(f"| {label} | "
              f"({best_meas.m},{best_meas.n},{best_meas.k}) "
              f"@ {best_meas.measured_ms:.2f}ms | "
              f"({best_pred.m},{best_pred.n},{best_pred.k}) "
              f"@ {best_pred.measured_ms:.2f}ms | "
              f"{regret:.2f}× | {rank}/{len(rows)} | {shape_mape:.1f}% |")


def _emit_calibration_block(
    samples: list[_Sample],
    t: float, b: float, a: float, mape: float,
    default_mape: float, default_t: float, default_b: float, default_a: float,
    regret_mean_d: float, regret_max_d: float,
    top1_default: float, top3_default: float, top5_default: float,
) -> None:
    print(f"  PER_CORE_TFLOPS  = {t}")
    print(f"  EFFECTIVE_DDR_BW = {b} GB/s")
    print(f"  SHARING_FACTOR   = {a}")
    top1_cal, _ = _top_k_accuracy(samples, t, b, a, k=1)
    top3_cal, _ = _top_k_accuracy(samples, t, b, a, k=3)
    top5_cal, _ = _top_k_accuracy(samples, t, b, a, k=5)
    regret_mean_c, regret_max_c = _regret_ratio(samples, t, b, a)
    print(f"\n  Wall-time MAPE   : {mape:.1f}%  "
          f"(default {default_mape:.1f}%)")
    print(f"  Top-1 best-split : {top1_cal:.1f}%  (default {top1_default:.1f}%)")
    print(f"  Top-3 best-split : {top3_cal:.1f}%  (default {top3_default:.1f}%)")
    print(f"  Top-5 best-split : {top5_cal:.1f}%  (default {top5_default:.1f}%)")
    print(f"  Mean regret      : {regret_mean_c:.3f}×  "
          f"(default {regret_mean_d:.3f}×)")
    print(f"  Max regret       : {regret_max_c:.3f}×  "
          f"(default {regret_max_d:.3f}×)")


def main() -> int:
    samples = _parse_phase1_results(PHASE1_RESULTS)
    print(f"Loaded {len(samples)} (shape, split) samples from "
          f"{os.path.basename(PHASE1_RESULTS)}")

    n_shapes = len({(s.label, s.M, s.N, s.K) for s in samples})
    print(f"Across {n_shapes} unique shapes")

    print("\n# Cost-model v1 validation — Phase 1.2\n")

    # --- Section 1: default constants ---
    default_t = cm.PER_CORE_TFLOPS
    default_b = cm.EFFECTIVE_DDR_BW_GBS
    default_a = cm.SHARING_FACTOR
    default_mape = _mape(samples, default_t, default_b, default_a)
    top1_default, _ = _top_k_accuracy(samples, default_t, default_b, default_a, k=1)
    top3_default, _ = _top_k_accuracy(samples, default_t, default_b, default_a, k=3)
    top5_default, _ = _top_k_accuracy(samples, default_t, default_b, default_a, k=5)
    regret_mean_d, regret_max_d = _regret_ratio(
        samples, default_t, default_b, default_a)

    print("## Default constants (initial guesses)\n")
    print(f"  PER_CORE_TFLOPS    = {default_t}")
    print(f"  EFFECTIVE_DDR_BW   = {default_b} GB/s")
    print(f"  SHARING_FACTOR     = {default_a}")
    print(f"  LAUNCH_FLOOR_MS    = {cm.LAUNCH_FLOOR_MS}")
    print(f"\n  Wall-time MAPE   : {default_mape:.1f}%")
    print(f"  Top-1 best-split : {top1_default:.1f}% ({n_shapes} shapes)")
    print(f"  Top-3 best-split : {top3_default:.1f}%")
    print(f"  Top-5 best-split : {top5_default:.1f}%")
    print(f"  Mean regret      : {regret_mean_d:.3f}×  (1.0 = optimal pick)")
    print(f"  Max regret       : {regret_max_d:.3f}×")

    # --- Section 2: calibration grid ---
    tflops_grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    bw_grid     = [50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1200.0, 2000.0]
    alpha_grid  = [0.0, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0]

    # Calibrate with two different objectives — MAPE is wall-time accuracy,
    # regret is "how good are the splits the model would pick". They can
    # land on different points.
    print("\n## Calibrated for wall-time MAPE\n")
    best_t, best_b, best_a, best_mape = _calibrate(
        samples, tflops_grid, bw_grid, alpha_grid, objective="mape")
    _emit_calibration_block(
        samples, best_t, best_b, best_a, best_mape,
        default_mape, default_t, default_b, default_a,
        regret_mean_d, regret_max_d,
        top1_default, top3_default, top5_default,
    )

    print("\n## Calibrated for mean regret (planner-relevant)\n")
    best_t_r, best_b_r, best_a_r, best_regret = _calibrate(
        samples, tflops_grid, bw_grid, alpha_grid, objective="regret")
    mape_r = _mape(samples, best_t_r, best_b_r, best_a_r)
    _emit_calibration_block(
        samples, best_t_r, best_b_r, best_a_r, mape_r,
        default_mape, default_t, default_b, default_a,
        regret_mean_d, regret_max_d,
        top1_default, top3_default, top5_default,
    )

    # --- Section 3: per-shape failure analysis with regret-min constants ---
    _print_per_shape_breakdown(samples, best_t_r, best_b_r, best_a_r)
    top1_cal, _ = _top_k_accuracy(samples, best_t_r, best_b_r, best_a_r, k=1)
    top3_cal, _ = _top_k_accuracy(samples, best_t_r, best_b_r, best_a_r, k=3)
    regret_mean_c, regret_max_c = _regret_ratio(
        samples, best_t_r, best_b_r, best_a_r)

    # --- Verdict ---
    print("\n## Verdict\n")
    print(f"  Top-1: {top1_cal:.0f}%   Top-3: {top3_cal:.0f}%   "
          f"Mean regret: {regret_mean_c:.2f}×   "
          f"Max regret: {regret_max_c:.2f}×")
    print()
    if regret_mean_c <= 1.05 and regret_max_c <= 1.15:
        print("  Mean regret ≤ 5%, max ≤ 15% — model is planner-usable. "
              "Proceed to Phase 2 (planner integration).")
    elif regret_mean_c <= 1.10:
        print("  Mean regret ≤ 10%. Useful as a tiebreaker on top of the "
              "current planner, but not yet good enough to replace it. "
              "Consider Phase 1.3 (refine missing terms).")
    else:
        print(f"  Mean regret = {regret_mean_c:.2f}× — model picks "
              "splits that lose meaningful wall time. Phase 1.3 should "
              "investigate why (split-dependent BW, sync cost, etc.).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
