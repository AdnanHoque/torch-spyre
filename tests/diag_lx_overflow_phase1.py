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

"""LX residency planner — Phase 1 diagnostic (Fix A: PSUM-side gate).

Successor to `diag_lx_overflow_phase0.py`, which gated on operand A
residency following Project B's hypothesis. Probes 1–3 (this branch)
showed the actual binding constraint is the **PSUM accumulator** —
M_per × N_per × dtype_psum bytes — and the new `lx_fit.py` predicate
gates on that.

This script re-runs the same scan as Phase 0 but with the corrected
predicate, and adds a side-by-side comparison: how do the two gates
classify the production matmuls and the validation set rows?

  Section A — per-op LX-fit status under both gates across decoder
              blocks of five popular models × four M values. Two
              columns: A-side fit (old, wrong) vs. C-PSUM fit (new,
              right). Where they disagree is informative.

  Section B — cost-model residual on the 30-row validation set,
              partitioned by C-PSUM fit status. The expectation:
              under-prediction rows correlate with C-PSUM overflow
              (catastrophic regime), and other residuals (over-pred
              on small-M kf rows; PSUM ring distance under id) are
              independent of LX fit.

  Section C — production-planner impact: which heuristic-fired splits
              would overflow C-PSUM, and what is the predicted wall
              hit?

Usage:
    python tests/diag_lx_overflow_phase1.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import predict, label  # noqa: E402
from tests.hmi_cost_model_phase1_block import (  # noqa: E402
    MODELS,
    block_ops,
)
from tests.lx_fit import lx_breakdown, LX_BYTES_PER_CORE  # noqa: E402
from tests.diag_hmi_cost_model_calibrate import VALIDATION  # noqa: E402


# ---- A-side check (the OLD predicate, kept for comparison only) -----

_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1, "int8": 1}


def _a_side_fits(shape, split, dtype="fp16",
                 lx_bytes=LX_BYTES_PER_CORE) -> bool:
    """The OLD A-residency predicate. Wrong, kept for comparison."""
    M, N, K = shape
    m, _, k = split
    M_per = M // m
    K_per = K // k
    return (M_per * K_per * _DTYPE_BYTES[dtype]) <= lx_bytes


def _a_side_overage(shape, split, dtype="fp16",
                    lx_bytes=LX_BYTES_PER_CORE) -> float:
    M, _, K = shape
    m, _, k = split
    return (M // m) * (K // k) * _DTYPE_BYTES[dtype] / lx_bytes


# ---- mirror PR 1933 heuristic ---------------------------------------

_ELEMS_PER_STICK = 64


def _heuristic_split(M, N, K, max_cores=32, n_sticks_gate=32):
    if max_cores != 32 or M < 32 or M > 512:
        return None
    n_sticks = N // _ELEMS_PER_STICK
    k_sticks = K // _ELEMS_PER_STICK
    if n_sticks >= n_sticks_gate or k_sticks < 32:
        return None
    for n in (16, 8, 4, 2):
        if max_cores % n != 0 or n_sticks % n != 0:
            continue
        k = max_cores // n
        if k_sticks < k or k_sticks % k != 0:
            continue
        return (1, n, k)
    return None


def _fmt_bytes(n):
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.2f} MB"
    if n >= 1024:
        return f"{n // 1024} KB"
    return f"{n} B"


def _mark(b):
    return "✓" if b else "✗"


# ---- Section A — production matmul scan -----------------------------

@dataclass
class OpRow:
    model: str
    M: int
    op_name: str
    shape: tuple
    pure_m_split: tuple
    pure_m_lx: object
    pure_m_a_overage: float
    heuristic_split: tuple | None
    heuristic_lx: object | None
    heuristic_a_overage: float | None


def _scan_blocks() -> list[OpRow]:
    rows = []
    for _, cfg in MODELS.items():
        for M in (32, 128, 512, 2048):
            for op in block_ops(cfg, M):
                if op.kind != "matmul":
                    continue
                shape = op.shape
                pure_m = (32, 1, 1)
                pure_m_lx = lx_breakdown(shape, pure_m, dtype="fp16")
                pure_m_aov = _a_side_overage(shape, pure_m)
                hsplit = _heuristic_split(*shape)
                if hsplit is not None:
                    hlx = lx_breakdown(shape, hsplit, dtype="fp16")
                    haov = _a_side_overage(shape, hsplit)
                else:
                    hlx = None
                    haov = None
                rows.append(OpRow(
                    model=cfg.name, M=M, op_name=op.name, shape=shape,
                    pure_m_split=pure_m, pure_m_lx=pure_m_lx,
                    pure_m_a_overage=pure_m_aov,
                    heuristic_split=hsplit, heuristic_lx=hlx,
                    heuristic_a_overage=haov,
                ))
    return rows


def _print_section_a(rows):
    print("## Section A — Per-op LX-fit: old A-side gate vs. new C-PSUM gate\n")
    print(f"LX capacity: {_fmt_bytes(LX_BYTES_PER_CORE)} per corelet\n")
    print("Showing only rows where either gate disagrees with the other "
          "OR where the heuristic fires.\n")
    print("| model | M | op | split | A_per | A fits? | C_psum | C fits? | gate disagree? |")
    print("|---|---:|---|---|---:|:-:|---:|:-:|:-:|")

    a_only_overflow = 0   # A says overflow but C fits
    c_only_overflow = 0   # C says overflow but A fits
    both_overflow = 0
    heur_fired = 0
    heur_c_overflow = 0
    heur_a_overflow = 0

    for r in rows:
        # Pure-M row
        a_ov_pm = (r.pure_m_a_overage > 1.0)
        c_ov_pm = not r.pure_m_lx.fits
        disagree_pm = (a_ov_pm != c_ov_pm)
        if a_ov_pm and not c_ov_pm:
            a_only_overflow += 1
        if c_ov_pm and not a_ov_pm:
            c_only_overflow += 1
        if a_ov_pm and c_ov_pm:
            both_overflow += 1

        # Print pure-M row only if interesting (disagree or heuristic fires)
        if disagree_pm or r.heuristic_split is not None:
            print(f"| {r.model} | {r.M} | {r.op_name} | (32,1,1) | "
                  f"{_fmt_bytes(r.pure_m_lx.a_bytes)} | {_mark(not a_ov_pm)} | "
                  f"{_fmt_bytes(r.pure_m_lx.c_psum_bytes)} | "
                  f"{_mark(not c_ov_pm)} | {'⚠' if disagree_pm else ''} |")

        if r.heuristic_split is not None:
            heur_fired += 1
            a_ov_h = (r.heuristic_a_overage > 1.0)
            c_ov_h = not r.heuristic_lx.fits
            disagree_h = (a_ov_h != c_ov_h)
            if c_ov_h:
                heur_c_overflow += 1
            if a_ov_h:
                heur_a_overflow += 1
            print(f"| {r.model} | {r.M} | {r.op_name} | "
                  f"{r.heuristic_split} | "
                  f"{_fmt_bytes(r.heuristic_lx.a_bytes)} | {_mark(not a_ov_h)} | "
                  f"{_fmt_bytes(r.heuristic_lx.c_psum_bytes)} | "
                  f"{_mark(not c_ov_h)} | {'⚠' if disagree_h else ''} |")

    print()
    print("Section A counts:")
    print(f"  total matmul instances scanned: {len(rows)}")
    print(f"  pure-M: A overflows {sum(1 for r in rows if r.pure_m_a_overage > 1)},"
          f" C overflows {sum(1 for r in rows if not r.pure_m_lx.fits)}")
    print(f"  pure-M gate-disagreement count (A says overflow, C says fits): "
          f"{a_only_overflow}")
    print(f"  pure-M gate-disagreement count (C says overflow, A says fits): "
          f"{c_only_overflow}")
    print(f"  heuristic fires on:               {heur_fired} instances")
    print(f"  heuristic-pick A-overflow:        {heur_a_overflow}")
    print(f"  heuristic-pick C-overflow (NEW):  {heur_c_overflow}")
    print()


# ---- Section B — validation residuals partitioned by C-PSUM fit -----

def _print_section_b():
    print("## Section B — Validation residuals partitioned by C-PSUM fit\n")
    print("Hypothesis: rows where C_psum > LX should show large under-prediction"
          " (catastrophic regime). Other residuals should be C-fit-independent.\n")
    print("| row | split | mode | A_per | A fits? | C_psum | C fits? | "
          "predicted | measured | rel err |")
    print("|---|---|---|---:|:-:|---:|:-:|---:|---:|---:|")

    fit_rel = []
    over_rel = []
    fit_under = []   # under-predictions (model < measured) on fitting rows
    over_under = []  # under-predictions on overflow rows

    for row_label, M, N, K, split, mode, measured in VALIDATION:
        lxb = lx_breakdown((M, N, K), split, dtype="fp16")
        a_ov = _a_side_overage((M, N, K), split)
        cb = predict((M, N, K), split, dtype="fp16",
                     k_fast=(mode == "kf"))
        rel = (cb.t_wall_ms - measured) / measured * 100
        if lxb.fits:
            fit_rel.append(abs(rel))
            if rel < 0:
                fit_under.append(abs(rel))
        else:
            over_rel.append(abs(rel))
            if rel < 0:
                over_under.append(abs(rel))

        split_str = f"({split[0]},{split[1]},{split[2]})"
        print(f"| {row_label} | {split_str} | {mode} | "
              f"{_fmt_bytes(lxb.a_bytes)} | {_mark(a_ov <= 1)} | "
              f"{_fmt_bytes(lxb.c_psum_bytes)} | {_mark(lxb.fits)} | "
              f"{cb.t_wall_ms:.2f} | {measured:.2f} | {rel:+.1f}% |")

    print()
    n_f = len(fit_rel)
    n_o = len(over_rel)
    fmean = sum(fit_rel) / n_f if fit_rel else 0
    omean = sum(over_rel) / n_o if over_rel else 0
    print("Section B summary (|rel error|):")
    print(f"  C-PSUM-fitting rows:    n={n_f}  mean={fmean:.1f}%  "
          f"max={max(fit_rel) if fit_rel else 0:.1f}%")
    print(f"  C-PSUM-overflow rows:   n={n_o}  mean={omean:.1f}%  "
          f"max={max(over_rel) if over_rel else 0:.1f}%")
    print(f"  Under-predictions on fitting rows:   {len(fit_under)}/{n_f}")
    print(f"  Under-predictions on overflow rows:  {len(over_under)}/{n_o}")
    print()


# ---- Section C — heuristic-impact summary --------------------------

def _print_section_c(rows):
    print("## Section C — Production-planner impact (PR 1933 heuristic + new gate)\n")
    print("If the new C-PSUM gate were added to the heuristic, which "
          "currently-firing splits would it reject?\n")
    print("| model | M | op | shape | split | C_psum | C fits? | predicted Δ vs pure-M |")
    print("|---|---:|---|---|---|---:|:-:|---:|")
    rejected = 0
    for r in rows:
        if r.heuristic_split is None:
            continue
        if r.heuristic_lx.fits:
            continue
        rejected += 1
        cb_pure = predict(r.shape, r.pure_m_split, k_fast=False)
        cb_h = predict(r.shape, r.heuristic_split, k_fast=True)
        delta = cb_pure.t_wall_ms - cb_h.t_wall_ms
        print(f"| {r.model} | {r.M} | {r.op_name} | {r.shape} | "
              f"{r.heuristic_split} | "
              f"{_fmt_bytes(r.heuristic_lx.c_psum_bytes)} | "
              f"{_mark(False)} | {delta:+.2f} ms |")
    if rejected == 0:
        print("(no heuristic-fired splits overflow the new C-PSUM gate)")
    print()
    print(f"Heuristic rows rejected by new gate: {rejected}")
    print()


# ---- main -----------------------------------------------------------

def main() -> int:
    print("# LX residency planner — Phase 1 (Fix A: PSUM-side gate)\n")
    print("Predicate: per-core PSUM accumulator (M_per × N_per × 4 bytes) ≤ "
          f"{_fmt_bytes(LX_BYTES_PER_CORE)}\n")
    rows = _scan_blocks()
    _print_section_a(rows)
    _print_section_b()
    _print_section_c(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
