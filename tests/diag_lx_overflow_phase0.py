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

"""LX residency planner — Phase 0 diagnostic.

Question: where does the production planner (or its companion k_fast
heuristic, PR 1933) pick a work-division split whose per-core operand
footprint overflows the 2 MB LX scratchpad? When that happens, the
kernel template must re-fetch operand chunks per N-iteration, which
the Phase 0 cost model (`tests/hmi_cost_model.py`) does NOT model —
Project B Phase 0 documented this as the dominant residual factor.

This script does NOT change torch_spyre code. It produces three
diagnostic tables:

  Section A — per-op LX-fit status across decoder blocks of five
              popular models at four M values (32, 128, 512, 2048).
              For each matmul: fit status under (a) planner pure-M
              (32, 1, 1) and (b) PR 1933 heuristic's chosen split.

  Section B — cost-model residual on the 30-row Project B validation
              set, partitioned by LX-fit status. Shows the predicted-
              vs-measured gap collapses on LX-fitting rows and blows
              up on overflow rows.

  Section C — block-level wall-time impact estimate. For each
              (model, M), how many ops would the heuristic put into
              an overflowing split, and what's the cost-model under-
              prediction in aggregate ms?

The output is the residual map for later phases:

  Phase 1 → add LX-overflow penalty to hmi_cost_model.predict().
  Phase 2 → add the gate to torch_spyre._inductor.core_division so the
            production planner avoids overflow splits.

Usage:
    python tests/diag_lx_overflow_phase0.py
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


# ---- mirror PR 1933's k_fast heuristic ------------------------------
# (copy of `_try_k_fast_split` from torch_spyre._inductor.core_division
# on AdnanHoque/feat-k-fast-planner-heuristic, also mirrored in
# tests/hmi_cost_model_strategic_compare.py)

_ELEMS_PER_STICK = 64        # fp16


def _heuristic_split(M: int, N: int, K: int, max_cores: int = 32,
                     n_sticks_gate: int = 32):
    if max_cores != 32:
        return None
    if M < 32 or M > 512:
        return None
    n_sticks = N // _ELEMS_PER_STICK
    k_sticks = K // _ELEMS_PER_STICK
    if n_sticks >= n_sticks_gate:
        return None
    if k_sticks < 32:
        return None
    for n in (16, 8, 4, 2):
        if max_cores % n != 0 or n_sticks % n != 0:
            continue
        k = max_cores // n
        if k_sticks < k or k_sticks % k != 0:
            continue
        return (1, n, k)
    return None


# ---- helpers --------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.2f} MB"
    if n >= 1024:
        return f"{n // 1024} KB"
    return f"{n} B"


def _fits_mark(fits: bool) -> str:
    return "✓" if fits else "✗"


# ---- Section A: per-op LX-fit across model × M sweep ----------------

@dataclass
class OpRow:
    model: str
    M: int
    op_name: str
    shape: tuple[int, int, int]
    pure_m_split: tuple[int, int, int]
    pure_m_lx: object           # LXBreakdown
    heuristic_split: tuple[int, int, int] | None
    heuristic_lx: object | None  # LXBreakdown or None


def _scan_blocks() -> list[OpRow]:
    rows: list[OpRow] = []
    for model_key, cfg in MODELS.items():
        for M in (32, 128, 512, 2048):
            for op in block_ops(cfg, M):
                if op.kind != "matmul":
                    continue
                shape = op.shape
                pure_m = (32, 1, 1)
                pure_m_lx = lx_breakdown(shape, pure_m, dtype="fp16")
                hsplit = _heuristic_split(*shape)
                hlx = (lx_breakdown(shape, hsplit, dtype="fp16")
                       if hsplit is not None else None)
                rows.append(OpRow(
                    model=cfg.name, M=M, op_name=op.name, shape=shape,
                    pure_m_split=pure_m, pure_m_lx=pure_m_lx,
                    heuristic_split=hsplit, heuristic_lx=hlx,
                ))
    return rows


def _print_section_a(rows: list[OpRow]) -> dict:
    print("## Section A — Per-op LX-fit across model × M sweep\n")
    print(f"LX capacity: {_fmt_bytes(LX_BYTES_PER_CORE)} per corelet")
    print("Predicate: A_per_core (stationary operand) ≤ LX. B streams "
          "through the data ring and is not counted.\n")
    print("| model | M | op | shape (M,N,K) | pure-M A_per | "
          "fits? | heuristic | h split | h A_per | h fits? |")
    print("|---|---:|---|---|---:|:-:|:-:|---|---:|:-:|")
    pure_overflow_count = 0
    heur_fired = 0
    heur_overflow_count = 0
    for r in rows:
        pure_a = _fmt_bytes(r.pure_m_lx.a_bytes)
        pure_mark = _fits_mark(r.pure_m_lx.fits)
        if not r.pure_m_lx.fits:
            pure_overflow_count += 1
        if r.heuristic_split is not None:
            heur_fired += 1
            h_split_str = f"({r.heuristic_split[0]},{r.heuristic_split[1]},{r.heuristic_split[2]})"
            h_a = _fmt_bytes(r.heuristic_lx.a_bytes)
            h_mark = _fits_mark(r.heuristic_lx.fits)
            if not r.heuristic_lx.fits:
                heur_overflow_count += 1
            heur_cell = "✓"
        else:
            h_split_str = "—"
            h_a = "—"
            h_mark = "—"
            heur_cell = "✗"
        # Trim noise: only print rows where something is interesting
        # (heuristic fires OR pure-M overflows)
        if r.heuristic_split is None and r.pure_m_lx.fits:
            continue
        shape_str = f"({r.shape[0]},{r.shape[1]},{r.shape[2]})"
        print(f"| {r.model} | {r.M} | {r.op_name} | {shape_str} | "
              f"{pure_a} | {pure_mark} | {heur_cell} | {h_split_str} | "
              f"{h_a} | {h_mark} |")
    print()
    print("Section A counts:")
    print(f"  total matmul instances scanned: {len(rows)}")
    print(f"  pure-M overflows LX:            {pure_overflow_count}")
    print(f"  heuristic fires:                {heur_fired}")
    print(f"  heuristic-pick overflows LX:    {heur_overflow_count}")
    print()
    return dict(
        total=len(rows),
        pure_overflow=pure_overflow_count,
        heur_fired=heur_fired,
        heur_overflow=heur_overflow_count,
    )


# ---- Section B: residual on Project B validation set ----------------

def _print_section_b() -> dict:
    print("## Section B — Cost-model residual partitioned by LX-fit\n")
    print("| row | split | mode | A_per | fits? | predicted | "
          "measured | rel err |")
    print("|---|---|---|---:|:-:|---:|---:|---:|")
    fit_rel = []
    overflow_rel = []
    for row_label, M, N, K, split, mode, measured in VALIDATION:
        lxb = lx_breakdown((M, N, K), split, dtype="fp16")
        cb = predict((M, N, K), split, dtype="fp16",
                     k_fast=(mode == "kf"))
        rel = (cb.t_wall_ms - measured) / measured * 100
        if lxb.fits:
            fit_rel.append(abs(rel))
        else:
            overflow_rel.append(abs(rel))
        split_str = f"({split[0]},{split[1]},{split[2]})"
        print(f"| {row_label} | {split_str} | {mode} | "
              f"{_fmt_bytes(lxb.a_bytes)} | {_fits_mark(lxb.fits)} | "
              f"{cb.t_wall_ms:.2f} | {measured:.2f} | {rel:+.1f}% |")
    print()
    n_fit = len(fit_rel)
    n_over = len(overflow_rel)
    fit_mean = sum(fit_rel) / n_fit if fit_rel else 0
    over_mean = sum(overflow_rel) / n_over if overflow_rel else 0
    fit_max = max(fit_rel) if fit_rel else 0
    over_max = max(overflow_rel) if overflow_rel else 0
    print("Section B summary (|rel error|):")
    print(f"  LX-fitting rows:    n={n_fit}  mean={fit_mean:.1f}%  "
          f"max={fit_max:.1f}%")
    print(f"  LX-overflow rows:   n={n_over}  mean={over_mean:.1f}%  "
          f"max={over_max:.1f}%")
    print()
    return dict(
        fit_n=n_fit, over_n=n_over,
        fit_mean=fit_mean, over_mean=over_mean,
        fit_max=fit_max, over_max=over_max,
    )


# ---- Section C: block-level wall-time under-prediction --------------

def _print_section_c(rows: list[OpRow]) -> None:
    print("## Section C — Block-level cost-model under-prediction "
          "from heuristic-overflow ops\n")
    print("| model | M | heuristic-overflow ops | predicted Δwall ms | "
          "actual measured Δ (where known) |")
    print("|---|---:|:-:|---:|---:|")

    # Build a quick map of validation rows for lookup: shape+split → measured
    val_lookup = {}
    for label_, M, N, K, split, mode, measured in VALIDATION:
        val_lookup[((M, N, K), tuple(split), mode)] = measured

    # Group OpRows by (model, M)
    by_block: dict[tuple[str, int], list[OpRow]] = {}
    for r in rows:
        by_block.setdefault((r.model, r.M), []).append(r)

    for (model, M), ops in sorted(by_block.items()):
        overflow_ops = [
            r for r in ops
            if r.heuristic_split is not None and not r.heuristic_lx.fits
        ]
        if not overflow_ops:
            continue
        op_names = ",".join(r.op_name for r in overflow_ops)
        # Predicted Δ wall = sum over overflow ops of
        #   predict(shape, heuristic, k_fast=True).t_wall_ms
        #   minus predict(shape, pure_m, k_fast=False).t_wall_ms
        # i.e. the cost-model-predicted savings the heuristic claims
        # (which won't materialise because of LX overflow).
        pred_delta = 0.0
        for r in overflow_ops:
            cb_pure = predict(r.shape, r.pure_m_split, k_fast=False)
            cb_h = predict(r.shape, r.heuristic_split, k_fast=True)
            pred_delta += cb_pure.t_wall_ms - cb_h.t_wall_ms
        # Actual measured Δ (where we have data): sum over ops with both
        # measurements available.
        actual_delta = 0.0
        any_measured = False
        for r in overflow_ops:
            key_pure = (r.shape, r.pure_m_split, "natural")
            key_h = (r.shape, tuple(r.heuristic_split), "kf")
            if key_pure in val_lookup and key_h in val_lookup:
                actual_delta += val_lookup[key_pure] - val_lookup[key_h]
                any_measured = True
        actual_str = f"{actual_delta:+.2f}" if any_measured else "—"
        print(f"| {model} | {M} | {op_names} | {pred_delta:+.2f} | {actual_str} |")
    print()


# ---- main ------------------------------------------------------------

def main() -> int:
    print("# LX residency planner — Phase 0 diagnostic\n")
    print("Goal: identify where the planner / k_fast heuristic picks "
          "splits that overflow the 2 MB LX scratchpad — the dominant "
          "residual in the Phase 0 cost model.\n")

    rows = _scan_blocks()
    a_stats = _print_section_a(rows)
    b_stats = _print_section_b()
    _print_section_c(rows)

    # Headline
    print("## Phase 0 verdict\n")
    print(f"  pure-M (32,1,1) overflows LX on "
          f"{a_stats['pure_overflow']}/{a_stats['total']} matmul instances "
          f"({a_stats['pure_overflow']/a_stats['total']*100:.0f}%)")
    print(f"  PR 1933 heuristic fires on {a_stats['heur_fired']} instances; "
          f"{a_stats['heur_overflow']} of those overflow LX "
          f"({a_stats['heur_overflow']/max(a_stats['heur_fired'],1)*100:.0f}%)")
    print(f"  cost-model rel error on overflow rows is "
          f"{b_stats['over_mean']:.0f}% mean / {b_stats['over_max']:.0f}% max, "
          f"vs. {b_stats['fit_mean']:.0f}% mean / {b_stats['fit_max']:.0f}% "
          "max on fitting rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
