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

"""Strategic comparison: how much Phase 1 headroom does k_fast capture?

Phase 1 said decode-regime block wall is ~72% HMI-bound, with a
theoretical 28% headroom upper bound if all non-HMI ops could be
overlapped behind HMI ops (Phase 2 would refine this under the dep
graph).

Question this script answers: of that 28%, how much does the
companion k_fast PR (planner heuristic that picks (1, n, k>1) splits
for narrow-N small-M shapes) already capture? If most, Phase 2 is
fighting for a small remainder. If little, Phase 2 still pencils out.

Three block-wall numbers per configuration:

  (A) baseline:   planner-natural pure-M (32, 1, 1) on every matmul
  (B) k_fast:     same, except matmuls matching the PR 1933 heuristic
                  use the (1, n, k>1) split it picks
  (C) overlap:    Phase 1 perfect-overlap upper bound — wall set by
                  the dominant op only (not realistic, but bounds
                  what any scheduler could ever achieve)

Coverage analysis: for each matmul, report whether the heuristic
fires, the chosen split, and the per-op delta under k_fast.

Usage:
    python tests/hmi_cost_model_strategic_compare.py
    python tests/hmi_cost_model_strategic_compare.py --m 128
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import predict, label, LAUNCH_FLOOR_MS
from tests.hmi_cost_model_phase1_block import (
    MODELS,
    block_ops,
    predict_op,
)


# ---- k_fast heuristic (mirror of PR 1933 _try_k_fast_split) ---------

ELEMS_PER_STICK = 64        # fp16


def heuristic_split(M: int, N: int, K: int, max_cores: int = 32,
                    n_sticks_gate: int = 32):
    """Apply PR 1933's k_fast heuristic and return (m, n, k) or None.

    Mirrors _try_k_fast_split in torch_spyre._inductor.core_division
    on AdnanHoque/feat-k-fast-planner-heuristic. The n_sticks_gate
    arg lets us probe what happens if the n_sticks ceiling is
    relaxed (the sweep wins on o_proj came from shapes with
    n_sticks=128, well above the 32 cutoff).
    """
    if max_cores != 32:
        return None
    if M < 32 or M > 512:
        return None
    n_sticks = N // ELEMS_PER_STICK
    k_sticks = K // ELEMS_PER_STICK
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


# ---- per-op prediction wrappers -------------------------------------

def predict_with_split(op_shape, split, k_fast):
    """Predict wall for a matmul under a specific split + emission."""
    cb = predict(op_shape, split, dtype="fp16", k_fast=k_fast)
    return dict(t_wall=cb.t_wall_ms, t_compute=cb.t_compute_ms,
                t_hmi=cb.t_hmi_ms, t_psum=cb.t_psum_ms,
                hmi_bytes=cb.hmi_bytes, label=label(cb))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama_70b",
                        choices=list(MODELS.keys()))
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n-sticks-gate", type=int, default=32,
                        help="upper bound on n_sticks for heuristic to fire "
                             "(PR 1933 default: 32; relax to probe what wider "
                             "coverage would predict)")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    ops = block_ops(cfg, args.m)

    rows = []
    for op in ops:
        baseline = predict_op(op)
        kfast = baseline.copy()
        kfast_fired = False
        chosen_split = None
        if op.kind == "matmul":
            split = heuristic_split(*op.shape, n_sticks_gate=args.n_sticks_gate)
            if split is not None:
                kfast_fired = True
                chosen_split = split
                pred = predict_with_split(op.shape, split, k_fast=True)
                kfast = dict(
                    name=op.name, kind=op.kind,
                    t_compute=pred["t_compute"], t_hmi=pred["t_hmi"],
                    t_wall=pred["t_wall"], hmi_bytes=pred["hmi_bytes"],
                    label=pred["label"],
                )
        rows.append((op, baseline, kfast, kfast_fired, chosen_split))

    print(f"# Strategic compare: {cfg.name} decoder block at M={args.m}\n")
    print("(A) baseline = planner pure-M (32, 1, 1) on every matmul")
    print("(B) k_fast   = same, except heuristic fires for (1, n, k>1)")
    print("(C) overlap  = Phase 1 perfect-overlap bound (dominant op only)\n")

    # Per-op coverage table
    print("## Per-op coverage\n")
    print("| op | shape | heuristic? | split chosen | A wall | B wall | Δ |")
    print("|---|---|:-:|:-:|---:|---:|---:|")
    for op, A, B, fired, split in rows:
        shape_str = "×".join(str(d) for d in op.shape)
        if op.kind != "matmul":
            mark = "—"
            split_str = "—"
        elif fired:
            mark = "✓"
            split_str = f"(1, {split[1]}, {split[2]})"
        else:
            mark = "✗"
            split_str = "(32, 1, 1)"
        delta = B['t_wall'] - A['t_wall']
        delta_str = f"{delta:+.2f}" if abs(delta) > 0.01 else "—"
        print(f"| {op.name} | {shape_str} | {mark} | {split_str} | "
              f"{A['t_wall']:.2f} | {B['t_wall']:.2f} | {delta_str} |")
    print()

    # Block-level totals
    A_total = sum(r[1]['t_wall'] for r in rows)
    B_total = sum(r[2]['t_wall'] for r in rows)
    # Perfect-overlap bound: max single-op wall (any scheduler pays at
    # least the longest op's wall). Looser bound = sum of HMI-bound op
    # walls (since those are the irreducible HMI work).
    hmi_walls_A = [r[1]['t_wall'] for r in rows if r[1]['label'] == 'HMI-bound']
    C_overlap = sum(hmi_walls_A) if hmi_walls_A else max(r[1]['t_wall'] for r in rows)

    print("## Block-level totals\n")
    print(f"  (A) baseline pure-M:        {A_total:7.2f} ms")
    print(f"  (B) k_fast heuristic:       {B_total:7.2f} ms  "
          f"({(A_total - B_total) / A_total * 100:+.1f}%)")
    print(f"  (C) perfect-overlap bound:  {C_overlap:7.2f} ms  "
          f"({(A_total - C_overlap) / A_total * 100:+.1f}%)")
    print()

    # Coverage stats
    matmul_rows = [r for r in rows if r[0].kind == 'matmul']
    fired_rows = [r for r in matmul_rows if r[3]]
    print("## Coverage stats\n")
    print(f"  matmul ops in block:        {len(matmul_rows)}")
    print(f"  heuristic fires on:         {len(fired_rows)} "
          f"({len(fired_rows) / max(len(matmul_rows), 1) * 100:.0f}%)")
    if fired_rows:
        print(f"  Σ per-op savings:           "
              f"{sum(r[1]['t_wall'] - r[2]['t_wall'] for r in fired_rows):.2f} ms")
    print()

    # Headroom math
    a_headroom = A_total - C_overlap
    b_headroom = B_total - C_overlap
    captured = (A_total - B_total) / a_headroom * 100 if a_headroom > 0 else 0
    print("## Headroom decomposition\n")
    print(f"  Phase 1 headroom (A − C):   {a_headroom:7.2f} ms  "
          f"({a_headroom / A_total * 100:.0f}% of baseline)")
    print(f"  k_fast captures (A − B):    {A_total - B_total:7.2f} ms  "
          f"({captured:.0f}% of available headroom)")
    print(f"  remaining for Phase 2:      {b_headroom:7.2f} ms  "
          f"({b_headroom / A_total * 100:.0f}% of baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
