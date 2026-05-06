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

"""End-to-end: FA joint SWP+WS savings flowed through to block wall.

The FA prototype showed 1.18-1.83× per-op speedup on attention.
This script asks: how much of that flows through to block-level
wall time, given the rest of the block is unchanged?

Method:
  1. Take Phase 1's per-op wall predictions for one decoder block
  2. Substitute the attention op's wall with the FA prototype's
     prediction under {serial, decoupled, joint} modes
  3. Run Phase 2's concurrent simulator on the modified block
  4. Compare block-level wall under each mode

The expectation, given Project B's verdict that decode-regime blocks
are HMI-bound: at decode M (≤512), attention is a small fraction of
block wall AND HMI is binding, so per-op savings on attention will
get absorbed. At prefill M (≥1024), the block is compute-bound and
attention savings can flow through.

Usage:
    python tests/joint_swp_ws_block_e2e.py --model llama_70b
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model_phase1_block import (
    MODELS, block_ops, predict_op
)
from tests.hmi_cost_model_phase2_concurrent import (
    SimOp, build_dep_graph, simulate
)
from tests.hmi_cost_model import LAUNCH_FLOOR_MS
from tests.joint_swp_ws_fa_prototype import solve_fa


# Cycle → ms conversion (assume 1 GHz)
CYC_TO_MS = 1e-6


def fa_wall_ms(M: int, n_heads: int, head_dim: int,
               sfp_softmax_cyc: int, mode: str) -> float:
    """Predict attention wall in ms via the FA prototype.

    Models the attention compute as a Q-tile loop: 32 Q-tiles per core
    (M/B_r/2 heads-per-core for Llama 70B M=2048), inner loop sweeps
    K/V tiles. Each inner iter is one FA prototype iteration.
    """
    B_r = 128  # Q tile rows
    B_c = 128  # K/V tile cols
    n_q_tiles_per_head = max(M // B_r, 1)
    n_kv_tiles = max(M // B_c, 1)
    heads_per_core = max(n_heads // 32, 1)

    # Inner-loop wall via FA prototype (16 iters by default)
    inner = solve_fa(n_kv_tiles, sfp_softmax_cyc, mode, time_limit=20.0)
    inner_wall_cyc = inner.wall

    total_q_tiles_per_core = n_q_tiles_per_head * heads_per_core
    total_cyc = inner_wall_cyc * total_q_tiles_per_core
    return total_cyc * CYC_TO_MS


def block_wall_with_attn_wall(cfg, M: int, attn_wall_ms: float) -> float:
    """Run Phase 2 concurrent simulator with attention wall substituted."""
    ops = block_ops(cfg, M)
    rows = []
    for op in ops:
        r = predict_op(op)
        if op.name == "attn_qkt_softmax_v":
            # Substitute the FA prototype's wall. Treat as a single op
            # with t_compute = attn_wall, t_hmi = 0 (the inner loop
            # already accounts for HMI).
            r = dict(r)
            r['t_compute'] = attn_wall_ms
            r['t_hmi'] = 0.0
            r['t_psum'] = 0.0
        rows.append(r)
    deps_map = build_dep_graph([o.name for o in ops])
    sim_ops = []
    for r in rows:
        sim_ops.append(SimOp(
            name=r['name'], kind=r['kind'],
            t_hmi_with_lf=r['t_hmi'] + LAUNCH_FLOOR_MS,
            t_compute=r['t_compute'],
            t_psum=r.get('t_psum', 0.0),
            deps=deps_map.get(r['name'], []),
        ))
    s = simulate(sim_ops, "serial")
    c = simulate(sim_ops, "concurrent")
    return s.total_wall, c.total_wall


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama_70b",
                        choices=list(MODELS.keys()))
    parser.add_argument("--m", type=int, default=2048,
                        help="batch-token count (decode 32-128, prefill 1024+)")
    parser.add_argument("--sfp", type=int, default=3000,
                        help="SFP softmax cycles per tile (regime knob)")
    parser.add_argument("--all", action="store_true",
                        help="sweep models × M ∈ {128, 2048} × SFP ∈ {1500, 3000, 5000}")
    args = parser.parse_args()

    if args.all:
        return run_sweep()

    cfg = MODELS[args.model]
    M = args.m

    print(f"# End-to-end: joint SWP+WS on attention → block wall\n")
    print(f"## Setup\n")
    print(f"  model:  {cfg.name}")
    print(f"  M:      {M}")
    print(f"  SFP softmax: {args.sfp} cycles/tile")
    print()

    # Per-op walls under each mode (just attention substitutes)
    print(f"## Attention op wall under each mode (single op, all 32 cores)\n")
    walls = {}
    for mode in ["serial", "decoupled", "joint"]:
        w = fa_wall_ms(M, cfg.n_heads, cfg.head_dim, args.sfp, mode)
        walls[mode] = w
        print(f"  attention {mode:<10}: {w:7.2f} ms")
    print()
    print(f"  joint vs decoupled per-op: "
          f"{walls['decoupled'] / walls['joint']:.2f}×")
    print()

    # Block wall with substituted attention
    print(f"## Block-level wall (under serial-runtime, today's path)\n")
    print("| attention mode | attention ms | block serial ms | block concurrent ms | "
          "block savings vs decoupled |")
    print("|---|---:|---:|---:|---:|")
    decoupled_serial, decoupled_conc = block_wall_with_attn_wall(
        cfg, M, walls['decoupled'])
    for mode in ["serial", "decoupled", "joint"]:
        s_wall, c_wall = block_wall_with_attn_wall(cfg, M, walls[mode])
        savings_serial = decoupled_serial - s_wall
        pct_serial = savings_serial / decoupled_serial * 100 if decoupled_serial else 0
        print(f"| {mode} | {walls[mode]:.2f} | {s_wall:.2f} | "
              f"{c_wall:.2f} | {pct_serial:+.1f}% |")
    print()

    # Headline interpretation
    s_serial, _ = block_wall_with_attn_wall(cfg, M, walls['serial'])
    d_serial, _ = block_wall_with_attn_wall(cfg, M, walls['decoupled'])
    j_serial, _ = block_wall_with_attn_wall(cfg, M, walls['joint'])
    print("## Block-wall savings\n")
    print(f"  joint vs decoupled (serial runtime):  "
          f"{d_serial - j_serial:+.2f} ms ({(d_serial - j_serial) / d_serial * 100:.1f}%)")
    print(f"  joint vs serial (per-op + serial):    "
          f"{s_serial - j_serial:+.2f} ms ({(s_serial - j_serial) / s_serial * 100:.1f}%)")

    return 0


def run_sweep() -> int:
    print("# Sweep — joint SWP+WS attention savings flowed to block wall\n")
    print("| model | M | SFP | attn dec ms | attn joint ms | block dec ms | "
          "block joint ms | block savings | block savings % |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model_key in ["llama_70b", "dsv3"]:
        cfg = MODELS[model_key]
        for M in [128, 2048]:
            for sfp in [1500, 3000, 5000]:
                w_dec = fa_wall_ms(M, cfg.n_heads, cfg.head_dim, sfp, "decoupled")
                w_jnt = fa_wall_ms(M, cfg.n_heads, cfg.head_dim, sfp, "joint")
                b_dec, _ = block_wall_with_attn_wall(cfg, M, w_dec)
                b_jnt, _ = block_wall_with_attn_wall(cfg, M, w_jnt)
                save = b_dec - b_jnt
                pct = save / b_dec * 100 if b_dec else 0
                print(f"| {cfg.name} | {M} | {sfp} | "
                      f"{w_dec:.2f} | {w_jnt:.2f} | {b_dec:.2f} | "
                      f"{b_jnt:.2f} | {save:.2f} ms | {pct:.1f}% |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
