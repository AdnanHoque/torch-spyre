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

"""Phase 0 Path B (FA variant): joint SWP+WS prototype for FlashAttention.

Companion to joint_swp_ws_ilp_prototype.py. Models FA's per-iteration
structure, which is the canonical Twill-style workload:

    iter i:  HMI[i] (K+V tile) → PT_QK[i] → SFP_softmax[i]
                                          → PT_OV[i] → SFP_update[i]

Crucially, two PT stages per iter (different matmuls, no PSUM dep
between them) and two SFP stages per iter. The "ping-pong" pattern
is: iter i+1's PT_QK runs in parallel with iter i's SFP_softmax,
while iter i's PT_OV runs in parallel with iter i+1's SFP_softmax.

Cross-iter dep: PT_OV[i+1] consumes O updated by PT_OV[i] (and
rescaled by SFP_update[i]). Serialized via PT and SFP no-overlap.

Cycle-count estimates per tile (B_r = B_c = 128, head_dim = 128):
  HMI fetch K+V:    ~1700 cycles  (32KB × 2 / 40 GB/s)
  PT Q·K^T:         ~4096 cycles  (128·128·128 / 512 lanes)
  SFP softmax:      varies — exp dominates. 1500 (4cyc/exp) to
                    5000 (16cyc/exp), depending on AIU SFP design.
  PT (P·V):         ~4096 cycles  (same as QK)
  SFP update:        ~300 cycles

Tests three modes (serial / decoupled / joint) across a sweep of
SFP softmax cost. Output: at what SFP cost does ping-pong matter,
and how big is the joint advantage on top?

Usage:
    python tests/joint_swp_ws_fa_prototype.py
    python tests/joint_swp_ws_fa_prototype.py --iters 32 --sfp 3000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ortools.sat.python import cp_model


# Default AIU cycle estimates per FA tile (B_r=B_c=d=128, fp16, 1 GHz)
HMI_CYC = 1700
PT_QK_CYC = 4096
PT_OV_CYC = 4096
SFP_UPDATE_CYC = 300


@dataclass
class SimResult:
    mode: str
    wall: int
    solve_time: float


def solve_fa(num_iters: int, sfp_softmax_cyc: int, mode: str,
             time_limit: float = 30.0) -> SimResult:
    """Solve FA scheduling under given mode and SFP softmax cost."""
    model = cp_model.CpModel()
    horizon = (HMI_CYC + PT_QK_CYC + sfp_softmax_cyc + PT_OV_CYC
               + SFP_UPDATE_CYC) * num_iters + 1000

    # Stages: (name, unit, duration). Two PT + two SFP per iter.
    stages = [
        ("hmi",         "HMI", HMI_CYC),
        ("pt_qk",       "PT",  PT_QK_CYC),
        ("sfp_softmax", "SFP", sfp_softmax_cyc),
        ("pt_ov",       "PT",  PT_OV_CYC),
        ("sfp_update",  "SFP", SFP_UPDATE_CYC),
    ]
    n_stages = len(stages)

    # Decision variables: start time of (iter, stage)
    starts = {}
    intervals = {}
    for i in range(num_iters):
        for s in range(n_stages):
            v = model.new_int_var(0, horizon, f"start_{i}_{s}")
            iv = model.new_interval_var(v, stages[s][2],
                                        v + stages[s][2], f"iv_{i}_{s}")
            starts[(i, s)] = v
            intervals[(i, s)] = iv

    # Intra-iteration deps: stage s starts after stage s-1 ends
    for i in range(num_iters):
        for s in range(1, n_stages):
            model.add(starts[(i, s)] >= starts[(i, s - 1)] + stages[s - 1][2])

    # Cross-iter: PT_OV[i+1] depends on SFP_update[i] (running stats).
    # In serial mode we'll force a stronger ordering.
    pt_ov_idx = next(s for s, st in enumerate(stages) if st[0] == "pt_ov")
    sfp_upd_idx = next(s for s, st in enumerate(stages) if st[0] == "sfp_update")
    for i in range(num_iters - 1):
        model.add(starts[(i + 1, pt_ov_idx)]
                  >= starts[(i, sfp_upd_idx)] + stages[sfp_upd_idx][2])

    # Per-unit no-overlap (HMI, PT, SFP). PT runs both QK and OV; SFP
    # runs both softmax and update. Each unit handles one task at a time.
    for u in ["HMI", "PT", "SFP"]:
        unit_intervals = [intervals[(i, s)] for i in range(num_iters)
                          for s in range(n_stages) if stages[s][1] == u]
        model.add_no_overlap(unit_intervals)

    # Mode-specific constraints
    if mode == "serial":
        # Each iter fully completes before next starts on any unit.
        for i in range(num_iters - 1):
            model.add(starts[(i + 1, 0)]
                      >= starts[(i, n_stages - 1)] + stages[-1][2])
    elif mode == "decoupled":
        # Per-unit greedy in iter order: same-unit tasks in iter
        # order with no cross-iter overlap on the same unit. The HMI
        # of iter i+1 cannot start until HMI of iter i is done — but
        # other units can overlap freely. This approximates a per-unit
        # SWP scheduler.
        for u in ["HMI", "PT", "SFP"]:
            unit_tasks = sorted(
                [(i, s) for i in range(num_iters)
                 for s in range(n_stages) if stages[s][1] == u],
                key=lambda x: (x[0], x[1]),
            )
            for k in range(len(unit_tasks) - 1):
                i_k, s_k = unit_tasks[k]
                i_k1, s_k1 = unit_tasks[k + 1]
                model.add(starts[(i_k1, s_k1)]
                          >= starts[(i_k, s_k)] + stages[s_k][2])
    # else 'joint': no extra constraints

    # Objective
    last_end = model.new_int_var(0, horizon, "last_end")
    for i in range(num_iters):
        model.add(last_end >= starts[(i, n_stages - 1)] + stages[-1][2])
    model.minimize(last_end)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)
    wall = solver.value(last_end) if status in (cp_model.OPTIMAL,
                                                 cp_model.FEASIBLE) else -1
    return SimResult(mode=mode, wall=wall, solve_time=solver.wall_time)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=16,
                        help="number of K/V tile iterations")
    parser.add_argument("--sfp", type=int, default=1500,
                        help="SFP softmax cycles per tile (1500 = 4-cyc/exp; "
                             "5000 = 16-cyc/exp)")
    parser.add_argument("--sweep", action="store_true",
                        help="sweep SFP softmax cost ∈ {500, 1500, 3000, "
                             "5000, 8000} to map balance regimes")
    parser.add_argument("--time-limit", type=float, default=30.0)
    args = parser.parse_args()

    print("# Joint SWP+WS — FlashAttention prototype\n")
    print(f"## Constants (per tile)\n")
    print(f"  HMI K+V fetch:   {HMI_CYC} cycles")
    print(f"  PT Q·K^T:        {PT_QK_CYC} cycles")
    print(f"  PT (P·V):        {PT_OV_CYC} cycles")
    print(f"  SFP softmax:     varies (probe arg)")
    print(f"  SFP update:      {SFP_UPDATE_CYC} cycles\n")

    if args.sweep:
        print("## Sweep over SFP softmax cost (iters={})\n".format(args.iters))
        print("| SFP softmax | PT/SFP ratio | serial | decoupled | joint | "
              "joint solve s | dec→joint speedup | ser→dec speedup |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        pt_per_iter = PT_QK_CYC + PT_OV_CYC
        for sfp in [500, 1500, 3000, 5000, 8000]:
            sfp_per_iter = sfp + SFP_UPDATE_CYC
            ratio = pt_per_iter / sfp_per_iter
            ser = solve_fa(args.iters, sfp, "serial", args.time_limit)
            dec = solve_fa(args.iters, sfp, "decoupled", args.time_limit)
            jnt = solve_fa(args.iters, sfp, "joint", args.time_limit)
            sp_sd = ser.wall / dec.wall if dec.wall else 0
            sp_dj = dec.wall / jnt.wall if jnt.wall else 0
            print(f"| {sfp} | {ratio:.2f}× | {ser.wall} | {dec.wall} | "
                  f"{jnt.wall} | {jnt.solve_time:.2f} | {sp_dj:.2f}× | "
                  f"{sp_sd:.2f}× |")
        return 0

    # Single-config run
    print(f"## Single config: SFP softmax = {args.sfp} cycles, "
          f"iters = {args.iters}\n")
    pt_per_iter = PT_QK_CYC + PT_OV_CYC
    sfp_per_iter = args.sfp + SFP_UPDATE_CYC
    print(f"  PT total per iter:  {pt_per_iter} cycles")
    print(f"  SFP total per iter: {sfp_per_iter} cycles")
    print(f"  PT/SFP balance:     {pt_per_iter / sfp_per_iter:.2f}× "
          f"({'PT-dominant' if pt_per_iter > sfp_per_iter * 1.2 else 'balanced' if abs(pt_per_iter - sfp_per_iter) < sfp_per_iter * 0.2 else 'SFP-dominant'})\n")

    results = {}
    for mode in ["serial", "decoupled", "joint"]:
        r = solve_fa(args.iters, args.sfp, mode, args.time_limit)
        results[mode] = r
        print(f"  {mode:<10} wall = {r.wall:>6} cycles, "
              f"solve = {r.solve_time:.3f} s")
    print()
    j = results["joint"].wall
    print("## Speedups\n")
    print(f"  joint vs serial:    {results['serial'].wall / j:.2f}×")
    print(f"  joint vs decoupled: {results['decoupled'].wall / j:.2f}×")
    print(f"  decoupled vs serial: "
          f"{results['serial'].wall / results['decoupled'].wall:.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
