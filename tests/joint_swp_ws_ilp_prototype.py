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

"""Phase 0 Path B: OR-tools ILP prototype for joint SWP+WS scheduling.

Tests whether the math behind the "A5.9 9-way Twill" proposal is
tractable on representative AIU shapes. Models a K-tiled matmul as a
4-stage pipeline (HMI fetch → LX stage → PT compute → SFP post) over
N iterations, with per-unit no-overlap constraints and PT
serialization for PSUM accumulation. Compares three schedules:

  serial:    each iteration fully completes before the next starts.
             Models a hypothetical worst-case decoupled scheduler.
  decoupled: each unit takes its tasks in iteration order, with no
             cross-iteration overlap on the SAME unit. PT serializes
             via PSUM dep; HMI/LX/SFP do not pipeline across iters.
             Approximates today's AIU compiler.
  joint:     full SWP+WS — units pipeline across iterations subject
             to deps and PT serialization. The Twill formulation.

Outputs:
  - wall (cycles) under each schedule
  - solve time for the joint ILP (key feasibility signal)
  - per-(iter, stage) Gantt-chart trace for the joint schedule

Phase 0 exit gate questions:
  - Does the joint ILP solve in seconds or hours? (scaling)
  - Does joint vs decoupled show a meaningful gap? (upper bound)

Usage:
    python tests/joint_swp_ws_ilp_prototype.py
    python tests/joint_swp_ws_ilp_prototype.py --iters 32 --verbose
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from ortools.sat.python import cp_model


# Stage definition for one K-tile of a matmul.
# Cycle counts proportional to typical AIU workloads (will calibrate
# later from the cost model; for prototype scaling, ratios matter).
# Two profiles to test where the joint formulation wins:
#   - HMI-dominant:    HMI >> compute. Decode regime, q_proj M=128.
#   - compute-balanced: HMI ≈ PT ≈ SFP. Attention compute, M=2048.
STAGE_PROFILES = {
    # HMI-bound (decode regime — Llama 70B q_proj M=128)
    "hmi_dominant": [
        ("hmi_fetch_b", "HMI", 10),
        ("lx_stage",    "LX",   2),
        ("pt_compute",  "PT",   8),
        ("sfp_post",    "SFP",  3),
    ],
    # Compute-balanced (prefill regime — attention QK·V M=2048-ish)
    "compute_balanced": [
        ("hmi_fetch_b", "HMI",  4),
        ("lx_stage",    "LX",   1),
        ("pt_compute",  "PT",  10),
        ("sfp_post",    "SFP",  8),
    ],
}
STAGES = STAGE_PROFILES["hmi_dominant"]  # default; --profile overrides


@dataclass
class SimResult:
    mode: str
    wall: int
    solve_time_s: float
    sched: list[tuple]   # [(iter, stage_idx, start_cycle)]


def solve(num_iters: int, mode: str, time_limit_s: float = 30.0,
          ws_choice: bool = False) -> SimResult:
    """Solve the scheduling problem under the given mode.

    ws_choice=True adds a 5th 'post' stage (4 cycles) that can run on
    either PT or SFP per iteration — modelling the FA-3 ping-pong
    case where a 'reduce/scale' op can target either unit. With this
    enabled, joint scheduling can alternate the assignment per
    iteration; decoupled cannot.
    """
    model = cp_model.CpModel()
    n_stages = len(STAGES)
    extra_dur = 4  # WS-choice stage duration
    horizon = sum(s[2] for s in STAGES) * num_iters + extra_dur * num_iters + 200

    # Decision variables: start time of (iter, stage)
    starts = {}
    intervals = {}
    for i in range(num_iters):
        for s in range(n_stages):
            dur = STAGES[s][2]
            v = model.new_int_var(0, horizon, f"start_{i}_{s}")
            iv = model.new_interval_var(v, dur, v + dur, f"iv_{i}_{s}")
            starts[(i, s)] = v
            intervals[(i, s)] = iv

    # Intra-iteration: stage s starts after stage s-1 finishes
    for i in range(num_iters):
        for s in range(1, n_stages):
            model.add(starts[(i, s)]
                      >= starts[(i, s - 1)] + STAGES[s - 1][2])

    # PT serialization across iterations (PSUM accumulation: each iter's
    # PT depends on the previous iter's PT having finished).
    pt_idx = next(s for s, st in enumerate(STAGES) if st[1] == "PT")
    for i in range(num_iters - 1):
        model.add(starts[(i + 1, pt_idx)]
                  >= starts[(i, pt_idx)] + STAGES[pt_idx][2])

    # Per-unit no-overlap: each unit (HMI, LX, PT, SFP) processes one
    # task at a time. This is the WS half — each stage maps to one unit
    # and that unit is exclusive.
    units = sorted({st[1] for st in STAGES})
    for u in units:
        unit_intervals = [
            intervals[(i, s)]
            for i in range(num_iters)
            for s in range(n_stages)
            if STAGES[s][1] == u
        ]
        model.add_no_overlap(unit_intervals)

    # WS-choice stage: a 5th 'post' stage that can run on either PT or
    # SFP per iteration. Modelled with two optional intervals per iter,
    # exactly one active. This is where joint scheduling gets its
    # ping-pong leverage — pinned-stage workloads can't benefit.
    post_pt_iv = {}
    post_sfp_iv = {}
    post_pt_present = {}
    post_sfp_present = {}
    if ws_choice:
        for i in range(num_iters):
            # Optional intervals: present iff the boolean flag is true
            pt_p = model.new_bool_var(f"post_pt_{i}")
            sfp_p = model.new_bool_var(f"post_sfp_{i}")
            pt_start = model.new_int_var(0, horizon, f"post_pt_start_{i}")
            sfp_start = model.new_int_var(0, horizon, f"post_sfp_start_{i}")
            pt_iv = model.new_optional_interval_var(
                pt_start, extra_dur, pt_start + extra_dur, pt_p,
                f"post_pt_iv_{i}",
            )
            sfp_iv = model.new_optional_interval_var(
                sfp_start, extra_dur, sfp_start + extra_dur, sfp_p,
                f"post_sfp_iv_{i}",
            )
            post_pt_iv[i] = (pt_iv, pt_start, pt_p)
            post_sfp_iv[i] = (sfp_iv, sfp_start, sfp_p)
            post_pt_present[i] = pt_p
            post_sfp_present[i] = sfp_p
            # Exactly one of PT or SFP runs the post stage for iter i
            model.add(pt_p + sfp_p == 1)
            # Post stage must come after the PT compute stage of same iter
            model.add(pt_start
                      >= starts[(i, pt_idx)] + STAGES[pt_idx][2]).only_enforce_if(pt_p)
            model.add(sfp_start
                      >= starts[(i, pt_idx)] + STAGES[pt_idx][2]).only_enforce_if(sfp_p)
            # Post stage must come before SFP post (existing stage),
            # since we model post as the optional 'extra work' before
            # the standard SFP post step.
            sfp_idx = next(s for s, st in enumerate(STAGES) if st[1] == "SFP")
            model.add(starts[(i, sfp_idx)]
                      >= pt_start + extra_dur).only_enforce_if(pt_p)
            model.add(starts[(i, sfp_idx)]
                      >= sfp_start + extra_dur).only_enforce_if(sfp_p)
        # Add post intervals to per-unit no-overlap
        pt_extra_intervals = [post_pt_iv[i][0] for i in range(num_iters)]
        sfp_extra_intervals = [post_sfp_iv[i][0] for i in range(num_iters)]
        # Re-add no-overlap including the optional intervals
        all_pt = [intervals[(i, s)] for i in range(num_iters)
                  for s in range(n_stages) if STAGES[s][1] == "PT"] + pt_extra_intervals
        all_sfp = [intervals[(i, s)] for i in range(num_iters)
                   for s in range(n_stages) if STAGES[s][1] == "SFP"] + sfp_extra_intervals
        model.add_no_overlap(all_pt)
        model.add_no_overlap(all_sfp)

    # Mode-specific extra constraints
    if mode == "serial":
        # Iter i+1 cannot start ANY stage until iter i's last stage ends
        for i in range(num_iters - 1):
            model.add(starts[(i + 1, 0)]
                      >= starts[(i, n_stages - 1)] + STAGES[-1][2])
    elif mode == "decoupled":
        # Decoupled: per-unit greedy in iter order. For ws_choice, we
        # also force the post-stage assignment to be the same across
        # all iters (no per-iteration ping-pong). This is the WS half
        # of "decoupled".
        if ws_choice:
            for i in range(num_iters - 1):
                model.add(post_pt_present[i] == post_pt_present[i + 1])
        for u in units:
            unit_starts = sorted(
                [(i, s, starts[(i, s)]) for i in range(num_iters)
                 for s in range(n_stages) if STAGES[s][1] == u],
                key=lambda x: (x[0], x[1]),
            )
            for k in range(len(unit_starts) - 1):
                _, sk, vk = unit_starts[k]
                _, sk1, vk1 = unit_starts[k + 1]
                model.add(vk1 >= vk + STAGES[sk][2])
    # else "joint": no extra constraints — full SWP+WS

    # Objective: minimize last iteration's completion
    last_end = model.new_int_var(0, horizon, "last_end")
    for i in range(num_iters):
        model.add(last_end >= starts[(i, n_stages - 1)] + STAGES[-1][2])
    model.minimize(last_end)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.solve(model)
    wall = solver.value(last_end) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1
    sched = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(num_iters):
            for s in range(n_stages):
                sched.append((i, s, solver.value(starts[(i, s)])))
    return SimResult(mode=mode, wall=wall,
                     solve_time_s=solver.wall_time, sched=sched)


def render_gantt(result: SimResult, max_width: int = 80) -> str:
    """ASCII Gantt chart of the schedule, one row per unit."""
    if not result.sched:
        return "(no schedule)"
    units = sorted({STAGES[s][1] for _, s, _ in result.sched})
    end = result.wall
    scale = max_width / max(end, 1)
    lines = [f"  wall = {end} cycles, scale ~ {scale:.2f} chars/cycle"]
    for u in units:
        cols = [" "] * max(int(end * scale) + 1, 10)
        for i, s, start in result.sched:
            if STAGES[s][1] != u:
                continue
            dur = STAGES[s][2]
            sym = STAGES[s][0][0].upper()  # first letter of stage name
            x0 = int(start * scale)
            x1 = max(x0 + 1, int((start + dur) * scale))
            for x in range(x0, min(x1, len(cols))):
                cols[x] = sym
        lines.append(f"  {u:<5} |{''.join(cols)}|")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=16,
                        help="number of K-tile iterations")
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="ILP solve time limit (seconds)")
    parser.add_argument("--verbose", action="store_true",
                        help="print Gantt chart for joint mode")
    parser.add_argument("--scale", action="store_true",
                        help="sweep iters ∈ {4, 8, 16, 32, 64, 128} to test scaling")
    parser.add_argument("--ws-choice", action="store_true",
                        help="enable optional 5th 'post' stage that can run on "
                             "either PT or SFP — exposes the joint advantage")
    parser.add_argument("--profile", default="hmi_dominant",
                        choices=list(STAGE_PROFILES.keys()),
                        help="stage cycle profile (regime-dependent)")
    args = parser.parse_args()
    global STAGES
    STAGES = STAGE_PROFILES[args.profile]
    print(f"# Profile: {args.profile}\n")

    print("# Joint SWP+WS ILP prototype\n")
    print("Stages (per K-tile iteration):")
    for name, unit, dur in STAGES:
        print(f"  {name:<14} → {unit:<4}  duration = {dur} cycles")
    print()

    if args.scale:
        print("## Scaling sweep — does ILP solve in reasonable time?\n")
        print("| iters | serial wall | decoupled wall | joint wall | "
              "joint solve s | speedup vs serial | speedup vs decoupled |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for n in [4, 8, 16, 32, 64, 128]:
            ser = solve(n, "serial", args.time_limit, args.ws_choice)
            dec = solve(n, "decoupled", args.time_limit, args.ws_choice)
            jnt = solve(n, "joint", args.time_limit, args.ws_choice)
            sp_ser = ser.wall / jnt.wall if jnt.wall > 0 else 0
            sp_dec = dec.wall / jnt.wall if jnt.wall > 0 else 0
            print(f"| {n} | {ser.wall} | {dec.wall} | {jnt.wall} | "
                  f"{jnt.solve_time_s:.3f} | {sp_ser:.2f}× | {sp_dec:.2f}× |")
        return 0

    print(f"## Schedule comparison (iters={args.iters}, "
          f"ws_choice={args.ws_choice})\n")
    results = {}
    for mode in ["serial", "decoupled", "joint"]:
        r = solve(args.iters, mode, args.time_limit, args.ws_choice)
        results[mode] = r
        print(f"  {mode:<10} wall = {r.wall:>5} cycles, "
              f"solve = {r.solve_time_s:.3f} s")

    print()
    print("## Speedup of joint vs other schedules\n")
    j = results["joint"].wall
    for mode in ["serial", "decoupled"]:
        sp = results[mode].wall / j if j > 0 else 0
        print(f"  joint vs {mode}: {sp:.2f}×")
    print()

    if args.verbose:
        for mode in ["serial", "decoupled", "joint"]:
            print(f"## {mode} schedule\n")
            print(render_gantt(results[mode]))
            print()

    print("## Phase 0 readout\n")
    j_solve = results["joint"].solve_time_s
    sp_dec = results["decoupled"].wall / j if j > 0 else 0
    print(f"  ILP solve time at iters={args.iters}: {j_solve:.3f} s")
    print(f"  joint speedup vs decoupled (today's compiler approx): {sp_dec:.2f}×")
    if j_solve < 5 and sp_dec >= 1.15:
        print("  → SCALING OK + GAP IS REAL: continue feasibility study (scaling sweep next).")
    elif j_solve < 5:
        print("  → SCALING OK, GAP IS SMALL: project may close — verify with realistic durations.")
    else:
        print("  → ILP TOO SLOW: horizon decomposition mandatory; effort estimate creeps up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
