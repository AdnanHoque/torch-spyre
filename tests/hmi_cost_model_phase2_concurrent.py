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

"""Phase 2 of Project B: concurrent decoder-block simulator.

Phase 1 measured serial decoder-block wall time and computed an
upper-bound headroom (sum of HMI-bound op walls = best possible if
all non-HMI ops fully hide behind HMI). Phase 2 computes the
*realistic* headroom under the dependency graph: the gap between
serial wall and what a list-scheduler with HMI prefetch could
achieve.

Resource model:

    HMI (capacity 1):  one op fetches weights at a time, taking
                       t_hmi + LF (Phase 0 found LF stacks on HMI).
    PT  (capacity 1):  one op computes at a time, taking t_compute.
    PSUM:              additive on op completion, t_psum.

Within one op: HMI and PT run in parallel (kernel-template prefetch).
Cross-op: op N+1's HMI can start as soon as op N's HMI is done — this
is the prefetch overlap Phase 2 measures.

Two scheduling modes:

    serial:     each op fully completes (HMI + PT) before next op
                starts on either resource. Today's runtime.
    concurrent: HMI is pipelined across ops; PT is queued. Op N+1
                claims HMI as soon as op N's HMI finishes; PT once
                free. This is what runtime cross-bundle support
                would enable.

Output: per-(model, M) wall under both modes + headroom report.

Usage:
    python tests/hmi_cost_model_phase2_concurrent.py
    python tests/hmi_cost_model_phase2_concurrent.py --model llama_70b --m 128 --verbose
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.hmi_cost_model import LAUNCH_FLOOR_MS
from tests.hmi_cost_model_phase1_block import (
    MODELS,
    block_ops,
    predict_op,
)


# ---- op + dep graph ---------------------------------------------------

@dataclass
class SimOp:
    name: str
    kind: str
    t_hmi_with_lf: float  # HMI machine claim (includes LF)
    t_compute: float       # PT machine claim
    t_psum: float          # additive at completion
    deps: list[str]


# Decoder-block dep graph. Linear chain except for sibling matmuls and
# residual streams.
def build_dep_graph(op_names: list[str]) -> dict[str, list[str]]:
    """Return name -> list of dep names for the standard decoder block."""
    deps = {
        "input_rmsnorm": [],
        "q_proj": ["input_rmsnorm"],
        "kv_proj": ["input_rmsnorm"],
        "attn_qkt_softmax_v": ["q_proj", "kv_proj"],
        "o_proj": ["attn_qkt_softmax_v"],
        "post_attn_residual": ["o_proj"],  # also conceptually depends on input
        "post_attn_rmsnorm": ["post_attn_residual"],
        "gate_proj": ["post_attn_rmsnorm"],
        "up_proj": ["post_attn_rmsnorm"],
        "silu_mul": ["gate_proj", "up_proj"],
        "down_proj": ["silu_mul"],
        "post_mlp_residual": ["down_proj"],  # also depends on post_attn_residual
    }
    return {n: deps.get(n, []) for n in op_names}


def to_sim_ops(rows, deps_map: dict) -> list[SimOp]:
    """Convert Phase 1 predict_op rows into SimOp."""
    ops = []
    for r in rows:
        # Phase 1 t_hmi excludes LF; the wall formula adds LF on top of HMI.
        t_hmi_with_lf = r['t_hmi'] + LAUNCH_FLOOR_MS
        # PSUM is 0 for k=1 (planner-natural in this script).
        t_psum = r.get('t_psum', 0.0)
        ops.append(SimOp(
            name=r['name'], kind=r['kind'],
            t_hmi_with_lf=t_hmi_with_lf,
            t_compute=r['t_compute'],
            t_psum=t_psum,
            deps=deps_map.get(r['name'], []),
        ))
    return ops


# ---- simulator -------------------------------------------------------

@dataclass
class OpTrace:
    name: str
    hmi_start: float
    hmi_end: float
    compute_start: float
    compute_end: float
    op_end: float


@dataclass
class SimResult:
    mode: str
    total_wall: float
    traces: list[OpTrace]
    hmi_total: float
    compute_total: float


def simulate(ops: list[SimOp], mode: str) -> SimResult:
    """Two-resource discrete-event simulator.

    Picks ready ops (deps met) one at a time. Modes:
    - serial:     each op claims both resources for max(hmi, compute)
                  + psum, blocking everything else.
    - concurrent: HMI pipeline is independent of PT pipeline. Op N+1's
                  HMI starts when prev HMI finishes; compute starts
                  when own HMI started AND PT free AND deps done.

    Returns SimResult with per-op trace.
    """
    state_hmi_busy = 0.0
    state_pt_busy = 0.0
    op_ends: dict[str, float] = {}
    traces: list[OpTrace] = []
    pending = list(ops)

    while pending:
        ready = [o for o in pending if all(d in op_ends for d in o.deps)]
        if not ready:
            raise RuntimeError(f"dep deadlock; pending={[o.name for o in pending]}")
        # Pick the op whose deps complete earliest (greedy list-scheduler).
        ready.sort(key=lambda o: max([op_ends[d] for d in o.deps] or [0.0]))
        op = ready[0]
        deps_ready_at = max([op_ends[d] for d in op.deps] or [0.0])

        if mode == "serial":
            op_start = max(deps_ready_at, state_hmi_busy, state_pt_busy)
            hmi_start = op_start
            hmi_end = hmi_start + op.t_hmi_with_lf
            compute_start = op_start
            compute_end = compute_start + op.t_compute
            op_end = max(hmi_end, compute_end) + op.t_psum
            state_hmi_busy = op_end  # op fully completes before next on EITHER resource
            state_pt_busy = op_end
        else:  # concurrent
            hmi_start = max(deps_ready_at, state_hmi_busy)
            hmi_end = hmi_start + op.t_hmi_with_lf
            # Compute needs: own HMI to have started (data loading), PT free, deps done.
            compute_start = max(hmi_start, state_pt_busy, deps_ready_at)
            compute_end = compute_start + op.t_compute
            op_end = max(hmi_end, compute_end) + op.t_psum
            state_hmi_busy = hmi_end
            state_pt_busy = compute_end

        op_ends[op.name] = op_end
        traces.append(OpTrace(op.name, hmi_start, hmi_end,
                              compute_start, compute_end, op_end))
        pending.remove(op)

    total_wall = max(op_ends.values())
    hmi_total = sum(o.t_hmi_with_lf for o in ops)
    compute_total = sum(o.t_compute for o in ops)
    return SimResult(mode, total_wall, traces, hmi_total, compute_total)


# ---- main -----------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama_70b",
                        choices=list(MODELS.keys()))
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--verbose", action="store_true",
                        help="print per-op trace under both schedules")
    parser.add_argument("--all", action="store_true",
                        help="sweep all models × M ∈ {32, 128, 512, 2048}")
    args = parser.parse_args()

    if args.all:
        return run_all_sweep()

    return run_single(args.model, args.m, args.verbose)


def run_single(model_key: str, M: int, verbose: bool) -> int:
    cfg = MODELS[model_key]
    ops = block_ops(cfg, M)
    rows = [predict_op(op) for op in ops]
    deps_map = build_dep_graph([o.name for o in ops])
    sim_ops = to_sim_ops(rows, deps_map)

    serial = simulate(sim_ops, "serial")
    concurrent = simulate(sim_ops, "concurrent")

    print(f"# Phase 2 concurrent simulator: {cfg.name} block at M={M}\n")

    print("## Block-level wall\n")
    print(f"  serial:        {serial.total_wall:7.2f} ms")
    print(f"  concurrent:    {concurrent.total_wall:7.2f} ms")
    saved = serial.total_wall - concurrent.total_wall
    pct = saved / serial.total_wall * 100 if serial.total_wall > 0 else 0
    print(f"  saved:         {saved:7.2f} ms ({pct:.1f}%)")
    print()

    print("## Resource utilization (concurrent mode)\n")
    print(f"  Σ HMI claim time:    {concurrent.hmi_total:7.2f} ms  "
          f"({concurrent.hmi_total / concurrent.total_wall * 100:.0f}% of wall)")
    print(f"  Σ compute time:      {concurrent.compute_total:7.2f} ms  "
          f"({concurrent.compute_total / concurrent.total_wall * 100:.0f}% of wall)")
    print(f"  HMI lower bound:     {max(concurrent.hmi_total, concurrent.compute_total):7.2f} ms  "
          f"(no scheduler can do better than max)")
    print()

    if verbose:
        print("## Per-op trace (concurrent mode)\n")
        print("| op | hmi start | hmi end | compute start | compute end | done |")
        print("|---|---:|---:|---:|---:|---:|")
        for t in sorted(concurrent.traces, key=lambda x: x.hmi_start):
            print(f"| {t.name} | {t.hmi_start:6.2f} | {t.hmi_end:6.2f} | "
                  f"{t.compute_start:6.2f} | {t.compute_end:6.2f} | {t.op_end:6.2f} |")
        print()

    # Verdict per scope doc
    print("## Verdict\n")
    if pct < 5:
        verdict = "< 5% — HMI is essentially fully utilized. Project B closes."
    elif pct < 15:
        verdict = "5–15% — small headroom. Marginal; weigh complexity."
    else:
        verdict = "> 15% — substantial headroom. Pursue scheduling heuristic + runtime conversation."
    print(f"  {pct:.1f}% saving → {verdict}")
    return 0


def run_all_sweep() -> int:
    M_VALUES = [32, 128, 512, 2048]
    print("# Phase 2 sweep — scheduling headroom across (model, M)\n")
    print("| model | M | serial ms | concurrent ms | saved ms | saved % | verdict |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for model_key, cfg in MODELS.items():
        for M in M_VALUES:
            ops = block_ops(cfg, M)
            rows = [predict_op(op) for op in ops]
            deps_map = build_dep_graph([o.name for o in ops])
            sim_ops = to_sim_ops(rows, deps_map)
            s = simulate(sim_ops, "serial")
            c = simulate(sim_ops, "concurrent")
            saved = s.total_wall - c.total_wall
            pct = saved / s.total_wall * 100 if s.total_wall else 0
            verdict = ("close" if pct < 5
                       else "marginal" if pct < 15
                       else "pursue")
            print(f"| {cfg.name} | {M} | {s.total_wall:.2f} | "
                  f"{c.total_wall:.2f} | {saved:.2f} | {pct:.1f}% | {verdict} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
