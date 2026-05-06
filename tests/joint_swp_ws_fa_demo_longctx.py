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

"""Long-context extension of joint_swp_ws_fa_demo.py.

The earlier demo at M=1024 showed reference attention beats even the
best joint+fused FA-tiled approach (17.8 ms vs 76.2 ms) because per-
tile launch floors overwhelm small-M HMI savings. This sweep measures
the crossover: at what M does FA tiling start winning?

For each M ∈ {1024, 2048, 4096, 8192}:
  1. Measure reference attention wall on AIU (full M×M materialized)
  2. Measure each FA-2 inner-loop op's wall on AIU
  3. Run the joint SWP+WS ILP under unfused / fused regimes
  4. Compare predicted FA wall to measured reference wall

Output: crossover analysis. Identifies M at which the joint+fused
FA approach becomes faster than reference materialized attention.

Usage:
    python tests/joint_swp_ws_fa_demo_longctx.py
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass

import torch
import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402

from ortools.sat.python import cp_model


N_HEADS = 8
HEAD_DIM = 128
K_TILE = 128
LAUNCH_FLOOR_MS = 3.0
WARMUP = 2
ITERS = 5  # smaller iters to keep total runtime manageable


@dataclass
class PerOpWalls:
    pt_qk: float
    sfp_amax: float
    sfp_where: float
    sfp_exp_p: float
    sfp_exp_r: float
    pt_pv: float
    sfp_acc_o: float
    sfp_sumexp: float
    sfp_acc_l: float


def _bench_compiled(fn, *args) -> float | None:
    torch._dynamo.reset()
    cfn = torch.compile(fn, dynamic=False)
    try:
        for _ in range(WARMUP): cfn(*args)
        _ts.synchronize()
        samples = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            cfn(*args)
            _ts.synchronize()
            samples.append(time.perf_counter() - t0)
        return statistics.median(samples) * 1e3
    except Exception:
        return None


def measure_reference(M: int) -> float | None:
    """Reference bmm + softmax + bmm wall."""
    q = torch.randn(1, N_HEADS, M, HEAD_DIM, dtype=torch.float16, device="spyre")
    k = torch.randn(1, N_HEADS, M, HEAD_DIM, dtype=torch.float16, device="spyre")
    v = torch.randn(1, N_HEADS, M, HEAD_DIM, dtype=torch.float16, device="spyre")

    SCALE = 1.0 / math.sqrt(HEAD_DIM)

    def ref(q, k, v):
        s = torch.matmul(q, k.transpose(-2, -1)) * SCALE
        p = torch.softmax(s, dim=-1)
        return torch.matmul(p, v)

    return _bench_compiled(ref, q, k, v)


def measure_per_op_walls(M: int) -> PerOpWalls:
    """Measure each FA-2 inner-loop op's wall at this M."""
    q = torch.randn(1, N_HEADS, M, HEAD_DIM, dtype=torch.float16, device="spyre")
    k = torch.randn(1, N_HEADS, M, HEAD_DIM, dtype=torch.float16, device="spyre")
    v = torch.randn(1, N_HEADS, M, HEAD_DIM, dtype=torch.float16, device="spyre")
    k_tile = k[..., :K_TILE, :].contiguous()
    v_tile = v[..., :K_TILE, :].contiguous()
    m_state = torch.full((1, N_HEADS, M, 1), -65000.0,
                         dtype=torch.float16, device="spyre")
    l_state = torch.zeros((1, N_HEADS, M, 1),
                          dtype=torch.float16, device="spyre")
    o_state = torch.zeros_like(q)
    SCALE = 1.0 / math.sqrt(HEAD_DIM)
    s_dummy = torch.randn(1, N_HEADS, M, K_TILE,
                          dtype=torch.float16, device="spyre")
    p_dummy = torch.randn(1, N_HEADS, M, K_TILE,
                          dtype=torch.float16, device="spyre")

    def op_pt_qk(q, k):
        return torch.matmul(q, k.transpose(-2, -1)) * SCALE

    def op_amax(s):
        return s.amax(dim=-1, keepdim=True)

    def op_where(m_t, m_s):
        return torch.where(m_t > m_s, m_t, m_s)

    def op_exp_p(s, m):
        return torch.exp(s - m)

    def op_exp_r(m_s, m_n):
        return torch.exp(m_s - m_n)

    def op_pt_pv(p, v):
        return torch.matmul(p, v)

    def op_acc_o(o, r, pv):
        return o * r + pv

    def op_sumexp(p):
        return p.sum(dim=-1, keepdim=True)

    def op_acc_l(l, r, sl):
        return l * r + sl

    return PerOpWalls(
        pt_qk=_bench_compiled(op_pt_qk, q, k_tile) or LAUNCH_FLOOR_MS,
        sfp_amax=_bench_compiled(op_amax, s_dummy) or LAUNCH_FLOOR_MS,
        sfp_where=_bench_compiled(op_where, m_state, m_state * 1.01) or LAUNCH_FLOOR_MS,
        sfp_exp_p=_bench_compiled(op_exp_p, s_dummy, m_state.expand(-1, -1, -1, K_TILE)) or LAUNCH_FLOOR_MS,
        sfp_exp_r=_bench_compiled(op_exp_r, m_state, m_state * 1.01) or LAUNCH_FLOOR_MS,
        pt_pv=_bench_compiled(op_pt_pv, p_dummy, v_tile) or LAUNCH_FLOOR_MS,
        sfp_acc_o=_bench_compiled(op_acc_o, o_state, m_state, torch.zeros_like(o_state)) or LAUNCH_FLOOR_MS,
        sfp_sumexp=_bench_compiled(op_sumexp, p_dummy) or LAUNCH_FLOOR_MS,
        sfp_acc_l=_bench_compiled(op_acc_l, l_state, m_state, l_state) or LAUNCH_FLOOR_MS,
    )


def unfused_iter_ops(walls: PerOpWalls):
    return [
        ("pt_qk",      walls.pt_qk,      "PT",  []),
        ("sfp_amax",   walls.sfp_amax,   "SFP", ["pt_qk"]),
        ("sfp_where",  walls.sfp_where,  "SFP", ["sfp_amax"]),
        ("sfp_exp_p",  walls.sfp_exp_p,  "SFP", ["sfp_where"]),
        ("sfp_exp_r",  walls.sfp_exp_r,  "SFP", ["sfp_where"]),
        ("pt_pv",      walls.pt_pv,      "PT",  ["sfp_exp_p"]),
        ("sfp_acc_o",  walls.sfp_acc_o,  "SFP", ["pt_pv", "sfp_exp_r"]),
        ("sfp_sumexp", walls.sfp_sumexp, "SFP", ["sfp_exp_p"]),
        ("sfp_acc_l",  walls.sfp_acc_l,  "SFP", ["sfp_sumexp", "sfp_exp_r"]),
    ]


def fused_iter_ops(walls: PerOpWalls):
    """One PT launch per matmul, one fused SFP launch per softmax block."""
    sfp_block_1 = [walls.sfp_amax, walls.sfp_where, walls.sfp_exp_p,
                   walls.sfp_exp_r, walls.sfp_sumexp]
    sfp_block_1_dur = LAUNCH_FLOOR_MS + sum(w - LAUNCH_FLOOR_MS for w in sfp_block_1)

    sfp_block_2 = [walls.sfp_acc_o, walls.sfp_acc_l]
    sfp_block_2_dur = LAUNCH_FLOOR_MS + sum(w - LAUNCH_FLOOR_MS for w in sfp_block_2)

    return [
        ("pt_qk",       walls.pt_qk,     "PT",  []),
        ("sfp_block_1", sfp_block_1_dur, "SFP", ["pt_qk"]),
        ("pt_pv",       walls.pt_pv,     "PT",  ["sfp_block_1"]),
        ("sfp_block_2", sfp_block_2_dur, "SFP", ["pt_pv"]),
    ]


def schedule(iter_ops, num_iters: int, mode: str,
             time_limit: float = 30.0) -> float:
    SCALE = 100  # cycle resolution
    ops_by_name = {n: (int(d * SCALE), u, deps) for n, d, u, deps in iter_ops}
    model = cp_model.CpModel()
    horizon = sum(d for d, _, _ in ops_by_name.values()) * num_iters + 1000
    starts = {}; intervals = {}
    for i in range(num_iters):
        for name in ops_by_name:
            dur, _, _ = ops_by_name[name]
            v = model.new_int_var(0, horizon, f"start_{i}_{name}")
            iv = model.new_interval_var(v, dur, v + dur, f"iv_{i}_{name}")
            starts[(i, name)] = v
            intervals[(i, name)] = iv
    for i in range(num_iters):
        for name in ops_by_name:
            dur, _, deps = ops_by_name[name]
            for d in deps:
                d_dur, _, _ = ops_by_name[d]
                model.add(starts[(i, name)] >= starts[(i, d)] + d_dur)
    units = sorted({u for _, u, _ in ops_by_name.values()})
    for u in units:
        unit_intervals = [intervals[(i, n)] for i in range(num_iters)
                          for n, (_, un, _) in ops_by_name.items() if un == u]
        model.add_no_overlap(unit_intervals)
    last_op = list(ops_by_name.keys())[-1]
    last_dur, _, _ = ops_by_name[last_op]
    first_op = list(ops_by_name.keys())[0]
    if mode == "decoupled":
        for u in units:
            unit_ops = [(i, name) for i in range(num_iters)
                        for name, (_, un, _) in ops_by_name.items() if un == u]
            for k in range(len(unit_ops) - 1):
                i_k, n_k = unit_ops[k]
                i_k1, n_k1 = unit_ops[k + 1]
                model.add(starts[(i_k1, n_k1)]
                          >= starts[(i_k, n_k)] + ops_by_name[n_k][0])
    elif mode == "serial":
        for i in range(num_iters - 1):
            model.add(starts[(i + 1, first_op)]
                      >= starts[(i, last_op)] + last_dur)
    last_end = model.new_int_var(0, horizon, "last_end")
    for i in range(num_iters):
        for name in ops_by_name:
            dur, _, _ = ops_by_name[name]
            model.add(last_end >= starts[(i, name)] + dur)
    model.minimize(last_end)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)
    return solver.value(last_end) / SCALE if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1


def main() -> int:
    print("# Joint SWP+WS — long-context FA-2 demonstration\n")
    print(f"# n_heads = {N_HEADS}, head_dim = {HEAD_DIM}, k_tile = {K_TILE}\n")

    M_VALUES = [1024, 2048, 4096, 8192]

    rows = []
    for M in M_VALUES:
        print(f"## M = {M}\n", flush=True)
        n_tiles = M // K_TILE
        print(f"  K-tiles per attention compute: {n_tiles}", flush=True)
        ref_ms = measure_reference(M)
        if ref_ms is None:
            print(f"  reference attention: ERR (compile failed)\n")
            continue
        print(f"  reference attention wall: {ref_ms:.2f} ms (measured)", flush=True)
        walls = measure_per_op_walls(M)
        print(f"  per-op walls (ms):", flush=True)
        print(f"    pt_qk={walls.pt_qk:.2f}  sfp_amax={walls.sfp_amax:.2f}  "
              f"sfp_where={walls.sfp_where:.2f}", flush=True)
        print(f"    sfp_exp_p={walls.sfp_exp_p:.2f}  sfp_exp_r={walls.sfp_exp_r:.2f}  "
              f"pt_pv={walls.pt_pv:.2f}", flush=True)
        print(f"    sfp_acc_o={walls.sfp_acc_o:.2f}  sfp_sumexp={walls.sfp_sumexp:.2f}  "
              f"sfp_acc_l={walls.sfp_acc_l:.2f}", flush=True)

        unfused_serial = schedule(unfused_iter_ops(walls), n_tiles, "serial")
        unfused_joint = schedule(unfused_iter_ops(walls), n_tiles, "joint")
        fused_serial = schedule(fused_iter_ops(walls), n_tiles, "serial")
        fused_joint = schedule(fused_iter_ops(walls), n_tiles, "joint")

        print(f"  unfused serial:   {unfused_serial:.2f} ms")
        print(f"  unfused joint:    {unfused_joint:.2f} ms")
        print(f"  fused serial:     {fused_serial:.2f} ms")
        print(f"  fused joint:      {fused_joint:.2f} ms  ← best FA-tiled prediction")
        ref_vs_fa = ref_ms / fused_joint
        wins = "FA wins" if fused_joint < ref_ms else "ref wins"
        print(f"  reference vs FA:  {ref_ms:.2f} vs {fused_joint:.2f} → {wins} ({ref_vs_fa:.2f}×)\n",
              flush=True)
        rows.append((M, ref_ms, unfused_serial, unfused_joint,
                     fused_serial, fused_joint))

    # Summary
    print("## Summary table\n")
    print("| M | ref ms | unfused serial | unfused joint | fused serial | "
          "fused joint | ref vs fused-joint |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for M, ref, us, uj, fs, fj in rows:
        verdict = f"FA {ref / fj:.2f}× win" if fj < ref else f"ref {fj / ref:.2f}× faster"
        print(f"| {M} | {ref:.2f} | {us:.2f} | {uj:.2f} | {fs:.2f} | "
              f"{fj:.2f} | {verdict} |")

    print()
    print("## Crossover analysis\n")
    crossover = None
    for M, ref, _, _, _, fj in rows:
        if fj < ref:
            crossover = M
            break
    if crossover is None:
        print("  No crossover within tested range — reference still wins at M=8192.")
        print("  FA tiling on AIU may need M ≥ 16K to break even.")
    else:
        print(f"  Crossover at M = {crossover}: FA-tiled (fused + joint scheduling)")
        print(f"  becomes faster than the reference materialized attention.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
