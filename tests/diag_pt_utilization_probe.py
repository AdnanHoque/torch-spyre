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

"""Probe — PT M-row utilisation curve.

Question: how does wall time depend on M_per_core relative to the PT
array's 8-M-row width? When M_per < 8, PT rows sit idle and per-core
compute time should scale up by 8/M_per.

Probe design: pure-M split (32, 1, 1) on a moderate (N, K) so HBM
isn't dominant. Sweep M ∈ {8, 16, ..., 512} → M_per ∈ {0.25, ..., 16}.
Wall time vs M_per gives the utilisation curve. Below M_per = 8 the
curve should rise sharply (PT underfed); above 8, it should flatten.

Shape: M variable, N=2048, K=2048, fp16.
  Compute work = M·N·K·2 ops scales linearly with M.
  HBM cost ~ M·K + K·N + M·N. Mostly K·N = 8 MB (constant) + linear-in-M.

Expected pattern: per-core compute should scale ∝ max(1, 8/M_per),
plus constant HBM and overhead. So total wall ~ a·M·max(1, 8/M_per)/32 + b.
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
import torch._inductor.config as _icfg  # noqa: E402

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/adnan/dt-inductor/torch-spyre")

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402

try:
    from torch_spyre._inductor import work_division as _planner  # noqa: E402
except ImportError:
    from torch_spyre._inductor import core_division as _planner  # noqa: E402

WARMUP = 3
ITERS = 12
DTYPE = torch.float16

_orig_multi = _planner.multi_dim_iteration_space_split


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        if target[0] * target[1] * target[2] != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}

    return _forced


@contextmanager
def _force_split(target):
    if target is None:
        yield
        return
    _planner.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _planner.multi_dim_iteration_space_split = _orig_multi


def _bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    _ts.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def measure(M, N, K, split):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(split):
                mm(a, b)

        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    print("# Probe — PT M-row utilisation curve (pure-M, M sweep)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: N=2048, K=2048 fixed. M sweeps to vary M_per = M/32.")
    print("  M_per < 8  → PT under-utilised (only M_per/8 of M-rows used).")
    print("  M_per ≥ 8  → PT M-rows fully used; further M scales linearly.")
    print()
    print("Expected: wall ≈ a · M · max(1, 8/M_per) / 32 + b")
    print()

    N, K = 2048, 2048
    Ms = [32, 64, 128, 256, 512, 1024, 2048]
    split = (32, 1, 1)

    print("| M | M_per | PT M-util | expected scaling | wall ms | wall / M_per (norm) |")
    print("|---:|---:|---:|---|---:|---:|")

    rows = []
    for M in Ms:
        M_per = M / 32
        util = min(1.0, M_per / 8.0)
        if M_per < 8:
            scaling_factor = 8.0 / M_per
            scaling_str = f"underfed × {scaling_factor:.1f}"
        else:
            scaling_factor = 1.0
            scaling_str = "PT-saturated"
        wall_ms, err = measure(M, N, K, split)
        if wall_ms is None:
            print(f"| {M} | {M_per:.2f} | {util:.2f} | {scaling_str} | ERR | — |")
            continue
        wall_per_mac = wall_ms / max(M_per, 1.0)
        print(f"| {M} | {M_per:.2f} | {util:.2f} | {scaling_str} | "
              f"{wall_ms:.3f} | {wall_per_mac:.3f} |")
        sys.stdout.flush()
        rows.append((M, M_per, util, scaling_factor, wall_ms))

    print()
    print("## Slope analysis")
    print()
    print("Below M_per = 8, doubling M doubles total work but PT util")
    print("stays at M_per/8. Wall should rise ~linearly in M_per.")
    print("At M_per = 8 and above, PT is fully used; wall ~ linear in M.")
    print()

    under = [r for r in rows if r[1] < 8]
    over = [r for r in rows if r[1] >= 8]
    if len(under) >= 2 and len(over) >= 2:
        # under-fed slope: ms per unit M
        u_slope = (under[-1][4] - under[0][4]) / (under[-1][0] - under[0][0])
        o_slope = (over[-1][4] - over[0][4]) / (over[-1][0] - over[0][0])
        print(f"  underfed-region slope:   {u_slope*1000:.2f} μs/unit-M")
        print(f"  PT-saturated slope:      {o_slope*1000:.2f} μs/unit-M")
        if o_slope > 0:
            print(f"  ratio:                   {u_slope/o_slope:.2f}×")
            print()
            print("  Note: under-fed region work-per-cycle is 1/8 of saturated,")
            print("  so per-M wall in under-fed should be ~8× the saturated slope.")
            print("  Less than 8× = HBM/overhead dominates; more than 8× = unusual.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
