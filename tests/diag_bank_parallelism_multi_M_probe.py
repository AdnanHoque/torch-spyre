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

"""Probe — bank parallelism falsification, extended to M=32/128/512.

The single-shape M=32 probe (diag_bank_parallelism_probe.py) showed
wall time is FLAT across B-distinct from 1 to 32 — the bank-parallelism
mechanism does not predict wall time on that shape.

This probe extends the test to two more M values (128 and 512) to
confirm the flatness holds across shapes, and to reconcile with the
head-to-head probe data that showed an apparent 16% gap between K-split
and mixed-MN on M=128 (which we suspect was a permutation effect, not
a split-intrinsic property).

For each M, hold N=K=4096, vary the split to span B-distinct from 1
to 32. All runs use k_fast permutation uniformly so the comparison
isn't confounded by permutation choice.
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

from torch_spyre._inductor import config as ts_config  # noqa: E402
from torch_spyre._inductor.codegen import compute_ops as _co  # noqa: E402

WARMUP = 2
ITERS = 8
DTYPE = torch.float16

_orig_multi = _planner.multi_dim_iteration_space_split
_orig_kfast_perm = _co._k_fast_core_id_permutation


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


def perm_kfast(m, n, k):
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(m * n * k)]


@contextmanager
def _force_perm(perm_func, split):
    m, n, k = split
    perm = perm_func(m, n, k)

    def _patched(num_cores, work_slices):
        if num_cores != m * n * k:
            return _orig_kfast_perm(num_cores, work_slices)
        return list(perm)

    _co._k_fast_core_id_permutation = _patched
    prev = ts_config.core_id_k_fast_emission
    ts_config.core_id_k_fast_emission = True
    try:
        yield
    finally:
        _co._k_fast_core_id_permutation = _orig_kfast_perm
        ts_config.core_id_k_fast_emission = prev


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
        with _force_perm(perm_kfast, split), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_perm(perm_kfast, split), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


SPLITS = [
    ((32, 1, 1),  1,  32),
    ((16, 2, 1),  2,  16),
    ((8, 4, 1),   4,   8),
    ((4, 8, 1),   8,   4),
    ((2, 16, 1), 16,   2),
    ((1, 32, 1), 32,   1),
    ((1, 16, 2), 32,   2),
    ((1, 8, 4),  32,   4),
    ((1, 4, 8),  32,   8),
    ((1, 2, 16), 32,  16),
    ((1, 1, 32), 32,  32),
]


def run_shape(M, N, K):
    print(f"## Shape: ({M}, {N}, {K})")
    print()

    A_bytes = M * K * 2
    B_bytes = K * N * 2
    C_bytes = M * N * 2

    print(f"  A = {A_bytes/1024/1024:.2f} MB, B = {B_bytes/1024/1024:.2f} MB, "
          f"C = {C_bytes/1024/1024:.2f} MB")
    print()
    print("| split | B-dist | A-dist | wall ms | agg BW est |")
    print("|---|---:|---:|---:|---:|")

    rows = []
    for split, b_dist, a_dist in SPLITS:
        m, n, k = split
        wall_ms, err = measure(M, N, K, split)
        if wall_ms is None:
            print(f"| {split} | {b_dist} | {a_dist} | ERR | — |")
            continue
        total_bytes = A_bytes + B_bytes + k * C_bytes
        agg_bw_gbs = (total_bytes / 1e9) / (wall_ms / 1e3)
        print(f"| {split} | {b_dist} | {a_dist} | {wall_ms:.3f} | "
              f"{agg_bw_gbs:.0f} GB/s |")
        sys.stdout.flush()
        rows.append((split, b_dist, a_dist, wall_ms, agg_bw_gbs))

    print()
    walls = [r[3] for r in rows]
    if walls:
        spread = max(walls) - min(walls)
        spread_pct = spread / min(walls) * 100
        print(f"  Spread across splits: {min(walls):.3f} → {max(walls):.3f} ms "
              f"({spread_pct:.1f}%)")
    print()


def main():
    print("# Probe — bank parallelism across M=32/128/512")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print("All runs use k_fast permutation uniformly.")
    print()
    print("If wall is flat across B-distinct values for each shape, the")
    print("bank-parallelism mechanism is empirically falsified across the M")
    print("decade. If wall varies more on larger M, the mechanism is shape-")
    print("dependent — possibly real at M=128/512 but not at M=32.")
    print()

    for M in [32, 128, 512]:
        run_shape(M, 4096, 4096)

    print("## Summary")
    print()
    print("Compare spread% across M values:")
    print("  - All under 5%: bank parallelism is NOT the mechanism on these shapes.")
    print("  - Spread grows with M: mechanism is shape-dependent.")
    print("  - Spread large at one M but flat at others: mechanism is regime-")
    print("    specific. Need to identify what's different.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
