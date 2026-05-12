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

"""Probe — discriminating test for per-K-iteration overhead theory.

Hypothesis under test: each K-iteration carries a fixed per-iter
overhead (instruction stream, sync barrier, HBM block setup, etc.),
so wall time scales linearly with K-iter count even when total work
is constant.

Test design:
  Hold total ops (2·M·N·K) and total HBM bytes approximately
  constant. Vary only K (with N inversely) so K-iter count is the
  ONLY thing that changes between data points.

  M=128 fixed.
  K=1024  N=8192:  compute=2.15G,  K-iters_per_kernel=128
  K=2048  N=4096:  compute=2.15G,  K-iters_per_kernel=256
  K=4096  N=2048:  compute=2.15G,  K-iters_per_kernel=512
  K=8192  N=1024:  compute=2.15G,  K-iters_per_kernel=1024

  Total HBM bytes vary minimally (A+B+C goes 18→17→17→18 MB), and
  the A vs C trade keeps things balanced.

Predictions:
  Per-K-iter overhead dominant:  wall(K=8192) ≈ 8× wall(K=1024)
  Compute-bound (constant work): wall ≈ same across shapes
  HBM-bound (constant bytes):    wall ≈ same across shapes
  Mixed:                          intermediate slope

Tested with split (1, 16, 2) k_fast — the same split used in our
production-shape measurements where K-split-and-bichain are engaged.
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
ITERS = 10
DTYPE = torch.float16
PT_PEAK_FP16 = 72.1e12
HBM_PEAK = 166e9

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


M = 128
# Each row keeps M·N·K constant (2.15G ops) by varying K and N inversely.
SHAPES = [
    (1024, 8192),
    (2048, 4096),
    (4096, 2048),
    (8192, 1024),
]

# Test under two splits: a k=1 (single corelet path) and a k=2 (bichain path).
SPLITS = [
    (4, 8, 1),    # mixed-MN k=1 — best for many shapes
    (1, 16, 2),   # K-split k=2 — engages bichain
]


def main():
    print("# Probe — per-K-iteration overhead discriminating test")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print(f"M={M} fixed. K varies 1024→8192 (8×) while N varies inversely")
    print("(8192→1024) to keep 2·M·N·K constant at 2.15G ops.")
    print("HBM bytes also approximately constant (within ~5%).")
    print()
    print("Test: if wall ∝ K, per-K-iter overhead is real.")
    print("      if wall ≈ const, the K-iter overhead theory is FALSIFIED.")
    print()

    for split in SPLITS:
        m, n, k = split
        print(f"## Split: {split}")
        print()
        print("| K | N | compute G·ops | HBM MB | K-iters/core | wall ms | "
              "ratio vs K=1024 |")
        print("|---:|---:|---:|---:|---:|---:|---:|")

        base_wall = None
        for K, N in SHAPES:
            wall_ms, err = measure(M, N, K, split)
            ops_g = 2 * M * N * K / 1e9
            hbm_mb = (M*K + K*N + k*M*N) * 2 / 1024 / 1024
            kiters_per_core = K // k // 8  # K-iters per core (PT step = 8)
            if wall_ms is None:
                print(f"| {K} | {N} | {ops_g:.2f} | {hbm_mb:.1f} | "
                      f"{kiters_per_core} | ERR ({err}) | — |")
                continue
            if base_wall is None:
                base_wall = wall_ms
                ratio_str = "1.00×"
            else:
                ratio_str = f"{wall_ms/base_wall:.2f}×"
            print(f"| {K} | {N} | {ops_g:.2f} | {hbm_mb:.1f} | "
                  f"{kiters_per_core} | {wall_ms:.3f} | {ratio_str} |")
            sys.stdout.flush()

        print()

    print("## Interpretation")
    print()
    print("Under the per-K-iter overhead theory, wall should scale ~8× from")
    print("K=1024 → K=8192 even though compute and HBM are constant.")
    print()
    print("If wall ratio at K=8192 vs K=1024 is:")
    print("  ≥ 5×  : per-K-iter overhead is the dominant cost. Theory")
    print("          STRONGLY SUPPORTED.")
    print("  2-5×  : K-iter cost is meaningful but not dominant. Mixed model.")
    print("  ≤ 1.5×: per-K-iter overhead is small. Theory FALSIFIED — wall")
    print("          is governed by compute, HBM, or other shape-invariant")
    print("          factors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
