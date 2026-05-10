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

"""K1 — kf K-sweep on a fixed-shape family.

Question: does kf savings (id wall − kf wall) scale with K, given fixed
M and N? Granite probe showed q_proj M=128 (K=4096) step = 0.28 ms,
down_proj M=128 (K=12800) step = 0.85 ms — same payload, 3× step.
But Llama 70B q_proj M=128 (K=8192) step = 1.07 ms, 70B down_proj
M=128 (K=28672) step = 0.43 ms — same payload, OPPOSITE direction.

This probe varies K at a single fixed M, N, split to isolate K's
effect. Ten shapes total.

Shape family: (128, 4096, K) at split (1, 16, 2):
  cohort_payload = 128 × 256 × 4 = 128 KB (constant)
  K varies → expect kf savings to scale (or not) with K
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

WARMUP = 3
ITERS = 8
DTYPE = torch.float16


def perm_identity():
    return list(range(32))


def perm_kfast(m, n, k):
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(32)]


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


_orig_kfast_perm = _co._k_fast_core_id_permutation


@contextmanager
def _force_perm(perm):
    def _patched(num_cores, work_slices):
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


def measure(M, N, K, split, perm):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_perm(perm), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_perm(perm), _force_split(split):
                mm(a, b)

        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    print("# Probe K1 — kf K-sweep at fixed M=128, N=4096, split (1,16,2)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: (128, 4096, K). cohort_payload = 128 KB (constant).")
    print("Question: does kf step scale with K?")
    print()

    M, N = 128, 4096
    Ks = [1024, 2048, 4096, 8192, 16384, 32768]
    split = (1, 16, 2)
    m, n, k_split = split

    print("| K | id wall ms | kf wall ms | step ms | step/K (µs/K) |")
    print("|---:|---:|---:|---:|---:|")

    rows = []
    for K in Ks:
        id_perm = perm_identity()
        kf_perm = perm_kfast(m, n, k_split)

        id_ms, id_err = measure(M, N, K, split, id_perm)
        kf_ms, kf_err = measure(M, N, K, split, kf_perm)

        if id_ms is None or kf_ms is None:
            err = id_err or kf_err
            print(f"| {K} | ERR ({err}) |")
            sys.stdout.flush()
            continue

        step = id_ms - kf_ms
        step_per_K = step * 1000 / K  # µs per unit K
        print(f"| {K} | {id_ms:.3f} | {kf_ms:.3f} | {step:.3f} | {step_per_K:.4f} |")
        sys.stdout.flush()
        rows.append((K, id_ms, kf_ms, step))

    # Linear regression of step vs K
    print("\n## Step vs K regression")
    print()
    if len(rows) >= 2:
        Ks_x = [r[0] for r in rows]
        steps = [r[3] for r in rows]
        n_pts = len(rows)
        sx = sum(Ks_x)
        sy = sum(steps)
        sxx = sum(x * x for x in Ks_x)
        sxy = sum(x * y for x, y in zip(Ks_x, steps))
        denom = n_pts * sxx - sx * sx
        if denom != 0:
            slope = (n_pts * sxy - sx * sy) / denom
            intercept = (sy - slope * sx) / n_pts
            print(
                f"  step(K) ≈ {intercept * 1000:.2f} µs + K × {slope * 1000:.4f} µs/K"
            )
            print(f"  intercept (K-independent component) = {intercept * 1000:.2f} µs")
            print(f"  slope (per-K component) = {slope * 1000:.4f} µs/K")
            print()
            if abs(slope) < 1e-6:
                print("  → step is essentially K-independent")
            elif intercept < 0.05 * max(steps):
                print("  → step is dominated by K (linear-in-K)")
            else:
                print("  → step has both K-dependent and K-independent parts")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
