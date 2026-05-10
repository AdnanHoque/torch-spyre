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

"""K2 — kf N-sweep at fixed M, K.

Question: above the 64 KB threshold, how does kf step scale with N
(and thus with cohort payload)? Granite probe gave us 64 KB (no
step), 128 KB (step), 400 KB (errored), 2 MB (step). This probe fills
in the 64-128 KB transition and extends to larger N to validate
linear scaling.

Shape: (128, N, 4096) at split (1, 16, 2). Each N gives a different
N_per = N/16 → cohort_payload = 128 × N/16 × 4 bytes.

N values:           N_per     cohort_payload
   1024              64           32 KB
   1536              96           48 KB
   2048             128           64 KB    ← threshold
   2560             160           80 KB
   3072             192           96 KB
   4096             256          128 KB    ← Granite q_proj equiv
   6144             384          192 KB
   8192             512          256 KB
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
    print("# Probe K2 — kf N-sweep at fixed M=128, K=4096, split (1,16,2)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()

    M, K = 128, 4096
    Ns = [1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192]
    split = (1, 16, 2)
    m, n, k_split = split

    print("| N | N_per | cohort_payload | id wall ms | kf wall ms | step ms |")
    print("|---:|---:|---:|---:|---:|---:|")

    rows = []
    for N in Ns:
        N_per = N // n
        if N_per * n != N:
            continue
        if N_per % 8 != 0:  # need 8-divisible for stick alignment
            continue
        payload = M * N_per * 4

        id_perm = perm_identity()
        kf_perm = perm_kfast(m, n, k_split)

        id_ms, id_err = measure(M, N, K, split, id_perm)
        kf_ms, kf_err = measure(M, N, K, split, kf_perm)

        if id_ms is None or kf_ms is None:
            err = id_err or kf_err
            print(f"| {N} | {N_per} | {payload / 1024:.0f} KB | ERR ({err}) |")
            sys.stdout.flush()
            continue

        step = id_ms - kf_ms
        print(
            f"| {N} | {N_per} | {payload / 1024:.0f} KB | {id_ms:.3f} | "
            f"{kf_ms:.3f} | {step:.3f} |"
        )
        sys.stdout.flush()
        rows.append((N, N_per, payload, id_ms, kf_ms, step))

    # Identify threshold and slope
    print("\n## Threshold + slope analysis")
    print()
    if rows:
        # Find threshold: smallest payload where step > 0.05 ms
        threshold_payload = None
        for r in rows:
            if r[5] > 0.05:
                threshold_payload = r[2]
                break
        if threshold_payload is not None:
            print(
                f"  threshold (first step > 0.05 ms): "
                f"{threshold_payload / 1024:.0f} KB payload"
            )
        # Linear fit on shapes ABOVE threshold
        above = [r for r in rows if r[5] > 0.05]
        if len(above) >= 2:
            xs = [r[2] for r in above]
            ys = [r[5] for r in above]
            n_pts = len(xs)
            sx = sum(xs)
            sy = sum(ys)
            sxx = sum(x * x for x in xs)
            sxy = sum(x * y for x, y in zip(xs, ys))
            denom = n_pts * sxx - sx * sx
            if denom != 0:
                slope = (n_pts * sxy - sx * sy) / denom
                intercept = (sy - slope * sx) / n_pts
                print(
                    f"  above-threshold linear fit: "
                    f"step ≈ {intercept:.3f} ms + payload × "
                    f"{slope * 1024 * 1024:.4f} ms/MB"
                )
                bw_GBps = (1.0 / slope) / 1e9 if slope > 0 else float("inf")
                print(
                    f"  effective BW from slope = {bw_GBps:.2f} GB/s "
                    f"(vs SFP peak 70.4 GB/s)"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
