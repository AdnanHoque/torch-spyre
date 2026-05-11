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

"""Probe — confirm B-weight is preloaded into LX (out of matmul critical path).

Question: doWeiPreload (deeptools/dsm/workOptimizer/baseOptimizer/weight_preload.cpp,
default ON per designSpaceConfig.h:75) is supposed to pre-position the
static B weight into each core's LX during model setup, removing B's
HBM cost from the matmul wall-time. Confirm empirically.

Shape choice: small A, LARGE B, small C → if B is preloaded, B's HBM
absence dominates; if not preloaded, B's HBM cost dominates and
splits should differ massively.

  M = 128, N = 8192, K = 2048, fp16
  A = M · K · 2 = 0.5 MB         ← small
  B = K · N · 2 = 32 MB          ← large; preload-test target
  C = M · N · 2 = 2 MB

Splits and predictions:

  (32, 1, 1) pure-M
    naive B traffic (no preload): 32 cores × 32 MB = 1024 MB
       → HBM time at 166 GB/s = 6.17 ms (just for B)
    preloaded B traffic during steady-state: 0 MB
       → matmul wall dominated by A, C, compute, overhead

  (1, 32, 1) pure-N
    naive B traffic: 1 × 32 MB = 32 MB
       → HBM time = 0.19 ms
    preloaded: 0 MB → same dominant terms as pure-M

  (4, 4, 2)
    naive B traffic: 4 × 32 MB = 128 MB
       → HBM time = 0.77 ms
    preloaded: 0 MB

If B preload is active:
  pure-M wall ≈ pure-N wall (B is not in the critical path for either)
  All three splits should be within ~0.5 ms of each other.

If B preload is NOT active:
  pure-M wall ≈ 6+ ms, pure-N ≈ 0.2 ms — huge 30× gap.

The wall times will tell us conclusively whether B is preloaded under
the default torch-spyre + deeptools config.
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
    print("# Probe — B-weight preload confirmation (small A, large B)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: M=128, N=8192, K=2048")
    print("  A = M·K·2 = 0.5 MB")
    print("  B = K·N·2 = 32 MB  (preload-test target)")
    print("  C = M·N·2 = 2 MB")
    print()

    M, N, K = 128, 8192, 2048

    splits = [
        ((32, 1, 1), "pure-M",    32, "naive B = 1024 MB"),
        ((1, 32, 1), "pure-N",    1, "naive B = 32 MB"),
        ((4, 4, 2),  "mixed",     4, "naive B = 128 MB"),
        ((8, 4, 1),  "mixed-MN",  8, "naive B = 256 MB"),
        ((4, 8, 1),  "mixed-MN",  4, "naive B = 128 MB"),
    ]

    print("| split | label | naive B per cluster | wall ms |")
    print("|---|---|---:|---:|")

    rows = []
    for split, label, m_factor, comment in splits:
        b_bytes = m_factor * K * N * 2
        b_bytes_mb = b_bytes / 1024 / 1024
        wall_ms, err = measure(M, N, K, split)
        if wall_ms is None:
            print(f"| {split} | {label} | {b_bytes_mb:.1f} MB | ERR ({err}) |")
            continue
        print(f"| {split} | {label} | {b_bytes_mb:.1f} MB | {wall_ms:.3f} |")
        sys.stdout.flush()
        rows.append((split, label, m_factor, b_bytes_mb, wall_ms))

    print()
    print("## Interpretation")
    print()
    if len(rows) >= 2:
        pure_m_wall = next(
            (r[4] for r in rows if r[0] == (32, 1, 1)), None
        )
        pure_n_wall = next(
            (r[4] for r in rows if r[0] == (1, 32, 1)), None
        )
        if pure_m_wall and pure_n_wall:
            naive_pure_m_hbm = 32 * K * N * 2 / 166e9 * 1000
            naive_pure_n_hbm = 1 * K * N * 2 / 166e9 * 1000
            gap = pure_m_wall - pure_n_wall
            print(f"  pure-M wall: {pure_m_wall:.3f} ms")
            print(f"  pure-N wall: {pure_n_wall:.3f} ms")
            print(f"  Δ (pure-M − pure-N): {gap:.3f} ms")
            print(f"  predicted Δ if B NOT preloaded: "
                  f"{naive_pure_m_hbm - naive_pure_n_hbm:.3f} ms")
            print()
            if pure_m_wall > 4.0 and pure_m_wall / pure_n_wall > 5.0:
                print("  → B is NOT preloaded; pure-M's full B reads dominate.")
            elif gap < 1.0:
                print("  → B IS preloaded; pure-M and pure-N similar")
                print("    (B not in steady-state HBM critical path).")
            else:
                print("  → Partial — B preload may be active for some splits but not others.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
