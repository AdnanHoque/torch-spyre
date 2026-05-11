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

"""Probe — heterogeneous K-cohort hypothesis test.

Hypothesis: K-cohort cores today have uniform K/k slices. If load is
naturally imbalanced (e.g., per-core memory locality differs, or some
slices underflow PT SIMD width), wall time vs k should be non-monotonic
or have discontinuities. That would indicate heterogeneous K-cohort
sizing could help.

If wall scales smoothly with k (no discontinuities), uniform slicing
is already well-balanced → heterogeneous K has no upside.

Vary k from 1 to 32 (keeping m·n·k = 32) on shapes spanning K=64
(stress test, K/16 and K/32 underflow PT SIMD), K=512 (moderate),
K=4096 (typical decoder).
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

WARMUP = 2
ITERS = 8
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


# Shapes: M=128 (constant compute target), N=4096 (constant), vary K
SHAPES = [
    ("K=64 (stress)",  128, 4096,   64),
    ("K=512",          128, 4096,  512),
    ("K=4096",         128, 4096, 4096),
]

# Splits where m=1 (pure K-cohort family), n varies, k varies — all m·n·k=32.
SPLITS = [
    (1, 32, 1),   # k=1, K_per=K
    (1, 16, 2),   # k=2, K_per=K/2
    (1, 8, 4),    # k=4, K_per=K/4
    (1, 4, 8),    # k=8, K_per=K/8
    (1, 2, 16),   # k=16, K_per=K/16
    (1, 1, 32),   # k=32, K_per=K/32
]


def main():
    print("# Probe — heterogeneous K-cohort hypothesis test")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("For each shape, vary k from 1 to 32 with m=1 (pure K-cohort family).")
    print("If wall(k) is smooth: K-slicing is uniform; hetero K has no upside.")
    print("If wall(k) has discontinuities: imbalance exists; hetero K could help.")
    print()

    for label, M, N, K in SHAPES:
        print(f"## {label} (M={M}, N={N}, K={K})")
        print()
        print("| split | k | K_per (elems) | wall ms | note |")
        print("|---|---:|---:|---:|---|")

        prev_wall = None
        for split in SPLITS:
            m, n, k = split
            K_per = K // k
            note = ""
            if K_per < 8:
                note = "K_per < 8 → SIMD underflow"
            elif K_per == 8:
                note = "K_per = 8 → SIMD just fits"

            wall_ms, err = measure(M, N, K, split)
            if wall_ms is None:
                print(f"| {split} | {k} | {K_per} | ERR ({err}) | — |")
                continue

            ratio = ""
            if prev_wall is not None:
                pct = (wall_ms - prev_wall) / prev_wall * 100
                ratio = f"Δ vs prev: {pct:+.1f}%"

            print(f"| {split} | {k} | {K_per} | {wall_ms:.3f} | "
                  f"{note}{(' / ' + ratio) if note and ratio else ratio} |")
            sys.stdout.flush()
            prev_wall = wall_ms
        print()

    print("## Interpretation")
    print()
    print("- Look for non-monotonic wall(k) or sharp jumps at K_per=8 boundary.")
    print("- Smooth scaling → uniform K-slicing is fine; hetero K has no upside.")
    print("- Sharp drop at large k → small slices waste cycles; hetero K could")
    print("  potentially recover by giving some cores more K work.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
