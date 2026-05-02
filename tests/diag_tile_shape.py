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

"""Tile-shape probe — Phase 1.3 data collection.

Phase 1.0 + per-axis analysis showed: at small K, pure n-split (1, 32, 1)
is fastest; at K=28672 (L3-70B MLP down), balanced (16, 2, 1) wins. The
v1 cost model can't explain the K-dependence with sharing factors alone.

Hypothesis: per-core compute throughput depends on the tile shape
`(M_per, N_per)`. At narrow-N tiles (e.g. N_per=256), the inner-K loop
runs less efficiently than at wide-N tiles (N_per=4096+). For small K
the DDR cost dominates so the n-split wins; for large K the compute
inefficiency at narrow-N dominates and m-split (which gives wider-N
tiles per core) wins.

This probe holds (M=128, N=8192) fixed and sweeps K ∈ {4096, 16384,
32768} — bracketing existing data's K range. For each K, it sweeps all
valid (m, n, 1) factorizations. Output: per-K (m, n) → wall ms table
that exposes the K → optimal-tile-shape trend cleanly.

Run: python tests/diag_tile_shape.py
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre
torch_spyre._autoload()  # ensure spyre device is registered
from torch_spyre import streams as _ts
from torch_spyre._inductor import core_division as _core_div


NUM_CORES = 32
STICK_ELEMS = 64  # fp16
WARMUP = 3
ITERS = 15


# ---- force-split monkey patch (same mechanism as diag_split_gap) -------

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target: tuple[int, int, int]):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        prod = 1
        for f in target:
            prod *= f
        if prod != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target: tuple[int, int, int]):
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)  # type: ignore[assignment]
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi  # type: ignore[assignment]


# ---- valid (m, n, 1) enumeration --------------------------------------

def _is_valid_mn1(M: int, N: int, K: int, m: int, n: int) -> bool:
    if M % m != 0:
        return False
    n_per = N // n
    if n_per < STICK_ELEMS or n_per % STICK_ELEMS != 0:
        return False
    if K < STICK_ELEMS or K % STICK_ELEMS != 0:
        return False
    return True


def _valid_factorizations_mn1(M: int, N: int, K: int) -> list[tuple[int, int]]:
    out = []
    for m in range(1, NUM_CORES + 1):
        if NUM_CORES % m != 0:
            continue
        n = NUM_CORES // m
        if not _is_valid_mn1(M, N, K, m, n):
            continue
        out.append((m, n))
    return out


# ---- bench primitive --------------------------------------------------

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


def _run(M: int, N: int, K: int, m: int, n: int) -> tuple[float | None, str]:
    a = torch.randn(M, K, dtype=torch.float16, device="spyre")
    b = torch.randn(K, N, dtype=torch.float16, device="spyre")

    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split((m, n, 1)):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split((m, n, 1)):
                mm(a, b)
        ms = _bench(step)
        return ms, ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


# ---- shapes -----------------------------------------------------------

@dataclass
class _Shape:
    M: int
    N: int
    K: int


# Fixed M=128, N=8192. K varies to bracket the K-dependence.
# K=4096: small (Phase 1.0 has shapes here, expect pure n-split win)
# K=16384: mid (between 8K and 28K, fills gap in Phase 1.0 data)
# K=32768: large (slightly bigger than L3-70B MLP down's 28K — confirms
#                  the trend extrapolates)
SHAPES: list[_Shape] = [
    _Shape(M=128, N=8192, K=4096),
    _Shape(M=128, N=8192, K=16384),
    _Shape(M=128, N=8192, K=32768),
]


def main() -> int:
    print(f"# Tile-shape probe — fixed M=128, N=8192, K ∈ "
          f"{[sh.K for sh in SHAPES]}")
    print(f"# warmup={WARMUP} iters={ITERS}")
    print()

    for sh in SHAPES:
        print(f"\n## Shape ({sh.M}, {sh.N}, {sh.K})  per-core flops = "
              f"{2 * sh.M * sh.N * sh.K // (NUM_CORES):,}\n")
        factors = _valid_factorizations_mn1(sh.M, sh.N, sh.K)
        print("| (m, n, 1) | M_per | N_per | wall ms | err |")
        print("|---|---:|---:|---:|---|")
        for (m, n) in factors:
            ms, err = _run(sh.M, sh.N, sh.K, m, n)
            M_per = sh.M // m
            N_per = sh.N // n
            if err:
                print(f"| ({m:>2},{n:>2},1) | {M_per} | {N_per} | err | "
                      f"{err[:50]} |")
            else:
                print(f"| ({m:>2},{n:>2},1) | {M_per} | {N_per} | "
                      f"{ms:.2f} | |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
