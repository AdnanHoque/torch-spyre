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

"""Probe — RIU CW/CCW separation hypothesis test.

Hypothesis: A-multicast and B-multicast share RIU ring bandwidth on
the same direction → contention. If true, splitting them onto
different ring directions (CW/CCW separation) would help. If false,
the directions are already independent (or there's no contention at
the volumes we see).

Test: hold compute roughly constant (same M, N, K), vary split family
to change the BALANCE of A-multicast vs B-multicast traffic. Compare
wall times:

  pure-M (32, 1, 1) — A unique per-core,  B multicast 32×
  pure-N (1, 32, 1) — A multicast 32×,    B unique per-core
  (4, 8, 1) mixed   — A multicast 8×,     B multicast 4×
  (16, 2, 1) wide-M — A multicast 2×,     B multicast 16×
  (2, 16, 1) wide-N — A multicast 16×,    B multicast 2×

If the chip has independent CW and CCW directions and uses them well:
  wall_mixed ≈ max(A_part, B_part) — looks like the larger operand

If A and B contend on shared ring path:
  wall_mixed ≈ A_part + B_part   — looks like sum

Shape choice: M=512, N=4096, K=4096 (PT-saturated for all splits with
M_per ≥ 16; HBM-bound regime so RIU usage is high).
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
HBM_PEAK = 166e9

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


SHAPES = [
    ("Llama-3.1-8B q M=512", 512, 4096, 4096),
    ("Granite-8B q M=512",   512, 4096, 4096),
    ("Granite-8B gate M=512", 512, 12800, 4096),
]

SPLITS = [
    ((32, 1, 1),  "pure-M",  "A unique / B mc=32"),
    ((1, 32, 1),  "pure-N",  "A mc=32 / B unique"),
    ((4, 8, 1),   "mixed",   "A mc=8 / B mc=4"),
    ((8, 4, 1),   "mixed",   "A mc=4 / B mc=8"),
    ((2, 16, 1),  "wide-N",  "A mc=16 / B mc=2"),
    ((16, 2, 1),  "wide-M",  "A mc=2 / B mc=16"),
]


def main():
    print("# Probe — RIU CW/CCW separation hypothesis test")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("For each shape, sweep split shapes that vary A-multicast vs B-")
    print("multicast intensity. If wall_mixed ≈ max(pure-M, pure-N) per shape,")
    print("ring directions are independent. If wall_mixed ≈ sum, they contend.")
    print()

    for label, M, N, K in SHAPES:
        print(f"## {label} (M={M}, N={N}, K={K})")
        print()

        # Compute theoretical HBM time (full multicast)
        hbm_bytes = (M * K + K * N + M * N) * 2
        hbm_floor_ms = hbm_bytes / HBM_PEAK * 1e3

        print(f"  Theoretical HBM floor (full multicast, k=1): "
              f"{hbm_floor_ms:.3f} ms")
        print()

        print("| split | family | mc pattern | wall ms | over floor |")
        print("|---|---|---|---:|---:|")

        results = {}
        for split, family, pattern in SPLITS:
            wall_ms, err = measure(M, N, K, split)
            if wall_ms is None:
                print(f"| {split} | {family} | {pattern} | ERR | — |")
                continue
            over_floor = wall_ms / hbm_floor_ms
            print(f"| {split} | {family} | {pattern} | "
                  f"{wall_ms:.3f} | {over_floor:.2f}× |")
            sys.stdout.flush()
            results[split] = wall_ms

        # Contention test
        print()
        pure_m = results.get((32, 1, 1))
        pure_n = results.get((1, 32, 1))
        mixed1 = results.get((4, 8, 1))
        mixed2 = results.get((8, 4, 1))
        if all(v is not None for v in [pure_m, pure_n, mixed1, mixed2]):
            max_pure = max(pure_m, pure_n)
            sum_pure = pure_m + pure_n
            best_mixed = min(mixed1, mixed2)

            print(f"  max(pure-M, pure-N):  {max_pure:.3f} ms")
            print(f"  sum(pure-M, pure-N):  {sum_pure:.3f} ms")
            print(f"  best mixed:           {best_mixed:.3f} ms")
            print()

            if best_mixed <= max_pure * 1.10:
                verdict = ("→ wall_mixed ≈ max(pure) → ring directions appear "
                           "INDEPENDENT (no major contention)")
            elif best_mixed >= sum_pure * 0.8:
                verdict = ("→ wall_mixed ≈ sum(pure) → ring directions appear "
                           "CONTENDED (CW/CCW separation could help)")
            else:
                verdict = ("→ wall_mixed between max and sum → partial "
                           "contention (some separation possible)")
            print(f"  {verdict}")
        print()

    print("## Overall interpretation")
    print()
    print("If most shapes show wall_mixed ≈ max(pure) → CW/CCW are already")
    print("independent or rarely contend → direction separation is unnecessary.")
    print()
    print("If most shapes show wall_mixed > max(pure) → contention exists →")
    print("RIU CW/CCW separation in deeptools would have empirical upside.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
