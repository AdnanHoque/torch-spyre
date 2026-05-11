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

"""Probe — isolate A-operand multicast vs n-replication.

Question: under pure-N (1, 32, 1) and other splits with n > 1, is A
(activation) multicast across the N-cohort, or is it replicated n
times in HBM reads?

Background: we established via doWeiPreload trace that B (weight) is
preloaded into LX out of the matmul critical path. A is NOT preloaded
because it changes per invocation. So A's HBM cost during the
measured iteration is real. The question is whether A's HBM cost
scales with n (replication, no multicast) or stays flat (multicast).

Shape choice: LARGE A, small B, small C → A dominates HBM cost.

  M = 2048, N = 128, K = 2048, fp16
  A = M · K · 2 = 8 MB           ← large; will dominate A traffic
  B = K · N · 2 = 0.5 MB         ← small (and preloaded, so free)
  C = M · N · 2 = 0.5 MB

Splits:
  (32, 1, 1) pure-M       A unique per core: M/32 × K × 2 = 256 KB
                          → no replication; per-cluster A = 8 MB.
  (1, 32, 1) pure-N       A same across cores; per-core = full A.
                          Without multicast: per-cluster = 32 × 8 = 256 MB.
                          With multicast:    per-cluster = 8 MB.
  (8, 4, 1) mixed-MN      A unique per (m_idx, k_idx=0); n=4 cohorts.
                          Without multicast: per-cluster = 4 × 8 = 32 MB.
                          With multicast:    per-cluster = 8 MB.

If A is multicast across the N-cohort, all three splits should have
similar HBM time. If A is replicated n times, pure-N should be
~1.5 ms slower than pure-M (Δ = 248 MB / 166 GB/s).

This probe is the empirical test for whether n_fast (a core-ID
permutation that places N-cohort cores adjacent on the RIU ring) has
upside — if A is already multicast, n_fast is moot; if A is
replicated, n_fast could enable real savings.
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
    print("# Probe — A-operand replication test (large A, small B)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: M=2048, N=128, K=2048")
    print("  A = M·K·2 = 8 MB   (large; not preloaded)")
    print("  B = K·N·2 = 0.5 MB (small; preloaded → free)")
    print("  C = M·N·2 = 0.5 MB")
    print()

    M, N, K = 2048, 128, 2048

    splits = [
        ((32, 1, 1), "pure-M", 1, "A per-cluster = 8 MB (no rep)"),
        ((1, 32, 1), "pure-N", 32, "A naive = 256 MB; multicast = 8 MB"),
        ((8, 4, 1), "mixed-MN", 4, "A naive = 32 MB; multicast = 8 MB"),
        ((4, 8, 1), "mixed-MN", 8, "A naive = 64 MB; multicast = 8 MB"),
        ((2, 16, 1), "mixed-MN", 16, "A naive = 128 MB; multicast = 8 MB"),
    ]

    print("| split | label | n | A traffic if no-mc | wall ms |")
    print("|---|---|---:|---:|---:|")

    rows = []
    for split, label, n_factor, comment in splits:
        a_bytes_no_mc = n_factor * M * K * 2
        a_bytes_no_mc_mb = a_bytes_no_mc / 1024 / 1024
        wall_ms, err = measure(M, N, K, split)
        if wall_ms is None:
            print(f"| {split} | {label} | {n_factor} | "
                  f"{a_bytes_no_mc_mb:.1f} MB | ERR ({err}) |")
            continue
        print(f"| {split} | {label} | {n_factor} | "
              f"{a_bytes_no_mc_mb:.1f} MB | {wall_ms:.3f} |")
        sys.stdout.flush()
        rows.append((split, label, n_factor, a_bytes_no_mc_mb, wall_ms))

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
            delta = pure_n_wall - pure_m_wall
            naive_delta_ms = (31 * M * K * 2) / 166e9 * 1000
            print(f"  pure-M wall: {pure_m_wall:.3f} ms")
            print(f"  pure-N wall: {pure_n_wall:.3f} ms")
            print(f"  Δ (pure-N − pure-M): {delta:.3f} ms")
            print(f"  predicted Δ if A replicated 32× at 166 GB/s: "
                  f"{naive_delta_ms:.3f} ms")
            print()
            if delta > 0.5 * naive_delta_ms:
                print("  → A IS replicated n× (multicast NOT active for A).")
                print("  → n_fast / A-multicast feature has real upside.")
            elif delta < 0.1 * naive_delta_ms:
                print("  → A is multicast across N-cohort (no detectable replication cost).")
                print("  → n_fast feature has no upside; A is already shared.")
            else:
                print("  → Inconclusive. Replication cost is partial; possibly")
                print("    HBM is partly saturated but coalescing-effects exist.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
