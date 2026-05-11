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

"""Probe — back out PSUM ring cost from k_fast vs identity wall-times.

For a K-split (m, n, k>1) work-division, K-cohort members are at
physical distance `mn` on the ring under identity, vs distance 1
under k_fast. Per-tile ring cost scales linearly with hop count, so:

  ring_cost_identity ≈ mn · ring_cost_per_unit_distance · n_tiles
  ring_cost_kfast    ≈  1 · ring_cost_per_unit_distance · n_tiles

Therefore:
  Δ = identity_wall − kfast_wall ≈ (mn − 1) · ring_cost_per_unit · n_tiles
  ⇒ kfast_ring_cost ≈ Δ / (mn − 1)

We then ask: how much of the (peak − kfast_wall) gap does that
ring cost explain? If ring is ~80% of the gap, ring optimization
is where to focus. If <20%, the gap is elsewhere (launch overhead,
PT pipeline fill, HBM-below-peak).
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


def perm_identity(m, n, k):
    return list(range(m * n * k))


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


def measure(M, N, K, split, perm_func):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_perm(perm_func, split), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_perm(perm_func, split), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def theoretical_bounds(M, N, K, k_split, dtype_bytes=2):
    compute_s = 2 * M * N * K / PT_PEAK_FP16
    hbm_bytes = (M * K + K * N + k_split * M * N) * dtype_bytes
    hbm_s = hbm_bytes / HBM_PEAK
    return compute_s * 1e3, hbm_s * 1e3, max(compute_s * 1e3, hbm_s * 1e3)


# K-split production shapes where k_fast applies. Pick the split family
# that won in the peak efficiency probe for each shape.
SHAPES_AND_SPLITS = [
    ("L3.1-8B q_proj M=32",     32,  4096,  4096, (2, 8, 2)),
    ("Granite-8B q_proj M=32",  32,  4096,  4096, (2, 8, 2)),
    ("Granite-8B gate M=32",    32, 12800,  4096, (2, 8, 2)),
    ("L3.1-8B q_proj M=128",   128,  4096,  4096, (2, 8, 2)),
    ("Granite-8B gate M=128",  128, 12800,  4096, (2, 8, 2)),
    ("Granite-8B down M=128",  128,  4096, 12800, (2, 8, 2)),
    ("L3.2-3B gate M=128",     128,  8192,  3072, (4, 4, 2)),
    # k=4 splits for variety
    ("L3.1-8B q_proj M=32 k4",   32,  4096,  4096, (1, 8, 4)),
    ("L3.1-8B q_proj M=128 k4", 128,  4096,  4096, (1, 8, 4)),
]


def main():
    print("# Probe — PSUM ring cost via k_fast vs identity delta")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("For each (shape, K-split), measure wall_id and wall_kf for")
    print("the SAME split. Δ = wall_id − wall_kf isolates the ring-cost")
    print("difference between mn-hops and 1-hop K-cohort placement.")
    print()
    print("Estimated kfast ring contribution ≈ Δ / (mn − 1).")
    print()

    print("| shape | split | mn | id wall ms | kf wall ms | Δ ms | "
          "kf ring est | peak ms | gap to peak | ring/gap |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    rows = []
    for label, M, N, K, split in SHAPES_AND_SPLITS:
        m, n, k = split
        mn = m * n

        id_ms, _ = measure(M, N, K, split, perm_identity)
        kf_ms, _ = measure(M, N, K, split, perm_kfast)
        if id_ms is None or kf_ms is None:
            print(f"| {label} | {split} | {mn} | ERR | ERR | — | — | — | — | — |")
            continue

        delta = id_ms - kf_ms
        kf_ring_est = delta / (mn - 1) if mn > 1 else 0

        _, _, peak_ms = theoretical_bounds(M, N, K, k)
        gap = kf_ms - peak_ms
        ring_over_gap = (kf_ring_est / gap * 100) if gap > 0 else 0

        print(f"| {label} | {split} | {mn} | {id_ms:.3f} | {kf_ms:.3f} | "
              f"{delta:.3f} | {kf_ring_est:.3f} | {peak_ms:.3f} | "
              f"{gap:.3f} | {ring_over_gap:.1f}% |")
        sys.stdout.flush()
        rows.append((label, kf_ring_est, gap, ring_over_gap))

    print()
    print("## Summary")
    print()
    if rows:
        ring_costs = [r[1] for r in rows]
        gaps = [r[2] for r in rows]
        ratios = [r[3] for r in rows if r[2] > 0]
        print(f"  Estimated k_fast ring cost (mean):  {statistics.mean(ring_costs)*1000:.0f} μs")
        print(f"  Peak gap (mean):                    {statistics.mean(gaps)*1000:.0f} μs")
        if ratios:
            print(f"  Ring cost / peak gap (mean):        {statistics.mean(ratios):.1f}%")
            print(f"  Ring cost / peak gap (median):      {statistics.median(ratios):.1f}%")
        print()
        print("  Interpretation:")
        print("  - <20%: ring is small share; most gap is launch/HBM-effi/PT-pipe")
        print("  - 20-50%: ring is meaningful; further ring opt has upside")
        print("  - >50%: ring is dominant; ring optimization is the leverage")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
