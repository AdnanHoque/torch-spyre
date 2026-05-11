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

"""Probe — % of theoretical peak with k_fast on production shapes.

For each shape, runs PR 1986's K-split + k_fast choice and the best
mixed-MN k=1 + identity, takes the faster of the two as "best
achievable today", then compares to two theoretical bounds:

  Compute peak    = 2·M·N·K / 72.1 TFLOPS                       fp16
  HBM peak        = (M·K + K·N + k·M·N) · 2 bytes / 166 GB/s    full A/B
                                                                multicast,
                                                                k partial
                                                                PSUMs

Theoretical wall = max(compute, hbm)   (assumes perfect overlap)
% of peak        = theoretical / observed × 100%

The 'k_fast efficiency' tells us how much room is left vs the
fundamental compute / HBM floor. Numbers below 50% suggest big
overheads; numbers above 80% suggest we're near the floor and
further work-division tuning has diminishing returns.
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
PT_PEAK_FP16 = 72.1e12     # FLOPS (per AIU, FMA = 2 ops/MAC)
HBM_PEAK = 166e9            # B/s (per AIU)

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
    """Return (compute_ms, hbm_ms, max_ms)."""
    compute_s = 2 * M * N * K / PT_PEAK_FP16
    hbm_bytes = (M * K + K * N + k_split * M * N) * dtype_bytes
    hbm_s = hbm_bytes / HBM_PEAK
    wall_s = max(compute_s, hbm_s)
    return compute_s * 1e3, hbm_s * 1e3, wall_s * 1e3


# Production shapes spanning M decade. For each, suggest the split
# family we expect to win (K-split for small-M, mixed-MN for large-M).
SHAPES = [
    # M=32 decode (k_fast firing zone — K-split should win)
    ("L3.1-8B q_proj M=32",     32,  4096,  4096, "K"),
    ("Granite-8B q_proj M=32",  32,  4096,  4096, "K"),
    ("Granite-8B gate M=32",    32, 12800,  4096, "K"),
    # M=128 decode (k_fast firing zone)
    ("L3.1-8B q_proj M=128",   128,  4096,  4096, "K"),
    ("Granite-8B gate M=128",  128, 12800,  4096, "K"),
    ("Granite-8B down M=128",  128,  4096, 12800, "K"),
    ("L3.2-3B gate M=128",     128,  8192,  3072, "K"),
    # M=512 (k_fast OFF for wide-N)
    ("L3.1-8B q_proj M=512",   512,  4096,  4096, "MN"),
    ("Granite-8B gate M=512",  512, 12800,  4096, "MN"),
    ("DSv3 q_b_proj M=512",    512, 24576,  1536, "MN"),
    # Large-M prefill (k_fast OFF)
    ("Mixtral gate M=1024",   1024, 16384,  6144, "MN"),
    ("Qwen-14B kv M=2048",    2048,  2048,  5120, "MN"),
    ("L3.1-70B q M=2048",     2048,  8192,  8192, "MN"),
    ("Granite-8B gate M=2048", 2048, 12800,  4096, "MN"),
]

K_SPLITS = [(1, 16, 2), (1, 8, 4), (4, 4, 2), (2, 8, 2)]
MN_SPLITS = [(8, 4, 1), (4, 8, 1), (16, 2, 1)]


def main():
    print("# Probe — % of theoretical peak with k_fast on production shapes")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print(f"PT peak: {PT_PEAK_FP16/1e12:.1f} TFLOPS fp16")
    print(f"HBM peak: {HBM_PEAK/1e9:.0f} GB/s")
    print()
    print("For each shape, the best of (K-split + k_fast) and (mixed-MN +")
    print("identity) is taken. Theoretical peak assumes full A and B")
    print("multicast on HBM, plus k partial PSUMs on C.")
    print()
    print("| shape | M | N | K | best split | best ms | compute ms | "
          "hbm ms | theor ms | % peak | bound |")
    print("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|")

    rows = []
    for label, M, N, K, family in SHAPES:
        # Try best of K-split + k_fast
        best_k = None
        if family == "K":
            for split in K_SPLITS:
                wall, _ = measure(M, N, K, split, perm_kfast)
                if wall is not None and (best_k is None or wall < best_k[1]):
                    best_k = (split, wall)
        # Try best of mixed-MN + identity
        best_mn = None
        for split in MN_SPLITS:
            wall, _ = measure(M, N, K, split, perm_identity)
            if wall is not None and (best_mn is None or wall < best_mn[1]):
                best_mn = (split, wall)

        candidates = [c for c in [best_k, best_mn] if c is not None]
        if not candidates:
            print(f"| {label} | {M} | {N} | {K} | ERR |")
            continue
        best = min(candidates, key=lambda t: t[1])
        chosen_split = best[0]
        wall_ms = best[1]
        k_used = chosen_split[2]

        compute_ms, hbm_ms, theor_ms = theoretical_bounds(M, N, K, k_used)
        pct = theor_ms / wall_ms * 100
        bound = "compute" if compute_ms > hbm_ms else "HBM"

        print(f"| {label} | {M} | {N} | {K} | {chosen_split} | "
              f"{wall_ms:.3f} | {compute_ms:.3f} | {hbm_ms:.3f} | "
              f"{theor_ms:.3f} | {pct:.1f}% | {bound} |")
        sys.stdout.flush()
        rows.append((label, family, pct, bound))

    print()
    print("## Summary")
    print()
    from collections import defaultdict
    by_family = defaultdict(list)
    by_bound = defaultdict(list)
    for label, family, pct, bound in rows:
        by_family[family].append(pct)
        by_bound[bound].append(pct)

    print("| group | shapes | mean %peak | median %peak | min | max |")
    print("|---|---:|---:|---:|---:|---:|")
    for fam, pcts in sorted(by_family.items()):
        mean_p = sum(pcts) / len(pcts)
        med_p = statistics.median(pcts)
        print(f"| family {fam} | {len(pcts)} | {mean_p:.1f}% | {med_p:.1f}% | "
              f"{min(pcts):.1f}% | {max(pcts):.1f}% |")
    for bnd, pcts in sorted(by_bound.items()):
        mean_p = sum(pcts) / len(pcts)
        med_p = statistics.median(pcts)
        print(f"| bound: {bnd} | {len(pcts)} | {mean_p:.1f}% | {med_p:.1f}% | "
              f"{min(pcts):.1f}% | {max(pcts):.1f}% |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
