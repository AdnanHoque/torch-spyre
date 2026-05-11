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

"""Probe — production shapes where m_fast can win.

Earlier head-to-head (diag_mfast_vs_kfast_production.py) showed:
  - PR 1986's K-split+k_fast is optimal for M=128 decoder shapes
  - mixed-MN+m_fast loses by 15-22% on those shapes

So m_fast's value is in regimes where:
  - The planner does NOT pick K-split (i.e., k_fast heuristic doesn't fire)
  - The planner naturally picks a mixed-MN k=1 split
  - The shape sits in m_fast's gated sweet spot

PR 1986's k_fast heuristic doesn't fire when:
  - M < 32 OR M > 512                                  (out of range)
  - K < 2048                                           (k_sticks < max_cores)
  - M > 128 AND N >= 2048                              (planner-priority gate)

So m_fast candidates: shapes where the planner picks mixed-MN. This
probe sweeps production shapes that fall in these regimes and reports
the m_fast lift over identity (the best the planner picks today).
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


def perm_mfast(m, n, k):
    nk = n * k
    return [(c % m) * nk + (c // m) for c in range(m * n * k)]


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


# Production shapes where PR 1986's k_fast heuristic does NOT fire and
# the planner naturally picks (or could pick) mixed-MN k=1.
# Heuristic OFF when: M > 512, OR (M > 128 AND N >= 2048), OR K < 2048.
SHAPES = [
    # K < 2048 — narrow K
    ("DSv3 q_b_proj M=512",     512, 24576, 1536),   # K=1536 < 2048
    ("Granite-8B kv M=128",     128,  2048,  1024),  # K=1024 < 2048
    # M > 128 AND N >= 2048 — wide N, medium M (k_fast OFF)
    ("Granite-8B q_proj M=512", 512,  4096,  4096),
    ("Granite-8B o_proj M=512", 512,  4096,  4096),
    ("Granite-8B gate M=512",   512, 12800,  4096),
    ("Granite-8B down M=512",   512,  4096, 12800),
    ("L3.1-8B q_proj M=512",    512,  4096,  4096),
    ("Mixtral q_proj M=512",    512,  6144,  6144),
    # M > 512 — large prefill (k_fast OFF)
    ("DSv3 kv_a_proj M=1024",   1024,  576,  7168),
    ("Mixtral gate M=1024",     1024, 16384, 6144),
    ("Qwen-14B kv M=2048",      2048,  2048, 5120),
    ("L3.1-70B q_proj M=2048",  2048,  8192, 8192),
    # Large-M projection layers
    ("L3.1-8B gate M=2048",     2048, 14336, 4096),
    ("Granite-8B gate M=2048",  2048, 12800, 4096),
]

# Mixed-MN k=1 candidate splits (m_fast-eligible)
SPLITS = [(8, 4, 1), (4, 8, 1), (16, 2, 1)]


def main():
    print("# Probe — production shapes where m_fast can win (mixed-MN k=1 wins)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Selecting shapes where PR 1986's k_fast heuristic does NOT fire,")
    print("so the planner naturally picks mixed-MN k=1. Reports the best")
    print("mixed-MN split's identity vs m_fast wall time.")
    print()

    print("| shape | M | N | K | best id split | id ms | best mf split | "
          "mf ms | Δ% | regime |")
    print("|---|---:|---:|---:|---|---:|---|---:|---:|---|")

    summary = []
    for label, M, N, K in SHAPES:
        # Find best identity and best m_fast across mixed-MN splits
        id_results = []
        mf_results = []
        for split in SPLITS:
            id_ms, _ = measure(M, N, K, split, perm_identity)
            mf_ms, _ = measure(M, N, K, split, perm_mfast)
            if id_ms is not None:
                id_results.append((split, id_ms))
            if mf_ms is not None:
                mf_results.append((split, mf_ms))

        if not id_results or not mf_results:
            print(f"| {label} | {M} | {N} | {K} | ERR | — | — | — | — | — |")
            sys.stdout.flush()
            continue

        best_id = min(id_results, key=lambda t: t[1])
        best_mf = min(mf_results, key=lambda t: t[1])
        delta_pct = (best_id[1] - best_mf[1]) / best_id[1] * 100

        # Regime label
        if K < 2048:
            regime = "narrow-K"
        elif M > 512:
            regime = "large-M"
        elif M > 128 and N >= 2048:
            regime = "mid-M wide-N"
        else:
            regime = "?"

        mf_str = f"**{best_mf[1]:.3f}**" if delta_pct > 1 else f"{best_mf[1]:.3f}"
        id_str = f"**{best_id[1]:.3f}**" if delta_pct < -1 else f"{best_id[1]:.3f}"

        print(f"| {label} | {M} | {N} | {K} | {best_id[0]} | {id_str} | "
              f"{best_mf[0]} | {mf_str} | {delta_pct:+.1f}% | {regime} |")
        sys.stdout.flush()
        summary.append((label, regime, delta_pct))

    print()
    print("## Summary by regime")
    print()
    from collections import defaultdict
    by_regime = defaultdict(list)
    for label, regime, delta in summary:
        by_regime[regime].append(delta)

    print("| regime | shapes | mean Δ% | median Δ% | best | worst |")
    print("|---|---:|---:|---:|---:|---:|")
    for regime, deltas in sorted(by_regime.items()):
        mean_d = sum(deltas) / len(deltas)
        med_d = statistics.median(deltas)
        print(f"| {regime} | {len(deltas)} | {mean_d:+.1f}% | {med_d:+.1f}% | "
              f"{max(deltas):+.1f}% | {min(deltas):+.1f}% |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
