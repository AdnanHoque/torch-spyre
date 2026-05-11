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

"""Probe — mixed-MN + m_fast vs K-split + k_fast vs K-split + composed.

Strategic question: for production decode shapes in PR 1986's firing
zone, which path wins?

  A) Mixed-MN k=1 + m_fast permutation     (new optimization)
  B) K-split k>1 + k_fast permutation      (PR 1986's current approach)
  C) K-split k>1 + composed m_fast+k_fast  (try to exploit both
                                            B-multicast AND PSUM ring)

For each shape we test a representative split from each family and
compare head-to-head. Also includes identity baselines so we can see
how far each permutation lifts the wall.

Composed permutation `mkfast`:
  K-cohort innermost stride (preserves k_fast PSUM-ring adjacency),
  M-cohort next stride (B-multicast benefit if any).
  Earlier kgt1 probe showed mkfast within 0.003 ms of k_fast — we
  reconfirm on these specific production shapes.
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


def perm_kfast(m, n, k):
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(m * n * k)]


def perm_mfast(m, n, k):
    nk = n * k
    return [(c % m) * nk + (c // m) for c in range(m * n * k)]


def perm_mkfast(m, n, k):
    """K-cohort innermost stride, M-cohort next stride."""
    mn = m * n
    out = []
    for c in range(m * n * k):
        i_k = c % k
        i_m = (c // k) % m
        i_n = (c // k) // m
        out.append(i_k * mn + i_m * n + i_n)
    return out


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


# Production decode/prefill shapes — M=128 sweet spot for m_fast
SHAPES = [
    ("L3.1-8B q_proj M=128",   128,  4096,  4096),
    ("L3.1-8B o_proj M=128",   128,  4096,  4096),
    ("Granite-8B q_proj M=128", 128, 4096,  4096),
    ("Granite-8B gate M=128",  128, 12800,  4096),
    ("Granite-8B down M=128",  128,  4096, 12800),
    ("L3.2-3B gate M=128",     128,  8192,  3072),
]

# Per shape, test family A (mixed-MN k=1), family B (K-split k>1)
FAMILY_A_SPLITS = [(8, 4, 1), (4, 8, 1)]                 # mixed-MN, m_fast applies
FAMILY_B_SPLITS = [(1, 16, 2), (1, 8, 4), (4, 4, 2),
                   (2, 8, 2)]                            # K-split, k_fast applies


def best_of(measurements):
    """Return (best_split, best_perm_label, best_ms) from list of
    (split, perm_label, ms) skipping None values."""
    valid = [(s, p, m) for (s, p, m) in measurements if m is not None]
    if not valid:
        return None
    return min(valid, key=lambda t: t[2])


def main():
    print("# Probe — mixed-MN+m_fast vs K-split+k_fast vs K-split+composed")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("For each shape, compare best-of-family:")
    print("  Family A: mixed-MN k=1 with m_fast")
    print("  Family B: K-split k>1 with k_fast")
    print("  Family C: K-split k>1 with mkfast (k_fast + M-cohort)")
    print()

    print("| shape | A best | A ms | B best | B ms | C best | C ms | "
          "winner | A vs B |")
    print("|---|---|---:|---|---:|---|---:|---|---:|")

    for label, M, N, K in SHAPES:
        # Family A: identity + m_fast on mixed-MN k=1 splits
        a_results = []
        for split in FAMILY_A_SPLITS:
            id_ms, _ = measure(M, N, K, split, perm_identity)
            mf_ms, _ = measure(M, N, K, split, perm_mfast)
            if id_ms is not None:
                a_results.append((split, "identity", id_ms))
            if mf_ms is not None:
                a_results.append((split, "m_fast", mf_ms))
        a_best = best_of(a_results)

        # Family B: identity + k_fast on K-split k>1 splits
        b_results = []
        for split in FAMILY_B_SPLITS:
            id_ms, _ = measure(M, N, K, split, perm_identity)
            kf_ms, _ = measure(M, N, K, split, perm_kfast)
            if id_ms is not None:
                b_results.append((split, "identity", id_ms))
            if kf_ms is not None:
                b_results.append((split, "k_fast", kf_ms))
        b_best = best_of(b_results)

        # Family C: composed mkfast on K-split k>1 splits
        c_results = []
        for split in FAMILY_B_SPLITS:
            mk_ms, _ = measure(M, N, K, split, perm_mkfast)
            if mk_ms is not None:
                c_results.append((split, "mkfast", mk_ms))
        c_best = best_of(c_results)

        def fmt(b):
            if b is None:
                return "—", "ERR"
            split_str = f"{b[0]}+{b[1]}"
            return split_str, f"{b[2]:.3f}"

        a_split, a_ms = fmt(a_best)
        b_split, b_ms = fmt(b_best)
        c_split, c_ms = fmt(c_best)

        # Determine overall winner
        candidates = [(a_best, "A"), (b_best, "B"), (c_best, "C")]
        valid = [(best, fam) for (best, fam) in candidates if best is not None]
        if not valid:
            overall = "—"
            a_vs_b_str = "—"
        else:
            best_overall = min(valid, key=lambda t: t[0][2])
            overall = f"{best_overall[1]} ({best_overall[0][2]:.3f})"

            if a_best and b_best:
                delta = (b_best[2] - a_best[2]) / b_best[2] * 100
                a_vs_b_str = f"{delta:+.1f}%"
            else:
                a_vs_b_str = "—"

        print(f"| {label} | {a_split} | {a_ms} | {b_split} | {b_ms} | "
              f"{c_split} | {c_ms} | {overall} | {a_vs_b_str} |")
        sys.stdout.flush()

    print()
    print("## Interpretation")
    print()
    print("- Family A wins ≥ Family B: planner should prefer mixed-MN +")
    print("  m_fast over K-split + k_fast. Suggests extending the planner")
    print("  heuristic to consider m_fast.")
    print("- Family B wins: PR 1986's current k_fast path is already")
    print("  capturing the best win on these shapes; m_fast is a separate")
    print("  optimization for non-firing shapes.")
    print("- Family C ≈ B: composed permutation adds nothing on top of")
    print("  k_fast, consistent with the earlier kgt1 probe (psum ring")
    print("  cost dominates so M-cohort B-multicast is invisible).")
    print("- Family C > B: surprising — composing m_fast + k_fast actually")
    print("  exploits both B-multicast AND PSUM ring; would motivate a")
    print("  composed _select_core_id_permutation for k>1 splits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
