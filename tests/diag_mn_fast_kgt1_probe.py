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

"""Probe — m_fast / n_fast COMPOSED with k_fast for k>1 splits.

Earlier mn_fast probe broke on (1,16,2) and (4,4,2) splits with
"DtException: Expect valid lower and upper bound parameters" from
L3DlOpsScheduler:927. The hypothesis: deeptools requires K-cohort
peers to be at specific physical positions (where k_fast places them);
naive m_fast/n_fast permutations break this invariant.

This probe tries COMPOSED permutations:
  - mkfast: K-cohort adjacent (k_fast property preserved), and
            M-cohort grouped over K-cohorts
  - nkfast: K-cohort adjacent, N-cohort grouped over K-cohorts

If these avoid the deeptools error AND show m_fast-style speedup,
we have a composed permutation that works for k>1 splits too.
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
ITERS = 12
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


# Logical encoding (K-outer): logical = i_k * (m*n) + i_m * n + i_n.

def perm_identity(m, n, k):
    return list(range(m * n * k))


def perm_kfast(m, n, k):
    """Existing k_fast — K-cohort adjacent on ring."""
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(m * n * k)]


def perm_mkfast(m, n, k):
    """K-cohort adjacent AND M-cohort grouped over K-cohorts.

    phys c → (i_k, i_m, i_n) where:
      i_k = c % k                 (K-cohort innermost stride)
      i_m = (c // k) % m          (M-cohort next stride)
      i_n = (c // k) // m         (N-cohort outermost)
    """
    mn = m * n
    out = []
    for c in range(m * n * k):
        i_k = c % k
        i_m = (c // k) % m
        i_n = (c // k) // m
        out.append(i_k * mn + i_m * n + i_n)
    return out


def perm_nkfast(m, n, k):
    """K-cohort adjacent AND N-cohort grouped over K-cohorts.

    phys c → (i_k, i_m, i_n) where:
      i_k = c % k                 (K-cohort innermost)
      i_n = (c // k) % n          (N-cohort next)
      i_m = (c // k) // n         (M-cohort outermost)
    """
    mn = m * n
    out = []
    for c in range(m * n * k):
        i_k = c % k
        i_n = (c // k) % n
        i_m = (c // k) // n
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


def main():
    print("# Probe — composed permutations for k>1 splits")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: M=2048, N=128, K=2048 + a couple production decode shapes.")
    print()
    print("Testing composed permutations on k>1 splits:")
    print("  mkfast: k_fast (K-cohort adjacent) + M-cohort next stride")
    print("  nkfast: k_fast (K-cohort adjacent) + N-cohort next stride")
    print()

    shapes = [
        ("synthetic large-A", 2048, 128, 2048),
        ("L3.1-8B q_proj M=32", 32, 4096, 4096),
        ("Granite-8B kv M=32", 32, 2048, 4096),
        ("DSv3 q_b_proj M=512", 512, 24576, 1536),
    ]

    splits = [
        (1, 16, 2),   # earlier this errored under raw m_fast / n_fast
        (4, 4, 2),    # earlier this errored too
        (1, 8, 4),    # bigger k
        (2, 8, 2),    # mixed-MN with k>1
    ]

    perm_funcs = [
        ("identity", perm_identity),
        ("kfast",    perm_kfast),
        ("mkfast",   perm_mkfast),
        ("nkfast",   perm_nkfast),
    ]

    for shape_label, M, N, K in shapes:
        print(f"## {shape_label} (M={M}, N={N}, K={K})")
        print()
        print("| split | identity | kfast | mkfast | nkfast | best winner |")
        print("|---|---:|---:|---:|---:|---|")
        for split in splits:
            results = {}
            for label, pf in perm_funcs:
                wall_ms, err = measure(M, N, K, split, pf)
                results[label] = wall_ms if wall_ms is not None else "ERR"

            cells = []
            best_label = None
            best_val = float("inf")
            for label, _ in perm_funcs:
                v = results[label]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}")
                    if v < best_val:
                        best_val = v
                        best_label = label
                else:
                    cells.append("ERR")
            best_str = f"{best_label} ({best_val:.3f})" if best_label else "—"
            print(f"| {split} | {cells[0]} | {cells[1]} | {cells[2]} | "
                  f"{cells[3]} | {best_str} |")
            sys.stdout.flush()
        print()

    print("## Notes")
    print()
    print("- If mkfast/nkfast run without ERR on the k>1 splits, the L3Dl")
    print("  scheduler invariant is just 'K-cohort must be adjacent' — easy")
    print("  to compose with.")
    print("- If mkfast wins by ≥ 5% vs kfast on splits with both m>1 and k>1,")
    print("  the M-cohort B-sharing benefit compounds with k_fast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
