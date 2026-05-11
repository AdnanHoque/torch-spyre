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

"""Probe — n_fast on prefill-scale M (8k / 16k / 32k).

Earlier production probe showed n_fast no-effect for M up to 2048.
This pushes much harder: at M ∈ {8k, 16k, 32k}, the per-N-cohort A-
fragment grows to 8/16/32 MB — comparable to or larger than the
per-M-cohort B-fragment.

If A-sharing is position-independent (HBM coalescing), n_fast still
won't help even at large M. If A-sharing has any position-dependent
component, large-M is the regime most likely to expose it.

Shape: M = {8192, 16384, 32768}, N = 4096, K = 4096, fp16.
  A_full = M · K · 2 = 64 / 128 / 256 MB
  B_full = K · N · 2 = 32 MB
  C_full = M · N · 2 = 64 / 128 / 256 MB
  Arithmetic intensity: ~M / 3 FLOPs/byte → 2700 / 5400 / 10900
  → compute-bound regime at large M.

For (8, 4, 1):
  Per-core compute = M·N·K / 32 = M·N·K / 32 ops
  At M=8k: 8192·4096·4096/32 = 4.3 G ops; at 72.1 TFLOPS / 32 = 2.25
  TFLOPS/core → ~1.9 ms compute. So wall is compute-dominated.

For (1, 32, 1) pure-N:
  Each core has M_per = M, N_per = 128. M-row fully saturated for any M.
  This is the test where n_fast = identity (n_fast on n=32 collapses
  to identity since the whole cluster is one N-cohort), so we use
  (8, 4, 1) where n_fast meaningfully differs from identity.
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
ITERS = 6
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
    """M-cohort adjacent."""
    nk = n * k
    return [(c % m) * nk + (c // m) for c in range(m * n * k)]


def perm_nfast(m, n, k):
    """N-cohort adjacent."""
    mn = m * n
    out = []
    for c in range(m * n * k):
        i_n = c % n
        i_m = (c // n) % m
        i_k = (c // n) // m
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
    print("# Probe — n_fast on prefill-scale M (8k / 16k / 32k)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Fixed N=4096, K=4096. M sweep: 8192 / 16384 / 32768.")
    print("Split: (8, 4, 1). m_fast pairs M-cohort=8; n_fast pairs N-cohort=4.")
    print("At large M, A-frag per N-cohort = M/8 · K · 2 grows large (8-32 MB).")
    print()

    N, K = 4096, 4096

    print("| M | A_frag per N-coh | identity | m_fast | n_fast | "
          "m_fast Δ% | n_fast Δ% |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    for M in [8192, 16384, 32768]:
        split = (8, 4, 1)
        a_frag_mb = (M // 8) * K * 2 / 1024 / 1024

        id_ms, id_err = measure(M, N, K, split, perm_identity)
        mf_ms, mf_err = measure(M, N, K, split, perm_mfast)
        nf_ms, nf_err = measure(M, N, K, split, perm_nfast)

        if id_ms is None:
            print(f"| {M} | {a_frag_mb:.1f}MB | ERR ({id_err}) | — | — | — | — |")
            sys.stdout.flush()
            continue
        mf_str = "ERR" if mf_ms is None else f"{mf_ms:.3f}"
        nf_str = "ERR" if nf_ms is None else f"{nf_ms:.3f}"
        mf_delta = ((id_ms - mf_ms) / id_ms * 100) if mf_ms else 0
        nf_delta = ((id_ms - nf_ms) / id_ms * 100) if nf_ms else 0

        print(f"| {M} | {a_frag_mb:.1f}MB | {id_ms:.3f} | {mf_str} | {nf_str} | "
              f"{mf_delta:+.1f}% | {nf_delta:+.1f}% |")
        sys.stdout.flush()

    print()
    print("## Also testing (4, 8, 1) where N-cohort=8 is the larger group:")
    print()
    print("| M | A_frag per N-coh | identity | m_fast | n_fast | "
          "m_fast Δ% | n_fast Δ% |")
    print("|---:|---:|---:|---:|---:|---:|---:|")

    for M in [8192, 16384, 32768]:
        split = (4, 8, 1)
        a_frag_mb = (M // 4) * K * 2 / 1024 / 1024

        id_ms, id_err = measure(M, N, K, split, perm_identity)
        mf_ms, mf_err = measure(M, N, K, split, perm_mfast)
        nf_ms, nf_err = measure(M, N, K, split, perm_nfast)

        if id_ms is None:
            print(f"| {M} | {a_frag_mb:.1f}MB | ERR ({id_err}) | — | — | — | — |")
            sys.stdout.flush()
            continue
        mf_str = "ERR" if mf_ms is None else f"{mf_ms:.3f}"
        nf_str = "ERR" if nf_ms is None else f"{nf_ms:.3f}"
        mf_delta = ((id_ms - mf_ms) / id_ms * 100) if mf_ms else 0
        nf_delta = ((id_ms - nf_ms) / id_ms * 100) if nf_ms else 0

        print(f"| {M} | {a_frag_mb:.1f}MB | {id_ms:.3f} | {mf_str} | {nf_str} | "
              f"{mf_delta:+.1f}% | {nf_delta:+.1f}% |")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
