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

"""Probe — m_fast / n_fast on production shapes.

Tests two hypotheses on real model shapes:

  1. m_fast (M-cohort adjacent on ring) — captured a ~10% speedup
     on mixed-MN k=1 splits in synthetic testing. Does this hold on
     production Llama/Granite/DSv3/Mixtral/Qwen shapes?

  2. n_fast (N-cohort adjacent) — showed no effect on the synthetic
     large-A shape (M=2048, N=128, K=2048). Does it help on
     shapes with TRULY large A relative to B? Specifically prefill-
     style shapes with large M.

For each shape we use a mixed-MN k=1 split (this is where the
synthetic probe showed m_fast wins). We avoid k>1 splits because
m_fast/n_fast permutations break the L3DlOpsScheduler invariants
under K-split (see broken (4,4,2) and (1,16,2) cases in the earlier
mn_fast probe).
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


# Permutations. Note: logical encoding is K-outer per
# compute_ops._k_fast_core_id_permutation; logical(i_k, i_m, i_n) =
# i_k * m*n + i_m * n + i_n.

def perm_identity(m, n, k):
    return list(range(m * n * k))


def perm_kfast(m, n, k):
    """K-cohort adjacent: phys c → logical (c%k)*mn + (c//k)."""
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(m * n * k)]


def perm_mfast(m, n, k):
    """M-cohort adjacent: phys c → logical (c%m)*nk + (c//m).

    Under K-outer logical encoding, M-cohort members have logical IDs
    differing by n. With nk = n*k and m strides of nk, consecutive
    phys cores 0..m-1 hit logical 0, nk, 2nk, ... which all have
    i_k=0 and i_n=0 but different i_m → M-cohort. ✓
    """
    nk = n * k
    return [(c % m) * nk + (c // m) for c in range(m * n * k)]


def perm_nfast(m, n, k):
    """N-cohort adjacent: phys c → logical i_k*mn + i_m*n + (c%n).

    Place n N-cohort members adjacent. Each group of n physical cores
    covers one N-cohort (same i_k, i_m). Across groups, i_m advances
    then i_k.
    """
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


# Shape catalog. Each entry: (label, M, N, K, primary split, alternative split).
# Splits are mixed-MN k=1 to keep within m_fast/n_fast's known-good zone.
SHAPES = [
    # Small-M decode-style (mixed-MN sweet spot for m_fast in earlier probe)
    ("L3.1-8B q_proj M=32",     32,   4096,  4096, (8, 4, 1)),
    ("L3.1-8B q_proj M=128",   128,   4096,  4096, (8, 4, 1)),
    ("Granite-8B q_proj M=32",  32,   4096,  4096, (8, 4, 1)),
    ("Granite-8B gate M=32",    32,  12800,  4096, (4, 8, 1)),
    ("Granite-8B down M=32",    32,   4096, 12800, (4, 8, 1)),
    ("L3.2-3B gate M=128",     128,   8192,  3072, (4, 8, 1)),
    # Medium-M (transition to PT-saturated regime)
    ("DSv3 q_b_proj M=512",    512,  24576,  1536, (8, 4, 1)),
    ("Mixtral q_proj M=512",   512,   6144,  6144, (8, 4, 1)),
    # Large-M (where A becomes substantial — n_fast territory)
    ("Mixtral gate M=1024",   1024,  16384,  6144, (8, 4, 1)),
    ("Qwen-14B kv_proj M=2048", 2048, 2048,  5120, (8, 4, 1)),
]


def main():
    print("# Probe — m_fast / n_fast on production shapes (k=1 splits)")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("For each shape, test identity / m_fast / n_fast on a mixed-MN")
    print("split. m_fast places M-cohort (sharing B) adjacent; n_fast places")
    print("N-cohort (sharing A) adjacent. m_fast won ~10% in synthetic; we")
    print("check if it survives on real workloads. n_fast hypothesis: helps")
    print("when A is large (large-M prefill shapes).")
    print()

    print("| shape | split | A | B | identity | m_fast | n_fast | "
          "m_fast Δ% | n_fast Δ% |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for label, M, N, K, split in SHAPES:
        A_mb = M * K * 2 / 1024 / 1024
        B_mb = K * N * 2 / 1024 / 1024

        id_ms, id_err = measure(M, N, K, split, perm_identity)
        mf_ms, mf_err = measure(M, N, K, split, perm_mfast)
        nf_ms, nf_err = measure(M, N, K, split, perm_nfast)

        if id_ms is None or mf_ms is None or nf_ms is None:
            err = id_err or mf_err or nf_err
            print(f"| {label} | {split} | {A_mb:.1f}MB | {B_mb:.1f}MB | "
                  f"ERR ({err}) |")
            sys.stdout.flush()
            continue

        mf_delta_pct = (id_ms - mf_ms) / id_ms * 100
        nf_delta_pct = (id_ms - nf_ms) / id_ms * 100

        mf_str = f"**{mf_ms:.3f}**" if mf_ms < id_ms - 0.02 else f"{mf_ms:.3f}"
        nf_str = f"**{nf_ms:.3f}**" if nf_ms < id_ms - 0.02 else f"{nf_ms:.3f}"

        print(f"| {label} | {split} | {A_mb:.1f}MB | {B_mb:.1f}MB | "
              f"{id_ms:.3f} | {mf_str} | {nf_str} | "
              f"{mf_delta_pct:+.1f}% | {nf_delta_pct:+.1f}% |")
        sys.stdout.flush()

    print()
    print("## Interpretation")
    print()
    print("- m_fast Δ ≥ +5% on small-M decode shapes → confirms B-cohort")
    print("  position-dependent sharing exists in production.")
    print("- n_fast Δ ≥ +5% on large-M prefill shapes (Mixtral 1024, Qwen")
    print("  14B kv_proj) → confirms A-cohort position-dependent sharing")
    print("  for large activations.")
    print("- Either or both ≤ noise (~2%) → HBM coalescing is the dominant")
    print("  sharing mechanism and core-ID permutation has minimal upside.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
