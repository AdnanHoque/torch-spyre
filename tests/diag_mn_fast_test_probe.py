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

"""Probe — does an m_fast / n_fast core-ID permutation change wall time?

Question: the A/B replication probes showed wall time is flat across
splits, suggesting A and B are already shared (most plausibly HBM
controller coalescing). The hypothesis was that any "multicast" is
position-independent.

If that hypothesis is wrong — and A/B sharing actually depends on
which cores are physically adjacent on a ring — then permuting core
IDs so M-cohort or N-cohort members are adjacent (analogous to
k_fast for K-cohort) should change wall time.

Probe design: take a split with non-trivial cohort sizes (e.g., (4, 4,
2) has m=4 M-cohorts, each of size n·k=8 cores). Run with three
permutations: identity, m_fast (M-cohort adjacent), n_fast (N-cohort
adjacent). If any permutation gives a different wall time, the
sharing IS position-dependent and the probe falsifies the HBM-
coalescing hypothesis.

Logical core indexing convention (for iteration space [M, N, K]):
  logical(i_m, i_n, i_k) = i_m * (n·k) + i_n * k + i_k

This is the planner's per-tile output ordering. m_fast and n_fast
formulas below permute physical → logical so the named cohort is
adjacent in physical-core space (assumed adjacent = adjacent on ring).

Tested splits and predicted cohort sizes for max_cores=32:
  (4, 4, 2) — k_fast pairs K=2; n_fast pairs N=4; m_fast pairs M=4
  (1, 16, 2) — k_fast pairs K=2; n_fast pairs N=16; m=1 so m_fast is identity
  (4, 8, 1) — k=1; m_fast pairs M=4; n_fast pairs N=8
  (32, 1, 1) — pure-M; m_fast pairs M=32; n_fast = identity (n=1)
  (1, 32, 1) — pure-N; n_fast pairs N=32; m_fast = identity (m=1)

Shape: M=2048, N=128, K=2048, fp16 (same as A-replication probe).
  Large A makes A-cohort multicast cost the largest single term IF
  the sharing were position-dependent.
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


# ─────────── Permutation strategies ───────────
# Iteration-space order is [M, N, K] → logical core c has
# (i_m, i_n, i_k) = (c // (n*k), (c // k) % n, c % k).

def perm_identity(m, n, k):
    return list(range(m * n * k))


def perm_kfast(m, n, k):
    """K-cohort adjacent: phys c → logical (c%k)*mn + (c//k)."""
    mn = m * n
    return [(c % k) * mn + (c // k) for c in range(m * n * k)]


def perm_mfast(m, n, k):
    """M-cohort adjacent: phys c → logical (c%m)*nk + (c//m)."""
    nk = n * k
    return [(c % m) * nk + (c // m) for c in range(m * n * k)]


def perm_nfast(m, n, k):
    """N-cohort adjacent: phys c → logical i_m*nk + (c%n)*k + i_k.

    Place n N-cohort members adjacent. Each group of n physical cores
    covers one N-cohort (same i_m, i_k). Across groups, i_k rotates
    then i_m advances.
    """
    nk = n * k
    out = []
    for c in range(m * n * k):
        i_n = c % n
        i_k = (c // n) % k
        i_m = (c // n) // k
        out.append(i_m * nk + i_n * k + i_k)
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
    print("# Probe — m_fast / n_fast core-ID permutation effect test")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Shape: M=2048, N=128, K=2048 (large-A bias)")
    print()
    print("For each split, run identity / m_fast / n_fast / k_fast perms.")
    print("If A/B sharing is position-independent (HBM coalescing model),")
    print("all four perms should give the SAME wall time per split.")
    print("If sharing requires ring adjacency, one of m_fast/n_fast should")
    print("give a noticeable speedup vs identity on splits where that")
    print("cohort > 1.")
    print()

    M, N, K = 2048, 128, 2048

    splits = [
        (32, 1, 1),   # pure-M: only m_fast = identity (rest trivial)
        (1, 32, 1),   # pure-N: only n_fast meaningful
        (8, 4, 1),    # mixed-MN: m_fast pairs M=8, n_fast pairs N=4
        (4, 8, 1),    # mixed-MN: m_fast pairs M=4, n_fast pairs N=8
        (1, 16, 2),   # mn+k: m_fast trivial; n_fast pairs N=16; k_fast pairs K=2
        (4, 4, 2),    # all three nontrivial
    ]

    perm_funcs = [
        ("identity", perm_identity),
        ("m_fast",   perm_mfast),
        ("n_fast",   perm_nfast),
        ("k_fast",   perm_kfast),
    ]

    print("| split | identity ms | m_fast ms | n_fast ms | k_fast ms | "
          "max Δ ms |")
    print("|---|---:|---:|---:|---:|---:|")

    for split in splits:
        results = {}
        for label, pf in perm_funcs:
            wall_ms, err = measure(M, N, K, split, pf)
            if wall_ms is None:
                results[label] = None
            else:
                results[label] = wall_ms
        # Compute max delta
        vals = [v for v in results.values() if v is not None]
        max_delta = (max(vals) - min(vals)) if len(vals) >= 2 else 0
        cells = []
        for label, _ in perm_funcs:
            v = results[label]
            cells.append(f"{v:.3f}" if v is not None else "ERR")
        print(f"| {split} | {cells[0]} | {cells[1]} | {cells[2]} | "
              f"{cells[3]} | {max_delta:.3f} |")
        sys.stdout.flush()

    print()
    print("## Verdict")
    print()
    print("If all max-Δ values are < 0.05 ms (essentially noise), A and B")
    print("sharing is position-independent — most likely HBM-controller")
    print("coalescing. m_fast / n_fast would not be useful optimizations.")
    print()
    print("If any max-Δ is ≥ 0.1 ms, that's evidence that physical core")
    print("layout DOES affect sharing — the multicast (if any) is ring-")
    print("dependent and the hypothesis was wrong. Look at which perm gives")
    print("the smallest wall and whether it matches the cohort with the")
    print("largest operand for that split.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
