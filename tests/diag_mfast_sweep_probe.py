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

"""Probe — broad m_fast sweep to validate the m_fast-vs-identity rule.

Hypothesis from earlier probes: for k=1 splits, the choice between
m_fast (M-cohort adjacent → B-multicast) and identity (N-cohort
adjacent → A-multicast) is determined by:

  B_side = m · K · N / n · sizeof(B)        # M-cohort × B-frag size
  A_side = n · M · K / m · sizeof(A)        # N-cohort × A-frag size

  if B_side > A_side: m_fast wins
  else:               identity wins

For fp16 this simplifies to (m/n)² > M/N for m_fast to win.

Sweep design:
  Shapes spanning:
   - M ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}
   - Three (K, N) families: small (2K, 2K), medium (4K, 4K), large (4K, 16K)
  Splits: (8, 4, 1), (4, 8, 1), (16, 2, 1), (2, 16, 1)
  Permutations: identity vs m_fast
  Record: wall ms each, predicted winner, observed winner.
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


def predicted_winner(m, n, k, M, N, K):
    # B-side = m · K · N / n,  A-side = n · M · K / m  (sizeof cancels)
    b_side = m * K * N / n
    a_side = n * M * K / m
    return "m_fast" if b_side > a_side else "identity"


def main():
    print("# Probe — broad m_fast vs identity sweep")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()
    print("Rule under test: m_fast wins when (m/n)² > M/N for fp16.")
    print("Equivalent: M-cohort × B-frag-size > N-cohort × A-frag-size.")
    print()

    # (label, M, N, K)
    shapes = [
        # Small-M decoder
        ("decode-32-4k-4k",    32,  4096,  4096),
        ("decode-64-4k-4k",    64,  4096,  4096),
        ("decode-128-4k-4k",  128,  4096,  4096),
        ("decode-256-4k-4k",  256,  4096,  4096),
        # Wide-N decoder (gate_proj style)
        ("gate-128-12k-4k",   128, 12800,  4096),
        ("gate-256-12k-4k",   256, 12800,  4096),
        # Medium-M prefill
        ("prefill-512-4k-4k",  512,  4096,  4096),
        ("prefill-1k-4k-4k",  1024,  4096,  4096),
        ("prefill-2k-4k-4k",  2048,  4096,  4096),
        # Large-M prefill
        ("prefill-4k-4k-4k",  4096,  4096,  4096),
        ("prefill-8k-4k-4k",  8192,  4096,  4096),
        # Asymmetric (large-N or large-K)
        ("wideN-128-16k-2k",  128, 16384,  2048),
        ("wideK-128-2k-8k",   128,  2048,  8192),
        ("wideN-512-24k-2k",  512, 24576,  1536),
    ]

    # k=1 splits where m_fast meaningfully differs from identity
    splits = [(8, 4, 1), (4, 8, 1), (16, 2, 1), (2, 16, 1)]

    print("| shape | split | pred | id ms | m_fast ms | Δ% | match? |")
    print("|---|---|---|---:|---:|---:|:---:|")

    correct, total = 0, 0
    for label, M, N, K in shapes:
        for split in splits:
            m, n, k = split
            pred = predicted_winner(m, n, k, M, N, K)

            id_ms, id_err = measure(M, N, K, split, perm_identity)
            mf_ms, mf_err = measure(M, N, K, split, perm_mfast)

            if id_ms is None or mf_ms is None:
                err = id_err or mf_err
                print(f"| {label} | {split} | {pred} | "
                      f"{'ERR' if id_ms is None else f'{id_ms:.3f}'} | "
                      f"{'ERR' if mf_ms is None else f'{mf_ms:.3f}'} | "
                      f"— | err |")
                sys.stdout.flush()
                continue

            delta_pct = (id_ms - mf_ms) / id_ms * 100
            # Threshold ±0.5% for "real" preference
            if delta_pct > 0.5:
                observed = "m_fast"
            elif delta_pct < -0.5:
                observed = "identity"
            else:
                observed = "tie"

            if observed == "tie":
                match = "~"
            else:
                match_bool = (observed == pred)
                match = "✓" if match_bool else "✗"
                total += 1
                if match_bool:
                    correct += 1

            mf_str = f"**{mf_ms:.3f}**" if delta_pct > 0.5 else f"{mf_ms:.3f}"
            id_str = f"**{id_ms:.3f}**" if delta_pct < -0.5 else f"{id_ms:.3f}"

            print(f"| {label} | {split} | {pred} | {id_str} | {mf_str} | "
                  f"{delta_pct:+.1f}% | {match} |")
            sys.stdout.flush()

    print()
    print(f"## Decision-rule validation: {correct}/{total} matches")
    print()
    if total > 0:
        pct = correct / total * 100
        print(f"  hit rate: {pct:.1f}%")
        print()
        if pct >= 85:
            print("  → Decision rule (m_fast iff B-side > A-side) holds robustly.")
            print("  → m_fast as a feature should use this rule as its trigger.")
        elif pct >= 60:
            print("  → Rule mostly works but has edge cases. Worth refining.")
        else:
            print("  → Rule doesn't generalise; need a different decision criterion.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
