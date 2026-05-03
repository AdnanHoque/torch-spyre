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

"""Validate the generalized `k_fast` permutation.

`k_fast` adapts the permutation to the planner's chosen split:
  perm[c] = (c mod k) * (m*n) + (c // k)

This packs the k K-collaborators into a contiguous physical block.
For different splits, k_fast should produce the win that the prior
shape-specific perms gave:

  (1, 16, 2): k_fast == block_cyclic  → kv_proj got 2.76x
  (4, 1, 8):  k_fast packs K-cluster to physical 0..7  (better than stride2's 14-hop chain)
  (2, 4, 4):  k_fast packs K-cluster to physical 0..3
  (1, 32, 1): k_fast == identity (k=1 means no K-collaborators)

For each test split, we expect:
  - k_fast doesn't crash
  - k_fast wins are >= the corresponding shape-specific perm wins
  - k_fast on (1, 32, 1) is identical to identity (safe default)
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
import sys

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402
from torch_spyre._inductor import core_division as _core_div  # noqa: E402


WARMUP = 5
ITERS = 25
DTYPE = torch.float16

# (label, M, N, K, split, comparison_perm)
TARGETS = [
    # K-split shapes — k_fast should win, matching or beating the
    # shape-specific known winner.
    ("kv_proj K-split (1,16,2)",   2048, 1024, 8192,  (1, 16, 2),  "block_cyclic"),
    ("q_proj K-split (4,1,8)",     128,  8192, 8192,  (4, 1, 8),   "stride2"),
    ("L3-8B MLP down K (4,1,8)",   128,  4096, 14336, (4, 1, 8),   "stride2"),
    # Mixed split where neither stride2 nor block_cyclic is K-fast.
    # k_fast should be the only optimum. (m=2, n=4, k=4 covers 32.)
    ("Mixed (2,4,4) synthetic",    128,  2048, 8192,  (2, 4, 4),   "stride2"),
    # Pure-N (no K-collab) — k_fast should be identical to identity.
    ("L3-70B q_proj pure-N",       128,  8192, 8192,  (1, 32, 1),  "identity"),
]


_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        prod = target[0] * target[1] * target[2]
        if prod != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target):
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


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


def _compile_and_bench(M, N, K, target, perm):
    ts_config.core_id_permutation = perm
    ts_config.core_emission_reverse = False
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(target):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(target):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:100]}"


def main() -> int:
    print("# k_fast validation — does the generalized perm win as predicted?\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    rows = []
    for label, M, N, K, target, comparison in TARGETS:
        print(f"### {label}  M={M} N={N} K={K}  split={target}")
        # Run identity, k_fast, and the shape-specific comparison perm.
        perms_to_test = ["identity", "k_fast"]
        if comparison != "identity" and comparison not in perms_to_test:
            perms_to_test.append(comparison)
        for tname, ordered in (("trial1", perms_to_test),
                                ("trial2", list(reversed(perms_to_test)))):
            print(f"  {tname} (order: {ordered}):")
            for perm in ordered:
                ms, err = _compile_and_bench(M, N, K, target, perm)
                if err:
                    print(f"    {perm:14s}: ERR {err[:80]}")
                else:
                    print(f"    {perm:14s}: {ms:.3f} ms")
            print()

    print("\n## Summary\n")
    print("Re-run the same configs to summarize speedups (re-bench):")
    print("| shape | split | identity ms | k_fast ms | reference ms | "
          "k_fast sp | reference sp |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for label, M, N, K, target, comparison in TARGETS:
        ms_id, _ = _compile_and_bench(M, N, K, target, "identity")
        ms_kf, err_kf = _compile_and_bench(M, N, K, target, "k_fast")
        if comparison != "identity":
            ms_ref, err_ref = _compile_and_bench(M, N, K, target, comparison)
        else:
            ms_ref = ms_id
            err_ref = ""
        if ms_id is None or ms_kf is None or ms_ref is None:
            print(f"| {label} | {target} | "
                  f"{ms_id or 'ERR'} | {ms_kf or 'ERR'} | {ms_ref or 'ERR'} | "
                  f"— | — |")
            continue
        sp_kf = ms_id / ms_kf
        sp_ref = ms_id / ms_ref
        print(f"| {label} | {target} | {ms_id:.3f} | {ms_kf:.3f} | "
              f"{ms_ref:.3f} ({comparison}) | "
              f"**{sp_kf:.3f}x** | {sp_ref:.3f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
