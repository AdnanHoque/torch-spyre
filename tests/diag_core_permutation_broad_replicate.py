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

"""Replicate the broad-sweep candidates with both trial orders.

The broad sweep (`diag_core_permutation_broad.py`) found two ≥2% wins:

  - L3-70B q_proj M=512 + block_cyclic → 1.023x (NEW finding)
  - L3-70B q_proj K-split (4,1,8) + stride2 → 1.035x (confirms narrow probe)

The K-split stride2 result has been replicated in
`diag_core_permutation_replicate.py` already. The new finding is the
M=512 prefill case where block_cyclic helps. This probe replicates
THAT specifically with two trial orders.

Also tests neighbours of the M=512 win to see if it's a local pocket
or a broader regime: M=256, 384, 768, 1024 with the same split and
permutation.
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

# (label, M, N, K, split, [perms_to_test])
TARGETS = [
    # The two ≥2% candidates from broad sweep
    ("L3-70B q_proj M=512 prefill",  512, 8192, 8192, (1, 32, 1),
     ["identity", "block_cyclic", "stride2", "bit_reverse"]),
    ("L3-70B q_proj K-split (4,1,8)", 128, 8192, 8192, (4, 1, 8),
     ["identity", "stride2", "block_cyclic"]),
    # Neighbour M values for the M=512 result — is it a local pocket?
    ("L3-70B q_proj M=256",  256,  8192, 8192, (1, 32, 1),
     ["identity", "block_cyclic"]),
    ("L3-70B q_proj M=1024", 1024, 8192, 8192, (1, 32, 1),
     ["identity", "block_cyclic"]),
    # Bonus: does block_cyclic help on L3-70B at M=2048 too? broad sweep
    # showed 1.014x there which is just sub-threshold.
    ("L3-70B q_proj M=2048", 2048, 8192, 8192, (1, 32, 1),
     ["identity", "block_cyclic"]),
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
        return None, f"{type(e).__name__}: {str(e)[:80]}"


def main() -> int:
    print("# Broad-sweep replication — confirm M=512 block_cyclic & K-split stride2\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    results = {}  # (label, perm, trial) -> ms or None
    for label, M, N, K, target, perms in TARGETS:
        print(f"### {label}  M={M} N={N} K={K}  split={target}")
        for tname, ordered in (("trial1", perms),
                                ("trial2", list(reversed(perms)))):
            print(f"  {tname} (order: {ordered}):")
            for perm in ordered:
                ms, err = _compile_and_bench(M, N, K, target, perm)
                if err:
                    print(f"    {perm:14s}: ERR {err[:50]}")
                    results[(label, perm, tname)] = None
                else:
                    print(f"    {perm:14s}: {ms:.3f} ms")
                    results[(label, perm, tname)] = ms
            print()

    # --- summary ---
    print("\n## Replication summary\n")
    print("| shape | perm | trial1 ms | trial2 ms | t1 sp vs id | "
          "t2 sp vs id | mean sp | consistent? |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for label, _M, _N, _K, target, perms in TARGETS:
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        for perm in perms:
            if perm == "identity":
                continue
            t1 = results.get((label, perm, "trial1"))
            t2 = results.get((label, perm, "trial2"))
            if t1 is None or t2 is None or id1 is None or id2 is None:
                print(f"| {label} | {perm} | "
                      f"{t1 or 'ERR'} | {t2 or 'ERR'} | — | — | — | — |")
                continue
            sp1 = id1 / t1
            sp2 = id2 / t2
            mean = (sp1 + sp2) / 2
            cons = ("✓ same dir" if (sp1 - 1) * (sp2 - 1) > 0
                    else "~ flipped")
            print(f"| {label} | {perm} | {t1:.3f} | {t2:.3f} | "
                  f"{sp1:.3f}x | {sp2:.3f}x | {mean:.3f}x | {cons} |")

    print()
    print("## Verdict\n")
    confirmed = []
    for label, _M, _N, _K, target, perms in TARGETS:
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        for perm in perms:
            if perm == "identity":
                continue
            t1 = results.get((label, perm, "trial1"))
            t2 = results.get((label, perm, "trial2"))
            if t1 is None or t2 is None or id1 is None or id2 is None:
                continue
            sp1, sp2 = id1 / t1, id2 / t2
            mean = (sp1 + sp2) / 2
            if (sp1 - 1) * (sp2 - 1) > 0 and mean >= 1.02:
                confirmed.append((label, perm, mean))

    if confirmed:
        print(f"  Confirmed wins (≥2% AND consistent across both orders):")
        for c in confirmed:
            print(f"    - {c[0]}: perm={c[1]}, mean speedup {c[2]:.3f}x")
    else:
        print("  No replicated wins ≥2%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
