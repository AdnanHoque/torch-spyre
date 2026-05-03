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

"""Replicate the strongest permutation signals across two trial orders.

Initial probe (`diag_core_permutation_probe.py`) found:
  - L3-70B q_proj K-split (4, 1, 8): stride2 → 1.038x vs identity
  - L3-70B MLP down (16, 2, 1): random_42 → 1.021x; stride2 → 1.019x

Plus: 4 permutations crashed dxp on K-split (reversed, antipodal,
random_42, random_7), which is itself a finding worth re-checking.

This probe re-runs each interesting (shape, perm) combo with two
trial orders and 2x the iters. If signals replicate AND crashes
reproduce, both findings are real.
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
ITERS = 30
DTYPE = torch.float16

# (label, M, N, K, split, [perms_to_test])
TARGETS = [
    (
        "L3-70B q_proj prefill (K-split)", 128, 8192, 8192, (4, 1, 8),
        ["identity", "stride2",
         # Test crashes for reproducibility:
         "reversed", "antipodal", "random_42",
         # And worst regressors:
         "block_cyclic", "bit_reverse"],
    ),
    (
        "L3-70B MLP down prefill (output)", 128, 8192, 28672, (16, 2, 1),
        ["identity", "random_42", "stride2", "block_cyclic", "bit_reverse"],
    ),
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
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def main() -> int:
    print("# Permutation REPLICATION — strong signals + crash reproducibility\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    results = {}  # (label, perm, trial) -> ms or None
    for label, M, N, K, target, perms in TARGETS:
        print(f"### {label}  M={M} N={N} K={K}  split={target}\n")
        # Two trial orders: forward and reverse
        trial_orders = [
            ("trial1", perms),
            ("trial2", list(reversed(perms))),
        ]
        for tname, ordered_perms in trial_orders:
            print(f"  {tname} (order: {ordered_perms}):")
            for perm in ordered_perms:
                ms, err = _compile_and_bench(M, N, K, target, perm)
                if err:
                    print(f"    {perm:14s}: ERR {err}")
                    results[(label, perm, tname)] = None
                else:
                    print(f"    {perm:14s}: {ms:.3f} ms")
                    results[(label, perm, tname)] = ms
            print()

    # --- summary ---
    print("\n## Replication summary\n")
    print("|shape | perm | trial1 ms | trial2 ms | "
          "trial1 sp vs id | trial2 sp vs id | mean sp | consistent? |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for label, _M, _N, _K, target, perms in TARGETS:
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        for perm in perms:
            if perm == "identity":
                continue
            t1 = results.get((label, perm, "trial1"))
            t2 = results.get((label, perm, "trial2"))
            if t1 is None and t2 is None:
                consistent = "✓ both crashed"
                t1_str = t2_str = "ERR"
                sp1_str = sp2_str = mean_str = "—"
            elif t1 is None or t2 is None:
                consistent = "✗ one crashed"
                t1_str = "ERR" if t1 is None else f"{t1:.3f}"
                t2_str = "ERR" if t2 is None else f"{t2:.3f}"
                sp1_str = sp2_str = mean_str = "—"
            else:
                sp1 = id1 / t1 if id1 else 0
                sp2 = id2 / t2 if id2 else 0
                mean = (sp1 + sp2) / 2
                t1_str = f"{t1:.3f}"
                t2_str = f"{t2:.3f}"
                sp1_str = f"{sp1:.3f}x"
                sp2_str = f"{sp2:.3f}x"
                mean_str = f"{mean:.3f}x"
                consistent = (
                    "✓ same direction" if (sp1 - 1) * (sp2 - 1) > 0
                    else "~ flipped (noise)"
                )
            print(
                f"| {label} | {perm} | {t1_str} | {t2_str} | "
                f"{sp1_str} | {sp2_str} | {mean_str} | {consistent} |"
            )
    print()

    # --- verdict ---
    print("## Verdict\n")
    confirmed_wins = []
    confirmed_crashes = []
    for label, _M, _N, _K, target, perms in TARGETS:
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        for perm in perms:
            if perm == "identity":
                continue
            t1 = results.get((label, perm, "trial1"))
            t2 = results.get((label, perm, "trial2"))
            if t1 is None and t2 is None:
                confirmed_crashes.append((label, perm))
                continue
            if t1 is None or t2 is None or id1 is None or id2 is None:
                continue
            sp1 = id1 / t1
            sp2 = id2 / t2
            if (sp1 - 1) * (sp2 - 1) > 0 and (sp1 + sp2) / 2 >= 1.02:
                confirmed_wins.append((label, perm, (sp1 + sp2) / 2))

    print(f"  Confirmed wins (>=2% on both trials): {len(confirmed_wins)}")
    for w in confirmed_wins:
        print(f"    - {w[0]}: perm={w[1]}, mean speedup {w[2]:.3f}x")
    print(f"  Reproducible crashes: {len(confirmed_crashes)}")
    for c in confirmed_crashes:
        print(f"    - {c[0]}: perm={c[1]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
