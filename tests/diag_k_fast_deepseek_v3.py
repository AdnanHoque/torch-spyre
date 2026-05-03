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

"""DeepSeek V3 + cross-model k_fast validation.

The k_fast theory writeup predicts that any modern model with a
matmul whose N-stick-count doesn't divide cleanly into 32 (forcing
the planner into a (1, 16, 2) split) will benefit from k_fast.

DeepSeek V3 architecture (hidden=7168, MLA, MoE) gives us multiple
distinct matmul shapes that hit this trigger:

  - o_proj:       (M, 7168, 16384) -> 112 sticks -> (1, 16, 2)
  - down_proj:    (M, 7168, 2048)  -> 112 sticks -> (1, 16, 2)
  - q_a_proj:     (M, 1536, 7168)  -> 24 sticks  -> (1, 8, 4)

This probe tests those shapes plus controls (pure-N, pure-K) plus
cross-model validation (Mixtral 8x7B, L3-70B reference).

Per-expert MoE: with 256 experts and top-8 routing, per-expert
M ≈ M_total / 32. Tested separately at M=64 to reflect reality.
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


WARMUP = 3
ITERS = 15
DTYPE = torch.float16

# (regime, label, M, N, K, split)
TARGETS = [
    # === Reference: known 2.76x win ===
    ("ref",   "L3-70B kv_proj M=2048",          2048,  1024,  8192, (1, 16, 2)),

    # === Cross-model validation ===
    ("cross", "Mixtral 8x7B kv_proj M=2048",    2048,  1024,  4096, (1, 16, 2)),

    # === DeepSeek V3: predicted big wins (1, 16, 2) ===
    ("dsv3",  "DSv3 o_proj M=2048",             2048,  7168, 16384, (1, 16, 2)),
    ("dsv3",  "DSv3 down_proj M=2048 (dense)",  2048,  7168,  2048, (1, 16, 2)),

    # === DeepSeek V3: per-expert MoE (M=64 ~ token-routing average) ===
    ("dsv3-moe", "DSv3 down_proj per-expert M=64",  64, 7168, 2048, (1, 16, 2)),

    # === DeepSeek V3: K=4 split (smaller predicted win) ===
    ("dsv3",  "DSv3 q_a_proj M=2048",           2048,  1536,  7168, (1, 8, 4)),

    # === DeepSeek V3: controls (no k_fast benefit expected) ===
    ("dsv3-ctrl", "DSv3 kv_a_proj M=2048 (pure-K)",  2048,   576,  7168, (1, 1, 32)),
    ("dsv3-ctrl", "DSv3 kv_b_proj M=2048 (pure-N)",  2048, 32768,   512, (1, 32, 1)),
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
    print("# DeepSeek V3 + cross-model k_fast validation\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")
    print(f"# Each (shape, perm) runs in BOTH trial orders.\n")

    results = {}  # (label, perm, trial) -> ms or None
    for regime, label, M, N, K, target in TARGETS:
        n_st = N / 64
        k_st = K / 64
        print(f"### [{regime}] {label}")
        print(f"      shape: M={M}, N={N} ({n_st:.0f}st), K={K} ({k_st:.0f}st), split={target}")
        for tname, ordered in (("trial1", ["identity", "k_fast"]),
                                ("trial2", ["k_fast", "identity"])):
            print(f"  {tname} (order: {ordered}):")
            for perm in ordered:
                ms, err = _compile_and_bench(M, N, K, target, perm)
                if err:
                    print(f"    {perm:14s}: ERR {err[:90]}")
                    results[(label, perm, tname)] = None
                else:
                    print(f"    {perm:14s}: {ms:.3f} ms")
                    results[(label, perm, tname)] = ms
            print()

    # --- summary ---
    print("\n## Summary — k_fast vs identity (mean of two trial orders)\n")
    print("| regime | shape | split | identity ms | k_fast ms | mean sp | consistent? |")
    print("|---|---|---|---:|---:|---:|---|")
    for regime, label, _M, _N, _K, target in TARGETS:
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        kf1 = results.get((label, "k_fast", "trial1"))
        kf2 = results.get((label, "k_fast", "trial2"))
        if None in (id1, id2, kf1, kf2):
            i_str = f"{id1:.3f}" if id1 is not None else "ERR"
            k_str = f"{kf1:.3f}" if kf1 is not None else "ERR"
            print(f"| {regime} | {label} | {target} | {i_str} | {k_str} | — | — |")
            continue
        sp1 = id1 / kf1
        sp2 = id2 / kf2
        mean = (sp1 + sp2) / 2
        cons = "✓" if (sp1 - 1) * (sp2 - 1) > 0 else "~ flipped"
        ident_avg = (id1 + id2) / 2
        kfast_avg = (kf1 + kf2) / 2
        marker = " 🚀" if mean >= 1.5 else (" ✓" if mean >= 1.05 else "")
        print(f"| {regime} | {label} | {target} | "
              f"{ident_avg:.3f} | {kfast_avg:.3f} | "
              f"**{mean:.3f}x**{marker} | {cons} |")
    print()

    # --- big wins ---
    print("## Confirmed big wins (≥1.5x mean, consistent direction)\n")
    huge = []
    for regime, label, _M, _N, _K, target in TARGETS:
        id1 = results.get((label, "identity", "trial1"))
        id2 = results.get((label, "identity", "trial2"))
        kf1 = results.get((label, "k_fast", "trial1"))
        kf2 = results.get((label, "k_fast", "trial2"))
        if None in (id1, id2, kf1, kf2):
            continue
        sp1, sp2 = id1 / kf1, id2 / kf2
        mean = (sp1 + sp2) / 2
        if (sp1 - 1) * (sp2 - 1) > 0 and mean >= 1.5:
            huge.append((label, target, mean, (id1+id2)/2, (kf1+kf2)/2))
    if huge:
        for w in huge:
            print(f"  {w[0]:50s} split={w[1]}  "
                  f"identity {w[3]:.2f}ms -> k_fast {w[4]:.2f}ms  ({w[2]:.3f}x)")
    else:
        print("  None.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
