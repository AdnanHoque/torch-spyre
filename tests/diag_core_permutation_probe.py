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

"""Test whether sequential core_id placement is empirically optimal.

The previous core-emission probes showed sequential (default) emission
isn't beaten by a simple dim-iteration reverse. But that doesn't prove
sequential is optimal — it only rules out one alternative ordering.

This probe tests several non-trivial permutations of physical core IDs:

  identity      [0, 1, 2, ..., 31]                  (baseline)
  reversed      [31, 30, ..., 0]                    (direction symmetry test)
  stride2       [0, 2, 4, ..., 30, 1, 3, ..., 31]   (interleaved half-rings)
  block_cyclic  [0, 16, 1, 17, 2, 18, ...]          (adjacent pairs on opposite halves)
  antipodal     [16, 17, ..., 31, 0, 1, ..., 15]    (halves swapped)
  bit_reverse   [0, 16, 8, 24, 4, 20, ...]          (recursive halving)
  random_42     (fixed-seed shuffle)                (upper bound on disruption)
  random_7      (different seed)                    (replication of random)

If sequential beats all of these → row-major-by-core-id is optimal,
not just by argument but empirically. If a permutation BEATS sequential
on any shape → we've found a new lever and the "ring lever is dead"
claim collapses entirely. If random is no slower than sequential →
ring topology effects don't matter at all on this hardware.
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

PERMUTATIONS = [
    "identity",
    "reversed",
    "stride2",
    "block_cyclic",
    "antipodal",
    "bit_reverse",
    "random_42",
    "random_7",
]

# (label, M, N, K, split). Splits picked from the prior K-split + reorder
# findings so we have known-non-zero baselines to compare against.
TARGETS = [
    # Production-default pure-N splits — if no perm beats identity here,
    # default is likely already near-optimal for the planner's typical pick.
    ("L3-70B q_proj prefill (pure-N)",     128, 8192, 8192,  (1, 32, 1)),
    ("L3-8B  q_proj prefill (pure-N)",     128, 4096, 4096,  (1, 32, 1)),
    # K-split mixed where reverse-emission already wins ~3.6%
    ("L3-70B q_proj prefill (K-split)",    128, 8192, 8192,  (4, 1, 8)),
    # Output-reorder regime where reverse-emission already wins ~2.1%
    ("L3-70B MLP down prefill (output)",   128, 8192, 28672, (16, 2, 1)),
]


# ---- machinery ---------------------------------------------------------

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


def _compile_and_bench(M: int, N: int, K: int, target, perm: str):
    ts_config.core_id_permutation = perm
    ts_config.core_emission_reverse = False  # we're testing perm, not reverse
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
    print("# Core-ID permutation probe — is sequential really optimal?\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")
    print("## Procedure\n")
    print(
        "For each (shape, split), run all 8 permutations back-to-back.\n"
        "Identity is the baseline. Any permutation faster than identity\n"
        "by >2% on any shape is a real new finding (random_42 vs random_7\n"
        "lets us bound noise — they should be similar to each other).\n"
    )

    all_rows = []
    for label, M, N, K, target in TARGETS:
        print(f"### {label}  M={M} N={N} K={K}  split={target}\n")
        baseline_ms = None
        rows = []
        for perm in PERMUTATIONS:
            ms, err = _compile_and_bench(M, N, K, target, perm)
            if err:
                print(f"  {perm:14s}: ERR {err}")
                continue
            if perm == "identity":
                baseline_ms = ms
                rel = 1.000
                marker = "(baseline)"
            else:
                rel = baseline_ms / ms if baseline_ms else 0.0
                marker = ("✓ FASTER" if rel >= 1.02 else
                          "✗ SLOWER" if rel <= 0.98 else "~ tie")
            print(f"  {perm:14s}: {ms:.3f} ms  rel={rel:.3f}x  {marker}")
            rows.append((perm, ms, rel))
        all_rows.append((label, target, rows))
        print()

    # --- compact summary ---
    print("## Summary table\n")
    header_perms = ", ".join(p for p in PERMUTATIONS)
    print(f"|shape | split | {' | '.join(PERMUTATIONS)} |")
    print("|---|---|" + "---:|" * len(PERMUTATIONS))
    for label, target, rows in all_rows:
        ms_by_perm = {p: ms for (p, ms, _r) in rows}
        cells = " | ".join(
            (f"{ms_by_perm.get(p, 0):.3f}" if ms_by_perm.get(p) else "err")
            for p in PERMUTATIONS
        )
        print(f"| {label} | {target} | {cells} |")
    print()

    print("## Speedup vs identity\n")
    print(f"|shape | split | {' | '.join(PERMUTATIONS)} |")
    print("|---|---|" + "---:|" * len(PERMUTATIONS))
    for label, target, rows in all_rows:
        rel_by_perm = {p: r for (p, _ms, r) in rows}
        cells = " | ".join(
            (f"{rel_by_perm.get(p, 0):.3f}x" if rel_by_perm.get(p) else "err")
            for p in PERMUTATIONS
        )
        print(f"| {label} | {target} | {cells} |")
    print()

    # --- verdict ---
    print("## Verdict\n")
    real_wins = []
    real_losses = []
    for label, target, rows in all_rows:
        for perm, ms, rel in rows:
            if perm == "identity":
                continue
            if rel >= 1.02:
                real_wins.append((label, target, perm, rel))
            elif rel <= 0.98:
                real_losses.append((label, target, perm, rel))

    if real_wins:
        print(f"  {len(real_wins)} (shape, perm) combos beat identity by ≥2%:")
        for w in real_wins:
            print(f"    - {w[0]} {w[1]}  perm={w[2]}  rel={w[3]:.3f}x")
        print(
            "  IDENTITY IS NOT EMPIRICALLY OPTIMAL. We need to dig into "
            "WHY a non-sequential ordering wins, and consider whether "
            "shipping a different default makes sense."
        )
    else:
        print("  No permutation beat identity by ≥2% on any shape.")
        print(f"  ({len(real_losses)} permutations were ≥2% SLOWER than identity.)")
        print(
            "  Sequential placement is empirically optimal among the "
            "permutations tested. The earlier 'row-major is optimal' "
            "claim is now supported by data, not just by argument."
        )

    # Random consistency check
    print()
    rand_diffs = []
    for label, target, rows in all_rows:
        ms_by_perm = {p: ms for (p, ms, _r) in rows}
        if "random_42" in ms_by_perm and "random_7" in ms_by_perm:
            r1, r2 = ms_by_perm["random_42"], ms_by_perm["random_7"]
            ident = ms_by_perm.get("identity", r1)
            rand_diffs.append((label, abs(r1 - r2) / ident))
    if rand_diffs:
        avg = sum(d for _, d in rand_diffs) / len(rand_diffs)
        print(f"  Random-vs-random (noise floor estimate): avg |Δ|/ident = "
              f"{avg*100:.2f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
