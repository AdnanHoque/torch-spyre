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

"""Multicast core_id permutation probe — Phase 0 verification.

Generalizes the k_fast permutation idea (PR 1932) from K-collaborator
chains to ARBITRARY sharing groups. Under split (m, n, k>1), k_fast
packs cores sharing a PSUM chain (k-collaborators) adjacent. Under
m·n splits like (8, 4, 1), the analogous question is: can we pack
cores sharing an HMI-fetched B chunk (n-collaborators) or an HMI-
fetched A chunk (m-collaborators) adjacent?

Current default emission for matmul iteration space [M, N, K] with
split (m, n, 1):
  m_slice = core_id % m
  n_slice = (core_id // m) % n
  k_slice = (core_id // (m·n)) % k

This already packs n-sharing groups (cores with fixed n_slice) at
adjacent core_ids — under (8, 4, 1), cores 0..7 share B chunk for
n_slice=0. So default is optimal for B broadcast.

But m-sharing groups (cores with fixed m_slice) are SPREAD across
the ring: cores {0, 8, 16, 24} share A chunk for m_slice=0. For
shapes where A is comparable to or bigger than B, packing m-sharing
groups adjacent could win.

This probe tests: does the permutation matter, and which sharing
axis dominates for what shapes?

Three permutations under (8, 4, 1):
  identity:   default. Packs n-sharing adjacent.
  m_adjacent: perm[c] = (c % 4) * 8 + (c // 4). Packs m-sharing adjacent.
  reversed:   perm[c] = (num_cores - 1) - c. Spreads everything (control).

Usage:
    python tests/diag_multicast_core_perm.py
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
import sys

import torch
from sympy import Symbol

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
from torch_spyre._inductor import core_division as _core_div  # noqa: E402
from torch_spyre._inductor.codegen import compute_ops as _co  # noqa: E402


WARMUP = 3
ITERS = 8
DTYPE = torch.float16


# ---- split forcing (same as prior probes) ----------------------------

_orig_multi = _core_div.multi_dim_iteration_space_split


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
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


# ---- permutation patching --------------------------------------------

_orig_generate_sdsc = _co.generate_sdsc


@contextmanager
def _force_perm(perm):
    """Override generate_sdsc to apply a core_id permutation.

    perm[c] is the logical core that physical core c will execute.
    """
    def _patched(sdsc_spec):
        result = _orig_generate_sdsc(sdsc_spec)
        # Substitute: physical core c -> work slice for logical perm[c]
        new_mapping = {
            str(c): {
                str(dim): int(expr.subs({Symbol("core_id"): perm[c]}))
                for dim, expr in sdsc_spec.core_id_to_work_slice.items()
            }
            for c in range(sdsc_spec.num_cores)
        }
        result[sdsc_spec.opfunc]["coreIdToWkSlice_"] = new_mapping
        return result

    _co.generate_sdsc = _patched
    try:
        yield
    finally:
        _co.generate_sdsc = _orig_generate_sdsc


# ---- permutation generators ------------------------------------------

def perm_identity(num_cores: int) -> list[int]:
    """No permutation. Default emission."""
    return list(range(num_cores))


def perm_m_adjacent(m: int, n: int, num_cores: int = 32) -> list[int]:
    """Pack m-sharing groups adjacent.

    Default: physical core c → work slice (m_slice = c % m, n_slice = (c // m)).
    Result: cores sharing same n are at consecutive ids; m-sharing spread.

    M-adjacent: physical core c → work slice (m_slice = c // n, n_slice = c % n).
    Result: cores sharing same m are at consecutive ids; n-sharing spread.

    To achieve this via permutation:
      perm[c] = (c % n) * m + (c // n)
    """
    return [(c % n) * m + (c // n) for c in range(num_cores)]


def perm_reversed(num_cores: int) -> list[int]:
    """Reversed mapping (control — both axes partially scrambled)."""
    return [(num_cores - 1) - c for c in range(num_cores)]


def perm_random(num_cores: int, seed: int = 42) -> list[int]:
    """Random permutation (worst-case control)."""
    import random
    rng = random.Random(seed)
    p = list(range(num_cores))
    rng.shuffle(p)
    return p


# ---- benchmark machinery ---------------------------------------------

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


def _compile_and_bench(M, N, K, split, perm):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(split), _force_perm(perm):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(split), _force_perm(perm):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


# ---- main ------------------------------------------------------------

def main() -> int:
    print("# Multicast core_id permutation probe — Phase 0\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    # Test shapes — pick varied A/B byte ratios
    SHAPES = [
        # (label, M, N, K, A_bytes, B_bytes, ratio_B_over_A)
        ("wide-B  M=128 (LLM-standard, B>>A)", 128, 8192, 8192,
         128 * 8192 * 2, 8192 * 8192 * 2, "32×"),
        ("wide-B  M=256",                       256, 8192, 8192,
         256 * 8192 * 2, 8192 * 8192 * 2, "16×"),
        ("square M=1024 (A≈B)",                1024, 1024, 4096,
         1024 * 4096 * 2, 4096 * 1024 * 2, "1×"),
        ("wide-A M=4096 (A>>B, atypical)",     4096, 256, 4096,
         4096 * 4096 * 2, 4096 * 256 * 2, "0.06×"),
    ]

    # Splits to test (all (m·n=32, k=1) configurations)
    SPLITS = [
        ((8, 4, 1), "(8,4,1)"),
        ((4, 8, 1), "(4,8,1)"),
    ]

    # Permutations
    def make_perms(m, n):
        return [
            ("identity",  perm_identity(32)),
            ("m_adj",     perm_m_adjacent(m, n)),
            ("reversed",  perm_reversed(32)),
            ("random",    perm_random(32)),
        ]

    print("| shape | B/A ratio | split | permutation | wall ms |")
    print("|---|---|---|---|---:|")

    rows = []
    for label, M, N, K, A_b, B_b, ratio in SHAPES:
        for split, split_label in SPLITS:
            m, n, _ = split
            for perm_name, perm in make_perms(m, n):
                ms, err = _compile_and_bench(M, N, K, split, perm)
                if err:
                    print(f"| {label} | {ratio} | {split_label} | "
                          f"{perm_name} | ERR: {err[:40]} |")
                    continue
                rows.append((label, ratio, split_label, perm_name, ms))
                print(f"| {label} | {ratio} | {split_label} | "
                      f"{perm_name} | {ms:.3f} |")
    print()

    # Per-(shape, split) comparison
    print("## Per-shape best/worst permutation\n")
    grouped = {}
    for label, ratio, split, perm_name, ms in rows:
        grouped.setdefault((label, split), []).append((perm_name, ms))
    for (label, split), entries in grouped.items():
        entries.sort(key=lambda x: x[1])
        best = entries[0]
        worst = entries[-1]
        spread = (worst[1] - best[1]) / best[1] * 100
        print(f"  {label} {split}: best = {best[0]} ({best[1]:.2f} ms), "
              f"worst = {worst[0]} ({worst[1]:.2f} ms), spread = {spread:.1f}%")
    print()

    print("## Reading guide\n")
    print(
        "  - Spread > 10%: permutation matters → torch_spyre lever exists.\n"
        "  - Spread < 5%: ring placement doesn't significantly affect wall.\n"
        "  - Identity wins on wide-B: default is optimal for LLM matmuls.\n"
        "  - m_adj wins on wide-A: A-broadcast is the bottleneck for those shapes.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
