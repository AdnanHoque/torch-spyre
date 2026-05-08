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

"""Phase 3 extension probe — test candidate splits on shapes where the
PR 1933 heuristic currently skips.

The 3-way campaign found 6 shapes in the suite where PR 1933's
heuristic returns None (n_sticks ≥ 32 or M out of band) and the
planner falls back to pure-M. This probe tests three candidate
splits on each of those shapes to determine whether the heuristic
should be extended to capture them:

  - (1, 16, 2) + kf — what a relaxed-gate heuristic would pick (k=2)
  - (1, 8, 4)  + kf — middle of (1, n, k) family
  - (1, 4, 8)  + kf — the verified small-M-wide-N winner

For each shape, the winner is the candidate that beats pure-M by the
largest margin (or none if all candidates regress).

Decision tree:
  - If (1, 16, 2)+kf wins everywhere → simplest extension: relax
    n_sticks gate for M ≤ 64
  - If only (1, 4, 8)+kf wins → heuristic must flip n iteration
    order at small M
  - If different shapes have different winners → need a richer rule

Usage:
    python tests/diag_k_fast_extension_candidates.py
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
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
from torch_spyre._inductor import core_division as _core_div  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16


@dataclass
class Shape:
    label: str
    M: int
    N: int
    K: int


# Shapes from the 3-way campaign where heuristic skipped.
SHAPES: list[Shape] = [
    Shape("L3-70B q_proj M=32",      32, 8192, 8192),
    Shape("DSv3 gate_proj M=32",     32, 18432, 7168),
    Shape("L3-70B q_proj M=128",    128, 8192, 8192),
    Shape("L3-70B q_proj M=512",    512, 8192, 8192),
    Shape("DSv3 down_proj M=128",   128, 7168, 18432),
    Shape("L3-70B kv_proj M=2048", 2048, 1024, 8192),
]


# ---- machinery ----------------------------------------------------

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
    if target is None:
        yield
        return
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


@contextmanager
def _kfast_emission(enabled: bool):
    prev = ts_config.core_id_k_fast_emission
    ts_config.core_id_k_fast_emission = enabled
    try:
        yield
    finally:
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


def _compile_and_bench(M, N, K, split, kfast):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _kfast_emission(kfast), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _kfast_emission(kfast), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:50]}"


def _is_valid(M, N, K, split):
    m, n, k = split
    if M % m or N % n or K % k:
        return False, "div"
    if (N // n) % 64 != 0:
        return False, "stick"
    return True, ""


# ---- main ----------------------------------------------------------

CANDIDATES = [
    ((1, 16, 2), "(1,16,2)+kf"),
    ((1,  8, 4), "(1,8,4)+kf"),
    ((1,  4, 8), "(1,4,8)+kf"),
]


def main() -> int:
    print("# Phase 3 extension candidates — heuristic-skipped shapes\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, k_fast emission ON\n")
    print("| shape | (M,N,K) | pure-M | (1,16,2) | (1,8,4) | (1,4,8) | "
          "winner | speedup |")
    print("|---|---|---:|---:|---:|---:|---|---:|")

    winners: dict[tuple[int, int, int], tuple] = {}

    for s in SHAPES:
        # pure-M baseline
        pm_ms, pm_err = _compile_and_bench(s.M, s.N, s.K, (32, 1, 1), False)
        cand_results = []
        for split, label in CANDIDATES:
            ok, why = _is_valid(s.M, s.N, s.K, split)
            if not ok:
                cand_results.append((split, label, None, why))
                continue
            ms, err = _compile_and_bench(s.M, s.N, s.K, split, True)
            cand_results.append((split, label, ms, err))

        # Find winner among candidates that ran
        valid = [(spl, lbl, ms) for (spl, lbl, ms, _) in cand_results
                  if ms is not None]
        if not valid or pm_ms is None:
            winner_label = "—"
            speedup_str = "—"
        else:
            best_split, best_label, best_ms = min(valid, key=lambda t: t[2])
            if best_ms < pm_ms * 0.97:  # require ≥3% to call a winner
                winner_label = best_label
                speedup_str = f"{pm_ms / best_ms:.2f}×"
                winners[(s.M, s.N, s.K)] = (best_split, best_label, best_ms, pm_ms)
            else:
                winner_label = "pure-M"
                speedup_str = "—"

        def _f(x):
            return f"{x:.2f}" if x is not None else "—"

        cells = [_f(ms) for (_, _, ms, _) in cand_results]
        print(f"| {s.label} | ({s.M},{s.N},{s.K}) | {_f(pm_ms)} | "
              f"{cells[0]} | {cells[1]} | {cells[2]} | "
              f"{winner_label} | {speedup_str} |")

    print()
    print("## Decision summary\n")
    if not winners:
        print("  No candidate splits beat pure-M on any heuristic-skipped shape.")
        print("  → Don't extend the heuristic. Ship PR 1933 as-is.")
    else:
        print(f"  {len(winners)}/{len(SHAPES)} skipped shapes have a winning candidate.")
        # Tally winning split families
        family_counts: dict[str, int] = {}
        for v in winners.values():
            family_counts[v[1]] = family_counts.get(v[1], 0) + 1
        print("  Winning splits by family:")
        for fam, n in sorted(family_counts.items(), key=lambda kv: -kv[1]):
            print(f"    {fam}: {n} shape(s)")
        print()
        # Decide extension shape
        if len(family_counts) == 1:
            fam = list(family_counts.keys())[0]
            print(f"  → Single winning family ({fam}). Extension is a "
                  "single-rule addition: drop the n_sticks gate (or M-conditional "
                  "version), keep the existing pick logic.")
        else:
            print("  → Multiple winning families. Heuristic needs a richer rule "
                  "(possibly per-(M, N, K)-regime selection of n).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
