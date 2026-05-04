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

"""Sweep M across real-world LLM matmul shapes to find the k_fast band.

The narrow-N kv-shape sweep showed that (1,16,2)+k_fast beats the
planner's pure-M pick when M ∈ [32, 1024] for kv N=1024 K=8192.
At M=2048, pure-M wins. So the band is real but bounded.

This probe runs the same comparison on the SHAPES we actually measured
the original "wins" on (L3-70B kv_proj, Mixtral 8x7B kv_proj, DSv3
o_proj, DSv3 down_proj, DSv3 q_a_proj) — but varying M from 32 to 2048
— to characterize where each shape benefits in production-like decode
or short-prefill regimes.

For each shape × M:
  A: natural-pick (planner's choice today)
  B: pure-M (32, 1, 1) [fallback if pure-N invalid]
  D: shape-canonical K-split + k_fast
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
ITERS = 12
DTYPE = torch.float16

# Real-world shapes from the DSv3 + cross-model probe.
# (label, N, K, k_split_to_test)
SHAPES = [
    ("L3-70B kv_proj",      1024,  8192, (1, 16, 2)),
    ("Mixtral 8x7B kv_proj", 1024,  4096, (1, 16, 2)),
    ("DSv3 o_proj",          7168, 16384, (1, 16, 2)),
    ("DSv3 down_proj",       7168,  2048, (1, 16, 2)),
    ("DSv3 q_a_proj",        1536,  7168, (1, 8, 4)),
]

M_VALUES = [32, 128, 512, 1024, 2048]


# ---- machinery (same shape as prior probes) ---------------------------

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


def _compile_and_bench(M, N, K, force_target, perm_name):
    ts_config.core_id_permutation = perm_name
    ts_config.core_emission_reverse = False
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(force_target):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(force_target):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:90]}"


def main() -> int:
    print("# k_fast win-band sweep across real-world LLM workloads\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    results = {}  # (label, M, config) -> ms

    for label, N, K, k_split in SHAPES:
        print(f"## {label}  (N={N}, K={K}, K-split tested = {k_split})\n")
        print(f"| M | A:natural | B:pure-M | D:K-split+k_fast | "
              f"D vs A | D vs best |")
        print(f"|---:|---:|---:|---:|---:|---|")
        for M in M_VALUES:
            a_ms, _err_a = _compile_and_bench(M, N, K, None, "identity")
            b_ms, _err_b = _compile_and_bench(M, N, K, (32, 1, 1), "identity")
            d_ms, _err_d = _compile_and_bench(M, N, K, k_split, "k_fast")

            results[(label, M, "A")] = a_ms
            results[(label, M, "B")] = b_ms
            results[(label, M, "D")] = d_ms

            valid_others = [v for v in (a_ms, b_ms) if v is not None]
            best_other = min(valid_others) if valid_others else None

            def fmt(v):
                return f"{v:.3f}" if v is not None else "ERR"

            d_vs_a = (
                f"{a_ms / d_ms:.3f}x"
                if a_ms is not None and d_ms is not None else "—"
            )
            if d_ms is not None and best_other is not None:
                vs_best = (
                    f"**WINS** ({best_other / d_ms:.3f}x)"
                    if d_ms < best_other * 0.98
                    else f"{best_other / d_ms:.3f}x"
                )
            else:
                vs_best = "—"

            print(f"| {M} | {fmt(a_ms)} | {fmt(b_ms)} | {fmt(d_ms)} | "
                  f"{d_vs_a} | {vs_best} |")
        print()

    # --- aggregate report: where does k_fast win? ---
    print("\n## Where does k_fast WIN over the planner?\n")
    print("Reporting (shape, M) where D < A by ≥2%:\n")
    found_wins = []
    for label, N, K, _ks in SHAPES:
        for M in M_VALUES:
            a = results.get((label, M, "A"))
            d = results.get((label, M, "D"))
            if a is None or d is None:
                continue
            if d < a * 0.98:
                pct = (a - d) / a * 100
                found_wins.append((label, N, K, M, a, d, pct))

    if found_wins:
        print("| shape | N | K | M | natural ms | k_fast ms | wins by |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for label, N, K, M, a, d, pct in found_wins:
            print(f"| {label} | {N} | {K} | {M} | {a:.3f} | {d:.3f} | "
                  f"{pct:.1f}% |")
    else:
        print("  None.")
    print()

    # --- per-shape M-band summary ---
    print("\n## Per-shape M-band where k_fast helps\n")
    for label, N, K, ks in SHAPES:
        winning_ms = []
        for M in M_VALUES:
            a = results.get((label, M, "A"))
            d = results.get((label, M, "D"))
            if a is None or d is None:
                continue
            if d < a * 0.98:
                winning_ms.append((M, a, d))
        if winning_ms:
            mlist = [str(m) for m, _, _ in winning_ms]
            print(f"  {label}: M ∈ {{{', '.join(mlist)}}} "
                  f"(K-split tested = {ks})")
            for M, a, d in winning_ms:
                print(f"    M={M:5d}  natural={a:.3f}  k_fast={d:.3f}  "
                      f"({(a-d)/a*100:.1f}% saved)")
        else:
            print(f"  {label}: NO WINS at any tested M (K-split tested = {ks})")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
