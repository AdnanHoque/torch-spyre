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

"""Broad permutation sweep — prefill, training, MoE, flash-attention shapes.

The narrow permutation probe (`diag_core_permutation_probe.py`) tested
4 (shape, split) configs and found identity is near-optimal among
runtime-accepted orderings, with stride2 matching reverse-emission
(1.036x on K-split) but no permutation beating that.

That sample was thin. This probe extends coverage across:

- Larger-M prefill (M=512, 2048) — does compute-dominance dilute
  any reorder effect further?
- Training-scale M (M=4096) — same, more extreme.
- MoE shapes (small per-expert M) — different shape character.
- Flash-attention-style matmuls — small K (head_dim=128), so
  K-split is structurally limited and per-call comm fraction is high.
  These are also currently many separate kernels on AIU (no fused
  attention), so wins compound across multiple matmul ops in a single
  attention forward pass.

Permutations tested (a subset that all worked on the narrow probe):
  identity, stride2, block_cyclic, bit_reverse

(reversed, antipodal, random_42 are excluded because they reproducibly
crash dxp on K-split shapes — see core_permutation_findings.md.)

Splits forced per shape so the comparison is apples-to-apples; only
splits that satisfy stick alignment for the shape are included.
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

# Permutations from the narrow probe that didn't crash dxp on K-split:
# identity (baseline), stride2 (best alt), block_cyclic (medium scatter),
# bit_reverse (max scatter, sanity for downside).
PERMUTATIONS = ["identity", "stride2", "block_cyclic", "bit_reverse"]


# (regime, label, M, N, K, split). Splits chosen for stick alignment.
# fp16 stick = 64 elements. N (output stick dim) must be divisible into
# the chosen number of N-pieces with each piece >= 1 stick.
TARGETS = [
    # ------- Dense PREFILL at increasing M -------
    ("prefill", "L3-8B q_proj M=128",  128,  4096, 4096, (1, 32, 1)),
    ("prefill", "L3-8B q_proj M=512",  512,  4096, 4096, (1, 32, 1)),
    ("prefill", "L3-8B q_proj M=2048", 2048, 4096, 4096, (1, 32, 1)),
    ("prefill", "L3-70B q_proj M=128", 128,  8192, 8192, (1, 32, 1)),
    ("prefill", "L3-70B q_proj M=512", 512,  8192, 8192, (1, 32, 1)),

    # ------- TRAINING-scale (large M) -------
    ("training", "L3-8B q_proj M=4096", 4096, 4096, 4096, (1, 32, 1)),
    ("training", "L3-70B q_proj M=2048", 2048, 8192, 8192, (1, 32, 1)),

    # ------- MoE (small per-expert M; only sticks-aligned splits) -------
    ("moe", "Mixtral expert down M=128",  128, 4096, 14336, (1, 32, 1)),
    ("moe", "MoE expert down M=512",      512, 4096, 14336, (1, 32, 1)),
    # Qwen3-MoE (N=1536=24 sticks) and DeepSeek-MoE (N=1408=22 sticks)
    # don't have any (m,n,k) with prod=32 that gives ≥1 stick of N per
    # core; their natural splits use SENCORES != 32. Skipped.

    # ------- Flash-attention-style matmuls -------
    # QK^T: M=N=seq_len, K=head_dim=128 (only 2 sticks of K). High M,
    # low K, so K-split is structurally limited to 2.
    # seq=512: N=8 sticks. (1,32,1) is invalid; use (4,8,1).
    ("attn", "QK^T seq=512 hd=128",   512,  512,  128, (4, 8, 1)),
    ("attn", "QK^T seq=2048 hd=128",  2048, 2048, 128, (1, 32, 1)),
    ("attn", "QK^T seq=4096 hd=128",  4096, 4096, 128, (1, 32, 1)),
    # AttnxV: M=seq, N=head_dim=128 (only 2 sticks). K varies. Need
    # m·n·k=32 with N split ≤ 2.
    # seq=512, K=8 sticks: (16,2,1) → M=32, N=1 stick.
    ("attn", "AV seq=512 hd=128",     512,  128, 512,  (16, 2, 1)),
    # seq=2048, K=32 sticks: (1,2,16) → K=2 sticks/core, N=1 stick.
    ("attn", "AV seq=2048 hd=128",    2048, 128, 2048, (1, 2, 16)),
    # seq=4096, K=64 sticks: (1,2,16) → K=4 sticks/core.
    ("attn", "AV seq=4096 hd=128",    4096, 128, 4096, (1, 2, 16)),

    # ------- K-split mixed (regime where stride2 wins on narrow probe) -------
    ("kmix", "L3-70B q_proj K-split (4,1,8)", 128, 8192, 8192,  (4, 1, 8)),
    ("kmix", "L3-8B MLP down K-split (4,1,8)", 128, 4096, 14336, (4, 1, 8)),
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
    print("# Broad permutation sweep — prefill / training / MoE / attention\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")
    print(f"# Permutations tested: {PERMUTATIONS}\n")

    all_rows = []
    for regime, label, M, N, K, target in TARGETS:
        print(f"### [{regime}] {label}  M={M} N={N} K={K}  split={target}")
        baseline_ms = None
        rows = []
        for perm in PERMUTATIONS:
            ms, err = _compile_and_bench(M, N, K, target, perm)
            if err:
                print(f"  {perm:14s}: ERR {err[:50]}")
                rows.append((perm, None, None))
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
        all_rows.append((regime, label, target, rows))
        print()

    # --- summary table grouped by regime ---
    print("\n## Summary — speedup vs identity\n")
    for regime in ("prefill", "training", "moe", "attn", "kmix"):
        regime_rows = [r for r in all_rows if r[0] == regime]
        if not regime_rows:
            continue
        print(f"### {regime.upper()}\n")
        print("| shape | split | " + " | ".join(PERMUTATIONS) + " |")
        print("|---|---|" + "---:|" * len(PERMUTATIONS))
        for _r, label, target, rows in regime_rows:
            rel_by = {p: rel for (p, _ms, rel) in rows}
            cells = " | ".join(
                ("err" if rel_by.get(p) is None else f"{rel_by[p]:.3f}x")
                for p in PERMUTATIONS
            )
            print(f"| {label} | {target} | {cells} |")
        print()

    # --- find candidates worth replicating ---
    print("## Candidates worth replicating (≥2% over identity)\n")
    cands = []
    for regime, label, target, rows in all_rows:
        for perm, ms, rel in rows:
            if perm == "identity" or rel is None:
                continue
            if rel >= 1.02:
                cands.append((regime, label, target, perm, rel))
    if cands:
        for c in cands:
            print(f"  [{c[0]}] {c[1]} {c[2]}  perm={c[3]}  rel={c[4]:.3f}x")
    else:
        print("  None.")
    print()

    # --- regression catalogue ---
    print("## Regressions (≤0.95) for negative-result archive\n")
    regs = []
    for regime, label, target, rows in all_rows:
        for perm, ms, rel in rows:
            if perm == "identity" or rel is None:
                continue
            if rel <= 0.95:
                regs.append((regime, label, target, perm, rel))
    if regs:
        for r in regs:
            print(f"  [{r[0]}] {r[1]} {r[2]}  perm={r[3]}  rel={r[4]:.3f}x")
    else:
        print("  None.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
