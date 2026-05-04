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

"""k_fast win-band sweep across popular vLLM-served LLM families.

Test the kv_proj (and a few interesting o_proj) shapes across the
models people actually serve. M values cover the production-relevant
band: M=32 (decode batched), M=128 (medium decode/short prefill),
M=512 (longer prefill).

The k_fast win-band on the previous sweep was M ≤ 128 → 512 for narrow-N
shapes. This probe tells us how universal that win is across model
zoos.

Each row reports natural-pick wall time vs forced-K-split + k_fast.
The K-split is chosen per shape based on N's stick count to keep the
split valid for that geometry.
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


def pick_k_split(n_sticks: int, k_sticks: int) -> tuple[int, int, int] | None:
    """Pick the largest valid (1, n, k) K-split for this shape.

    n must divide both n_sticks and 32, with k = 32 / n. We want the
    largest n possible so K-cluster is shortest under k_fast (smallest k).
    Skip if n_sticks < 1 or only k=1 split is valid (no K-split benefit).
    """
    candidates = []
    for n in (16, 8, 4, 2):
        if n_sticks % n == 0 and 32 % n == 0:
            k = 32 // n
            # Make sure K is divisible by k (each core gets ≥1 stick of K)
            if k_sticks >= k and k_sticks % k == 0:
                candidates.append((1, n, k))
    return candidates[0] if candidates else None


# (model, op, M_label, N, K, expected_K_split)
# K_split=None means "compute it from sticks"
SHAPES = [
    # ─── Llama 3.1 family (most-served foundation models) ──────────────
    ("Llama 3.1 8B",   "kv_proj",  1024,  4096, None),
    ("Llama 3.1 8B",   "o_proj",   4096,  4096, None),
    ("Llama 3.1 70B",  "kv_proj",  1024,  8192, None),
    ("Llama 3.1 70B",  "o_proj",   8192,  8192, None),
    ("Llama 3.1 405B", "kv_proj",  1024, 16384, None),

    # ─── Llama 3.2 (small models, becoming common in serving) ──────────
    ("Llama 3.2 1B",   "kv_proj",   512,  2048, None),  # head_dim=64, narrow
    ("Llama 3.2 3B",   "kv_proj",  1024,  3072, None),

    # ─── Mistral / Mixtral ─────────────────────────────────────────────
    ("Mistral 7B v0.3", "kv_proj", 1024,  4096, None),  # = Llama 3.1 8B
    ("Mixtral 8x7B",    "kv_proj", 1024,  4096, None),  # already tested
    ("Mixtral 8x22B",   "kv_proj", 1024,  6144, None),

    # ─── Qwen 2.5 (heavily used on vLLM) ───────────────────────────────
    ("Qwen 2.5 7B",  "kv_proj", 512,  3584, None),  # 4 KV heads × 128 = 512
    ("Qwen 2.5 14B", "kv_proj", 1024, 5120, None),
    ("Qwen 2.5 32B", "kv_proj", 1024, 5120, None),
    ("Qwen 2.5 72B", "kv_proj", 1024, 8192, None),

    # ─── Phi-3 ──────────────────────────────────────────────────────────
    ("Phi-3 medium", "kv_proj", 1280, 5120, None),  # 10 KV heads × 128 = 1280

    # ─── Granite (IBM) ─────────────────────────────────────────────────
    ("Granite 8B",   "kv_proj", 1024,  4096, None),
    ("Granite 34B",  "kv_proj", 1024,  8192, None),

    # ─── DeepSeek V3 (already validated) ───────────────────────────────
    ("DSv3", "o_proj",    7168, 16384, None),
    ("DSv3", "down_proj", 7168,  2048, None),
    ("DSv3", "q_a_proj",  1536,  7168, None),

    # ─── Gemma 2 (control: wide kv_proj, expect no benefit) ─────────────
    ("Gemma 2 27B",  "kv_proj", 2048,  4608, None),  # head_dim=128 × 16 = 2048

    # ─── GPT-OSS family (estimated; may need verification) ─────────────
    # gpt-oss-20b: hidden ~2880, 8 KV heads × 64 head_dim = 512
    # gpt-oss-120b (MoE): hidden 4096, 8 KV heads × 128 = 1024
    ("gpt-oss-20b",  "kv_proj_est",  512, 2880, None),
    ("gpt-oss-120b", "kv_proj_est", 1024, 4096, None),
]

M_VALUES = [32, 128, 512]


# ---- machinery --------------------------------------------------------

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
    print("# k_fast win-band across popular vLLM-served LLM families\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    results = {}  # (model, op, M, "A"/"D") -> ms

    print("\n## Per-shape per-M results\n")
    for model, op, N, K, k_split_override in SHAPES:
        n_st = N // 64
        k_st = K // 64
        k_split = k_split_override or pick_k_split(n_st, k_st)
        if k_split is None:
            print(f"### {model}  {op}  N={N}({n_st}st) K={K}({k_st}st):  "
                  f"NO valid K-split — skipping")
            continue

        print(f"### {model}  {op}  N={N}({n_st}st) K={K}({k_st}st)  "
              f"K-split={k_split}\n")
        print(f"| M | natural ms | k_fast ms | wins by |")
        print(f"|---:|---:|---:|---:|")
        for M in M_VALUES:
            a_ms, _err_a = _compile_and_bench(M, N, K, None, "identity")
            d_ms, _err_d = _compile_and_bench(M, N, K, k_split, "k_fast")
            results[(model, op, M, "A")] = a_ms
            results[(model, op, M, "D")] = d_ms

            if a_ms is None or d_ms is None:
                print(f"| {M} | ERR | ERR | — |")
                continue
            pct = (a_ms - d_ms) / a_ms * 100
            marker = " ✓" if pct >= 2 else ("" if pct > -2 else " ✗")
            print(f"| {M} | {a_ms:.3f} | {d_ms:.3f} | "
                  f"{pct:+.1f}%{marker} |")
        print()

    # --- aggregate: pivot to (model × M) summary ---
    print("\n## Compact summary — k_fast speedup (% saved over natural-pick)\n")
    print("| model / op | M=32 | M=128 | M=512 |")
    print("|---|---|---|---|")
    for model, op, N, K, _override in SHAPES:
        cells = []
        for M in M_VALUES:
            a = results.get((model, op, M, "A"))
            d = results.get((model, op, M, "D"))
            if a is None or d is None:
                cells.append("ERR")
                continue
            pct = (a - d) / a * 100
            if pct >= 2:
                cells.append(f"**+{pct:.1f}%**")
            elif pct <= -2:
                cells.append(f"-{abs(pct):.1f}%")
            else:
                cells.append(f"~{pct:+.1f}%")
        print(f"| {model} {op} | " + " | ".join(cells) + " |")
    print()

    # --- best-case finder ---
    print("\n## Highest-impact (model, M) combinations (≥5% saved)\n")
    big_wins = []
    for model, op, N, K, _override in SHAPES:
        for M in M_VALUES:
            a = results.get((model, op, M, "A"))
            d = results.get((model, op, M, "D"))
            if a is None or d is None:
                continue
            pct = (a - d) / a * 100
            if pct >= 5:
                big_wins.append((pct, model, op, M, N, K, a, d))
    big_wins.sort(reverse=True)
    if big_wins:
        print("| rank | model | op | M | N | K | natural ms | k_fast ms | saved |")
        print("|---:|---|---|---:|---:|---:|---:|---:|---:|")
        for i, (pct, model, op, M, N, K, a, d) in enumerate(big_wins[:20], 1):
            print(f"| {i} | {model} | {op} | {M} | {N} | {K} | "
                  f"{a:.3f} | {d:.3f} | **{pct:.1f}%** |")
    else:
        print("  None.")

    # --- coverage / counts ---
    print("\n## Coverage stats\n")
    total = 0
    wins_by_m = {m: 0 for m in M_VALUES}
    losses_by_m = {m: 0 for m in M_VALUES}
    for model, op, N, K, _override in SHAPES:
        for M in M_VALUES:
            a = results.get((model, op, M, "A"))
            d = results.get((model, op, M, "D"))
            if a is None or d is None:
                continue
            total += 1
            pct = (a - d) / a * 100
            if pct >= 2:
                wins_by_m[M] += 1
            elif pct <= -2:
                losses_by_m[M] += 1
    print(f"  Total (model, op, M) configs measured: {total}")
    for M in M_VALUES:
        print(f"  M={M:5d}:  k_fast wins ≥2%: "
              f"{wins_by_m[M]}, loses ≥2%: {losses_by_m[M]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
