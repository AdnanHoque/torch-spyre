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

"""Small-M decode-regime spread analysis — Llama + DeepSeek linear layers.

Sweeps every linear layer (q/kv/o/gate/up/down + DSv3 MLA variants)
in Llama 3.1/3.2 and DeepSeek V3 at decode batch sizes
(M ∈ {1, 32, 128}). Deduplicates by (M, N, K). For each unique shape,
runs the same 4-category focused probe as
diag_kfast_essential_driver.py:

    pure-M    : (32, 1, 1) identity (planner default)
    k=1       : best of pure-M / pure-N / mixed-(m, n, 1) family
    k>1 + id  : best of (m, n, k>1) family + identity emission
    k>1 + kf  : best of (m, n, k>1) family + k_fast emission

Reports the spread:
  - Histogram of which category wins per shape
  - Top suboptimality gaps (where current planner / PR pick are far
    from optimum)
  - Distribution of speedup vs pure-M baseline

Subprocess-isolated; line-buffered output for visible progress.

Usage:
    python tests/diag_small_m_spread_driver.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

ELEMS_PER_STICK = 64
MEASURE_SCRIPT = str(Path(__file__).resolve().parent / "diag_kfast_essential_measure.py")
TIMEOUT_S = 90

# Decode-regime batch sizes.
M_VALUES = (1, 32, 128)


@dataclass(frozen=True)
class StdConfig:
    name: str
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def kv_proj_out(self) -> int:
        return 2 * self.n_kv_heads * self.head_dim

    @property
    def q_proj_out(self) -> int:
        return self.n_heads * self.head_dim


@dataclass(frozen=True)
class DSV3Config:
    name: str = "DeepSeek V3"
    hidden: int = 7168
    intermediate: int = 18432
    n_heads: int = 128
    head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_rope_dim: int = 64
    qk_nope_dim: int = 128
    v_head_dim: int = 128


LLAMA_MODELS = [
    StdConfig("Llama 3.1 8B",   4096, 14336, 32, 8, 128),
    StdConfig("Llama 3.1 70B",  8192, 28672, 64, 8, 128),
    StdConfig("Llama 3.1 405B", 16384, 53248, 128, 8, 128),
    StdConfig("Llama 3.2 1B",   2048, 8192,  32, 8, 64),
    StdConfig("Llama 3.2 3B",   3072, 8192,  24, 8, 128),
]

DSV3 = DSV3Config()


def llama_ops(cfg: StdConfig, M: int) -> list[tuple]:
    H, I = cfg.hidden, cfg.intermediate
    Nq = cfg.q_proj_out
    Nkv = cfg.kv_proj_out
    return [
        (cfg.name, "q_proj/o_proj", M, H,   Nq if Nq != H else H),  # symmetric
        (cfg.name, "q_proj",        M, Nq,  H),
        (cfg.name, "kv_proj",       M, Nkv, H),
        (cfg.name, "o_proj",        M, H,   Nq),
        (cfg.name, "gate/up_proj",  M, I,   H),
        (cfg.name, "down_proj",     M, H,   I),
    ]


def dsv3_ops(M: int) -> list[tuple]:
    cfg = DSV3
    return [
        (cfg.name, "q_a_proj",  M, cfg.q_lora_rank, cfg.hidden),
        (cfg.name, "q_b_proj",  M, cfg.n_heads * (cfg.qk_rope_dim + cfg.qk_nope_dim),
         cfg.q_lora_rank),
        (cfg.name, "kv_a_proj", M, cfg.kv_lora_rank + cfg.qk_rope_dim, cfg.hidden),
        (cfg.name, "kv_b_proj", M, cfg.n_heads * (cfg.qk_nope_dim + cfg.v_head_dim),
         cfg.kv_lora_rank),
        (cfg.name, "o_proj",    M, cfg.hidden, cfg.n_heads * cfg.v_head_dim),
        (cfg.name, "gate/up",   M, cfg.intermediate, cfg.hidden),
        (cfg.name, "down_proj", M, cfg.hidden, cfg.intermediate),
    ]


def build_unique_shapes() -> list[tuple]:
    """Return a list of (label, M, N, K) for unique shapes."""
    seen: dict[tuple[int, int, int], tuple] = {}
    for M in M_VALUES:
        for cfg in LLAMA_MODELS:
            for entry in llama_ops(cfg, M):
                _, op, m, n, k = entry
                key = (m, n, k)
                if key not in seen:
                    seen[key] = (f"{cfg.name} {op}", m, n, k)
        for entry in dsv3_ops(M):
            _, op, m, n, k = entry
            key = (m, n, k)
            if key not in seen:
                seen[key] = (f"{DSV3.name} {op}", m, n, k)
    return list(seen.values())


# Candidate split families (same as focused probe).
K1_CANDIDATES = [
    (32, 1, 1), (1, 32, 1),
    (16, 2, 1), (8, 4, 1), (4, 8, 1), (2, 16, 1),
]
KGT1_CANDIDATES = [
    (1, 16, 2), (1, 8, 4), (1, 4, 8), (1, 2, 16), (1, 1, 32),
    (16, 1, 2), (8, 1, 4), (4, 1, 8), (2, 1, 16),
    (8, 2, 2), (4, 4, 2), (4, 2, 4), (2, 8, 2), (2, 4, 4), (2, 2, 8),
]


def _is_valid(M, N, K, split):
    m, n, k = split
    if m * n * k != 32:
        return False
    if M % m or N % n or K % k:
        return False
    if (N // n) % ELEMS_PER_STICK != 0:
        return False
    return True


def _measure(M, N, K, split, kfast):
    m, n, k = split
    try:
        result = subprocess.run(
            ["python", MEASURE_SCRIPT,
             str(M), str(N), str(K), str(m), str(n), str(k), str(int(kfast))],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1].strip()
    if last.startswith("ERR:"):
        return None
    try:
        return float(last)
    except ValueError:
        return None


def _best_in_category(M, N, K, splits, kfast):
    best = None
    for s in splits:
        if not _is_valid(M, N, K, s):
            continue
        ms = _measure(M, N, K, s, kfast)
        if ms is None:
            continue
        if best is None or ms < best[1]:
            best = (s, ms)
    return best  # (split, ms) or None


def main() -> int:
    shapes = build_unique_shapes()
    print(f"# Small-M decode-regime spread analysis\n")
    print(f"Unique shapes: {len(shapes)}  (Llama 3.1/3.2 + DeepSeek V3, M ∈ {M_VALUES})\n")
    print(f"Subprocess timeout {TIMEOUT_S}s.\n")

    print("| label | (M, N, K) | pure-M | best k=1 | best k>1+id | best k>1+kf | winner | speedup |")
    print("|---|---|---:|---:|---:|---:|---|---:|")

    summary = []
    for (label, M, N, K) in shapes:
        pm = _measure(M, N, K, (32, 1, 1), False)
        k1 = _best_in_category(M, N, K, K1_CANDIDATES, False)
        kid = _best_in_category(M, N, K, KGT1_CANDIDATES, False)
        kf = _best_in_category(M, N, K, KGT1_CANDIDATES, True)

        cands = []
        if pm is not None:  cands.append(("pure-M", pm, (32, 1, 1)))
        if k1 is not None:  cands.append(("k=1 mixed", k1[1], k1[0]))
        if kid is not None: cands.append(("k>1+id", kid[1], kid[0]))
        if kf is not None:  cands.append(("k>1+kf", kf[1], kf[0]))

        if not cands:
            print(f"| {label} | ({M},{N},{K}) | — | — | — | — | ERR | — |")
            continue

        winner_cat, winner_ms, winner_split = min(cands, key=lambda c: c[1])
        baseline = pm if pm is not None else max(c[1] for c in cands)
        speedup = baseline / winner_ms if winner_ms > 0 else 0

        def _f(c): return f"{c[1]:.2f}" if c is not None else "—"
        if isinstance(pm, float):
            pm_s = f"{pm:.2f}"
        else:
            pm_s = "—"
        k1_s = _f(k1)
        kid_s = _f(kid)
        kf_s = _f(kf)
        print(f"| {label} | ({M},{N},{K}) | {pm_s} | {k1_s} | {kid_s} | {kf_s} | "
              f"{winner_cat} {winner_split} | {speedup:.2f}× |")

        summary.append((label, M, N, K, pm, k1, kid, kf, winner_cat, winner_split,
                        winner_ms, speedup))

    # Aggregate spread
    print("\n## Winner category histogram\n")
    cat_count = {}
    for row in summary:
        cat_count[row[8]] = cat_count.get(row[8], 0) + 1
    for cat in ("pure-M", "k=1 mixed", "k>1+id", "k>1+kf"):
        n = cat_count.get(cat, 0)
        bar = "#" * n
        print(f"  {cat:<12}: {n:>3} / {len(summary)}  {bar}")

    # Speedup distribution
    speedups = sorted([r[11] for r in summary])
    if speedups:
        n = len(speedups)
        median = speedups[n // 2]
        max_su = speedups[-1]
        min_su = speedups[0]
        # geomean
        from math import log, exp
        geomean = exp(sum(log(s) for s in speedups) / n)
        print(f"\n## Speedup spread (winner vs pure-M baseline)\n")
        print(f"  shapes: {n}")
        print(f"  min:     {min_su:.2f}×")
        print(f"  median:  {median:.2f}×")
        print(f"  geomean: {geomean:.2f}×")
        print(f"  max:     {max_su:.2f}×")
        # Bucket histogram
        buckets = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 100)]
        print()
        print("  histogram:")
        for lo, hi in buckets:
            n_in = sum(1 for s in speedups if lo <= s < hi)
            label = f"  [{lo}× – {hi}×)"
            bar = "#" * n_in
            print(f"  {label:<14}: {n_in:>3}  {bar}")

    # k_fast essentiality
    print("\n## Where is k_fast STRICTLY essential?\n")
    n_essential = 0
    for row in summary:
        (label, M, N, K, pm, k1, kid, kf, winner_cat, winner_split, winner_ms, _) = row
        if winner_cat != "k>1+kf":
            continue
        if kid is None or kf is None:
            continue
        if kid[1] > kf[1] * 1.05:
            n_essential += 1
            ratio = kid[1] / kf[1]
            print(f"  {label} ({M},{N},{K}): kf {kf[1]:.2f}ms vs id {kid[1]:.2f}ms ({ratio:.2f}×)")
    if n_essential == 0:
        print("  (none)")
    print(f"\n  k_fast strictly essential on {n_essential}/{len(summary)} shapes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
