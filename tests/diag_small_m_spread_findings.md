# Small-M decode-regime spread analysis

Exhaustive split-space search over Llama 3.1/3.2 + DeepSeek V3
+ Granite 3.x linear-layer shapes at decode batch sizes (M ∈ {1, 32, 128}).
84 unique shapes after dedup. Every shape measured under all 21
ordered (m, n, k) splits with m·n·k = 32, plus identity / k_fast
emission for the k>1 family.

For each shape the probe reports the best of four categories:

- **pure-M**: (32, 1, 1) identity — current production planner default
- **best k=1 mixed**: best of {pure-M, pure-N, mixed M+N} with k=1
- **best k>1 + id**: best of K-split family with identity emission
- **best k>1 + kf**: best of K-split family with k_fast emission

Then picks the overall global winner. Run config:
WARMUP=3, ITERS=12, dtype=fp16, SENCORES=32.

Source: `tests/diag_small_m_spread_driver.py`,
raw output in `tests/diag_small_m_spread_results.txt`.

## Aggregate

84/84 shapes measured cleanly, 0 errors.

| | M=1 | M=32 | M=128 |
|---|---:|---:|---:|
| Shapes | 28 | 28 | 28 |
| Min speedup vs pure-M | 1.00× | 1.03× | 1.02× |
| Median speedup | 1.02× | 3.12× | 3.10× |
| **Geomean speedup** | **1.03×** | **2.60×** | **2.58×** |
| Max speedup | 1.13× | 3.65× | 3.64× |

At M=1 the gains are slim (parallelism is already pinned: only K
can be split usefully). At M=32 and M=128 the global optimum is
2.5-3× ahead of pure-M.

## Winner-category histogram

Across all 84 shapes:

| category | wins |
|---|---:|
| **k=1 mixed (M+N split)** | **30** (36%) |
| k>1 + id (1, n, k) | 16 (19%) |
| k>1 + kf (m, n, k) mixed | 15 (18%) |
| k>1 + kf (1, n, k) | 13 (15%) |
| k>1 + id (m, n, k) mixed | 8 (10%) |
| pure-M | 2 (2%) |

Per-shard:

| category | M=1 | M=32 | M=128 |
|---|---:|---:|---:|
| k=1 mixed | 1 | 12 | 17 |
| k>1+id (1, n, k) | 14 | 2 | 0 |
| k>1+kf (1, n, k) | 11 | 1 | 1 |
| k>1+kf mixed | 0 | 7 | 8 |
| k>1+id mixed | 0 | 6 | 2 |
| pure-M | 2 | 0 | 0 |

## Key finding — the k=1 mixed-M+N family wins most often

At **M=32 and M=128**, the global empirical optimum is most often a
mixed-M+N split with no K-split — overwhelmingly `(4, 8, 1)`,
occasionally `(2, 16, 1)` or `(16, 2, 1)`. 29 of 56 shapes (52%) at
M=32 / M=128 prefer this family.

This split is **not in the PR 1933 heuristic's candidate set**
(the heuristic only considers `(1, n_split, k_split>1)`). On those
shapes the PR's pick is a *local* optimum (better than pure-M) but
not the *global* optimum.

The PR is still a clear net win vs main:

- Production planner today picks pure-M on these shapes.
- PR's `(1, n, k>1)` consistently beats pure-M.
- But the empirically optimal `(4, 8, 1)`-family also beats pure-M,
  often by more — and is reachable by a planner that can split on
  M+N without K-split.

This corroborates the earlier exhaustive-split finding
(`diag_exhaustive_split_findings.md`) on a 7× larger shape suite
that now includes Llama 3.1/3.2 + DeepSeek V3 + Granite 3.x.

## When does k_fast emission actually matter

k_fast emission only changes the K-split family. Its contribution
shows up most clearly in two regimes:

- **M=1 K-split shapes (11/28 shapes):** kf wins over id at the
  same `(1, n, k>1)` split. Speedups are slim (1.01-1.13×) because
  M=1 has very little compute to cover the latency in the first
  place.
- **M=32 / M=128 triple-mixed splits (15/56 shapes):** kf wins on
  splits like `(4, 4, 2)`, `(2, 4, 4)`, `(2, 2, 8)`. These are
  shapes where K-split is the right family but the optimal split
  has m > 1 — outside the PR heuristic's `(1, n, k)` form.

Combined that's 26/84 shapes (31%) where k_fast emission gives a
strict win at the chosen split.

## Top-10 speedups

| speedup | shape | (M, N, K) | winning split |
|---:|---|---|---|
| 3.65× | Granite 3 8B gate/up_proj | (32, 12800, 4096)  | k=1 mixed (4, 8, 1) |
| 3.64× | Llama 3.1 70B q_proj/o_proj | (128, 8192, 8192)  | k=1 mixed (4, 8, 1) |
| 3.64× | Granite 3 8B down_proj | (128, 4096, 12800) | k=1 mixed (4, 8, 1) |
| 3.61× | Llama 3.1 70B q_proj/o_proj | (32, 8192, 8192)   | k>1+id (4, 4, 2) |
| 3.57× | DeepSeek V3 o_proj | (128, 7168, 16384) | k>1+kf (4, 4, 2) |
| 3.56× | Llama 3.1 405B kv_proj | (32, 2048, 16384) | k=1 mixed (4, 8, 1) |
| 3.55× | Granite 3 8B gate/up_proj | (128, 12800, 4096) | k=1 mixed (4, 8, 1) |
| 3.54× | DeepSeek V3 o_proj | (32, 7168, 16384)  | k>1+kf (4, 2, 4) |
| 3.54× | Llama 3.1 405B kv_proj | (128, 2048, 16384) | k=1 mixed (4, 8, 1) |
| 3.50× | Granite 3 8B down_proj | (32, 4096, 12800)  | k>1+id (4, 4, 2) |

Granite 3 8B unique shapes appear in 3 of the top 10 — the
12800-wide intermediate is a sweet spot for the (4, 8, 1) M+N split.

## Where the heuristic gives nothing

Two shapes (both M=1) hit pure-M as the global winner — no split is
materially better:

- Llama 3.1 8B q_proj/o_proj (1, 4096, 4096): all categories tie at 0.28 ms
- Llama 3.2 1B gate/up_proj (1, 8192, 2048): all categories tie at 0.28 ms

These are correctness-preserving null cases for the heuristic.

## Implication for PR 1986

- **Refactored heuristic preserves wins** vs pure-M baseline on
  all shapes where it fires (verified separately in
  `diag_k_fast_combined_findings_normalized.md`).
- **There remains material headroom** the PR's heuristic doesn't
  capture: the k=1 mixed-M+N family wins 30/84 shapes globally and
  29/56 at M=32/128.
- A follow-up that lets the planner consider mixed-M+N splits
  (within the existing work_distribution priority logic, not as a
  separate heuristic) would close most of this gap. Not a
  prerequisite for shipping PR 1986, since today's planner doesn't
  pick those splits either — it's a delta vs the empirical
  optimum, not a regression vs main.
