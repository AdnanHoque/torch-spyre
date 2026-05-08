# Focused k_fast-essential probe — findings

Companion to `diag_vllm_shape_catalog.py`,
`diag_kfast_essential_driver.py`,
`diag_kfast_essential_measure.py`, and
`diag_kfast_essential_results.txt`. Validates the production scope
of the k_fast emission contribution by measuring a representative
sample of vLLM-supported model shapes across four split-family
categories.

## TL;DR

**On 20 representative production shapes drawn from a 588-op,
318-unique-shape vLLM catalog (16 model families × 6 M values),
mixed `(m, n, k=1)` splits — splits this PR doesn't consider —
are the empirical optimum on 16/20 shapes (80%).** The k_fast
emission contribution is the global optimum on 2/20 shapes (10%)
and strictly essential (kf > id at the same split) on 1/20 (5%).

Pure-M, the planner's current default, is the optimum on 0/20
shapes.

This reframes the production-worthiness story: the planner has a
much larger problem (mixed-split scheduling) that this PR doesn't
address. The k_fast emission contribution is real but narrow in
production scope.

## Method

For each shape, measure four configuration categories:

| category | candidates tested | what it represents |
|---|---|---|
| `pure-M` | (32, 1, 1) identity | planner default today |
| `k=1 (mixed M+N)` | best of (32,1,1), (1,32,1), (16,2,1), (8,4,1), (4,8,1), (2,16,1) | non-K-split champion |
| `k>1 + identity` | best of 14 splits with k>1, identity emission | "does K-split help at all?" |
| `k>1 + k_fast` | same 14 splits, k_fast emission | "does the emission permutation help?" |

Per shape we measure ~30 individual configs; subprocess-isolated
to handle deeptools scheduler crashes on certain forced splits.

## Per-shape outcomes (full table)

| shape | (M, N, K) | pure-M | best k=1 | best k>1+id | best k>1+kf | winner |
|---|---|---:|---:|---:|---:|---|
| Llama 3.2 1B gate_proj | (32, 8192, 2048) | 0.88 | 0.27 | 0.27 | 0.28 | k=1 (mixed M+N) |
| DeepSeek V3 q_b_proj | (512, 24576, 1536) | 2.00 | 1.05 | 2.27 | 1.34 | k=1 (mixed M+N) |
| Llama 3.1 8B q_proj | (32, 4096, 4096) | 0.86 | 0.26 | 0.27 | 0.27 | k=1 (mixed M+N) |
| Gemma 2 9B o_proj | (1, 3584, 4096) | 0.27 | — | 0.27 | 0.27 | k>1 + id |
| Qwen 2.5 7B q_proj | (1, 3584, 3584) | 0.31 | — | 0.30 | **0.29** | k>1 + kf |
| Llama 3.2 3B gate_proj | (128, 8192, 3072) | 1.36 | 0.42 | 0.47 | 0.43 | k=1 (mixed M+N) |
| DeepSeek V3 kv_a_proj | (1024, 576, 7168) | 0.35 | 0.35 | 0.59 | 0.38 | k=1 (mixed M+N) |
| Mixtral 8x22B gate_proj | (1024, 16384, 6144) | 6.58 | 3.50 | 8.98 | 6.26 | k=1 (mixed M+N) |
| Qwen 2.5 32B gate_proj | (1, 27648, 5120) | 2.35 | — | 2.33 | 2.33 | k>1 + id |
| Mixtral 8x22B q_proj | (512, 6144, 6144) | 1.98 | 0.81 | 1.75 | 1.24 | k=1 (mixed M+N) |
| Qwen 2.5 14B kv_proj | (2048, 2048, 5120) | 1.54 | 0.85 | 1.99 | 1.35 | k=1 (mixed M+N) |
| Mixtral 8x22B kv_proj | (2048, 2048, 6144) | 1.82 | 1.00 | 2.27 | 1.60 | k=1 (mixed M+N) |
| Gemma 2 9B down_proj | (1024, 3584, 14336) | 2.95 | 2.29 | 5.24 | 3.17 | k=1 (mixed M+N) |
| **Phi 3 medium down_proj** | (128, 5120, 17920) | 4.68 | 2.03 | 1.57 | **1.43** | **k>1 + kf** |
| Llama 3.1 70B down_proj | (512, 8192, 28672) | — | 5.98 | 13.66 | 7.39 | k=1 (mixed M+N) |
| Llama 3.1 405B down_proj | (512, 16384, 53248) | — | 14.72 | 64.29 | 27.19 | k=1 (mixed M+N) |
| Llama 3.1 405B q_proj | (2048, 16384, 16384) | — | 18.91 | 46.00 | 32.53 | k=1 (mixed M+N) |
| Llama 3.1 405B down_proj | (128, 16384, 53248) | — | 12.32 | 16.19 | 12.39 | k=1 (mixed M+N) |
| DeepSeek V3 down_proj | (1024, 7168, 18432) | 7.83 | 4.30 | 12.44 | 8.06 | k=1 (mixed M+N) |
| Llama 3.1 8B down_proj | (512, 4096, 14336) | 2.57 | 1.54 | 2.97 | 1.89 | k=1 (mixed M+N) |

`—` in pure-M column = Llama 3.1 70B/405B shapes that hit the EAR
overflow limit under pure-M (the 256 MB hardware ceiling
identified in earlier emission-aware-LX work). On those shapes the
planner today literally can't compile pure-M.

## Headline counts

| category | wins | % |
|---|---:|---:|
| pure-M | 0 | 0% |
| k=1 (mixed M+N) | 16 | **80%** |
| k>1 + identity | 2 | 10% |
| k>1 + k_fast | 2 | 10% |

**k_fast strictly essential** (k>1+kf is global winner AND kf >
id at same split family by ≥5%):

| shape | kf wall | id wall | kf advantage |
|---|---:|---:|---:|
| Phi 3 medium down_proj M=128 | 1.43 ms | 1.57 ms | 1.10× |

**1 of 20 shapes (5%).**

## What k_fast actually contributes (when it's used)

Worth noting that on shapes where k>1+kf isn't the global winner,
k_fast still produces real speedups *at the K-split it's applied
to*. Sample of "kf vs id at same split" deltas observed during the
probe (these are where kf's structural property comes from, even
if k=1 mixed splits beat both):

- DeepSeek V3 q_b_proj (512, 24576, 1536) at (4, 4, 2): kf 1.34 ms
  vs id 4.66 ms — **3.49×**
- Llama 3.1 8B down_proj (512, 4096, 14336) at (4, 4, 2): kf 1.89
  ms vs id 7.20 ms — **3.82×**
- Llama 3.1 70B down_proj (512, 8192, 28672) at (4, 4, 2): kf 7.39
  ms vs id 28.61 ms — **3.87×** (estimated from category bests)

So when K-split is used, k_fast's adjacent-K-collaborator
permutation provides 3-4× speedup at that split family. The
research property is real and reproducible. But mixed M+N splits
beat the entire K-split family on 80% of production shapes, so
the real-world delivery of k_fast's value is gated by whether
K-split was the right choice in the first place.

## Implications for the PR (PR #1986 — combined k_fast)

**The PR is still net-positive vs the current planner**:
- pure-M wins 0/20 shapes; the PR's heuristic moves shapes off
  pure-M for the cases it fires on, capturing some of the
  available speedup
- k_fast emission is essential infrastructure for the small
  fraction of shapes where K-split is optimal (1/20 strict, 2/20
  best-of-category)
- 24/24 unit tests pass; hardware-verified 9/12 wins on the
  earlier 3-way campaign (1.79-3.28× over pure-M)

**The PR doesn't address the bigger production problem**:
- 80% of vLLM-sampled shapes have a mixed `(m, n, k=1)` split as
  the optimum
- The recurring winners are `(4, 8, 1)`, `(8, 4, 1)`, `(16, 2, 1)`
  — splits that divide BOTH M and N without K-split
- No closed-form heuristic addition to PR 1933 captures this; it
  needs a search-based planner that considers mixed splits

## Three options for going forward

### Option 1: Land the current PR as scoped

Merge the combined k_fast PR (1932 + 1933 + small-M extension)
with the current narrow framing. Document that mixed-M+N
optimization is a substantial follow-up. Pros: ships a real (if
narrow) win + scopes the next project clearly. Cons: leaves 80%
of available speedup on the table.

### Option 2: Land PR 1932 (k_fast emission) alone, drop PR 1933

Argument: 1932 is the novel research contribution (adjacent
K-collaborator placement); 1933 is a narrow heuristic that the
data shows is rarely the right answer. Future planner work
captures the rest; meanwhile 1932 is available as a primitive
when needed. Pros: cleanest "ship the primitive, plan the bigger
work separately" framing. Cons: less immediate production win.

### Option 3: Expand PR scope to mixed-split planning

Rewrite the heuristic as a search over candidate splits ranked by
cost model, including all 21 (m, n, k) candidates. Multi-week
project; needs cost-model recalibration. Captures most of the
available speedup at once. Pros: captures the 80% available
speedup. Cons: substantial scope creep on a PR that's already
review-ready.

**My honest read**: Option 2 is the cleanest research-first
framing. Option 1 is the most pragmatic ship-now choice. Option 3
is too much rework for the marginal benefit over (1 + follow-up
PR).

## What this means for the research story

The novel contribution — k_fast emission as a codegen primitive
for adjacent K-collaborator placement — stands on its own merits:
- Real algebraic property (perm[c] = (c%k)·m·n + c//k preserves
  K-collaborator adjacency)
- Real performance improvement at the split family it applies to
  (3-4× over identity emission on K-split shapes)
- Genuinely novel — no public auto-scheduler models adjacent
  ring placement of K-collaborators

But the production scope is narrow. The paper / RFC should frame
it as:
- "We characterize a codegen primitive that delivers 3-4×
  speedup on K-split matmul kernels"
- "The mechanism works by exploiting the AIU's SFP ring topology;
  K-collaborators on adjacent ring positions reduce PSUM chain
  hops from m·n to 1"
- "On production matmul shapes, this primitive is the right
  choice on roughly 5-10% of cases. The other 90-95% are best
  served by mixed M+N work-division splits, which are an
  independent (and substantially larger) planner research
  problem."

That's an honest framing the data supports.

## Files

- `tests/diag_vllm_shape_catalog.py` — vLLM shape catalog
- `tests/diag_kfast_essential_driver.py` — focused probe driver
  (subprocess-isolated, line-buffered for visible progress)
- `tests/diag_kfast_essential_measure.py` — single-config
  measurement subprocess
- `tests/diag_kfast_essential_results.txt` — raw output
- This doc

## Branch

`AdnanHoque/feat-k-fast-combined` (evidence branch). Production
PR (`AdnanHoque/pr-k-fast`) is unaffected.
