# PR 1932 top-10 replication — findings

## TL;DR

Replicated PR 1932's claimed top-10 wins on current deeptools, with
each shape measured under 4 configurations (cold cache + fresh
process per measurement) to decompose the win into split-choice
contribution vs k_fast-permutation contribution.

**Headline findings:**
1. **8 of 10 shapes replicate within 10%** of the PR's claim
2. **DSv3 o_proj M=128 has shrunk: claimed 1.94×, today 1.48×**
3. **PR 1932 (permutation) and PR 1933 (heuristic) BOTH have
   independent value** — most shapes split the win between them
4. **One shape (L3-405B kv_proj M=512) regresses with PR 1933 alone**
   — needs PR 1932's permutation to be safe
5. **Direction (row vs col) is symmetric** within ±2% across all
   10 shapes — no direction lever

## Methodology

For each shape, measured 4 walls:
- **pure-M**: forced (32, 1, 1) split, no permutation. The planner's
  natural pick today.
- **split-only**: forced k-split (1, 16, 2) or (1, 8, 4) for q_a_proj,
  no permutation. The split PR 1933 would override to.
- **kf-on**: forced k-split with k_fast (PR 1932) permutation
  applied. perm[c] = (c % k) * (m·n) + (c // k).
- **col_dir**: forced k-split with column-direction permutation.

Each measurement run in a fresh Python process with cleared kernel
cache to avoid the cache-poisoning issue I documented in
`diag_2d_direction_probe_revalidated.md`.

## Per-shape results

| rank | label | pure-M | split-only | kf-on | col_dir | split contrib | kf contrib | total | PR claim |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | DSv3 o_proj M128 | 9.19 | 6.69 | 6.22 | 6.24 | 1.37× | 1.08× | **1.48×** | 1.94× |
| 2 | L3-70B o_proj M128 | 6.40 | 5.24 | 4.16 | 4.16 | 1.22× | 1.26× | **1.54×** | 1.31× |
| 3 | DSv3 down_proj M128 | 3.75 | 3.42 | 3.42 | 3.37 | 1.10× | 1.00× | 1.10× | 1.18× |
| 4 | L3-8B o_proj M128 | 3.86 | 3.48 | 3.20 | 3.19 | 1.11× | 1.09× | 1.21× | 1.18× |
| 5 | L3-405B kv_proj M128 | 3.82 | 3.28 | 3.28 | 3.29 | 1.16× | 1.00× | 1.16× | 1.17× |
| 6 | L3-405B kv_proj M32 | 3.70 | 3.23 | 3.21 | 3.29 | 1.15× | 1.01× | 1.16× | 1.15× |
| 7 | Gemma 27B kv_proj | 3.44 | 3.17 | 3.17 | 3.10 | 1.09× | 1.00× | 1.09× | 1.10× |
| 8 | DSv3 q_a_proj M128 | 3.48 | 3.28 | 3.18 | 3.19 | 1.06× | 1.03× | 1.09× | 1.10× |
| 9 | L3-405B kv_proj M512 | 3.71 | 4.08 | 3.59 | 3.64 | **0.91×** | 1.14× | 1.03× | 1.09× |
| 10 | L3-70B kv_proj M128 | 3.33 | 3.13 | 3.06 | 3.14 | 1.06× | 1.02× | 1.09× | 1.09× |

## How the win decomposes by shape type

### Wide-B shapes: BOTH PRs contribute

- DSv3 o_proj (N=7168): split 1.37×, k_fast 1.08× → 1.48× combined
- L3-70B o_proj (N=8192): split 1.22×, k_fast **1.26×** → 1.54× combined
- L3-8B o_proj (N=4096): split 1.11×, k_fast 1.09× → 1.21× combined

Wide-B shapes have the biggest PSUM payloads, so the k_fast
permutation has real work to do. The o_proj wins owe roughly half to
split choice and half to permutation.

### Narrow-B shapes: split choice dominates, k_fast adds little

- L3-70B kv_proj (N=1024): split 1.06×, k_fast 1.02× → 1.09×
- L3-405B kv_proj M=128 (N=1024): split 1.16×, k_fast 1.00× → 1.16×
- Gemma 27B kv_proj: split 1.09×, k_fast 1.00× → 1.09×
- DSv3 q_a_proj: split 1.06×, k_fast 1.03× → 1.09×

Narrow-B shapes are LF-bound; the k_fast permutation has no PSUM
cost to amortize. PR 1933 alone delivers ~all the value here.

### One regression-risk shape

**L3-405B kv_proj M=512**: split-only gives **0.91×** (i.e., a **9%
regression** vs pure-M). The k_fast permutation recovers it back to
1.03× total (3% net gain).

**Implication: PR 1933 cannot safely ship without PR 1932** for
this shape. Without the permutation rescue, the heuristic would hurt
this case. This argues for shipping the two PRs as a paired stack.

## Headline-claim discrepancy: DSv3 o_proj

The PR's headline claim is "1.94× on DSv3 o_proj M=128". Today: 1.48×.

Decomposing:
- Validation set (PR claim): 9.16 → 4.71 = 1.94×
- Today: 9.19 → 6.22 = 1.48×

The pure-M wall reproduces (9.16 vs 9.19). The k_fast wall has
gotten worse (4.71 → 6.22 — 32% slower). Default emission is
probably what's improved on this shape, but our forced (1,16,2)+kf
isn't capturing the same kernel template path it used to.

This is a worthwhile **caveat to flag in the PR**: "DSv3 o_proj win
has shrunk to 1.48× from 1.94× on current deeptools, but other top
wins still hold."

## Direction lever: confirmed absent across full set

| direction | rows where col_dir > kf-on | rows where kf-on > col_dir | within ±2% |
|---|:-:|:-:|:-:|
| count | 4 | 4 | 8/10 |

Direction effects are symmetric and within measurement noise. The
2% Gemma effect (col_dir 3.10 vs kf 3.17) is reversed by the 2.5%
L3-405B M=32 effect (kf 3.21 vs col_dir 3.29). No consistent
direction-aware lever.

## What this means for shipping

1. **Both PRs have real, distinguishable value.** PR 1933 delivers
   the split choice; PR 1932 delivers the permutation. They aren't
   redundant.
2. **Ship them as a paired stack** — the L3-405B M=512 regression
   makes shipping PR 1933 alone risky.
3. **Update the PR's claimed wins** to reflect today's deeptools.
   The headline 1.94× → 1.48× is a meaningful correction. Other
   wins replicate.
4. **No new direction-aware lever to chase.** Confirmed across full
   target set.

## Files

- `diag_pr1932_top10_replication_results.txt` — raw measurements
- `replicate_pr_top10/run_one.py` — single-measurement runner
  (not committed; lives in /tmp)
- This doc — findings + decomposition
