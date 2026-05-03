# Why does element-priority make matmul faster on Spyre?

A first-principles writeup of the speedup mechanism behind the
`output_element_priority` heuristic.

## The bug in the default planner

For matmul `C[M, N] = A[M, K] @ B[K, N]`, the default planner ranks
output dimensions by **stick-adjusted iteration-space size** before
greedily handing them cores. For fp16 matmul:

- M is non-stick → `M_iter = M` (e.g. 128 iteration units)
- N is innermost stick (64 elem/stick) → `N_iter = N / 64`
  (e.g. 4096 elem → 64 iteration units)

So the planner sees `M=128 > N=64`, gives M priority, and spends all
32 cores on M. The captured pick is `(m=32, n=1, k=1)`. But N is
**32× larger than M in elements** — that's the dimension we should
have split.

This is purely a unit mismatch. Stick-adjustment is correct for
deciding *valid divisors* (each core must hold a whole number of
sticks) but wrong for deciding *priority*.

## Why N-split is cheaper — naive DDR arithmetic

Phase 0 measured the cross-core DDR formula for each split:

| split | A reads | B reads | C writes |
|---|---|---|---|
| `(m, n, k)` | n·|A| | m·|B| | k·|C| |

For LLM prefill, B is the large tensor (`|B| = K·N`), A is small
(`|A| = M·K` with M small), and C is medium. So:

- `(32, 1, 1)` total ≈ **32·|B|** + |A| + |C| — 32× the largest tensor
- `(1, 32, 1)` total ≈ **|B|** + 32·|A| + |C| — 32× the smallest tensor

For L3-70B q_proj (M=128, N=8192, K=8192): pure-M total = ~4100 MB,
pure-N total = ~196 MB. **A 21× reduction in cross-core DDR.** With
the LPDDR5 channel as the bottleneck, wall time tracks DDR — hence
the measured 1.61× wall-time speedup.

This is the dominant first-order effect. The whole speedup table is
explained by it.

## Why the speedup is larger than naive DDR predicts

The naive count assumes every byte transits DDR independently. Spyre
has **cross-core operand sharing**: Phase 0 measurements showed
effective bandwidth exceeding LPDDR5 peak by 1.4–3.4× on certain
splits. That can only happen if cores share rather than re-fetch.

Sharing isn't symmetric across axes. In Spyre's row-major core
emission ([superdsc.py:131-149](torch_spyre/_inductor/codegen/superdsc.py#L131-L149)),
core IDs are mapped as:

- cores `[0..n-1]` cover one M-band, varying along N
- core ID `m·n` jumps to the next M-band

So **N-axis cores are physically adjacent**: they all need the same A
row, broadcast cheaply across the interconnect. **M-axis cores are
spaced `n` apart**: they each need full B, which would need to
broadcast across the whole topology. The per-axis analysis we did on
Phase 1.0 data confirmed this — the median ratio
`(1,n,1)/(m,1,1) = 0.70×` across big shapes.

So the speedup decomposes as:
1. **Naive DDR** (~80% of the win): N-split puts the 32× redundancy on
   the small tensor A instead of the big tensor B.
2. **Sharing topology** (~20%): N-axis sharing is contiguous and
   efficient; M-axis sharing is non-contiguous and degrades.

Both effects point the same way, which is why pure-N consistently
beats pure-M in the measured table.

## Why mixed splits emerge for some shapes

For two shapes (L3-8B / L3-70B GQA kv_proj prefill, both N=1024) the
heuristic picks `(2, 16, 1)` rather than `(1, 32, 1)`. Reason:
N=1024 with 64-elem fp16 sticks gives only 16 valid N-divisors before
falling under the stick-alignment limit. The remaining 2 cores fall
through to M. This isn't a different optimum — it's the same N-first
priority, capped by stick-alignment validity.

## When the heuristic intentionally does nothing

Six of 13 shapes show 1.00× speedup. These fall into three buckets:

1. **Launch-floor-bound** (decode shapes, GQA TP=8): wall time ≈ 3 ms
   from per-launch overhead. Any split looks the same.
2. **Default already picks well** (L3-8B MLP gate/up): when
   `N_iter > M_iter` even after stick adjustment, default ranking
   already prefers N. Heuristic is a no-op.
3. **Span-pre-split forces the choice** (L3-70B MLP down): N's per-core
   span exceeds 256 MB, so `must_split_vars` forces n≥2 before priority
   runs. The pre-split N is filtered out of `it_space_remaining`
   upstream, leaving only M as a remaining output dim — so the
   heuristic's "≥2 output dims" guard fails and it doesn't fire. The
   default planner's pick stands, which is also the empirical best
   for this shape.

These bucketings are what give the heuristic **zero regressions across
the production shape catalog**.
