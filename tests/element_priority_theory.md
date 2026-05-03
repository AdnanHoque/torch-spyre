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

The naive count assumes every byte transits HMI (the on-chip DRAM
interface) independently per core. In practice, Spyre's HMI sits on
the on-chip ring, so cross-core operand sharing and DRAM streaming
**compete for the same ring bandwidth**. Phase 0 measurements showed
effective bandwidth exceeding LPDDR5 peak by 1.4–3.4× on certain
splits — that can only happen if cores share operands across the ring
rather than each pulling separately from HMI.

The dominant first-order story is therefore: **pure-N reduces the
total bytes that have to go through the HMI-ring path.** With pure-N,
the small input A (a few MB) is the redundantly-needed operand; with
pure-M, the large weight B (hundreds of MB) is. Pushing fewer total
bytes through HMI lets per-core compute proceed without stalling on
DRAM, regardless of whether those bytes are eventually fanned out via
ring-share or streamed per-core.

The per-axis analysis we ran on Phase 1.0 data showed pure-N
empirically beats pure-M by ~30% at fixed split count
(`(1,n,1)/(m,1,1) = 0.70× median` across big shapes). That gap is
mostly explained by the operand-size argument above. There is also
likely a topology component — adjacent core IDs are adjacent on the
ring (16×2 core layout, ring wrapped through HMI/QGI), so neighbor
sharing is contiguous along whichever axis is fast-changing in the
core-ID emission — but we couldn't isolate the topology effect from
the operand-size effect with the measurements we have, and the
core-ordering reorder sweep we ran later showed the topology lever is
flat for production-sized shapes (HMI dominates).

So the speedup is best understood as a single mechanism:
**operand-size asymmetry on the HMI-bottlenecked ring.** Pure-N is
the natural way to put the 32× redundancy on the small operand. Pure-M
would put it on the large operand, saturating HMI for much longer.

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
   span exceeds the 256 MB hardware limit, so `must_split_vars`
   forces n≥2 before priority runs. The pre-split N is filtered out
   of `it_space_remaining` upstream, leaving only M as a remaining
   output dim — so the heuristic's "≥2 output dims" guard fails and
   it doesn't fire. The default planner's pick stands, which is also
   the empirical best for this shape.

These bucketings are what give the heuristic **zero regressions across
the production shape catalog**.

## Why core-ordering tweaks don't add anything on top

A natural follow-up was: with the operand-size win already captured,
could reordering the `core_id → slice` mapping squeeze out more by
putting neighbor-shared operands on physically adjacent ring cores?
We tested this with a `core_emission_reverse` flag on a separate
branch and saw flat (±1%) wall-time across 13 production shapes plus
12 forced mixed splits.

Two architectural facts explain the flat result:

1. **HMI is on the ring.** For all the production shapes, weights are
   far larger than per-core scratchpad and must stream from DRAM
   through HMI. The same bytes go through HMI regardless of which
   cores end up sharing what — reordering doesn't change total HMI
   traffic, so it doesn't change wall time.
2. **Overlapped input fetch is built into the kernel templates.** The
   weight-stationary dataflow already issues cross-core operand
   fetches in chunks concurrently with PT compute, with soft-syncs
   that don't gate compute on the fetch completing. Whatever ring
   topology effect a manual reorder might have had is already being
   hidden by this overlap.

Net: at production scale, the lever exists but the wall-time floor
underneath it is set by HMI bandwidth and template-level overlap, not
by sharing topology. Reordering would only matter for shapes that fit
entirely in LX (no HMI streaming) AND have a mixed split where the
shared operand is small enough for ring-broadcast to actually fire.
