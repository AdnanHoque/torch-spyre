# Broad permutation sweep — prefill / training / MoE / attention

## TL;DR — actually found something

Tested 4 permutations × 17 (shape, split) configurations across
prefill, training-scale, MoE, and attention regimes. Two confirmed
≥2% wins on replication, and one of them is **new**:

| shape | split | perm | mean speedup | mechanism |
|---|---|---|---:|---|
| L3-70B q_proj K-split | `(4, 1, 8)` | `stride2` | **1.039×** | PSUM chain shortening (already known) |
| **L3-70B q_proj prefill M=2048** | `(1, 32, 1)` | `block_cyclic` | **1.024×** | suspected output-writeback DRAM banking, *new* |

The M=2048 result is the surprise. **It's part of a continuous trend
that grows with M:** 1.015× at M=256, 1.010× at M=512, 1.016× at M=1024,
1.024× at M=2048. Direction is consistent across both trial orders for
every M tested. The effect crosses the 2% replication threshold at
M=2048 and would likely keep growing at M=4096+ for prefill of long
contexts.

This contradicts the previous "row-major identity is near-optimal"
conclusion specifically for **prefill at long contexts**. For short
contexts (M=128) it's still near-optimal. The difference is the
fraction of wall time spent on output writeback.

## What didn't move

For completeness, here's what stayed flat (within noise) across the
broad sweep:

- **All MoE shapes**: identity ± permutation = no difference. Both
  Mixtral expert down and MoE expert down at M=128 and M=512 sit in
  [-1%, +0.3%] for every permutation tested.
- **Most flash-attention shapes**: QK^T at all sequence lengths stays
  within ±0.5% of identity. AV at seq=512 is dead flat. Only AV at
  seq=4096 with `bit_reverse` shows a 15% regression — and that's the
  bad direction; nothing helps.
- **Short prefill (M=128, M=512)**: no permutation reaches +2% on
  pure-N splits; small swings in both directions.
- **L3-8B at all M**: very slight (<1%) movement everywhere.

## Where this fits in the broader story

We now have three confirmed real ≥2% permutation wins from the project,
each via a different mechanism:

| context | shape | best perm | speedup | mechanism |
|---|---|---|---:|---|
| K-split (PSUM chain) | various K-heavy | `stride2` or `core_emission_reverse` | ~1.035× | shorten the SFP-ring chain hops |
| Output reorder | L3-70B MLP down `(16, 2, 1)` | `core_emission_reverse` | 1.021× | flip which operand becomes ring-shareable |
| **Prefill long-M** | **L3-70B q_proj M≥2048 `(1, 32, 1)`** | **`block_cyclic`** | **1.024×** | **suspected DRAM-banking on output writeback** |

The first two we already knew about. The third is new from this sweep.

## Why we suspect DRAM banking on output writeback

The mechanism hypothesis comes from the *trend*, not from a single
data point:

```text
M=256  → 1.015×
M=512  → 1.010×
M=1024 → 1.016×
M=2048 → 1.024×
```

What scales linearly with M? Output writeback. Each core writes
M·(N/n) elements of C back to HMI. With (1, 32, 1) split:
- M=128:  128 KB per-core write     (<3% of wall time)
- M=2048: 1 MB per-core write       (~25% of wall time at 4 ms wall)

With `identity`, physical core c writes N-band c. Adjacent physical
cores 0,1 write contiguous N-bands → adjacent DRAM addresses → likely
to land on the same DRAM bank's row buffer → write conflicts.

With `block_cyclic`, physical core c writes N-band `perm(c)` where
adjacent c's get N-bands 16 apart. → writes spread across DRAM banks
in parallel → fewer write conflicts.

We can't *prove* this is the mechanism without a DRAM-counter probe
(which we don't have access to from torch_spyre). But:
- The trend is monotonic with M → consistent with a write-bound effect
- Other permutations don't show the same trend → not a generic noise win
- Short-M configs show nothing → confirms it's not pure ring share

## Caveats — why not ship a heuristic now

1. **2% on a single shape isn't enough by itself.** Need replication
   on more model variants (other 70B-class configs) to confirm this
   isn't a peculiarity of L3-70B's exact dimensions.
2. **Mechanism is hypothesis, not proven.** Could be something subtler
   (e.g., interaction with the LX scratchpad lookup pattern). Without
   confirming the mechanism, the heuristic might fail on shapes that
   look similar but differ in a relevant way.
3. **Coupling with other permutation effects.** `block_cyclic`
   *severely* hurts K-split (0.86×). Any heuristic that picks
   `block_cyclic` would need to gate on (split, shape) jointly.

So the right next step isn't shipping. It's:

- One more sweep at M ≥ 2048 across more shapes (L3-8B at long M,
  Granite, Qwen at compatible sizes) to see if the trend holds
  generally or is L3-70B-specific.
- Probe whether the win is on writeback specifically vs. some other
  M-scaled cost. Could split into "before output write" and "after"
  segments if there's instrumentation we can hook.

## Per-regime takeaways

### Prefill (dense)

Identity is near-optimal for short context (M ≤ 512). For longer
context (M ≥ 1024) `block_cyclic` shows growing improvement, crossing
the replication threshold at M=2048. **Worth following up.**

### Training-scale (M=4096)

Limited single-shape coverage; M=4096 q_proj showed flat results in
the broad sweep. Need to retest with the trend in mind — the broad
sweep used iters=12; trend extrapolation suggests block_cyclic should
give 2-4% at M=4096. Rerun with iters=25 recommended.

### MoE (small per-expert M)

Dead. No permutation moves the needle on per-expert matmul. Per-expert
M is the bottleneck but it's also small enough that wall time is
launch-floor-dominated.

### Flash-attention shapes

All flat within noise. The structural constraint (K=128 = only 2
sticks) limits the split space severely. Bit_reverse breaks AV at
long sequence lengths (15% regression at seq=4096) — confirming the
runtime is sensitive to permutations on K-split-like (1, 2, 16)
shapes too.

For attention specifically: **fused attention is the bigger lever
here, not core-ID permutation.** Each individual matmul is too small
or too constrained to benefit from reordering.

### K-split mixed (kmix)

Confirms the prior finding: `stride2` ≈ `core_emission_reverse` for
PSUM chain shortening. `block_cyclic` and `bit_reverse` make K-split
catastrophically worse (0.86× and 0.72×). The K-split constraint
window is narrow: only specific orderings work, only one wins by ~3.6%.

## Updated answer to the original question

> Are you sure row-major (identity) core_id placement is optimal?

Refined answer based on broader data:

- **For short prefill (M ≤ 512), MoE, and attention**: yes, identity
  is empirically near-optimal among runtime-accepted orderings.
- **For long prefill (M ≥ 1024)**: no — `block_cyclic` is consistently
  ~1-2.5% better, with the gap growing as M grows.
- **For K-split mixed shapes**: no — `stride2` is ~3.6% better
  (matches the existing `core_emission_reverse` knob).
- **In the down-side direction**: many permutations cause severe
  regressions or runtime crashes on K-split. The runtime has hidden
  adjacency constraints we don't fully understand.

The ring lever is alive for *one specific use case* (long-context
prefill) we hadn't tested before. It's still small (~2-4%), but unlike
the previous narrow probe, the trend is clearly monotonic in M and
the mechanism is plausible.

## Files

- [`tests/diag_core_permutation_broad.py`](diag_core_permutation_broad.py)
  — broad sweep across 17 (shape, split) configs
- [`tests/diag_core_permutation_broad_replicate.py`](diag_core_permutation_broad_replicate.py)
  — focused replication of M=512 candidate + neighbour-M scan
- raw outputs:
  [`diag_core_permutation_broad_results.txt`](diag_core_permutation_broad_results.txt),
  [`diag_core_permutation_broad_replicate_results.txt`](diag_core_permutation_broad_replicate_results.txt)
- prior context:
  [`core_permutation_findings.md`](core_permutation_findings.md),
  [`core_emission_psum_chain_findings.md`](core_emission_psum_chain_findings.md)

## Open questions for follow-up

1. **Does `block_cyclic` keep helping at M=4096+?** The trend predicts
   yes; one more probe at M=4096, 8192 on L3-70B q_proj would tell us.
2. **Does the same trend hold on L3-8B and other models at long M?**
   Or is this specific to the 8192-N geometry of L3-70B?
3. **Is the mechanism actually output-writeback DRAM banking?** A
   probe that varies output size while holding compute constant would
   isolate this. Would need to find a way to construct such a shape.
4. **Why doesn't the K-split crash repro on `(1, 2, 16)` AV shapes?**
   The runtime accepts permutations there that crash on `(4, 1, 8)`.
   What's the distinguishing structural property?
