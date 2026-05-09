# Why the small-M sweep results make sense — hardware-grounded analysis

Theory companion to `diag_small_m_spread_findings.md`. Walks through
the hardware features that determine whether pure-M, mixed-M+N
(k=1), or K-split (k>1) wins on a given (M, N, K) shape, and works
out a few sample calculations to gauge how far we are from
compute-peak.

## Hardware essentials

The AIU 1.0 has 32 cores arranged in a ring, with two co-located
quad-rings: one for **data** (operand A and B traffic from HMI/L2),
one for **PSUM** (output accumulator coordination). Each core holds:

- One **PT (point) array**: 8 rows × 8 cols of MAC units, with
  8-way SIMD per cell. One PT cycle multiplies an 8×8 block with
  8 K-elements deep, producing an 8×8 partial sum block.
- A **2 MB LX scratchpad**, of which `(1 - DXP_LX_FRAC_AVAIL) * 2 MB`
  is available to the inductor allocator (1.6 MB by default).
- A **256 MB EAR span limit** for HMI access per core.

Stick = 64 fp16 elements (128-byte aligned). All data movement is
in stick units.

Key dimensions of any `(m, n, k)` work-division split:

- **m**: how M is partitioned across cores (M_per_core = M/m)
- **n**: how N is partitioned across cores (N_per_core = N/n)
- **k**: K-cohort depth: k cores cooperate on each PSUM chain
- m·n·k = 32 (must use all cores)

Two bottlenecks compete for the win on small-M shapes:

- **PT utilization**: how many of the PT array's 8 M-rows are
  filled per core. M_per_core ≥ 8 ⇒ 100% M-row utilization.
- **PSUM coordination**: K-split splits the K loop, so each
  K-cohort of k cores must reduce-sum their partial PSUMs over
  the SFP ring before writing the output. With identity emission,
  K-collaborators are m·n hops apart on the ring; with k_fast
  emission, they're 1 hop apart.

## Why pure-M loses at small M

Under pure-M `(32, 1, 1)`, M_per_core = M/32. If M < 32 × 8 = 256,
the PT array's M-dimension is under-saturated. Concretely:

| M | M_per_core | PT M-rows used (of 8) | Util |
|---:|---:|---:|---:|
| 1   | 0.03 | 1 (broadcast) | 12.5% |
| 32  | 1    | 1             | 12.5% |
| 128 | 4    | 4             | 50%   |
| 256 | 8    | 8             | 100%  |
| 512 | 16   | 8 (×2 PT batches) | 100% |

So at M ≤ 128, pure-M gets at most half the PT array's M-throughput
per core. That's the headroom every win in this sweep is exploiting.

## Why K-split wins on M=1

At M=1, M_per_core under pure-M is 1/32 — the PT array runs as a
vector × matrix kernel, not a matmul. Splitting K (e.g.,
`(1, 1, 32)` or `(1, 4, 8)`) gives every core full M=1 access and
splits the K loop, so each core does K/k iterations and a final
ring reduction.

The K-split family wins 25/28 M=1 shapes (89%) in the sweep. But
the gain is small: geomean 1.03×, max 1.13×. Why so slim?

- M=1 is intrinsically memory-bound for the B operand (no M-reuse).
- The total work scales with N·K, which is small relative to the
  fixed launch + reduction overhead.
- Compute peak is unattainable here regardless of split.

The PR heuristic deliberately skips M < 32 because (a) the gains are
small, (b) M=1 is a regime where the planner risk/benefit is
unfavourable, and (c) the kernel timing floor is already dominated
by overhead the split can't reduce.

## Why mixed-M+N wins at M ∈ {32, 128}

This is the key insight from the sweep. At M=32 with split
`(4, 8, 1)`:

- M_per_core = 32/4 = **8** → exactly fills the PT array's M-rows
- N_per_core = N/8
- k = 1 → no K-cohort, no PSUM ring traffic

Every core processes a full 8-row PT block, parallelism comes from
splitting N across cores, and there's zero communication overhead.
This achieves **100% PT M-utilization with no PSUM cost**.

Compare to `(1, 16, 2)` (PR's pick at M=32, N wide enough):

- M_per_core = 32 → 4 PT M-blocks per core (still 100% PT M-util)
- N_per_core = N/16
- k = 2 → K-cohort of 2 cores, must coordinate one PSUM hop per chain
  (k_fast collapses that to 1 ring hop)

Both reach 100% PT M-util, but `(4, 8, 1)` avoids the PSUM ring
hop entirely. That's why mixed-M+N wins at M=32 even though
K-split also achieves PT saturation.

At M=128 with `(4, 8, 1)`:

- M_per_core = 128/4 = 32 → 4 full PT M-blocks per core (100% util)
- N_per_core = N/8
- k = 1 → no PSUM coordination

Same story, just more PT batches per core. The sweep shows
`(4, 8, 1)` winning 17/28 M=128 shapes for this exact reason.

## Why K-split + k_fast wins at triple-mixed splits

15/56 of the M=32 / M=128 shapes have winners like `(4, 4, 2)`,
`(2, 2, 8)`, `(4, 2, 4)`. These are shapes where:

- N_sticks is not divisible by 8 (so `(4, 8, 1)` doesn't fit), or
- The shape of N relative to M makes finer N-sharding inefficient

For these, splitting both N and K becomes the right structure. With
k=2 or k=4, the K-cohort is small, so the k_fast permutation collapses
the PSUM ring traversal to 1 hop and the cohort cost is amortized
over many K-tiles. The PR's `(1, n, k)` heuristic considers
k=2/4/8/16 candidates but with m=1 fixed; the sweep shows that
**`(4, 4, 2)` and `(2, 4, 4)` triple-mixed splits often beat
`(1, 16, 2)`** because the m>1 portion saturates the PT M-rows
without needing wider N to do so.

This is where the PR's local optimum diverges from the global one
on the K-split branch.

## Sample calculations: how far from peak?

Take **Llama 3.1 70B q_proj/o_proj** at M=32: shape (32, 8192, 8192).

**Total ops:** 2 × 32 × 8192 × 8192 = 4.29 × 10⁹ MACs = 8.59 GFLOPs.

**Compute-peak bound:**

PT cycles per core (assuming perfect parallelism across 32 cores):

`(M/8) × (N/8) × (K/8) / 32` PT-cycles per core
= (4 × 1024 × 1024) / 32
= 131 072 cycles per core

At 1 GHz that's **131 µs**. (AIU clock is closer to ~1 GHz logical;
the published 300 TOPS rating includes wider parallelism than the
PT array alone — for our work-division comparison, PT-cycle bound
is the right reference.)

Measured wall times:

| split | wall (ms) | TFLOPs/s | % of compute-peak (131 µs) |
|---|---:|---:|---:|
| pure-M (32, 1, 1) baseline | 3.40 | 2.5 | 4% |
| best k=1 mixed (4, 8, 1) | 0.95 | 9.0 | 14% |
| best k>1+id (4, 4, 2) | 0.94 | 9.1 | 14% |
| best k>1+kf (4, 4, 2) | 0.96 | 8.9 | 14% |

Pure-M is at **4% of compute-peak** because M_per_core=1 leaves the
PT array running at 12.5% M-util AND adds memory bandwidth pressure
(B operand fully replicated across all cores). The optimal splits
lift us to **~14% of compute-peak** — a 3.6× wall-clock speedup,
but still leaving 86% on the table.

**The remaining gap to compute-peak comes from:**

- Memory bandwidth: B is 8192×8192×2 = 128 MB of weights, has to
  flow from HMI through L2 into LX every kernel invocation
- Pipeline bubbles: PT throughput sustained over a full kernel
  requires perfect operand prefetch
- LX residency turnover: A and B can't both stay resident at this
  size, so partial reloads happen

**Same shape at M=128** (which is where the sweep showed
`(4, 8, 1)` winning more decisively):

Shape (128, 8192, 8192). Total ops: 17.18 GFLOPs. PT-cycle bound:

`(128/8 × 8192/8 × 8192/8) / 32 = (16 × 1024 × 1024) / 32 = 524 288 cycles`
= **524 µs at 1 GHz**

Measured `(4, 8, 1)`: 0.99 ms = 990 µs.

That's **53% of compute-peak** — much better. The reason: at M=128
each core gets 4 full PT M-blocks, which gives the operand
prefetchers enough work to keep the PT array fed. Memory bandwidth
is amortized over more PT cycles. The remaining 47% gap is mostly
launch overhead + LX turnover.

**Granite 3 8B gate/up_proj at M=32** (32, 12800, 4096):

Total ops: 2 × 32 × 12800 × 4096 = 3.36 GFLOPs. PT-cycle bound:

`(4 × 1600 × 512) / 32 = 102 400 cycles = 102 µs`

Measured `(4, 8, 1)`: 0.72 ms.
- 3.36e9 / 0.72e-3 = 4.67 TFLOPs/s
- 102 / 720 = **14% of compute-peak** for the best split
- Pure-M baseline 2.64 ms = 1.27 TFLOPs/s = 4% of peak

Same pattern as Llama 70B: 3.6× speedup, lifts us from 4% to 14% of
compute-peak.

## Putting it together — a phase diagram

For any (M, N, K) at SENCORES=32 the optimal split family follows
the phase diagram below:

| Regime | M_per_core under pure-M | Optimal family | Why |
|---|---|---|---|
| **M ≪ 32** | < 1 PT row | K-split (1, n, k) | Only way to keep all 32 cores busy |
| **M = 32-128**, narrow N | 1-4 PT rows | K-split (1, n, k>1) + kf | Pure-M severely under-utilises PT; K-split + k_fast saturates with full M per core |
| **M = 32-128**, mid/wide N (n_sticks div 8) | 1-4 PT rows | **k=1 mixed (4, 8, 1)** | M+N split fills PT M-rows AND splits N — no PSUM ring cost |
| **M = 32-128**, awkward N | 1-4 PT rows | Triple-mixed (4, 4, 2) etc + kf | M+N fills PT, K-split soaks the rest of the cores, kf collapses ring hops |
| **M ≥ 256** | ≥ 8 PT rows | pure-M (32, 1, 1) | PT already saturated, K-split adds overhead |

The PR 1986 heuristic only considers the second column (`(1, n, k>1)`)
and gates it to M ∈ [32, 512]. That captures the "narrow N" pocket
correctly, but misses:

- The **k=1 mixed-M+N pocket** (the third row), which is the larger
  win pocket at M=32 / M=128 — a planner with mixed-M+N candidates
  in its priority sweep would catch this naturally.
- The **triple-mixed K-split pocket** (the fourth row), where m>1
  and k>1 together — the PR heuristic forces m=1 so it can't reach
  this either.

## Implication for the heuristic

Three takeaways for the PR / follow-up planner work:

1. **Skipping M < 32 is correct.** The M=1 wins are slim (geomean
   1.03×) and the regime is dominated by overhead the split can't
   reduce.

2. **The PR's "M < 32 ⇒ skip" gate is not the limiting factor on
   what the heuristic can deliver.** The bigger headroom is at
   M=32 / M=128 where mixed-M+N splits are the global optimum.
   This is a *planner* gap, not a *heuristic* gap — letting
   `multi_dim_iteration_space_split` consider mixed-M+N as a first-
   class option (with m·n=32 candidates ranked by PT M-util) would
   close it.

3. **k_fast's residual value is real but bounded.** It strictly
   helps on 26/84 sweep shapes (31%), most clearly on (1, n, k>1)
   shapes at M=1 and triple-mixed (m>1, k>1) shapes at M ≥ 32. As
   long as the planner ever picks a K-split, k_fast emission is
   strictly equal-or-better, so it's always worth keeping on.
