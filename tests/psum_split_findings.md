# Project A — K-split / PSUM split selection findings

A characterization of when K-split (using the dedicated SFP ring for
partial-sum reduction) is faster than the `output_element_priority`
default of pure-N split, and what that means for a planner heuristic.

## TL;DR

The K-split lever is real but its impact is much narrower than the
"PSUM uses a free SFP ring" hypothesis predicted. **Pure K-split is
worse than pure-N at every K/N ratio tested.** A mixed `(m=2, n=1, k)`
split wins by ~13% on exactly one production shape pattern (L3-8B
MLP down / Mixtral down), and that win is gated by total per-core
compute exceeding the launch floor — not by the K/N ratio per se.

**Decision: don't ship a heuristic.** Targeting one shape pattern
isn't worth the planner-code complexity. The mechanism is documented
here so the work can be picked up if the workload mix changes.

## The hypothesis we started with

The IBM AIU architecture doc (slide 30) says cross-core partial-sum
reduction uses a *dedicated* 32 B SFP ring, separate from the 128 B
data rings. Today's planner avoids K-split because it accounts ring
traffic naively (treats all rings as one). For shapes where data ring
is congested but SFP ring is idle, K-split should be cheaper than the
cost model predicts.

## Probe 1 — Project A focused split sweep

6 production / synthetic shapes, 4-5 forced splits each. Found:

- **L3-8B MLP down (128, 4096, 14336)**: `(2, 1, 16)` at 4.11 ms vs
  `(1, 32, 1)` at 4.60 ms — **1.12× speedup**. Confirmed Phase 1.0
  hint that K-split wins here.

- **Pure K-split is BAD across the board** — at every shape tested,
  `(1, 1, 32)` regresses vs the element_priority pick. Even on
  L3-8B MLP down (where K-split wins), pure K is worse than mixed.

- **The winner on MLP-down is not pure-K, it's `(2, 1, 16)`**: 16-core
  PSUM chain at 0.5 MB partials per core. Pure K would be a 32-core
  chain at 1 MB partials — both factors worse.

So the mechanism isn't "K-split is free on the SFP ring" — it's
**"PSUM cost scales as (chain length) × (per-core partial size)";
mixed (m, 1, k) splits halve both factors compared to pure-K**.

## Probe 2 — K/N ratio sweep (5 shapes, M=128 fixed)

Designed to find the K/N threshold where K-split starts winning, by
sweeping K/N from 1.0 to 16.0:

| K/N | shape | best split | speedup vs pure-N | per-core compute |
|---|---|---|---:|---|
| 1.0 | (128, 4096, 4096) | `(8, 4, 1)` | 1.01× (noise) | 67M (LFB) |
| 2.0 | (128, 4096, 8192) | `(8, 4, 1)` | 1.01× (noise) | 134M (LFB) |
| 3.5 | (128, 4096, 14336) | `(2, 1, 16)` | **1.13×** | 234M (escapes!) |
| 8.0 | (128, 2048, 16384) | `(1, 32, 1)` | 1.00× (noise) | 134M (LFB) |
| 16.0 | (128, 1024, 16384) | `(8, 4, 1)` | 1.00× (noise) | 67M (LFB) |

**The K/N=3.5 win isn't because of the ratio.** It's because that's
the only shape in the test with per-core compute large enough
(2.3 ms at peak) to escape the ~3 ms launch floor. K/N=8.0 has more
K but less total compute, so it stays at the floor regardless of
split choice.

## The actual fire condition

The K-split lever fires when ALL of:

1. K is large enough that K-split provides meaningful parallelism
   (e.g., `K ≥ 8 × num_cores · stick_size` at fp16 = `K ≥ ~16K`)
2. **Total compute is large enough to escape the launch floor**
   (`M · N · K ≥ 9 GFLOPs total ≈ 270 MFLOPs/core`)
3. K/N ratio is moderate (~3-4) — small enough that pure-N still
   has reasonable per-core compute
4. No span-pre-split on output dims

In the production catalog, the only shapes meeting all four:

- **L3-8B MLP down** (128, 4096, 14336): K=14K, M·N·K = 7.5 GFLOPs ≈ launch floor border, K/N=3.5
- **Mixtral down per-expert** (128, 4096, 14336): identical shape

Both give ~13% speedup. Same shape, two model contexts.

## Why we're not shipping a heuristic

**Two production shapes (the same shape pattern) is too narrow to
justify a planner-level heuristic.** The complexity-per-shape ratio
isn't favorable:

- A K-split heuristic needs careful gating to avoid regressions on
  the 11 other Phase 1.0 shapes
- The compound interactions with `output_element_priority` would need
  validation
- The existing `k_split_heuristic` (splitk-matmul branch, picks pure-K)
  is now wrong post-EP — would have to be rewritten anyway

For the 13% win we'd capture on 2 shapes, the planner-code complexity
isn't worth it. Better to leave this documented and revisit if:

- The workload mix shifts toward more K-heavy matmul
- We find a cleaner algorithmic framing that captures more shapes
- Higher-priority projects don't materialize

## Open question: what happened to the "PSUM-is-free" prediction?

The architecture doc clearly says SFP ring is dedicated. So why
doesn't pure-K consistently win on K-heavy shapes?

Two plausible explanations, neither tested:

1. **SFP ring bandwidth is real-but-saturable.** 32 B/cycle vs 128 B/
   cycle data rings. For 1 MB partials × 31 hops = 31 MB through a
   32 B-wide ring. At 1 GHz that's ~1 ms just for the ring transit —
   comparable to the wall-time delta we're measuring. So PSUM cost is
   real, just on a different ring than data movement.

2. **The PT array has to wait for the PSUM result.** PSUM happens
   AFTER compute (the architecture doc shows the dataflow as PT → PE
   → SFP). So PSUM time is on the critical path of the kernel. SFP
   ring being separate from data ring doesn't change that the
   reduction has to complete before output writes.

Either way, the "free PSUM" prediction was too optimistic.

## What this rules out for future work

This investigation rules out a broad K-split heuristic as a viable
project. It does NOT rule out:

- A **specific** L3-8B MLP down optimization (if it's in the critical
  path of a deployment)
- An **aux-op-aware** scheduling layer that overlaps PSUM with the
  next op's compute
- A fundamentally different K-axis approach (e.g., split-K with
  in-LX accumulation rather than ring PSUM)

But none of those are clean Phase-2-style projects.
