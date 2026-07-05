# Deeptools change log — LSE ring-fold merge (BET 3)

Status: **written rationale only.** No backend code changes here. Everything a
frontend can compose is composed in the frontend (see
`torch_spyre/_inductor/carousel/fold.py` and `reference.py`). This file records
the one backend primitive that does **not** exist on the intra-AIU ring, the
precise ask if the composed cost measures too high, and the frontend-wiring
dependency that gates the cross-core lowering.

## What BET 3 does

Once the flash `Lk` (KV) reduction is split across **cores** (not just streamed
in time as in BET 1), each core owns a KV shard and emits an unnormalized flash
partial `(m, l, A)`. Merging the P partials is an associative reduction under the
log-sum-exp combine — **not** a plain PSUM sum. The default cross-core reduction
path (a K-split lowered to a PSUM-SUM ring, priced by `psum_us`) is numerically
wrong for a softmax reduction, so the merge is composed as a **neighbor
reduce-scatter fold**: one ring move + one SFP `lse_combine` per hop.

The validated schedule + cost is `fold.py` (`ring_fold`, topology
`B_REDUCE_SCATTER`); the numeric merge is `lse_combine` (fp32 `(m, l, A)` with
the dead-lane `-inf` guard). The device-free validator now proves the cross-core
reduce-scatter schedule end to end (`check_fold_reduce_scatter_matches_softmax`,
13/13).

## Frontend vs backend split

**Frontend composition (buildable now — no backend change):**

1. The ring order and the per-hop op **pair**: a cross-core MOVE (a
   restickify/copy with a core-shifted output layout, which deeptools already
   lowers to `STCDPOpLx` on the exact distance-1 adjacency the K-cohort
   core-to-slice map builds in `superdsc.py`), followed by a separate SFP
   `lse_combine` Pointwise sub-run.
2. The numerically exact compound `(m, l, A)` combine with the dead-lane guard.
3. The reduce-scatter + all-gather schedule (`fold._hops_for(B_REDUCE_SCATTER)`):
   P-1 combining neighbor hops of payload/P, then P-1 move-only all-gather hops,
   endpoint **head-split** — the layout the O-projection consumes relayout-free.
4. The cost term that ranks topology B (neighbor reduce-scatter, ~130 GB/s
   uniform-shift band) over topology A (linear fold-to-one, ~36 GB/s contention
   floor), gated **structurally on hop distance (=1 for a neighbor
   reduce-scatter)** — the same broadcast/shared-operand analysis BET 2 uses to
   tell a multicast from a scatter, never a tuned scalar.

**Backend primitive ask (this file — no code here):**

The intra-AIU 32-core ring has `STCDPOpLx` (pure movement) and a PSUM-SUM ring
reduction, but **no fused move-then-reduce primitive**, and reduce/accumulate
lives only in the cross-chip collective layer (AllReduce/AllGather), not on the
32-core ring. So each fold hop is composed as **move + a separate SFP op = two
sequential program steps (a barrier per hop)**.

Precise ask, **only if** the composed per-hop overhead measures too high on
device (this is the trigger — do not pre-emptively build it):

1. **Fuse move + `lse_combine` into one hop primitive** on the intra-AIU ring, so
   a reduce-scatter hop is a single program step rather than a move-barrier-SFP
   pair.
2. **Overlap a hop's move with the previous hop's combine.** This needs the
   `overlapInpFetchWithCompute` gate — today restricted to Conv consumers —
   extended to accept matmul/SFP consumers so an SFP `lse_combine` can run while
   the next shard is in flight.

Neither is required to build BET 3; both are latency optimizations. If they are
never added, the fold still wins the HBM round-trip that the default
score-materializing path pays (BET 1) — the composed-barrier overhead only caps
the *additional* fold speedup, and that measured cap is what would justify the
ask.

`HOP_LATENCY_US` (`fold.py`) is the flagged strawman: the A-vs-B topology
crossover is latency-set and sequence-length-independent, so this constant must
be **measured on device**, not overfit. A wrong value can invert the topology
choice.

## Frontend-wiring dependency (honest status)

Pieces 1-2 of the plan (the cross-core LSE-fold lowering pass, sibling to
`_propagate_tiled_reduction_op`, and the `_matmul_split_cost` fold-cost term
gated on `is_lse_fold`) are **not yet wired into the compiler.** BET 1's
device-grounded finding is that `Lk` does **not** currently route as a tiled
reduction dim on the flash matmul+softmax graph: `coarse_tile._hints_levels`
reads a level's `is_reduction` from the first hinted op in the group (a pointwise
`keys*scale`, where `Lk.is_reduction=False`), so `tiles={"Lk": P}` is classified
as an output-dim level and never populates `loop_tiled_reduction_dims`. Until
that hint-to-level follow-up lands, a cross-core LSE-fold pass keyed on "work
division committed a core split on the `Lk` reduction dim AND `is_lse_fold`"
would never fire, and nothing produces the `is_lse_fold` tag. Wiring an inert
pass now would be dead code built on a path device evidence shows does not
execute.

What ships in this change is therefore the **validated frontend schedule + cost
(`fold.py`) and the closed validator gap (`reference.py` topology B)** — the
numerically exact, device-free proof that the neighbor reduce-scatter fold
equals a single softmax, is all distance-1 hops, and lands head-split. The
compiler wiring is the immediate next step, unblocked by the BET 1 hint-to-level
fix, and the cost term substitutes `ring_fold(B).total_us` for `psum_us` gated
strictly on `is_lse_fold` so non-attention matmuls keep the banked
min-cores/shared-weight pricing unchanged.

## Device probes still owed (per ablations-over-static-analysis)

- Confirm the neighbor MOVE lowers to a **distance-1 `STCDPOpLx`**, not a DRAM
  round-trip (a copy through `memId=-1` loses the 130 GB/s band). Do not assert
  the band from code-read alone.
- Measure `HOP_LATENCY_US` and the composed 2-step-per-hop barrier overhead to
  decide whether the fused-hop / overlap asks above are worth filing.
