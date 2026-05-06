# Core-id placement levers — definitive Phase 0 findings

## TL;DR

Two related ideas — multicast core_id permutation, and inter-op
alignment — both close as torch_spyre levers. Neither delivers
measurable wall-time changes on AIU.

The architectural lesson: **physical core placement matters for the
PSUM ring (where k_fast wins) but NOT for the data ring** (HMI
multicast or inter-op data flow). They use fundamentally different
mechanisms.

## Probe 1: multicast core_id permutation (broader sweep)

Tested 4 representative LLM matmul shapes × 3 M values × 2 splits ×
4 permutations (identity, m-adjacent, reversed, random) — 24 (shape,
M, split) combinations × 4 perms = 96 measurements.

| metric | value |
|---|---|
| structured-perm spread, median | 0.60% |
| structured-perm spread, max | 2.04% |
| rows with >2% spread | 1/24 |
| rows with >5% spread | 0/24 |

The single 2.04% outlier is L3-70B gate at M=2048 (8,4,1) — a 0.4 ms
difference on a 18 ms wall, well within measurement noise.

Across **all tested LLM shapes and M values**, the choice of core_id
permutation moves walls by less than 1% in 23/24 cases. **The
multicast lever does not exist** for HMI data movement on AIU.

## Probe 2: inter-op core_id alignment

Chained matmul1 → matmul2 (op2's input is op1's output), forced op1
to (8, 4, 1), varied op2's split across 5 candidates (matched,
partial-match, mismatched, pure-N, pure-M).

If alignment matters, chained walls should differ — matched should
beat mismatched.

Result:

| case | T_chain across op2 splits | variance |
|---|---|---|
| M=128, kv→o-style | 3.86–3.87 ms | **<0.3%** |
| M=128, o→kv-style | 5.42–5.46 ms | **<1%** |
| M=256, o→kv-style | 5.24–5.29 ms | **<1%** |

**T_chain is essentially constant regardless of op2's split.** Even
with a wildly mismatched split (pure-M when op1 is (8,4,1)),
the chained wall is the same as for the "matched" (8,1,4) split.

Side observation: T_chain < T_solo1 + T_solo2 by 1.1–2.3 ms across
all cases. The compiler bundles the two ops into one launch and
amortizes LF. That's a real saving but not the alignment lever
we were testing.

## Why both close: architectural interpretation

The k_fast PR's mechanism worked because **PSUM accumulation is
sequential**: each k-collaborator passes its partial sum to the next
along the SFP ring. Ring distance directly determines hop count.
Placement matters.

For HMI multicast and inter-op data flow on AIU:

- **HMI multicast**: probably implemented at the HMI port level. One
  fetch broadcasts to all subscribed cores in parallel. Source: chip
  HMI port; sinks: any core. Ring distance doesn't matter because
  the broadcast isn't ring-walking.
- **Inter-op data flow**: data goes through HMI between ops, OR
  through the L0/scratchpad hierarchy. Neither depends on physical
  core placement. The compiler bundles back-to-back ops and the
  intermediate stays on-chip; the placement of the producer cores
  vs consumer cores is invisible at the wall-time level.

**Placement matters for sequential ring traversal; it doesn't matter
for parallel multicast or for non-ring data movement.**

## Meta lesson, sharpened

This was the 4th idea derived from GPU/Twill literature to close
via Phase 0:

| project | closure mechanism |
|---|---|
| Project B (HMI scheduling) | HMI is binding, already saturated |
| Joint SWP+WS at decomposition layer | Per-tile launch floor dominates |
| Multicast core_id permutation | Data ring placement-independent |
| Inter-op core_id alignment | Same — placement-independent |

All four were extrapolated from GPU literature where the analogous
levers exist (GPU SMs have explicit physical layouts; warp scheduling
overlaps; CUDA streams enable cross-op pipelining). On AIU, these
either don't apply at the same layer, or AIU's hardware abstracts
them away.

**Project selection bias**: ideas grounded in already-measured AIU
behavior (k_fast, m-n split, LX residency, SDPA regression) survive
Phase 0; ideas extrapolated from GPU theory mostly don't.

## What's still on the table for solo torch_spyre work

Updated tier-1 list:

1. **LX residency planner** — strongest remaining candidate. Operates
   on a different lever (memory hierarchy) untouched by these
   placement probes. Phase 1 cost model already showed ~22 ms / Llama
   block goes to non-matmul ops with avoidable HMI traffic.
2. **Fix SDPA-to-bmm regression** — quick win, real bug, single file
   scope.
3. **Cost-model-driven planner heuristic** — extends k_fast / m-n
   split work.
4. **Op fusion audit** — investigate what's already fused, find gaps.

Notably ABSENT from this list (closed):
- ❌ Project B / HMI-aware scheduling
- ❌ Joint SWP+WS at Python layer
- ❌ Multicast core_id permutation
- ❌ Inter-op core_id alignment

## Files

- `diag_multicast_core_perm_sweep.py` + `_results.txt` — broader sweep
- `diag_inter_op_alignment.py` + `_results.txt` — alignment probe
- This doc — combined findings + meta lesson
