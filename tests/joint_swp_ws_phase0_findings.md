# Joint SWP + WS scheduling — Phase 0 findings

## Pitch (one-paragraph summary)

Generalize the Twill paper's joint software-pipelining + warp-specialization
ILP formulation (arXiv:2512.18134) from 4 GPU warpgroups to **9 AIU DAE
units**. The claim: today's AIU compiler does software pipelining (SWP)
and unit assignment (WS) as decoupled passes, missing schedules where
reassigning ops to different units would unlock a better pipeline. A
joint ILP formulation could recover patterns like "PT runs current GEMM
tile while SFP normalizes the previous tile while LX prefetches the
next-next tile via L0" — patterns no AIU compiler emits today.

Pitch claims: 12-16 weeks. Top-1 cost-model accuracy 23%. Patent + MLSys
2027 venue.

## Phase 0 question: is the premise actually true?

The pitch's load-bearing claim is that **scheduling and unit assignment
are decoupled in the AIU compiler**. If they're already joint, the
project closes. Phase 0 set out to verify this from the codebase.

## What we found

**Premise verified.** The relevant scheduler is
`RCUIntraEntityScheduler` in
`deeptools/dsm/perfEstimator/IntraEntityScheduler/RCUIntraEntityScheduler.cpp`
(1914 lines). Two structural decouplings:

### 1. Data transfer and compute scheduled separately

The entry point dispatches by entity type:

```cpp
void RCUIntraEntityScheduler::performIntraEntitySchedulingForEntity(
    perfEntity* entity, ..., const SentientSystem& ssys) {
  if (entity->type == EntityType::DATA_TRANSFER) {
    doAboveLxScheduling(entity, ...);   // HMI ↔ LX moves
  } else if (entity->type == EntityType::COMPUTE) {
    doBelowLxScheduling(entity, ...);   // PT, SFP work
  }
}
```

`doAboveLxScheduling` and `doBelowLxScheduling` are independent passes.
Neither sees the other's schedule decisions. This is the **WS half** of
the proposal's claim: data transfer and compute don't co-optimize.

### 2. No cross-iteration pipelining within an entity

Inside `formSETaskSubGraphForCompEntity`, the per-iteration task graph
connects `curQC[iter=i]` → `nextQC[iter=i]` pairwise. I did not find
construction of edges of the form `curQC[iter=i+1]` overlapping with
`curQC[iter=i]`'s downstream stages. This is the **SWP half** of the
proposal's claim: within-loop pipelining across iterations isn't joint
with cross-unit dispatch.

### 3. `functional_overlapped` edge type is defined but appears unused

`perfEstimator.h:26` defines `seTaskEdgeType::functional_overlapped`,
and `interEntityScheduler.cpp:1242` *reads* it during inter-entity
scheduling — but I could not find code that **creates** edges of this
type. Two possible explanations:

- The construction lives in code I didn't grep (worth confirming).
- The type is defined for a future feature that was never wired up.

Either way, the existence of the type alongside missing construction
is consistent with the pitch's premise: the infrastructure for cross-
unit overlap modeling exists in skeleton form, but the actual scheduler
isn't producing those edges.

## What we couldn't determine alone

**How much wall the decoupling costs.** This requires per-unit
utilization stats from a real compile — for example, "PT runs at 17%
utilization while SFP is idle 80% of wall on Llama 70B q_proj M=128."
That number is the upper bound on what joint scheduling could save.

The cost-model accuracy claim ("top-1 23%") in the pitch is suggestive
but indirect. It could mean:
- The scheduler picks suboptimal schedules → joint ILP wins by finding
  better ones.
- The cost model is inaccurate at predicting which schedule the
  scheduler picks → the issue is the model, not the scheduler. In this
  case joint ILP wouldn't help wall time, only prediction.

Phase 0 cannot disambiguate from torch_spyre alone.

## Suggested questions for the deeptools owner

If/when this is handed off:

1. Where is `seTaskEdgeType::functional_overlapped` created? Is the
   cross-entity overlap modeling actively used or vestigial?
2. Can you produce per-unit utilization output for one Llama 70B
   q_proj M=128 compile? (PT, SFP, LX, L0, RIU, Mni, etc. — % busy
   over wall.)
3. Where do you know the current scheduler leaves performance on the
   table? Specific shape regimes? Specific patterns?
4. Has anyone tried joint scheduling, even informally? What blocked it?

## Path forward — proceeding to Path B (OR-tools ILP prototype)

The Phase 0 plan was two parallel tracks:

- **Path A**: deeptools owner conversation. Cheapest, gates everything.
- **Path B**: OR-tools ILP prototype on a single shape. Tests whether
  the math is even tractable at AIU scale.

We're proceeding to Path B without waiting for Path A. The reasoning:

- Premise (decoupling) is verified in the code, so the pitch isn't
  wrong on its central assumption.
- Path B independently answers the second-biggest risk: ILP scaling.
  If the prototype solves a single shape in seconds, scale to a block;
  if it churns for hours, the 12-16 weeks creeps toward 20+.
- Path A still useful but not blocking — its output (utilization
  numbers) calibrates the prototype's win prediction once we have one.

Path B prototype scope:

- One shape: Llama 70B q_proj M=128 (M=128, N=8192, K=8192).
- 3-9 unit machines (start small at 3, expand if tractable).
- 4-10 K-tile iterations.
- Per-iteration stages: HMI fetch B, LX stage, PT compute, SFP post.
- Compare optimal joint schedule vs. greedy decoupled schedule on
  total wall time.

Output: a number for "joint vs decoupled wall-time delta on this
shape" and a runtime measurement for "OR-tools solve time at this
problem size." Both feed the Phase 0 exit gate.

## Files

- This doc — Phase 0 findings
- (next) `joint_swp_ws_ilp_prototype.py` — Path B OR-tools prototype
