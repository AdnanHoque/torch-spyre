# Joint Layout-Work-Mapping Coordination for Ring-Aware Restickify

**Authors:**

* @AdnanHoque

**Status:** Draft (v2 — significant revision)

## Summary

Restickify is the dominant data-movement primitive in the inductor backend.
Its cost on the RIU ring is not bounded by any single compiler decision —
it is the joint product of **layout** (which dim is stick-oriented),
**work-distribution** (split factors per op), and **core-id mapping**
(physical core assignment). All three are picked independently today.
Empirical measurement of single-lever optimization (work-distribution
alignment) showed either no win (0%) or harmful regressions (−48.7%).
This RFC proposes a joint cost model spanning all three decisions for
restickify-bounded producer-consumer chains, validated incrementally
against an upper-bound telemetry that already exists.

## Motivation

### What changed since v1 of this RFC

The v1 draft proposed "align consumer splits to producer splits" as a
work_distribution hint. I prototyped that on
`AdnanHoque/rfc-ring-aware-restickify` and measured the results:

| Scope of alignment hint | Wall-clock delta vs baseline |
|---|---:|
| All consumer ops | **−48.7%** (regression) |
| Restickify-only | **+0.15%** (no-op) |

The regression came from forcing producer-aligned splits onto matmul
consumers, which have PT-utilization constraints the producer doesn't
share. The no-op came from the fact that restickify's stick-adjusted
extent on the aligned dim is too small (2 sticks on M=128/64) to take
the producer's full split factor — so work_distribution falls back to
the orthogonal axis for remaining cores.

Both results invalidate the v1 framing. **Local alignment at the
work_distribution layer doesn't have enough degrees of freedom to
reduce restickify ring cost.** What does is documented below.

### The real constraint chain

For any restickify between producer op A (output buffer T) and consumer
op B (reading T'), the decisions that determine ring cost are:

```
1. Layout decision  — picks stick dim for T and T'
        ↓ determines
2. Stick adjustment — converts stick-dim extent from elements to sticks
        ↓ determines
3. Split feasibility — which split factors each dim can take
        ↓ determines
4. Work distribution — which dim gets split, by how much
        ↓ determines
5. Core-id mapping  — which physical core gets which slice
        ↓ determines
6. Restickify ring cost
```

The work-distribution layer (step 4) is where v1 tried to intervene.
It is the wrong layer in isolation: it can only choose among the splits
that step 3 makes feasible, and steps 1-3 are themselves driven by
constraints the work-distribution layer can't see (consumer's compute
shape, downstream layout requirements).

**Concrete example from the 4-layer Granite-3.3-8B probe:**

* Producer matmul outputs `k` as `(M=128, H=4096)` with K as stick dim
  → producer's `d0=M` has extent 128 elements (not stick-adjusted)
  → producer can choose split `d0=32`. Default compute-optimal pick.

* Restickify takes `k` and produces `k'` in a layout where `M` is the
  stick dim (because the consumer `q @ k.t()` wants `M` rows contiguous
  in stick layout)
  → restickify's `d0=M` has extent **2 sticks** (M=128 / 64 elems/stick)
  → restickify cannot take `d0=32` even when hinted; max is `d0=2`.

* Consumer matmul `q @ k.t()` would naturally split its `M_q` axis
  → consumer's split is on a different physical dim than producer's.

The mapping between producer and restickify is fixed-orthogonal
**because of layout decision in step 1**. The work-distribution lever
can promote the right dim, but the extent constraint makes the promotion
ineffective.

### Implication

Real restickify ring-cost reduction requires **joint coordination of
layout, work-distribution, and (optionally) core-id mapping decisions
across each producer-restickify-consumer chain**. This RFC proposes
that joint optimization.

## Proposed Implementation

The proposal is a four-phase plan. Phase 0 telemetry is already
prototyped and validated. Phases 1-3 are new and reflect the joint
framing.

### Phase 0 — Telemetry (done)

Read-only diagnostic pass after work_distribution that reports per-
restickify:
* bytes moved
* precise hops/byte (computed pairwise from per-core slice ownership)
* physical alignment status via stride-matching
* whether the cost source is "precise" or "coarse fallback"

Implemented in `restickify_telemetry.py` and `mapping_alignment.py` on
the branch above. Validated on:
* two-matmul chain probe (probe4): ring cost 4.3 MB-hops, precise math
  agrees with coarse estimate to 3%.
* 4-layer Granite probe: 33 MB-hops total across 4 q@k.t() restickifies,
  all mismatched (stride-matching confirms no false alignment).

Telemetry is shippable as a standalone diagnostic regardless of whether
the optimization phases land.

### Phase 1 — Joint cost model & off-line validator (~1 week)

Goal: prove or disprove that joint coordination can beat the current
independent-greedy decisions, before any production integration.

Build an off-line evaluator that takes a single producer-restickify-
consumer chain (extracted from the operations list) and:

1. Enumerates the feasible space of (layout, split-A, split-restickify,
   split-B) tuples.
2. Scores each tuple with the cost model below.
3. Reports the joint optimum and how it compares to the
   independent-greedy choice that today's pipeline would make.

**Cost model**:

```
total_cost(config) = compute_cost(A, config)
                   + restickify_ring_cost(config)
                   + compute_cost(B, config)
```

Where:

* **compute_cost(op, config)** uses the PT-utilization model already
  baked into k_fast's heuristic: for matmul, rows-per-core relative to
  `_PT_ROWS`, and HBM bandwidth bound for the per-core read pattern. For
  reductions and pointwise, simpler models scale with per-core bytes.

* **restickify_ring_cost(config)** = `Σ_{(p,c)} ring_dist(p, c) ×
  overlap_bytes(p, c)`, computed exactly using the per-core slice
  enumeration already in `mapping_alignment.compute_precise_hop_cost`.

* **ring_dist(p, c)** = `min(|p - c|, num_cores - |p - c|)`. Latency
  proxy; bandwidth model deferred to Phase 2 if needed.

**Feasibility constraints** (encoded in the enumeration):

* Layout(T') stick dim ∈ {dims of T'}; restricted to dims compatible
  with consumer's access pattern.
* Split factor on dim D ≤ stick-adjusted extent of D after layout choice.
* Product of split factors ≤ `max_cores`.
* Span-reduction commits respected.

**Search size estimate** (per chain, max_cores=32):
* Layouts to consider: ≤ 4 stick-dim candidates per intermediate tensor.
* Splits per op: each op has ≤ 8 plausible split tuples (pure dim,
  mixed, k_fast-style).
* Per chain: ≤ 4 × 8 × 4 × 8 = ~1000 configurations to evaluate.
* Per configuration: O(num_cores²) = 1024 pair evaluations for ring
  cost.
* Total per chain: ~1M operations. Sub-second in Python.

**Phase 1 deliverable**: a `.md` report on the 4-layer Granite probe
showing, for each of the 4 q@k.t() chains:
* Current cost (today's independent-greedy decisions).
* Joint-optimum cost.
* Delta in milliseconds (using the cost model's wall-clock conversion).

**Kill criterion**: if the joint optimum reduces total cost by < 5%
relative to today's choices, the lever isn't worth pursuing further;
ship Phase 0 telemetry only and close the project.

### Phase 2 — Integration as a constraint provider (~2 weeks, conditional)

Goal: turn Phase 1's joint optimum into a real pipeline override.

The cleanest integration point is a new pass that runs **before**
`propagate_layouts`/`optimize_restickify_locations`/`work_distribution`,
identifying restickify-bounded chains and computing the joint optimum.
The pass then attaches per-op constraints:

* `op._spyre_layout_hint` (read by propagate_layouts)
* `op._spyre_split_hint` (read by work_distribution)

Each hint is advisory: downstream passes consult it but may refuse if
the hint conflicts with hard constraints (span limits, hardware-illegal
layouts, etc.). Hints come with a fallback path so the pipeline never
worsens when the override is refused.

Initial scope: matmul-restickify-matmul chains only (the q@k.t() and
mlp-tail patterns from Granite). Generalize after measurement.

### Phase 3 — Multi-op chains & cost-model refinement (≥3 weeks, deferred)

Generalizes Phase 2 from 3-op chains (A-restickify-B) to longer chains
including pointwise ops, reductions, and multi-fanout. Also refines the
cost model with hardware-measured calibration (ring bandwidth utilization
under contention, PT compute model under multi-cohort layouts).

Explicit non-goals for v2 of this RFC: Phase 3 is sketched only as a
direction. Commit will happen only after Phase 2 measured wins.

## Metrics

Primary:
* **Joint-optimum cost reduction over independent-greedy** on the
  4-layer Granite probe, measured by the Phase 1 evaluator (~5% kill
  threshold; >10% would be a clear win).
* **End-to-end wall-clock** on the same probe with Phase 2 hints
  enabled, vs baseline. Target: ≥ Phase 1's predicted delta minus
  measurement noise (~1%).

Secondary:
* Number of restickifies removed entirely (vs reduced in ring cost).
* Compute-cost-vs-ring-cost ratio across optimized configurations (tells
  us whether the optimization is biased toward ring savings or compute
  preservation).

Diagnostic:
* Phase 0 telemetry's "before/after" delta across a model forward pass.

## Drawbacks

1. **Architectural scope.** This couples three passes (layout, work-
   distribution, mapping) that are independent today. The team
   reshaping `propagate_layouts` (#1941) and `work_distribution`
   (#1989) needs to coordinate; this RFC must not block their work.

2. **Search-space risk.** ~1000 configurations per chain is fine for
   one-shot compilation; in dynamic-shape scenarios it could be too
   slow. Mitigation: cache by shape-signature, skip when AOT compile
   isn't in play.

3. **Cost-model fidelity.** First-order analytic compute model risks
   over- or under-counting reality. Mitigation: Phase 1 reports both
   predicted and (if possible) measured cost; Phase 2 only ships if
   they agree within ~10%.

4. **Negative result is plausible.** If the natural-split compute cost
   dominates restickify ring cost, the joint optimum is the
   independent-greedy choice and the project produces only telemetry.
   This is an acceptable outcome.

## Alternatives

1. **Layout-pass-only optimization.** Solve the problem upstream by
   choosing layouts that don't force restickify in the first place.
   This is what `optimize_restickify_locations` already does (cost =
   element count). Replacing its cost function with ring-cost is a
   simpler change than joint optimization but doesn't benefit from the
   work-distribution lever. Worth considering as a Phase 1.5 if joint
   optimization is too complex but a single-pass change is feasible.

2. **Pure runtime overlap.** If hardware overlaps RIU traffic with PT
   compute well enough that restickify is fully hidden, the entire
   project is moot. Phase 0 telemetry estimates an upper bound but does
   not measure actual on-critical-path time. A profiler-level
   measurement (deferred Phase 0.5 work) would tell us how much of the
   1.8 ms predicted is on the critical path vs hidden.

3. **Doing nothing.** Restickify ring cost is bounded; today's pipeline
   already chooses non-pessimal splits via `optimize_restickify_locations`
   for the element-count metric. The opportunity cost of inaction is
   bounded by Phase 1's predicted delta.

## Prior Art

* **k_fast PR (#1986).** Same general technique (coordinate physical
  core IDs with cost model) applied to PSUM ring reduction in matmul.
  k_fast measured a 1.73× geomean speedup on 20 production shapes by
  picking a slightly-non-natural split combined with adjacent core IDs.
  The cost-model approach in this RFC is conceptually identical;
  difference is that this RFC operates across ops (chain optimization)
  rather than within a single op.

* **TVM AutoTVM and Halide schedules.** Standard joint-search problem
  in tensor-program optimization. Tractable for small chains because
  the search space is small.

* **GPU compilers' fused-kernel selection.** Choosing a fused kernel
  variant trades off compute per-op vs interconnect cost; structurally
  similar tradeoff.

## How we teach this

* **For maintainers:** add `docs/source/compiler/restickify_ring_cost.md`
  explaining the chain-of-constraints picture and pointing at the
  telemetry pass. Update `work_division_planning.md` to mention that
  the join-cost hints exist when Phase 2 lands.

* **For users:** invisible; the optimization is automatic.

## Unresolved questions

1. **Cost-model calibration.** First-order analytic compute models may
   not match hardware closely enough. The Phase 1 evaluator's
   `predicted vs measured` delta will tell us; if delta > 20% on the
   probe, we'll need a hardware-calibrated lookup table.

2. **Multi-fanout.** If tensor T is consumed by multiple downstream ops
   with different preferred splits, joint optimization across all
   consumers becomes constrained. Phase 1 ignores this (picks the
   single largest consumer); Phase 3 must handle it.

3. **Symbolic shapes.** The precise hop math currently requires
   concrete extents. Dynamic-shape support would need either symbolic
   computation or per-shape caching.

4. **Layout-team coordination.** PR #1941 (scratchpad refactor) and
   #1989 (reduction-split) just landed in the area this RFC touches.
   Phase 2 design must align with whatever direction the layout team
   is moving. **Hard prerequisite**: a 30-minute conversation with the
   layout owners before Phase 2 code lands.

5. **Bmm and multi-output-dim cases.** Excluded from k_fast's scope and
   should be excluded from this RFC's initial scope. Generalization is
   future work.

## Resolution

To be filled in after review.

### Level of Support

To be filled in after review.

#### Tracking issue

To be filed.

#### Next Steps

1. Circulate this v2 draft to Olivier and the layout-team owners (PRs
   #1941, #1989). The shift from "work-distribution alignment" to
   "joint layout-work-mapping coordination" needs their early input
   before Phase 1 code lands.
2. Build the Phase 1 off-line evaluator on a single chain.
3. Decide kill/proceed based on the Phase 1 measurement.
4. If proceed, draft formal RFC at https://github.com/torch-spyre/rfcs.
