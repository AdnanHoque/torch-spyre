# Comms-collectives progress report

**North star:** eliminate all non-weight HBM spills in a Granite block by moving
activations core-to-core over the on-chip ring instead of round-tripping HBM.

This branch collects the analysis scaffolding toward that goal: a spill/comms
taxonomy, an architecture assessment of the relayout mechanism, a gap readout with
implementation designs for the uncovered classes, a live-code verification of what
the backend already supports, and a running deeptools change log. A first-iteration
communication cost model lives on the companion branch `ah/comm-cost-model`.

Production PRs are referred to by function only; no PR or issue numbers/links appear
in any artifact here.

## Contents

- [`01-spill-comms-taxonomy.md`](01-spill-comms-taxonomy.md) — every non-weight spill
  edge in Granite and flash attention, with and without WSR (coarse tiling), placed
  on a 2-axis lattice (movement cardinality × stick-form change).
- [`02-architecture-assessment.md`](02-architecture-assessment.md) — ground-up
  assessment of the frontend/backend split and whether DLDSC is the right interface.
- [`03-gap-readout-and-design.md`](03-gap-readout-and-design.md) — per-class gap +
  buildable implementation design, prioritizing restickify and all-gather.
- [`04-deeptools-capability-verification.md`](04-deeptools-capability-verification.md)
  — which backend capabilities are live-in-pipeline vs present-but-dead vs absent.
- [`deeptools-change-log.md`](deeptools-change-log.md) — every backend change, why it
  is needed, and what breaks without it.

## The two facts that reframe the current state

1. **The measured ~1.19× (≈14.70 → ≈12.34 ms/iter) is a full-LX-residency win, not a
   collective-lowering win.** On the full Granite block the collective classes never
   fire — zero relayout-classification rows are emitted, and the explicit HBM
   restickify rows are unchanged across baseline / full-LX / collectives-on. The
   speedup comes from keeping more fused-chain intermediates LX-resident. Scatter's
   *edge coverage* is high (≈14 of ≈16 classified activation spill edges); its
   isolated *latency* contribution is not yet measured.

2. **The backend already does more than "scatter only."** It auto-inserts on-chip
   `STCDPOpLx` relayout for resident scatter mismatches, and one-source-to-many
   replication (multicast) is a live property of that same lowering. The real gaps are
   narrower and specific: the value-side attention operand (all-gather) hits a backend
   op that is hard-pinned to graph inputs, and layout-changing restickify has a live op
   but no pass that derives the pre/post layout from a coordinate mismatch.

## The architecture verdict (short form)

The frontend-owns-policy / backend-owns-realization split is right, and the DLDSC
coordinate contract is the right spine — with one correction and two additions:

- **Correction:** the frontend should *not* hand-classify each collective. Movement
  cardinality (1:1 scatter, 1:many broadcast, many:many all-gather) is *implied* by the
  producer and consumer distributions, so the backend can derive it — one coordinate-
  driven path already covers scatter and multicast. Per-class labeling in the frontend
  is the part that does not scale; push it down. The frontend's job is detection: is
  there a mismatch worth moving?
- **Two additive fields, not new lanes:** a stick-**form** field (source/dest layout +
  operand identity) so restickify can be planned rather than renamed-and-hoped, and a
  reduce-**op** field for the arithmetic collectives — which stay out of scope to design
  today, because on-chip reduce is PSUM/array compute and cross-core reduce has no
  single-AIU movement primitive.

## Honest status

| class | status |
|---|---|
| scatter (1:1, no form change) | implemented; backend path live; win measured (as LX-residency) |
| broadcast / all-gather operand | classified; multicast primitive live; blocked by a backend op pinned to graph inputs |
| restickify (layout-form change) | op live; no auto-insertion; guarded opfunc-swap prototype, unmeasured |
| gather | falls out of the all-gather fix; deferred |
| reduce / all-reduce | out of the single-AIU LX lane; state, do not design |
| weight restickify, capacity spills | out of scope (offline prelayout / streaming own them) |

## Next steps

1. **Restickify A/B (cheapest):** the backend already self-promotes the HBM restickify
   to its LX form under an all-args-resident precondition. Measure whether the Torch
   opfunc-swap is load-bearing or redundant against that promotion — this single missing
   measurement gates the whole lane.
2. **All-gather (highest value):** generalize the loop-scoped operand-fetch backend op
   off its hard-coded graph-input type and wire it into the compile pipeline; route
   replication through the live multicast path rather than materializing a full
   per-core resident view (which is what fails today).

Both land as frontend changes on the torch-spyre fork plus backend changes on the
deeptools fork; neither needs a new ring primitive.
