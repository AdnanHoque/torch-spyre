# Granite Block LX Relayout Architecture Comparison

Date: 2026-06-30

Branch: `ah/comms-collectives`

Primary question: given the Granite block SDSCs, should the long-term LX
communication substrate be built around the dl-dsc coordinate contract used by
PR #2939, or around the explicit movement plan used by PR #2789?

## Architect's Read

If designing this from scratch, I would use the PR #2939 direction as the
production contract:

1. The frontend chooses work divisions and emits tensor-distribution coordinates.
2. The frontend classifies and costs the mismatch because that cost must feed
   back into work-division selection.
3. The backend synthesizes the physical movement, buffering, chunking, and ring
   schedule from the dl-dsc coordinate mismatch.
4. Later scheduling work overlaps the synthesized movement with compute.

PR #2789 is still valuable. It is a direct explicit-movement prototype: Torch
plans concrete source-to-destination LX transfers and emits them through an
STCDPOpLx/mixed-SDSC carrier. That makes narrow prototypes faster because the
frontend can force exactly the movement it wants. But it also pushes physical
carrier and scheduling decisions up into Torch. That gets awkward as soon as the
communication is not a simple resident scatter.

The short version:

| approach | best property | main long-term risk |
|---|---|---|
| PR #2939 dl-dsc metadata | Clean frontend/backend contract: coordinates and compute demand in dl-dsc; backend realizes movement | Backend must grow enough relayout synthesis to handle more than scatter |
| PR #2789 explicit movement | Fastest way to force a prototype and prove a transfer shape | Frontend starts owning carrier details, schedule rows, chunking, and backend-specific constraints |

So for the full Granite communication goal, PR #2939 is the better center of
gravity. PR #2789 should remain the oracle/prototype lane.

## First Principles Model

Each SDSC row describes one lowered operation. The relevant fields are:

- `labeledDs_`: tensors and whether each allocation is in HBM or LX.
- `scheduleTree_`: compute work division and tensor distribution coordinates.
- `coreIdToWkSlice_`: which logical slice each core computes.
- `coordinates_.coreIdToWkSlice_`: which logical tensor slice already resides
  on each core for an LX input.

The HBM spill problem appears when a producer creates a tensor using one core
division, then a consumer wants that tensor using another division. Without
on-chip relayout, the compiler materializes the edge through HBM, usually as a
`ReStickifyOpHBM` row or as HBM-backed consumer inputs.

The production-shaped contract is:

```text
producer output tensor distribution
  + consumer compute distribution
  -> coordinate mismatch
  -> communication class
  -> backend movement synthesis
```

Coordinates describe what data each core owns and what data each consumer core
needs. The communication class tells the compiler how expensive and schedulable
the mismatch is.

## Granite SDSC Evidence

Source inventory:

```text
docs/results/granite_e2e/comms_collectives_guarded_spill_inventory_20260630.md
docs/results/granite_e2e/comms_collectives_guarded_spill_inventory_20260630.csv
```

The guarded Granite prefill run produced this classification:

| kind | realized | count | meaning |
|---|---:|---:|---|
| `scatter` | yes | 14 | resident LX redistribution/permutation that fits |
| `layout_restickify_activation` | no | 1 | computed activation needs a true LX layout transform |
| `matmul_operand_broadcast` | no | 1 | attention operand needs all-gather/replicate style movement |
| `layout_restickify_weight` | no | 4 | weight prelayout problem, out of runtime scope |

The explicit HBM restickify rows are:

| SDSC | op | scope | communication class |
|---|---|---|---|
| attention QKV projection weight `sdsc_7` | `ReStickifyOpHBM` | out of scope | offline weight prelayout |
| attention activation `sdsc_9` | `ReStickifyOpHBM` | in scope | activation layout restickify |
| attention output projection weight `sdsc_0` | `ReStickifyOpHBM` | out of scope | offline weight prelayout |
| fused FFN gate/up weight `sdsc_10` | `ReStickifyOpHBM` | out of scope | offline weight prelayout |
| FFN down-projection weight `sdsc_0` | `ReStickifyOpHBM` | out of scope | offline weight prelayout |

The runtime communication scope excludes weight restickifies. Those should be
solved by offline/preload weight layout work. The remaining Granite runtime
spills are:

1. A computed attention activation layout restickify.
2. A value-side attention matmul operand broadcast/all-gather.

## What PR1 Covers

PR1 means the current scatter-oriented pair:

- Torch PR #2939: `inductor: add LX planner scatter relayout metadata`
- Deeptools companion: import/use dl-dsc tensor distribution metadata and
  synthesize scatter-like LX relayout through existing backend machinery.

PR1 covers resident scatter/permutation:

```text
producer LX tensor view != consumer compute view
same tensor values
same stick/layout form
destination resident view fits in LX
```

Artifact evidence:

- 14 `scatter` classifications are realized.
- Guarded Granite prefill remains correct and avoids the unsafe all-gather path.
- The remaining runtime gaps are explicitly classified, not silently forced
  through resident scatter.

PR1 does not cover:

- full attention-sized resident replication;
- loop-scoped matmul operand movement;
- true LX layout restickify with different pre/post stick layouts;
- reductions or all-reductions;
- offline weight restickification.

## Covered And Uncovered SDSC Edges

| file/op | current class | PR1 status | why |
|---|---|---|---|
| attention fused SDSCs `1_mul`, `4_mul`, `11_add`, `17_identity` | `scatter` | covered | same values, new resident per-core owner view |
| FFN fused SDSCs `12_silu`, `13_mul`, `2_mul` | `scatter` | covered | pointwise chain can keep intermediates LX resident when views differ by scatter/permutation |
| attention `sdsc_9 ReStickifyOpHBM` | `layout_restickify_activation` | not covered | changes physical stick/layout form, not only owner core |
| attention `sdsc_10 batchmatmul` | `layout_restickify_activation` + `scatter` | partially covered | scatter part is covered; layout transform and dependent operand movement are not |
| attention `sdsc_18 batchmatmul` | `matmul_operand_broadcast` / `all_gather_replicate` | not covered | full resident replication is too large; needs tiled/loop-scoped fetch |
| weight restickify rows | `layout_restickify_weight` | intentionally out of scope | offline weight prelayout/preload owns these |

## Communication Classes

### Scatter / Permutation

Definition: each destination slice comes from exactly one source slice, and
there is no arithmetic reduction. This is a one-to-one or many-core permutation
of ownership.

PR #2939 path:

- Frontend: emit producer tensor distribution coordinates on the consumer LX
  input; classify as `scatter` for observability/costing.
- Backend: compare tensor distribution and compute distribution; synthesize
  STCDPOpLx movement; keep compute SDSC as dl-dsc.
- Status: PR1 class. This is the right production shape.

PR #2789 path:

- Frontend: compute exact source/destination cells and emit explicit movement
  rows.
- Backend: realize the provided transfer list.
- Status: easier to prototype, but Torch owns physical movement details.

### Broadcast / Multicast

Definition: one source slice is consumed by multiple destination cores.
Broadcast usually means all relevant consumers get the piece; multicast is the
same idea for a subset.

PR #2939 path:

- Frontend: emit coordinates where the same logical producer region is required
  by multiple consumer compute regions; classify as `broadcast` or `multicast`
  and estimate resident bytes.
- Backend: choose whether to materialize a small resident view, use multicast
  ring metadata, or lower as loop-scoped movement.
- Gap: backend must not blindly materialize a full replicated operand.

PR #2789 path:

- Frontend: explicitly enumerate fanout transfers.
- Backend: execute the movement rows.
- Gap: frontend now has to choose chunking/fanout shape and avoid backend
  resource limits.

### All-Gather / Replicate

Definition: each consumer needs pieces produced by many or all producer cores.
The attention value-side operand is the first Granite example.

PR #2939 path:

- Frontend: classify as `matmul_operand_broadcast` or `all_gather_replicate`;
  include read index / operand identity and cost estimate.
- Backend: synthesize tiled or loop-scoped input movement around the consumer
  matmul operand fetch. This should use the existing input-neighbor/STCDPOpLx
  machinery, generalized beyond ordinary `INPUT` tensors.
- Current gap: whole-operand resident all-gather is unsafe. Experiments hit
  IBUFF or hardware bus-fence failures. The fix direction is loop-scoped
  movement, not bigger resident buffers.

PR #2789 path:

- Frontend: can enumerate staged all-gather rows directly.
- Backend: lowers those rows.
- Current risk: this can get an isolated compile further faster, but the
  frontend ends up encoding stage size, ordering, and backend legalities.

### Gather

Definition: one destination core needs pieces from multiple producer cores, but
the result is not necessarily replicated to all consumers.

PR #2939 path:

- Frontend: classify fan-in coordinate mismatch and estimate bytes.
- Backend: synthesize multiple source reads into the destination resident view
  or into a loop-scoped operand stream.
- Gap: on-chip gather has not been validated end to end in the current branch.

PR #2789 path:

- Frontend: explicitly emits the fan-in transfer list.
- Backend: carries it out.
- Gap: easy to describe for small cases, but physical fan-in scheduling leaks
  into Torch.

### Layout Restickify Activation

Definition: the tensor is not merely moving between cores; its physical
stick/layout form changes. This is the remaining computed attention activation
`ReStickifyOpHBM`.

PR #2939 path:

- Frontend: intervene before the HBM `ReStickifyOpHBM` is baked in; emit pre
  and post layout coordinates or an explicit `layout_restickify_activation`
  contract.
- Backend: synthesize `ReStickifyOpLx` or equivalent on-chip layout transform.
- Gap: current PR1 scatter metadata is not enough because coordinates describe
  ownership, but the backend also needs pre/post stick-layout semantics.

PR #2789 path:

- Frontend: could explicitly emit a movement/layout-transform carrier if the
  exact cells are computed.
- Backend: realizes explicit rows.
- Gap: unless the carrier is a true layout-transform op, this risks encoding a
  layout algorithm in Torch.

### Reduce

Definition: multiple source pieces contribute arithmetically to one destination
piece.

PR #2939 path:

- Frontend: classify as reduction, not scatter/gather; include reduction op
  semantics and partial/final ownership.
- Backend: synthesize the reduction schedule using compute/reduction support,
  not a pure copy primitive.
- Gap: not implemented in PR1; should be a separate primitive family.

PR #2789 path:

- Frontend: explicit copy rows are insufficient because arithmetic is required.
  It would need either explicit reduce rows or a compute-side reduction carrier.
- Backend: still needs a reduction primitive.
- Gap: PR #2789 does not avoid the need for backend reduction support.

### All-Reduce

Definition: reduce values across producers, then make the final value available
to multiple/all consumers.

PR #2939 path:

- Frontend: classify as reduction plus broadcast/all-gather of the final value;
  cost it as a collective.
- Backend: synthesize reduce-scatter/all-gather or another legal collective
  schedule.
- Gap: future work; not needed for the current proven Granite prefill speedup.

PR #2789 path:

- Frontend: would need to emit a multi-phase movement and reduction schedule.
- Backend: still needs collective reduction support.
- Gap: explicit movement alone does not solve the arithmetic collective.

## Why PR #2939 Is The Better Long-Term Base

The important distinction is not "coordinates versus transfers." It is policy
versus realization.

The frontend must own:

- work-division selection;
- communication-class classification;
- cost estimates that feed back into the planner;
- whether an edge is worth keeping on chip.

The backend should own:

- physical ring movement;
- chunking/staging;
- legal STCDPOpLx/InputFetchNeighbor/ReStickifyOpLx lowering;
- resource constraints such as IBUFF, LX capacity, GTR/GTRIMM legality;
- eventual compute/movement overlap.

PR #2939 matches that split better. It lets Torch describe the logical tensor
and compute coordinates, then lets Deeptools synthesize the movement. PR #2789
is better when we need to force an experiment before backend synthesis exists.

## Practical Recommendation

Use three lanes:

1. Production lane: keep PR #2939 as PR1 and land resident scatter.
2. Backend lane: extend Deeptools from scatter to loop-scoped operand movement
   and LX layout restickify using the dl-dsc contract.
3. Prototype lane: keep PR #2789 available to prove transfer shapes quickly
   when backend synthesis is missing.

For Granite specifically, the next production-shaped work is:

1. Keep weight restickifies out of scope.
2. Implement `layout_restickify_activation` as LX-to-LX layout transform.
3. Implement `matmul_operand_broadcast` as loop-scoped all-gather/input-neighbor
   movement, not full resident replication.
4. Add gather/multicast/reduce/all-reduce only when a real Granite or adjacent
   transformer edge demands them, with each class costed in the frontend and
   synthesized in the backend.

The path through PR #2939 is slower for the next prototype but cleaner for the
final system. The path through PR #2789 is faster for isolated experiments but
less clean once the communication classes grow beyond resident scatter.
