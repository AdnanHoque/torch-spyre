# Input and Weight Prelayout: Eliminating Restickify at the Graph Boundary

## Authors

- Adnan Hoque

## Summary

A *restickify* materializes a tensor layout boundary: it is inserted when an
edge crosses from one legal Spyre stick layout to the layout a downstream op
requires, and it lowers to a real on-device copy kernel
(`spyre.restickify` → a `Pointwise` copy). A large share of restickify traffic
today originates not from in-graph producers but from **graph inputs and
weights** — edges whose source has no in-graph producer.

The companion *On-Chip Restickify* RFC targets in-graph producer→consumer edges
and depends on new deeptools contracts. This RFC targets the disjoint and
currently larger bucket — input/weight edges — and eliminates those restickifies
**at the source, entirely within the Inductor backend, with no deeptools
dependency**:

- **Weights / parameters** are constant across invocations, yet a weight that
  needs a different stick layout pays a `spyre.restickify` copy on *every*
  forward pass. The layout conversion is itself constant, so it should be done
  **once at compile time** (prepack) and reused.
- **Activation inputs** arrive via a host→device transfer that *already*
  performs a layout conversion. If that transfer targets the first consumer's
  required layout directly, the restickify is **absorbed** into an operation
  that has to happen anyway.

The work is staged in three tiers:

| Tier | Capability | Scope |
|---|---|---|
| P0 | Source-kind classification + prelayout telemetry | Inductor backend |
| P1 | **Weight prepack** — fold input-sourced restickifies into frozen weight constants at compile time | Inductor backend (inference freezing) |
| P2 | **Activation input prelayout** — choose a non-parameter input's layout and realize it in the host→device staging path | Inductor backend + staging path |

## Motivation

### The bucket this targets

A restickify telemetry survey (see Prior Art) found that input/weight-sourced
restickifies dominate the mix — roughly half of sampled restickify traffic was
HBM-load / weight-prep, versus a much smaller in-graph post-compute share. The
on-chip movement RFC, by construction, cannot touch input/weight edges (they
have no in-graph producer to align to or gather from). So the largest current
bucket is left entirely to this RFC.

### Two distinct wastes

**Weights.** Today the backend draws no distinction between a parameter and an
ordinary graph input — both arrive as `InputBuffer`, and there is no freezing,
constant-folding, or prepack mechanism anywhere
(`torch_spyre/_inductor/`). Consequently a weight whose first consumer (commonly
a matmul) needs a different stick layout has a `spyre.restickify` copy spliced in
front of that consumer (`insert_restickify`), and that copy re-runs on **every
invocation**. Because the weight value — and therefore its relaid-out form — is
constant, this is pure recurring waste: a per-call HBM read/transform/write plus
a kernel launch, repeated for the life of the model.

**Activation inputs.** A graph input's device layout is currently taken as-is
from the runtime tensor (`propagate_spyre_tensor_layouts` reads
`real_input.device_tensor_layout()` and pins the input to a single candidate
layout). The host→device staging path (`spyre_to` / `spyre_empty` in
`_monkey_patch.py`) already converts a host-contiguous tensor into a sticked
device layout — a layout conversion that must happen regardless. Today it targets
a default layout and a *separate* restickify then fixes it for the consumer.
Choosing the staging target to match the consumer's required layout removes the
second conversion at zero marginal cost.

### Why this is the pragmatic near-term win

Unlike the on-chip RFC, this work needs **no cross-team deeptools contract**. It
is frontend/compiler work the Inductor backend owns end to end, and it attacks
the larger bucket. It should be sequenced as the higher-ROI near-term effort, with
the on-chip RFC pursued in parallel for the in-graph bucket.

### Honest scoping

This RFC does not claim a fixed end-to-end speedup. The benefit scales with how
much of a model's restickify traffic is input/weight-sourced and how often those
copies recur. Reports must follow the Metrics section and separate one-time
compile cost from per-call savings.

## Proposed Implementation

### Background: the current layout pipeline

The relevant pass order (`passes.py`, `CustomPreSchedulingPasses`) is:

```
propagate_spyre_tensor_layouts   # assign candidate device layouts (STLs)
optimize_restickify_locations    # commit one STL per op (min total restick cost)
finalize_layouts                 # wrap to FixedTiledLayout; build restickify_plan
insert_restickify                # splice spyre.restickify copies before consumers
... work_distribution ; scratchpad_planning
```

Key facts the design relies on:

- Inputs (including weights) are pinned to a **single** candidate
  `SpyreTensorLayout`, read from the runtime tensor in
  `propagate_spyre_tensor_layouts`; both optimizers seed inputs as fixed.
- The first consumer's required input layout is already available through
  `cost_fn.required_input_stls` (`optimize_restickify.py`) and is consumed in
  `finalize_layouts` to build `V.graph.restickify_plan`.
- There is **no source-kind classifier**: graph inputs are detected only
  structurally (`TensorBox(StorageBox(InputBuffer))`), and parameters are
  indistinguishable from activations (both are leading positional `InputBuffer`s
  under AOTAutograd inlining).
- There is **no freezing / constant-folding / prepack** and no backend
  `ConstantBuffer` handling. A restickify always materializes as a recurring
  copy kernel.
- A `TENSOR_MATCH` guard recompiles if a graph input's `SpyreTensorLayout`
  changes (`_monkey_patch.py`), so any compiler-chosen input layout is
  recompile-safe.

### Tier P0 — Source-kind classification and prelayout telemetry

**What.** Introduce an explicit source-kind classifier for each restickify edge —
`{in_graph_computed, parameter, activation_input, constant, mutation}` — and
default-off telemetry that records, per input/weight restickify: source kind,
bytes moved, per-call copy cost, dtype/shape, the input and target layouts, and
whether the source recurs across invocations.

**Why first.** The backend currently cannot tell a weight from an activation, so
it cannot target either remedy. P0 makes the bucket measurable and assigns each
edge an actionable class. Parameter identification uses AOTAutograd's positional
convention (leading-N inputs are parameters) initially, with the freezing-based
`ConstantBuffer` signal taking over in P1.

**Success criterion.** Every input/weight restickify carries a class label and a
per-call cost, so P1/P2 candidates can be ranked by recurring savings.

### Tier P1 — Weight prepack

**What.** Eliminate input-sourced restickifies on parameters by performing the
layout conversion once at compile time.

**Primary path — inference freezing.** Enable PyTorch Inductor freezing
(`config.freezing`) so parameters fold to constants. This requires the backend
to gain the `ConstantBuffer` support it currently lacks: assign a
`SpyreTensorLayout` to a frozen constant and stage it to device. Then add a
**constant-fold-restickify pass** (after `finalize_layouts`, before
`insert_restickify`): for any restickify whose source is a frozen weight
constant, compute the relaid-out constant at compile time, stage that single
relaid-out tensor to device, and rewrite the consumer to read it — so no
`spyre.restickify` copy is emitted for that edge. This is the standard weight
prepack pattern (cf. oneDNN/mkldnn weight prepack under Inductor freezing).

**Alternative path — runtime prepack cache (no freezing).** If freezing is
undesirable for a workload, identify parameter inputs positionally and maintain a
runtime side-table that converts each parameter's layout once on first use and
reuses it on subsequent calls, keyed on input identity and guarded by the
existing `TENSOR_MATCH` mechanism. This avoids the inference-only constraint at
the cost of guard/lifetime complexity.

**Gate.** Default-off config flag; inference-only for the freezing path; HBM
restickify remains the fallback when a weight is not eligible.

**Result.** For every eligible weight, the per-call restickify copy and its HBM
traffic are removed entirely, replaced by a one-time compile/stage cost.

### Tier P2 — Activation input prelayout

**What.** For a genuine (non-parameter) graph input whose first consumer requires
a different layout, let the compiler choose the input's layout and realize it in
the host→device transfer rather than via a separate restickify.

**Design.**

1. In `propagate_spyre_tensor_layouts`, where an activation input is currently
   pinned to its single eager layout, allow it to take the first consumer's
   required layout (already available via `required_input_stls`) as its target
   when that is legal.
2. Propagate the chosen input layout to the staging path (`spyre_to`) so the
   host→device transfer materializes the consumer's layout directly, absorbing
   the restickify into the conversion that already occurs.
3. Rely on the `TENSOR_MATCH` guard to recompile if a caller supplies a
   different layout, preserving correctness.

**Gate.** Default-off; falls back to the stock per-call restickify when the
staging path cannot realize the chosen layout.

**Result.** The restickify is eliminated by folding it into the unavoidable
host→device layout conversion.

## Metrics

- input/weight restickifies eliminated (count, and as a fraction of all
  restickifies);
- recurring device copies removed = eliminated edges × invocations;
- HBM bytes and kernel launches saved per forward pass;
- one-time compile/stage cost for prepacked weights (reported separately from
  per-call savings);
- percentage of model runtime spent in eliminated copies;
- recompile rate under the `TENSOR_MATCH` guard for P2 (a regression signal);
- correctness vs CPU within existing probe tolerance.

A worked reminder: a per-call copy that is 1% of a forward pass, eliminated for a
model run 10,000 times, is a 1% steady-state win that compounds over the
deployment — the value is in the recurrence, not a single-call delta.

## Drawbacks

- **Freezing is inference-only.** The P1 primary path precludes training /
  parameter mutation and recompiles when weights change; baked constants enlarge
  artifacts. Acceptable for a Spyre inference target, but a real constraint.
- **No `ConstantBuffer` support exists today.** P1 requires adding frozen-constant
  handling to the backend (layout assignment + staging), which is genuine
  integration work, not a flag flip.
- **P2 couples compiler and staging.** The chosen input layout must be realizable
  by `spyre_to`; mis-coordination shows up as extra recompiles under the guard.
- **Disjoint from in-graph edges.** This RFC does nothing for in-graph
  producer→consumer restickifies; those are the on-chip RFC's domain.

## Alternatives

- **On-Chip Restickify RFC (companion).** Targets the in-graph bucket via
  core-to-core LX movement; needs deeptools contracts. Complementary — the two
  RFCs partition the restickify problem by source kind.
- **Runtime prepack cache instead of freezing** (P1 alternative above) — broader
  applicability, more lifetime/guard complexity.
- **Status quo.** Keep per-call input/weight restickify copies. Simple, always
  correct, pays the recurring cost.

## Prior Art

- **oneDNN / mkldnn weight prepack under Inductor freezing (CPU).** The canonical
  pattern P1 mirrors: fold a layout/format conversion of a frozen weight into a
  compile-time constant.
- **Restickify telemetry survey (`rfc-restickify-first-principles`).** Showed the
  input/weight-sourced bucket dominates the restickify mix and motivated
  classifying restickifies by source kind.
- **RFC 0047 — Tensors with Device-Specific Layouts.** Defines
  `SpyreTensorLayout`, sticks, and the stride map the eligibility checks read.
- **RFC 0171 — Spyre Device Construct in PyTorch.** Covers device registration
  and the host→device path P2 extends.
- **On-Chip Restickify RFC.** The companion proposal for the in-graph bucket.

## How we teach this

- All controls are internal, default-off config flags until validated on device.
- The teaching frame is the **source-kind taxonomy**: a restickify's remedy is
  determined by where its source comes from — *weight* → prepack once
  (P1); *activation input* → choose the staging layout (P2); *in-graph producer*
  → move it on-chip (companion RFC). This gives a single map from "where did this
  boundary come from?" to "which mechanism removes it?".
- Documentation lands in the tensors-and-layouts user guide (the prelayout
  contract for inputs/weights) and the compiler front-end docs (the
  classification pass and the prepack fold).

## Unresolved questions

- Does enabling `config.freezing` compose cleanly with the backend's custom
  `compile_fx` wrapper and the existing custom pre-scheduling passes?
- Is AOTAutograd's positional "leading-N inputs are parameters" convention
  reliable enough for P0/P1 parameter identification across configurations, or is
  freezing's `ConstantBuffer` signal required from the start?
- Can the staging path (`spyre_to`) realize an arbitrary chosen device layout for
  P2, or only a subset of target stick layouts?
- How should persistent cross-graph state (e.g. KV cache inputs) be handled — do
  they need a layout contract that spans graph boundaries?
- How does prepack interact with the Dynamo guard / compiled-artifact cache, and
  what is the recompile behavior when a weight tensor is replaced?

## Resolution

### Level of Support

To be determined by review.

### Additional Context

The mechanisms this RFC needs — layout candidates for inputs, the consumer's
required layout, and the host→device conversion — already exist in the pipeline;
what is missing is source-kind classification, frozen-constant handling, and the
prepack fold. None of it requires deeptools.

### Next Steps

- **Tracking issue:** open an issue in `torch-spyre/rfcs` linking this RFC and
  the companion On-Chip Restickify RFC.
- **Sequence:**
  1. **P0** — source-kind classification + prelayout telemetry; rank
     input/weight restickifies by recurring cost.
  2. **P1** — weight prepack (freezing + `ConstantBuffer` support +
     constant-fold-restickify pass); validate elimination of per-call weight
     copies vs CPU.
  3. **P2** — activation input prelayout via the staging path; validate guard
     behavior and absorbed-conversion correctness.
- **Exceptions:** none requested.
