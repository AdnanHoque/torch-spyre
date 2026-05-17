# Stage 24: Ring-Aware Continuity And Multicast Plan

## Summary

This stage splits the next ring-aware work into two compiler projects:

1. **Ring-aware core-division continuity**: keep producer and consumer work
   ownership aligned across adjacent in-graph ops when the consumer has legal
   flexibility.
2. **Multicast-aware constant/weight fanout**: detect read-only graph inputs,
   weights, and constants that are repeatedly loaded or restickified for many
   consumers, then evaluate whether Deeptools' GTR multicast path can serve
   those loads more efficiently.

The first project is the direct generalization of Stage 3B. The second project
targets the source class that dominated the fused-block telemetry:
`graph_input_or_weight`.

## Hardware Grounding

The KB records three relevant facts:

- AIU has 32 persistent cores connected by the RIU bidirectional ring.
- The ring carries both off-chip memory to LX transfers and cross-core LX to LX
  transfers.
- The Deeptools schedule tree has a `coreIdToGTRInfo_` field on transfer nodes
  for per-core group tag register multicast routing.

Torch-Spyre currently emits `coreIdToWkSlice_` for work ownership, but a source
scan did not find any torch-spyre emission of `coreIdToGTRInfo_`. That means
core-division continuity can start as an Inductor-side prototype immediately,
while GTR multicast needs a backend contract probe before we change compiler
behavior.

## Project A: Ring-Aware Core-Division Continuity

### Claim

For some producer-consumer edges, the compiler can reduce RIU traffic by giving
the consumer the same physical-core ownership of logical tensor regions that the
producer already used. This preserves tensor layouts and semantics; it only
changes how work slices are mapped to cores.

Stage 3B proved this for one narrow edge:

```text
producer -> restickify -> consumer
```

The continuity project asks whether the same idea helps more generally:

```text
producer -> consumer
producer -> pointwise join
producer -> matmul/reduction
producer -> view/restickify -> consumer
```

### Stage A0: Continuity Telemetry

Add a default-off telemetry pass after `work_distribution` and before
`scratchpad_planning`.

Proposed flags:

```text
SPYRE_CORE_CONTINUITY_TELEMETRY=1
SPYRE_CORE_CONTINUITY_TELEMETRY_JSONL=/path/to/file.jsonl
```

For every in-graph producer-consumer edge:

- classify producer kind and consumer kind
- record source tensor name and layout stride maps
- decode producer and consumer `op_it_space_splits`
- map producer output symbols to consumer input symbols where unambiguous
- estimate logical bytes crossing ownership boundaries
- estimate RIU byte-hops with the same ring-distance model used by Stage 3B
- report whether the consumer appears flexible enough to preserve producer
  ownership

Initial rows should be attribution-only; no behavior change.

### Stage A1: Conservative Continuity Hint

Add a second default-off behavior flag only after telemetry finds nonzero
candidate byte-hops:

```text
SPYRE_ALIGN_CORE_DIVISION_CONTINUITY=1
```

Eligibility should be strict:

- source is `in_graph_computed`
- single dominant producer for the consumer input being preserved
- unambiguous symbol correspondence
- producer and consumer can use the same number of cores
- consumer output/reduction legality is unchanged
- no span-reduction or k-fast constraint is violated

For pointwise consumers, prefer the output dimension order that preserves the
producer split. For reductions/matmuls, first collect telemetry before changing
the N/K split heuristic, because the compute-side cost can dominate locality.

### Stage A2: Certificate

Re-use the Stage 3B locality-certification idea:

- if an override is attached, recompute ownership byte-hops
- allow the override only when modeled byte-hops do not increase
- optionally require exactly zero byte-hops for a debug/assert mode

This makes the prototype understandable: every behavior change has an attached
locality proof.

### First Probe Families

Start with shapes where Stage 3B already showed signal:

- `adds_then_matmul_x` around `2048 x 2048`
- pointwise chains with transposed joins
- matmul then pointwise chains
- fused MLP stress joins
- attention score/value joins
- Mamba/MoE stress joins

The success metric is first byte-hop reduction, then kernel-time reduction.
Runtime claims should remain secondary until profiler traces and memory counters
show the movement is on the critical path.

## Project B: Multicast-Aware Constant/Weight Fanout

### Claim

Many realistic restickifies came from `graph_input_or_weight` sources. Stage 3B
cannot optimize those because there is no in-graph producer ownership to align
with. A different optimization family should look at the first use of read-only
inputs, weights, and constants and ask whether repeated per-core/per-consumer
loads can be replaced by a multicast-aware staging plan.

### Stage B0: Fanout Telemetry

Add default-off attribution first:

```text
SPYRE_INPUT_FANOUT_TELEMETRY=1
SPYRE_INPUT_FANOUT_TELEMETRY_JSONL=/path/to/file.jsonl
```

For every graph input, weight-like input, and constant/external source:

- count consumers
- classify consumers by op kind
- count restickifies sourced from that tensor
- sum bytes moved by restickifies
- record target layouts/stick dims requested by consumers
- record whether consumers use the same layout or incompatible layouts
- record approximate core sets and work slices for each consumer

This answers whether the problem is:

- one source, many consumers, same layout: good multicast/preload candidate
- one source, many incompatible layouts: prepacking/layout-selection candidate
- one source, one consumer: probably not worth GTR work

### Stage B1: GTR Backend Contract Probe

Before emitting compiler behavior, prove the backend accepts and uses GTR fields.

Experiment:

1. Generate or hand-edit a tiny SDSC bundle with a read-only source consumed by
   multiple cores.
2. Add `coreIdToGTRInfo_` to the relevant transfer node.
3. Run `dxp_standalone --bundle`.
4. Inspect generated Dataflow IR / logs for GTR or multicast lowering.
5. Time the same tiny case with and without the GTR field if the backend accepts
   it.

If Deeptools ignores or rejects the field from torch-spyre-generated SDSC, this
project becomes a backend-enablement task rather than a pure Inductor task.

### Stage B2: Compiler Prototype

Only after Stage B1 passes:

- reserve GTR groups for high-value read-only fanout sources
- cap at the hardware/backend-supported group count
- restrict to immutable sources with identical consumer layout needs
- attach multicast metadata through `OpSpec.op_info` into the SDSC generator
- keep behavior default-off

The first compiler prototype should avoid layout-changing weights. It should
target the easiest safe case: read-only source, same layout, many consumers.

## Recommended Order

1. Implement Project A telemetry first. It is an Inductor-local extension of the
   machinery already built for Stage 3B.
2. In parallel or immediately after, implement Project B telemetry. It will tell
   us whether the fused-block `graph_input_or_weight` rows are repeated enough
   to justify a multicast/prepack project.
3. Run the GTR SDSC contract probe before emitting any torch-spyre multicast
   behavior.
4. Only then choose one behavior prototype:
   - continuity override if in-graph byte-hops are common
   - input/weight multicast or prepack if graph-input fanout dominates

## Tests

Project A:

- unit tests for producer-consumer symbol mapping
- unit tests for ownership byte-hop estimation on non-restickify edges
- tests proving flags-off generated SDSC is unchanged
- synthetic probe confirming byte-hop reduction on at least one producer-consumer
  chain

Project B:

- unit tests for source fanout classification
- tests that graph inputs, constants, externs, and mutation targets are separated
- SDSC JSON smoke test for any emitted GTR metadata
- backend compile smoke for the hand-edited GTR bundle before any production
  compiler path is added

## Production Bar

Neither project should be default-on from the start. A production-ready change
needs:

- zero correctness regressions
- flags-off byte-for-byte stability
- clear telemetry evidence on more than one synthetic shape
- profiler or counter evidence that the optimized movement is material to kernel
  time
- an escape hatch env flag until the behavior has model-slice coverage

