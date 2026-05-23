# On-Chip Restickify: A Staged Path to Core-to-Core LX Data Movement

## Authors

- Adnan Hoque

## Summary

A *restickify* materializes a tensor layout boundary: it is inserted when a
tensor edge crosses from one legal Spyre stick layout to another that a
downstream op requires. Today restickify is always implemented as
`ReStickifyOpHBM` — a read-transform-write round trip through HBM — even when
both the source and the destination of the edge already live on-chip in
per-core LX scratchpad.

This RFC proposes eliminating that round trip by moving the data **core-to-core
over the RIU ring**, keeping the value flow in LX. The narrow goal is a faster
restickify. The broader goal, and the reason this is worth a staged investment,
is to establish a **general core-to-core LX data-movement capability** in the
Spyre compiler. Restickify is the first and most self-contained customer of that
capability; MoE token dispatch/combine, tensor-parallel collectives, and
flash-attention KV movement are later customers of the same fabric.

The work is organized as three tiers, each a standalone milestone with its own
correctness proof:

| Tier | Capability | New deeptools surface | Status / proof |
|---|---|---|---|
| 0 | Producer-aligned work division (no movement; reduce ring byte-hops *inside* the HBM world) | none | Prototyped (≈1.03× @2048) |
| 1 | **Same-stick cross-core transport** (replace HBM restickify when no layout change is needed) | mixed-bundle import + consumer binding | Fabric op proven; needs packaging |
| 2 | **Layout-changing PT/LX bridge** (replace HBM restickify with on-chip transpose) | Tier-1 contracts + remote-fragment-aware coordinate remap | Value-correct on aligned shape (≈1.3× @2048); needs the transform primitive to generalize |

Tier 0 is in scope for the Inductor backend today and requires no deeptools
changes. Tiers 1 and 2 require deeptools contract changes; this RFC specifies
them precisely and shows that the fabric, the packaging, and the upside are
already demonstrated by prototype.

## Motivation

### The waste

Each Spyre core has a private LX scratchpad (≈4.5 TB/s aggregate) and all cores
share HBM (≈166 GB/s). A restickify's source and destination both already exist
as LX-resident tiles on cores — HBM is used only as a rendezvous point. So the
stock path pays two slow HBM trips to relay data that never needed to leave the
chip.

The cores sit on the RIU bidirectional ring (166 GB/s per direction). The
`L3LU`/`L3SU` units that fetch from HBM are *ring-facing*: the same hardware that
pulls a tile from HBM can pull it from a neighbor core's LX. A cross-core LX→LX
move therefore appears in `senprog` traces as `L3LU`/`L3SU` with
`ringDT-ring-lx` / `ringDT-lx-ring` and **no HBM tokens**. The capability to move
data between cores without HBM already exists at the fabric level; what is
missing is the compiler path to express and schedule it.

### Why this is more than restickify

A general core-to-core movement primitive is reusable infrastructure. The same
gather/scatter-over-ring mechanism underlies:

- MoE token dispatch to experts and expert-output combine (often
  layout-preserving — a Tier 1 customer);
- tensor-parallel all-gather / reduce-scatter style joins;
- flash-attention / long-context KV and state movement;
- hybrid Mamba-2 + attention dataflow with mixed split preferences.

Framing the work as a movement *primitive* rather than a restickify *patch*
broadens the customer base and justifies the foundational deeptools contract
changes once rather than per-feature.

### Honest scoping

This RFC does **not** claim a large end-to-end speedup on any current model. A
telemetry survey (see Prior Art) found that most restickify traffic today is
graph-input/weight sourced (a different problem, addressed by prelayout, not by
this RFC), and that the in-graph post-compute restickify share this RFC targets
is currently small. The justification is (a) a measured per-op upside on the
eligible class, (b) forward-looking layout pressure from MoE/hybrid/long-context
models, and (c) the reusable-primitive argument above. Reports must follow the
measurement principles in the Metrics section and keep per-op and end-to-end
numbers separate.

## Proposed Implementation

### Background: the data path

A restickify decomposes into two operations:

1. **Transport** — move the source tile's bytes from the producer's cores to the
   consumer's cores. Because the producer's work-split can be finer than a tile,
   a tile's bytes may be *scattered across several producer cores*.
2. **Transpose** — change the stick dimension locally (e.g. `mb/out → out/mb`).
   This is a per-core PE/PT operation.

Deeptools already provides the same-stick transport ops `STCDPOpLx` and
`InputFetchNeighbor`, and a local transpose op `ReStickifyOpWithPTLx`. The
tiers below differ in *which* of transport/transpose they require and therefore
in *which* deeptools surface they depend on.

### Tier 0 — Producer-aligned work division (in scope today)

**What.** When a restickify has an in-graph producer, steer the restickify's
work-division so it splits along the dimension the producer split, preserving
logical core ownership. This does not move the boundary or change layout; it
reduces the *modeled ring byte-hops* the eventual movement would cost, and in
the aligned case drives them to zero.

**Inductor-side design.** During `work_division`, for an eligible restickify,
reorder the candidate output split dims so the producer-corresponding dim is
considered first by the existing splitter. Eligibility: exactly one in-graph
producer, unambiguous stride-based symbol correspondence, a single dominant
producer split dim, and the preferred dim can absorb the remaining cores. Skip
and log otherwise. No change to restickify count, placement, layout, or values.

**Telemetry.** A default-off ring cost model attributes per-edge byte-hops:

```
byte_hops = Σ over (producer-core P, restickify-core R):
              overlap_elements(P, R) × elem_bytes × ring_distance(P, R)
```

emitted as one JSONL row per restickify, classified by source kind (in-graph /
graph-input / weight / mutation).

**Status.** Prototyped. On `(a + b.t() + c.t()) @ d`: 100% byte-hop reduction at
size 2048 (producer `d1:32`, restickify steered `d0:32 → d1:32`) yielding a
≈1.03× isolated kernel time; 52% at 512; no change at 128 (correctly declined).
Default-off, correctness-preserving (existing restickify tests pass).

**Dependencies.** None outside the Inductor backend. Tier 0 is the only tier
shippable without deeptools changes and should land first.

### Tier 1 — Same-stick cross-core transport

**What.** For an in-graph edge whose producer output and consumer input require
the **same stick layout** but live on the **wrong cores**, replace
`ReStickifyOpHBM` with an in-bundle `STCDPOpLx` / `InputFetchNeighbor` cross-core
LX→LX move. No transpose is involved, so there is no value transform to certify
— the only correctness object is the address map.

**Eligibility gate** (all must hold; else fall back to HBM):

- `source_kind == in_graph_computed`;
- producer output `SpyreTensorLayout` stick order **equals** consumer required
  stick order (this gate is what separates Tier 1 from Tier 2);
- producer core ownership differs from consumer core ownership (else the edge is
  already free);
- producer, edge, and consumer are co-schedulable in one fused bundle.

**Out of scope for Tier 1.** Stick-changing edges (Tier 2); graph-input,
weight, constant, and persistent-state sources (prelayout, separate work);
cross-bundle edges (LX does not survive a launch boundary).

**Inductor-side design.**

1. *Reuse Tier 0 metadata.* Producer ownership from `decode_op_splits`, the
   consumer's committed split from `finalize_layouts`, and the dimension
   correspondence from the restickify symbol map.
2. *Build a transfer plan.* For each consumer-owned tile, list the producer
   core(s) and LX offsets holding its bytes. Because the stick is unchanged this
   is a pure address remap `(src_core, src_lx_off) → (dst_core, dst_lx_off)` with
   no coordinate arithmetic.
3. *Emit one mixed SuperDSC.* Consumer DL op in `dscs_`, transport op(s) in
   `dataOpdscs_`, and `coreIdToDscSchedule` entries running the transport before
   the consumer. The schedule step form is
   `[datadsc_idx, dldsc_idx, after_sync, before_sync]`.
4. *Default-off flag; HBM fallback* whenever any gate fails.

**Value-correctness contract.** Same-stick transport is identity on values, so
the certificate reduces to: (a) the address map is consistent with ownership
metadata (checkable offline), and (b) single-bundle LX lifetime — the producer
output LX is live when the transport reads, and the transport output LX *is* the
consumer's input endpoint.

**Why Tier 1 should generalize across shapes.** The multi-source same-stick
gather was bitwise-verified across all 32 cores at arbitrary ownership in the
prototype (HBM=0). Unlike Tier 2, Tier 1's gather is a *proven* fabric op for
every shape; the only open work is packaging, not a missing op.

**Deeptools requirement — the Foundation contract (shared with Tier 2).**

1. *Mixed DL + data-op bundle import.* DXP bundle import currently rejects any
   SuperDSC carrying `dataOpdscs_` ("Datadsc not allowed, use dldsc"). Accept it
   when `dscs_` and a populated `coreIdToDscSchedule` coexist, and route such
   SuperDSCs through `runDcgForDataOpsDlOps`. A ≈50-line patch of this shape
   already exists in tree (see Prior Art) and was shown sufficient to lower the
   mixed SuperDSC HBM-free under standalone DCC.
2. *A supported producer→consumer LX binding hook.* The current graph API binds
   by graph-port index (`Edge::Pair.index_`), which is not the internal
   `labeledDs_` index the consumer input needs. A first-class hook is required
   so the transport output can be bound to a specific consumer input without
   launch-time artifact splicing.

### Tier 2 — Layout-changing PT/LX bridge

**What.** For an in-graph edge that requires a stick change, replace
`ReStickifyOpHBM` with a three-stage on-chip bridge:

```
STCDPOpLx / InputFetchNeighbor   gather producer LX fragments into a bounded
                                 per-core tile workspace
ReStickifyOpWithPTLx             local PT/SFP tile transform (changes stick)
STCDPOpLx / InputFetchNeighbor   scatter the consumer-owned LX tile
```

**Inductor-side design.** As Tier 1, plus a per-tile gather/transform/scatter
plan and a `bridge_core` per tile. The mixed SuperDSC carries the three data-ops
ordered before the consumer DL op.

**Deeptools requirement — the Transform contract (Tier 2 only).** The three ops
above do not currently *compose* value-correctly:

- `STCDPOpLx` refuses to change the stick during the gather (`stickDimOrder_`
  must match in==out) — acceptable, since the transform happens in the middle;
- `ReStickifyOpWithPTLx` emits its output in a native internal tile descriptor
  (`j_, i_, out_, mb_`) that does not match the consumer's 2D LX input
  descriptor — the missing **consumer-endpoint adapter**;
- no single primitive performs "gather scattered remote fragments **and** change
  the stick **and** present a consumer-shaped LX output" — the missing
  **remote-fragment-aware coordinate remap**.

The ask is therefore: a `remote-fragment-aware-ptlx-coordinate-remap` op, or
equivalently an option for `ReStickifyOpWithPTLx` to emit a consumer-shaped LX
descriptor (the endpoint adapter), such that the three-stage chain certifies
value-correct across non-degenerate shapes.

**Why Tier 2 already works at one shape but not generally.** When producer and
restickify ownership are perfectly aligned (e.g. both `d0:32` at size 2048), the
gather is degenerate — each core already holds exactly the tile it must
transpose — so only the *local* transpose runs and no remote-fragment primitive
is needed. This is precisely the case Tier 0 alignment produces. At
non-aligned shapes (e.g. `d0:8, d1:4` at size 512) a tile's fragments span
multiple cores and the missing primitive is required; forced no-HBM runs at 512
are value-incorrect (≈40–50% mismatched elements). The aligned-shape success is
real evidence the path is sound, not evidence the general case is close.

### Summary of the two deeptools asks

- **Foundation contract** (unlocks Tier 1 and the aligned Tier 2): mixed-bundle
  import + binding hook. *Acceptance test:* the size-2048
  `computed_transpose_adds_then_matmul` mixed bridge runs value-correct through
  **stock** deeptools, with the LD_PRELOAD shim removed.
- **Transform contract** (unlocks general Tier 2): remote-fragment-aware
  coordinate remap / consumer-endpoint adapter. *Acceptance test:* the size-512
  case runs value-correct without forced descriptor overrides.

## Metrics

Per the measurement discipline this project must hold to, reports separate
per-op from end-to-end numbers and never justify the work on a local kernel
speedup alone:

- modeled ring byte-hops, before/after (Tier 0 telemetry);
- HBM bytes eliminated on the eligible edge (from the lowered unit summary;
  `HBM=0` is the success signal);
- affected-kernel speedup (isolated probe, fresh processes, warmup + repeated
  medians);
- percentage of model runtime spent in eligible restickify-heavy regions;
- eligible vs skipped restickifies and the skip-reason distribution;
- correctness vs CPU within existing probe tolerance.

A worked reminder: a 5% kernel speedup in a kernel that is 10% of runtime is a
0.5% end-to-end speedup. End-to-end claims require workload-share evidence, not
probe medians.

## Drawbacks

- **Cross-team dependency.** Tiers 1 and 2 cannot land until deeptools accepts
  the Foundation contract; Tier 2 also needs the Transform contract. The Inductor
  backend cannot complete these alone.
- **Shape dependence.** The benefit appears only when ownership and divisibility
  line up; small/awkward shapes correctly fall back to HBM.
- **Small current share.** On today's models the eligible in-graph restickify
  bucket is a minor fraction of runtime; the strong justification is
  forward-looking and infrastructural.
- **Scheduling/lifetime complexity.** Co-scheduling producer, transport, and
  consumer in one bundle with a shared LX allocation adds compiler invariants
  (LX liveness, sync placement) that must be enforced and tested.
- **Prototype debt.** The existing prototype reaches the working 2048 result via
  an `LD_PRELOAD` shim that interposes mangled deeptools symbols. This RFC
  proposes to *delete* that mechanism; it must not be a shipping dependency.

## Alternatives

- **Status quo (HBM restickify).** Simple and always correct; pays the round
  trip. The baseline this RFC improves on.
- **Graph-input / weight prelayout.** Addresses the *larger* current restickify
  bucket by choosing input/parameter layouts to match the first consumer, or
  prepacking at load. Complementary, not competing — it targets edges with no
  in-graph producer, which Tiers 1–2 explicitly exclude. Recommended as a
  parallel RFC.
- **Hand-authored DDL `unit="dataring"` template.** Rejected: prior work showed
  the SFP ring is psum-only (FMA-fused accumulation), not a pure data ring, and
  hand-authoring raw ring templates is brittle and bypasses the scheduler.
  Building on the existing `STCDPOpLx`/`InputFetchNeighbor` ops is more robust.
- **Cross-bundle LX handoff.** Rejected: LX does not persist across runtime
  launch boundaries; split launches produce a Compute CB hardware error. The
  value flow must stay within one bundle, which is why the Foundation contract is
  about *mixed bundles* rather than separate transport bundles.

## Prior Art

- **`rfc-ring-aware-restickify` (Phase A/B).** Registered `STCDPOpLx` as a
  restickify template and established that the SFP ring is psum-only; originated
  the "emit a cross-core op instead of HBM" direction.
- **`rfc-restickify-first-principles` investigation.** A staged exploration that
  produced: the Tier 0 telemetry + work-division prototype; a bitwise-verified
  HBM-free same-stick cross-core gather across 32 cores; the value-correct
  size-2048 mixed PT/LX bridge (≈1.3×, 1.317 → 1.010 ms); the standalone-DCC
  proof that the mixed SuperDSC lowers HBM-free; the in-tree
  mixed-bundle-import patch sketch; and the precise final blockers
  (`missing-three-stage-remote-fragment-ptlx-lowering`,
  `native-ptlx-output-needs-consumer-endpoint-adapter`).
- **Deeptools `InputFetchNeighbor` / `STCDPOpLx`.** Existing same-stick ring
  transport ops, reused here rather than reinvented.
- **RFC 0047 — Tensors with Device-Specific Layouts.** Defines
  `SpyreTensorLayout`, sticks, and the stride map this RFC's eligibility gates
  read.

## How we teach this

- All controls are internal compiler configuration flags, default-off, until
  validated on device.
- The **tier model** is the teaching frame: Tier 0 changes *where work runs*,
  Tier 1 changes *where bytes are*, Tier 2 changes *where bytes are and their
  layout*. Each tier adds exactly one capability.
- Documentation lands in `docs/source/compiler/work_division_planning.md` (Tier
  0 steering and telemetry) and the tensors-and-layouts guide (restickify and
  the on-chip path), with the success signal stated explicitly: HBM-free lowering
  shows `HBM=0` with `ringDT-*-lx` L3 traffic.

## Unresolved questions

- Which deeptools surface implements the Transform contract: a new
  `remote-fragment-aware-ptlx-coordinate-remap` op, or a consumer-shaped output
  mode on `ReStickifyOpWithPTLx`?
- What is the exact API of the producer→consumer LX binding hook, and does it
  generalize beyond two-op edges?
- Does Tier 1 same-stick transport at 2D splits compose cleanly via
  multi-source `InputFetchNeighbor`, or does it need additional packaging?
- Which non-restickify customers (MoE dispatch/combine, TP collectives) should
  be the first post-Tier-1 adopters, and do they need the Transform contract or
  only the Foundation contract?
- Is ring *distance* sufficient for the cost model, or do high-value cases need
  ring-link *contention* modeling?

## Resolution

### Level of Support

To be determined by review.

### Additional Context

The fabric (HBM-free cross-core movement), the packaging (mixed-bundle HBM-free
lowering), and the upside (≈1.3× on the aligned case) are each demonstrated by
the prior-art prototype. What separates the current state from general support is
two named, testable deeptools contracts.

### Next Steps

- **Tracking issue:** open an issue in `torch-spyre/rfcs` linking this RFC and
  the prior-art branches.
- **Sequence:**
  1. Land **Tier 0** in the Inductor backend (no deeptools dependency).
  2. Land the **Foundation contract** in deeptools; validate by re-running the
     size-2048 mixed bridge through stock deeptools with the `LD_PRELOAD` shim
     removed (acceptance test above).
  3. Implement **Tier 1** end-to-end on the Foundation contract; validate
     value-correctness across sizes where ownership genuinely differs.
  4. Land the **Transform contract**; implement and validate **Tier 2** at the
     size-512 case without forced overrides.
- **Exceptions:** none requested.
