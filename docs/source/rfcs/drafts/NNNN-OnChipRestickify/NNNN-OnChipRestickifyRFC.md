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
| 1 | **Same-layout cross-core handoff** (general on-chip LX remap; keeps re-partitioned activations on-chip — *not* a restickify replacement) | multi-op SuperDSC import + consumer binding | Fabric op proven; needs packaging |
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

### Tier 1 — Same-layout cross-core handoff (general on-chip LX remap)

**What — and what it is *not*.** Tier 1 is **not** a restickify replacement. A
same-layout re-partition does not produce a `ReStickifyOpHBM` at all:
`compute_restickify_needed` only fires on stick *incompatibility*. When a
producer and consumer share a stick layout but the consumer re-partitions the
data across cores differently, the activation simply round-trips HBM at the
**bundle/SDSC boundary** (how separate SDSCs hand off). Tier 1 is therefore a
**general internal LX-remap / on-chip handoff**: keep that same-layout activation
resident in LX and move the re-partitioned bytes core-to-core over the ring,
instead of spilling the whole activation to HBM and reloading it.

**Customers.** This is the substrate for the on-chip collective patterns, not
restickify: MoE token dispatch/combine (all-to-all), tensor-parallel all-gather
between sharded GEMMs, and residual / elementwise joins across divergent core
splits. The reduce half already exists (`CrossCoreReduceOpLx`); Tier 1 adds the
gather / scatter / all-to-all half. (Restickify itself is a Tier 2 customer,
since it changes the stick.)

**Eligibility** (all must hold; else leave the stock HBM handoff in place):

- the edge is an in-graph producer→consumer activation handoff;
- producer output and consumer input share the same stick layout (no transpose —
  this separates Tier 1 from Tier 2);
- producer and consumer core ownership differ (else the handoff is already
  local);
- producer and consumer can be co-scheduled into one SuperDSC (see Realization).

**Out of scope.** Stick-changing edges (Tier 2); graph-input / weight /
constant / persistent-state sources (prelayout); cross-bundle edges where LX
cannot persist.

**Architecture — a planner, in two stages.** The on-chip realization is *not*
self-contained in one inductor pass, because the on-chip unit is the **SDSC**,
not the bundle: `generate_bundle` emits one SDSC per op, and LX does not persist
across `sdsc_execute`, so even same-*bundle* op-to-op handoff goes through HBM.

1. *Detection / planning (inductor, after `work_distribution`).* Core ownership
   is known after work division. A new pass — best framed as an **extension of
   LX-residency / scratchpad planning** rather than a standalone op — detects
   same-layout edges whose producer/consumer ownership differs and that would
   otherwise spill to HBM, and builds the transfer plan: for each consumer-owned
   tile, the producer core(s) and LX offsets holding its bytes. Because the stick
   is unchanged this is a pure address remap
   `(src_core, src_lx_off) → (dst_core, dst_lx_off)`, no coordinate arithmetic.
   Reuse the ownership / symbol-map / byte-hop primitives from
   `restickify_ring.py` (Tier 0); do **not** hang this off `insert_restickify.py`
   — these edges are not restickifies. Note: such edges are **invisible to the
   existing restickify telemetry**, so detecting them is itself net-new and is
   the planner's first job.
2. *Realization (SDSC codegen, gated on the Foundation contract).* Keeping the
   intermediate LX-resident across the edge requires co-scheduling producer and
   consumer into one SuperDSC with the cross-core transfer as a scheduled step
   (`coreIdToDscSchedule`). Stock deeptools will not do this from bundle import
   (`runDcgForDlOpsStandalone` is one-op-per-SDSC), so this stage depends on the
   Foundation contract below. **The planner must fail closed:** when the contract
   is absent or any gate fails, leave the stock HBM handoff in place.

**Value-correctness contract.** Same-layout transport is identity on values, so
the certificate reduces to (a) the address map is consistent with ownership
metadata (checkable offline) and (b) single-SuperDSC LX lifetime — the producer
output LX is live when the transport reads, and the transport output LX *is* the
consumer's input endpoint.

**Why Tier 1 should generalize across shapes.** The multi-source same-stick
gather was bitwise-verified across all 32 cores at arbitrary ownership in the
prototype (HBM=0). Unlike Tier 2, Tier 1's gather is a *proven* fabric op for
every shape; the open work is packaging, not a missing op.

**Deeptools requirement — the Foundation contract (shared with Tier 2).**

1. *Multi-op SuperDSC codegen reachable from bundle import.* The codegen that
   keeps an intermediate in LX across multiple scheduled ops
   (`runDcgForDataOpsDlOps`) already **exists** in deeptools but is not reachable
   from bundle import — stock `dxp.cpp` only dispatches to `runDcg`
   (data-op-only) or `runDcgForDlOpsStandalone` (one DL op per SDSC), and import
   rejects any SuperDSC carrying `dataOpdscs_` ("Datadsc not allowed, use
   dldsc"). The change **wires an existing function** (≈50-line patch in tree,
   see Prior Art: relax the import gate when `dscs_` + a populated
   `coreIdToDscSchedule` coexist, and route to `runDcgForDataOpsDlOps`); it is
   *not* a new fabric primitive. It was shown sufficient to lower the multi-op
   SuperDSC HBM-free under standalone DCC.
2. *A supported producer→consumer LX binding hook.* The current graph API binds
   by graph-port index (`Edge::Pair.index_`), not the internal `labeledDs_`
   index the consumer input needs. A first-class hook lets the transport output
   bind to a specific consumer input without launch-time artifact splicing.

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

## Device Validation Findings (2026-05)

The prior-art prototype results above were obtained on an isolated/standalone
stack. This section records what has since been **validated on the Spyre device**
from a clean, isolated-environment implementation, separating what device runs
have proven from what remains to be confirmed. Numbers stated here are device
results; the design prose above is unchanged.

### Tier 0 — device-validated

The Tier 0 capability (ring telemetry plus the Stage 3B producer-aligned
work-division steering) is implemented and device-validated, benchmarked across
sizes from small to large. The byte-hop reductions and the isolated kernel-time
behavior described in the Tier 0 status are reproduced on hardware;
correctness is preserved and the path remains default-off.

### Tier 1 — planner device-validated end-to-end

The Tier 1 same-layout cross-core handoff planner (`run_onchip_handoff_planner`)
is device-validated end-to-end on the compiled path. On the compiled flow it
detects eligible same-layout divergent-split edges and emits a valid handoff plan
while remaining fail-closed where the Foundation contract is absent.

### Mixed-SuperDSC on-chip realization runs on device

A **mixed SuperDSC** — a consumer DL op in `dscs_` together with data-ops in
`dataOpdscs_` / `datadscs_` and a `coreIdToDscSchedule` — carrying a single
`STCDPOpLx` data-op **runs on device**: the producer→consumer activation handoff
stays resident in LX, the round trip through HBM is eliminated (`hbmSize_=0`, with
L3 ring tokens present in the senprog), and the whole-graph result is
value-correct. This was realized on the size-2048 fused-add-mm case via a
synthesized mixed bundle (the `onchip_bridge.py` synthesizer, gold-verified
byte-for-byte against a known-good reference) spliced onto a real compiled bundle
and compiled with a minimally-patched `dxp_standalone`. This is the on-device
confirmation of the Foundation-contract packaging that the prior-art section
demonstrated only under standalone DCC.

**Caveat — this first STCDP test was a degenerate same-core configuration.**
Source and destination were both `out:32`-split with distinct LX bases, so each
core copied its own slice LX→LX **locally**. The run therefore proves that the
mixed data-op **control path** executes on device and is value-correct, and that
the HBM round trip is eliminated; it does **not** by itself prove cross-core ring
movement.

### Compute-CB fault isolated to the transpose

The Tier-2 layout-changing bridge — a `ReStickifyOpWithPTLx` local
stick-transpose followed by `STCDPOpLx` — faults on device with
`RAS::RUNTIMESCHEDULER::ComputeHardwareError` ("Compute CB hardware error
detected", code `0x7b1b`). Because the pure-`STCDPOpLx` mixed bundle runs clean
while the transpose-bearing bundle faults, the fault is **isolated to the
`ReStickifyOpWithPTLx` (PT/compute) transpose op**, not to the `STCDPOpLx`
data-move or to the mixed-dispatch machinery. This sharpens the Transform
contract: the missing work is on the compute/transpose primitive, while the
data-move and mixed-dispatch paths are demonstrated sound on device.

### The two precise deeptools asks

The device runs confirm that the only deeptools changes required are these two;
runtime dispatch and the `dcc` `runDcgForDataOpsDlOps` codegen are already stock.

1. **Relax the mixed-bundle import gate.** The `Dxp::importSdsc` gate currently
   rejects mixed bundles ("Datadsc not allowed, use dldsc"). It must admit a
   SuperDSC with non-empty `dataOpdscs_`, non-empty `dscs_`, and non-empty
   `coreIdToDscSchedule` as a mixed DL+data-op SuperDSC.
2. **Dispatch the existing multi-op codegen.** Dispatch `runDcgForDataOpsDlOps`
   when `coreIdToDscSchedule` covers all used cores.

### Validation methodology note

For reproducibility: the runtime's `g_artifact_cache` is keyed on `code_dir` and
is per-process, so swapping a senprog on disk under a `code_dir` the process has
already loaded is shadowed by the cache. Device validation must therefore
redirect the kernel runner to a **fresh `code_dir` path the process has never
seen**, and must include a **negative control** (remove the spliced senprog → the
run must FAIL) to prove the device is actually executing the spliced program and
not a cached or baseline one.

### Genuine cross-core ring STCDP — confirmed on device

The degenerate same-core STCDP above proves the mixed control path runs, but does
no ring traffic. To prove genuine **cross-core** movement we built a 2-STCDP round
trip with a reversed-ownership intermediate:

```text
producer add output  (linear   @LX 16384, slice i on core i)
  --STCDP1-->  scratch (REVERSED @LX 1048576, slice i on core 31-i)
  --STCDP2-->  consumer add input (linear @LX 8192, slice i on core i)
```

STCDP1 moves slice `i` from core `i` to core `31-i`; STCDP2 moves it back. All 32
slices cross cores in both moves, yet the round trip lands data in the consumer's
native (linear) layout, so the whole-graph result stays value-correct **without
any consumer-reshard surgery**. No transpose / PT compute op is involved, which
isolates the ring data path from the Compute-CB-faulting `ReStickifyOpWithPTLx`.

Three independent layers of evidence confirm genuine cross-core ring movement:

1. **Microcode (senprog).** `DXP_VERBOSE=1` dump of the compiled bundle shows all
   32 cores emit `L3_LDU` **and** `L3_STU` RIU ring transfers, and core `i`
   targets core `31-i` (`(31-i) << 14` in the instruction's remote-core field:
   0→31, 1→30, …). The degenerate same-split STCDP emits **zero** `L3_LDU`/`L3_STU`
   — it is a pure same-core copy with the ring transfers dead-code-eliminated. The
   difference is the cross-core ring traffic, at the hardware-instruction level.
2. **Device execution.** The round trip runs on hardware **value-correct**
   (`max_err 0.0137`, identical to baseline) and with **no `ComputeHardwareError`**
   (code `0x7b1b`). The remove-the-senprog negative control fails as required,
   proving the device executed the spliced cross-core program.
3. **Logical lock.** Value-correctness *requires* the path core `i` → core `31-i`
   → core `i`: the consumer's `LX@8192` on core `i` is only ever written by
   STCDP2, whose `dataIN` reads core `31-i`'s scratch, which is only written by
   STCDP1 reading core `i`'s producer output. Since `i ≠ 31-i` for all 32 cores,
   a correct result is impossible without cross-core movement; the senprog
   confirms it is realized as ring transfers (not collapsed into a same-core copy).

**Conclusion.** A genuine cross-core ring STCDP — every core's activation slice
moving over the RIU ring to a remote core and back — runs on Spyre hardware,
value-correct, HBM-free, inside a mixed DL+data-op SuperDSC. The remaining
Compute-CB fault is confined to the `ReStickifyOpWithPTLx` transpose; the
cross-core **data movement** primitive itself is proven sound on device.

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
