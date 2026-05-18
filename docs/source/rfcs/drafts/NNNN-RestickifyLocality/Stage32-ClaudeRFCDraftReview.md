# Stage 32: Review Of Claude Ring-Aware Restickify RFC Draft

## Context

Claude pushed a draft RFC to branch:

```text
AdnanHoque/rfc-ring-aware-restickify
```

Reviewed commit:

```text
eaeade4 rfc: ring-aware restickify draft v1
```

Draft path:

```text
tests/rfc_ring_aware_restickify_draft.md
```

This stage is a review note only. It does not file a PR, merge anything, or
copy the draft into the upstream RFC repo.

## High-Level Read

The draft is pointing at a real and interesting path:

> Replace HBM round-trip restickify for truly cross-core, in-graph relayouts
> with an on-chip shuffle represented as an inductor-emitted `ComputeNode`.

It also identifies a plausible Deeptools integration blocker:

```text
Ddc::attachToPrefilledSchedule()
```

The local Deeptools source confirms that the function explicitly handles
`ALLOCATE`, `TRANSFER`, `LOOP`, and `BLOCK`, but has no `COMPUTE` branch. Since
prefilled nodes are already inserted into `metadata.externalNodes_`, the real
work is probably not "make DDC notice compute nodes" in the abstract. The work
is to validate and wire externally supplied compute nodes into the metadata and
downstream scheduling assumptions that DDL-created compute nodes normally
satisfy.

## What Looks Solid

### The Contract Gap Is Real

`attachToPrefilledSchedule()` currently validates prefilled schedule nodes and
fills several metadata structures:

- LX allocate nodes get `calculateClStartAddress(...)`.
- transfer nodes populate `prefilledExternalTransferToDataConnectToFill_`.
- loop nodes populate `dimToCoreChunkLoops_`.
- the `lx_below_schedule` block is discovered.

There is no equivalent compute-node path.

The draft's smallest Deeptools patch direction is therefore plausible: start by
adding careful `COMPUTE` handling to this contract and then address the first
downstream assumption that breaks.

### Schedule IR / DSC2 Can Represent The Concept

The Spyre knowledge base's Schedule IR design already describes compute as a
coarse node at the boundary between above-scratchpad scheduling and below-
scratchpad dataflow lowering. It also allows transfer metadata such as GTR for
multicast, and points to DSC2 schedule trees as the current production analog.

That does not prove the exact `STCDPOpLx` route is end-to-end ready, but it does
support the general architectural direction: the schedule boundary is the right
place to express data movement and coarse compute.

### The Motivation Matches Our Measurement Direction

Our earlier measurements showed that `ReStickifyOpHBM` behaves like an HBM
round trip: read HBM into core-local work, transform layout, write back. If a
future lowering can keep source and destination on-chip, the avoided HBM traffic
is real.

The draft is therefore complementary to Stage 3B:

- Stage 3B reduces modeled ring byte-hops inside the existing HBM restickify
  placement and work-division world.
- The Claude draft targets a bigger change: replacing the HBM restickify kernel
  for a subset of fundamental relayouts with an on-chip shuffle.

## What Needs Tightening Before An Upstream RFC

### Bandwidth Framing

The draft headline says the AIU RIU BiRing offers about `10.6 TB/s aggregate`
and uses that next to an HBM effective bandwidth number. This needs a careful
rewrite.

The canonical AIU 1.0 numbers we have been using are:

| Quantity | Value |
|---|---:|
| RIU data ring, one direction | `128 B/cyc * 1.3 GHz = 166 GB/s` |
| RIU data ring, bidirectional physical aggregate | `333 GB/s` |
| LPDDR / off-chip memory | about `166-170 GB/s` |

`10.6 TB/s` is `32 cores * 2 directions * 166 GB/s`. That is best understood
as whole-ring aggregate byte-hop capacity under idealized full utilization, not
as the payload bandwidth that a single tensor shuffle receives.

The draft's own `diag_ring_speedup_model.py` is more nuanced: it uses
`166e9` as the per-direction link bandwidth and distinguishes bisection,
uniform all-to-all, and aggregate models. The RFC should lead with that nuance
instead of the aggregate number.

### Per-Op Speedup vs End-To-End Speedup

The draft mixes a strong per-restickify claim with layer-level claims. Those
need to stay separate.

A fast ring shuffle can give a large per-op win for a restickify that is truly
HBM-bound and replaceable. But model-level speedup depends on how much of the
layer or graph is actually that replaceable traffic.

Claude's supporting findings already say a sampled cache mix was roughly:

| Category | Share |
|---|---:|
| HBM-load restickify / weight prep | 52% |
| Fundamental post-compute restickify | 4% |
| Matmul / pointwise compute | 44% |

That means ring-only replacement of the fundamental 4% bucket cannot produce a
large end-to-end speedup on that sample. The RFC should say this plainly, then
position the bigger gains as forward-looking: long-context, flash-attention-like
paths where fundamental restickify becomes a larger share.

### "All Three Pay HBM" Is Too Broad

The draft lists:

1. explicit `ReStickifyOpHBM`
2. optimizer absorption
3. `mm_t` fusion

and says all three pay HBM bandwidth at runtime. That needs qualification.

An explicit `ReStickifyOpHBM` is the cleanest HBM round-trip case. Absorption
and `mm_t` fusion may pay through non-natural matmul access patterns or extra
input traffic, but they are not necessarily the same standalone HBM
read-transform-write kernel. Treat them as "HBM-pressure or layout-compromise
mechanisms" rather than identical HBM round trips.

### "Every Other Layer Supports This" Is Too Strong

The draft says every other layer already supports the path and only the
prefilled-schedule contract is incomplete. The evidence is promising, but the
draft also lists hardware errors, matcher elision, and scheduler crashes from
the experiments.

Safer wording:

> Lower layers contain the relevant concepts and partial mechanisms, but an
> end-to-end valid prefilled ComputeNode path from torch-spyre SDSC through DDC,
> L3 scheduling, Dataflow IR, and device execution is not proven yet.

### The RFC Should Name The Current Unknowns

Before this is filed upstream, it should explicitly list:

- Which exact schedule-tree metadata DDL-created compute nodes carry that
  prefilled compute nodes must also carry.
- Whether externally supplied compute nodes need surrounding allocate, loop,
  sync, and data-connect nodes, or whether DDC can infer them.
- Whether `SNComputeLowering` handles the desired `SFPRING` input/output shape
  once the schedule reaches it.
- Whether the proposed path is RIU data ring, SFP ring, or both depending on
  lowering. The draft currently uses RIU ring, SFPRING, and `STCDPOpLx` language
  close together; the RFC should be precise about the fabric.

## Suggested RFC Reframe

The strongest upstream framing is:

> Torch-spyre can classify relayouts whose source data is already produced
> in-graph and whose consumer needs a different core ownership. Today, the only
> robust implementation path materializes that boundary through HBM. We want a
> schedule-level way to express a pure on-chip shuffle for those boundaries.

That avoids overclaiming Granite or full-model speedup while preserving the
important compiler architecture argument.

## Suggested Patch Strategy For The Deeptools Side

The smallest patch should come with a tiny Deeptools unit test before it is
plumbed through torch-spyre:

1. Create a minimal SDSC with a prefilled `ComputeNode` under the expected
   block, plus all required LX allocations and data infos.
2. Make `attachToPrefilledSchedule()` accept the compute node and validate:
   - compute node fields
   - input/output `DataInfo`
   - relevant components
   - dominance by allocations and loops
3. Run through `L3DlOpsScheduler_standalone`.
4. Inspect the scheduled JSON and lowered IR for the intended send/receive
   pattern.
5. Only then wire torch-spyre to emit such a node for one fundamental pattern.

This keeps the Deeptools patch narrow and makes the contract testable without
requiring a full model compile.

## Relationship To Our Current Branch

This draft is not a replacement for Stage 3B.

| Track | What It Tries To Do | Status |
|---|---|---|
| Stage 3B | Align producer and restickify work distribution to reduce modeled RIU byte-hops inside the current restickify world | Implemented as default-off prototype |
| GTR fanout | Use existing GTR multicast for shared HBM-pinned inbound data | Backend contract confirmed; torch-spyre shaping still open |
| Claude RFC | Replace some HBM restickifies with an explicit on-chip shuffle | Promising, but needs contract proof and tighter bandwidth claims |

## Recommendation

Do not file the draft as-is. Keep it as an experimental RFC seed.

The best next step is a two-part evidence package:

1. A Deeptools-only minimal prefilled-`ComputeNode` test that survives DDC and
   L3 scheduling.
2. A cleaned RFC that uses the per-direction RIU bandwidth model, separates
   per-op from end-to-end speedup, and states exactly which fabric and lowering
   path are being proposed.
