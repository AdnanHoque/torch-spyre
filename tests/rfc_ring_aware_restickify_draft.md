# RFC: Ring-aware restickify — enabling inductor-emitted cross-core data shuffle

**Status:** draft
**Author:** Adnan Hoque (torch-inductor team)
**Target repo:** github.com/torch-spyre/rfcs
**Related deeptools files:**
- [`ddc/ddcv1.cpp:2170`](../../deeptools/ddc/ddcv1.cpp) — `attachToPrefilledSchedule()`
- [`dsc/dsc2.cpp:1606-1668`](../../deeptools/dsc/dsc2.cpp) — ComputeNode JSON parser
- [`dsc/dsc2.cpp:699-747`](../../deeptools/dsc/dsc2.cpp) — ComputeNode JSON serializer
- [`dcc/src/Conversion/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:540-704`](../../deeptools/dcc/src/Conversion/DSC2ToDataflowIR/V3/SNComputeLowering.cpp) — SFPRING-aware lowering
- [`dcc/test/Transform/SetSendDestinationRE/sfp-basic-1.mlir`](../../deeptools/dcc/test/Transform/SetSendDestinationRE/sfp-basic-1.mlir) — working Sentient IR pattern

## 1. Summary

For attention-heavy workloads the inductor side detects a category of
relayout — **FUNDAMENTAL restickify** — where the producer and consumer
operations require different per-core partitions and a cross-core
shuffle is the only correct lowering. Today every such relayout pays an
HBM round-trip (~2·B/107 GB/s effective). The AIU's on-chip RIU BiRing
offers ~10.6 TB/s aggregate, ~64× the effective HBM bandwidth, so a
ring-based shuffle (`STCDPOpLx`) saves ~5–28× per FUNDAMENTAL restickify
and ~1.13–1.31× per attention layer at M=8k (validated empirically; see
§3).

The proposal: **expand the SDSC JSON contract between torch-spyre
inductor and deeptools to allow inductor-emitted `ComputeNode` entries
that drive pure cross-core data shuffle**, by adding a `COMPUTE` branch
to `attachToPrefilledSchedule()`. Every other layer of the stack
(hardware, Sentient IR, DSC2 → Dataflow IR lowering, JSON
serializer/parser) already supports this; only the pre-filled-schedule
contract is incomplete.

## 2. Motivation

torch-spyre's planner identifies three FUNDAMENTAL-restickify absorption
mechanisms today (probe v1+v2 empirical study, see
[`tests/diag_restickify_lx_findings.md`](diag_restickify_lx_findings.md)
for full data):

1. **Explicit restickify kernel** — emitted as `ReStickifyOpHBM`, 2·HBM round-trip.
2. **Optimizer absorption** — `optimize_restickify` picks a non-natural matmul STL so the producer aligns with the consumer; matmul perf hit instead of HBM hit, but the cost is paid in the matmul kernel.
3. **`mm_t` kernel fusion** — matmul lowers to a specialized `mm_t` kernel that handles the transposed input inline; HBM cost paid in the kernel's input read pattern.

All three pay HBM bandwidth at runtime. The ring-aware-restickify
project replaces all three with an on-chip `STCDPOpLx` shuffle when the
restickify is FUNDAMENTAL (producer and consumer want incompatible
partitions). Probe v2 at HD=4096, SENCORES=32, LX_PLANNING=1, sweep
M={128, 512, 2048, 8192} validated the HBM cost model:
`Δ_measured/Δ_pred = 0.85–0.89×` where `Δ_pred = 2·|X|/107 GB/s`. The
24× per-op speedup ceiling (~22× empirical floor) holds when the
cross-core shuffle is real ring transport.

The 1H 2026 roadmap workloads (Llama, Mistral, Granite, GPT-OSS) are
attention-heavy. Eliminating the HBM tax on FUNDAMENTAL restickifies
unblocks measurable per-layer wins on every model in the lineup.

## 3. What works today vs. what doesn't

### Inductor side — complete

- **Classifier** that distinguishes FUNDAMENTAL vs INCIDENTAL vs HBM_LOAD restickifies. See [`tests/diag_restickify_lx_trace.py`](diag_restickify_lx_trace.py); decision rule in [`tests/restickify_categories.md`](restickify_categories.md).
- **Cost-function gate** in `optimize_restickify.EdgeCostMap` keyed on `torch_spyre.config.ring_aware_restickify`; on, the optimizer leaves FUNDAMENTAL restickifies explicit rather than absorbing.
- **Codegen swap** at [`spyre_kernel.py:516`](../torch_spyre/_inductor/spyre_kernel.py#L516) — swaps `RESTICKIFY_OP` from `"ReStickifyOpHBM"` to `"STCDPOpLx"` when the gate is on.
- **Scratchpad fix** (commit `1f26ba2`) — tolerates op users without `op_it_space_splits`, required for the gate-on path.

End-to-end pipeline proven: gate fires → inductor emits SDSC bundle
with `STCDPOpLx` SDSC → DCC matches a registered template → bundle
compiles → device dispatches. **Verified with Phase B iter 2** (PE
pass-through template, see [`PHASE_B_UPDATE.md` in spyre-profile](https://github.ibm.com/Adnan-Hoque1/spyre-profile/blob/main/PHASE_B_UPDATE.md)):
bitwise-identical to HBM baseline, wall time ~1.04× (real work, no
cross-core savings — PE is a pass-through, data still round-trips
through each core's local LX).

### Deeptools side — partial

The hardware supports pure cross-core data movement. The lowering stack
supports it from DSC2 downward. But the *contract* by which inductor
hands work to deeptools doesn't have a way to express it.

#### Five-variant DDL template sweep (see [`PHASE_B_ITER3_FINDING.md`](https://github.ibm.com/Adnan-Hoque1/spyre-profile/blob/main/PHASE_B_ITER3_FINDING.md))

| Variant | Compile | SFP in senprog | Runtime |
|---|---|---|---|
| v1 (FMA16 + `sfp_input`, no PE) | 24 KB hex | ✓ present | **Compute CB hardware error** |
| v2 (MACC + `sfp_lx_input`, topk-style) | 11 KB hex | ✗ absent | matcher elided → LX direct (= iter 2) |
| v3 (two separate SFP LRFs) | 11 KB hex | ✗ absent | same as v2 |
| v4 (v1 + `vias=["pe"]`) | byte-identical to v1 | ✓ present | same HW error as v1 |
| v5 (bmm-style PE feeder) | 11 KB hex | ✗ absent | matcher elided → LX direct |

Pattern: `data_connect="sfp_input"` keeps the sfpring chain alive but
generates SFP_FMA instructions whose PE-input port is unwired (HW
error). Switching to "routed" data_connect names or adding a
pass-through PE feeder makes the matcher recognize the degenerate
pattern and elide the sfpring entirely. The DDC matcher cannot
distinguish "pure cross-core data shuffle" from "broken reduction
template" — by design, since `core_to_core_communication` is the only
cross-core DDL constructor and it's psum-shaped.

#### SDSC JSON injection experiment

Hand-edited the captured `STCDPOpLx` SDSC to add a `ComputeNode` with
`inputs_=["lx", "one", "zero"]`, `outputs_=["sfpring"]`, and per-core
`startAddr_.data_` table encoding a cyclic-shift dest mapping (the
exact pattern STCDP needs). Ran via `dxp_standalone`.

Progressive crash points trace the contract gap:

| Step | Result |
|---|---|
| `prev_: "<allocate-node>"` | segfault at [`dsc2.cpp:1336`](../../deeptools/dsc/dsc2.cpp#L1336) — `static_cast<BlockNode*>` on an Allocate |
| `prev_: ""` (root) | parser accepts; DDC starts processing |
| Full ComputeNode body | reaches [`L3DlOpsScheduler::fillLoopOffsetsAndAddresses` at line 5182](../../deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp#L5182) — `DataLocation → int` map missing key for tensor LX allocations our ComputeNode references (which don't exist yet in the prefilled schedule, since DDL parsing normally creates them) |

Each crash is a downstream pass assuming "compute nodes only come from
DDL template parsing, which is guaranteed to set up the surrounding
LX allocations, loops, syncs first." The pre-filled schedule contract
([`attachToPrefilledSchedule()`](../../deeptools/ddc/ddcv1.cpp#L2170))
handles `ALLOCATE`, `TRANSFER`, `LOOP`, `BLOCK` explicitly and has no
`COMPUTE` branch.

#### What works at every layer

| Layer | Status | Evidence |
|---|---|---|
| Hardware | ✓ | `SFP_FMA(0, 1, lx) → sfpring + set_send_dst(core_id)` documented in KB and tested in Sentient IR test fixtures |
| Sentient IR | ✓ | [`sfp-basic-1.mlir`](../../deeptools/dcc/test/Transform/SetSendDestinationRE/sfp-basic-1.mlir) — `vector_mac` with `opA=zero, opB=one, opC=lx, ResultForwarding=[sfpring]` and explicit per-core `set_send_dst` |
| Dataflow IR | ✓ | unit types include `"l3lu"`, `"l3su"`, `"ring"`, `"sfpring"`; `dataflow.send/receive_op` with cross-core unit handles |
| DSC2 → Dataflow IR | ✓ | [`SNComputeLowering.cpp:540-704`](../../deeptools/dcc/src/Conversion/DSC2ToDataflowIR/V3/SNComputeLowering.cpp#L540) handles `inputs_[i] == SFPRING` with arbitrary per-core `startAddr_` |
| ComputeNode JSON ser/de | ✓ | [`dsc2.cpp:699-747`](../../deeptools/dsc/dsc2.cpp#L699) writes, [`:1606-1668`](../../deeptools/dsc/dsc2.cpp#L1606) reads — full round-trip with SFPRING inputs/outputs and per-core fold tables |
| DDC `attachToPrefilledSchedule()` | ✗ | no COMPUTE branch |

## 4. Proposal — three options, ordered by size

### Option 1 (recommended) — extend the pre-filled schedule contract

Add a `COMPUTE` branch to `attachToPrefilledSchedule()` at
[`ddcv1.cpp:2170`](../../deeptools/ddc/ddcv1.cpp#L2170), and corresponding
support in downstream passes that currently assume DDL-only origin for
compute nodes (most notably the `DataLocation → int` map in
[`L3DlOpsScheduler::fillLoopOffsetsAndAddresses`](../../deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp#L5182)).

The minimum change set, as far as our local exploration has gone:

1. **`attachToPrefilledSchedule()` COMPUTE branch** — validate node fields, register `relevantComps_`, mark as external.
2. **`L3DlOpsScheduler` allocation-lookup paths** — for externally-supplied compute nodes that reference LX, either accept inductor-pre-populated LX allocations or short-circuit the lookup when the ComputeNode is marked external.
3. **`dsc2.cpp:1334` cast safety** — replace `static_cast<BlockNode*>` with a checked cast or document `prev_` must reference a `BlockNode`.

Inductor side, no API changes — torch-spyre's existing SDSC emitter
gets a new code path that populates `scheduleTree_` with the right
ComputeNode entries when `config.ring_aware_restickify` is on. The
inductor team has the FUNDAMENTAL-vs-INCIDENTAL context that the
backend doesn't, so it's the natural place to emit the per-core
addressing tables.

**Why recommended**: smallest deeptools change, narrowest interface,
preserves backend autonomy over the rest of the schedule tree. Every
other layer below already supports the path. Once the contract is
extended, both inductor and any future frontend can emit cross-core
data shuffle without needing new primitives.

### Option 2 — provide a worked DDL template example

Publish a DDL template that uses existing primitives (`sfpring` +
`core_to_core_communication`) for pure data shuffle, surviving DDC's
matcher elision. Our five-variant sweep didn't find one; the team
that owns DDC's matcher would have a far better view of what
incantation works.

### Option 3 — add a new DDL DSL primitive

Add `unit="cross_core_send"` or `ddl.cross_core_addr_table` that
generates ComputeNode entries with explicit per-core SFPRING
addressing, bypassing the psum-shaped `core_to_core_communication`.
Largest change, but a clean primitive for any future template author.

## 5. Cost model and validation plan

Cost model (from [`tests/diag_ring_speedup_model.py`](diag_ring_speedup_model.py)):

- HBM round-trip: `2·|X| / (107 GB/s effective)` per FUNDAMENTAL restickify
- Ring shuffle: `|X| / (effective ring BW)` where effective ring BW varies with chain depth and pattern
- Speedup ratio: `~0.04 × HBM` for the validated 10.6 TB/s aggregate ring bandwidth

Validation steps once Option 1 lands:

1. Inductor emits ComputeNode entries for one FUNDAMENTAL pattern (e.g., `(a@b).t() @ c` at S=512, 32 cores).
2. Verify via `dxp_standalone`: schedule tree contains ComputeNode with SFPRING I/O, lowered Sentient IR has `vector_mac` with `ResultForwarding=[sfpring]`, senprog has per-core `set_send_dst`.
3. Run E2E through torch-spyre + Phase 3 gate. Check:
   - Bitwise correctness vs HBM baseline (must match — same numerical operation, different transport).
   - Wall-clock speedup at multiple shapes. Expected: ~0.04× the iter 2 baseline (i.e., real cross-core savings).
4. Confirm classifier correctly identifies FUNDAMENTAL vs INCIDENTAL — INCIDENTAL patterns must still use ReStickifyOpHBM (no regression).

## 6. Out of scope

- Inter-chip collectives (separate library, see `dsm/coll/`).
- The original probe-v0 "ring-aware core permutation within ReStickifyOpHBM" — empirically buys nothing (HBM bandwidth is flat ~105-107 GB/s regardless of core permutation).
- Any speedup for HBM_LOAD-classified restickifies (graph-input weight prep — data is in HBM by definition, ring can't help).

## 7. Open questions for the deeptools team

1. **Architectural preference**: Option 1 (extend the contract so inductor can emit cross-core) vs Option 2 (worked DDL example) vs Option 3 (new DDL primitive). Inductor team's preference is Option 1; what's deeptools'?
2. **Scope of `attachToPrefilledSchedule` extension**: is "compute node passthrough" sufficient, or do downstream passes have other DDL-origin assumptions we'd need to surface?
3. **Backwards compatibility**: any existing DDC consumers relying on "scheduleTree_ from inductor only contains ALLOCATE/TRANSFER/LOOP/BLOCK" that would break?
