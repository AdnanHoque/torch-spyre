# DLDSC LX Communication Substrate Coverage - 2026-07-07

This note records the current evidence for the Granite communication-substrate
goal. It separates communication support from WSR/tile-scoping support.

## Scope Boundary

This branch owns the bounded communication substrate:

- classify the producer-to-consumer edge;
- express the edge in DLDSC metadata;
- let Deeptools realize one bounded LX-resident tile correctly;
- fail closed or preserve HBM fallback when the full tensor is too large.

This branch does not own full-tensor streaming. If a Granite activation is too
large to materialize safely in LX, that is WSR-shaped work: split the live
region into smaller tiles first, then apply the same communication substrate to
each tile.

## Code Heads

Torch:

```text
repo: AdnanHoque/torch-spyre
branch: gather-restickify
head: 6bc8b00d inductor: emit partial-view gather relayout metadata
previous bounded-budget checkpoint: 99f95650 test: cover bounded matmul operand relayout budget
```

Deeptools:

```text
repo: Adnan-Hoque1/deeptools
branch: ah/comms-collectives
head: faa78233e [DXP] fail closed for partial-view gather relayout
previous bounded-broadcast checkpoint: 9e9b20b42 [DXP] test bounded matmul operand broadcast patterns
```

## Current Validation Evidence

Focused checks from CDX:

```text
Torch tests/inductor/test_lx_relayout_dldsc.py at 99f95650: 30/30 passed
Torch partial-view update at 6bc8b00d: py_compile passed
Torch partial_view_gather helper smoke at 6bc8b00d: passed with _C stub
Deeptools LayoutAllgatherRestickify.*: 32/32 passed
Deeptools MatmulOperandBroadcastPattern*: 2/2 passed
Deeptools CoreWorkDivIncomptLxRelayout*: 2/2 passed
Deeptools MatmulOperandBroadcastChunkCapFailsClosed: passed
Deeptools PartialViewGatherFailsClosedBeforeGenericLxRelayout: passed
```

The partial-view Torch pytest could not be run in that exact CDX worktree
because `_C.so` was linked against a different Flex ABI. The helper was
validated by stubbing `_C`, and the Deeptools behavior was validated through the
real DXP unit test.

Concrete flash structural probe:

```text
run: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_one_flag_gather_restickify_20260706_175419
returncode: 0
ReStickifyOpHBM: 0 in the structural summary
ReStickifyOpLx: 64 files mentioning ReStickifyOpLx
matmul_operand_broadcast plans: 32
backend plans realized: 32
communication_pattern: all_gather_replicate
realization_strategy: gather_then_restickify
logical transfers: 8192
```

Concrete Granite structural probe:

```text
run: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_bounded_substrate_default4m_20260707_050254
case: prefill, seq_len=512, sdpa_causal
status: generated SDSC and backend plans, then failed later in an unrelated flex convert_address runtime path
```

That Granite run still proves classification and DXP lowering for bounded
attention-side communication. It does not prove end-to-end Granite timing.

## Communication Coverage Matrix

| Class | Meaning | Torch classification / DLDSC metadata | Deeptools bounded realization | Concrete artifact | Status |
|---|---|---|---|---|---|
| Scatter / permutation | N to N ownership reassignment, no duplication, no arithmetic | Covered by `LXRelayoutTopology` and `test_coordinate_topology_classifies_one_to_one_scatter`; generic relayout metadata records `kind=scatter` | Covered by generic core-work-division relayout tests `CoreWorkDivIncomptLxRelayout*` | PR1 scatter path and unit tests; not the main flash evidence path | Supported for bounded same-stick relayout, but current Granite/flash artifact set is not primarily exercising this class |
| Broadcast | One source piece copied to all consumer cores, no arithmetic | Covered by topology classification and generic contract tests; matmul-operand path maps to `communication_pattern=broadcast` | Utility planner supports `grouped_broadcast`; DXP compile-positive fixture mutates the compact generated SDSC to a bounded 1-source / 32-consumer broadcast | `DxpTestFixture.MatmulOperandBroadcastPatternBroadcastCompiles` | Supported for bounded named matmul-operand broadcast |
| Multicast | One source piece copied to a subset/cohort of consumer cores, no arithmetic | Covered by topology classification and generic contract tests; matmul-operand path maps to `communication_pattern=multicast` | Utility planner supports `grouped_multicast`; DXP compile-positive fixture mutates the compact generated SDSC to a bounded 4-source / 32-consumer grouped multicast | `DxpTestFixture.MatmulOperandBroadcastPatternMulticastCompiles` plus earlier synthetic value grouped fanout artifact | Supported for bounded named matmul-operand multicast |
| Gather | Many distinct source pieces assembled onto one consumer core or compact set, no arithmetic | Covered by topology classification; `partial_view_gather` metadata now uses TensorArg source provenance and `source_offset_elems` | Generic fan-in cardinality is covered by `CoreWorkDivIncomptLxRelayoutCardinality`; named `partial_view_gather` now fails closed before generic relayout so Deeptools cannot silently ignore the source offset | `DxpTestFixture.PartialViewGatherFailsClosedBeforeGenericLxRelayout`; Torch helper smoke for `buf33 + 12800` | Generic bounded fan-in is covered; offset-aware `partial_view_gather` physical realization remains a next substrate gap |
| All-gather / replicate | N source pieces replicated so each consumer cohort can see the needed pieces, no arithmetic | Covered by topology classification and matmul operand contracts using `matmul_operand_broadcast` plus `communication_pattern=all_gather_replicate` | Generic replicated cardinality is covered by `CoreWorkDivIncomptLxRelayoutCardinality`; staged matmul operand all-gather is covered by `LayoutAllgatherRestickify.*`, the DXP chunk-cap fail-closed fixture, flash structural run, and Granite bounded plan artifacts | Flash probe removes HBM restickify structurally; Granite bounded plans emit 64/256/1024 transfer cases | Strongest currently proven class |
| Reduce / all-reduce | Many inputs combined arithmetically | Not in PR1 substrate scope | Needs an arithmetic reduction primitive, not just copy movement | None | Future work |

## Granite Spill Readout

| Granite edge / region | Observed class | Current behavior | Owner for remaining work |
|---|---|---|---|
| Flash attention value-side matmul operand | All-gather / replicate, staged as `matmul_operand_broadcast` | Bounded plans are emitted and realized through `gather_then_restickify`; flash structural probe has zero `ReStickifyOpHBM` | Communication substrate, currently best-covered |
| Attention partial view / compact view handoffs | Gather-like `partial_view_gather` | Torch can emit source-base plus constant-offset metadata; Deeptools now refuses the named class before generic relayout can value-wrong the offset | Communication substrate next gap: implement bounded offset-aware partial-view gather lowering |
| Generic producer/consumer core-div mismatch | Scatter / permutation | PR1-style DLDSC relayout path is covered by focused tests | Communication substrate, but needs concrete Granite edge artifact if it becomes performance-critical |
| MLP down-projection activation | Large matmul operand fanout/gather-like relayout | Explicitly skipped at 13,107,200 bytes with 4 MiB bounded budget; HBM fallback preserved | WSR/tile-scoping first, then communication substrate per tile |
| Weight restickifies | Weight layout issue | Not targeted | Offline/preload weight layout work |
| Reduce/all-reduce from split-K partials | Arithmetic collective | Not implemented | Future backend primitive plus frontend classification/costing |

## Why The Bounded Guard Matters

The MLP down-projection activation previously expanded into thousands of
movement chunks. That is not a healthy proof of the communication substrate.
The guard keeps bounded collectives enabled while preventing a large full tensor
from becoming an illegal or silently truncated descriptor list.

The intended behavior is:

```text
bounded tile fits -> classify, emit DLDSC, realize on chip
full tensor too large -> preserve HBM fallback or fail closed with a clear reason
large activation needs performance -> wait for WSR to tile the live region
```

## Current Gaps

1. Add bounded offset-aware physical realization for Granite-specific
   `partial_view_gather`. Current behavior is intentionally fail-closed so a
   source-offset view cannot be lowered as a generic offset-zero LX relayout.
2. Keep all oversized Granite activation cases classified as WSR/tile-scoping until WSR provides bounded live tiles.
3. Keep reduce/all-reduce separate because those require arithmetic, not just movement.

## One-Line Status

The branch is currently a bounded communication substrate with strong
all-gather/replicate evidence, scatter test coverage, broadcast/multicast DXP
compile-positive coverage for named matmul-operand metadata, and explicit
partial-view gather source-offset classification plus fail-closed backend
guarding. It intentionally does not solve offset-aware partial-view physical
lowering or large full-activation streaming yet; the latter belongs to WSR.
