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
head: 99f95650 test: cover bounded matmul operand relayout budget
```

Deeptools:

```text
repo: Adnan-Hoque1/deeptools
branch: ah/comms-collectives
head: 61ffb6b3a [DXP] test matmul operand relayout chunk cap
```

## Current Validation Evidence

Focused checks from CDX:

```text
Torch tests/inductor/test_lx_relayout_dldsc.py: 30/30 passed
Deeptools LayoutAllgatherRestickify.*: 32/32 passed
Deeptools CoreWorkDivIncomptLxRelayout*: 2/2 passed
Deeptools MatmulOperandBroadcastChunkCapFailsClosed: passed
```

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
| Broadcast | One source piece copied to all consumer cores, no arithmetic | Covered by topology classification and generic contract tests; matmul-operand path maps to `communication_pattern=broadcast` | Utility planner supports `grouped_broadcast` and rejects malformed contracts | No current DXP positive fixture from a full SDSC replay | Metadata and planner support exist; add a bounded DXP positive fixture before claiming full backend coverage |
| Multicast | One source piece copied to a subset/cohort of consumer cores, no arithmetic | Covered by topology classification and generic contract tests; matmul-operand path maps to `communication_pattern=multicast` | Utility planner supports `grouped_multicast` and rejects malformed contracts | Synthetic value run proves grouped fanout shape through the staged matmul-operand path, but the archived plan reports `all_gather_replicate` | Mostly covered at utility/planner level; add a clean DXP positive fixture whose plan explicitly reports `multicast` |
| Gather | Many distinct source pieces assembled onto one consumer core or compact set, no arithmetic | Covered by topology classification and `partial_view_gather` metadata emission | Generic fan-in cardinality is covered by `CoreWorkDivIncomptLxRelayoutCardinality`; the named `partial_view_gather` path is not yet proven at the same level | Granite SDSCs contain `partial_view_gather` classifications | Generic bounded fan-in is covered; Granite-specific `partial_view_gather` realization remains a next substrate gap |
| All-gather / replicate | N source pieces replicated so each consumer cohort can see the needed pieces, no arithmetic | Covered by topology classification and matmul operand contracts using `matmul_operand_broadcast` plus `communication_pattern=all_gather_replicate` | Generic replicated cardinality is covered by `CoreWorkDivIncomptLxRelayoutCardinality`; staged matmul operand all-gather is covered by `LayoutAllgatherRestickify.*`, the DXP chunk-cap fail-closed fixture, flash structural run, and Granite bounded plan artifacts | Flash probe removes HBM restickify structurally; Granite bounded plans emit 64/256/1024 transfer cases | Strongest currently proven class |
| Reduce / all-reduce | Many inputs combined arithmetically | Not in PR1 substrate scope | Needs an arithmetic reduction primitive, not just copy movement | None | Future work |

## Granite Spill Readout

| Granite edge / region | Observed class | Current behavior | Owner for remaining work |
|---|---|---|---|
| Flash attention value-side matmul operand | All-gather / replicate, staged as `matmul_operand_broadcast` | Bounded plans are emitted and realized through `gather_then_restickify`; flash structural probe has zero `ReStickifyOpHBM` | Communication substrate, currently best-covered |
| Attention partial view / compact view handoffs | Gather-like `partial_view_gather` | Classified in artifacts; backend realization is not yet proven at the same level as all-gather | Communication substrate next gap |
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

1. Add a small DXP positive fixture for the named matmul-operand path with
   `communication_pattern=broadcast`.
2. Add a small DXP positive fixture for the named matmul-operand path with
   `communication_pattern=multicast`.
3. Add a bounded backend proof for the Granite-specific `partial_view_gather`
   metadata path, or make its fail-closed behavior explicit.
4. Keep all oversized Granite activation cases classified as WSR/tile-scoping until WSR provides bounded live tiles.
5. Keep reduce/all-reduce separate because those require arithmetic, not just movement.

## One-Line Status

The branch is currently a bounded communication substrate with strong
all-gather/replicate evidence, scatter test coverage, broadcast/multicast
planner coverage, and incomplete gather backend evidence. It intentionally does
not solve large full-activation streaming; that belongs to WSR.
