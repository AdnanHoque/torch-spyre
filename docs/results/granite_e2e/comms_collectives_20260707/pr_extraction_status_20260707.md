# DLDSC LX Collectives PR Extraction Status - 2026-07-07

This note records the clean branch extraction from the broad LX collective prototype. The goal is to avoid one large PR and instead stage communication classes on top of the PR1 scatter foundation.

## Source branches

- Torch artifact/lab branch: `ah/comms-collectives`
- Torch broad prototype source: `gather-restickify`
- Deeptools broad prototype source: `ah/comms-collectives`
- Torch PR1 base: `pr-lx-relayout-scatter`
- Deeptools PR1 base: `adnan/lx-relayout-scatter-sizing` / PR 4408 head

## Torch extraction

These branches are pushed to `AdnanHoque/torch-spyre` and are stacked in this order:

1. `pr-lx-relayout-allgather-restickify`
   - Base: `origin/pr-lx-relayout-scatter`
   - Extracts all-gather/restickify classification and bounded matmul operand contract scaffolding.
   - Local static check: `py_compile` passed for touched Python modules.
   - Local pytest was not run on the Mac because this environment does not have `torch`.

2. `pr-lx-relayout-broadcast-multicast`
   - Base: `pr-lx-relayout-allgather-restickify`
   - Tiny enablement slice for multicast matmul operand relayout classification.
   - Local static check: `py_compile` passed for `torch_spyre/_inductor/lx_relayout.py`.

3. `pr-lx-relayout-partial-view-gather`
   - Base: `pr-lx-relayout-broadcast-multicast`
   - Extracts partial-view/source-offset gather metadata and tests.
   - Local static check: `py_compile` passed for touched Python modules.

## Deeptools extraction

These experimental extraction branches now live in the user fork, `Adnan-Hoque1/deeptools`.
The official `ai-chip-toolchain/deeptools` repository is reserved for active/review-ready PR
branches. As of this cleanup, the official repo keeps only the active PR1 scatter branch:

- `adnan/lx-relayout-scatter-sizing` at `611687dc34e53fd7be0ceb37e74cfbf85010abf1`

The fork `master` was fast-forwarded to official `master`:

- `master` at `ff1c7c676cdc8f319f90fe7baa666db2a1103327`

Mirrored fork experiment branches:

1. `adnan/lx-relayout-allgather-restickify`
   - Fork SHA: `d6ac31ea82b445f6a65e3bb9ee314a1cf0e63fc9`
   - Base: PR 4408 head / PR1 scatter base `611687dc34e53fd7be0ceb37e74cfbf85010abf1`.
   - This branch was force-updated from the old broad prototype pointer to a clean PR2 slice.
   - Contains the hand-sliced bounded all-gather/restickify materializer:
     - LX relayout classification preservation and `LayoutAllgatherRestickify` utility.
     - Matmul operand broadcast/all-gather contract parsing for `kind=matmul_operand_broadcast` and `communication_pattern=all_gather_replicate`.
     - Bounded DXP materialization that emits `STCDPOpLx` gather chunks followed by local `ReStickifyOpLx` chunks before the consumer compute.
     - Chunk-cap and malformed/missing metadata fail-closed behavior.
     - Focused utility tests and a bounded SuperDSC fixture under `dxp/test/test_matmul_operand_broadcast_chunk_cap`.
   - Explicitly excluded from this PR2 slice:
     - `partial_view_gather` and source-offset gather.
     - Generic broadcast/multicast enablement.
     - Full Granite streaming / WSR behavior.
   - Hygiene:
     - `git diff --check origin/adnan/lx-relayout-scatter-sizing..HEAD` passes.
     - `git clang-format --diff origin/adnan/lx-relayout-scatter-sizing -- <touched C++ files>` reports no further modifications.
     - Scope grep over touched files found no `partial_view`, `PartialView`, broadcast/multicast enablement, source-offset gather, or WSR/streaming code.
   - Validation on CDX pod:
     - Pod: `adnan-cdx-spyre-dev-pf`.
     - Checkout: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools`.
     - Log directory: `/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/logs_pr2_clean_d6ac31ea82_20260708_024845`.
     - Build command: `ninja-build util/util_unit_test dxp/dxp_unit_test`.
     - `SPYRE_LX_PLANNER_RELAYOUT=1 util/util_unit_test --gtest_filter=LayoutAllgatherRestickify.*`: 19/19 passed.
     - `SPYRE_LX_PLANNER_RELAYOUT=1 dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*:DxpTestFixture.MatmulOperandAllGatherRestickify*"`: 3/3 passed.
   - Debugging notes from this slice:
     - The first hand-slice build missed `TransferNode::lxNeighborRingTransfers_`; this was fixed by adding the narrow transfer payload to `dsc/dsc2.h`.
     - The next DXP run reached DCC but failed because inserted data-op descriptors had no PCFGs. The root cause was DXP routing scheduled mixed DL+data SDSCs through `runDcgForDlOpsStandalone`; the branch now routes them through `runDcgForDataOpsDlOps`.
     - After routing was fixed, the remaining failure was only the test's stale plan-artifact filename; the emitted artifact name is derived from `sdsc name + lds name + classification key`.

2. `adnan/lx-relayout-broadcast-multicast`
   - Fork SHA: `3558acd1423a6d20eabadcb4d8148d0c66a34c6c`
   - Currently stacked on `adnan/lx-relayout-allgather-restickify`.
   - Branch pointer reserved.
   - Attempting to cherry-pick bounded broadcast/multicast immediately showed a dependency on the not-yet-extracted bounded `gather_then_restickify` materializer.

3. `adnan/lx-relayout-partial-view-gather`
   - Fork SHA: `3558acd1423a6d20eabadcb4d8148d0c66a34c6c`
   - Currently stacked on `adnan/lx-relayout-broadcast-multicast`.
   - Branch pointer reserved.
   - Should receive the partial-view/source-offset hand-slice only after Deeptools PR2 materialization is isolated.

4. `ah/comms-collectives-dldsc-agent`
   - Fork SHA: `b0d94ac421cdde2d0472e0d2a89df962d4e0751e`
   - Broad experimental agent branch mirrored out of the official repo.

Cleanup performed:

- Mirrored the official experimental refs into `Adnan-Hoque1/deeptools`.
- Deleted the non-PR experiment refs from `ai-chip-toolchain/deeptools`:
  - `adnan/lx-relayout-allgather-restickify`
  - `adnan/lx-relayout-broadcast-multicast`
  - `adnan/lx-relayout-partial-view-gather`
  - `ah/comms-collectives-dldsc-agent`
- Retargeted the local Deeptools checkout so `adnan/lx-relayout-allgather-restickify`
  tracks the fork remote branch instead of the now-deleted official branch.

## Why Deeptools needed hand-slicing

The broad Deeptools prototype history was not authored in clean PR-order. The useful bounded materialization commit, `[DXP] add bounded gather restickify relayout path`, was created after partial-view gather experiments. As a result, direct cherry-pick brings partial-view helpers/tests into PR2 context. I stopped instead of letting PR2 become a disguised mega-PR.

The Deeptools PR2 branch now manually lifts only the bounded `gather_then_restickify` pieces:

- DXP: create bounded gather/restickify rows from `matmul_operand_broadcast` / `all_gather_replicate` metadata.
- DCG/DDC/DCC: reuse existing `STCDPOpLx` and `ReStickifyOpLx` lowering; no new public data-op kind.
- Tests: keep the bounded all-gather/restickify fixture and chunk-cap/fail-closed tests.
- Exclude: partial-view source offsets, generic broadcast/multicast enablement, full Granite streaming, and WSR behavior.

## Public flag policy

The Torch-facing public feature flag remains:

```bash
SPYRE_LX_PLANNER_RELAYOUT=1
```

Other Deeptools knobs that appear in prototype history should be treated as diagnostic or extraction scaffolding, not as user-facing feature flags.
