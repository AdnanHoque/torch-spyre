# DLDSC Communication Substrate Update - 2026-07-02

## Current Position

The DLDSC path is the right direction for the Granite communication work. The contract is:

1. Torch owns work division and records the logical tensor distribution expected at each producer/consumer edge.
2. Torch classifies the coordinate mismatch so the cost model can reason about scatter, broadcast/multicast, gather, and all-gather shaped traffic.
3. Deeptools owns physical movement synthesis from DLDSC coordinates using the existing STCDPOpLx/L3 ring machinery.
4. Scheduling/overlap remains a later optimization layer after value-correct movement is visible in emitted SDSCs.

This update folds in the findings from the ah/comms-collectives-dldsc-agent branch without replacing the richer artifact branch state.

## Torch Artifact Branch Delta

Branch: ah/comms-collectives

Files changed in this checkpoint:

- torch_spyre/_inductor/config.py
  - Adds SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1 as a guarded research knob for non-primary matmul operand coordinate collectives.
- torch_spyre/_inductor/lx_relayout.py
  - Adds coordinate-overlap topology classification.
  - Records consumer coordinates alongside producer coordinates.
  - Records communication_class, communication_pattern, max_fanout, max_fanin, and transfer_count.
  - Keeps physical transfer derivation in Deeptools; Torch only emits the logical contract/classification.
- tests/inductor/test_lx_relayout_dldsc.py
  - Adds focused topology tests for one-to-one scatter, broadcast, multicast, gather, and all-gather.

Local validation on adnan-spyre-dev-pf:

- python3 -m py_compile torch_spyre/_inductor/lx_relayout.py torch_spyre/_inductor/config.py tests/inductor/test_lx_relayout_dldsc.py: passed.
- git diff --check: passed.
- Focused pytest is blocked in this pod by the local runtime ABI mismatch: libspyre_comms.so.1: undefined symbol: flex::AllocationDirective...

## Backend Evidence

Backend branch inspected on adnan-cdx-spyre-dev-pf: Deeptools ah/comms-collectives at 966f1149e9e6cb02f8c5a2a102a9e6cc01083fc3.

Passing backend checks reported from CDX:

- ./util/util_unit_test --gtest_filter=LayoutAllgatherRestickify.*: 13/13 passed.
- ./dcg/dcg_unit_test --gtest_filter=stcdpLibtest.relayoutComplex:stcdpLibtest.relayoutDynMVLoop:stcdpLibtest.multicastSimple:stcdpLibtest.multicastSimpleZP: 4/4 passed.
- ./dxp/dxp_unit_test --gtest_filter=DxpTestFixture.CoreWorkDivIncomptLxRelayout: 1/1 passed.
- DCC FileCheck tests passed for L3 gather, L3 scatter, and core-to-core multicast.

Backend support status:

| Class | Current backend status | Notes |
| --- | --- | --- |
| scatter | backend generation and senpcfg pass for LX-overlap cases | Generated SDSC uses ScatterOpHBM naming, not literal STCDPOpLx, but the program verifies through the L3/LX movement path. |
| broadcast | supported as 1:many copy/multicast | Physical movement derives from coordinate overlap. |
| multicast | supported with literal STCDPOpLx evidence | Covered by DCG/DCC multicast tests. |
| gather | backend generation and senpcfg pass for LX-overlap cases | Generated SDSC uses GatherOpHBM naming, not literal STCDPOpLx, and this is not reduction semantics. |
| all-gather | narrow layout_allgather_restickify contract checker and logical movement-plan artifact only | Not yet wired into full DXP/DSM physical lowering. |
| reduce | not supported by this relayout path | Needs value-combining semantics, not pure copy relayout. |
| all-reduce | not supported by this relayout path | Existing DSM collective machinery is separate from DLDSC LX relayout. |

## Flash Runtime Status

Runtime validation on adnan-clc-spyre-dev-pf did not yet demonstrate flash spill removal.

Latest repo test_flash.py failed before SDSC emission with:

- NotImplementedError: buf10 (Pointwise): no mechanism to resolve stick incompatibility
- No mechanism to gather elements from multiple sticks into single stick

The pod-local all-gather flash probe also failed before SDSC emission with:

- Unexpected stick expression 4*(Mod(d4, 16))

Existing replay artifacts show the planner saw candidates, but persisted SDSCs still had no LX residency metadata/classification and removed zero ReStickifyOpHBM rows. The remaining flash rows are still the activation path:

mul -> ReStickifyOpHBM -> batchmatmul

This is a layout/restickify plus grouped all-gather handoff, not the simple scatter class.

## Interpretation

The backend copy-movement substrate is ahead of the current frontend e2e path for scatter/gather/multicast-style movement. The next useful implementation work is getting Torch to emit the DLDSC contract for the flash/restickify edge and survive stick-layout codegen to SDSC. The narrow flash layout-allgather-restickify path still needs full physical lowering after the contract is emitted.

The current branch now has the classification vocabulary needed for cost-model and artifact analysis. It does not yet prove that flash HBM round trips are removed.

## Next Tasks

1. Fix the flash frontend pointwise stick-choice issue so latest test_flash.py reaches SDSC emission. A minimal patch now keeps the zero-stick candidate available for mixed zero/non-zero multi-arg pointwise joins.
2. Preserve staged relayout metadata through the allocation/codegen path and confirm lxRelayoutClassifications_ appears in the batchmatmul SDSC.
3. Make the activation mul -> restickify -> batchmatmul handoff emit an LX-resident ReStickifyOpLx/layout-all-gather contract instead of an HBM ReStickifyOpHBM row.
4. Replay the emitted SDSC through current Deeptools and confirm no DXP/runtime failure.
5. Only then rerun flash/granite profiling and claim HBM spill removal or speedup.
