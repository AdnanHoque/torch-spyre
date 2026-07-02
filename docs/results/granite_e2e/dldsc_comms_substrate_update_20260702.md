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

Runtime validation on adnan-clc-spyre-dev-pf now has two useful checkpoints.

First, latest repo test_flash.py originally failed before SDSC emission with:

- NotImplementedError: buf10 (Pointwise): no mechanism to resolve stick incompatibility
- No mechanism to gather elements from multiple sticks into single stick

The zero-stick pointwise candidate patch gets past that frontend failure and emits SDSCs.

Second, with full Torch LX planning, SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1, and the split DXP wrapper, the flash activation edge changes shape in emitted SDSCs:

- Before: 32 ReStickifyOpHBM rows, 0 layout_allgather_restickify classifications.
- After: 0 ReStickifyOpHBM rows, 32 ReStickifyOpLx rows, and 32 layout_allgather_restickify classifications.
- Each classification records transfer_count=256, max_fanout=8, and max_fanin=8.

Representative artifacts are in docs/results/granite_e2e/flash_contract_20260702.

The run is still not end-to-end correct. DXP aborts with: Scheduler failed to find a suitable op mapping for sdsc: 2_ReStickifyOpLx. This is now the backend physical-lowering gap after the frontend contract is visible.

## Interpretation

The backend copy-movement substrate is ahead of the current frontend e2e path for scatter/gather/multicast-style movement. Torch can now emit the narrow flash layout-allgather-restickify contract into SDSC, including ReStickifyOpLx rows and batchmatmul classifications. The remaining narrow flash gap is full DXP/DSM physical lowering/scheduling for that ReStickifyOpLx plus grouped all-gather contract.

The current branch now has the classification vocabulary needed for cost-model and artifact analysis. It does not yet prove that flash HBM round trips are removed.

## Next Tasks

1. Wire Deeptools physical lowering/scheduling for the emitted ReStickifyOpLx plus layout_allgather_restickify contract.
2. Replay the emitted SDSC through current Deeptools and confirm no DXP/runtime failure.
3. Then rerun flash/granite profiling and claim HBM spill removal or speedup only after correctness and profiler traces pass.
4. Separately address the older pod-local all-gather probe failure: Unexpected stick expression 4*(Mod(d4, 16)).
