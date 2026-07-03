# Attention 105_batchmatmul Contract Isolation - 2026-07-03

This note records the CDX isolation experiment for the latest flash-attention workload. The goal was to determine whether the remaining attention HBM spill is a PR1-style scatter, a layout-aware all-gather/restickify, or a generic matmul operand all-gather.

## Environment

- Pod: `adnan-cdx-spyre-dev-pf`
- Root: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525`
- Torch branch: `ah/comms-collectives`
- Deeptools branch: local experimental DLDSC collective checkout under the same root
- Split LX env: Torch sees `DXP_LX_FRAC_AVAIL=0`; wrapper maps `DXP_BACKEND_LX_FRAC_AVAIL=1` for the DXP subprocess
- Workload: `test-spyre-scripts/test_flash.py`

## Result 1: fissioned layout_allgather_restickify

Run directory:

`/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_allgather_only105_fission4_sourceonly_20260703_065227`

Key flags:

- `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1`
- `DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_ONLY_SDSC=105_batchmatmul`
- `DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_FISSION_ROWS=4`
- `DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_FISSION_SLICE=16`

Observed result:

- DXP/runtime completes.
- Correctness fails: `15530529 / 16777216` mismatched elements (`92.6%`), greatest absolute difference `inf`.
- Backend plan artifact: `105_batchmatmul_Tensor1_layout_allgather_restickify_fission4_sourceonly_plan_20260703.json`.
- Plan shape: 32 source cores, 32 consumer cores, 4 groups, 8 producer chunks per group, 8 consumer cores per group, 256 logical transfers.

Interpretation: this path is executable after the phase-cardinality and program-packaging fixes, but it is value-wrong. The 256-transfer contract is too compressed for this attention operand. It captures a grouped layout restickify summary, not the complete operand materialization needed by the matmul.

## Result 2: filtered generic matmul_operand_broadcast

Run directory:

`/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_filtered_20260703_070937`

Key flags:

- `SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=0`
- `SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1`
- `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_ONLY_SDSC=105_batchmatmul`

Observed result:

- Torch emits the generic `matmul_operand_broadcast` contract for `105_batchmatmul` Tensor1.
- Backend plan artifact: `105_batchmatmul_Tensor1_matmul_operand_broadcast_filtered_plan_20260703.json`.
- Plan shape: 32 source cores, 32 consumer cores, 1 group, 32 producer chunks per group, 32 consumer replicas, 1024 logical transfers.
- DXP aborts before runtime with `query fold dimension with higher fold factor` in `dsc/foldManager/foldInfrastructure.h`.

Interpretation: the richer contract has the expected transfer cardinality for full operand materialization, but current Deeptools cannot yet lower that loop-scoped matmul operand fetch into the consumer schedule. This is a different backend gap than the value-wrong 256-transfer layout path.

## Takeaway

The attention edge is not covered by PR1 scatter. We need at least one of the following production paths:

1. A correct transform-aware `layout_allgather_restickify` lowering that derives movement from full DLDSC tensor/compute coordinates and layout rename, not from the compact 256-transfer summary alone.
2. A robust loop-scoped `matmul_operand_broadcast`/all-gather lowering that can handle the 1024-transfer operand materialization without fold-manager or IBUF failures, preferably with fission/staging rather than resident full replication.

The second shape is conceptually cleaner for the matmul operand, but it needs backend scheduling support. The first shape may be cheaper if the layout transform can be fused into the movement, but the current prototype is not value-correct.
