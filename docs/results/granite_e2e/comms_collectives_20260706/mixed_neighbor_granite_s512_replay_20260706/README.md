# Mixed Neighbor Granite S512 Replay Probe

Date: 2026-07-06
Pod: adnan-cdx-spyre-dev-pf
Deeptools worktree: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
Base Deeptools head: 9092d48e0ae6af1d7cc66e4bd6128f2196e7f495
Replay source bundle: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_oneflag_no_wrapper_20260706_202931/block_prefill/cache/inductor-spyre/sdsc_fused_add_linear_mul_silu_split_with_sizes_3_clt9lx2o
Replay output: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_clean_mixed_neighbor_pass_20260706_223805

## Result

Focused tests passed:

- LayoutAllgatherRestickify.*: 32/32
- DxpTestFixture.CoreWorkDivIncomptLxRelayout*: 2/2

Granite S512 DXP replay passed:

```
rc=0
out=/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/granite_s512_clean_mixed_neighbor_pass_20260706_223805
plan_count=1
```

## Flags Used

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
export DEEPTOOLS_ALLOW_DIRECT_KERNEL_NEIGHBOR_LAYOUT_BYPASS=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR=<run>/plans
```

This is not yet the final one-gate handoff. It is the backend proof that the mixed HBM + neighbor matmul operand path can pass DXP replay after the DDC/runtime gaps below are addressed.

## Gaps Closed By This Probe

1. Row-bundling retry cycle for KERNEL-neighbor LX allocation feeding a single PT row transfer.
2. Missing loop-distribution metadata for synthetic relayout transfers; fallback to data-stage loop offsets under relayout gate.
3. Coordinate-offset solving on relayout transfer nodes without explicit core maps; skip coordinate offsets for those transfer nodes.
4. Preserve source allocation and allocate a dedicated destination for matmul operand neighbor movement.
5. Allow INPUT/KERNEL operand classification in the matmul broadcast/gather restickify utility path.

## Next Work

- Remove/merge remaining diagnostic flags behind `SPYRE_LX_PLANNER_RELAYOUT` where appropriate.
- Clean the patch into the Deeptools `ah/comms-collectives` branch.
- Run AIU/runtime validation for flash and Granite block, not just DXP replay.
- Verify value correctness, since this replay only proves DXP/DCC acceptance.
