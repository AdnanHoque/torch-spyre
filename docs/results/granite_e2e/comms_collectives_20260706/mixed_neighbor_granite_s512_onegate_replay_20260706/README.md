# Granite S512 mixed-neighbor replay with one public gate

This directory archives the DXP replay proving the mixed-neighbor gather/restickify carrier now runs with only the public relayout gate enabled.

## Source

- Deeptools branch: Adnan-Hoque1/deeptools ah/comms-collectives-mixed-neighbor-probe
- Deeptools head: da52ebdee
- Upstream master observed at generation: 949cfeea8
- SDSC input: runs/granite_s512_oneflag_no_wrapper_20260706_202931/block_prefill/cache/inductor-spyre/sdsc_fused_add_linear_mul_silu_split_with_sizes_3_clt9lx2o

## Command shape

```bash
export SPYRE_LX_PLANNER_RELAYOUT=1
export DXP_LX_FRAC_AVAIL=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR=/path/to/plans
unset DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR
unset DEEPTOOLS_ALLOW_DIRECT_KERNEL_NEIGHBOR_LAYOUT_BYPASS
unset DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC
dxp_standalone -d /path/to/sdsc_bundle
```

DXP_LX_FRAC_AVAIL=1 is a direct-DXP-replay capacity setting. It is not a feature flag. Torch-launched runs use SPYRE_LX_PLANNER_RELAYOUT=1 as the feature gate.

## Result

- DXP replay return code: 0
- Backend plan count: 1
- Plan kind: matmul_operand_broadcast
- Communication pattern: all_gather_replicate
- Physical lowering status: lowered_loop_scoped_kernel_neighbor
- Logical transfers: 128

The previous diagnostic flags are intentionally unset in this replay.
