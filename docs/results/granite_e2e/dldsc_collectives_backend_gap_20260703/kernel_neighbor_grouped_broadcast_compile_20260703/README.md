# Grouped KERNEL Operand Broadcast Replay Proof

This artifact records the first clean DXP replay for the Granite attention grouped RHS/KERNEL operand broadcast case on the DLDSC relayout path.

## Result

- Pod: `adnan-clc-spyre-dev-pf`
- Torch branch: `ah/comms-collectives`
- Torch SHA: `bad57f80cf898803c8e292cc56f4fb5c541c70d1`
- Deeptools branch: `ah/comms-collectives`
- Deeptools SHA: `1446330381d84c6086e5131742011727f6883d5b`
- Focused unit gate: `LayoutAllgatherRestickify.*`, 22/22 passed
- DXP replay: `RC=0`
- Plan file: `8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- Plan status: `realized=True`, `physical_lowering_status=lowered_loop_scoped_kernel_neighbor`
- Communication pattern: `all_gather_replicate`
- Logical transfers in plan: `512`

## Why This Mattered

This Granite attention edge was not a simple 1:1 scatter. It is a grouped matmul operand broadcast/all-gather-like case where the producer-owned RHS/KERNEL shards must be visible to multiple consumer matmul corelets. The earlier backend path tried to make the full KERNEL operand resident in LX for the consumer and failed capacity.

The fixed path marks the operand as a loop-scoped KERNEL-neighbor input. DCC then schedules the LX-neighbor transfer inside the matmul transfer loop instead of allocating the full per-consumer operand.

## Key Evidence

`logs/allocation_trace_comparison.txt` shows the important capacity change:

- Before: `8_batchmatmul/Tensor1` attempted `allocationSize=2097152` and failed against LX capacity.
- After: the same operand is staged with `allocationSize=65536`.

`plans/8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json` shows the backend plan is now realized with `lowered_loop_scoped_kernel_neighbor`.

## Reproduction

```bash
ROOT=/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108
B=$ROOT/runs/granite_relayout_s512_failclosed_20260703_173131/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_e0v78nrw
RUN=$ROOT/runs/dxp_replay_kernel_neighbor_realized_patch_20260703_233721
cd $ROOT/build/deeptools
env \
  DEEPTOOLS_PATH=$ROOT/deeptools \
  DXP_LX_FRAC_AVAIL=1 \
  DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1 \
  DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1 \
  DEEPTOOLS_MATMUL_OPERAND_BROADCAST_PLAN_DIR=$RUN/plans \
  DEEPTOOLS_LAYOUT_ALLGATHER_RESTICKIFY_PLAN_DIR=$RUN/plans \
  timeout 180 ./dxp/dxp_standalone -b sentient --bundle -d "$B" \
  >"$RUN/stdout.log" 2>"$RUN/stderr.log"
```

Focused unit test:

```bash
cd $ROOT/build/deeptools
./util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"
```

## Current Scope

This proves the grouped RHS/KERNEL operand broadcast compiles through DXP replay. It is not yet the full AIU runtime/performance proof for the Granite block. Weight restickifies are out of scope for this line of work because they should be handled by weight preload/layout work.

## Next Checks

1. Run the full Granite block on AIU with this Deeptools SHA.
2. Compare SDSC before/after for remaining non-weight HBM spills.
3. Continue classify/fix gather, all-gather, reduce, all-reduce, and form-changing LX restickify cases.
4. Fold the flash-attention `test_flash.py` replay evidence into the same taxonomy once baseline rerun is stable.
