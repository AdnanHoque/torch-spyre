# Granite DLDSC Collectives Spill Classification - 2026-07-04

This note classifies the currently archived enabled Granite S512 prefill run. It separates weight-shaped HBM restickifies from activation HBM handoffs that still need compiler work.

## Source

- Run: /home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/runs/granite_relayout_s512_both_flags_prefer_matmul_fixed_env_20260704_022432/block_prefill/cache/inductor-spyre
- Torch SHA: 99192b8a140f2e48ba9284161a460f7cf5470e0f
- Deeptools SHA: fa36c57166ed4541216c55cf97527f96819d7d5e

## Explicit ReStickify Rows

| Bundle | file | op | N | split | classification |
|---|---|---|---|---|---|
| sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_u7ziutnz | sdsc_9.json | 9_ReStickifyOpLx | mb_=32, x_=512, out_=128 | {'mb': 32, 'x': 1, 'out': 1} | activation on-chip relayout |
| sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_awctlogs | sdsc_0.json | 0_ReStickifyOpHBM | mb_=4096, out_=4096 | {'mb': 32, 'out': 1} | weight/layout-preload |
| sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_awctlogs | sdsc_10.json | 10_ReStickifyOpHBM | mb_=25600, out_=4096 | {'mb': 25, 'out': 1} | weight/layout-preload |
| sdsc_fused_add_linear_mul_3_e4nnnbwu | sdsc_0.json | 0_ReStickifyOpHBM | mb_=4096, out_=12800 | {'mb': 1, 'out': 25} | weight/layout-preload |
| sdsc_fused_linear_rms_norm_0_a32x0r6r | sdsc_7.json | 7_ReStickifyOpHBM | mb_=6144, out_=4096 | {'mb': 32, 'out': 1} | weight/layout-preload |

Interpretation: the remaining explicit ReStickifyOpHBM rows are weight-shaped: QKV, attention output projection, fused gate/up projection, and down projection. Those are out of scope for this pass because weight preload/prelayout should handle them offline.

## Planned DLDSC Relayout Classes

| Bundle | file | op | kind | class | pattern | fanout | fanin | transfers | producer -> consumer |
|---|---|---|---|---|---|---:|---:|---:|---|
| sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_u7ziutnz | sdsc_10.json | 10_batchmatmul | matmul_operand_broadcast | all_gather | all_gather_replicate | 32 | 32 | 1024 | restickify -> batchmatmul |
| sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_u7ziutnz | sdsc_18.json | 18_batchmatmul | matmul_operand_broadcast | all_gather | all_gather_replicate | 32 | 32 | 1024 | clone -> batchmatmul |
| sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_awctlogs | sdsc_1.json | 1_batchmatmul | matmul_operand_broadcast | all_gather | all_gather_replicate | 4 | 4 | 128 | restickify -> batchmatmul |
| sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_awctlogs | sdsc_11.json | 11_batchmatmul | matmul_operand_broadcast | all_gather | all_gather_replicate | 8 | 4 | 128 | restickify -> batchmatmul |
| sdsc_fused_add_linear_mul_3_e4nnnbwu | sdsc_1.json | 1_batchmatmul | matmul_operand_broadcast | all_gather | all_gather_replicate | 32 | 25 | 800 | restickify -> batchmatmul |
| sdsc_fused_linear_rms_norm_0_a32x0r6r | sdsc_8.json | 8_batchmatmul | matmul_operand_broadcast | all_gather | all_gather_replicate | 4 | 4 | 128 | restickify -> batchmatmul |

## Remaining Non-Weight Activation Handoffs

The fused FFN/SwiGLU region still has HBM-backed activation operands even though the explicit HBM restickifies are weight-shaped. In generated code, the gate/up projection is 2x wide, split by split_with_sizes, consumed by silu and mul, and the resulting activation is passed through the shared pool into the down projection kernel. This is not represented as a simple direct producer-buffer to consumer-buffer LX edge, so the current single-writer PerCoreView relayout planner does not mark it as a relayout source.

That means the next FFN class is split/view-aware activation handoff across the pool-backed SuperDSC boundary. It needs either:

1. frontend metadata that ties split/view aliases back to the producer activation layout and declares the desired consumer layout, or
2. a larger LX pool/region handoff contract that lets Deeptools synthesize the movement between the pointwise result and the down-projection input.

This is distinct from the PR1 scatter case and from the current matmul RHS all-gather/broadcast path.
