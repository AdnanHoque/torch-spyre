# Golden Granite-8B matmul ops

Distinct matmuls in the Granite-3.3-8B e2e graph (bs=1, prefill sl=512,
padded decode M=64), with shape, emitted work-division split, and a
representative fused kernel. For microbenchmark lift-and-shift.

- Shape **B x M x N x K** (elements); matmul `[B,M,K] @ [B,K,N]`.
- `split` = emitted `numWkSlicesPerDim_` (x=batch, mb=M, out=N, in=K).
- Prefill splits = upstream-main planner; decode = retuned branch.
- `count` = matmuls of this shape per layer block.

## Projection + MLP matmuls (B=1)

| phase | M | N | K | role | split | count | example kernel |
|---|--:|--:|--:|---|---|--:|---|
| prefill | 512 | 12800 | 4096 | MLP gate+up (w1,w3) | `mb4,out8,in1` | 2 | `sdsc_fused_add_linear_mul_rms_norm_silu_3` |
| prefill | 512 | 4096 | 12800 | MLP down (w2) | `mb4,out8,in1` | 1 | `sdsc_fused_add_linear_mul_rms_norm_silu_3` |
| prefill | 512 | 4096 | 4096 | Q/O proj | `mb4,out8,in1` | 2 | `sdsc_fused_add_linear_mul_rms_norm_silu_view_2` |
| prefill | 512 | 1024 | 4096 | K,V proj (GQA) | `mb4,out8,in1` | 2 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__to_copy__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1` |
| decode | 64 | 12800 | 4096 | MLP gate+up (w1,w3) | `mb4,out8,in1` | 16 | `sdsc_fused_add_linear_mul_rms_norm_5` |
| decode | 64 | 4096 | 4096 | Q/O proj | `mb4,out8,in1` | 17 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_transpose_unsqueeze_view_4` |
| decode | 64 | 4096 | 12800 | MLP down (w2) | `mb4,out8,in1` | 8 | `sdsc_fused_add_linear_mul_rms_norm_5` |
| decode | 64 | 1024 | 4096 | K,V proj (GQA) | `mb4,out8,in1` | 16 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_mul_sum_transpose_unsqueeze_view_1` |

## Attention batched bmms (heads folded into batch)

| phase | B | M | N | K | kind | split | count | example kernel |
|---|--:|--:|--:|--:|---|---|--:|---|
| prefill | 512 | 32 | 512 | 128 | QK^T scores | `x4,mb1,out8,in1` | 1 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__to_copy__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1` |
| prefill | 32 | 512 | 128 | 512 | attn @ V | `x1,mb32,out1,in1` | 1 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__to_copy__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1` |
| decode | 64 | 32 | 576 | 128 | QK^T scores | `x32,mb1,out1,in1` | 8 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2` |
| decode | 32 | 64 | 128 | 576 | attn @ V | `x1,mb16,out2,in1` | 8 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_transpose_unsqueeze_view_4` |
