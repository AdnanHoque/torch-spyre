# Granite Block Kineto Breakdown

Run root: /home/adnan/dt-inductor/profiler_runs/granite_block_cost_model_isolated_20260623_224926

Comparison uses current main plus the same local Granite compile prerequisite as baseline, and that same baseline plus the cost-model work_division.py as candidate. Kernel event names in this profiler build are path labels, so per-kernel rows map Kineto launch-order buckets back to the generated SDSC inventory.

## Summary

| regime | baseline wall median ms | candidate wall median ms | baseline kernel ms/iter | candidate kernel ms/iter | kernel speedup |
|---|---:|---:|---:|---:|---:|
| prefill | 27.867 | 28.467 | 16.149 | 16.574 | 0.974x |
| decode_expand | 18.868 | 14.880 | 14.765 | 10.621 | 1.390x |

## prefill Kernel Buckets

| launch | baseline kernel | baseline split hints | baseline mean ms | candidate kernel | candidate split hints | candidate mean ms | delta ms |
|---:|---|---|---:|---|---|---:|---:|
| 1 | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_... | 16_batchmatmul:{'in': 1, 'mb': 32, 'out': 1, 'x': 1}; 7_ReStickifyOpHBM:{'mb': 32, 'out':... | 1.901 | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_... | 16_batchmatmul:{'in': 1, 'mb': 32, 'out': 1, 'x': 1}; 7_ReStickifyOpHBM:{'mb': 32, 'out':... | 1.895 | -0.006 |
| 2 | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mu... | 0_ReStickifyOpHBM:{'mb': 32, 'out': 1}; 1_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 1.773 | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mu... | 0_ReStickifyOpHBM:{'mb': 32, 'out': 1}; 1_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 1.990 | +0.217 |
| 3 | sdsc_fused_add_linear_mul_silu_split_with_sizes_3 | 0_ReStickifyOpHBM:{'mb': 25, 'out': 1}; 1_batchmatmul:{'in': 1, 'mb': 4, 'out': 8}; 7_ReS... | 1.450 | sdsc_fused_add_linear_mul_silu_split_with_sizes_3 | 0_ReStickifyOpHBM:{'mb': 25, 'out': 1}; 1_batchmatmul:{'in': 1, 'mb': 4, 'out': 8}; 7_ReS... | 1.446 | -0.004 |
| 4 | sdsc_fused_linear_rms_norm_0 | 6_ReStickifyOpHBM:{'mb': 32, 'out': 1}; 7_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 11.025 | sdsc_fused_linear_rms_norm_0 | 6_ReStickifyOpHBM:{'mb': 32, 'out': 1}; 7_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 11.243 | +0.218 |

## decode_expand Kernel Buckets

| launch | baseline kernel | baseline split hints | baseline mean ms | candidate kernel | candidate split hints | candidate mean ms | delta ms |
|---:|---|---|---:|---|---|---:|---:|
| 1 | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_... | 3_batchmatmul:{'in': 1, 'mb': 32, 'out': 1, 'x': 1}; 5_ReStickifyOpHBM:{'mb': 32, 'out': ... | 1.336 | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_... | 3_batchmatmul:{'in': 1, 'mb': 4, 'out': 1, 'x': 8}; 5_ReStickifyOpHBM:{'mb': 32, 'out': 1... | 1.283 | -0.053 |
| 2 | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_... | 4_ReStickifyOpHBM:{'mb': 32, 'out': 1, 'x': 1}; 5_batchmatmul:{'in': 1, 'mb': 1, 'out': 1... | 0.006 | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_... | 4_ReStickifyOpHBM:{'mb': 32, 'out': 1, 'x': 1}; 5_batchmatmul:{'in': 2, 'mb': 4, 'out': 1... | 0.006 | -0.000 |
| 3 | sdsc_fused__scaled_dot_product_fused_attention_overrideable_mul_split_wit... |  | 2.655 | sdsc_fused__scaled_dot_product_fused_attention_overrideable_mul_split_wit... |  | 0.570 | -2.085 |
| 4 | sdsc_fused__scaled_dot_product_fused_attention_overrideable_unsqueeze_3 |  | 0.049 | sdsc_fused__scaled_dot_product_fused_attention_overrideable_unsqueeze_3 |  | 0.049 | +0.000 |
| 5 | sdsc_fused_add_linear_mul_rms_norm_silu_split_with_sizes_5 | 8_ReStickifyOpHBM:{'mb': 25, 'out': 1}; 9_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 1.488 | sdsc_fused_add_linear_mul_rms_norm_silu_split_with_sizes_5 | 8_ReStickifyOpHBM:{'mb': 25, 'out': 1}; 9_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 0.979 | -0.508 |
| 6 | sdsc_fused_add_linear_mul_silu_split_with_sizes_6 | 3_ReStickifyOpHBM:{'mb': 1, 'out': 25}; 4_batchmatmul:{'in': 1, 'mb': 32, 'out': 1} | 5.197 | sdsc_fused_add_linear_mul_silu_split_with_sizes_6 | 3_ReStickifyOpHBM:{'mb': 1, 'out': 25}; 4_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 5.194 | -0.003 |
| 7 | sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0 | 6_ReStickifyOpHBM:{'mb': 32, 'out': 1}; 7_batchmatmul:{'in': 1, 'mb': 1, 'out': 32} | 4.035 | sdsc_fused_linear_mul_rms_norm_split_with_sizes_sum_unsqueeze_view_0 | 6_ReStickifyOpHBM:{'mb': 32, 'out': 1}; 7_batchmatmul:{'in': 1, 'mb': 4, 'out': 8} | 2.540 | -1.495 |

