# Granite Block Layer Probe

- case: `prefill`
- returncode: `0`
- fused_weights: `True`
- compile_block: `True`
- input_shape: `[1, 512, 4096]`
- position_ids_shape: `[1, 512]`
- mask_shape: `[1, 512, 512]`
- past_key_value_shape: `None`
- generated SDSC exact normalized match: `False`
- generated SDSC overlap: `0/20`

## Timing

- profile_enabled: `True`
- median_ms: `28.084874153137207`
- all_ms: `[28.059, 27.521, 28.38, 28.283, 28.013, 28.072, 27.694, 28.026, 28.354, 28.303, 27.686, 28.855, 28.477, 28.098, 27.693, 27.963, 28.113, 28.562, 28.053, 28.373]`
- trace_summary_path: `/home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414/baseline_off/block_prefill/trace_summary.json`
- kernel_ms_per_iter: `14.700179199999996`
- memory_ms_per_iter: `0.3081238500000001`

## Generated SDSCs

| normalized kernel | split samples |
|---|---|
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_` | `{'mb': 32, 'x': 1, 'y': 1, 'i': 1, 'out': 1} ; {'mb': 32, 'x': 1, 'y': 1, 'i': 1, 'out': 1}` |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2` | `{'mb': 32, 'out': 1} ; {'mb': 4, 'out': 8, 'in': 1}` |
| `sdsc_fused_add_linear_mul_silu_split_with_sizes_3` | `{'mb': 25, 'out': 1} ; {'mb': 4, 'out': 8, 'in': 1}` |
| `sdsc_fused_linear_rms_norm_0` | `{'mb': 32, 'out': 1} ; {'mb': 32, 'out': 1}` |

## Missing vs Antoni Trace

- `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_unsqueeze_view_3`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_transpose_unsqueeze_view_4`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_2`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_mul_sum_transpose_unsqueeze_view_1`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_unsqueeze_3`
- `sdsc_fused_add_linear_mul_4`
- `sdsc_fused_add_linear_mul_rms_norm_silu_5`
- `sdsc_fused_add_linear_mul_silu_6`
- `sdsc_fused_add_mean_mul_rsqrt_0`
- `sdsc_fused_add_mul_5`
- `sdsc_fused_bmm_transpose_unsqueeze_0`
- `sdsc_fused_div_0`
- `sdsc_fused_linear_mul_rms_norm_silu_3`
- `sdsc_fused_linear_mul_rms_norm_silu_4`
- `sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0`
- `sdsc_fused_linear_overwrite_slice_transpose_view_1`
- `sdsc_fused_mul_0`

## Extra vs Antoni Trace

- `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_`
- `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2`
- `sdsc_fused_add_linear_mul_silu_split_with_sizes_3`
- `sdsc_fused_linear_rms_norm_0`
