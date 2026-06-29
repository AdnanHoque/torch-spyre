# SDSC Communication Classes: Baseline vs dldsc LX Relayout

This artifact classifies Granite block prefill `batchmatmul` operands by communication class. It deliberately separates working-set/capacity questions from communication-class coverage: WSR can decide how much data is staged at once, while the relayout planner still needs to name the movement pattern.

## Timing Context

| variant | kernel_ms_per_iter | median_wall_ms |
|---|---|---|
| baseline_off | 12.4741 | 19.146 |
| dldsc_full_torch_lx | 10.978 | 17.7715 |

## Class Counts

| comm_class | baseline_off | dldsc_full_torch_lx |
|---|---|---|
| hbm_input_roundtrip_candidate | 5 | 0 |
| hbm_kernel_operand | 5 | 5 |
| hbm_output_spill | 5 | 0 |
| lx_input_same_view | 1 | 1 |
| lx_output | 1 | 6 |
| missing_matmul_operand_collective | 1 | 1 |
| scatter | 0 | 5 |

## Key Rows

| variant | bundle | sdsc | n_shape | compute_split | tensor | role | component | layout | coord_core_map_len | comm_class | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_72z2kyes | sdsc_8 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor0 | INPUT | hbm | mb,in,x | 0 | hbm_input_roundtrip_candidate | consumer input read from HBM rather than LX |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_72z2kyes | sdsc_8 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor2 | OUTPUT | hbm | out,x,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_72z2kyes | sdsc_16 | x_=32,mb_=512,out_=128,in_=512 | mb:32 | Tensor1 | KERNEL | hbm | out,in,x | 0 | missing_matmul_operand_collective | attention value operand remains HBM; planner classifies as all_gather_replicate, not resident relayout |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_cqhp_h68 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | in,mb | 0 | hbm_input_roundtrip_candidate | consumer input read from HBM rather than LX |
| baseline_off | sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_transpose_view_2_cqhp_h68 | sdsc_1 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3_2zutqjow | sdsc_1 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input read from HBM rather than LX |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3_2zutqjow | sdsc_1 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3_2zutqjow | sdsc_5 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input read from HBM rather than LX |
| baseline_off | sdsc_fused_add_linear_mul_silu_split_with_sizes_3_2zutqjow | sdsc_5 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| baseline_off | sdsc_fused_linear_rms_norm_0_qceujboj | sdsc_7 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | hbm | mb,in | 0 | hbm_input_roundtrip_candidate | consumer input read from HBM rather than LX |
| baseline_off | sdsc_fused_linear_rms_norm_0_qceujboj | sdsc_7 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor2 | OUTPUT | hbm | out,mb | 0 | hbm_output_spill | producer output materialized in HBM |
| dldsc_full_torch_lx | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_kksuzzc9 | sdsc_10 | x_=512,mb_=32,out_=512,in_=128 | out:2,x:16 | Tensor0 | INPUT | lx | mb,in,x | 32 | scatter | input already in LX with producer coordinate map; backend inserted resident relayout |
| dldsc_full_torch_lx | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_kksuzzc9 | sdsc_18 | x_=32,mb_=512,out_=128,in_=512 | mb:32 | Tensor1 | KERNEL | hbm | out,in,x | 0 | missing_matmul_operand_collective | attention value operand remains HBM; planner classifies as all_gather_replicate, not resident relayout |
| dldsc_full_torch_lx | sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_kksuzzc9 | sdsc_21 | mb_=512,out_=4096,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | lx | in,mb | 32 | scatter | input already in LX with producer coordinate map; backend inserted resident relayout |
| dldsc_full_torch_lx | sdsc_fused_add_linear_mul_rms_norm_silu_split_with_sizes_2_1baqouvz | sdsc_9 | mb_=512,out_=25600,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | lx | mb,in | 32 | scatter | input already in LX with producer coordinate map; backend inserted resident relayout |
| dldsc_full_torch_lx | sdsc_fused_add_linear_mul_rms_norm_silu_split_with_sizes_2_1baqouvz | sdsc_13 | mb_=512,out_=4096,in_=12800 | mb:8,out:4 | Tensor0 | INPUT | lx | mb,in | 32 | scatter | input already in LX with producer coordinate map; backend inserted resident relayout |
| dldsc_full_torch_lx | sdsc_fused_linear_rms_norm_0_nmecuyoi | sdsc_8 | mb_=512,out_=6144,in_=4096 | mb:4,out:8 | Tensor0 | INPUT | lx | mb,in | 32 | scatter | input already in LX with producer coordinate map; backend inserted resident relayout |

## Readout

- `scatter` rows are the class PR1 now covers: producer data is resident in LX, consumer wants a different resident view, and dldsc coordinates let Deeptools synthesize the LX relayout.
- `missing_matmul_operand_collective` is the remaining Granite attention value path. Treating it as a resident scatter remap asks for a full value operand on every consumer core, which is the 4 MiB/core failure seen in the DXP-only repro.
- WSR should own tiling/staging for capacity. The relayout planner should still classify this as `matmul_operand_broadcast` / `all_gather_replicate` so the backend can realize the right collective instead of falling back to HBM or attempting resident full-materialization.
