# Antoni Trace Golden Granite Ops

Ground-truth launch-kernel inventory from Antoni's two Granite e2e traces:

- `/Users/adnan/Downloads/aviros-spyre-test_1673899.1780607390507520214.pt.trace.json`
- `/Users/adnan/Downloads/aviros-spyre-test_52053.1781021126249019455.pt.trace.json`

Both traces have the same normalized `launch_kernel:sdsc_*` set: **20/20 exact kernel names match**. Counts also match across both traces. Durations differ slightly, as expected.

Important scope note: the Kineto traces prove exact launch names, launch counts, ordering, and launch-event durations. They do **not** carry full B/M/N/K shape metadata or `numWkSlicesPerDim_`; any shape/split/role fields below are inferred from Granite execution order and the prior Granite shape inventory. For shape/split ground truth, pair this with the corresponding `inductor-spyre` SDSC cache.

## Role-Oriented Golden Sequence

### Prefill Block Sequence

This sequence repeats once per Granite layer during prompt/prefill. The roles are inferred from launch order in the layer body.

| order | phase | highlighted role | expected shape family | exact Antoni trace kernel |
|---:|---|---|---|---|
| 1 | prefill | **attention input norm + Q/K/V projections** | `[1,512,4096] @ [4096,{4096,1024}]` projection family | `sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0` |
| 2 | prefill | **prefill SDPA / attention core** | QK scores + attention @ V family | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1` |
| 3 | prefill | **attention output + residual / next norm setup** | O projection / post-attention fused family | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2` |
| 4 | prefill | **MLP gate/up projection + SiLU** | `[1,512,4096] @ [4096,12800]` gate/up family | `sdsc_fused_linear_mul_rms_norm_silu_3` |
| 5 | prefill | **MLP down projection + residual** | `[1,512,12800] @ [12800,4096]` down family | `sdsc_fused_add_linear_mul_4` |

### Decode Block Sequence: Cat-Cache Variant

This sequence appears in decode regions that read/concatenate with existing KV cache.

| order | phase | highlighted role | expected shape family | exact Antoni trace kernel |
|---:|---|---|---|---|
| 1 | decode | **attention input norm + Q/K/V projections** | padded decode projection family, M=64 | `sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0` |
| 2 | decode | **decode K/V projection or QK setup** | GQA K/V projection / attention setup | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_mul_sum_transpose_unsqueeze_view_1` |
| 3 | decode | **decode QK^T scores over cached K** | batched attention score BMM | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2` |
| 4 | decode | **decode attention intermediate / layout prep** | attention layout helper | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_unsqueeze_3` |
| 5 | decode | **decode attention @ V + output projection** | attention context / O-projection family | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_transpose_unsqueeze_view_4` |
| 6 | decode | **MLP gate/up projection + SiLU** | padded decode MLP gate/up family, M=64 | `sdsc_fused_add_linear_mul_rms_norm_silu_5` |
| 7 | decode | **MLP down projection + residual** | padded decode MLP down family, M=64 | `sdsc_fused_add_linear_mul_silu_6` |

### Decode Block Sequence: Overwrite-Slice Variant

This second decode pattern appears in regions that update/write KV cache slices.

| order | phase | highlighted role | expected shape family | exact Antoni trace kernel |
|---:|---|---|---|---|
| 1 | decode | **attention input norm + Q/K/V projections** | padded decode projection family, M=64 | `sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0` |
| 2 | decode | **KV cache overwrite / transpose layout** | KV-cache update helper | `sdsc_fused_linear_overwrite_slice_transpose_view_1` |
| 3 | decode | **decode attention core / projection setup** | attention score/context setup | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_2` |
| 4 | decode | **attention output + residual / MLP norm setup** | post-attention fused family | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_unsqueeze_view_3` |
| 5 | decode | **MLP gate/up projection + SiLU** | padded decode MLP gate/up family, M=64 | `sdsc_fused_linear_mul_rms_norm_silu_4` |
| 6 | decode | **MLP residual/elementwise tail** | post-MLP elementwise residual tail | `sdsc_fused_add_mul_5` |

### Small Setup / Helper Kernels

These are exact trace kernels but are not the main projection/MLP/attention matmul rows.

| highlighted role | exact Antoni trace kernel |
|---|---|
| **input scaling / embedding multiplier helper** | `sdsc_fused_mul_0` |
| **RMSNorm helper** | `sdsc_fused_add_mean_mul_rsqrt_0` |
| **small attention/cache BMM helper** | `sdsc_fused_bmm_transpose_unsqueeze_0` |
| **logit or scale division helper** | `sdsc_fused_div_0` |

## Exact Launch-Kernel Inventory

Counts are total `launch_kernel:sdsc_*` occurrences across each whole trace. Durations are summed launch-event `dur` values from Kineto, converted to ms. These are useful for exact matching, not for final device timing claims.

| trace count | dur ms trace 1673899 | dur ms trace 52053 | exact Antoni trace kernel | role group |
|---:|---:|---:|---|---|
| 320 | 10.566 | 10.155 | `sdsc_fused_linear_mul_rms_norm_sum_unsqueeze_view_0` | attention projection / norm, shared prefill+decode family |
| 160 | 3.094 | 2.841 | `sdsc_fused_linear_overwrite_slice_transpose_view_1` | decode KV-cache overwrite/update helper |
| 160 | 2.973 | 2.690 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_2` | decode overwrite-slice attention core/setup |
| 160 | 2.852 | 2.615 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_unsqueeze_view_3` | decode overwrite-slice attention output / MLP norm setup |
| 160 | 2.811 | 2.593 | `sdsc_fused_linear_mul_rms_norm_silu_4` | decode MLP gate/up family |
| 160 | 3.354 | 2.667 | `sdsc_fused_add_mul_5` | decode MLP residual/elementwise tail |
| 80 | 1.701 | 1.525 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_linear_mul_sum_transpose_unsqueeze_view_1` | prefill SDPA / attention core |
| 80 | 1.581 | 1.416 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_add_clone_expand_linear_mul_rms_norm_transpose_unsqueeze_view_2` | prefill attention output / post-attn norm setup |
| 80 | 1.552 | 1.402 | `sdsc_fused_linear_mul_rms_norm_silu_3` | prefill MLP gate/up family |
| 80 | 1.428 | 1.320 | `sdsc_fused_add_linear_mul_4` | prefill MLP down/residual family |
| 80 | 1.547 | 1.443 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_mul_sum_transpose_unsqueeze_view_1` | decode cat-cache K/V or attention setup |
| 80 | 1.541 | 1.382 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_transpose_unsqueeze_view_2` | decode cat-cache QK score BMM |
| 80 | 1.401 | 1.285 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable_linear_unsqueeze_3` | decode cat-cache attention layout helper |
| 80 | 1.486 | 1.363 | `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_cat_clone_expand_linear_transpose_unsqueeze_view_4` | decode cat-cache attention @ V / O-projection family |
| 80 | 1.542 | 1.348 | `sdsc_fused_add_linear_mul_rms_norm_silu_5` | decode MLP gate/up family |
| 80 | 1.453 | 1.302 | `sdsc_fused_add_linear_mul_silu_6` | decode MLP down/residual family |
| 8 | 0.642 | 0.465 | `sdsc_fused_mul_0` | small setup/helper |
| 8 | 0.273 | 0.266 | `sdsc_fused_add_mean_mul_rsqrt_0` | small RMSNorm helper |
| 8 | 0.241 | 0.241 | `sdsc_fused_bmm_transpose_unsqueeze_0` | small attention/cache BMM helper |
| 8 | 0.243 | 0.243 | `sdsc_fused_div_0` | small division/helper |

## Practical Use

Use the exact kernel names above as the Antoni-trace golden for checking whether a new Granite microbenchmark is representative. A benchmark is trace-representative at the launch-name level if its emitted `launch_kernel:sdsc_*` names are a subset of this inventory, and it is fully e2e-representative only if it also preserves the prefill/decode sequence families above.

Do not treat this file as final split ground truth. For split validation, collect and diff the matching `sdsc_*.json` files from the same run.
