# Guarded Granite Prefill Spill Inventory - 2026-06-30

Source run: `/home/adnan/codex-isolated/comms_collectives_20260629/runs/granite_prefill_collectives_guarded_20260630_114425`.

This artifact is generated from the archived SuperDSC cache and records what the current `ah/comms-collectives` branch handles versus what remains as a communication/lowering gap. Weight restickifies are explicitly marked out of scope.

## Classification Counts

| kind | realized | count |
|---|---:|---:|
| `layout_restickify_activation` | `False` | 1 |
| `layout_restickify_weight` | `False` | 4 |
| `matmul_operand_broadcast` | `False` | 1 |
| `scatter` | `True` | 14 |

## HBM Restickify Rows

| file | op | locations | scope | reason |
|---|---|---|---|---|
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_9.json` | `9_ReStickifyOpHBM` | `hbm:1 lx:1` | remaining-runtime-gap | computed attention activation layout restickify; downstream `sdsc_10` classifies it as `layout_restickify_activation` |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_0.json` | `0_ReStickifyOpHBM` | `hbm:2` | out-of-scope | attention output projection weight; offline prelayout owns this |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_10.json` | `10_ReStickifyOpHBM` | `hbm:2` | out-of-scope | fused FFN gate/up projection weight; offline prelayout owns this |
| `sdsc_fused_add_linear_mul_3_300wv0lo/sdsc_0.json` | `0_ReStickifyOpHBM` | `hbm:2` | out-of-scope | FFN down-projection weight; offline prelayout owns this |
| `sdsc_fused_linear_rms_norm_0_9x5xiiym/sdsc_7.json` | `7_ReStickifyOpHBM` | `hbm:2` | out-of-scope | attention QKV projection weight; offline prelayout owns this |

## Relayout/Spill Rows

| file | op | locations | relayout kinds | scope | reason |
|---|---|---|---|---|---|
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_1.json` | `1_mul` | `hbm:1 lx:2` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_10.json` | `10_batchmatmul` | `hbm:2 lx:1` | `layout_restickify_activation:1 scatter:1` | remaining-runtime-gap | computed activation layout transform plus operand movement |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_11.json` | `11_add` | `hbm:2 lx:1` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_17.json` | `17_identity` | `hbm:2` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_18.json` | `18_batchmatmul` | `hbm:1 lx:2` | `matmul_operand_broadcast:1` | remaining-runtime-gap | attention-sized all-gather skipped by resident-size guard |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_4.json` | `4_mul` | `hbm:1 lx:2` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1_7yt9sfzx/sdsc_9.json` | `9_ReStickifyOpHBM` | `hbm:1 lx:1` | `scatter:1` | remaining-runtime-gap | input scatter is realized, but the op still writes the computed activation through HBM for a layout restickify |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_0.json` | `0_ReStickifyOpHBM` | `hbm:2` | `` | out-of-scope | weight/parameter restickify; offline prelayout owns this |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_1.json` | `1_batchmatmul` | `hbm:2 lx:1` | `layout_restickify_weight:1 scatter:1` | out-of-scope | weight/parameter restickify; offline prelayout owns this |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_10.json` | `10_ReStickifyOpHBM` | `hbm:2` | `` | out-of-scope | weight/parameter restickify; offline prelayout owns this |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_11.json` | `11_batchmatmul` | `hbm:3` | `layout_restickify_weight:1 scatter:1` | out-of-scope | weight/parameter restickify; offline prelayout owns this |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_12.json` | `12_silu` | `hbm:1 lx:1` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_13.json` | `13_mul` | `hbm:2 lx:1` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2__5fr7fpm/sdsc_2.json` | `2_mul` | `hbm:2 lx:1` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused_add_linear_mul_3_300wv0lo/sdsc_0.json` | `0_ReStickifyOpHBM` | `hbm:2` | `` | out-of-scope | weight/parameter restickify; offline prelayout owns this |
| `sdsc_fused_add_linear_mul_3_300wv0lo/sdsc_1.json` | `1_batchmatmul` | `hbm:3` | `layout_restickify_weight:1 scatter:1` | out-of-scope | weight/parameter restickify; offline prelayout owns this |
| `sdsc_fused_add_linear_mul_3_300wv0lo/sdsc_2.json` | `2_mul` | `hbm:2 lx:1` | `scatter:1` | handled | resident scatter/permutation realized through dldsc relayout |
| `sdsc_fused_linear_rms_norm_0_9x5xiiym/sdsc_7.json` | `7_ReStickifyOpHBM` | `hbm:2` | `` | out-of-scope | weight/parameter restickify; offline prelayout owns this |
| `sdsc_fused_linear_rms_norm_0_9x5xiiym/sdsc_8.json` | `8_batchmatmul` | `hbm:3` | `layout_restickify_weight:1 scatter:1` | out-of-scope | weight/parameter restickify; offline prelayout owns this |

## Remaining Runtime Gaps

- `layout_restickify_activation`: one computed attention activation still uses an HBM `ReStickifyOpHBM`. This is a layout transform, not a pure shard scatter, and needs an LX layout-restickify contract before it can be removed.
- `matmul_operand_broadcast`: one attention value-side operand is an all-gather/replicate class. The tensor is 4,194,304 bytes, so full resident replication is unsafe; it must be tiled or loop-scoped around the matmul operand fetch.
- Weight restickifies: four rows are graph-input/parameter layout preparation and are intentionally left to offline weight prelayout/preload work.

## CSV

Companion CSV: [`comms_collectives_guarded_spill_inventory_20260630.csv`](./comms_collectives_guarded_spill_inventory_20260630.csv).
