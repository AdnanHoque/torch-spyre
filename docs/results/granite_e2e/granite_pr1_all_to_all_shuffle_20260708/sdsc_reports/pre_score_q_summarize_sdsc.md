
SDSC Operations Summary - Batch Report
Directory: /home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414/baseline_off/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable__unsafe_view_clone_expand_mul_split_with_sizes_sum_transpose_unsqueeze_view_1__tq1btul
Total sdsc.json files found: 18

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| sumnonstick | INPUT (lx), OUTPUT (lx) |  |
| identity | INPUT (hbm), OUTPUT (hbm) |  |
| ReStickifyOpHBM | INPUT (hbm), OUTPUT (hbm) |  |
| batchmatmul | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |
| add | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| max | INPUT (lx), OUTPUT (lx) |  |
| sub | INPUT (lx), INPUT (lx), OUTPUT (lx) |  |
| exp | INPUT (lx), OUTPUT (lx) |  |
| sum | INPUT (lx), OUTPUT (lx) |  |
| realdiv | INPUT (lx), INPUT (lx), OUTPUT (lx) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mul | 32 | 0_hbm | INPUT | 2, 2, 512/32, 64*, 1 | 2 x 2 x 16 x 64 x 1 | 8.00 KB | 0x-1 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 2, 1, 512/32, 64*, 32 | 2 x 32 x 16 x 64 x 1 | 128.00 KB | 0x-21 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 2, 2, 512/32, 64*, 32 | 16 x 2 x 2 x 32 x 64 | 256.00 KB | 0x0 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) |  |
| sumnonstick | 32 | 0_lx | INPUT | 512/32, 2, 2, 32, 64* | 16 x 2 x 2 x 32 x 64 | 256.00 KB | 0x0 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_lx | OUTPUT | 512/32, 1, 2, 32, 64* | 16 x 2 x 32 x 64 x 1 | 128.00 KB | 0x40000 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 512/32, 128*, 32 | 16 x 128 x 32 | 128.00 KB | 0x40000 | mb=core_id x=0 out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_hbm | INPUT | 1, 1*, 1 | 1 x 1 x 1 | 0.00 KB | 0x-41 | mb=core_id x=0 out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 512/32, 128*, 32 | 32 x 128 x 16 | 128.00 KB | 0x-42 | mb=core_id x=0 out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 2, 2, 512/32, 64*, 1 | 2 x 2 x 16 x 64 x 1 | 8.00 KB | 0x-62 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_hbm | INPUT | 2, 1, 512/32, 64*, 8 | 2 x 8 x 16 x 64 x 1 | 32.00 KB | 0x-82 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 2, 2, 512/32, 64*, 8 | 16 x 2 x 2 x 8 x 64 | 64.00 KB | 0x0 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) |  |
| sumnonstick | 32 | 0_lx | INPUT | 512/32, 2, 2, 8, 64* | 16 x 2 x 2 x 8 x 64 | 64.00 KB | 0x0 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_hbm | OUTPUT | 512/32, 1, 2, 8, 64* | 16 x 2 x 8 x 64 x 1 | 32.00 KB | 0x-a2 | mb=core_id x=0 y=0 i=0 out=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_hbm | INPUT | 512/32, 128*, 8, 1 | 16 x 128 x 8 x 1 | 32.00 KB | 0x-c2 | mb=0 x=0 y=core_id out=0 | SEN169_FP16 (2B) | sdsc_5 |
|  |  | 1_hbm | OUTPUT | 512/32, 128*, 8, 4 | 128 x 16 x 4 x 8 | 128.00 KB | 0x-e2 | mb=0 x=0 y=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 512/32, 32 | 128 x 16 x 32 | 128.00 KB | 0x-102 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_hbm | INPUT | 1*, 1, 1 | 1 x 1 x 1 | 0.00 KB | 0x-122 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 128*, 512/32, 32 | 32 x 128 x 16 | 128.00 KB | 0x-123 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_hbm | INPUT | 32/32, 128*, 512 | 1 x 128 x 512 | 128.00 KB | 0x-143 | mb=core_id x=0 out=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_hbm | OUTPUT | 32/32, 512*, 128 | 1 x 512 x 128 | 128.00 KB | 0x-163 | mb=core_id x=0 out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 32, 128*, 512/16 | 32 x 128 x 32 | 256.00 KB | 0x-183 | {x=0:15} mb=0 {out=0:1} in=0 | SEN169_FP16 (2B) | sdsc_8 |
|  |  | 1_hbm | INPUT | 32, 512*/2, 128 | 32 x 256 x 128 | 2.00 MB | 0x-193 | {x=0:15} mb=0 {out=0:1} in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 512*/2, 512/16, 32 | 256 x 32 x 32 | 512.00 KB | 0x-195 | {x=0:15} mb=0 {out=0:1} in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 512*, 512/32, 32 | 512 x 16 x 32 | 512.00 KB | 0x-1b5 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  | 1_hbm | INPUT | 512*, 512/32, 1 | 512 x 16 x 1 | 16.00 KB | 0x-1d5 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 512*, 512/32, 32 | 32 x 512 x 16 | 512.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_10 |
|  |  | 1_lx | OUTPUT | 32, 1*, 512/32 | 32 x 16 x 1 | 1.00 KB | 0x80000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_11 |
|  |  | 1_lx | INPUT | 32, 1*, 512/32 | 32 x 16 x 1 | 1.00 KB | 0x80000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_12 |
|  |  | 1_lx | OUTPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_13 |
|  |  | 1_lx | OUTPUT | 32, 1*, 512/32 | 32 x 16 x 1 | 1.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| realdiv | 32 | 0_lx | INPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_14 |
|  |  | 1_lx | INPUT | 32, 1*, 512/32 | 32 x 16 x 1 | 1.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_hbm | INPUT | 128*, 8, 512/32, 1 | 128 x 8 x 16 x 1 | 32.00 KB | 0x-1f5 | mb=0 x=0 y=core_id out=0 | SEN169_FP16 (2B) | sdsc_15 |
|  |  | 1_hbm | OUTPUT | 128*, 8, 512/32, 4 | 128 x 16 x 4 x 8 | 128.00 KB | 0x-215 | mb=0 x=0 y=core_id out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 32, 512*, 512/32 | 32 x 512 x 16 | 512.00 KB | 0x90000 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) | sdsc_16 |
|  |  | 1_hbm | INPUT | 128*, 512, 32 | 128 x 512 x 32 | 4.00 MB | 0x-235 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 512/32, 32 | 128 x 16 x 32 | 128.00 KB | 0x0 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_lx | INPUT | 128*, 512/32, 32 | 128 x 16 x 32 | 128.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_17 |
|  |  | 1_hbm | OUTPUT | 128*, 512/32, 32 | 128 x 32 x 16 | 128.00 KB | 0x-236 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
