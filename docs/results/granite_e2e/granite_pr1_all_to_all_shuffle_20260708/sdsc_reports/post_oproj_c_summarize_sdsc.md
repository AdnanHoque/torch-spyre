
SDSC Operations Summary - Batch Report
Directory: /home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414/rescue_full_torch_lx_backend1/block_prefill/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_4jydowcu
Total sdsc.json files found: 14

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| ReStickifyOpHBM | INPUT (hbm), OUTPUT (hbm) |  |
| batchmatmul | INPUT (lx), INPUT (hbm), OUTPUT (hbm) |  |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| add | INPUT (lx), INPUT (lx), OUTPUT (lx) |  |
| mean | INPUT (lx), OUTPUT (lx) |  |
| rsqrt | INPUT (lx), OUTPUT (lx) |  |
| silu | INPUT (hbm), OUTPUT (lx) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReStickifyOpHBM | 32 | 0_hbm | INPUT | 4096/32, 4096* | 128 x 4096 | 1.00 MB | 0x-1 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | OUTPUT | 4096, 4096*/32 | 4096 x 128 | 1.00 MB | 0x-21 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4096*, 512/4 | 4096 x 128 | 1.00 MB | 0x40000 | in=0 mb=core_id | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_hbm | INPUT | 4096, 4096*/8 | 4096 x 512 | 4.00 MB | 0x-41 | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4096*/8, 512/4 | 512 x 128 | 128.00 KB | 0x-49 | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x-69 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_hbm | INPUT | 1*, 1 | 1 x 1 | 0.00 KB | 0x-89 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| mean | 32 | 0_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_5 |
|  |  | 1_lx | OUTPUT | 1*, 512/32 | 16 x 1 | 0.03 KB | 0x40000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x40000 | out=core_id x=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_hbm | INPUT | 1, 64* | 1 x 64 | 0.12 KB | 0x-8a | out=core_id x=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x40000 | out=core_id x=0 | SEN169_FP16 (2B) |  |
| rsqrt | 32 | 0_lx | INPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x40000 | out=core_id x=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_lx | OUTPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x40000 | out=core_id x=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_8 |
|  |  | 1_lx | INPUT | 1*, 512/32 | 16 x 1 | 0.03 KB | 0x40000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32 | 16 x 4096 | 128.00 KB | 0x40800 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 512/32, 4096* | 16 x 4096 | 128.00 KB | 0x40800 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  | 1_hbm | INPUT | 1, 4096* | 4096 x 1 | 8.00 KB | 0x-8b | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 512/32, 4096* | 16 x 4096 | 128.00 KB | 0x-8c | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | 25600/25, 4096* | 1024 x 4096 | 8.00 MB | 0x-ac | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_10 |
|  |  | 1_hbm | OUTPUT | 4096, 25600*/25 | 4096 x 1024 | 8.00 MB | 0x-c5 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 512/4, 4096* | 128 x 4096 | 1.00 MB | 0x-de | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) | sdsc_11 |
|  |  | 1_hbm | INPUT | 4096, 25600*/8 | 4096 x 3200 | 25.00 MB | 0x-e2 | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 25600*/8, 512/4 | 3200 x 128 | 800.00 KB | 0x-ea | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) |  |
| silu | 32 | 0_hbm | INPUT | 12800*, 512/32 | 12800 x 16 | 400.00 KB | 0x-10a | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_12 |
|  |  | 1_lx | OUTPUT | 12800*, 512/32 | 16 x 12800 | 400.00 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 512/32, 12800* | 16 x 12800 | 400.00 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_13 |
|  |  | 1_hbm | INPUT | 512/32, 12800* | 12800 x 16 | 400.00 KB | 0x-12a | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 512/32, 12800* | 16 x 12800 | 400.00 KB | 0x-14a | mb=core_id out=0 | SEN169_FP16 (2B) |  |
