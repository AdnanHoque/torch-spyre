
SDSC Operations Summary - Batch Report
Directory: /home/adnan/spyre-envs/main-e3a79c56-aggressive/runs/sdpa_h4_upstream_main_aggressive_20260709_213048/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_eg5sudhj
Total sdsc.json files found: 22

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| identity | INPUT (hbm), OUTPUT (lx) |  |
| max | INPUT (lx), OUTPUT (lx) |  |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| ReStickifyOpHBM | INPUT (lx), OUTPUT (hbm) |  |
| batchmatmul | INPUT (lx), INPUT (hbm), OUTPUT (lx) |  |
| maximum | INPUT (lx), INPUT (lx), OUTPUT (lx) |  |
| sub | INPUT (lx), INPUT (lx), OUTPUT (lx) |  |
| exp | INPUT (lx), OUTPUT (lx) |  |
| add | INPUT (lx), INPUT (lx), OUTPUT (hbm) |  |
| sum | INPUT (lx), OUTPUT (lx) |  |
| realdiv | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 32 | 0_hbm | INPUT | 1*, 1, 1 | 1 x 1 x 1 | 0.00 KB | 0x-1 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_lx | OUTPUT | 128*, 512/32, 4 | 128 x 16 x 4 | 16.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_hbm | INPUT | 1, 1, 64* | 1 x 1 x 64 | 0.12 KB | 0x-2 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_lx | OUTPUT | 512/32, 4, 64* | 16 x 4 x 64 | 8.00 KB | 0x4000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 512/32, 4, 64* | 16 x 4 x 64 | 8.00 KB | 0x4000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_lx | OUTPUT | 512/32, 4, 64* | 4 x 16 x 64 | 8.00 KB | 0x6000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_hbm | INPUT | 1, 1, 64* | 1 x 1 x 64 | 0.12 KB | 0x-3 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_lx | OUTPUT | 512/32, 4, 64* | 16 x 4 x 64 | 8.00 KB | 0x8000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 512/32, 4, 64* | 16 x 4 x 64 | 8.00 KB | 0x8000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_lx | OUTPUT | 512/32, 4, 64* | 4 x 16 x 64 | 8.00 KB | 0xa000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 512/32, 4 | 128 x 16 x 4 | 16.00 KB | 0x-4 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_5 |
|  |  | 1_hbm | INPUT | 1*, 1, 1 | 1 x 1 x 1 | 0.00 KB | 0x-24 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 512/32, 4 | 128 x 16 x 4 | 16.00 KB | 0xc000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/32, 4 | 128 x 128 x 4 | 128.00 KB | 0x-25 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_hbm | INPUT | 1*, 1, 1 | 1 x 1 x 1 | 0.00 KB | 0x-45 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 4096/32, 4 | 128 x 128 x 4 | 128.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/32, 4 | 128 x 128 x 4 | 128.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_hbm | OUTPUT | 4096*/32, 128, 4 | 128 x 128 x 4 | 128.00 KB | 0x-46 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 128*, 512/32, 4 | 128 x 16 x 4 | 16.00 KB | 0xc000 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) | sdsc_8 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4 | 4096 x 128 x 4 | 4.00 MB | 0x-66 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  | 1_lx | OUTPUT | 1*, 512/32, 4 | 4 x 16 x 1 | 0.12 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x6000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) | sdsc_10 |
|  |  | 1_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x90000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x90000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x6000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) | sdsc_11 |
|  |  | 1_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x90000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x6000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x6000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) | sdsc_12 |
|  |  | 1_lx | OUTPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x6000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 128*, 512/32, 4 | 128 x 16 x 4 | 16.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_13 |
|  |  | 1_lx | INPUT | 1*, 512/32, 4 | 4 x 16 x 1 | 0.12 KB | 0x6000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 512/32, 4 | 4 x 128 x 16 | 16.00 KB | 0x92000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_14 |
|  |  | 1_lx | INPUT | 1*, 512/32, 4 | 4 x 16 x 1 | 0.12 KB | 0x90000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_15 |
|  |  | 1_lx | OUTPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) | sdsc_16 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4 | 128 x 4096 x 4 | 4.00 MB | 0x-67 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 512/32, 4 | 128 x 16 x 4 | 16.00 KB | 0x0 | x=0 mb=core_id out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4, 128*, 512/32 | 4 x 128 x 16 | 16.00 KB | 0x92000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_17 |
|  |  | 1_lx | INPUT | 4, 128*, 512/32 | 128 x 16 x 4 | 16.00 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4, 128*, 512/32 | 128 x 16 x 4 | 16.00 KB | 0x-68 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0xa000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) | sdsc_18 |
|  |  | 1_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x6000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0xa000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4096*, 512/32, 4 | 4096 x 16 x 4 | 512.00 KB | 0x10000 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_19 |
|  |  | 1_lx | OUTPUT | 1*, 512/32, 4 | 4 x 16 x 1 | 0.12 KB | 0x0 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0xa000 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) | sdsc_20 |
|  |  | 1_lx | INPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x0 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4, 512/32, 64* | 4 x 16 x 64 | 8.00 KB | 0x-88 | mb=0 out=core_id y=0 | SEN169_FP16 (2B) |  |
| realdiv | 32 | 0_hbm | INPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x-a8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_21 |
|  |  | 1_hbm | INPUT | 1*, 512/8, 4/4 | 1 x 64 x 1 | 0.12 KB | 0x-c8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x-e8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
