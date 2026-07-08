
SDSC Operations Summary - Batch Report
Directory: /home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414/baseline_off/block_prefill/cache/inductor-spyre/sdsc_fused_linear_rms_norm_0_ny_j4_e_
Total sdsc.json files found: 8

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| mean | INPUT (lx), OUTPUT (lx) |  |
| add | INPUT (lx), INPUT (hbm), OUTPUT (lx) |  |
| rsqrt | INPUT (lx), OUTPUT (lx) |  |
| ReStickifyOpHBM | INPUT (hbm), OUTPUT (hbm) |  |
| batchmatmul | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mul | 32 | 0_hbm | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x-1 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x-1 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| mean | 32 | 0_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_lx | OUTPUT | 1*, 512/32 | 16 x 1 | 0.03 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x20000 | out=core_id x=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_hbm | INPUT | 1, 64* | 1 x 64 | 0.12 KB | 0x-21 | out=core_id x=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x20000 | out=core_id x=0 | SEN169_FP16 (2B) |  |
| rsqrt | 32 | 0_lx | INPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x20000 | out=core_id x=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_lx | OUTPUT | 512/32, 64* | 16 x 64 | 2.00 KB | 0x20000 | out=core_id x=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x-22 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_lx | INPUT | 1*, 512/32 | 16 x 1 | 0.03 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32 | 16 x 4096 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_lx | INPUT | 512/32, 4096* | 16 x 4096 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_5 |
|  |  | 1_hbm | INPUT | 1, 4096* | 4096 x 1 | 8.00 KB | 0x-42 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 512/32, 4096* | 16 x 4096 | 128.00 KB | 0x-43 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_hbm | INPUT | 6144/32, 4096* | 192 x 4096 | 1.50 MB | 0x-63 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_hbm | OUTPUT | 4096, 6144*/32 | 4096 x 192 | 1.50 MB | 0x-83 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 512/4, 4096* | 128 x 4096 | 1.00 MB | 0x-a3 | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_hbm | INPUT | 4096, 6144*/8 | 4096 x 768 | 6.00 MB | 0x-a7 | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 6144*/8, 512/4 | 768 x 128 | 192.00 KB | 0x-af | {mb=0:3} {out=0:7} in=0 | SEN169_FP16 (2B) |  |
