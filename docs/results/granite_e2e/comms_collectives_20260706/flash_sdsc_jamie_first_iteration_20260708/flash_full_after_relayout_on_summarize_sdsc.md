
SDSC Operations Summary - Batch Report
Directory: /Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/generated_flash_sdsc_jamie_20260708/after_relayout_on
Total sdsc.json files found: 550

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| max | INPUT (hbm), OUTPUT (hbm) |  |
| identity | INPUT (hbm), OUTPUT (hbm) |  |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| add | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| sub | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| exp | INPUT (lx), OUTPUT (hbm) |  |
| maximum | INPUT (hbm), INPUT (lx), OUTPUT (hbm) |  |
| ReStickifyOpLx | INPUT (lx), OUTPUT (lx) |  |
| sum | INPUT (lx), OUTPUT (lx) |  |
| batchmatmul | INPUT (hbm), INPUT (lx), OUTPUT (hbm) |  |
| realdiv | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max | 32 | 0_hbm | INPUT | 4096/32, 32, 64* | 128 x 32 x 64 | 512.00 KB | 0x-43 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_hbm | OUTPUT | 4096/32, 32, 64* | 32 x 128 x 64 | 512.00 KB | 0x-63 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_hbm | INPUT | 1, 1, 64* | 1 x 1 x 64 | 0.12 KB | 0x-83 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_hbm | OUTPUT | 4096/32, 32, 64* | 128 x 32 x 64 | 512.00 KB | 0x-84 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_hbm | INPUT | 4096/32, 32, 64* | 128 x 32 x 64 | 512.00 KB | 0x-a4 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_hbm | OUTPUT | 4096/32, 32, 64* | 32 x 128 x 64 | 512.00 KB | 0x-c4 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_hbm | INPUT | 1*, 1, 1 | 1 x 1 x 1 | 0.00 KB | 0x-1 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | OUTPUT | 128*, 4096/32, 32 | 128 x 128 x 32 | 1.00 MB | 0x-2 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| identity | 32 | 0_hbm | INPUT | 1, 1, 64* | 1 x 1 x 64 | 0.12 KB | 0x-22 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_hbm | OUTPUT | 4096/32, 32, 64* | 128 x 32 x 64 | 512.00 KB | 0x-23 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-39d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_26 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3bd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-233 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_17 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-253 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-254 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2d5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_21 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2f5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-a3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-c3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-16b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-18b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-274 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_18 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-294 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-10b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-12b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_14 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1f3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-21 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-22 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-33d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_24 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-35d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_8 |
| (0, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-cb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-eb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2fd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_23 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-31d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_25 |
| (0, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-37d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-42 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-62 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_27 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3cf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_2 |
| (0, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_11 |
| (0, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_10 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-19d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_5 |
| (0, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_15 |
| (0, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_16 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-213 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_19 |
| (0, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-63 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-83 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-295 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_20 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2b5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_22 |
| (0, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_29 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3e1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_12 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1af | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_13 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1b3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_28 |
| (0, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-e37 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_109 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-e57 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-405 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_31 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-425 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-b9d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_89 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-bbd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-afb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_85 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-b1b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-b1c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-c05 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_92 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-c25 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-697 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_51 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-6b7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-6b8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-d6e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_103 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-d8e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-ccd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_99 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-ced | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-801 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_60 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-821 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-a33 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_77 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-a53 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-a9b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_82 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-abb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-c65 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_94 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-c85 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-d2d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_102 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-d4d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-d4e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-56f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_41 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-58f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-4a6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_35 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-4c6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-7a1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_58 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-7c1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-9d3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_75 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-9f3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-465 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_34 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-485 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-486 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-5cf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_43 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-5ef | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-739 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_55 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-759 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-507 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_38 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-527 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-96b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_72 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-98b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-637 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_48 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-657 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-8c9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_68 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-8e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-8ea | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-6d8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_52 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-6f8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-dcf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_106 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-def | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-90a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_69 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-92a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-b3c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_86 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-b5c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_70 |
| (1, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_66 |
| (0, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_46 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-613 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_104 |
| (1, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_56 |
| (0, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_95 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-c97 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_83 |
| (1, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_61 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-833 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_36 |
| (0, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_39 |
| (0, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_78 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-a65 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_97 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-ca9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_62 |
| (0, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_63 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-845 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_100 |
| (1, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_79 |
| (1, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_96 |
| (1, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_80 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-a77 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_87 |
| (1, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_107 |
| (1, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_44 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-601 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_45 |
| (0, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_53 |
| (0, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-d8f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_105 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-daf | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-b5d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_88 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-b7d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-92b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_71 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-94b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4c7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_37 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4e7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-6f9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_54 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-719 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_90 |
| (1, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_49 |
| (0, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_73 |
| (1, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_32 |
| (0, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-869 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_65 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-889 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_50 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-677 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_93 |
| (1, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-c45 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_84 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-adb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_47 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-617 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_67 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-8a9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_30 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3e5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-df7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_108 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-e17 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-52f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_40 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-54f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_76 |
| (1, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-a13 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_98 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-cad | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-761 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_57 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-781 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_81 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-a7b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_101 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-d0d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_42 |
| (0, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-5af | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-993 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_74 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-9b3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_59 |
| (0, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-7e1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_64 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-849 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_33 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-445 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-bc5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_91 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-be5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1131 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_133 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1151 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-10c9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_128 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-10e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-13c3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_153 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-13e3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-13e4 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-12fb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_145 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-131b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1404 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_154 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1424 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-f5f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_119 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-f7f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-f80 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1001 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_123 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1021 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1697 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_174 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-16b7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-152d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_162 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-154d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-129b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_143 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-12bb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1233 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_140 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1253 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-11d2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_137 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-11f2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14cd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_160 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14ed | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-fa0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_120 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-fc0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-16ff | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_177 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-171f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1191 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_136 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-11b1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-11b2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-eff | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_116 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-f1f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1465 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_157 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1485 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-15f5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_170 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1615 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1616 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1363 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_150 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1383 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-e97 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_111 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-eb7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1069 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_126 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1089 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1636 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_171 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1656 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1595 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_167 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-15b5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1029 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_125 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1049 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_144 |
| (2, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-12db | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-125b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_142 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-127b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_178 |
| (2, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-173f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-148d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_159 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14ad | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-16bf | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_176 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-16df | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_161 |
| (2, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-150d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_127 |
| (1, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-10a9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_110 |
| (1, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-e77 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_130 |
| (1, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_148 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-133f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_172 |
| (2, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_164 |
| (2, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_113 |
| (1, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_129 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-10fb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_168 |
| (2, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_112 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-ec9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_165 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1571 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_124 |
| (1, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_158 |
| (2, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_134 |
| (1, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_163 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-155f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_175 |
| (2, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_138 |
| (2, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_155 |
| (2, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_114 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-edb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_117 |
| (1, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_121 |
| (1, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1657 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_173 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1677 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-11f3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_139 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1213 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-fc1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_122 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-fe1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1425 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_156 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1445 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_141 |
| (2, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_131 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-110d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_132 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1111 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_149 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1343 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_115 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-edf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_166 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1575 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_146 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-132d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_147 |
| (2, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_151 |
| (2, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_152 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-13a3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_169 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-15d5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_135 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1171 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_118 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-f3f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-18c9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_191 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-18e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1827 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_187 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1847 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1848 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2130 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_256 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2150 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-19f9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_201 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1a19 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1c8b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_221 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1cab | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1cac | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2027 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_247 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2047 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1df5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_230 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-1e15 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1b63 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_211 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1b83 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1991 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_196 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-19b1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-175f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_179 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-177f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1fc7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_245 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1fe7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1a59 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_204 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1a79 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1a7a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d95 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_228 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1db5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1afb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_208 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1b1b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1d2d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_225 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1d4d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1a9a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_205 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1aba | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-208f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_252 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-20af | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1931 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_194 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1951 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1bc3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_213 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-1be3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1c2b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_218 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1c4b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2191 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_259 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-21b1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1ccc | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_222 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1cec | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-20ef | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_255 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-210f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2110 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1ebd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_238 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1edd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1ede | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1efe | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_239 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1f1e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1f5f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_242 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1f7f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-17c7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_184 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-17e7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1e5d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_235 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1e7d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1b23 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_210 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1b43 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d55 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_227 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d75 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_246 |
| (3, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2007 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_212 |
| (3, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1ba3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_195 |
| (2, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1971 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_229 |
| (3, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1dd5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1f87 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_244 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1fa7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-18f1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_193 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1911 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1868 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_188 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1888 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_236 |
| (3, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_257 |
| (3, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_216 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1c07 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_240 |
| (3, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_260 |
| (3, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_206 |
| (3, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_197 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-19c3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_181 |
| (2, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_226 |
| (3, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_231 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1e27 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_180 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1791 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_250 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-206b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_253 |
| (3, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_249 |
| (3, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_199 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-19d5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_232 |
| (3, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_233 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1e39 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_209 |
| (3, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_198 |
| (2, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_248 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2059 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_182 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-17a3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_200 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-19d9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_217 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1c0b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_251 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-206f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_183 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-17a7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_189 |
| (2, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_234 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1e3d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_214 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1bf5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_185 |
| (2, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_243 |
| (3, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_202 |
| (2, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_192 |
| (2, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_215 |
| (3, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_223 |
| (3, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1f1f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_241 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1f3f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1889 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_190 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-18a9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1abb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_207 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1adb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1ced | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_224 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1d0d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2151 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_258 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2171 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_219 |
| (3, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_220 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1c6b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_186 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1807 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_237 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1e9d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_203 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1a39 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_254 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-20cf | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-29f8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_324 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2a18 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2b21 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_332 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2b41 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-26bd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_298 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-26dd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-288f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_313 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-28af | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-23c3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_276 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-23e3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2553 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_289 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2573 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2574 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-29b7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_323 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-29d7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-29d8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2725 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_303 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2745 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-28ef | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_315 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-290f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-25f5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_293 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2615 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2c2a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_341 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2c4a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2594 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_290 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-25b4 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-22c1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_269 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-22e1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-24f3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_286 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2513 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2362 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_273 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2382 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2957 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_320 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2977 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2259 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_264 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2279 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2321 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_272 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2341 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2342 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2b89 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_337 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2ba9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2be9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_340 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2c09 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2c0a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2ac1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_330 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2ae1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2785 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_306 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-27a5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-27a6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2827 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_310 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2847 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-242b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_279 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-244b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-265d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_296 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-267d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-248b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_281 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-24ab | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-27c6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_307 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-27e6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-21f9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_262 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2219 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-21b9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_261 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-21d9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-284f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_312 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-286f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-261d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_295 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-263d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2a81 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_329 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2aa1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_314 |
| (4, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-28cf | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_263 |
| (3, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2239 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_280 |
| (4, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-246b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-23eb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_278 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-240b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_297 |
| (4, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-269d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2a59 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_327 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2a79 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_277 |
| (4, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_308 |
| (4, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_328 |
| (4, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_282 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-24bd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_294 |
| (4, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_304 |
| (4, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_283 |
| (4, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_333 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2b53 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_299 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-26ef | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_325 |
| (4, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_284 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-24cf | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_338 |
| (4, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_267 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-229d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_334 |
| (4, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_318 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2933 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_335 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2b65 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_270 |
| (3, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_266 |
| (3, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_300 |
| (4, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_316 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2921 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_265 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-228b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_321 |
| (4, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_287 |
| (4, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_305 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2765 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_288 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2533 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_322 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2997 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_271 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2301 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_339 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2bc9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_291 |
| (4, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_317 |
| (4, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_301 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2701 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_302 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2705 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_319 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2937 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_285 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-24d3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_336 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2b69 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_268 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-22a1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_311 |
| (4, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_274 |
| (4, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-27e7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_309 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2807 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-25b5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_292 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-25d5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2383 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_275 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-23a3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2a19 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_326 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2a39 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_331 |
| (4, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2b01 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2d53 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_349 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2d73 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2c8b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_344 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2cab | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-327f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_391 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-329f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-32a0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2f25 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_364 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2f45 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-33e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_400 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3409 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-308e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_375 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-30ae | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2e5c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_358 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2e7c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-304d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_374 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-306d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-306e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2dbb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_354 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2ddb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-361b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_417 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-363b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3157 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_381 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3177 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-30ef | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_378 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-310f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2e1b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_357 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2e3b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2e3c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3683 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_422 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-36a3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2ebd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_361 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2edd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3389 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_398 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-33a9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3321 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_395 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3341 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-31b7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_383 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-31d7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-35bb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_415 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-35db | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-321f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_388 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-323f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2fed | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_371 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-300d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-34b1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_408 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-34d1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-34d2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2cf3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_347 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2d13 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3553 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_412 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3573 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-32c0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_392 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-32e0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3451 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_405 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3471 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-34f2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_409 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3512 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_406 |
| (5, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_369 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2fc9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_386 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-31fb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_410 |
| (6, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_345 |
| (5, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_352 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2d97 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_368 |
| (5, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_372 |
| (5, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_379 |
| (5, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_396 |
| (5, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_355 |
| (5, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_420 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-365f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_359 |
| (5, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_362 |
| (5, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_342 |
| (5, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_401 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-341b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_402 |
| (5, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_418 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-364d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_376 |
| (5, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_419 |
| (6, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_403 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-342d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_367 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2fb7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_351 |
| (5, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_384 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-31e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_393 |
| (5, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3513 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_411 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3533 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2c4b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_343 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2c6b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-32e1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_394 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3301 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-30af | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_377 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-30cf | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2e7d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_360 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2e9d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_385 |
| (5, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_413 |
| (6, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_350 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2d85 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_389 |
| (5, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2f85 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_366 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2fa5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_373 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-302d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_365 |
| (5, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2f65 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_390 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-325f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_353 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2d9b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_387 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-31ff | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_407 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3491 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_348 |
| (5, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2d33 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3117 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_380 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3137 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_416 |
| (6, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-35fb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2ee5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_363 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2f05 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_421 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3663 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3349 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_397 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3369 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-357b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_414 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-359b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_382 |
| (5, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3197 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_399 |
| (5, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-33c9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_356 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2dfb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_404 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3431 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2cb3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_346 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2cd3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_370 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2fcd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3724 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_426 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3744 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3a7f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_451 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3a9f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3fab | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_493 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3fcb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3fcc | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3ee3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_485 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3f03 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-39b7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_446 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-39d7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c51 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_466 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c71 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3fec | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_494 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-400c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3dba | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_477 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3dda | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3d79 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_476 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3d99 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3d9a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3b88 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_460 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3ba8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3e83 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_483 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3ea3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3ae7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_456 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3b07 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3e1b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_480 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3e3b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3956 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_443 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3976 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3be9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_463 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3c09 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-384d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_434 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-386d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3b47 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_459 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3b67 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3b68 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3915 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_442 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3935 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3936 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-38b5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_439 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-38d5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-404d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_497 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-406d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3d19 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_473 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3d39 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-37ed | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_432 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-380d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3a1f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_449 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3a3f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3f4b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_490 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3f6b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-40b5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_500 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-40d5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3cb1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_468 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3cd1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3785 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_429 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-37a5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_430 |
| (6, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_488 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3f27 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_471 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3cf5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_447 |
| (6, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_427 |
| (6, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_470 |
| (6, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_457 |
| (6, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_436 |
| (6, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_461 |
| (6, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_498 |
| (7, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_437 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3891 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_495 |
| (7, 1) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_440 |
| (6, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_435 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-387f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_474 |
| (6, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_423 |
| (6, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_454 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3ac3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_481 |
| (7, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_478 |
| (7, 0) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_486 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3f15 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_469 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3ce3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_453 |
| (6, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_452 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3ab1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_444 |
| (6, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-400d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_496 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-402d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3ddb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_479 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3dfb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3ba9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_462 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3bc9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3745 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_428 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3765 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3977 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_445 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3997 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_487 |
| (7, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_491 |
| (7, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_464 |
| (6, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_467 |
| (6, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c91 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_484 |
| (7, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3ec3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_492 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3f8b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_450 |
| (6, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3a5f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_489 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3f2b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-37ad | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_431 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-37cd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_441 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-38f5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3e43 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_482 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3e63 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4075 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_499 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4095 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_438 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3895 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_455 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3ac7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_475 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3d59 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_458 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3b27 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_424 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-36c3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c11 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_465 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c31 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_501 |
| (7, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-40f5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-39df | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_448 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-39ff | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_433 |
| (6, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-382d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_472 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3cf9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-36e3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_425 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3703 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3704 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4519 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_534 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4539 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-427f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_514 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-429f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4115 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_502 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-4135 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4347 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_519 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-4367 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-4450 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_528 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-4470 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-43af | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_524 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-43cf | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-41dd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_510 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-41fd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-41fe | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-440f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_527 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-442f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4430 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-44b1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_531 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-44d1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-421e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_511 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-423e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-417d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_507 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-419d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4579 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_536 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-4599 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-45e1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_541 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4601 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-42e7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_517 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4307 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_518 |
| (7, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4327 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_535 |
| (7, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4559 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-44d9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_533 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-44f9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-42a7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_516 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-42c7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_537 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-45ab | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_522 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-438b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_538 |
| (7, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_539 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-45bd | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_503 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-4147 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_542 |
| (7, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_515 |
| (7, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_512 |
| (7, 2) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_504 |
| (7, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_508 |
| (7, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_532 |
| (7, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_525 |
| (7, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_543 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4621 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_509 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-41bd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_529 |
| (7, 3) |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x20000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_505 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-4159 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_523 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-438f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-423f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_513 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-425f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_506 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-415d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4471 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_530 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x20000 | {out=0:3} {in=0:7} x=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4491 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_526 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-43ef | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_520 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-4379 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_540 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-45c1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_521 |
| (7, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| realdiv | 32 | 0_hbm | INPUT | 32, 128*, 4096/32 | 32 x 128 x 128 | 1.00 MB | 0x-1 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 32, 1*, 4096/32 | 32 x 128 x 1 | 8.00 KB | 0x-21 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 32, 128*, 4096/32 | 32 x 128 x 128 | 1.00 MB | 0x-41 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
