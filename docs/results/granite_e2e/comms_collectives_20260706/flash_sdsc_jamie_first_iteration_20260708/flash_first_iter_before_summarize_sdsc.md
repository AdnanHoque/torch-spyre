
SDSC Operations Summary - Batch Report
Directory: /Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/generated_flash_sdsc_jamie_20260708/first_iteration/before_relayout_off
Total sdsc.json files found: 17

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |
| ReStickifyOpHBM | INPUT (lx), OUTPUT (hbm) |  |
| batchmatmul | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |
| add | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| max | INPUT (lx), OUTPUT (lx) |  |
| maximum | INPUT (hbm), INPUT (lx), OUTPUT (hbm) |  |
| sub | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| exp | INPUT (lx), OUTPUT (hbm) |  |
| sum | INPUT (lx), OUTPUT (lx) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-21 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-22 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-42 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-62 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-63 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-83 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-a3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-a7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-c7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-e7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_5 |
|  |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-ef | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-10f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-12f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_8 |
|  |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-16f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-18f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-1af | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_10 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1c1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_11 |
|  |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_12 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1d3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_13 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1d7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1f7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_14 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-217 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_15 |
|  |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_16 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-237 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
