
SDSC Operations Summary - Batch Report
Directory: /Users/adnan/Documents/Codex/2026-05-23-we-are-continuing-torch-spyre-on/generated_flash_sdsc_jamie_20260708/before_relayout_off
Total sdsc.json files found: 550

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| max | INPUT (hbm), OUTPUT (hbm) |  |
| identity | INPUT (hbm), OUTPUT (hbm) |  |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| batchmatmul | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |
| add | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| sub | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| maximum | INPUT (hbm), INPUT (lx), OUTPUT (hbm) |  |
| exp | INPUT (lx), OUTPUT (hbm) |  |
| ReStickifyOpHBM | INPUT (lx), OUTPUT (hbm) |  |
| sum | INPUT (lx), OUTPUT (lx) |  |
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
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-44d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_31 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-46d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-703 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_51 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-723 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-724 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-83 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-a3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-a7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3e5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_26 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-405 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-257 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_17 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-277 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-278 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-31d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_21 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-33d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-52f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_37 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-54f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-553 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-c7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-e7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2d9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_20 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2f9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2fd | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-18f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-1af | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-5db | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_41 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-5fb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-4ee | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_35 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-50e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-298 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_18 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2b8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4ad | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_34 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-4cd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4ce | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-12f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1f7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_14 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-217 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-63b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_43 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-65b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-573 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_38 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-593 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-21 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-22 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-6a3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_48 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-6c3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-385 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_24 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3a5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-59b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_40 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-5bb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_8 |
| (0, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-16f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_42 |
| (0, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-61b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-ef | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-10f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-345 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_23 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-365 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_25 |
| (0, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-42 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-62 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_27 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-417 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_2 |
| (0, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-63 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_11 |
| (0, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_46 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-67f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_10 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1c1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_5 |
| (0, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_36 |
| (0, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-50f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_39 |
| (0, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_15 |
| (0, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_19 |
| (0, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-2b9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_22 |
| (0, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_29 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-429 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_44 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-66d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_12 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1d3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_47 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-683 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_30 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-42d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_13 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1d7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_45 |
| (0, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_28 |
| (0, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_49 |
| (0, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_32 |
| (0, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_50 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-6e3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_16 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-237 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_33 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-48d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-f33 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_109 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-f53 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-c75 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_89 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-c95 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-e87 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_105 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-ea7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-eab | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-baf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_85 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-bcf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-bd0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-11e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_128 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-1209 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-cdd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_92 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-cfd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-c31 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_88 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-c51 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-c55 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-9db | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_71 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-9fb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-9ff | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-e46 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_103 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-e66 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-105b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_119 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-107b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-107c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1121 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_123 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1141 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-da5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_99 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-dc5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-891 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_60 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-8b1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-ae7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_77 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-b07 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-10dd | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_122 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-10fd | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1101 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-b4f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_82 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-b6f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-d3d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_94 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-d5d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-e05 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_102 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-e25 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-e26 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-785 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_54 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-7a5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-7a9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-831 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_58 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-851 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-109c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_120 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-10bc | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-a87 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_75 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-aa7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-ffb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_116 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-101b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-7c9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_55 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-7e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-a1f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_72 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-a3f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-959 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_68 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-979 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-97a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-744 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_52 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-764 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-f93 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_111 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-fb3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-ecb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_106 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-eeb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-99a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_69 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-9ba | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-bf0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_86 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-c10 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1189 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_126 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-11a9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_130 |
| (1, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_70 |
| (1, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-9bb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_66 |
| (0, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_113 |
| (1, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_129 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-121b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_104 |
| (1, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-e67 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_112 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-fc5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_124 |
| (1, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_56 |
| (0, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_95 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-d6f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_83 |
| (1, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_61 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-8c3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_114 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-fd7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_78 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-b19 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_97 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-d81 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_117 |
| (1, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_62 |
| (0, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_121 |
| (1, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-10bd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_63 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-8d5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_100 |
| (1, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_79 |
| (1, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_96 |
| (1, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_80 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-b2b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_131 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-122d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_87 |
| (1, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-c11 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_107 |
| (1, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_53 |
| (0, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-765 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_90 |
| (1, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_73 |
| (1, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-8f9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_65 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-919 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1149 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_125 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1169 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_93 |
| (1, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-d1d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_84 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-b8f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_67 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-939 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-ef3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_108 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-f13 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_115 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-fdb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_76 |
| (1, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-ac7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_98 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-d85 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_118 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-103b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-7f1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_57 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-811 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_81 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-b2f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_101 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-de5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-a47 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_74 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-a67 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_59 |
| (0, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-871 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_64 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (0, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-8d9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_127 |
| (1, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-11c9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-c9d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_91 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-cbd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_110 |
| (1, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-f73 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1251 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_133 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1271 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1a79 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_191 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1a99 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-19b3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_187 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-19d3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-19d4 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1a35 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_190 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1a55 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1a59 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1ba9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_201 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1bc9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1507 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_153 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1527 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1528 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-143f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_145 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-145f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-17df | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_173 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-17ff | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1803 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1548 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_154 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1568 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1333 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_139 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1353 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1357 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1823 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_174 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1843 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1695 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_162 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-16b5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1b41 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_196 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-1b61 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-18eb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_179 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-190b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-13df | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_143 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-13ff | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1c09 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_204 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1c29 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1c2a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1589 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_156 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-15a9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-15ad | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1377 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_140 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1397 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-12f2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_137 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1312 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1635 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_160 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1655 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-188b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_177 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-18ab | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-12b1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_136 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-12d1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-12d2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-15cd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_157 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-15ed | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1c4a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_205 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1c6a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1ae1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_194 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1b01 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-175d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_170 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-177d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-177e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14a7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_150 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14c7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1953 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_184 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1973 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-179e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_171 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-17be | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-16fd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_167 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-171d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_144 |
| (2, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-141f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-139f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_142 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-13bf | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_178 |
| (2, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-18cb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-15f5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_159 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1615 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_195 |
| (2, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1b21 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-184b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_176 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-186b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_161 |
| (2, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1675 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1aa1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_193 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1ac1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-19f4 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_188 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1a14 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_148 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1483 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_172 |
| (2, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-17bf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_164 |
| (2, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_168 |
| (2, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_165 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-16d9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_197 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1b73 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_181 |
| (2, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_158 |
| (2, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_134 |
| (1, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_163 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-16c7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_175 |
| (2, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_138 |
| (2, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-1313 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_180 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-191d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_155 |
| (2, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-1569 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_199 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1b85 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_198 |
| (2, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_141 |
| (2, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_182 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-192f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_200 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1b89 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_132 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1231 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_149 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1487 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_183 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1933 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_166 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-16dd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_189 |
| (2, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-1a15 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_185 |
| (2, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_202 |
| (2, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_146 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1471 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_147 |
| (2, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_151 |
| (2, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_152 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-14e7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_186 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1993 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_169 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-173d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_135 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (1, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1291 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_203 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (2, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1be9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_192 |
| (2, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2137 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_241 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2157 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-215b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-234c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_256 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-236c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2627 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_276 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2647 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1e5f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_221 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1e7f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1e80 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2243 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_247 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2263 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1fed | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_230 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-200d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-27b7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_289 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-27d7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-27d8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d37 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_211 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d57 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1c8b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_207 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1cab | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1caf | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-21e3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_245 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2203 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-27f8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_290 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2818 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1f8d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_228 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1fad | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2501 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_269 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2521 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2757 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_286 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2777 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1ccf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_208 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1cef | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-25a2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_273 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-25c2 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1ee1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_224 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1f01 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1f05 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2499 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_264 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-24b9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1f25 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_225 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-1f45 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2561 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_272 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2581 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2582 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-22ab | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_252 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-22cb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1d97 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_213 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-1db7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1dff | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_218 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1e1f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-23d1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_259 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-23f1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-25e3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_275 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2603 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2607 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-1ea0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_222 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-1ec0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-230b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_255 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-232b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-232c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-268f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_279 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-26af | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-20b5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_238 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-20d5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-20d6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-26ef | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_281 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-270f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-20f6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_239 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2116 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-217b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_242 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-219b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_215 |
| (3, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_236 |
| (3, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_277 |
| (4, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_282 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2721 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_257 |
| (3, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-236d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_216 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-1ddb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_240 |
| (3, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-2117 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_283 |
| (4, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_260 |
| (3, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_284 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2733 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_206 |
| (3, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-1c6b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_267 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-24dd | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_226 |
| (3, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_270 |
| (3, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_231 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-201f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_266 |
| (3, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_250 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2287 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_253 |
| (3, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_249 |
| (3, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_232 |
| (3, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_265 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-24cb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_233 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2031 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_209 |
| (3, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_248 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2275 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_287 |
| (4, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_214 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-1dc9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_243 |
| (3, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2439 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_262 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2459 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2055 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_235 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2075 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-23f9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_261 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2419 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_217 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-1ddf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_251 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-228b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1cf7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_210 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d17 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1f4d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_227 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1f6d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_246 |
| (3, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2223 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_285 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2737 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_212 |
| (3, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1d77 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_268 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-24e1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_229 |
| (3, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1fcd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-21a3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_244 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-21c3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_234 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2035 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_263 |
| (3, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2479 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_280 |
| (4, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-26cf | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-264f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_278 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-266f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_223 |
| (3, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-1ec1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_274 |
| (4, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-25c3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-238d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_258 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-23ad | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-23b1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_219 |
| (3, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_220 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-1e3f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_237 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2095 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_288 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2797 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_271 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2541 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_254 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (3, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-22eb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2ca4 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_324 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2cc4 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2df1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_332 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2e11 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2945 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_298 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2965 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3047 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_349 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3067 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2b3b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_313 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2b5b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2f7f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_344 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2f9f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2a8f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_309 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2aaf | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2ab3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-323d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_364 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-325d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2839 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_292 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2859 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-285d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2f3b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_343 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2f5b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2f5f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-33a6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_375 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-33c6 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3150 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_358 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3170 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3365 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_374 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3385 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3386 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2c63 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_323 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2c83 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2c84 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-30af | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_354 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-30cf | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-29ad | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_303 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-29cd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2b9b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_315 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-2bbb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-287d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_293 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-289d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-310f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_357 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-312f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3130 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2efa | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_341 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2f1a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-31d5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_361 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-31f5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2c03 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_320 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2c23 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3191 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_360 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-31b1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-31b5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2e59 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_337 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2e79 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2eb9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_340 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2ed9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2eda | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2d91 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_330 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2db1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3305 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_371 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3325 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2ce5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_326 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2d05 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2d09 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2a0d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_306 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2a2d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-2a2e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2ad3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_310 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2af3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2fe7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_347 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3007 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-28e5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_296 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2905 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-2a4e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_307 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-2a6e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2d29 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_327 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-2d49 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_308 |
| (4, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-2a6f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_328 |
| (4, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_369 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-32e1 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_294 |
| (4, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_345 |
| (5, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_304 |
| (4, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_352 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-308b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_368 |
| (5, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_333 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2e23 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_299 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2977 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_372 |
| (5, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_325 |
| (4, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-2cc5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_338 |
| (4, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_355 |
| (5, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_334 |
| (4, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_318 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2bdf | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_359 |
| (5, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-3171 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_335 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2e35 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_362 |
| (5, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_342 |
| (5, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-2f1b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_300 |
| (4, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_316 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-2bcd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_321 |
| (4, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_376 |
| (5, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-33c7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_291 |
| (4, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-2819 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_317 |
| (4, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_301 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-2989 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_367 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-32cf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_351 |
| (5, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_311 |
| (4, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_350 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3079 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-329d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_366 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-32bd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_373 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3345 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_365 |
| (5, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-327d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2afb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_312 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2b1b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_353 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-308f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_305 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-29ed | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-28a5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_295 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-28c5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2d51 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_329 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2d71 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_348 |
| (5, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3027 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_314 |
| (4, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2b7b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_302 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-298d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_322 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2c43 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-31fd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_363 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-321d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_319 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2be3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_339 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2e99 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_336 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (4, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-2e39 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_356 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-30ef | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_297 |
| (4, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2925 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2fa7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_346 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2fc7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_370 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-32e5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_331 |
| (4, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-2dd1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3aa8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_426 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3ac8 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3e4b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_451 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3e6b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3d83 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_446 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3da3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3893 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_411 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-38b3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-38b7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-35bb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_391 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-35db | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-35dc | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3749 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_400 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3769 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3f54 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_460 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3f74 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-399f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_417 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-39bf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3493 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_381 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-34b3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-342b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_378 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-344b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3eb3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_456 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3ed3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3cfe | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_443 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3d1e | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-363d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_394 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-365d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3661 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3a07 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_422 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3a27 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3bf5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_434 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3c15 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-36e9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_398 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3709 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-33e7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_377 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3407 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-340b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3f13 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_459 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3f33 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3f34 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3681 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_395 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-36a1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-34f3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_383 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-3513 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-393f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_415 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-395f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3cbd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_442 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3cdd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3cde | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c5d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_439 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c7d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3b95 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_432 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3bb5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3deb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_449 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3e0b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-355b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_388 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-357b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3811 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_408 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3831 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3832 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3ae9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_428 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3b09 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3b0d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3d3f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_445 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3d5f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3d63 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-38d7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_412 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-38f7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-35fc | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_392 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-361c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-37b1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_405 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-37d1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3b2d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_429 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3b4d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-3852 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_409 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3872 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_430 |
| (6, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_406 |
| (5, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_386 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3537 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_410 |
| (6, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-3873 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_447 |
| (6, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_427 |
| (6, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-3ac9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_457 |
| (6, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_379 |
| (5, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_396 |
| (5, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_436 |
| (6, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_420 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-39e3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_437 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3c39 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_440 |
| (6, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_401 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-377b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_402 |
| (5, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_418 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-39d1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_419 |
| (6, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_435 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3c27 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_423 |
| (6, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_454 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-3e8f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_403 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-378d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_384 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3525 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_453 |
| (6, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_452 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-3e7d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_393 |
| (5, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-361d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_385 |
| (5, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_413 |
| (6, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_387 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-353b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_450 |
| (6, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3e2b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3b55 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_431 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3b75 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3453 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_380 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3473 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_416 |
| (6, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-397f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_421 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-39e7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-36a9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_397 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-36c9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_438 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3c3d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-38ff | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_414 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-391f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_382 |
| (5, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-34d3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_455 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3e93 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_399 |
| (5, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3729 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_404 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-3791 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_444 |
| (6, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-3d1f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_389 |
| (5, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_390 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-359b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_407 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (5, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-37f1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_441 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3c9d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_458 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3ef3 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_424 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3a47 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3dab | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_448 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3dcb | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_433 |
| (6, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-3bd5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3a67 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_425 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-3a87 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3a88 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4999 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_534 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-49b9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-46db | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_514 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-46fb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-454d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_502 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-456d | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-43bf | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_493 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-43df | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-43e0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-42f7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_485 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-4317 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4041 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_466 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4061 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-47a3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_519 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-47c3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-48ac | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_528 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-48cc | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-4400 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_494 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-4420 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-480b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_524 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-482b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-41aa | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_477 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-41ca | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4169 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_476 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-4189 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-418a | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4297 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_483 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-42b7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4697 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_513 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-46b7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-46bb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4441 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_496 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4461 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4465 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-41eb | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_479 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-420b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-420f | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-422f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_480 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-424f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4615 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_510 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-4635 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-4636 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-48ed | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_530 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-490d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4911 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3fd9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_463 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3ff9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-486b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_527 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-488b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-488c | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-3f95 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_462 |
|  |  | 1_hbm | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3fb5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-3fb9 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4931 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_531 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4951 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-4656 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_511 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-4676 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-45b5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_507 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-45d5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-4485 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_497 |
|  |  | 1_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-44a5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-49f9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_536 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-4a19 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4109 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_473 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4129 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-435f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_490 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-437f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-44ed | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_500 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-450d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4a61 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_541 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4a81 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4743 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_517 |
|  |  | 1_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4763 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-40a1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_468 |
|  |  | 1_hbm | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x-40c1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_537 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-4a2b | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_488 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-433b | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_522 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-47e7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_471 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-40e5 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_538 |
| (7, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_539 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-4a3d | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_503 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-457f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_542 |
| (7, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_515 |
| (7, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_470 |
| (6, 3) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_512 |
| (7, 2) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-4677 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_504 |
| (7, 1) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_508 |
| (7, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_461 |
| (6, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-3f75 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_532 |
| (7, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_498 |
| (7, 1) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_525 |
| (7, 2) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_495 |
| (7, 1) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-4421 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_529 |
| (7, 3) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-48cd | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_505 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-4591 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_474 |
| (6, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_481 |
| (7, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpHBM | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_478 |
| (7, 0) |  | 1_hbm | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x-41cb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_520 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-47d5 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_486 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-4329 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_469 |
|  |  | 1_hbm | INPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x-40d3 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_487 |
| (7, 0) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_491 |
| (7, 0) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x104000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_521 |
| (7, 2) |  | 1_lx | OUTPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_lx | INPUT | 4/4, 4096*, 1024/8 | 1 x 4096 x 128 | 1.00 MB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_464 |
| (6, 3) |  | 1_lx | OUTPUT | 4/4, 1*, 1024/8 | 1 x 128 x 1 | 0.25 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_518 |
| (7, 2) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4783 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_467 |
| (6, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4081 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_484 |
| (7, 0) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-42d7 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_492 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-439f | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_535 |
| (7, 3) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-49d9 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_489 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-433f | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_523 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-47eb | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4257 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_482 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 0) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4277 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4959 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_533 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4979 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-44ad | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_499 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-44cd | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_509 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-45f5 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_506 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 1) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-4595 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_475 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4149 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_526 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-484b | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4001 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_465 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4021 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| maximum | 32 | 0_hbm | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4703 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_516 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 2) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4723 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_540 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-4a41 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_501 |
| (7, 1) |  | 1_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-452d | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x100000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_472 |
|  |  | 1_lx | INPUT | 4/4, 128*, 1024/8 | 128 x 128 x 1 | 32.00 KB | 0x108000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| (6, 3) |  | 2_hbm | OUTPUT | 4/4, 128*, 1024/8 | 1 x 128 x 128 | 32.00 KB | 0x-40e9 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x100000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) | sdsc_543 |
|  |  | 1_lx | INPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x104000 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| (7, 3) |  | 2_hbm | OUTPUT | 4/4, 1024/8, 64* | 1 x 128 x 64 | 16.00 KB | 0x-4aa1 | {mb=0:3} {out=0:7} y=0 | SEN169_FP16 (2B) |  |
| realdiv | 32 | 0_hbm | INPUT | 32, 128*, 4096/32 | 32 x 128 x 128 | 1.00 MB | 0x-1 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 32, 1*, 4096/32 | 32 x 128 x 1 | 8.00 KB | 0x-21 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 32, 128*, 4096/32 | 32 x 128 x 128 | 1.00 MB | 0x-41 | mb=0 x=core_id out=0 | SEN169_FP16 (2B) |  |
