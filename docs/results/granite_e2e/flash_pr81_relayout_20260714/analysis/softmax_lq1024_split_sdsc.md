
SDSC Operations Summary - Batch Report
Directory: /home/adnan/codex-isolated/flash_pr81_relayout_20260714/runs/pr81_matrix/flash_attn_softmax/dev/lq1024_mask0_split/cache/inductor-spyre
Total sdsc.json files found: 10

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| ReStickifyOpLx | INPUT (lx), OUTPUT (lx) |  |
| batchmatmul | INPUT (lx), INPUT (lx), OUTPUT (hbm) |  |
| max | INPUT (hbm), OUTPUT (lx) |  |
| sub | INPUT (hbm), INPUT (lx), OUTPUT (lx) |  |
| exp | INPUT (lx), OUTPUT (lx) |  |
| sum | INPUT (lx), OUTPUT (lx) |  |
| realdiv | INPUT (lx), INPUT (lx), OUTPUT (lx) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mul | 32 | 0_hbm | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-21 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-22 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-42 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x8000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 4096/8, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x8000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_lx | OUTPUT | 4096*/8, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x28000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_lx | INPUT | 4096*, 128, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x28000 | {out=0:7} {in=0:3} x=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-43 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| max | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-63 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  | 1_lx | OUTPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | 32 | 0_hbm | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x-83 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_5 |
|  |  | 1_lx | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | 32 | 0_lx | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  | 1_lx | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sum | 32 | 0_lx | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  | 1_lx | OUTPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| realdiv | 32 | 0_lx | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_8 |
|  |  | 1_lx | INPUT | 1*, 1024/8, 4/4 | 1 x 128 x 1 | 0.25 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_lx | INPUT | 4096*, 1024/8, 4/4 | 4096 x 128 x 1 | 1.00 MB | 0x4000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  | 1_hbm | INPUT | 128*, 4096, 4/4 | 128 x 4096 x 1 | 1.00 MB | 0x-a3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 128*, 1024/8, 4/4 | 128 x 128 x 1 | 32.00 KB | 0x-a7 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
