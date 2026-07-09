
SDSC Operations Summary - Batch Report
Directory: /home/adnan/codex-isolated/sdpa_pr2_allgather_20260709_035348/runs/granite_collectives_backend_l512_20260709_042551/relayout_on/cache/inductor-spyre/sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_yytqnma4
Total sdsc.json files found: 11

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
| identity | INPUT (lx), OUTPUT (hbm) |  |

Tensor Summary Table:

| FA role | Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mul | mul | 32 | 0_hbm | INPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x-1 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_0 |
|  |  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-21 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  |  | 2_lx | OUTPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| mul | mul | 32 | 0_hbm | INPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x-22 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  |  | 1_hbm | INPUT | 1*, 1, 4/4 | 1 x 1 x 1 | 0.00 KB | 0x-42 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  |  | 2_lx | OUTPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| relayout | ReStickifyOpLx | 32 | 0_lx | INPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x4000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  |  | 1_lx | OUTPUT | 512*/8, 128, 4/4 | 64 x 128 x 1 | 16.00 KB | 0x8000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| matmul | batchmatmul | 32 | 0_lx | INPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x0 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  |  | 1_lx | INPUT | 512*, 128, 4/4 | 512 x 128 x 1 | 128.00 KB | 0x8000 | {out=0:7} in=0 {x=0:3} | SEN169_FP16 (2B) |  |
|  |  |  | 2_hbm | OUTPUT | 512*, 512/8, 4/4 | 512 x 64 x 1 | 64.00 KB | 0x-43 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| init: denom2 | max | 32 | 0_hbm | INPUT | 512*, 512/8, 4/4 | 512 x 64 x 1 | 64.00 KB | 0x-63 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_4 |
|  |  |  | 1_lx | OUTPUT | 1*, 512/8, 4/4 | 1 x 64 x 1 | 0.12 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| sub | sub | 32 | 0_hbm | INPUT | 512*, 512/8, 4/4 | 512 x 64 x 1 | 64.00 KB | 0x-83 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_5 |
|  |  |  | 1_lx | INPUT | 1*, 512/8, 4/4 | 1 x 64 x 1 | 0.12 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  |  | 2_lx | OUTPUT | 512*, 512/8, 4/4 | 1 x 512 x 64 | 64.00 KB | 0x2000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| exp | exp | 32 | 0_lx | INPUT | 4/4, 512*, 512/8 | 1 x 512 x 64 | 64.00 KB | 0x2000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_6 |
|  |  |  | 1_lx | OUTPUT | 4/4, 512*, 512/8 | 1 x 512 x 64 | 64.00 KB | 0x2000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  | sum | 32 | 0_lx | INPUT | 4/4, 512*, 512/8 | 1 x 512 x 64 | 64.00 KB | 0x2000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_7 |
|  |  |  | 1_lx | OUTPUT | 4/4, 1*, 512/8 | 1 x 64 x 1 | 0.12 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| divide | realdiv | 32 | 0_lx | INPUT | 4/4, 512*, 512/8 | 1 x 512 x 64 | 64.00 KB | 0x2000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) | sdsc_8 |
|  |  |  | 1_lx | INPUT | 4/4, 1*, 512/8 | 1 x 64 x 1 | 0.12 KB | 0x0 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
|  |  |  | 2_lx | OUTPUT | 4/4, 512*, 512/8 | 1 x 512 x 64 | 64.00 KB | 0x2000 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
| matmul | batchmatmul | 32 | 0_lx | INPUT | 4/4, 512*, 512/8 | 1 x 512 x 64 | 64.00 KB | 0x2000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) | sdsc_9 |
|  |  |  | 1_hbm | INPUT | 128*, 512, 4/4 | 128 x 512 x 1 | 128.00 KB | 0x-a3 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
|  |  |  | 2_lx | OUTPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x12000 | {x=0:3} {mb=0:7} out=0 in=0 | SEN169_FP16 (2B) |  |
| identity | identity | 32 | 0_lx | INPUT | 128*, 512/8, 4/4 | 128 x 64 x 1 | 16.00 KB | 0x12000 | out=0 {x=0:7} {mb=0:3} | SEN169_FP16 (2B) | sdsc_10 |
|  |  |  | 1_hbm | OUTPUT | 128*, 512/8, 4/4 | 128 x 1 x 64 | 16.00 KB | 0x-a7 | {mb=0:3} {x=0:7} out=0 | SEN169_FP16 (2B) |  |
