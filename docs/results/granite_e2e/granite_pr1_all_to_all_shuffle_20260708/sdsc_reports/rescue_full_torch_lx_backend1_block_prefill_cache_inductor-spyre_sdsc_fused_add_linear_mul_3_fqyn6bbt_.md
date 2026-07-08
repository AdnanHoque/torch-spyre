
SDSC Operations Summary - Batch Report
Directory: /home/adnan/codex-isolated/pr1_rescue_compare_20260708/runs/granite_rescue_device_20260708_200414/rescue_full_torch_lx_backend1/block_prefill/cache/inductor-spyre/sdsc_fused_add_linear_mul_3_fqyn6bbt
Total sdsc.json files found: 4

Operations Summary:

| Op | Tensors | Move Op |
| --- | --- | --- |
| ReStickifyOpHBM | INPUT (hbm), OUTPUT (hbm) |  |
| batchmatmul | INPUT (hbm), INPUT (hbm), OUTPUT (hbm) |  |
| mul | INPUT (hbm), INPUT (hbm), OUTPUT (lx) |  |
| add | INPUT (lx), INPUT (lx), OUTPUT (hbm) |  |

Tensor Summary Table:

| Op (outer, inner) loop iter | cores | alloc_tensor {i}_{loc} (except relayout) | Role | Layout* extent/wkSlices | Tile Shape | Tile Size | Address | coreIdToWkSlice | Format | json files |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReStickifyOpHBM | 25 | 0_hbm | INPUT | 4096, 12800*/25 | 4096 x 512 | 4.00 MB | 0x-1 | mb=0 out=core_id | SEN169_FP16 (2B) | sdsc_0 |
|  |  | 1_hbm | OUTPUT | 12800/25, 4096* | 512 x 4096 | 4.00 MB | 0x-1a | mb=0 out=core_id | SEN169_FP16 (2B) |  |
| batchmatmul | 32 | 0_hbm | INPUT | 512/8, 12800* | 64 x 12800 | 1.56 MB | 0x-33 | {mb=0:7} {out=0:3} in=0 | SEN169_FP16 (2B) | sdsc_1 |
|  |  | 1_hbm | INPUT | 12800, 4096*/4 | 12800 x 1024 | 25.00 MB | 0x-3b | {mb=0:7} {out=0:3} in=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4096*/4, 512/8 | 1024 x 64 | 128.00 KB | 0x-3f | {mb=0:7} {out=0:3} in=0 | SEN169_FP16 (2B) |  |
| mul | 32 | 0_hbm | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x-5f | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_2 |
|  |  | 1_hbm | INPUT | 1*, 1 | 1 x 1 | 0.00 KB | 0x-7f | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_lx | OUTPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
| add | 32 | 0_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x20000 | mb=core_id out=0 | SEN169_FP16 (2B) | sdsc_3 |
|  |  | 1_lx | INPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x0 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
|  |  | 2_hbm | OUTPUT | 4096*, 512/32 | 4096 x 16 | 128.00 KB | 0x-80 | mb=core_id out=0 | SEN169_FP16 (2B) |  |
