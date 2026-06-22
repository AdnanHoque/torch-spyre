# Fused SwiGLU Prefill HBM Round-Trip Comparison

Shape: `B=1 S=512 E=4096`. Baseline is upstream Torch-Spyre main at `c6b357a`; the STCDPOpLx LX-relayout run uses `pr-lx-planner-relayout-extension` at `0f9bbcb` with `SPYRE_LX_PLANNER_RELAYOUT=1` and `SPYRE_LX_PLANNER_RELAYOUT_REALIZE=1`.

| Edge | Baseline SDSC evidence | LX relayout SDSC evidence | Interpretation |
| --- | --- | --- | --- |
| First projection output into pointwise chain | `Op=batchmatmul; alloc_tensor {i}_{loc}=2_hbm; Role=OUTPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0xc800000..0xdacaf00 (32 unique); coreIdToWkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique); json files=sdsc_1` | `Op=batchmatmul; alloc_tensor {i}_{loc}=2_lx; Role=OUTPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0x0; coreIdToWkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique); json files=sdsc_1` | Projection output changes from HBM to LX, enabling the following pointwise consumer to read on chip. |
| SiLU neg input | `Op=neg; alloc_tensor {i}_{loc}=0_hbm; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0xc800000..0xe038000 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_2` | `Op=neg; alloc_tensor {i}_{loc}=0_lx; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0x100000; coreIdToWkSlice=mb=core_id out=0; json files=sdsc_2` | First-half activation input to neg is now LX-backed after the STCDPOpLx relayout. |
| SiLU realdiv numerator input | `Op=realdiv; alloc_tensor {i}_{loc}=0_hbm; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0xc800000..0xe038000 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_5` | `Op=realdiv; alloc_tensor {i}_{loc}=0_lx; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0x100000; coreIdToWkSlice=mb=core_id out=0; json files=sdsc_5` | The realdiv path consumes the same first-half activation from LX instead of re-reading HBM. |
| Gate/up mul second input | `Op=mul; alloc_tensor {i}_{loc}=1_hbm; Role=INPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0xc806400..0xe03e400 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | `Op=mul; alloc_tensor {i}_{loc}=1_lx; Role=INPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0x100000; coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | Fixed branch now relayouts the second fused-projection half on chip before `mul`; this matches the older coordinate-remap relay-fix behavior. |
| Pointwise product output to down projection | `Op=mul; alloc_tensor {i}_{loc}=2_hbm; Role=OUTPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0x0..0xf800 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | `Op=mul; alloc_tensor {i}_{loc}=2_hbm; Role=OUTPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0x0..0xf800 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | Still written to HBM for the downstream matmul; down-projection fan-out/streaming is follow-up work. |

## STCDPOpLx Movement Rows

| Chunk | Coverage | Address bases | Ranges | Expanded movements | Bytes |
| --- | --- | --- | ---: | ---: | ---: |
| sdsc_2:dataop_0_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 124 | 6200 | 12697600 |
| sdsc_2:dataop_1_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 112 | 112 | 229376 |
| sdsc_2:dataop_2_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 112 | 112 | 229376 |
| sdsc_2:dataop_3_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 88 | 88 | 180224 |
| sdsc_2:dataop_4_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 88 | 88 | 180224 |
| sdsc_6:dataop_0_lx->lx | `[0, 200, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 124 | 6200 | 12697600 |
| sdsc_6:dataop_1_lx->lx | `[0, 200, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 112 | 112 | 229376 |
| sdsc_6:dataop_2_lx->lx | `[0, 200, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 112 | 112 | 229376 |
| sdsc_6:dataop_3_lx->lx | `[0, 200, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 88 | 88 | 180224 |
| sdsc_6:dataop_4_lx->lx | `[0, 200, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 88 | 88 | 180224 |

## Structural Counters

| Metric | Baseline | LX relayout |
| --- | ---: | ---: |
| SDSC count | 9 | 9 |
| Jamie-table rows | 23 | 33 |
| SDSCs with data ops | 0 | 2 |
| STCDPOpLx data-op chunks | 0 | 10 |
| STCDPOpLx mixed SDSCs | 0 | 2 |
| STCDPOpLx movement ranges | 0 | 1048 |
| Expanded STCDPOpLx movements | 0 | 13200 |
| STCDPOpLx bytes moved | 0 | 27033600 |

## What This Proves

- The fixed production LX planner extension emits two mixed SDSCs with ranged `STCDPOpLx` data ops: one before `neg`, and one before `mul`.
- Both halves of the fused projection are moved through LX-to-LX ring traffic instead of being re-read from HBM by the pointwise consumers.
- The final pointwise product still writes to HBM for the downstream matmul; down-projection fan-out/streaming remains follow-up work.
- This artifact includes a profiler-enabled Kineto run. Use `prefill_three_way_kernel_comparison.md` for the timing table and keep wall time separate from trace-derived kernel time.
