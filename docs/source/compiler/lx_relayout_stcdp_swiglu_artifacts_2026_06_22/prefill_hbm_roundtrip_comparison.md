# Fused SwiGLU Prefill HBM Round-Trip Comparison

Shape: `B=1 S=512 E=4096`. Baseline is the same production branch with `SPYRE_LX_PLANNER_RELAYOUT=0`; LX relayout enables `SPYRE_LX_PLANNER_RELAYOUT=1` and `SPYRE_LX_PLANNER_RELAYOUT_REALIZE=1`.

| Edge | Baseline SDSC evidence | LX relayout SDSC evidence | Interpretation |
| --- | --- | --- | --- |
| First projection output into pointwise chain | `Op=batchmatmul; alloc_tensor {i}_{loc}=2_hbm; Role=OUTPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0xc800000..0xdacaf00 (32 unique); coreIdToWkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique); json files=sdsc_1` | `Op=batchmatmul; alloc_tensor {i}_{loc}=2_lx; Role=OUTPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0x0; coreIdToWkSlice=in=0 mb=0:3 (4 unique) out=0:7 (8 unique); json files=sdsc_1` | Projection output changes from HBM to LX, enabling the following pointwise consumer to read on chip. |
| SiLU neg input | `Op=neg; alloc_tensor {i}_{loc}=0_hbm; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0xc800000..0xe038000 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_2` | `Op=neg; alloc_tensor {i}_{loc}=0_lx; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0x100000; coreIdToWkSlice=mb=core_id out=0; json files=sdsc_2` | First-half activation input to neg is now LX-backed after the STCDPOpLx relayout. |
| SiLU realdiv numerator input | `Op=realdiv; alloc_tensor {i}_{loc}=0_hbm; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0xc800000..0xe038000 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_5` | `Op=realdiv; alloc_tensor {i}_{loc}=0_lx; Role=INPUT; Layout* extent/wkSlices=layout=out,mb; stick=out; stick_size=[64]; Address=0x100000; coreIdToWkSlice=mb=core_id out=0; json files=sdsc_5` | The realdiv path consumes the same first-half activation from LX instead of re-reading HBM. |
| Gate/up mul second input | `Op=mul; alloc_tensor {i}_{loc}=1_hbm; Role=INPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0xc806400..0xe03e400 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | `Op=mul; alloc_tensor {i}_{loc}=1_hbm; Role=INPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0xc806400..0xe03e400 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | Still HBM-backed in this PR1 production branch; this is narrower than the older relay-fix coordinate-remap artifact. |
| Pointwise product output to down projection | `Op=mul; alloc_tensor {i}_{loc}=2_hbm; Role=OUTPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0x0..0xf800 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | `Op=mul; alloc_tensor {i}_{loc}=2_hbm; Role=OUTPUT; Layout* extent/wkSlices=layout=mb,out; stick=out; stick_size=[64]; Address=0x0..0xf800 (32 unique); coreIdToWkSlice=mb=core_id out=0; json files=sdsc_6` | Still written to HBM for the downstream matmul; down-projection fan-out/streaming is follow-up work. |

## STCDPOpLx Movement Rows

| Chunk | Coverage | Address bases | Ranges | Expanded movements | Bytes |
| --- | --- | --- | ---: | ---: | ---: |
| dataop_0_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 124 | 6200 | 12697600 |
| dataop_1_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 112 | 112 | 229376 |
| dataop_2_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 112 | 112 | 229376 |
| dataop_3_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 88 | 88 | 180224 |
| dataop_4_lx->lx | `[0, 0, 0, 0]+[512, 200, 1, 64] of [512, 400, 1, 64]` | `src_lx=0x0 dst_lx=0x100000` | 88 | 88 | 180224 |

## Structural Counters

| Metric | Baseline | LX relayout |
| --- | ---: | ---: |
| SDSC count | 9 | 9 |
| Jamie-table rows | 23 | 28 |
| SDSCs with data ops | 0 | 1 |
| STCDPOpLx data-op chunks | 0 | 5 |
| STCDPOpLx mixed SDSCs | 0 | 1 |
| STCDPOpLx movement ranges | 0 | 524 |
| Expanded STCDPOpLx movements | 0 | 6600 |
| STCDPOpLx bytes moved | 0 | 13516800 |

## What This Proves

- The current production LX planner extension emits one mixed SDSC with ranged `STCDPOpLx` data ops before the pointwise consumer.
- The first half of the fused projection is moved through LX-to-LX ring traffic instead of an HBM round trip before `neg` and `realdiv`.
- This artifact intentionally documents the latest PR1 behavior. It does not claim that the second-half `mul` input or final pointwise product output are eliminated from HBM yet.
- The fresh artifact run is structural evidence. The short profiler capture reported zero kernel events, so timing claims should use the earlier archived Kineto runs, not wall time from this directory.
