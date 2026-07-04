# Flash Attention DLDSC Relayout Carrier Diagnostics

This note captures the July 4 follow-up on the flash attention all-gather relayout path.

## Setup

- Torch branch: `ah/comms-collectives`
- Deeptools branch: `ah/comms-collectives`
- Device pod: `adnan-cdx-spyre-dev-pf`
- Test script: `test_flash.py` from `github.ibm.com/aviros/test-spyre-scripts`
- Shape: the script default flash-attention value-correctness case
- Frontend relayout flags: DLDSC LX planner relayout, collectives, layout all-gather restickify, matmul operand contract, and restickify outputs enabled
- Backend split-LX setup: Torch sees `DXP_LX_FRAC_AVAIL=0`; DXP subprocess sees `DXP_BACKEND_LX_FRAC_AVAIL=0.2`

## Results

| Carrier mode | Compile result | Runtime correctness | Observation |
| --- | --- | --- | --- |
| Kernel-neighbor marker | Passes compile/run | Fails, `31.5%` mismatch | Logical plan is present, but the DL scheduler emits only a dummy `NO_COMPONENT -> LX` marker. No exact ring movement is produced. |
| Combined IFN data-op | Fails in DXP | Not reached | Aborts at `inputNeighFetchOp.cpp:2293` on `inpSPIdxToChunkRank`. Existing IFN chunk-rank assumptions do not match this grouped all-gather operand case. |
| Standalone combined STCDP | Passes compile/run | Fails, `90.7%` mismatch with `inf` | A pre-DL STCDP row is not enough. The movement is not correctly bound to the loop-scoped matmul operand consumption point. |

The latest two raw summaries are:

- `single_combined_ifn_summary.json`
- `single_standalone_stcdp_summary.json`

## Interpretation

The frontend and Deeptools classification are now doing the right logical thing: the backend plan identifies `all_gather_replicate` matmul operand broadcasts and expands source-to-destination core groups.

The remaining gap is physical lowering. Reusing IFN unchanged is not sufficient, and standalone STCDP materialization is value-wrong for this loop-scoped matmul operand. The next implementation should teach the DL path to lower exact relayout groups into ring movement at the consumer schedule point, rather than treating LX-neighbor as a boolean marker or a standalone prefetch row.

That means the next patch should focus on group-aware DL ring lowering:

- carry source core/address, destination core/address, byte/stick extent, and consumer LDS identity into the DL transfer path;
- emit `LX -> RING` and `RING -> LX` movement at the same schedule level as the consuming matmul;
- keep the DLDSC coordinate contract as the frontend/backend handoff;
- avoid relying on deprecated mixed data DSC behavior for production.
