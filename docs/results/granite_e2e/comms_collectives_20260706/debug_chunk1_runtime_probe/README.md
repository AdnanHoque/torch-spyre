# Granite staged gather/restickify chunk-1 runtime probe

Date: 2026-07-06

## Question

Does the Granite runtime timeout come from the scale of the staged matmul-operand gather/restickify lowering, or does even the first staged chunk trigger the runtime completion problem?

## Setup

- Torch artifact branch: `ah/comms-collectives`
- Deeptools branch: `ah/comms-collectives`
- Deeptools debug commit: `3562e70bf446a3c3791bc599dedd81d04c2cd6f5`
- CLC was reserved for a separate Claude agent and was not used.
- CDX was used for DXP replay/build validation.
- DEV was used for AIU runtime validation.

Important runtime flags:

```bash
SPYRE_LX_PLANNING=1
SPYRE_LX_PLANNER_RELAYOUT=1
SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1
SPYRE_LX_PLANNER_RELAYOUT_LAYOUT_ALLGATHER_RESTICKIFY=1
SPYRE_LX_PLANNER_RELAYOUT_MATMUL_OPERAND_CONTRACT=1
SPYRE_LX_PLANNER_RELAYOUT_RESTICKIFY_OUTPUTS=1
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_CHUNKS=1
DXP_LX_FRAC_AVAIL=0
DXP_BACKEND_LX_FRAC_AVAIL=1
```

`DXP_BACKEND_LX_FRAC_AVAIL=1` is passed through the split wrapper so the Torch-side planner can use full LX while the DXP subprocess still has backend chunk space.

## CDX DXP replay result

Artifact: `cdx_chunk1_replay/summary.json`

The copied Granite attention bundle replay passes DXP with `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_MAX_CHUNKS=1`.

- return code: `0`
- backend plans: `2`
- `8_batchmatmul`: `lowered_gather_then_restickify`, `debug_max_chunks=1`
- `16_batchmatmul`: `lowered_gather_then_restickify`, `debug_max_chunks=1`

This proves the capped staged gather/restickify plan can be imported/lowered by DXP.

## DEV environment repair

Artifact: `dev_deeptools_rebuild/build.log`

DEV's existing build tree wanted to rerun CMake but the local LLVM source checkout was incomplete. The environment was repaired by copying source metadata from CDX:

- `/home/adnan-cdx/dt-inductor-mixed/llvm-project/llvm/cmake` -> `/home/adnan/dt-inductor/llvm-project/llvm/cmake`
- `/home/adnan-cdx/dt-inductor-mixed/llvm-project/llvm/include` -> `/home/adnan/dt-inductor/llvm-project/llvm/include`
- `/home/adnan-cdx/dt-inductor-mixed/llvm-project/mlir/include` -> `/home/adnan/dt-inductor/llvm-project/mlir/include`

After that, DEV rebuilt `dxp_standalone` successfully against DEV's own build libraries. A direct copy of the CDX executable was rejected as invalid because it loaded DEV's installed Deeptools libraries and segfaulted on the first RMS norm bundle.

## DEV AIU result

Artifact: `dev_aiu_chunk1_timeout/run_summary.json`

The full Granite `B=1,S=512,H=4096` causal prefill smoke still timed out with chunking capped to the first staged gather/restickify chunk per relayout site.

- return code: `124` from the 420s timeout wrapper
- backend plans: `2`
- both plans present and realized as `gather_then_restickify`
- both plans have `debug_max_chunks=1`
- runtime signature includes `RuntimeStream::synchronize() still waiting after 60000ms: in_flight_=1 device=0` and later timeout termination in `RuntimeScheduler::issueBarrier`

## Interpretation

This narrows the issue substantially. The bug is not just that the full all-gather/restickify plan is too large. Even the first staged chunk is sufficient to reproduce the runtime completion failure in the full Granite execution path.

Most likely remaining gap: the generated data-op/compute schedule or barrier interaction for the staged `STCDPOpLx -> ReStickifyOpLx -> consumer matmul operand` path is invalid for runtime, even though DXP replay accepts the bundle.

Next debugging should focus on a smaller AIU reproducer around one attention matmul-operand broadcast site, not another full Granite sweep.
