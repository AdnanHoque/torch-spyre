# Flash Attention `105_batchmatmul` Matmul-Operand Broadcast Experiments

Date: 2026-07-03  
Pod: `adnan-cdx-spyre-dev-pf`  
Torch workspace: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/torch-spyre`  
Deeptools workspace: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/deeptools`  
Test: `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/test-spyre-scripts/test_flash.py`

## What This Is Testing

This isolates the attention edge where `sdsc_105.json` / `105_batchmatmul` consumes Tensor1 as a KERNEL operand. The producer-side LX tensor distribution is split over `(out, in)` while the consumer matmul compute is split over `(x, mb)`. That is not PR1 scatter. It is an all-gather/broadcast style contract: every consumer core needs all producer chunks for this operand.

The Torch-side metadata emits:

- `kind = matmul_operand_broadcast`
- `communication_pattern = all_gather_replicate`
- `realization_strategy = loop_scoped_input_fetch`
- `logical_transfer_count = 1024`
- `source_core_count = 32`
- `consumer_core_count = 32`

## Runs

All runs used the split frontend/backend LX knob setup:

- Torch sees `DXP_LX_FRAC_AVAIL=0`
- DXP subprocess sees `DXP_BACKEND_LX_FRAC_AVAIL=1` through the local wrapper

| Variant | CDX run directory | DXP/runtime result | Correctness result | Interpretation |
| --- | --- | --- | --- | --- |
| `phasefix` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_fission4_phasefix_20260703_075649` | DXP and runtime complete | `15,589,744 / 16,777,216` mismatched (`92.9%`), max diff `inf` | Phase/cardinality fixes are enough to run, but physical movement is value-wrong. |
| `dstoffset` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_fission4_dstoffset_20260703_080553` | DXP and runtime complete | `14,908,697 / 16,777,216` mismatched (`88.9%`), max diff `inf` | Adding destination shard offsets helps slightly, so overwrite/address placement is part of the bug, but not sufficient. |
| `srcdstoffset` | `/home/adnan-cdx/codex-isolated/dldsc_flash_runtime_lrfimm_20260702_145525/runs/test_flash_direct_matmul_operand_only105_fission4_srcdstoffset_20260703_081110` | DXP and runtime complete | `15,619,793 / 16,777,216` mismatched (`93.1%`), max diff `inf` | Source shards are likely already addressed as per-core local shards; applying source offsets makes the result worse. |

## Artifact Files

Each variant directory contains:

- `run_command.sh`: exact run command and environment.
- `105_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`: backend plan artifact with the 1024 logical source/destination core pairs.

`cdx_deeptools_current_experimental_diff.patch` records the CDX experimental backend diff at the best current diagnostic state. This is not production-clean; it includes investigation hooks and filters.

## Current Read

The matmul-operand broadcast contract is correctly identified at the Torch/DLDSC level, and the backend can now compile and run the isolated edge with fissioned movement SDSCs. The remaining failure is semantic: a 32-source to 32-destination operand all-gather cannot be represented as only `(source_core, destination_core)` transfers. The physical lowering also needs the logical destination slot for each producer chunk, and likely needs the matmul operand reader to consume the staged chunks with the same logical coordinates.

This is why PR1 scatter is not enough for this attention spill. Scatter is one producer shard to one consumer shard. This edge is all-gather/replicate: every consumer core needs the full producer-sharded operand.

## Next Backend/Frontend Gap

The next clean implementation should make `matmul_operand_broadcast` a first-class backend-lowered communication class:

1. Preserve the DLDSC coordinate contract from Torch: producer tensor distribution, consumer compute distribution, operand index, communication kind.
2. Backend derives source chunk coordinates and destination slots, not just source/destination core pairs.
3. Backend materializes either a full resident replicated operand or a loop-scoped input fetch that feeds the matmul reader without overwriting chunks.
4. Add a small value-pattern unit test before re-running full flash attention.

