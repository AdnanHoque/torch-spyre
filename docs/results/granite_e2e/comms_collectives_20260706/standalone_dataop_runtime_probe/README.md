# Standalone DataOp Runtime Probe - 2026-07-06

## Question
Can the `matmul_operand_broadcast` gather/restickify path avoid mixed data-op rows by emitting a standalone DataOp-only SuperDSC before the attention consumer?

## Result
DXP replay now passes, but AIU runtime still loses completion.

The key Deeptools fix for replay was to mirror the generic `LxRelayout` timeline behavior: when a standalone relayout SuperDSC is inserted, call `memTrackers->insertPsBefore(ps)` before LX scratch allocations. Without that, program-correction allocation saw inconsistent timeline state and produced an invalid HBM segment address or failed allocation.

## Evidence

| Probe | Result | Meaning |
| --- | --- | --- |
| `replay_gather_only_minimal` | rc=0 | Standalone DataOp-only `STCDPOpLx` replay succeeds with the timeline fix. |
| `replay_gather_plus_restickify_minimal` | rc=0 | Standalone gather plus local `ReStickifyOpLx` replay succeeds with the same fix. |
| `aiu_full_chunk1_timeout` | rc=124 | Hardware reaches attention `run()` return, then hangs at final `torch.accelerator.synchronize()`. |
| `aiu_gather_only_chunk1_timeout` | rc=124 | Same lost-completion behavior with only the gather `STCDPOpLx`, so `ReStickifyOpLx` is not required for the runtime failure. |

## Current Read
The standalone DataOp carrier is structurally accepted by DXP after the timeline fix, but is not yet a working hardware carrier for the attention relayout. This keeps pointing us away from both mixed data-op rows and pure DataOp-only standalone rows as production carriers for this path.

The remaining direction is to either:

1. confirm/fix runtime support for standalone DataOp-only SuperDSCs, or
2. express the communication through DLDSC/KTIR-compatible relayout constructs so the backend owns physical movement without data-dsc program rows.

## Local Patch
`local_dev_standalone_dataop_insertps_minimal.diff` records the DEV-local Deeptools patch that makes standalone DataOp replay pass.
