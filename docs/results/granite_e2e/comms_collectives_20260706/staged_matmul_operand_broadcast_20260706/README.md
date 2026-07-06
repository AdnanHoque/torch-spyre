# Staged matmul operand broadcast replay checkpoint - 2026-07-06

## What this checkpoint proves

The flash-attention bundle that previously exposed `matmul_operand_broadcast` / all-gather-like operand movement now passes DXP replay with the staged lowering enabled.

The staged lowering is:

1. Use `STCDPOpLx` to gather producer LX pieces into temporary LX staging pieces on the destination side.
2. Use local `ReStickifyOpLx` to convert the staged source-layout pieces into the consumer matmul KERNEL operand layout.
3. Schedule the movement rows before the consumer batchmatmul rows only on cores that own real input/output pieces for that staged restickify.

This is the safe alternative to direct KERNEL-neighbor movement for layout-changing matmul operands. Direct KERNEL-neighbor copies can remove the structural HBM spill, but they are value-unsafe when the operand also needs stick/layout conversion.

## Evidence

- Deeptools checkout: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools`
- Deeptools branch: `ah/comms-collectives`
- Torch artifact checkout: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/torch-spyre`
- Torch branch: `ah/comms-collectives`
- Replay root: `/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/staged_gather_restickify_replay_clean_20260706_054214`
- DXP replay result: `rc=0`
- Backend plan count: `32`
- Total logical transfers represented across plans: `8192`
- Plan class: `matmul_operand_broadcast`, `all_gather_replicate`
- Realization: `gather_then_restickify`
- Physical lowering status: `lowered_gather_then_restickify`
- Layout conversion required: `true`

Focused tests:

- `./build-deeptools/util/util_unit_test --gtest_filter="LayoutAllgatherRestickify.*"`: 27 passed
- `./build-deeptools/dxp/dxp_unit_test --gtest_filter="DxpTestFixture.CoreWorkDivIncomptLxRelayout*"`: 2 passed

Artifacts in this directory:

- `backend_plans/`: all 32 backend plan JSON artifacts from the passing replay
- `backend_plan_summary.json`: compact machine-readable summary
- `commands.txt`: exact replay command and environment
- `replay.stdout`, `replay.stderr`: raw replay output

## Backend change made in this checkpoint

`dxp/SdscRelayoutInsertion.cpp` now has a gated prototype path behind:

```bash
DEEPTOOLS_ENABLE_MATMUL_OPERAND_GATHER_RESTICKIFY=1
```

The path parses the source LX tensor and target matmul operand tensor contracts from the DLDSC relayout metadata, builds temporary LX pieces, emits one `STCDPOpLx` gather data-op and one `ReStickifyOpLx` data-op per affected SuperDSC, and schedules them before the consumer DL op.

The important correctness fix from this checkpoint is that `ReStickifyOpLx.coreIdsUsed_` and `coreIdToDscSchedule` are now derived from actual staged pieces, not from every logical destination core. DCG requires each scheduled `ReStickifyOpLx` core to have LX input and output pieces; otherwise `apeOp.cpp` fails with an empty `iPieceOrder`.

## What this does not prove yet

This is a DXP/DCG/codegen replay gate, not a full AIU correctness/performance gate.

Remaining work:

- Run the flash script end-to-end on AIU once the independent broadcast/zero-stride correctness issue is not masking results.
- Validate that the staged pieces cover the exact matmul operand values consumed by each core, not merely that DXP can lower the rows.
- Decide whether the final production path should be full pre-materialized staging, loop-scoped staging, or a carousel-style schedule for large Granite attention operands.
- Extend the same discipline to the other communication classes in the Epic: multicast/broadcast, gather, all-gather, reduce, and all-reduce.
