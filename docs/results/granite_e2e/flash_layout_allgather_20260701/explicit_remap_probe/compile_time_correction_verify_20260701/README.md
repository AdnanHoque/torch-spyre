# Explicit Remap Compile-Time Correction Verify

Archived run: 2026-07-01 `explicit_remap_ctc_verify_20260701_095803`

Source pod: `adnan-clc-spyre-dev-pf`

Source workspace: `/home/adnan/codex-isolated/explicit_range_real_attention_20260701_015212`

Source bundle:
`runs/flash_explicit_layout_restickify_20260701_062020/concrete_layout_allgather_range_probe_20260701_083445/dxp_bundle_replay_senulator/bundle_input`

## Result

This verifies the copied-bundle CLC explicit-remap senulator replay in two modes.
The default senulator replay fails during runtime program-correction scatter
generation. The same copied bundle passes when replayed with
`DXP_ENABLE_COMPILE_TIME_CORRECTION=1`.

| Run | Key setting | Result | Evidence |
| --- | --- | --- | --- |
| `default` | `DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1` | `rc=134` | `dxp_replay.stderr` reports `DtException: skv.second <= layoutSize` from `transfer_compute.cpp line 639`. `explicit_range_diag.txt` records `parsedRangeCount: 256`, `dtTableCount: 256`, then `pieceVerificationFailure dataOpDsc=ProgCorrectionScatter0 op=ScatterOpHBM lds=ProgCorrectionFlit piece=p0 dim=d1 pieceSize=1 layoutSize=0`. |
| `compile_time_correction` | `DEEPTOOLS_EXPLICIT_LX_RANGE_PROTO=1`, `DXP_ENABLE_COMPILE_TIME_CORRECTION=1` | `rc=0` | `dxp_replay.stderr` is empty. `explicit_range_diag.txt` records the same `parsedRangeCount: 256` and `dtTableCount: 256` without the default run's `ProgCorrectionScatter0` failure. |

## Interpretation

`DXP_ENABLE_COMPILE_TIME_CORRECTION=1` validates this copied-bundle replay by
avoiding zero-flit runtime program-correction scatter generation. This is not a
full production flash explicit-remap signoff: the replay is a prototype copied
bundle, and the production path still needs to bind the LX-resident `KERNEL`
view into downstream batchmatmul and remove or replace the original
`ReStickifyOpHBM` materialization.

Smallest backend follow-up identified by the run: skip runtime correction SDSC
generation when the correction flit count is zero and there are no unresolved
correction symbols, or make `createDataDscProgCorrection` emit a no-op for zero
correction flits instead of forcing a one-element `ScatterOpHBM` piece on a
zero-sized `ProgCorrectionFlit` LDS dimension.

## Archived Files

- `verification_summary.txt`: original run summary.
- `default/command.txt`: default replay command.
- `default/dxp_replay.result`: default replay return code.
- `default/dxp_replay.stdout`: default replay stdout.
- `default/dxp_replay.stderr`: default replay stderr.
- `default/explicit_range_diag.txt`: default explicit range diagnostic dump.
- `compile_time_correction/command.txt`: compile-time-correction replay command.
- `compile_time_correction/dxp_replay.result`: compile-time-correction return code.
- `compile_time_correction/dxp_replay.stdout`: compile-time-correction stdout.
- `compile_time_correction/dxp_replay.stderr`: compile-time-correction stderr.
- `compile_time_correction/explicit_range_diag.txt`: compile-time-correction diagnostic dump.
