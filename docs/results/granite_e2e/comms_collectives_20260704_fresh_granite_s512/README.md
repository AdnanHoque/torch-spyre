# Fresh Granite S512 DLDSC Collectives Checkpoint - 2026-07-04

This folder archives a fresh profiled Granite block prefill run from `adnan-spyre-dev-pf` on the current `ah/comms-collectives` branch.

## Result

| Variant | kernel_ms_per_iter | wall median ms | return code |
|---|---:|---:|---:|
| disabled control | 14.7257902 | 27.6074409 | 0 |
| enabled DLDSC relayout | 13.8212856 | 26.5204906 | 0 |

Kernel speedup: **1.0654x** (6.14% improvement).

Wall speedup: **1.0410x** (3.94% improvement).

## What This Proves

The current Torch and Deeptools collectives branches still produce an end-to-end profiled Granite S512 win. The enabled run emits backend `matmul_operand_broadcast` plans lowered as loop-scoped kernel-neighbor movement.

## Run Roots

- Disabled control: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_disabled_control_20260704_035333`
- Enabled relayout: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_relayout_backend02_20260704_035559`
- Workspace: `/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404`

## Included Artifacts

- `disabled_control/`: command, env, return code, summary, result, and trace summary
- `enabled_relayout/`: command, env, return code, summary, result, and trace summary
- `backend_plans/`: emitted backend relayout plans for the enabled run
- `summary.json`: machine-readable comparison
