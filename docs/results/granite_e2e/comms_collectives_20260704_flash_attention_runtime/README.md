# Flash Attention Runtime Probe - DLDSC Collectives - 2026-07-04

This archive captures the `test_flash.py` probe from `adnan-cdx-spyre-dev-pf` using the current `ah/comms-collectives` Torch and Deeptools branches.

## Result

| Variant | Runtime status | SDSC files | HBM restickify SDSCs | LX restickify SDSCs | Backend plans |
|---|---|---:|---:|---:|---:|
| relayout off | runtime_success | 550 | 32 | 0 | 0 |
| relayout on | runtime_success | 550 | 0 | 32 | 32 |

The optimized run replaces **32** HBM restickify SDSCs with **32** LX restickify SDSCs and emits **32** backend matmul operand broadcast plans. Both variants reached runtime success under the compile-probe path.

## Why This Matters

This is the clearest attention-side evidence for the DLDSC collectives substrate: the before bundle hides the handoff through HBM restickify rows; the after bundle represents the same class as LX restickify plus backend-synthesized loop-scoped kernel-neighbor movement.

## Included Artifacts

- `before_relayout_off/summary.json` and `after_relayout_on/summary.json`
- `restickify_inventory.csv` and `restickify_inventory.json`
- `sdsc_samples/`: one representative before HBM restickify SDSC and one representative after LX restickify SDSC
- `after_relayout_on/backend_plans/`: all emitted backend plan files
- command/env/stdout/stderr tails for both runs

## Source Run Root

`/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507`
