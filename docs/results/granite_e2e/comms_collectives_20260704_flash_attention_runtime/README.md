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


## Value-Correctness Follow-up

After the compile-probe archive above, we reran `test_flash.py` without skipping the CPU reference. This used Deeptools commit `ebb662cb7`, which derives `matmul_operand_broadcast` groups from explicit producer/consumer coordinates.

| Variant | Value status | First-edge logical groups | Mismatch |
|---|---|---|---:|
| loop-scoped kernel-neighbor marker | fail | corrected to interleaved groups such as `0 -> {0,4,8,...,28}` | 31.5% |
| unsafe explicit STCDP materialization | fail | corrected to interleaved groups such as `0 -> {0,4,8,...,28}` | 90.7% |

This isolates the current blocker. The frontend contract and logical backend plan now identify the correct all-gather/replicate groups for the first flash edge, but the physical lowering still does not use those exact source/destination groups when feeding the matmul RHS. The explicit STCDP materialization path is also not equivalent for this loop-scoped matmul operand.

Artifacts are in `value_correctness_followup/`:

- `summary.json`
- `marker_kernel_neighbor/3_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- `unsafe_explicit_stcdp/3_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json`
- full stdout/stderr/env for both value-correctness attempts

## Source Run Root

`/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507`
