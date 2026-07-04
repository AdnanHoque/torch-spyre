# DLDSC Collectives Backend Kernel-Neighbor Checkpoint - 2026-07-04

This artifact packet records the current `ah/comms-collectives` state after the backend KERNEL-neighbor fixes.

## Source State

- Torch branch: `ah/comms-collectives`
- Torch SHA: `796e0fb1998f599e5b058336335ff290334dfc54`
- Deeptools branch: `ah/comms-collectives`
- Deeptools SHA: `fa36c57166ed4541216c55cf97527f96819d7d5e`
- Pod source: `adnan-clc-spyre-dev-pf`

## Granite Prefill S512

Shape: `B=1, S=512, E=4096`, causal prefill Granite block probe with fake/fused weights.

| Variant | kernel_ms_per_iter | wall/median_ms | memory_ms_per_iter |
|---|---:|---:|---:|
| relayout disabled/control | 14.734058 | 27.222633 | 0.385967 |
| relayout enabled | 13.869813 | 25.944948 | 0.330816 |

Kernel speedup: `1.062x`.
Wall/median speedup: `1.049x`.

Notes:
- This is the current archived useful Granite win, not the older isolated `~1.19x` split-env result.
- A fresh rerun from pushed branches is in progress on another pod.

## Flash Attention DXP Replay

Bundle: `/home/adnan/codex-isolated/dldsc_granite_clean_relayout_20260703_163108/runs/flash_current_collectives_20260704_025212/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_gf_a7qey`

| Field | Value |
|---|---:|
| DXP replay rc | 0 |
| stderr lines | 0 |
| backend plan count | 32 |
| `ReStickifyOpHBM` bundle count | 0 |
| `ReStickifyOpLx` bundle count | 32 |

Representative plan class: `matmul_operand_broadcast` / `all_gather_replicate` / `lowered_loop_scoped_kernel_neighbor`.

## What Changed In Deeptools

Two narrow backend fixes were needed for staged KERNEL-neighbor relayouts:

1. KERNEL-neighbor transfers must participate in LX allocation fit checks. Without this, DXP selected 524 KB chunks where only about 406 KB was free after frontend reservations.
2. KERNEL-neighbor allocation spatial folds must use chunk-stage coordinates, not full core-stage coordinates. Without this, DDC tried to solve a transfer target at offset 256 against a corelet fold starting at offset 2048.

## Current Interpretation

We have advanced from frontend metadata exists to flash attention all-gather-replicate relayout compiles through DXP. The remaining proof step is full AIU runtime/profiler validation for flash and a fresh Granite rerun from the pushed branches.
