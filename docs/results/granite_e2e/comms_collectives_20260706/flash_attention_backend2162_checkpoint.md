# Flash Attention DLDSC Collectives Checkpoint - 2026-07-06

This records the standalone flash attention verification on `adnan-spyre-dev-pf` using the current `ah/comms-collectives` Torch/Deeptools branches.

## Source State

- Torch branch: `AdnanHoque/torch-spyre ah/comms-collectives`
- Torch checkout SHA: `db16aab3021a355482e13333b99f84d6882a1706`
- Deeptools branch: `Adnan-Hoque1/deeptools ah/comms-collectives`
- Deeptools checkout SHA: `2162efb3e09dd8c2ba3e6c8cd29139c7c5d8abe8`
- Flash script checkout: `aviros/test-spyre-scripts main @ afda166e58b23519d0b4ca871350b011b56d91a3`
- Pod: `adnan-spyre-dev-pf`
- Run root: `/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751`

## Baseline Relayout-Off Run

Run:

```text
/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/runs/baseline_relayout_off_20260706_005811
```

Result:

| metric | value |
|---|---:|
| return code | 0 |
| SDSC files | 550 |
| backend plan files | 0 |
| `ReStickifyOpHBM` files | 33 |
| `ReStickifyOpHBM` occurrences | 97 |
| `ReStickifyOpLx` files | 0 |
| `ReStickifyOpLx` occurrences | 0 |

The baseline compile probe completed successfully. The harness used the compile-probe runtime patch, so value comparison was intentionally skipped:

```text
[runtime_patch] assert_close skipped for compile probe | SUCCESS
```

## Relayout-On Run

Run:

```text
/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/runs/relayout_on_20260706_010236
```

Result:

| metric | value |
|---|---:|
| return code | 1 |
| SDSC files | 549 |
| backend plan files | 32 |
| plan kind | `matmul_operand_broadcast` x 32 |
| communication pattern | `all_gather_replicate` x 32 |
| `ReStickifyOpHBM` files | 0 |
| `ReStickifyOpHBM` occurrences | 0 |
| `ReStickifyOpLx` files | 97 |
| `ReStickifyOpLx` occurrences | 193 |

The structural result is the expected direction: the flash bundle no longer contains explicit HBM restickify rows for these handoffs, and Deeptools emits 32 all-gather/replicate matmul-operand plans.

The run does not yet pass end-to-end. DXP/DDC aborts during fold propagation:

```text
DtException: [buildFoldFromAllocation] Can not propagate coordinates for coreletSplit dimensionout from allocateNode allocate_lds1_lx with custom coreIdToWkSlice.
```

Source location reported by the run:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/deeptools/ddc/ddc_fold.cpp line 3209
```

## Interpretation

This is not a device wedge and not a missing frontend classification. The Torch side emits the intended LX relayout structure for the flash workload, and Deeptools creates the expected 32 `all_gather_replicate` backend plans.

The current blocker is a Deeptools DDC folding gap: an internal LX allocation with custom `coreIdToWkSlice` cannot propagate coordinates for the `out` corelet split dimension. This is the next backend/runtime issue to debug before claiming flash correctness or performance.

## What This Proves

- The relayout-on flash bundle structurally removes the HBM restickify rows that appear in the relayout-off baseline.
- The current backend prototype recognizes the flash matmul operand handoffs as `matmul_operand_broadcast` / `all_gather_replicate`.
- The path is not yet value-correct or performance-measured for this standalone flash script because DDC aborts before a completed run.

## Next Debug Step

Minimize one failing SDSC from:

```text
/home/adnan/codex-isolated/flash_attention_comms_backend2162_20260706_005751/runs/relayout_on_20260706_010236/cache/inductor-spyre/sdsc_fused_add_amax_exp_maximum_mul_sub_sum_transpose_unsqueeze_1_zoxlnzj7
```

Then decide whether the right fix is:

1. Teach `buildFoldFromAllocation` how to propagate coordinates through this custom `coreIdToWkSlice` allocation, or
2. Avoid emitting the problematic custom allocation for this specific internal fold path.

Until that is fixed, treat flash as structurally promising but not complete.
