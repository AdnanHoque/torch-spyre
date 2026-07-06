# Flash Current `ah/comms-collectives` Status

Date: 2026-07-06

This records a fresh CDX rerun of the flash compile/lowering probe using the current Deeptools `ah/comms-collectives` head after the DLDSC extent fix.

This is not a value-correctness claim. The run uses the existing compile-probe patch that skips host-to-device copies and skips the CPU reference assertion so we can isolate Torch/DXP lowering behavior.

## Branches

```text
Torch root: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/torch-spyre
Torch SHA: b84528d7e32ad0aea5f31d7de107344b35617695

Deeptools root: /home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/deeptools
Deeptools SHA: 3095cdb33
Deeptools branch: ah/comms-collectives
```

## Runs

Default current run:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_current_ah_comms_collectives_20260706_164837
```

Diagnostic run with `DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1`:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_current_ah_comms_collectives_diag_20260706_165024
```

## Result

Both runs fail before successful DXP lowering.

| Run | rc | SDSC files | ReStickifyOpHBM string hits | ReStickifyOpLx string hits | Backend plan files |
|---|---:|---:|---:|---:|---:|
| default | 1 | 549 | 0 | 160 | 0 |
| diagnostic | 1 | 549 | 0 | 160 | 0 |

The structural frontend side is still doing the expected thing: no HBM restickify rows appear and LX restickify rows are generated. The failure is in backend realization/scheduling before plan artifacts are emitted.

## Failure Modes

Default run fails on a backend scheduling guard:

```text
DtException: Do not support double buffering and input-neighbor fetch coexisting in the same DSC.
Set DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1 only for diagnostic probes.
file .../dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1700
```

Diagnostic run gets past that guard, then fails in fold solving:

```text
[fillDataInfo diagnostic] exception=DtException: Solution cannot be found
processor=transfer_lds1_src:lxlu_dst:ptrow0
allocation=allocate_lds1_ptxrf
loop=loop_ds2_ds3_in
dataConnect=xrf_kernel
phase=coordinate_offsets
coordCoreMap=32

DtException: Solution cannot be found
file .../dsc/foldManager/foldInfrastructure.h line 2983
```

## Interpretation

The older clean `gather-restickify` split branch had a successful flash compile/lowering probe:

```text
ReStickifyOpHBM: 0
ReStickifyOpLx: 32 top-level rows
backend plans: 32
kind: matmul_operand_broadcast
communication_pattern: all_gather_replicate
realization_strategy: gather_then_restickify
```

The current prototype branch regressed for flash because the newer kernel-neighbor / input-fetch path collides with double buffering and then fails fold solving even under the diagnostic escape hatch. This is the next backend gap for flash attention.

Granite S512 is not blocked by this specific flash failure: Granite causal prefill successfully replaces the in-scope attention activation/layout HBM spill with `ReStickifyOpLx` on the current branch.

## Next Work

1. Decide whether flash matmul operand all-gather should use the older `gather_then_restickify` realization instead of the newer kernel-neighbor/input-fetch route for this shape.
2. If kernel-neighbor remains the intended carrier, define a production scheduling rule for coexistence with double buffering instead of relying on the diagnostic env.
3. Fix the fold-solving failure for `lxlu -> ptrow0` coordinate offsets after the diagnostic guard is bypassed.
4. Re-run the compile probe and archive backend plan files once DXP lowering succeeds.

