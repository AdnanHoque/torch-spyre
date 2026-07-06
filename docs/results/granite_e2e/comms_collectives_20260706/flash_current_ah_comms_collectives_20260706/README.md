# Flash Current `ah/comms-collectives` Status

Date: 2026-07-06

This records fresh CDX reruns of the flash compile/lowering probe using the current Deeptools `ah/comms-collectives` head after the DLDSC extent fix.

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

Gather/restickify carrier only:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_current_ah_comms_collectives_gather_only_20260706_165643
```

Kernel-neighbor carrier enabled:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_current_ah_comms_collectives_20260706_164837
```

Kernel-neighbor carrier with `DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1`:

```text
/home/adnan-cdx/codex-isolated/gather_restickify_clean_20260706_113236/runs/test_flash_current_ah_comms_collectives_diag_20260706_165024
```

## Result

The current branch succeeds when flash uses the `gather_then_restickify` carrier. The experimental kernel-neighbor/input-fetch carrier still fails on this flash shape.

| Run | rc | SDSC files | ReStickifyOpHBM string hits | ReStickifyOpLx string hits | Backend plan files |
|---|---:|---:|---:|---:|---:|
| gather/restickify carrier only | 0 | 550 | 0 | 160 | 32 |
| kernel-neighbor carrier | 1 | 549 | 0 | 160 | 0 |
| kernel-neighbor diagnostic | 1 | 549 | 0 | 160 | 0 |

The successful gather/restickify run emits:

```text
realization_strategy: gather_then_restickify
communication_pattern: all_gather_replicate
backend plans: 32
logical_transfer_count per plan: 256
stdout: SUCCESS
```

This means the current branch can still remove the flash activation HBM restickifies structurally, as long as the carrier selection stays on the staged gather/restickify path.

## Failure Modes

The kernel-neighbor carrier fails on a backend scheduling guard:

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

The older clean `gather-restickify` split branch and the current `ah/comms-collectives` branch agree on the important structural result when the carrier is gather/restickify:

```text
ReStickifyOpHBM: 0
backend plans: 32
kind: matmul_operand_broadcast
communication_pattern: all_gather_replicate
realization_strategy: gather_then_restickify
```

The current prototype branch only regresses for flash when the newer kernel-neighbor/input-fetch path is forced. That path collides with double buffering and then fails fold solving even under the diagnostic escape hatch. This is the next backend gap for the more aggressive carrier, not for the staged gather/restickify carrier.

Granite S512 is not blocked by this specific flash failure: Granite causal prefill successfully replaces the in-scope attention activation/layout HBM spill with `ReStickifyOpLx` on the current branch.

## Next Work

1. Treat `gather_then_restickify` as the current production candidate carrier for flash all-gather/restickify.
2. Keep kernel-neighbor/input-fetch behind an experimental gate until double-buffer coexistence has a real scheduling rule.
3. Fix the fold-solving failure for `lxlu -> ptrow0` coordinate offsets before using the kernel-neighbor path for this flash shape.
4. Re-run value correctness only after the known baseline zero-stride/broadcast issue is fixed or bypassed.
