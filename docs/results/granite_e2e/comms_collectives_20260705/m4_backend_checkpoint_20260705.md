# M4 Backend Collectives Checkpoint - 2026-07-05

This note records the latest DLDSC collectives progress across the three AIU pods. It is an artifact-branch checkpoint for `ah/comms-collectives`, not a production design note.

## Summary

The useful current state is:

- Granite S512 has a previously passing run where all in-scope activation HBM root restickifies were removed. The remaining HBM root restickifies are weight/prelayout rows and are out of scope for this lane.
- The current all-collectives Granite path still aborts in Deeptools on the attention RHS `matmul_operand_broadcast` edge with an LX chunk-fit failure.
- The flash `test_flash.py` compile/runtime probe still structurally removes explicit HBM restickifies, but it skips correctness with `PATCH_MODE=no_h2d,skip_cpu_ref`.
- The CDX synthetic M4 RHS operand all-gather probe isolates the backend blocker to DDC fold/coordinate handling for artificial LX-neighbor transfers. Ring traffic alone is not enough for value correctness.

## CLC Granite S512 Status

Read-only inspection was done on:

```text
pod: adnan-clc-spyre-dev-pf
root: /home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404
```

Passing marker run:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_relayout_backend02_20260704_035559
```

Counts:

| Metric | Count |
|---|---:|
| `ReStickifyOpHBM` root SDSC rows | 4 |
| raw `ReStickifyOpHBM` occurrences | 12 |
| in-scope activation HBM remaining | 0 |
| out-of-scope weight/prelayout HBM rows | 4 |
| in-scope activation `ReStickifyOpLx` root rows | 1 |
| raw `ReStickifyOpLx` occurrences | 3 |

The four remaining HBM restickifies are weight/prelayout rows:

```text
sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_j3z0ehfl/sdsc_0.json
sdsc_fused__scaled_dot_product_fused_attention_overrideable_add_linear_mul_rms_norm_silu_split_with_sizes_transpose_view_2_j3z0ehfl/sdsc_10.json
sdsc_fused_add_linear_mul_3_9v_l7803/sdsc_0.json
sdsc_fused_linear_rms_norm_0_cou1jp4i/sdsc_7.json
```

Current July 5 enabled archive:

```text
/home/adnan/codex-isolated/granite_s512_comms_collectives_20260704_033404/runs/granite_s512_collectives_archive_20260705_081840/relayout_enabled_collectives_backend1
```

Current enabled-run status:

```text
returncode.txt = 1
DXP replay return code = 134
```

Failure signature:

```text
DtException: Unable to map graph within architecture constraints:
The initial chunk parameters must fit in LX for SuperDSC: 8_batchmatmul
```

The generated in-scope plan is:

```text
backend_plans/8_batchmatmul_Tensor1_0_matmul_operand_broadcast_plan.json
kind = matmul_operand_broadcast
communication_pattern = all_gather_replicate
```

Interpretation: the frontend/DLDSC side is classifying the right edge, but the backend realization for the attention RHS all-gather plus layout conversion is not yet robust.

## DEV Flash `test_flash.py` Probe

Fresh rerun was done on:

```text
pod: adnan-spyre-dev-pf
root: /home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129
fresh run: /home/adnan/codex-isolated/flash_attention_devpf_verify_20260705_070129/runs/relayout_on_verify_backend1_20260705_174127
script: /tmp/test-spyre-scripts/test_flash.py
script sha1: 6facf47e7c66198c44c352e5c809955c5d558afe
```

Counts:

| Run | rc | SDSCs | Backend plans | Matmul operand plans | `ReStickifyOpHBM` | `ReStickifyOpLx` | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| archived `075332` | 0 | 550 | 32 | 32 | 0 files / 0 occurrences | 33 files / 97 occurrences | `SUCCESS` |
| fresh `174127` | 0 | 550 | 32 | 32 | 0 files / 0 occurrences | 33 files / 97 occurrences | `SUCCESS` |

Strictly over `sdsc_*.json`, the LX count is 32 files / 96 occurrences; the artifact-wide cache count is 33 / 97 because one generated cache Python file also contains the string.

What this proves:

- the patched flash compile/runtime probe can generate and run with zero explicit `ReStickifyOpHBM`;
- it emits 32 backend relayout plans and 32 matmul operand broadcast plans;
- `ReStickifyOpLx` replaces the explicit HBM restickify rows;
- no fatal DXP/runtime failure was observed in the patched probe.

What this does not prove:

- value correctness;
- H2D correctness;
- an unpatched full `test_flash.py` result.

The run uses:

```text
PATCH_MODE=no_h2d,skip_cpu_ref
```

and prints:

```text
[runtime_patch] assert_close skipped for compile probe
```

Both archived and fresh runs include a runtime stream warning that waits for roughly 60 seconds before completing. The final return code remains 0, so this is a warning to track, not the immediate blocker.

## CDX Synthetic M4 Backend Diagnostic

Active CDX workspace:

```text
pod: adnan-cdx-spyre-dev-pf
root: /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
run root: /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/runs/min_stable_matmul_operand_broadcast_20260704_100506
```

Known-good archived synthetic control:

```text
kernel_neighbor_carousel_M4_212148
ALLCLOSE True
MAX_DIFF 0.0
MISMATCH 0 / 1024
```

Current diagnostic edits:

- added a preference for finalized RHS broadcast transfer candidates over pre-finalize template transfers in `L3DlOpsScheduler.cpp`;
- rebuilt `dxp_standalone`;
- reran M4 with `DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1`.

Current run without fold skip:

```text
single_stage_prefer_finalized_M4_174304
```

Result:

```text
terminate called after throwing an instance of 'std::out_of_range'
what(): map::at
```

GDB backtrace:

```text
ddc::Ddc::buildFoldForTransfer
ddc::Ddc::buildFoldFromAllocation
```

The failing transfer shape is:

```text
transfer_lds1_src:lxlu_dst:ptrow*
```

Current run with fold skip:

```text
single_stage_foldskip_M4_174558
```

Result:

```text
ALLCLOSE False
MAX_DIFF 0.7490234375
MISMATCH 949 / 1024
ROWMAP_OUT0 [0.0, 0.0, 0.0, 0.75]
ROWMAP_REF0 [0.0, 0.25, 0.5, 0.75]
```

Interpretation:

- DDC fold construction is currently crashing on artificial LX-neighbor PT-row transfers because the fold/coordinate metadata is incomplete.
- Simply skipping that fold lets the program run, but it is value-wrong. The fold is not just optional metadata; it is needed to make the PT consumer read the correct logical row/chunk placement.
- Ring movement alone is not sufficient. The backend must either construct valid coordinates/folds for the artificial transfer or lower the local conversion through a real backend-owned carrier that already has valid coordinate/fold semantics.

## Next Backend Hypothesis

The current free-standing LX-neighbor transfer path is too fragile unless it carries full DDC coordinate metadata. The next useful implementation step is:

1. Either populate valid coordinate/fold metadata for the artificial `transfer_lds1_src:lxlu_dst:ptrow*` nodes so DDC can build folds without `map::at`;
2. or stop representing the local leg as a synthetic transfer and route the local layout conversion through an existing `ReStickifyOpLx`/`STCDPOpLx`-style carrier;
3. keep the ring all-gather staging loop-scoped, because full resident materialization is too large for Granite attention.

Acceptance for the next probe:

```text
M4: ALLCLOSE True, MISMATCH 0 / 1024
M16/M64: same row-pattern correctness
Flash: rerun without correctness skip
Granite S512: rerun with trace-derived kernel_ms_per_iter and no in-scope activation HBM rows
```

## Commands Worth Reusing

Build CDX DXP:

```bash
/home/adnan-cdx/dt-inductor-mixed/.venv/bin/cmake \
  --build /home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507/build-deeptools-comms-clean-fast \
  --target dxp_standalone \
  -j8
```

Run M4 synthetic probe:

```bash
ROOT=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
RUN=$ROOT/runs/min_stable_matmul_operand_broadcast_20260704_100506
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
export DEEPTOOLS_LX_NEIGHBOR_FORCE_VIEW_GUARD_SKIP=1
export DEEPTOOLS_LX_NEIGHBOR_FOLD_GUARD_DIAGNOSTIC=1
bash "$RUN/run_one_64diag.sh" "case_name" mul_one 1 4 64 256
```

Run DXP under gdb:

```bash
ROOT=/home/adnan-cdx/codex-isolated/flash_attention_verify_comms_20260704_033507
REAL=$ROOT/build-deeptools-comms-clean-fast/dxp/dxp_standalone
RUN=$ROOT/runs/min_stable_matmul_operand_broadcast_20260704_100506/single_stage_prefer_finalized_M4_174304
BUNDLE=$(find "$RUN/cache/inductor-spyre" -maxdepth 2 -name bundle.mlir -print -quit | xargs dirname)
export DXP_LX_FRAC_AVAIL=1
export DEEPTOOLS_MATMUL_OPERAND_BROADCAST_KERNEL_NEIGHBOR=1
export DEEPTOOLS_ALLOW_MIXED_HBM_IFN_DIAGNOSTIC=1
gdb -batch -ex run -ex "bt 50" --args "$REAL" --bundle -d "$BUNDLE"
```
