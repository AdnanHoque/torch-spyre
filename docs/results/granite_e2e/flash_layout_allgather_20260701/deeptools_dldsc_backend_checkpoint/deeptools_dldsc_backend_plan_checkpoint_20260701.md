# Deeptools DLDSC Backend Plan Checkpoint - 2026-07-01

Branch: `Adnan-Hoque1/deeptools:ah/comms-collectives`
Head: `6d7a720f5fddeae1d416a1f19877069fcc0a4bb8`
Base: `0a9da5eb19d08712383312bb7dec18fbd7caf711`

## What This Checkpoint Adds

This checkpoint keeps the flash attention layout-all-gather/restickify edge in the DLDSC contract lane. It does not claim physical lowering is complete.

The backend now preserves and recognizes Torch-emitted `lxRelayoutClassifications_` metadata, validates the flash `layout_allgather_restickify` contract, and emits a deterministic backend-plan artifact at the DXP relayout mutation point. The plan records:

- communication class: grouped all-gather with layout restickify
- 4 groups
- 8 producer chunks per group
- 8 consumer cores per group
- 256 logical transfers
- target consumer operand: `batchmatmul.KERNEL`

The checkpoint intentionally marks the edge `realized=false` and skips the old generic 1:1 relayout path so we do not silently miscompile the flash all-gather as scatter.

## Validation Reported By Worker

- `git diff --check`: pass
- `cmake --build build-codex-util --target util_unit_test -j16`: pass
- `./build-codex-util/util/util_unit_test --gtest_filter=LayoutAllgatherRestickify.*`: pass, 10 tests
- `cmake --build build-codex-util --target dsc_unit_test -j16`: pass
- full `dxp_standalone` build: still blocked/slow in heavy MLIR/LLVM external configure/build

## Remaining Backend Work

1. Generate grouped `STCDPOpLx` movement for the 256 logical transfers.
2. Allocate/bind the post-restickify LX KERNEL view.
3. Patch the consumer `batchmatmul` input `LabeledDs`/allocation coordinates to consume the new LX view.
4. Schedule movement before consumer compute.
5. Re-run the H=4 flash bundle and then the full `test_flash.py` shape.

## Files

See `deeptools_ah_comms_collectives.diffstat.txt`, `deeptools_ah_comms_collectives_commits.txt`, and `deeptools_ah_comms_collectives.patch` in this directory.

## 2026-07-01 Update

The backend plan checker now accepts both `ReStickifyOpHBM` metadata from older staged artifacts and `ReStickifyOpLx` metadata from the latest Torch probe. It still normalizes the physical plan to `ReStickifyOpLx`. Focused `LayoutAllgatherRestickify.*` unit tests pass with 11 tests.

