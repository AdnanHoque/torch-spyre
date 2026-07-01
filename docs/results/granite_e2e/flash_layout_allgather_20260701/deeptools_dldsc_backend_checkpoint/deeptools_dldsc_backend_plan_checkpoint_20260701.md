# Deeptools DLDSC Backend Checkpoint - 2026-07-01

Branch: Adnan-Hoque1/deeptools ah/comms-collectives
Head: 4ef9b53a5ca8b39cbc3f3bc151e753a593b49a41
Base: 0a9da5eb19d08712383312bb7dec18fbd7caf711

## Current Direction

The flash attention H=4 spill is classified as : an LX-resident KERNEL tensor produced with one coordinate distribution is consumed by  with an incompatible compute distribution. The current Deeptools checkpoint keeps the logical classification explicit, but routes physical realization through the existing generic LX relayout insertion path. That path creates STCDPOpLx data movement from producer tensor coordinates to consumer compute coordinates.

## Key Changes

- Accept staged Torch metadata that uses compact device-dimension keys and explicit producer/consumer core counts.
- Accept  imported either as the current map form or as the list form seen in archived staged SDSCs.
- Keep  as a supported restickify op for this flash all-gather contract.
- Expand the compact plan to 256 logical source/destination core transfers for the H=4 flash case.
- Let validated flash all-gather edges fall through to the generic STCDPOpLx LX relayout insertion path instead of stopping after a diagnostic artifact.

## Validation So Far

- Note: Google Test filter = LayoutAllgatherRestickify.*
[==========] Running 0 tests from 0 test suites.
[==========] 0 tests from 0 test suites ran. (0 ms total)
[  PASSED  ] 0 tests. passes on CDX with 13 focused tests.
- Full DXP/e2e validation is still pending because the CDX DXP build directory is not currently configured with a Makefile.

## Files

- : full patch against current Deeptools master merge-base.
- : diffstat for the patch.
- : commit list on the artifact branch.
