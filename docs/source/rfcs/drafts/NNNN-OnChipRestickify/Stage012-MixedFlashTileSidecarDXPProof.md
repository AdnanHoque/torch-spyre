# Stage 012: Mixed Flash Tile Sidecar DXP Proof

Date: 2026-05-26

## Purpose

Stage 011 emitted mixed flash-pipeline sidecar artifacts from the real generated
flash-prefill SDSC graph.  Stage 012 runs those artifacts through the clean
on-chip Foundation DXP build and adjusts the artifact shape to the current
Foundation contract.

The important finding is that current DXP accepts mixed SuperDSCs with one DL
compute DSC plus data-ops.  It rejects mixed SuperDSCs with multiple compute
DSCs.

## Implementation

Code changes:

- `torch_spyre/_inductor/onchip_realize.py`
  - The full proof artifact now inherits required SuperDSC top-level metadata
    (`sdscFoldProps_`, `sdscFolds_`, `coreFoldProp_`, `coreletFoldProp_`,
    `coreIdToDsc_`, `numWkSlicesPerDim_`, `coreIdToWkSlice_`) from the first
    generated compute SDSC.
  - Added `build_flash_attention_pipeline_tile_artifacts`, which emits one
    DXP-compatible mixed sidecar per generated flash-prefill `batchmatmul` tile.
- `torch_spyre/_inductor/codegen/bundle.py`
  - Sidecar emission now writes per-tile files named
    `sdsc_mixed_flash_pipeline_tile_<n>.json` in addition to the full proof
    artifact.
- `tests/_inductor/test_onchip_realize_logic.py`
  - Added coverage proving each tile sidecar has exactly one compute DSC, two
    `STCDPOpLx` data-ops, and the expected schedule shape.

## DXP Findings

Initial direct-DXP run on the Stage 011 full sidecars:

```text
sidecar_1: DXP_RC=134
what(): DtException: Unexpected json type for integer type import
```

Cause: the sidecar body did not include required top-level SuperDSC metadata.
After inheriting that metadata:

```text
sidecar_1: DXP_RC=0
sidecar_2: DXP_RC=134
what(): DtException: sdsc->dscs_.size() == 1, dxp.cpp line 366
```

Conclusion: multi-compute mixed flash pipeline artifacts remain a Foundation/DXP
contract gap.  The production-shaped executable step must start with one compute
DSC per mixed SuperDSC.

## Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Standalone tests on pod:

```text
tests/_inductor/test_onchip_realize_logic.py         17/17 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
```

Compile/value control plus tile sidecar generation:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-tile-sidecar-dxp-cache-1779818689
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q
```

Result:

```text
1 passed, 6 deselected in 13.67s
```

Direct DXP root:

```text
/tmp/sdpa-tile-sidecar-dxp-1779818716
```

Each generated `sdsc_mixed_flash_pipeline_tile_*.json` was copied into a small
single-SDSC bundle and compiled with:

```sh
DXP_DEBUG=1 "$PATCHED_DXP" --bundle -d <tile_bundle_dir>
```

Results:

| Tile bundle | DXP RC | HBM | L3_LDU | L3_STU | LX_LDSTU |
|---|---:|---:|---:|---:|---:|
| tile_1 | 0 | 0 | 0 | 0 | 192 |
| tile_2 | 0 | 0 | 0 | 0 | 160 |
| tile_3 | 0 | 0 | 0 | 0 | 192 |
| tile_4 | 0 | 0 | 0 | 0 | 160 |

All four tile sidecars generated `senprog.txt` under their debug directories.

## Interpretation

This proves the first DXP-compatible flash-attention mixed artifact shape:

```text
STCDPOpLx(K lane) + STCDPOpLx(V lane) + one generated batchmatmul compute DSC
```

It is still not a value-flow replacement.  The data-ops are staged alongside the
original compute DSC; the compute DSC still uses its generated inputs.  The next
implementation step is to replace one generated flash-prefill `batchmatmul`
SDSC in `bundle.mlir` with its corresponding one-compute mixed tile sidecar and
verify value correctness/device execution.  After that, we can wire the data-op
outputs into the compute input descriptors.
