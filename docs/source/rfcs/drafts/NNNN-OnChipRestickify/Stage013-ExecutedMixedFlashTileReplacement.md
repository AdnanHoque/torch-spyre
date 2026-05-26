# Stage 013: Executed Mixed Flash Tile Replacement

Date: 2026-05-26

## Purpose

Stage 012 proved the per-tile mixed flash sidecars compile through the clean
on-chip Foundation DXP build and generate `senprog.txt`.  Stage 013 makes the
first executable replacement: one generated flash-prefill `batchmatmul` SDSC in
`bundle.mlir` is replaced by its matching one-compute mixed tile sidecar.

This is still a staging step.  The mixed sidecar includes the original generated
compute DSC plus two `STCDPOpLx` data-ops.  The compute DSC still uses its
generated inputs; the data-op outputs are not yet wired into compute inputs.
The purpose of this stage is to prove the real SDPA bundle can execute a mixed
tile SDSC without breaking value correctness.

## Implementation

New gate:

```sh
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=<tile-index>
```

Behavior:

- `-1` keeps all mixed flash sidecars non-executed.
- `0` replaces the first generated flash-prefill `batchmatmul` tile in each
  generated bundle with `sdsc_mixed_flash_pipeline_tile_0.json`.
- `1`, `2`, etc. replace that tile index where it exists.

Code changes:

- `torch_spyre/_inductor/config.py`
  - Added `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE`.
- `torch_spyre/_inductor/onchip_realize.py`
  - Tile sidecars now record `flashAttentionPipeline_.replaces_sdsc`.
- `torch_spyre/_inductor/codegen/bundle.py`
  - When the execute-tile flag is non-negative, `bundle.mlir` uses the matching
    mixed tile sidecar filename instead of the original generated SDSC filename.
  - The original SDSC is still written to disk for inspection.
- `tests/_inductor/test_onchip_realize_logic.py`
  - Added coverage for `replaces_sdsc` metadata.

## Validation

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Logic tests:

```text
tests/_inductor/test_onchip_realize_logic.py         17/17 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
```

All device runs used:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export DXP_DEBUG=1
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q
```

### Tile 0

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=0
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-exec-mixed-tile0-1779818936
```

Result:

```text
1 passed, 6 deselected in 19.87s
```

Bundle refs:

```text
bundle 0: sdsc_mixed_flash_pipeline_tile_0.json
bundle 1: sdsc_mixed_flash_pipeline_tile_0.json
          sdsc_4_batchmatmul.json
          sdsc_14_batchmatmul.json
```

Mixed senprog:

```text
bundle 0 tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=192
bundle 1 tile_0: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=160
```

### Tile 1

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-exec-mixed-tile1-1779818991
```

Result:

```text
1 passed, 6 deselected in 12.93s
```

Bundle refs:

```text
bundle 0: sdsc_5_batchmatmul.json
bundle 1: sdsc_0_batchmatmul.json
          sdsc_mixed_flash_pipeline_tile_1.json
          sdsc_14_batchmatmul.json
```

Mixed senprog:

```text
bundle 1 tile_1: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=192
```

### Tile 2

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=2
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-exec-mixed-tile2-1779819027
```

Result:

```text
1 passed, 6 deselected in 12.00s
```

Bundle refs:

```text
bundle 0: sdsc_5_batchmatmul.json
bundle 1: sdsc_0_batchmatmul.json
          sdsc_4_batchmatmul.json
          sdsc_mixed_flash_pipeline_tile_2.json
```

Mixed senprog:

```text
bundle 1 tile_2: HBM=0 L3_LDU=0 L3_STU=0 LX_LDSTU=160
```

## Conclusion

Every generated flash-prefill `batchmatmul` tile in the tested SDPA shape can be
replaced one at a time by a mixed sidecar SDSC and still pass device value
correctness.  This proves the first executed production-shaped mixed flash tile.

The next step is to make the mixed tile semantically useful: flip the compute
input descriptors to consume the sidecar data-op LX outputs for one tile, then
rerun the same one-tile replacement tests.
