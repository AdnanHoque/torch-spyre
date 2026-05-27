# Stage 051: Warp IFN-Prefix Probe Gate

Date: 2026-05-27

## Purpose

The active attention target is an AIU analogue of warp-specialized prefill
attention: overlap tile staging with current-tile compute.  Previous overlap
probes proved that independent synthetic `STCDPOpLx` sidecars can reach device
execution, but they are not value-safe when scheduled in the same row as flash
batchmatmul compute.

This stage adds a sharper default-off probe for the next hypothesis: use the
existing InputFetchNeighbor-shaped overlap-prefix artifact, which attaches the
movement to the DL input it is meant to feed, instead of relying on an
independent sidecar.

## Change

Added:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE=1
tools/onchip_sdpa_sweep.py --variants warp_ifn_prefix_probe
```

The new sweep variant sets:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE=0
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE=1
```

Normal execution remains conservative: when the mixed-pipeline sidecar path
emits IFN-prefix overlap artifacts, they are not selected unless the force flag
is set.  Forced artifacts are marked with:

```text
flashAttentionPipeline_.ifn_runtime_forced = true
```

## Interpretation

This is not a production enablement flag.  It is a device-facing probe that
exposes the actual Foundation/DXP blocker for an InputFetchNeighbor-attached
overlap row.  A compile/runtime failure or value mismatch is still expected at
this stage, but it is a better failure than the older independent-sidecar
overlap corruption because the executed descriptor now has the data movement
attached to the DL input it is meant to feed.

## Validation

Torch-free validation:

```text
tests/_inductor/test_config_logic.py
tests/_inductor/test_onchip_realize_logic.py
tests/_inductor/test_onchip_sdpa_sweep_logic.py
```

Pod validation copied the changed Torch files to:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

and reran the torch-free tests above.

Device command:

```sh
"$PYTHON" tools/onchip_sdpa_sweep.py \
  --lengths 128 \
  --variants warp_ifn_prefix_probe \
  --batch 1 --heads 2 --dim 64 --block-size 64 \
  --warmup 1 --iters 2 \
  --timeout-s 480 \
  --cache-prefix /tmp/sdpa-stage051-warp-ifn-prefix \
  --output-json /tmp/sdpa-stage051-warp-ifn-prefix.json
```

Result:

```text
L=128 warp_ifn_prefix_probe status=failed rc=1
cache=/tmp/sdpa-stage051-warp-ifn-prefix-warp_ifn_prefix_probe-B1-H2-L128-D64-C0-628887-229165
```

The generated bundle selected the forced IFN-prefix artifact:

```text
bundle.mlir -> sdsc_mixed_flash_pipeline_tile_0.json
source=generated-flash-prefill-overlap-prefix-ifn-tile
replaces_sdsc=0_batchmatmul
overlap_prefix=true
ifn_attached_input_idx=0
ifn_runtime_safe=false
ifn_runtime_forced=true
```

DXP aborted while stitching the selected bundle:

```text
DtException: unit already set for associated schedule step
file /home/adnan-cdx/dt-inductor-mixed/deeptools-onchip-foundation-clean/dcc/src/Stitcher/ModuleStitcher.cpp line 264
```

## Current Status

The Torch-side probe is doing the intended thing: it reaches an
InputFetchNeighbor-attached overlap-prefix descriptor instead of the older
independent synthetic sidecar.  The remaining blocker is now the Foundation/DXP
schedule representation for an overlap row that shares the associated schedule
step with the batchmatmul compute unit.  The next stage should make that row
shape stitchable without falling back to the value-unsafe independent-sidecar
path.
