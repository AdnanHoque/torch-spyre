# Stage 053: Layout-Xform Overlap Pair Is Stitchable, Values Wrong

Date: 2026-05-27

## Purpose

Stage052 proved that the forced IFN-prefix same-row probe can compile and run
under the diagnostic DXP stack, but it failed value correctness.  That probe had
no real predecessor producer, so Stage053 tries the closest current Torch-side
shape with a real predecessor: the Stage039 layout-transform pair, with the
consumer-side STCDPOpLx scheduled in the same row as the consumer batchmatmul.

This is a diagnostic, not the final warp-specialized attention shape.  It tests
whether replacing the synthetic same-SDSC IFN source with a predecessor-backed
LX source is enough for same-row execution to become value-correct.

## Change

A default-off gate was added:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_OVERLAP=1
```

The sweep harness exposes it as:

```text
layout_xform_pair_overlap_auto
```

When enabled together with the layout-transform pair auto selector, the generated
consumer sidecar changes from the safe serial schedule:

```text
[[0, -1, 0, 1], [-1, 0, 1, 0]]
```

to the diagnostic same-row schedule:

```text
[[0, 0, 0, 0]]
```

The consumer data-op is named with `prefetch_` so the pod's diagnostic DXP
stitcher patch recognizes it as the intentional duplicate-unit overlap probe:

```text
0_STCDPOpLx_prefetch_layout_xform_Tensor0_idx{idx}_tile{tile}
```

The selected artifact is marked:

```text
layout_xform_overlap_consumer=true
layout_xform_runtime_safe=false
layout_xform_runtime_forced=true
```

## Local Validation

```text
python3 -m py_compile torch_spyre/_inductor/config.py \
  torch_spyre/_inductor/codegen/bundle.py \
  torch_spyre/_inductor/onchip_realize.py \
  tools/onchip_sdpa_sweep.py \
  tests/_inductor/test_config_logic.py \
  tests/_inductor/test_onchip_realize_logic.py \
  tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_config_logic.py
python3 tests/_inductor/test_onchip_sdpa_sweep_logic.py
python3 tests/_inductor/test_onchip_realize_logic.py
python3 tests/_inductor/test_onchip_flash_pipeline_logic.py
git diff --check
```

Results:

```text
test_config_logic.py: 10/10 passed
test_onchip_sdpa_sweep_logic.py: 10/10 passed
test_onchip_realize_logic.py: 51/51 passed
test_onchip_flash_pipeline_logic.py: 11/11 passed
git diff --check: clean
```

The same torch-free test set passed after syncing the touched files into the pod
stage tree.

## Device Result

The run used the stage tree plus the patched local DXP binary:

```text
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
cache=/tmp/sdpa-stage053-layout-xform-overlap-localdxp-layout_xform_pair_overlap_auto-B1-H2-L128-D64-C0-635916-598572
```

The bundle selected the intended predecessor-backed overlap pair:

```text
bundle.mlir -> sdsc_mixed_flash_pipeline_tile_layout_xform_pair_2_predecessor.json
bundle.mlir -> sdsc_mixed_flash_pipeline_tile_layout_xform_pair_2_consumer.json
source=generated-flash-prefill-layout-xform-overlap-pair-consumer
datadsc=0_STCDPOpLx_prefetch_layout_xform_Tensor0_idx0_tile2
schedule=[[0, 0, 0, 0]]
replaces_sdsc=15_batchmatmul
layout_xform_predecessor_sdsc=14_ReStickifyOpHBM
layout_xform_attached_input_idx=0
layout_xform_runtime_safe=false
```

The program compiled and ran, then failed the value check:

```text
Mismatched elements: 16234 / 16384 (99.1%)
Greatest absolute difference: nan at index (0, 1, 1, 33)
Greatest relative difference: nan at index (0, 1, 1, 33)
```

## Current Status

Stage053 confirms that a real predecessor-backed layout-transform copy can be
stitched and launched in the same row as batchmatmul compute under the diagnostic
DXP stack, but that same-input overlap is still not value-correct.  The likely
hazard is read-after-write: the row computes from the consumer input while the
copy that populates that input is scheduled in the same row.

The next value-plausible direction is a fail-closed lookahead builder: copy the
current input before compute, then pair current compute with a prefetch for a
different future input whose producer has already executed.  That is closer to
the intended warp-specialized prefill analogue than another same-input overlap
probe.
