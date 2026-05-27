# Stage 054: Fail-Closed Layout-Xform Lookahead Builder

Date: 2026-05-27

## Purpose

Stage053 showed that same-row copy+compute for the same consumer input is
stitchable but value-wrong.  Stage054 adds the next more plausible shape for a
warp-specialized prefill analogue:

```text
row 0: copy current input
row 1: compute current tile while prefetching a different future input
```

The builder is intentionally fail-closed.  It only emits the lookahead artifact
when the future input's producer is already earlier than the current consumer in
bundle order.  If the future producer is the current compute, or appears later,
the builder rejects the candidate instead of constructing another same-row
read-after-write hazard.

## Change

A new default-off tile gate was added:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_LOOKAHEAD_TILE=-2
```

The sweep harness exposes:

```text
layout_xform_lookahead_auto
```

That variant also enables the existing layout-transform pair fallback:

```text
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE=-2
```

Bundle precedence is:

1. use lookahead if it is legal;
2. otherwise fall back to the predecessor-backed layout-transform pair if it is
   configured and legal;
3. otherwise leave generated HBM-backed SDSCs.

The lookahead artifact shape is:

```text
*_current_predecessor
  original current producer, output LX-pinned

*_future_predecessor
  original future producer, output LX-pinned

*_current_consumer
  datadsc 0: layout-transform copy for current input
  datadsc 1: prefetch layout-transform copy for future input
  compute 0: current batchmatmul
  schedule: [[0, -1, 0, 1], [1, 0, 1, 0]]

*_future_consumer
  original future consumer, future input LX-pinned to the prefetched buffer
```

The future-prefetch data-op name contains `prefetch_` so it matches the
diagnostic DXP duplicate-unit predicate used by the Stage052/Stage053 probes.

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
test_config_logic.py: 11/11 passed
test_onchip_sdpa_sweep_logic.py: 12/12 passed
test_onchip_realize_logic.py: 55/55 passed
test_onchip_flash_pipeline_logic.py: 11/11 passed
git diff --check: clean
```

The key new unit test builds a synthetic legal graph with two earlier
layout-transform producers feeding current and future batchmatmuls.  It verifies
the intended two-row schedule and future-consumer LX pinning.

## Pod Result

The touched files were synced into:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-stage039-two-sdsc-ifn
```

Torch-free pod tests passed:

```text
test_onchip_realize_logic.py: 55/55 passed
test_onchip_sdpa_sweep_logic.py: 12/12 passed
```

The real SDPA run used:

```text
variant=layout_xform_lookahead_auto
cache=/tmp/sdpa-stage054-layout-xform-lookahead-fallback-localdxp-layout_xform_lookahead_auto-B1-H2-L128-D64-C0-637913-590925
```

The current generated graph had no legal lookahead candidate:

```text
Requested layout-transform lookahead flash attention pair was not realizable:
['tile0:current:input0:no_latest_producer']
```

The layout-transform pair fallback also had no candidate in that particular
wrapper run:

```text
Requested layout-transform flash attention pair was not realizable:
['tile0:input0:no_latest_producer']
```

The cache confirmed no lookahead or pair sidecars were selected; bundle.mlir
referenced only original generated SDSCs.  The subsequent value mismatch is
therefore not evidence against the lookahead schedule itself.  It is the raw
mixed/HBM path for that generated graph:

```text
Mismatched elements: 7266 / 16384 (44.3%)
Greatest absolute difference: 0.90576171875 at index (0, 0, 22, 29)
```

## Current Status

The Torch-side lookahead builder now exists and is covered by local structural
tests, including a legal synthetic graph.  The real L128 SDPA graph produced by
the current wrapper does not yet expose the producer ordering needed to exercise
the lookahead path on device.

The next implementation step is to either find a real fused-attention shape that
has an earlier future producer, or broaden the builder beyond LX-to-LX
layout-transform edges so it can prefetch an HBM-backed future input without
requiring an already-executed predecessor sidecar.
