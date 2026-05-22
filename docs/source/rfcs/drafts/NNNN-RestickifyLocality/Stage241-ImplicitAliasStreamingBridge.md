# Stage 241: Implicit Alias Streaming Bridge

## Summary

Stage 240 showed that use-specific restickify insertion can avoid the old
name-wide rewrite bug, but it can leave a consumer reading one LX allocation
through two logical layouts.  That shape is not a valid backend contract: the
consumer sees the producer value directly and the restickified view is implicit.

This stage adds a default-off bridge patcher for that shape.  When all prototype
flags are enabled, the bundle writer can detect:

```text
producer output in LX
consumer input 0 reads producer layout from that same LX base
consumer input 1 reads a different logical view from that same LX base
consumer output expects the second layout
```

and replace the mismatched consumer input with an explicit streaming PT-LX bridge
before the consumer runs.

## Code Changes

- Added `patch_implicit_restickify_ptlx_aliases()` in
  `torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py`.
- Wired it into `generate_bundle()` after the normal mixed-schedule PT-LX patch.
- Added a focused hardware-free test:
  `test_implicit_alias_streaming_patch_materializes_consumer_input_bridge`.

The path is still default-off behind the existing prototype flags:

```sh
SPYRE_RESTICKIFY_USE_SPECIFIC_INSERT=1
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
```

## Validation

Hardware-free:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
26 passed
```

Guarded pod compile:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_self_transpose_join \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --output-dir /tmp/stage241-implicit-macro \
  --fail-on-error
```

The patcher fired and produced an LX-only value-flow contract:

```text
kind                     ptlx-implicit-alias-streaming
size                     512
tile_size                512
total_tiles              1
datadsc_count            3
hbm_placements           0
has_hbm_restickify       false
value_flow_contract      valid
```

The compile still failed in Deeptools:

```text
DtException: There must be at least one valid candidate.
L3DlOpsScheduler.cpp line 1075
```

## Interpretation

The Torch-Spyre side now makes the value-flow explicit: producer LX, bridge
workspace LX, bridge output LX, then consumer input LX.  The remaining blocker is
not that the compiler cannot describe the internal value.  The blocker is that
the generated all-to-one / one-to-all `STCDPOpLx + ReStickifyOpWithPTLx +
STCDPOpLx` data-op sequence is not accepted by the current Deeptools L3 data-op
scheduler for this fragmented transpose-style shape.

For `computed_self_transpose_join` at 512, the producer split is effectively
`mb:32` and the consumer split is `out:32`.  A 64x64 bridge tile therefore
fragments into four source cores and four destination cores.  Coalescing to one
512x512 macro tile reduces the bridge to three data ops, but the scheduler still
does not find a valid candidate.

## Next Step

The next production-shaped step is not another Torch-Spyre alias rewrite.  It is
one of:

1. generate a Deeptools-native remote-LX bridge that the L3 data-op scheduler
   accepts for all-to-all fragments, likely through the existing
   `InputFetchNeighbor`/interslice-transfer family; or
2. steer work distribution before SDSC generation so eligible PT-LX bridges have
   tile-aligned producer and consumer ownership, then fall back to
   `ReStickifyOpHBM` for fragmented all-to-all shapes.

Until one of those is true, this path should remain prototype-only and
default-off.
