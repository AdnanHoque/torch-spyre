# Stage 221: Streaming PT-LX Descriptor

## Summary

This stage turns the Stage 220 streaming PT-LX contract into a concrete
codegen-only descriptor. The descriptor is not yet a Deeptools SuperDSC or a
runtime artifact. It is the next production-shaped boundary object: lowering
can consume it to build a tiled producer-to-consumer LX bridge without replacing
the stock HBM restickify fallback.

The target remains:

- keep `ReStickifyOpHBM` as the fallback path;
- only use PT-LX when the compiler has a valid producer/bridge/consumer LX
  value-flow contract;
- handle non-2048 sizes by streaming bounded tiles through LX scratchpad
  workspace rather than requiring the whole bridge tensor to fit at once.

## Descriptor Shape

The new helper `generate_streaming_ptlx_artifact(...)` emits a descriptor with:

- tile geometry and total tile count;
- explicit LX buffer bases for producer, consumer, and tile workspace;
- a bounded workspace contract;
- one or more materialized tile records;
- per-tile stages:
  1. `STCDPOpLx` gather from producer-owned source fragments;
  2. `ReStickifyOpWithPTLx` local tile restickify;
  3. `STCDPOpLx` write/scatter into the consumer LX layout.

For the known skipped 512 case, the sample descriptor reports `fan_in=4` and
source cores `[0, 1, 2, 3]` for the first destination tile. That is the shape we
need for a streaming bridge: the destination tile can be assembled from several
producer-owned fragments, restickified locally, then written to the consumer's
expected tile layout.

## Validation

Validated in the Spyre pod at `/tmp/torch-spyre-bench216` with:

```bash
python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ptlx_streaming.py \
  tools/restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_tile_ownership_probe.py

python -m pytest tests/inductor/test_restickify_tile_ownership_probe.py -q
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q

python tools/restickify_tile_ownership_probe.py \
  --streaming-ptlx \
  --streaming-artifact \
  --size 512 \
  --source-work-slices mb:32,out:1 \
  --dest-work-slices mb:4,out:8 \
  --artifact-max-tiles 1
```

Results:

- `tests/inductor/test_restickify_tile_ownership_probe.py`: `11 passed`
- `tests/inductor/test_restickify_lx_dataop.py`: `18 passed`
- sample descriptor kind: `streaming_ptlx_restickify_descriptor`
- sample descriptor status: `codegen-only`
- sample descriptor phases:
  `gather-source-fragments`, `local-ptlx-restickify`, `write-dest-tile`
- sample descriptor ops:
  `STCDPOpLx`, `ReStickifyOpWithPTLx`, `STCDPOpLx`

## What This Proves

This proves the compiler can now materialize the missing intermediate contract
for non-2048 shapes at the tile/fragments level. The descriptor identifies the
remote producer fragments, the local PT-LX restickify tile, and the consumer
layout write. It also keeps the LX workspace bounded by using reusable tile
buffers.

This does not yet prove runtime correctness for non-2048 sizes. The descriptor
still has to be lowered into actual Deeptools data-op/SuperDSC objects and then
validated on hardware.

## Next Step

Lower one descriptor tile into a real Deeptools artifact and statically inspect
the generated op-funcs. The first acceptance target is compile/codegen only:

- no generated `ReStickifyOpHBM` for the patched producer/restickify/consumer
  boundary;
- explicit LX gather and scatter data-op stages;
- bounded tile workspace;
- stock HBM restickify remains available for all unsupported cases.

Only after that static shape is correct should this path run on hardware again.
