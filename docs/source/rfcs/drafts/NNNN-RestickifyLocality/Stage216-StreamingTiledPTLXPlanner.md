# Stage 216: Streaming Tiled PT-LX Planner

## Summary

This stage adds the first hardware-free planner for a streaming tiled PT-LX
restickify bridge. The current mixed PT-LX prototype emits one full-tensor
bridge. That works for the high-signal `2048x2048` case, but it still skips
other sizes when a full bridge would either:

- require producer pieces smaller than one 64-element stick,
- require mismatched producer/restickify core counts, or
- reserve too much per-core LX scratchpad for producer endpoint, bridge
  intermediate, and consumer endpoint at the same time.

The streaming planner makes the next contract explicit: move one logical
`64x64` tile at a time, gather producer fragments into a bridge core when the
producer split is finer than one tile, apply the local restickify, then scatter
the tile to the destination ownership expected by the consumer/restickify side.

## Why The 2 MB LX Scratchpad Matters

Each AIU core has its own 2 MB LX scratchpad. PT-LX avoids the stock
`ReStickifyOpHBM` round trip only when the compiler can keep the value flow in
that per-core LX address space.

For a full-tensor bridge, each core may need three live ranges at once:

1. producer output endpoint,
2. bridge intermediate,
3. consumer input endpoint.

For `2048x2048` fp16, the tensor is 8 MB total, or about 256 KB per core across
32 cores. Three ranges are still comfortably below 2 MB/core, so the current
PT-LX bridge can work. For `4096x4096` fp16, the tensor is 32 MB total, or about
1 MB per core. Three full ranges can exceed the 2 MB/core budget before other
compiler/runtime allocations are considered.

A streaming bridge changes the storage scale. A `64x64` fp16 tile is only 8192
bytes, so the bridge can reuse a small per-core tile buffer rather than holding
the whole restickified tensor in LX at once.

## What Changed

- Added `torch_spyre/_inductor/codegen/restickify_ptlx_streaming.py`.
- Extended `tools/restickify_tile_ownership_probe.py` with:

```sh
python3 tools/restickify_tile_ownership_probe.py \
  --streaming-ptlx \
  --size 512 \
  --source-work-slices mb:32,out:1 \
  --dest-work-slices mb:4,out:8
```

- Extended PT-LX skip audit rows so cases skipped for piece-size or intermediate
  LX-space reasons also include a `streaming_ptlx_candidate` block.

## Planner Results

Representative probe output:

```text
512  tiles=64    max_fan_in=4  max_fan_out=1  tile_buffer=8192  source-piece-smaller-than-tile
1024 tiles=256   max_fan_in=2  max_fan_out=1  tile_buffer=8192  source-piece-smaller-than-tile
1536 tiles=576   max_fan_in=2  max_fan_out=2  tile_buffer=8192  source-piece-smaller-than-tile
2048 tiles=1024  max_fan_in=1  max_fan_out=1  tile_buffer=8192  single-tile-bridge-contract-compatible
3072 tiles=2304  max_fan_in=2  max_fan_out=1  tile_buffer=8192  source-dest-core-count-mismatch
4096 tiles=4096  max_fan_in=1  max_fan_out=1  tile_buffer=8192  single-tile-bridge-contract-compatible
```

Interpretation:

- `512/1024/1536` are not inherently impossible for PT-LX. They need a gather
  stage because one 64-row tile spans multiple producer cores.
- `3072` needs a tile scheduler that is not locked to equal producer and
  restickify core counts.
- `4096` does not need unusual fan-in, but it needs streaming storage because
  full endpoints/intermediate are too large to reserve naively.
- `2048` is the current happy path because each 64-row producer piece already
  matches one tile.

## Validation

Local static validation:

```sh
python3 -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ptlx_streaming.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tools/restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_tile_ownership_probe.py
```

Direct planner smoke:

```sh
python3 - <<'PY'
from tools.restickify_tile_ownership_probe import default_core_mapping, plan_streaming_ptlx_tiles
for size, source, dest in [
    (512, {"mb": 32, "out": 1}, {"mb": 4, "out": 8}),
    (1024, {"mb": 32, "out": 1}, {"mb": 8, "out": 4}),
    (1536, {"mb": 32, "out": 1}, {"mb": 16, "out": 2}),
    (2048, {"mb": 32, "out": 1}, {"mb": 1, "out": 32}),
    (3072, {"mb": 32, "out": 1}, {"mb": 24, "out": 1}),
    (4096, {"mb": 32, "out": 1}, {"mb": 1, "out": 32}),
]:
    s = plan_streaming_ptlx_tiles(
        size=size,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
    )
    print(size, s.total_tiles, s.max_fan_in, s.max_fan_out, s.tile_buffer_bytes, s.notes)
PY
```

Desktop Python does not have `pytest` installed, so the focused pytest suite
should be run in the pod environment.

## Next Step

The next implementation stage is lowering, not more modeling:

1. add an explicit `SPYRE_RESTICKIFY_PTLX_STREAMING_E2E` gate,
2. turn each planner tile into a small gather/restickify/scatter data-op group,
3. reuse one per-core tile buffer across the tile stream,
4. preserve the fail-closed behavior: if any tile cannot be represented safely,
   leave the stock `ReStickifyOpHBM` path untouched.

The acceptance target should be a value-correct `512` or `1024` run with no
`ReStickifyOpHBM`, followed by a `4096` run that avoids full-tensor LX
workspace pressure.
