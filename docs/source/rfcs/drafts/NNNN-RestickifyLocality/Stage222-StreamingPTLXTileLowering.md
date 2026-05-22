# Stage 222: Static Streaming PT-LX Tile Lowering

## Summary

Stage 221 produced a streaming PT-LX descriptor. This stage lowers one
materialized descriptor tile into a SuperDSC-shaped object containing explicit
LX data-op stages:

1. `STCDPOpLx` gathers producer-owned source fragments into a bridge core's LX
   tile workspace;
2. `ReStickifyOpWithPTLx` restickifies the gathered tile locally;
3. `STCDPOpLx` writes the restickified tile into the consumer layout.

This remains a static/codegen-only prototype. It does not replace the stock
`ReStickifyOpHBM` fallback, and it has not yet been accepted by Deeptools or
run on hardware.

## What Changed

Added `generate_streaming_ptlx_tile_bridge_sdsc(...)`, which consumes a
`streaming_ptlx_restickify_descriptor` and emits a SuperDSC-shaped payload for
one tile. The generated payload has:

- no compute `dscs_`;
- three `datadscs_`;
- explicit fragment-level `PieceInfo`;
- LX-only `PlacementInfo`;
- the normal fallback recorded as `ReStickifyOpHBM`;
- `streamingPTLXTile_.status = static-codegen-only`.

The descriptor now also carries `source_core_count` and `dest_core_count` so
the static tile payload can preserve the original 32-core schedule shape even
when the sample tile only touches a subset of cores.

## Validation

Validated in the Spyre pod at `/tmp/torch-spyre-bench216`:

```bash
python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_lx_dataop.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_streaming.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
python -m pytest tests/inductor/test_restickify_tile_ownership_probe.py -q
```

Results:

- `tests/inductor/test_restickify_lx_dataop.py`: `19 passed`
- `tests/inductor/test_restickify_tile_ownership_probe.py`: `11 passed`

The sample lowered 512 tile reported:

```text
dscs 0 datadscs 3 ops ['STCDPOpLx', 'ReStickifyOpWithPTLx', 'STCDPOpLx'] hbm_placements 0
status static-codegen-only num_cores 32
gather_input_memids [0, 1, 2, 3]
gather_output_memids [0, 0, 0, 0]
```

This is the intended tile-level shape: four producer-core fragments are gathered
into bridge core 0, restickified locally, and then made available for the
consumer-layout write without any HBM placement in the static payload.

## What This Proves

The compiler-side representation no longer has to be a single full-tensor
PT-LX bridge. We can represent a bounded tile bridge with explicit remote LX
fragments, a local PT-aware restickify, and a consumer-layout write while
preserving the stock fallback path.

This is a meaningful step toward the production-shaped fix, but it is not
runtime proof. Deeptools may still reject the fragment-level data-op payload or
require additional scheduling/metadata fields.

## Next Step

Try to feed the one-tile static payload to the Deeptools compile path in a
compile-only mode. The acceptance target for the next stage is:

- DDC/DXP accepts the three-data-op tile payload, or reports a precise metadata
  blocker;
- generated op-funcs still contain no `ReStickifyOpHBM` for the tile payload;
- no hardware launch until the compile-only artifact is structurally clean.
