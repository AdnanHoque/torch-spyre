# Stage 309: LX Neighbor Production Contract

## Summary

This stage made the LX-neighbor sidecar describe whether a generated 64x64
bridge candidate is production-valid or still only diagnostic evidence.

The branch already emitted non-executable sidecar bridge candidates from real
producer and consumer SDSC ownership metadata.  Those candidates now carry a
`production_contract` field that separates:

- same-layout LX ownership remaps, which can be certified from bridge metadata;
- real PT-LX layout transforms, which still require a remote-fragment-aware
  three-stage lowering before they may replace `ReStickifyOpHBM`.

## Code Changes

`restickify_lx_neighbor_streaming_bridge_edge_<idx>.json` candidates now report:

```text
production_valid
production_blocker
production_contract
```

The `production_contract` includes:

```text
bridge_kind
endpoint_contract_valid
semantic_transform_certified
bounded_workspace_ok
tile_contract
required_primitive
required_lowering
fallback
```

The `tile_contract` records whether every tile is materialized, fan-in/fan-out
histograms, remote gather/scatter counts, bounded workspace bytes, tile buffer
bytes, and modeled byte-hops.

## Current Gate

Same-layout ownership remap candidates can report:

```text
production_valid = true
production_blocker = null
```

Actual PT-LX layout transforms report:

```text
production_valid = false
production_blocker = missing-three-stage-remote-fragment-ptlx-lowering
required_primitive = remote-fragment-aware-ptlx-coordinate-remap
```

and the required lowering is:

```text
STCDPOpLx/InputFetchNeighbor gather producer LX fragments into bounded workspace
local PT/interslice tile transform changes stick/layout semantics
STCDPOpLx/InputFetchNeighbor writes or scatters the consumer-owned LX tile
```

This keeps the stock `ReStickifyOpHBM` fallback until the sidecar can be turned
into a value-correct executable bridge.

## Validation

Pod validation:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m py_compile \
  torch_spyre/_inductor/codegen/lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py

TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_dataop.py \
  -q
```

Result:

```text
58 passed in 4.64s
```

## Next Step

Lower the non-production PT-LX candidate into the actual three-stage bridge
shape.  The first executable prototype should target a single-source,
single-destination 64x64 tile because that isolates the local PT/interslice
value transform before adding multi-fragment gathers for smaller sizes such as
512.
