# Stage 319: Normal Sidecar Selection Hook

## Summary

Stage 319 wires the Stage317/318 artifact into the normal LX-neighbor streaming
sidecar selection path behind a new default-off flag:

```text
SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_E2E=1
```

With this flag enabled for kernel-to-output PT-LX edges, the compiler emits a
sidecar candidate using:

```text
STCDPOpLx gather
ReStickifyOpWithPTLx native local tile transform
ReStickifyOpWithPTLx valid-gap consumer endpoint adapter
STCDPOpLx valid-gap endpoint scatter
```

The stock `ReStickifyOpHBM` path remains the runnable fallback. The sidecar is
still not inserted into `bundle.mlir`.

## Code Path

The new full-bridge helper is:

```text
generate_streaming_ptlx_native_validgap_endpoint_full_bridge_sdsc(...)
```

It combines every materialized 64x64 tile using the Stage317 per-tile shape and
records:

```text
coalescing = native-validgap-endpoint-scatter-64x64-tiles
native_local_transform_contract = true
validgap_endpoint_adapter_contract = true
validgap_endpoint_scatter_contract = true
semantic_transform_certified = false
fallback = ReStickifyOpHBM
```

Selection is available from both:

- LX-neighbor sidecar emission in `lx_neighbor_descriptor.py`;
- the bridge payload selector used by the PT-LX boundary probes.

## Validation

```sh
python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_dataop.py -q
```

Result: `63 passed`.

Focused selector check:

```sh
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  -k bridge_selector -q
```

Result: `1 passed`.

## Interpretation

This makes the new artifact reachable from the normal compiler sidecar path,
but it remains explicitly diagnostic:

- no public API change;
- no default behavior change;
- no automatic HBM replacement;
- no semantic certificate yet.

Next step: generate a real Torch-Spyre code directory with this flag enabled,
verify that the emitted `restickify_lx_neighbor_streaming_bridge_edge_*.json`
uses `native-validgap-endpoint-scatter-64x64-tiles`, then package that sidecar through
the Stage318 export path.
