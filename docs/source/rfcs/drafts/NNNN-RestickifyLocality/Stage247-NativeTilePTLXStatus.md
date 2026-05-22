# Stage 247: Native-Tile PT-LX Status

## Summary

This stage moved the production-shaped PT-LX restickify path from a large,
single bridge toward a bounded streaming tile lowering.  The new prototype
emits one native Deeptools local-transform tile per logical 64x64 tile:

1. gather the producer tile from LX with `STCDPOpLx`
2. transform the tile with `ReStickifyOpWithPTLx`
3. scatter the transformed tile to the consumer LX endpoint with `STCDPOpLx`

The native tile and full multi-tile bridge compile through Deeptools.  The
bridge remains default-off and is not allowed to replace `ReStickifyOpHBM`
unless the compiler can certify the value transform.

## What Works

- Hardware-free unit coverage validates the generated native tile shape,
  4D local-transform contract, all-LX placements, and bridge selector.
- `dcg_standalone` accepts both a single native tile and a full multi-tile
  generated bridge.
- A real Torch-Spyre compile for `adds_then_matmul`, size 512, can emit a
  cross-bundle native tile bridge that `dcg_standalone` accepts.
- The semantic guard now keeps the native path fail-closed.  If the native
  tile bridge is enabled but not value-certified, the graph falls back to the
  stock `ReStickifyOpHBM` path and remains value-correct.

## What Does Not Work Yet

The native tile bridge is not value-correct on hardware.  The size-512 launch
compiled and ran, but produced wrong values before the downstream matmul.  The
failure did not match a simple whole-tensor transpose variant, which suggests
the remaining issue is a physical coordinate/layout contract mismatch between:

- the producer's real LX output layout
- the bridge's native 4D tile-local layout
- the consumer's expected LX input layout

Until that contract is fixed, `streamingPTLXFull_` marks the native bridge as
`semantic_transform_certified: false`, and the value-flow verifier rejects it.

## Safety Fix

An important fallback bug was fixed in this stage.  The earlier prototype could
force producer/restickify buffers into LX before the later streaming bridge
guard rejected the bridge.  That meant the fallback SDSC still looked like
stock `ReStickifyOpHBM`, but its scratchpad metadata had already been changed.

The allocator now refuses uncertified streaming PT-LX endpoint forcing.  With
native tiles enabled at size 512, the audit reports:

```text
status=skipped
kind=ptlx-streaming-cross-bundle-handoff
reason=native-ptlx-tile-bridge-compiles-but-needs-value-correct-coordinate-contract
fallback=ReStickifyOpHBM
```

The corresponding correctness probe completes with zero errors.

## Validation

```bash
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_mapping_alignment.py \
  -q
```

Result:

```text
56 passed
```

Probe:

```bash
LX_PLANNING=1 \
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7 \
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1 \
SPYRE_RESTICKIFY_PTLX_NATIVE_TILE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576 \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 512 \
  --output-dir /tmp/stage247-native-fallback-512c \
  --fail-on-error
```

Result:

```text
ok size=512 case=adds_then_matmul restickifies=2 bytes=1048576 byte_hops=0
Completed 1 rows with 0 errors
```

## Distance To Wide-Size Enablement

The current state is compile-proven and safety-guarded, but not production
ready.  To enable PT-LX across a wide set of sizes, the remaining work is:

- make the native tile coordinate contract value-correct;
- prove the bridge writes exactly the physical layout the consumer reads;
- keep scratchpad endpoint reservation atomic with bridge certification;
- add tail handling for sizes that are not multiples of 64;
- benchmark only after value correctness holds for 512, 1024, 2048, and larger
  tiled shapes.

The stock HBM restickify path remains the fallback.
