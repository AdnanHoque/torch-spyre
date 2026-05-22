# Stage 227-229: Production-Shaped Streaming PT-LX Handoff

## Summary

This stage moved the streaming PT-LX restickify prototype from standalone/static artifacts toward normal Torch-Spyre lowering.

The important finding is that the real `adds_then_matmul` graph is not shaped like the synthetic adjacent triple. With the stock bundle tensor cap, the matmul-side restickify is emitted as the first SDSC in the matmul bundle:

```text
0001_sdsc_fused_add_t_0:
  sdsc_0_ReStickifyOpHBM.json
  sdsc_1_add.json
  sdsc_2_add.json

0002_sdsc_fused_mm_1:
  sdsc_0_ReStickifyOpHBM.json
  sdsc_1_batchmatmul.json
```

That means a producer/restickify/consumer LX contract cannot be formed by a bundle-local adjacent patch alone.

## Changes

- Added a default-preserving `SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS` knob so we can probe bundle shape without changing normal behavior. The default remains `6`.
- Added a default-off `SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E` deferred compile window:
  - `SpyreAsyncCompile.sdsc()` can now emit bundle JSON and delay DXP compilation.
  - `SpyreAsyncCompile.wait()` can patch adjacent runtime bundles before DXP sees them.
- Added a cross-bundle streaming PT-LX handoff patch:
  - Detects `producer -> trailing ReStickifyOpHBM` in one bundle and the next bundle's first consumer.
  - Replaces the trailing in-graph `ReStickifyOpHBM` with a tiled streaming `ReStickifyOpWithPTLx` data-op bridge.
  - Patches the next bundle consumer input to read the same LX endpoint.
  - Keeps graph-input restickifies on the stock HBM path.

## Probe Results

Baseline compile-only real graph:

```sh
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
python tools/restickify_scenario_probe.py --case adds_then_matmul --size 512 \
  --skip-correctness --skip-kernel-launch --copy-kernel-code \
  --output-dir /tmp/stage227-real-compile --fail-on-error
```

Result: compile succeeded, but the PT-LX patch skipped because the restickify was not between adjacent SDSCs. The generated files still contained `sdsc_0_ReStickifyOpHBM.json` in both runtime bundles.

Single-bundle cap probe:

```sh
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7
```

Result: compile succeeded and moved the in-graph restickify to the end of the producer bundle:

```text
0001_sdsc_fused_add_t_0:
  sdsc_0_ReStickifyOpHBM.json
  sdsc_1_add.json
  sdsc_2_add.json
  sdsc_3_ReStickifyOpHBM.json

0002_sdsc_fused_mm_1:
  sdsc_0_batchmatmul.json
```

The audit reported a valid streaming candidate for `sdsc_index=3`.

Cross-bundle handoff probe:

```sh
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=65536
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
```

Result: the cross-bundle patch fired and produced a valid value-flow contract:

```json
{
  "status": "patched",
  "kind": "ptlx-streaming-cross-bundle-handoff",
  "replacement_sdsc": "3_CrossBundleStreamingReStickifyOpWithPTLx",
  "streaming_summary": {
    "size": 512,
    "total_tiles": 64,
    "tile_buffer_bytes": 8192,
    "total_transfer_bytes": 1048576,
    "total_byte_hops": 1376256,
    "max_fan_in": 4,
    "max_fan_out": 1
  },
  "value_flow_contract": {
    "valid": true,
    "has_hbm_restickify": false,
    "hbm_placements": 0,
    "gather_count": 64,
    "scatter_count": 64,
    "datadsc_count": 192
  }
}
```

The emitted cache artifact confirmed:

```text
sdsc_fused_add_t_0:
  sdsc_3_CrossBundleStreamingReStickifyOpWithPTLx.json

sdsc_fused_mm_1:
  sdsc_0_batchmatmul.json
    Tensor0 allocation component: lx
```

## Current Blocker

DXP rejects the patched producer bundle before launch:

```text
DtException: Datadsc not allowed without dldsc schedule
file .../deeptools/dxp/SdscTree.cpp line 155
```

So the compiler-side value-flow contract now exists, but the generated bridge is still packaged as `datadscs_` in a normal bundle. The next production-shaped step is to lower the streaming bridge through a DXP-accepted DLDSC/dldsc schedule form, or otherwise use a backend entry point that accepts the same data-op sequence in a normal Torch-Spyre runtime bundle.

## Validation

```sh
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
23 passed
```

## Conclusion

We are no longer blocked on finding the production-shaped boundary. The real boundary is a cross-bundle handoff:

```text
producer bundle: producer add -> streaming PT-LX bridge
consumer bundle: matmul reads bridged input from LX
```

The remaining gap is backend packaging, not compiler-side ownership discovery.
