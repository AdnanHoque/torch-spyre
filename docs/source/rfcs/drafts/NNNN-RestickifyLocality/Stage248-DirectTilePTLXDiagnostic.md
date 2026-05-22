# Stage 248: Direct-Tile PT-LX Diagnostic

## Summary

This stage tested a narrower PT-LX bridge shape after the native 4D tile
diagnostic compiled but produced wrong values. The new diagnostic emits
64x64 direct tiles using `ReStickifyOpWithPTLx` with the same 2D
`mb_/out_ -> out_/mb_` contract as the earlier full-tensor PT-LX prototype.
When producer fragments are smaller than one 64x64 tile, the diagnostic first
coalesces them in LX with a same-layout `STCDPOpLx` gather, then writes the
restickified tile directly to the consumer endpoint.

The result is useful but not yet production-ready:

- The direct/gather-direct tile bridge compiles through Deeptools.
- The endpoint contract verifier sees no HBM placement and no
  `ReStickifyOpHBM`.
- Hardware validation at `adds_then_matmul`, size `512`, produced wrong values.
- The compiler now treats this path as semantically uncertified and falls back
  to stock `ReStickifyOpHBM`.

## Why This Stage Exists

The previous native 4D tile shape likely wrote a local
`j_,i_,out_,mb_` physical layout while the downstream consumer still expected
the normal 2D input contract. This stage removed the final `STCDPOpLx` scatter
and tried to let `ReStickifyOpWithPTLx` write directly into the consumer's
logical `out_/mb_` endpoint.

That still did not prove value correctness. This means the remaining blocker is
not only the final scatter. The bridge must be generated from, or verified
against, the consumer's actual input layout descriptor rather than a synthetic
destination fragment model.

## Implementation Notes

New default-off diagnostic flag:

```sh
SPYRE_RESTICKIFY_PTLX_DIRECT_TILE_E2E=1
```

New generator functions:

- `generate_streaming_ptlx_direct_tile_bridge_sdsc`
- `generate_streaming_ptlx_direct_full_bridge_sdsc`

The direct path is selected before the native 4D diagnostic when both flags are
enabled. The path remains fail-closed:

- `endpoint_contract_valid=true` only proves LX endpoints and schedule shape.
- `semantic_transform_certified=false` prevents replacement of the stock HBM
  restickify.
- The audit records the reason
  `direct-ptlx-tile-bridge-needs-hardware-value-validation`.

## Hardware Evidence

The forced-patch diagnostic at `adds_then_matmul`, size `512`, failed
correctness:

```text
Mismatched elements: 210465 / 262144 (80.3%)
Greatest absolute difference: 2.17578125
```

The audit for that run showed the old, overly optimistic state:

```json
{
  "status": "patched",
  "replacement_sdsc": "3_CrossBundleProducerStreamingReStickifyOpWithPTLx",
  "value_flow_contract": {
    "coalescing": "direct-64x64-tiles",
    "endpoint_contract_valid": true,
    "semantic_transform_certified": true,
    "valid": true,
    "hbm_placements": 0,
    "has_hbm_restickify": false
  }
}
```

After this stage, the same diagnostic flag builds the same endpoint candidate
but refuses to patch it:

```json
{
  "status": "skipped",
  "reason": "direct-ptlx-tile-bridge-needs-hardware-value-validation",
  "value_flow_contract": {
    "coalescing": "direct-64x64-tiles",
    "endpoint_contract_valid": true,
    "semantic_transform_certified": false,
    "valid": false,
    "hbm_placements": 0,
    "has_hbm_restickify": false
  }
}
```

The guarded run passed via fallback:

```text
ok size=512 case=adds_then_matmul restickifies=2 bytes=1048576 byte_hops=0
```

Artifacts:

- `artifacts/stage248_direct_tile_ptlx/audit_wrong_values_512.jsonl`
- `artifacts/stage248_direct_tile_ptlx/restickify_scenarios_wrong_values_512.jsonl`
- `artifacts/stage248_direct_tile_ptlx/audit_fallback_512.jsonl`
- `artifacts/stage248_direct_tile_ptlx/restickify_scenarios_fallback_512.csv`

## Validation

Focused pod tests:

```sh
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_mapping_alignment.py \
  -q
```

Result:

```text
58 passed in 3.66s
```

Guarded fallback probe:

```sh
LX_PLANNING=1 \
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7 \
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1 \
SPYRE_RESTICKIFY_PTLX_DIRECT_TILE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576 \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 512 \
  --output-dir /tmp/stage248-direct-fallback-512 \
  --copy-kernel-code \
  --fail-on-error
```

Result:

```text
Completed 1 rows with 0 errors
```

## Next Step

The next productive step is to generate the PT-LX bridge from the consumer's
actual input layout contract. The verifier should compare the bridge output
`PieceInfo`, `layoutDimOrder_`, `stickDimOrder_`, and placement/base addresses
against the real consumer input `labeledDs_`. A PT-LX replacement should only
patch the bundle when that descriptor-level contract proves the bridge writes
the exact layout the consumer will read.

Until then, stock `ReStickifyOpHBM` remains the safe fallback.
