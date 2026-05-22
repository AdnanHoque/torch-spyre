# Stage 255: Output-Alias ValidGap PT-LX Probe

## Goal

Retest the forced consumer-shaped PT-LX bridge after making the sparse
`validGap_` axis match Deeptools' `ReStickifyOpWithPTLx` validation rule.

The prior Stage254 descriptor used:

```text
input layout: out_, mb_, in_
input stick:  out_
input validGap out_: [[1, 63]]
output layout: mb_, in_
output stick:  in_
```

Inspection of Deeptools' `determineSubOp` showed that the input valid-gap check
is applied on the output stick dimension.  For this diagnostic shape, that means
the sparse alias must be `in_`, not `out_`.

## Change

The descriptor now keeps the source stick lanes dense and sparsifies the output
stick alias:

```text
input layout: out_, mb_, in_
input stick:  out_
input validGap out_: [[64, 0]]
input validGap mb_:  [[64, 0]]
input validGap in_:  [[1, 63]]
output layout: mb_, in_
output stick:  in_
```

The local unit tests were updated to assert this shape.

## Validation

Static/unit validation in the pod:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_mapping_alignment.py -q
```

Result:

```text
64 passed in 3.27s
```

Forced hardware probe:

```text
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_VALIDGAP_CONSUMER_TILE_E2E=1
SPYRE_RESTICKIFY_PTLX_FORCE_VALIDGAP_CONSUMER_TILE_E2E=1
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 512 \
  --output-dir /tmp/stage255-asdout-validgap-force-512 \
  --copy-kernel-code
```

Result:

```text
error size=512 case=adds_then_matmul restickifies=0 bytes=0 byte_hops=0
Mismatched elements: 204455 / 262144 (78.0%)
Greatest absolute difference: 1.8662109375 at index (15, 289)
```

The audit row still proved that the stock HBM restickify was replaced:

```text
status: patched
replacement_sdsc: 3_CrossBundleProducerStreamingReStickifyOpWithPTLx
coalescing: validgap-consumer-64x64-tiles
has_hbm_restickify: false
hbm_placements: 0
endpoint_contract_valid: true
consumer_descriptor_valid: true
validgap_tile_count: 64
```

The device remained healthy afterward; a stock `adds_then_matmul` size-128 smoke
completed with zero errors.

## Interpretation

This is a useful negative result.  The sparse output-alias descriptor is closer
to the Deeptools validation rule, but it still does not implement the logical
restickify transform correctly.  Both the source-sparse and output-alias-sparse
variants compile, launch, remove `ReStickifyOpHBM`, and then return wrong
values.

That means the current consumer-shaped `ReStickifyOpWithPTLx` alias route should
stay diagnostic-only.  The next production-shaped path should pivot back to the
value-correct inter-slice DDL/`interslicetranspose_fp16` route and solve its
capacity/tile scheduling blocker, or use an explicit data-op bridge that
materializes the consumer view.

## Artifacts

```text
artifacts/stage255_asdout_validgap_force_512/
```
