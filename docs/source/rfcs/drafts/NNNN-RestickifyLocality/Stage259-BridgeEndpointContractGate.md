# Stage 259: Bridge Endpoint Contract Gate

## Goal

Check whether the Stage258 direct tiled bridge candidate can be inserted into
the normal mixed schedule, and make the sidecar report the endpoint contract
that blocks insertion.

## What We Tried

First, the mixed-schedule path was run with:

```text
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_DIRECT_TILE_E2E=1
```

The initial probe skipped before building a bridge:

```text
producer-endpoint-not-allocator-backed:prototype-default
```

With the existing force-env endpoint escape hatch and non-overlapping endpoint
ranges:

```text
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=262144
```

the patcher built the direct tiled candidate but still correctly refused to
insert it:

```text
status: skipped
kind: ptlx-streaming-mixed-schedule
reason: direct-ptlx-tile-bridge-needs-hardware-value-validation
coalescing: direct-64x64-tiles
datadsc_count: 128
has_hbm_restickify: false
endpoint_contract_valid: true
value_preservation_valid: true
semantic_transform_certified: false
consumer_descriptor_valid: false
```

## Important Discovery

For the current 512 generated case, the direct PT-LX sidecar output descriptor
does not match the actual destination/restickify descriptor:

```text
bridge output:       layout out,mb  stick mb
destination expects: layout mb,out  stick out
```

That means "no HBM inside the sidecar" is not enough.  The executable gate must
also prove that the bridge output descriptor matches the restickify/consumer
destination descriptor.

## Code Change

The LX-neighbor sidecar now records:

```text
bridge_endpoint_contract
bridge_endpoint_contract_valid
```

For the real 512 probe:

```text
bridge_endpoint_contract_valid: false
reason: layout-dim-order-mismatch
bridge_layout: out,mb
bridge_stick: mb
destination_layout: mb,out
destination_stick: out
```

This makes the current blocker explicit in the generated compiler artifact.

## Validation

Focused unit test:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py -q
```

Result:

```text
10 passed in 7.25s
```

Real compiler artifact probe:

```text
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code \
  --output-dir /tmp/stage259-endpoint-contract-sidecar-512 \
  --fail-on-error
```

Result:

```text
status: emitted
size: 512
datadsc_count: 128
bridge_endpoint_contract_valid: false
reason: layout-dim-order-mismatch
```

## Interpretation

We are not ready to insert the bridge.  The next production-shaped fix must
choose the bridge primitive from the actual destination descriptor:

- if source and destination have the same layout/stick but different ownership,
  emit a tiled LX ownership remap rather than a PT-LX layout transform;
- if source and destination differ in stick/layout, emit the PT-LX tile
  transform and prove the output descriptor matches the destination;
- only then allow the bridge to replace `ReStickifyOpHBM`.

This is a useful narrowing: the next problem is no longer tile planning or HBM
avoidance.  It is selecting the correct per-edge bridge semantics from the
actual source/destination descriptors.

