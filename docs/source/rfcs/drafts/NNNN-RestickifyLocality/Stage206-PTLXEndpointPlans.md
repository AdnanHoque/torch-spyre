# Stage 206: PT/LX Endpoint Plans

## Summary

Stage205 introduced an `OpSpec`-level mixed bridge plan.  This stage makes the
producer and consumer LX endpoints explicit inside that plan.

Instead of carrying only loose `producer_base` and `consumer_base` values, the
plan now carries two endpoint records:

- `producer_endpoint`: the producer output LDS that must become LX-backed;
- `consumer_endpoint`: the consumer input LDS that must read the bridge output
  from LX.

The SDSC emitter still performs JSON mutation, but the mutation is now driven
by typed endpoint intent.

## Code Changes

- Added `PTLXLXEndpointPlan`.
- `PTLXMixedSchedulePlan` now contains:
  - `producer_endpoint`
  - `consumer_endpoint`
- Added endpoint materialization helpers:
  - `_materialize_producer_lx_endpoint(...)`
  - `_materialize_consumer_lx_endpoint(...)`
- Added role checks so a producer-output endpoint cannot be used as a
  consumer-input endpoint by accident.
- Extended unit coverage to assert endpoint roles, direction, and owning SDSC
  indices.

## Audit Shape

For the 2048 probe, the audit row now contains:

```json
{
  "plan": {
    "sdsc_index": 1,
    "producer_index": 0,
    "consumer_index": 2,
    "producer_endpoint": {
      "role": "producer_output",
      "sdsc_index": 0,
      "lds_idx": 2,
      "arg_index": 3,
      "base": 16384,
      "is_input": false
    },
    "consumer_endpoint": {
      "role": "consumer_input",
      "sdsc_index": 2,
      "lds_idx": 1,
      "arg_index": 4,
      "base": 8192,
      "is_input": true
    }
  }
}
```

The verified value-flow contract remains:

```json
{
  "valid": true,
  "producer_to_bridge_input_match": true,
  "bridge_output_to_consumer_match": true,
  "producer_core_count": 32,
  "bridge_input_core_count": 32,
  "bridge_output_core_count": 32,
  "consumer_core_count": 32
}
```

## Validation

Focused tests:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
10 passed
```

2048 hardware probe with locality and value-flow assertions enabled:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

## Interpretation

This is another small move from patching toward planned lowering.  The current
prototype still mutates generated SDSC JSON, but the inputs to that mutation
are now explicit compiler intent:

```text
producer output endpoint -> PT/LX bridge -> consumer input endpoint
```

The next cleanup target is the bridge data-op endpoint itself: generate its
input and output endpoint bindings from the same endpoint plan, rather than
constructing independent per-core maps at the SDSC-emission site.
