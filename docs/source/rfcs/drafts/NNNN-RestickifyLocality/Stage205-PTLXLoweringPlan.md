# Stage 205: PT/LX Mixed Bridge Lowering Plan

## Summary

Stage204 added a verifier for the generated PT/LX mixed bridge value-flow
contract.  This stage moves one level earlier in the compiler: Torch-Spyre now
builds an explicit `OpSpec`-level plan for eligible
producer/restickify/consumer triples before SDSC JSON is generated.

The mixed bridge is still emitted as a generated SDSC payload mutation, but the
eligibility decision and endpoint intent now live in a small lowering plan
rather than being discovered entirely from already-generated JSON.

## Code Changes

- Added `PTLXMixedSchedulePlan`.
- Added `plan_restickify_ptlx_mixed_schedules(specs)`.
- `generate_bundle(...)` now computes the PT/LX plan before compiling OpSpecs.
- `patch_restickify_ptlx_mixed_schedules(...)` consumes that plan.
- The audit row now includes a `plan` block.
- Added unit coverage for:
  - planning a certified adjacent producer/restickify/consumer triple;
  - skipping an uncertified restickify.

## Planned Contract

For the high-signal 2048 case, the audit row now reports:

```json
{
  "plan": {
    "sdsc_index": 1,
    "producer_index": 0,
    "consumer_index": 2,
    "producer_lds_idx": 2,
    "consumer_lds_idx": 1,
    "producer_arg_index": 3,
    "consumer_arg_index": 4,
    "producer_base": 16384,
    "consumer_base": 8192
  }
}
```

The generated value-flow verifier still reports:

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

Focused unit tests:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
10 passed
```

2048 compile-only probe with locality and value-flow assertions:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

2048 hardware probe with the same assertions:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

Both probes emitted:

```text
replacement_sdsc: 1_MixedReStickifyOpWithPTLxConsumer
value_flow_contract.valid: true
```

## Interpretation

This is a small but important separation of concerns:

- `OpSpec` planning decides whether the compiler intends to lower an internal
  restickify edge as a PT/LX mixed bridge.
- SDSC emission still performs the concrete backend JSON edits needed by the
  current prototype.
- The value-flow verifier checks that the backend artifact matches the planned
  producer and consumer endpoints.

The next step is to move more of the backend-specific mutation behind planned
fields, especially producer output LX materialization and consumer input LX
materialization, so the final code path becomes generation rather than patching.
