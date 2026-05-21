# Stage 211: Fail-Closed PT-LX Endpoint Requirements

## Summary

Stage 211 makes the PT-LX mixed-schedule prototype fail closed.

Earlier stages preserved prototype fallback bases:

- producer fallback: `16384`
- consumer fallback: `8192`

Those were useful for proving the bridge, but they are not acceptable for a
production-shaped lowering path. This stage requires real allocator-backed
endpoint bases before the PT-LX bridge is allowed to replace
`ReStickifyOpHBM`.

If allocator-backed endpoint information is missing or invalid, the compiler
skips the PT-LX bridge and leaves the stock HBM restickify path in place.

## Required Conditions

The PT-LX mixed schedule now requires:

- producer endpoint `base_source == "op-spec-allocation"`
- consumer endpoint `base_source == "op-spec-allocation"`
- `ptlx_endpoint_allocation` exists in `OpSpec.op_info`
- `endpoint_allocation.overlap_check.valid == true`
- recorded producer range starts at the producer endpoint base
- recorded consumer range starts at the consumer endpoint base

Failure reasons are emitted in the PT-LX audit row. Examples:

```text
producer-endpoint-not-allocator-backed:prototype-default
consumer-endpoint-not-allocator-backed:prototype-default
missing-endpoint-allocation
invalid-endpoint-overlap-check
producer-endpoint-base-mismatch
consumer-endpoint-base-mismatch
```

## Validation

Focused pod test:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
15 passed in 0.23s
```

The added negative cases verify that:

- a certified restickify without allocator-backed endpoints does not get a
  PT-LX plan
- an endpoint allocation record with an invalid overlap check does not get a
  PT-LX plan

## Hardware Guardrails

Without scratchpad planning:

```text
LX_PLANNING unset
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
case=computed_transpose_adds_then_matmul_tuple
size=2048
status=skipped
reason=producer-endpoint-not-allocator-backed:prototype-default
```

The run still completed successfully because the compiler fell back to the
stock `ReStickifyOpHBM` path.

With scratchpad planning:

```text
LX_PLANNING=1
SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING unset
case=computed_transpose_adds_then_matmul_tuple
size=2048
status=patched
restickifies=1
bytes=8388608
byte_hops=0
device_events=0
```

Allocator-backed audit highlights:

```text
producer base_source=op-spec-allocation base=0
consumer base_source=op-spec-allocation base=262144
endpoint_allocation.overlap_check.valid=true
value_flow_contract.valid=true
replacement_sdsc=1_MixedReStickifyOpWithPTLxConsumer
```

## Interpretation

This stage removes the most important prototype hazard: the bridge no longer
silently invents LX addresses when real scratchpad allocations are absent.

The path is now:

1. certify zero-hop restickify locality
2. request narrow LX endpoint buffers
3. record and validate endpoint allocation ranges
4. require those allocator-backed ranges before replacing the HBM restickify
5. otherwise leave the stock path untouched

## Remaining Work

Next useful steps:

- add an explicit negative overlap unit test with a deliberately conflicting
  allocator
- consolidate the flag cluster into one guarded PT-LX prototype option
- run a timing comparison between stock HBM restickify and fail-closed PT-LX
  mixed restickify with `LX_PLANNING=1`
