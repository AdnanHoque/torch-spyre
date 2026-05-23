# Stage 327: Mixed Schedule Contract Check

## Summary

This stage moved the normal-lowering PT-LX path from a launch-oriented splice
probe back into the compiler verifier.  The goal was to answer whether the
chunked no-HBM bridge has a valid producer-to-bridge-to-consumer endpoint
contract before it is allowed to replace `ReStickifyOpHBM`.

The answer is now precise:

- chunked endpoint accounting is valid across all bridge roots;
- the bridge contains no HBM placements and no `ReStickifyOpHBM`;
- producer and consumer LX base starts match the planned endpoints;
- live element preservation is valid;
- replacement is still blocked because the bridge output descriptor does not
  match the actual consumer input descriptor.

## Fix

`_streaming_value_flow_contract(...)` previously inspected only the first
top-level payload root.  Row-chunked PT-LX bridges are multi-root payloads, so
the verifier counted only the first chunk and incorrectly failed the endpoint
contract.

The verifier now aggregates:

- `datadscs_` across all payload roots;
- chunk metadata such as `tile_count`, `datadsc_count`, and
  `logical_tile_count`;
- all consumer-visible endpoint outputs, not just the last data-op in a chunk.

A focused unit test covers this shape:

```text
test_streaming_ptlx_chunked_contract_aggregates_all_roots
```

## Probe

Forced compile-only probe:

```sh
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_E2E=1
SPYRE_RESTICKIFY_PTLX_NATIVE_VALIDGAP_ENDPOINT_TILE_CHUNK_SIZE=0
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=262144
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
python tools/restickify_scenario_probe.py \
  --case matmul_then_add \
  --size 512 \
  --skip-correctness \
  --skip-kernel-launch \
  --copy-kernel-code
```

Result after the verifier fix:

```text
endpoint_contract_valid: true
producer_start_valid: true
consumer_start_valid: true
gather_count: 64
validgap_tile_count: 64
scatter_count: 64
datadsc_count: 256
hbm_placements: 0
has_hbm_restickify: false
```

The replacement is still correctly rejected:

```text
production_blocker: bridge-output-does-not-match-consumer-lx-input
production_blocker_stage: consumer-descriptor
```

Consumer descriptor mismatch:

```text
bridge layout/stick:   mb,in / in
consumer layout/stick: out,mb / mb
bridge roots:          8
bridge outputs:        64
bridge pieces:         64
consumer pieces:       0
```

## Interpretation

This removes an accounting false negative and confirms the real next blocker.
The bridge is now structurally complete as a chunked no-HBM LX endpoint flow,
but its final visible descriptor is not the descriptor consumed by the next
Torch-Spyre op.

The next implementation target is consumer-driven bridge output lowering:
generate the final bridge endpoint from the real consumer input contract, or
patch the consumer input contract to match the bridge only when the compiler can
also prove value semantics.

Until then the compiler must keep `ReStickifyOpHBM` as the fallback.

## Validation

Pod unit test:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
54 passed in 1.02s
```

Artifacts:

```text
artifacts/stage327_mixed_contract_check/default_512.jsonl
artifacts/stage327_mixed_contract_check/forced_512.jsonl
artifacts/stage327_mixed_contract_check/forced_512_v2.jsonl
```
