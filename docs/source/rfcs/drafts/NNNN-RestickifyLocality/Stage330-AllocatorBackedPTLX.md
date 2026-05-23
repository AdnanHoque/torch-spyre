# Stage 330: Allocator-Backed PT-LX Validation

## Summary

Stage 330 reran the normal mixed PT-LX path on hardware after restoring
OpenShift access.  The key result is that the 2048 same-bundle restickify edge
works without forced environment-provided LX bases when `LX_PLANNING=1` is
enabled.

This is the current best prototype shape:

- allocator-backed producer LX endpoint;
- allocator-backed consumer LX endpoint;
- explicit endpoint overlap check;
- `MixedReStickifyOpWithPTLxConsumer` generated during normal lowering;
- no `ReStickifyOpHBM` in the fused-add bundle;
- hardware correctness passes.

## Code Change

Scratchpad endpoint discovery was too strict about `restickify_source_kind`.
During scratchpad planning, the explicit source-kind marker can be absent even
when the restickify still has enough information for `producer_for_restickify`
to prove an in-graph producer.

The preliminary gate now rejects only explicit non-in-graph source kinds:

```text
graph_input_or_weight -> skip
unknown/non-in-graph -> skip
missing source_kind -> allow later producer lookup to decide
```

The later producer lookup remains the proof.  If no in-graph producer can be
found, the edge still skips.

## Hardware Command

```sh
LX_PLANNING=1
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_AUDIT_JSONL=/tmp/stage330-stage195-mixed-allocator-lxplanning-2048.jsonl
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --copy-kernel-code \
  --output-dir /tmp/stage330-stage195-mixed-allocator-lxplanning-2048 \
  --fail-on-error
```

No forced-base variables were set:

```text
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS unset
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE unset
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE unset
```

## 2048 Result

Probe:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple
restickifies=1 bytes=8388608 byte_hops=0
Completed 1 rows with 0 errors
```

Audit:

```text
status: patched
replacement_sdsc: 1_MixedReStickifyOpWithPTLxConsumer
producer_lx_unique_starts: [0]
consumer_lx_unique_starts: [262144]
value_flow_contract.valid: true
producer_to_bridge_input_match: true
bridge_output_to_consumer_match: true
```

Endpoint allocation:

```text
producer buffer buf0: [0, 262144)
consumer buffer buf3: [262144, 524288)
overlap_check.valid: true
overlap_check.overlaps: []
```

Core locality:

```text
locality_certified: true
locality_assertion: passed
certified_byte_hops: 0
has_core_mapping_override: true
```

Generated fused-add bundle:

```text
sdsc_0_add.json
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json
```

String check:

```text
sdsc_1_MixedReStickifyOpWithPTLxConsumer.json:
  ReStickifyOpHBM: false
  ReStickifyOpWithPTLx: true
  STCDPOpLx: true
```

## Size Sweep

Allocator-backed sweep:

```text
sizes: 512, 1024, 1536, 2048
case: computed_transpose_adds_then_matmul_tuple
LX_PLANNING=1
forced bases unset
```

Correctness:

```text
512:  ok
1024: ok
1536: ok
2048: ok
```

Patch eligibility:

| Size | Result |
|---:|---|
| 512 | fallback, `producer-endpoint-not-allocator-backed:prototype-default` |
| 1024 | fallback, `producer-endpoint-not-allocator-backed:prototype-default` |
| 1536 | fallback, `producer-endpoint-not-allocator-backed:prototype-default` |
| 2048 | patched, allocator-backed endpoints |

This matches the current full-tensor PT-LX contract: the production-shaped
allocator-backed path is proven for the high-signal 2048 shape.  Smaller shapes
still need the streaming/chunked bridge family, but the currently tested direct
tile spelling is not value-correct.

## Validation

Focused pod tests:

```text
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_scratchpad_patterns.py -q

66 passed, 7 xfailed in 0.91s
```

## Next Step

Package the 2048 allocator-backed path as the narrow production-shaped
prototype:

- keep it default-off;
- require `LX_PLANNING=1`;
- require allocator-backed endpoints and zero-hop locality;
- keep smaller/non-stick-sized cases on `ReStickifyOpHBM`;
- run timing against the stock HBM restickify path for 2048.

The next separate research track is a value-correct streaming/chunked bridge
for 512/1024/1536.
