# Stage 210: PT-LX Endpoint Lifetime Audit

## Summary

Stage 210 hardens the narrow PT-LX scratchpad endpoint path by recording the
actual endpoint allocations and validating their live LX ranges before codegen.

Stage 209 proved that certified PT-LX restickify edges can request only the
producer and restickify-output buffers on LX, without enabling the broad
scratchpad allowlist. This stage makes that decision visible and auditable.

## Code Change

The scratchpad planner now attaches a `ptlx_endpoint_allocation` record to a
certified restickify op when both endpoint buffers are allocated on LX.

The record includes:

- producer endpoint buffer
- consumer/restickify-output endpoint buffer
- per-core LX address range for each endpoint
- live LX ranges at the restickify op
- an explicit overlap check

If either endpoint overlaps with another live LX allocation, the compiler raises
before bundle emission.

`SpyreKernel` carries this record into `OpSpec.op_info`, and the PT-LX mixed
schedule audit row emits it as `endpoint_allocation`.

## Validation

Focused pod tests:

```text
python -m pytest tests/inductor/test_scratchpad_patterns.py -k certified_ptlx -q
1 passed, 14 deselected in 0.18s

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
13 passed in 0.08s
```

Hardware validation:

```text
LX_PLANNING=1
SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING unset
case=computed_transpose_adds_then_matmul_tuple
size=2048
restickifies=1
bytes=8388608
byte_hops=0
device_events=0
value_flow_contract.valid=true
```

Audit highlights:

```text
producer base_source=op-spec-allocation base=0
consumer base_source=op-spec-allocation base=262144

endpoint_allocation:
  producer_buffer=buf0
  producer=[0, 262144)
  consumer_buffer=buf3
  consumer=[262144, 524288)
  live_lx_ranges=[buf0, buf3]
  overlap_check.valid=true
  overlap_check.overlaps=[]
```

## Interpretation

The PT-LX path now has a visible allocation chain:

1. Stage 3B certifies the restickify edge has zero modeled byte-hops.
2. Scratchpad planning forces only the certified endpoint buffers onto LX.
3. Scratchpad planning records the selected endpoint ranges and proves no live
   overlap for those endpoints.
4. `SpyreKernel` serializes the record into `OpSpec.op_info`.
5. PT-LX mixed lowering consumes the real `TensorArg.allocation["lx"]` bases and
   emits the allocation record in the audit row.

This still remains default-off, but the path is no longer a blind endpoint
rewrite. It now has allocation provenance and an overlap certificate.

## Remaining Work

The next step is to reduce the prototype surface area:

- collapse the flag cluster into one guarded PT-LX prototype option
- keep telemetry/locality/assert flags as diagnostics
- add a negative overlap/unit test with a deliberately conflicting allocator
- run a small timing check comparing stock HBM restickify and PT-LX mixed
  restickify with `LX_PLANNING=1`
