# Stage 209: Narrow PT-LX Scratchpad Endpoint Requests

## Summary

Stage 209 removes the need for broad scratchpad allowlisting in the validated
PT-LX mixed-schedule path.

Before this stage, allocator-backed endpoint bases required:

```text
LX_PLANNING=1
SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING=1
```

That proved the bridge could consume real `TensorArg.allocation["lx"]` values,
but it was too broad for a production-shaped path. This stage adds a narrow
scratchpad request for only the certified restickify edge:

```text
producer output -> certified ReStickifyOpHBM -> consumer input
```

The generic scratchpad allowlist remains unchanged.

## Code Change

The scratchpad planner now pre-scans operations for certified PT-LX restickify
ops when the mixed-schedule prototype is enabled.

An edge is eligible only when:

- `SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1`
- the restickify source kind is `in_graph_computed`
- a core mapping override exists
- the locality certificate says `locality_certified=true`
- the certified byte-hop cost is exactly `0`
- the source is not a graph input
- the restickify output is not a graph output

For those edges, the planner forces exactly two buffers onto LX:

- the producer output read by the restickify
- the restickify output read by the consumer

The existing generic `core_div_mismatch` guard remains in place for normal LX
reuse, but certified PT-LX endpoint buffers bypass it because the locality
certificate is a stronger, restickify-specific proof than structural
`op_it_space_splits` equality.

## Validation

Focused pod tests:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
13 passed in 0.15s

python -m pytest tests/inductor/test_scratchpad_patterns.py -k certified_ptlx -q
1 passed, 14 deselected in 0.03s
```

Hardware validation without the broad allowlist:

```text
LX_PLANNING=1
SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING unset
case=computed_transpose_adds_then_matmul_tuple
size=2048
restickifies=1
bytes=8388608
byte_hops=0
device_events=0
```

Audit highlights:

```text
producer base_source=op-spec-allocation base=0
consumer base_source=op-spec-allocation base=262144
producer_allocation_patches before_component=lx
value_flow_contract.valid=true
```

## Interpretation

This is the first version where the PT-LX mixed bridge can get real scratchpad
endpoint addresses through a narrow compiler request instead of a global
"allow all ops" scratchpad policy.

The bridge still remains default-off and still requires the prototype flags, but
the allocation story is now much closer to a production lowering:

1. Stage 3B certifies the restickify edge has zero modeled byte-hops.
2. Scratchpad planning reserves the certified producer/restickify endpoint
   buffers.
3. PT-LX endpoint planning consumes the resulting `TensorArg.allocation["lx"]`
   bases.
4. The mixed bridge validates producer -> bridge -> consumer value flow before
   bundle files are written.

## Remaining Work

The next production-shaped step is lifetime/overlap hardening:

- record which buffers were forced for the certified PT-LX edge
- prove the chosen producer and consumer endpoint ranges do not overlap with
  other live LX allocations
- report the endpoint allocation decision in the PT-LX audit row
- then collapse the current cluster of prototype flags into a smaller guarded
  option
