# Stage 208: Allocator-Aware PT-LX Endpoint Bases

## Summary

Stage 208 moves the PT-LX mixed-schedule prototype one step closer to normal
Torch-Spyre lowering by allowing the endpoint plan to use existing scratchpad
allocation metadata.

Before this stage, the planned producer and consumer LX endpoints always used
fixed prototype bases unless explicit debug environment overrides were set:

- producer endpoint: `16384`
- consumer endpoint: `8192`

This stage changes the planning order to:

1. explicit debug environment override, if present
2. `TensorArg.allocation["lx"]`, if scratchpad planning assigned one
3. prototype fallback base

The old fallback path is preserved, so the validated no-LX-planning path keeps
working.

## Code Change

`PTLXLXEndpointPlan` now records both:

- `base`
- `base_source`

The source is one of:

- `env:<variable>`
- `op-spec-allocation`
- `prototype-default`

The mixed bridge materialization still uses the same endpoint contract as
Stage 207. The only change is where the endpoint base comes from.

## Validation

Focused pod validation:

```text
python3 -m py_compile torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py tests/inductor/test_restickify_lx_dataop.py
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q

13 passed in 0.22s
```

Default fallback hardware guardrail:

```text
case=computed_transpose_adds_then_matmul_tuple
size=2048
restickifies=1
bytes=8388608
byte_hops=0
device_events=0
producer base_source=prototype-default base=16384
consumer base_source=prototype-default base=8192
value_flow_contract.valid=true
```

Allocator-backed hardware validation with scratchpad planning enabled:

```text
LX_PLANNING=1
SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING=1
case=computed_transpose_adds_then_matmul_tuple
size=2048
restickifies=1
bytes=8388608
byte_hops=0
device_events=0
producer base_source=op-spec-allocation base=0
consumer base_source=op-spec-allocation base=262144
value_flow_contract.valid=true
```

## Interpretation

This is the first version where the PT-LX bridge can consume real scratchpad
planner addresses instead of only using hand-picked prototype offsets.

The important detail is that the producer was already emitted as LX-resident in
the allocator-backed run:

```text
producer_allocation_patches: before_component=lx
```

So the bridge endpoint no longer has to invent the producer side. It reads the
planned producer LX base and patches the bridge to that same address. The
consumer endpoint follows the same pattern.

## Remaining Work

The path still depends on broad LX-planning enablement for this synthetic
example because the current scratchpad allowlist does not normally allocate all
pointwise producer/consumer values to LX.

Next production-shaped steps:

1. narrow the scratchpad planner hook so this specific producer/restickify/
   consumer edge can request LX endpoints without enabling
   `SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING`
2. add overlap/lifetime checks for the chosen producer and consumer endpoint
   bases
3. replace the prototype flags with a single guarded compiler option once the
   endpoint request and lifetime checks are explicit
