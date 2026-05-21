# Stage 207: Planned PT-LX Bridge Endpoint Bindings

## Summary

Stage 207 moves one more piece of the PT-LX mixed-schedule prototype from
emission-time inference into the planned lowering contract.

The previous stage introduced explicit producer and consumer LX endpoint plans:

- producer output endpoint: LX base `16384`
- consumer input endpoint: LX base `8192`

This stage makes the bridge data-op endpoint `PieceInfo` bindings derive from
that plan instead of rebuilding the per-core start maps inline in the mixed
schedule emitter.

## Code Change

The bridge endpoint patch now flows through a small materialization helper:

- `_materialize_bridge_lx_endpoints(...)`
- `_endpoint_core_starts(...)`

The helper turns the planned producer and consumer endpoints into the per-core
start maps used by `_patch_bridge_endpoint_pieces(...)`.

This keeps the current behavior unchanged, but makes the contract clearer:

1. plan producer/consumer LX endpoints
2. materialize bridge endpoint pieces from that plan
3. emit the mixed producer -> bridge -> consumer bundle

No default behavior changes. The path still requires the existing prototype
flags.

## Validation

Focused unit/static validation in the pod:

```text
python3 -m py_compile torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py tests/inductor/test_restickify_lx_dataop.py
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q

12 passed in 0.19s
```

Hardware validation on the high-signal `2048` case:

```text
case=computed_transpose_adds_then_matmul_tuple
size=2048
restickifies=1
bytes=8388608
byte_hops=0
device_events=0
```

Audit highlights:

```text
replacement_sdsc: 1_MixedReStickifyOpWithPTLxConsumer
producer endpoint base: 16384
consumer endpoint base: 8192
producer -> bridge input match: true
bridge output -> consumer match: true
value_flow_contract.valid: true
```

## Interpretation

This does not yet make the feature production-ready. It does remove another
small ad hoc emission-time decision and turns it into an explicit planned
binding. The producer, bridge, and consumer now agree through a single planned
endpoint contract for the validated path.

The next step is allocator-aware endpoint planning: replace the fixed prototype
LX bases with endpoint bases chosen from the real LX allocation plan.
