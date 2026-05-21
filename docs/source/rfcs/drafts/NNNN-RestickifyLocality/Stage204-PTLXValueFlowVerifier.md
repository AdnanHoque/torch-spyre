# Stage 204: PT/LX Mixed Bridge Value-Flow Verifier

## Summary

Stage203 showed that the mixed PT/LX bridge can replace the stock
`ReStickifyOpHBM` path for the high-signal 2048 in-graph case and run
value-correctly on hardware.

This stage adds a narrow verifier for the generated mixed bundle contract.  It
checks the SDSC JSON before bundle files are written and confirms that:

- the producer output allocation is LX-backed;
- the first bridge data-op input pieces read the same per-core LX addresses;
- the last bridge data-op output pieces write the expected per-core LX
  addresses;
- the consumer input reads those same bridge output LX addresses.

The verifier is default-off behind:

```text
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
```

## Code Changes

- Added `restickify_ptlx_value_flow_assert` config.
- Added `_mixed_value_flow_contract(...)` in
  `codegen/restickify_ptlx_boundary.py`.
- The mixed schedule audit row now includes a `value_flow_contract` summary.
- Added unit coverage for:
  - matching producer/bridge/consumer LX endpoints;
  - an intentionally corrupted bridge input endpoint.

## Contract Shape

For the 2048 probe, the verifier reported:

```json
{
  "valid": true,
  "producer_to_bridge_input_match": true,
  "bridge_output_to_consumer_match": true,
  "producer_core_count": 32,
  "bridge_input_core_count": 32,
  "bridge_output_core_count": 32,
  "consumer_core_count": 32,
  "producer_unique_starts": [16384],
  "bridge_input_unique_starts": [16384],
  "bridge_output_unique_starts": [8192],
  "consumer_unique_starts": [8192]
}
```

In simple terms:

```text
producer LX output -> bridge input:  same address on every core
bridge output      -> consumer input: same address on every core
```

## Validation

Focused unit tests:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
8 passed
```

Compile-only 2048 probe with the verifier enabled:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

Hardware 2048 probe with the verifier enabled:

```text
ok size=2048 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=8388608 byte_hops=0 device_events=0
```

Both runs emitted:

```text
replacement_sdsc: 1_MixedReStickifyOpWithPTLxConsumer
value_flow_contract.valid: true
producer_to_bridge_input_match: true
bridge_output_to_consumer_match: true
```

## Interpretation

This does not make the PT/LX bridge production-ready by itself.  It does make
the prototype safer and easier to reason about: when the verifier is enabled,
Torch-Spyre will fail before launch if the generated mixed bridge no longer
connects the producer, data-op bridge, and consumer through a single coherent
LX value-flow contract.

The remaining production work is still to remove JSON patching as the main
implementation technique and express this contract as planned lowering data
before SDSC generation.
