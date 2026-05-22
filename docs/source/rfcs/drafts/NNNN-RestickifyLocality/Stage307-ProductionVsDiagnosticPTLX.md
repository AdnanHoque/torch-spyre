# Stage 307: Separate Production PT-LX Proof From Diagnostic Force

## Summary

This stage made the PT-LX value-flow contract distinguish a compiler proof from
a diagnostic force flag.

Before this stage, a validation-only flag such as
`SPYRE_RESTICKIFY_PTLX_FORCE_DIRECT_TILE_E2E=1` could make
`_streaming_value_flow_contract(...)` report `valid: true`. That is useful for
hardware experiments, but it is not a production proof that the compiler can
replace `ReStickifyOpHBM`.

The contract now reports both:

```text
valid              # true for diagnostic forced probes or real certificates
production_valid   # true only for non-forced compiler/bridge metadata proof
```

## Code Changes

- `_streaming_semantic_transform_certificate(...)` now returns structured
  certificate metadata:
  - `certified`
  - `reason`
  - `source`
  - `forced`
- `_streaming_value_flow_contract(...)` now includes:
  - `semantic_transform_forced`
  - `semantic_certificate_source`
  - `production_valid`
- Metadata-certified bridges such as same-layout LX ownership remaps report:

```text
semantic_certificate_source: bridge-metadata
semantic_transform_forced: false
production_valid: true
```

- Diagnostic-forced PT-LX variants report:

```text
semantic_certificate_source: forced-...-env
semantic_transform_forced: true
valid: true
production_valid: false
```

## Why This Matters

The requested end state is not just "make a PT-LX path run." It is:

```text
only enabling the path when the compiler can prove a valid
producer-bridge-consumer LX contract
```

This stage prevents diagnostic experiments from being mistaken for that proof.
It keeps the stock `ReStickifyOpHBM` fallback as the production behavior for
uncertified PT-LX layout transforms, even when developers use force flags to
collect hardware data.

## Validation

Pod validation:

```sh
python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  -q
```

Result:

```text
58 passed
```

## Status

This stage does not make the streaming PT-LX layout transform value-correct.
It tightens the production gate so that once a future tiled bridge does become
value-correct, the compiler can distinguish that proof from a diagnostic escape
hatch.
