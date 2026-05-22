# Stage 306: Streaming PT-LX Value-Flow Gate Tightening

## Summary

This stage tightened the verifier around the production-shaped PT-LX path
without making the PT-LX path default-on.

The key distinction is now explicit:

- a direct 64x64 PT-LX bridge may have a valid LX endpoint contract even when
  it scatters a tile into multiple consumer-local LX offsets;
- that endpoint validity is still not enough to replace `ReStickifyOpHBM`
  unless the bridge also has a semantic transform certificate.

## Code Changes

- `_streaming_value_flow_contract(...)` now accepts scattered consumer LX
  offsets for direct 64x64 PT-LX bridges when:
  - the minimum output start equals the planned consumer base;
  - every output start stays inside the bounded per-core LX allocation window;
  - the bridge remains HBM-free and uses no `ReStickifyOpHBM`.
- The contract reports `consumer_start_valid` separately from
  `endpoint_contract_valid`, so endpoint failures are easier to distinguish
  from semantic transform failures.
- The older full-tensor mixed PT-LX replacement now uses candidate copies for
  producer/consumer endpoint patching. It commits the replacement only after
  `_mixed_value_flow_contract(...)` passes.
- If the mixed value-flow contract fails, the pass returns:

```text
status: skipped
reason: mixed-value-flow-contract-invalid
fallback: ReStickifyOpHBM
```

and leaves the original producer, `ReStickifyOpHBM`, and consumer SDSCs in
place.

## Why This Matters

The wide tiled path needs scatter-style consumer endpoints. A 512 example with
destination split `mb:32,out:1` produces 64 scatter operations and multiple
consumer LX start offsets within the consumer allocation. Before this stage the
verifier treated that as an invalid endpoint because it expected a singleton
consumer base.

That was too strict for the target design. The target design is not "one LX
start per consumer tensor"; it is "all consumer-visible pieces live inside the
planned LX allocation and match the consumer descriptor."

This stage fixes that part while preserving the important safety rule:

```text
endpoint valid + semantic uncertified => keep ReStickifyOpHBM
```

## Validation

Pod validation:

```sh
python -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tests/inductor/test_restickify_lx_dataop.py

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
```

Result:

```text
47 passed
```

## Status

This is still not the final production implementation. It moves the compiler
contract closer to the requested end state by making the verifier understand
the shape of a valid tiled consumer scatter and by ensuring mixed PT-LX
replacement is committed only when producer, bridge, and consumer endpoint
metadata agree.

The remaining blocker is unchanged: the direct tiled PT-LX transform still
reports `semantic_transform_certified: false`, so it is not allowed to replace
the stock HBM fallback.
