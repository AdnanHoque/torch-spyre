# Stage 244: Streaming PT-LX Semantic Guard

## Summary

The streaming PT-LX prototype previously marked its value-flow contract valid
when the generated bridge had no HBM placements and the producer/consumer LX
endpoint bases matched. Hardware correctness checks showed that this was too
weak: the bridge could be LX-only and still produce wrong values because the
generated data-op sequence did not prove the restickify coordinate transform.

This stage tightens the verifier:

- `endpoint_contract_valid` means the bridge is HBM-free and points at the
  intended LX producer/consumer bases.
- `semantic_transform_certified` means the bridge has also proven the logical
  restickify value transform.
- `valid` is now the conjunction of both fields.

The current streaming `STCDPOpLx -> ReStickifyOpWithPTLx -> STCDPOpLx` shape
keeps `semantic_transform_certified=false`.

## Why

Deeptools inspection showed that plain `STCDPOpLx` creates subpieces from
overlapping input/output coordinate regions. It is good for LX movement when
the coordinate spaces overlap, but it is not a general gather/scatter
coordinate remapper.

That explains the Stage 243 results:

- global-coordinate intermediate pieces can compile but are not value-correct;
- compact tile-local intermediate pieces fail Deeptools coverage checks;
- therefore endpoint matching alone is not enough to certify a streaming
  PT-LX restickify.

## Validation

Pod tests:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_mapping_alignment.py -q

52 passed in 0.39s
```

Compile-only guard with the risky cross-bundle streaming path and
`SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1` now stops before launch:

```text
RuntimeError: cross-bundle streaming PT-LX restickify value-flow contract failed
semantic_skip_reason:
  streaming-ptlx-stcdp-gather-scatter-does-not-certify-coordinate-remap
endpoint_contract_valid: True
semantic_transform_certified: False
```

## Interpretation

This does not finish the production-shaped wide-size PT-LX path. It makes the
prototype safer and more honest: the compiler can still build/audit the
streaming shape under explicit prototype flags, but the assert gate no longer
allows it to claim value-flow validity.

The next implementation step is to replace the uncertified STCDP gather/scatter
remap with one of:

- a Deeptools-native coordinate-remap data movement primitive;
- an `InputFetchNeighbor`-backed bridge;
- a hardware-validated `ReStickifyOpWithPTLx` contract that directly consumes
  producer LX pieces and writes the consumer LX view.
