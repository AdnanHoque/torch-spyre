# Stage 252: PT-LX Value-Preservation Guard

## Summary

Stage 252 adds another fail-closed verifier for streaming PT-LX bridge
candidates. In addition to checking LX endpoints and consumer descriptors, the
compiler now checks that every `ReStickifyOpWithPTLx` data-op preserves live
element count between its input and output LDS descriptors.

This is needed because Stage 251 found expanded descriptors that compile in
Deeptools but are not yet safe tensor transformations: some compile only by
introducing an extra live axis.

## Contract

For each `ReStickifyOpWithPTLx` data-op:

```text
sum(input PieceInfo live elements) == sum(output PieceInfo live elements)
```

The helper understands normal `validGap_` entries of the form:

```json
{"dim": [[valid, gap]]}
```

If a candidate introduces extra live coordinates, the verifier reports:

```text
restickify-live-element-count-mismatch
```

## Validation

Focused pod tests:

```text
python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_mapping_alignment.py \
  -q
```

Result:

```text
61 passed in 8.38s
```

The tests cover both:

- direct 64x64 tiles, where input and output are both 4096 live elements; and
- an expanded descriptor with `out=64, mb=64, in=64` input and `mb=64, in=64`
  output, which is rejected as a live-element mismatch.

## Probe Audit

The guarded `adds_then_matmul`, size `512`, run still passes via stock HBM
fallback. The direct-tile candidate now records three independent gates:

```json
{
  "endpoint_contract_valid": true,
  "consumer_descriptor_valid": false,
  "value_preservation_valid": true,
  "semantic_transform_certified": false,
  "valid": false
}
```

This is the desired state: the direct-tile candidate preserves element count,
but does not match the matmul consumer descriptor and is not semantically
certified. Therefore the compiler skips it and keeps `ReStickifyOpHBM`.

Artifacts:

- `artifacts/stage252_value_preservation_guard/audit_512.jsonl`
- `artifacts/stage252_value_preservation_guard/restickify_scenarios_512.csv`

## Next Step

The next candidate generator must satisfy all verifier layers at once:

1. LX endpoint contract
2. Consumer descriptor contract
3. Value-preservation contract
4. Semantic transform certificate

For the matmul-input family, the remaining open problem is to generate a
consumer-shaped PT-LX bridge that matches `mb/in` stick `in` without
introducing extra live source coordinates.
