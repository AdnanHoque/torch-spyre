# Stage 328: Direct Tile Diagnostic

## Summary

Stage 328 tested the existing direct tiled PT-LX bridge on the
`matmul_then_add`, size `512` restickify edge after Stage 327 fixed chunked
contract accounting.

This path is useful because it emits the consumer-visible descriptor shape:

```text
bridge layout/stick:   out,mb / mb
consumer layout/stick: out,mb / mb
```

That means it clears the consumer descriptor blocker that the chunked
native-validGap endpoint path still hits.  However, the hardware value check
fails, so this is not a production solution.

## Code Change

`SPYRE_RESTICKIFY_PTLX_FORCE_DIRECT_TILE_E2E=1` was documented as a
validation-only switch that runs the direct tiled candidate as the real mixed
bridge, but the replacement gate still required `production_valid`.

The gate now allows a diagnostic replacement only when:

- the full diagnostic contract is valid;
- the semantic certificate source is one of the explicit forced diagnostic
  sources;
- production validity remains false and visible in telemetry.

This does not change default behavior and does not certify the path for
production.

## Compile-Only Result

Forced direct tile compile-only:

```text
status: patched
replacement_sdsc: 1_StreamingMixedReStickifyOpWithPTLxConsumer
hbm_placements: 0
has_hbm_restickify: false
direct_tile_count: 64
datadsc_count: 128
endpoint_contract_valid: true
consumer_descriptor_valid: true
semantic_transform_certified: true
semantic_certificate_source: forced-direct-tile-env
production_valid: false
```

This proves the candidate can be inserted into the normal mixed schedule and
compiled as a no-HBM bridge for this shape.

## Hardware Value Result

Forced direct tile hardware run at size `512` failed correctness:

```text
Mismatched elements: 175247 / 262144 (66.9%)
Greatest absolute difference: 1.48046875
```

The device remained healthy after the failed value check; a stock tiny
Torch-Spyre smoke returned `2.0`.

## Interpretation

The direct tiled descriptor solves the consumer descriptor shape problem but
does not prove the element coordinate transform.  The current failure means the
direct tile is likely reading producer fragments in the wrong logical coordinate
order or placing them in the wrong consumer coordinates.

The next production-shaped step should not be descriptor relabeling.  It should
return to the value-correct Stage195 full-tensor PT-LX bridge contract and move
that contract from late artifact splice into normal mixed-schedule lowering.

## Validation

Pod unit test:

```text
python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
55 passed in 0.98s
```

Artifacts:

```text
artifacts/stage327_mixed_contract_check/direct_force_compile_512_v2.jsonl
artifacts/stage327_mixed_contract_check/direct_force_run_512.jsonl
artifacts/stage327_mixed_contract_check/direct_force_run_audit_512.jsonl
```
