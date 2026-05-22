# Stage 253: Valid-Gap Consumer-Shaped PT-LX Diagnostic

## Summary

This stage adds a default-off PT-LX diagnostic path that gets closer to a
production-shaped restickify bridge without taking over the stock
`ReStickifyOpHBM` path.

The new guarded selector is:

```sh
SPYRE_RESTICKIFY_PTLX_VALIDGAP_CONSUMER_TILE_E2E=1
```

It emits each 64x64 tile as:

1. `STCDPOpLx` gathers producer-owned fragments into a bounded per-core LX tile
   workspace.
2. `ReStickifyOpWithPTLx` reads an expanded input descriptor
   `out_, mb_, in_` with stick `out_`.
3. The input piece keeps `out_` physically sized at 64, but marks only one
   lane live with `validGap_["out_"] = [[1, 63]]`.
4. The output descriptor is consumer-shaped: `mb_, in_` with stick `in_`.

This satisfies the Deeptools shape constraints we found:

- every output dimension is present in the input descriptor;
- input and output stick dims differ;
- the input still carries a full source-stick span;
- live input elements equal live output elements.

It is still not value-certified. The compiler contract therefore remains
fail-closed: the candidate is audited, but the emitted executable bundle keeps
the stock HBM restickify fallback.

## Compile Probe

The standalone Deeptools probe compiled these valid-gap variants:

| Probe | Result |
|---|---|
| `out_valid1_mb1_in64_to_mb1_in64` | compiled |
| `out_valid1_mb64_in64_to_mb64_in64` | compiled |
| `out_valid8_mb64_in64_to_mb64_in64` | compiled |

The key log line is:

```text
Computing Re-StickifyOpWithPT (Special re-stickify) transfer function..
```

Artifacts:

- `artifacts/stage253_validgap_source_stick/`

## E2E Guarded Probe

Command shape:

```sh
LX_PLANNING=1 \
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7 \
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1 \
SPYRE_RESTICKIFY_PTLX_VALIDGAP_CONSUMER_TILE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576 \
SPYRE_RESTICKIFY_PTLX_BRIDGE_AUDIT_JSONL=/tmp/stage253-validgap-e2e-512/audit.jsonl \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 512 \
  --output-dir /tmp/stage253-validgap-e2e-512 \
  --copy-kernel-code \
  --fail-on-error
```

Results:

| Size | Tiles | Data ops | Candidate HBM placements | Endpoint contract | Consumer descriptor | Value preservation | Semantic certificate | Emitted path |
|---:|---:|---:|---:|---|---|---|---|---|
| 512 | 64 | 128 | 0 | pass | pass | pass | fail | HBM fallback |
| 1024 | 256 | 512 | 0 | pass | pass | pass | fail | HBM fallback |

The semantic certificate deliberately fails with:

```text
validgap-consumer-ptlx-tile-needs-hardware-value-validation
```

This is the right state for this stage. The candidate now satisfies the
compiler-side descriptor and live-element contracts across non-2048 sizes, but
we have not proven that hardware execution produces the same values as the
stock HBM restickify.

Artifacts:

- `artifacts/stage253_validgap_e2e_512/`
- `artifacts/stage253_validgap_e2e_1024/`

## Tests

Ran in the Spyre pod:

```text
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_dataop.py \
  tests/inductor/test_restickify_mapping_alignment.py -q

63 passed in 6.50s
```

## Current Interpretation

This stage moves the PT-LX path from "descriptor shape mismatch" to
"descriptor and element-count contracts pass, but value semantics still need
hardware validation."

The next step is to run the valid-gap candidate as the actual bridge inside a
small controlled value-correct graph, still behind an explicit force flag, and
compare the bridge output before matmul. Only after that passes should this
path be allowed to replace `ReStickifyOpHBM`.
