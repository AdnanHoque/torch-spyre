# Stage 220: Streaming PT-LX Lowering Contract

## Summary

Started the production-shaped PT-LX fix by adding an explicit streaming
contract, still default-off and still fail-closed. This does not delete the HBM
path. It gives the compiler a structured answer for skipped PT-LX cases:

- what tile movement phases are required,
- whether bounded tile workspace fits in the 2 MB/core LX scratchpad,
- whether the edge needs gather, scatter, or core-count adaptation,
- and why the current full-tensor bridge is insufficient.

The new flag is:

```text
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
```

## What Changed

- Added `restickify_ptlx_streaming_e2e` config.
- Added `streaming_ptlx_contract(...)`.
- Extended PT-LX skip audit rows so skipped full-bridge cases can report a
  `streaming_ptlx_candidate` contract.
- Kept behavior default-off and preserved `ReStickifyOpHBM` fallback.

For a `512x512` skipped case, the audit now reports:

```json
{
  "reason": "producer-endpoint-not-allocator-backed:prototype-default",
  "status": "skipped",
  "streaming_ptlx_candidate": {
    "available": true,
    "notes": ["source-piece-smaller-than-tile"],
    "max_fan_in": 4,
    "tile_buffer_bytes": 8192,
    "contract": {
      "phases": [
        "gather-source-fragments",
        "local-ptlx-restickify",
        "write-dest-tile"
      ],
      "bounded_workspace_bytes": 24576,
      "fits_lx_workspace": true,
      "requires_gather": true,
      "requires_scatter": false
    }
  }
}
```

## Why This Is Production-Shaped

The core production rule is:

```text
emit PT-LX only when the full edge contract is representable;
otherwise keep ReStickifyOpHBM
```

The previous full-bridge prototype only had one representable shape in this
family: `2048x2048`. The streaming contract generalizes the decision so future
lowering can handle:

- `512/1024/1536`: gather multiple sub-stick producer fragments into one tile;
- `3072`: adapt unequal producer/restickify core-count ownership;
- `4096+`: reuse a small tile buffer instead of reserving full producer,
  intermediate, and consumer ranges in LX.

## Validation

Local static validation:

```sh
python3 -m py_compile \
  torch_spyre/_inductor/codegen/restickify_ptlx_streaming.py \
  torch_spyre/_inductor/codegen/restickify_ptlx_boundary.py \
  tools/restickify_tile_ownership_probe.py \
  tests/inductor/test_restickify_tile_ownership_probe.py \
  torch_spyre/_inductor/config.py
```

Pod tests:

```sh
python -m pytest tests/inductor/test_restickify_tile_ownership_probe.py -q
# 10 passed

python -m pytest tests/inductor/test_restickify_lx_dataop.py -q
# 18 passed
```

Pod audit smoke:

```sh
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1 \
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1 \
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --ring-telemetry \
  --skip-correctness \
  --skip-kernel-launch
```

Result: stock HBM fallback remains, but the audit now carries the streaming
contract the next lowering phase needs.

## Next Step

The next stage should consume the contract and emit the first codegen-only
streaming data-op artifact for `512`:

1. one or more `STCDPOpLx` gather data ops for source fragments,
2. one local `ReStickifyOpWithPTLx` tile restickify,
3. one destination write/consumer handoff,
4. no generated `ReStickifyOpHBM` for the patched boundary,
5. no hardware launch until the JSON artifact passes static inspection.
