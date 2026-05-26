# Stage 014: Flash Value-Flow Tile Probe

Date: 2026-05-26

## Purpose

Stage 013 proved that generated flash-prefill `batchmatmul` tiles can be
replaced by mixed SuperDSCs and still pass device value correctness.  Those
mixed tiles were execution-shaped but not value-flow-shaped: the `STCDPOpLx`
data-ops ran next to the compute DSC, while the compute DSC still consumed its
original HBM inputs.

Stage 014 adds the first stricter value-flow probe.  For one requested flash
tile, the compiler looks for a real latest-producer edge feeding a
single-consumer `batchmatmul` input.  If the producer output and consumer input
are same physical stick layout, the producer output and consumer input are
flipped to LX, two `STCDPOpLx` roundtrip rows are folded into a one-compute mixed
tile sidecar, and `bundle.mlir` executes that sidecar in place of the generated
consumer SDSC.

This is intentionally stricter than the Stage 013 replacement.  If the edge is
layout-changing, shared by another consumer, over LX capacity, or otherwise not
same-stick Tier 1, it fails closed and keeps the generated HBM-backed SDSC.

## Implementation

New diagnostic gate:

```sh
SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE=<tile-index>
```

Code changes:

- `torch_spyre/_inductor/config.py`
  - Added `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE`.
- `torch_spyre/_inductor/onchip_realize.py`
  - Added `build_flash_attention_value_flow_tile_artifact`.
  - Added latest-producer lookup by HBM base address.
  - Requires exactly one future consumer for the selected producer address.
  - Requires same physical stick layout, allowing `out_`/`in_` stick renaming
    only when stick position and paired extents match.
  - Flips the producer output and consumer input to LX and builds a
    two-`STCDPOpLx` roundtrip before the copied compute DSC.
- `torch_spyre/_inductor/codegen/bundle.py`
  - Executes the value-flow sidecar when realization succeeds.
  - Gives value-flow replacement precedence over the older generic tile
    replacement.
  - Logs a warning when the requested value-flow tile is not realizable and the
    compiler keeps the generated HBM-backed SDSC.
- `tests/_inductor/test_onchip_realize_logic.py`
  - Added positive coverage for a synthetic same-stick producer
    `batchmatmul -> batchmatmul` value-flow tile.
  - Added negative coverage for missing producer and multi-consumer cases.

## Validation

Local standalone tests:

```text
tests/_inductor/test_onchip_realize_logic.py         19/19 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
py_compile(config.py, onchip_realize.py, bundle.py)   passed
git diff --check                                      passed
```

Pod:

```text
adnan-cdx-spyre-dev-pf
DTI_PROJECT_ROOT=/home/adnan-cdx/dt-inductor-mixed
PATCHED_DXP=/home/adnan-cdx/dt-inductor-mixed/build/deeptools-onchip-foundation-clean/dxp/dxp_standalone
worktree=/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Pod standalone tests after fast-forwarding to `c8d0880`:

```text
tests/_inductor/test_onchip_realize_logic.py         19/19 passed
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
```

Device command:

```sh
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1
export SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE=0
export DXP_DEBUG=1
export TORCHINDUCTOR_CACHE_DIR=/tmp/sdpa-value-flow-tile0-bg-1779819565
"$PYTHON" -m pytest tests/inductor/test_building_blocks.py \
  -k "flash_attention_mixed_pipeline_selects_prefill" -q -s
```

Result:

```text
1 passed, 6 deselected in 14.61s
```

The value-flow probe intentionally did not replace the generated real SDPA
tiles for this shape.  `bundle.mlir` kept:

```text
sdsc_5_batchmatmul.json
```

and emitted only the non-executed generic sidecar:

```text
sdsc_mixed_flash_pipeline_tile_0.json
```

## Real SDPA Rejection Evidence

For the first generated flash bundle, the only `batchmatmul` tile was
`5_batchmatmul`.

Input 0:

```text
producer: 2_mul output 2
consumer: 5_batchmatmul input 0
future consumers: [5_batchmatmul input 0]
producer layout/stick: [x_, mb_, out_] / out_
consumer layout/stick: [mb_, x_, in_] / in_
decision: reject, non-stick layout order changes x_ <-> mb_
```

Input 1:

```text
producer: 4_ReStickifyOpHBM output 1
consumer: 5_batchmatmul input 1
future consumers: [5_batchmatmul input 1, 7_maxnonstick input 1]
producer layout/stick: [x_, mb_, out_] / out_
consumer layout/stick: [in_, x_, out_] / out_
decision: reject, fanout plus layout mismatch
```

The second generated flash bundle had three `batchmatmul` tiles.  All candidate
real producer edges were rejected for the same reason: generated flash prefill
still contains layout-changing producer-to-consumer boundaries, not pure Tier 1
same-stick handoffs.

## Interpretation

The Stage 014 code path is now production-shaped for an eligible real value-flow
edge, and the unit test proves the descriptor mutation and mixed replacement
shape.  The generated SDPA prefill graph tested here does not expose such an
edge.  It needs either a certified PT-LX/restickify path for the layout-changing
boundaries or an upstream flash-attention lowering that produces same-layout
K/V tile feeds directly.

This keeps the current architecture honest: Tier 1 `STCDPOpLx` remains
same-stick only, and real flash attention does not claim on-chip value-flow
success until the layout transform piece is certified.
