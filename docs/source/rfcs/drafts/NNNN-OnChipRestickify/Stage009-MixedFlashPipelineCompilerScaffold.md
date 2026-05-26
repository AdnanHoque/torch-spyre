# Stage 009: Mixed Flash Pipeline Compiler Scaffold

Date: 2026-05-26

## Purpose

Stage 008 added the low-level double-buffered data-op schedule proof for a
warp-specialized flash-attention pipeline.  Stage 009 moves that proof one step
closer to the compiler path by:

- making the mixed-pipeline flag select the existing Inductor flash-prefill
  decomposition; and
- adding a production-shaped mixed SuperDSC wrapper for tiled flash compute DSCs.

This still is not the final SDPA replacement.  It is the scaffold that lets the
next step fold real generated flash-prefill tile compute SDSCs behind the
already-tested data-op schedule.

## Implementation

Code changes:

- `torch_spyre/_inductor/decompositions.py`
  - `_can_use_flash_attention_prefill` now accepts either
    `SPYRE_FLASH_ATTENTION_PREFILL=1` or
    `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE=1`.
  - This lets the mixed pipeline flag opt into the online-softmax prefill graph
    without also setting the older generic flash-prefill flag.
- `torch_spyre/_inductor/codegen/onchip_bridge.py`
  - Added `build_flash_attention_pipeline_mixed_sdsc`.
  - The helper wraps tiled compute DSCs with `datadscs_`, `opFuncsUsed_`, and
    `coreIdToDscSchedule` into one mixed SuperDSC body.
  - The helper validates every schedule row before emitting JSON, including
    data-op index bounds and compute DSC index bounds.
- `tests/_inductor/test_onchip_flash_pipeline_logic.py`
  - Added coverage for the mixed SuperDSC wrapper and invalid schedule refs.
- `tests/inductor/test_building_blocks.py`
  - Added a test that the mixed pipeline flag selects the flash-prefill
    decomposition even when `flash_attention_prefill` is false.

## Why This Step

The previous proof could build the data-op side and schedule shape, but it did
not yet have an artifact shaped like a real mixed SDSC with multiple compute
tiles.  A compiler realizer needs that exact shape before it can safely fold
real SDPA tile compute into one descriptor.

This step keeps the design honest:

- serial schedule remains the safe default;
- overlap rows are represented but still opt-in;
- the code still does not claim HBM-to-LX K/V prefetch;
- `STCDPOpLx` remains the only movement primitive used by the proof.

## Validation

Local standalone checks:

```text
tests/_inductor/test_onchip_flash_pipeline_logic.py   9/9 passed
tests/_inductor/test_onchip_streaming_logic.py        9/9 passed
tests/_inductor/test_onchip_realize_logic.py         13/13 passed
tests/_inductor/test_onchip_handoff_logic.py          3/3 passed
```

Static checks:

```text
python3 -m py_compile torch_spyre/_inductor/codegen/onchip_bridge.py \
  torch_spyre/_inductor/decompositions.py \
  tests/_inductor/test_onchip_flash_pipeline_logic.py
git diff --check
```

Both passed locally.  Device execution and senprog evidence remain the next
step after this scaffold is synced to the pod branch.

## Next Step

Use this wrapper to combine real flash-prefill tile compute SDSCs from the
Inductor decomposition into one mixed SDSC, first in serial mode, then with
`SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1` only after Foundation/DXP
accepts and executes rows containing both data-op and DL-op indices.
