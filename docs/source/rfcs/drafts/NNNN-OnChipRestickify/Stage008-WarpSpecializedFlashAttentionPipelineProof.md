# Stage 008: Warp-Specialized Flash Attention Pipeline Proof

Date: 2026-05-25

## Purpose

Stage 004 showed that the production SDPA score handoff can keep the QK score
matrix on chip for the first softmax consumers.  Stage 008 starts the next step:
a proof-shaped mixed-SDSC schedule for flash-attention prefill where data
movement and compute can be expressed as a double-buffered pipeline.

On AIU this is the analog of warp specialization.  The scheduler roles are not
CUDA warps; they are mixed SuperDSC rows.  Data-op rows stage the next tile into
LX while DL rows compute on the current tile.  The first implementation is
descriptor-only so we can validate the contract before wiring it into the
compiler SDPA path.

## Implementation

Code changes:

- `torch_spyre/_inductor/codegen/onchip_bridge.py`
  - Added `FLASH_PIPELINE_TILE_BYTES`.
  - Added `allocate_flash_attention_pipeline_bases`.
  - Added `flash_pipeline_schedule`.
  - Added `build_flash_attention_pipeline_bridge`.
- `torch_spyre/_inductor/config.py`
  - Added `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE`.
  - Added `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP`.
- `tests/_inductor/test_onchip_flash_pipeline_logic.py`
  - Added standalone descriptor tests for allocation, K/V lane staging,
    ping-pong buffer alternation, serial schedule fallback, and overlap schedule
    rows.

The helper stages one or more LX-resident payload lanes through ping-pong
buffers using `STCDPOpLx`.  The attention target is two lanes, K and V, but the
builder is generic so a one-lane score-tile proof can use the same schedule
machinery.

## Schedule Shape

The safe control is serial double buffering:

```text
prefetch K[t]
prefetch V[t]
compute tile t
```

The candidate overlap schedule is:

```text
prefetch K[0], V[0]
prefetch K[t+1] + compute tile t
prefetch V[t+1]
compute last tile
```

Rows that contain both a data-op index and a DL-op index are intentionally
behind `SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP=1`.  Until a Foundation/DXP
run proves those rows execute concurrently and value-correctly, the production
path must use the serial fallback.

## Important Limitation

This stage does not claim HBM-to-LX K/V prefetch.  The only certified movement
primitive used here is same-stick LX-to-LX `STCDPOpLx`, matching the core-to-core
handoff recipe.  A full flash-attention compiler realization needs either:

- K/V tiles already produced into LX by an earlier op, or
- a separately certified HBM-to-LX data movement contract.

That distinction is why this stage is a proof artifact rather than a compiler
replacement for SDPA.

## Validation Plan

Initial validation is local and torch-free:

```text
tests/_inductor/test_onchip_flash_pipeline_logic.py
tests/_inductor/test_onchip_streaming_logic.py
tests/_inductor/test_onchip_realize_logic.py
tests/_inductor/test_onchip_handoff_logic.py
```

Device validation is deferred until pod access is available again.  The next
device step should generate both serial and overlap mixed SDSCs, inspect
`senprog.txt` for `L3_LDU/L3_STU`, and only claim overlap if the runtime gives
value correctness without scheduler or Compute CB errors.

## Conclusion

The mixed-SDSC architecture now has a reusable schedule/data-op proof for
double-buffered flash-attention staging.  This is the right bridge between the
current score-handoff implementation and a future Inductor-level flash-prefill
realizer: it isolates the schedule contract first, keeps uncertified HBM-load
behavior out of the claim, and gives us a serial fallback for device bring-up.
