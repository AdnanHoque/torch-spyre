# SwiGLU On-Chip Movement Diagnostics

This branch tracks experiments toward a scalable LX-to-LX movement path for
SwiGLU-style matmul-to-pointwise reshards.  The current target shape is the
matmul output layout used by the small smoke:

- producer: `{mb:4, out:8}`, logical layout `["mb", "out", "x"]`
- consumer: `{mb:32}`, logical layout `["out", "mb"]`
- tensor: `[1, 512, 512]` activation represented as `[512, 8, 1, 64]`

## Current Status

There is no value-correct working prototype yet for this reshaping handoff.

The implementation experiment is split across four sibling branches, all
branched from `a344643920f10674139425bc863677506af445f5`:

- `swiglu-ws-dxp`: continue the current mixed-SDSC `STCDPOpLx`/DXP probe path.
- `swiglu-ws-co-remap`: prototype a native logical coordinate-remap movement
  primitive.
- `swiglu-ws-input-fetch`: test whether InputFetchNeighbor can carry general
  `mb/out` relayouts.
- `swiglu-ws-existing-ops`: test whether existing LX-capable ops can express
  the relayout without changing the matmul split.

The planner and mixed-SDSC artifact generation can identify and emit a
candidate movement.  Unit coverage verifies the common-refinement cells,
separate producer/consumer logical layouts, collapsed size-one `x`, schedule
ordering, and the diagnostic `dense_actual` output-piece mode.

The AIU smoke result is still negative:

- `valid_gap` output pieces compile with the DXP physicalizer but produced NaNs
  before the STCDP overlap probe.
- `dense_actual` output pieces avoid NaNs but are value-wrong versus the HBM
  baseline: max difference `0.362060546875`, mean `0.053619384765625`.
- a Deeptools STCDP valid-gap overlap probe lets the isolated sparse `p2`
  bundle pass DXP coverage, but the full hardware smoke timed out after 280 s
  with an RB timeout.

## Mixed STCDP Findings

Plain `STCDPOpLx` is an overlap copier.  It is not a general coordinate-remap
or scatter primitive.  The current mixed carrier only works for this reshaping
case because the DXP physicalizer probe rewrites LX placements outside normal
STCDP semantics.

Two output encodings were tested:

- `valid_gap`: output pieces span the full stick dimension and use `validGap_`
  to select the 64-wide stripe.
- `dense_actual`: output pieces describe only the actual 64-wide stripe.

`dense_actual` is useful diagnostically because it avoids the leading-gap import
path, but it still cannot express the consumer layout's row-strided placement.
The movement remains contiguous at the STCDP subpiece level while the consumer
layout needs per-row placement within an `["out", "mb"]` local layout.

## Deeptools Probes

Two probe patches are recorded in `tools/`:

- `deeptools_onchip_move_physicalizer_filter_probe.patch`
  - routes mixed data-op SDSCs through the DXP physicalizer probe;
  - supports logical `mb/out/x` dims;
  - computes sparse-piece placement from the first valid coordinate.
- `deeptools_stcdp_validgap_overlap_probe.patch`
  - changes STCDP overlap/subpiece generation to treat leading valid gaps as
    positive valid-coordinate offsets;
  - fixes isolated DXP coverage for a sparse second stripe;
  - does not make the full AIU smoke runnable.

## InputFetchNeighbor Check

Deeptools has an InputFetchNeighbor path, but it is not triggered by a separate
JSON op.  It is triggered when a mixed schedule step carries both a data DSC and
a DL DSC index, for example `[0, 0, 1, 0]`.

Changing the generated bundle to use that combined schedule does enter the IFN
path, but current Deeptools asserts on missing `i/j` coordinates for this
`mb/out` tensor.  IFN also supports only one neighbor input today, which is a
problem for full SwiGLU where both gate/up activations may need handoff.

## Next Direction

Do not keep tuning the current mixed `STCDPOpLx` encoding as the scalable
solution.  The next useful implementation slice is one of:

1. Add a Deeptools-native coordinate-remap LX dataop that pairs source and
   destination logical coordinates directly.
2. Generalize InputFetchNeighbor beyond `i/j` assumptions and one neighbor
   input, then emit the combined mixed schedule from torch-spyre.
3. Prototype a local gather/restickify/scatter carrier using existing
   `ReStickifyOpLx` / `ReStickifyOpWithPTLx` only if it can preserve the
   matmul-preferred split and avoid HBM fallback.

Warp specialization should remain secondary until the movement path is
value-correct.
