# SwiGLU On-Chip Movement Diagnostics

This branch tracks experiments toward a scalable LX-to-LX movement path for
SwiGLU-style matmul-to-pointwise reshards.  The current target shape is the
matmul output layout used by the small smoke:

- producer: `{mb:4, out:8}`, logical layout `["mb", "out", "x"]`
- consumer: `{mb:32}`, logical layout `["out", "mb"]`
- tensor: `[1, 512, 512]` activation represented as `[512, 8, 1, 64]`

## Current Status

There is no value-correct working prototype yet for this reshaping handoff.

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

## Existing-Op Carrier Check

Branch `swiglu-ws-existing-ops` prototyped an `existing_ops` carrier that builds
a mixed DL-SDSC with an existing `ReStickifyOpLx` row before the pointwise
consumer.  The artifact keeps the matmul producer and pure-M consumer work
divisions separate: it does not co-assign the producer to the consumer split.

The generated shape is:

- producer output is patched to LX at `SPYRE_ONCHIP_MOVE_PRODUCER_LX_BASE`;
- a `ReStickifyOpLx` DL row reads that LX allocation and writes the consumer
  LX allocation at `SPYRE_ONCHIP_MOVE_CONSUMER_LX_BASE`;
- the consumer input is patched to read the consumer LX allocation;
- the mixed schedule runs restickify first, then the consumer DL op.

This is not a viable carrier for the target `{mb:4,out:8}` producer to pure-M
consumer edge.  The planned common-refinement cells include remote handoffs such
as source core 4 to destination core 0.  `ReStickifyOpLx` describes local
input/output allocations for each executing core, but it has no per-piece
placement field equivalent to the mixed `STCDPOpLx` `PlacementInfo` pairs.  It
can retile data already local to a core's LX; it cannot express "consumer core
0 reads this piece from producer core 4" without adding a separate gather,
remote-read primitive, or falling back to co-assignment.

Temp pressure is not the blocker.  For the small `[512,8,1,64]` activation, the
producer and consumer per-core LX regions are both 16 KiB, so the prototype has
roughly the same two-region LX footprint as the DXP/STCDP path.  It would add a
DL restickify stage before the consumer instead of a data-op STCDP stage, but it
still needs an unimplemented cross-core source selection mechanism.  Composing
extra gather/scatter steps around it would be more indirect than the DXP path
and would likely add at least one more temporary region or stage.

## Next Direction

Do not keep tuning the current mixed `STCDPOpLx` encoding as the scalable
solution.  The next useful implementation slice is one of:

1. Add a Deeptools-native coordinate-remap LX dataop that pairs source and
   destination logical coordinates directly.
2. Generalize InputFetchNeighbor beyond `i/j` assumptions and one neighbor
   input, then emit the combined mixed schedule from torch-spyre.
3. Avoid the local `ReStickifyOpLx` carrier for the current SwiGLU cross-core
   reshuffle unless another existing operation can supply the missing remote
   source selection without co-assignment or HBM fallback.

Warp specialization should remain secondary until the movement path is
value-correct.
