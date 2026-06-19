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

### `swiglu-ws-dxp` Branch Check

This branch adds an artifact-level diagnostic for the dense-actual encoding:
`diagnose_stcdp_output_layout_contiguity(...)`.  It checks whether each output
`PieceInfo` is a contiguous span in the consumer's local logical layout.  The
small SwiGLU-shaped case reproduces the same failure without a full AIU run:

```text
output piece p1:
  layoutDimOrder_: ["out", "mb", "x"]
  start: {"out": 64, "mb": 0, "x": 0}
  size:  {"out": 64, "mb": 16, "x": 1}

first mismatch:
  contiguous STCDP element delta: 64
  required consumer-layout delta: 512
  contiguous STCDP byte delta: 128
  required consumer-layout byte delta: 1024
```

So the first 64 `out` elements in `mb=0` are contiguous, but the next value
belongs at `mb=1,out=64`, which is 512 elements from the first value in the
consumer `["out", "mb", "x"]` layout.  A single dense STCDP output piece writes
that value at element delta 64 instead.  This is a concrete value-correctness
blocker for the current mixed `STCDPOpLx` carrier.

### Second Iteration Backend Check

The second iteration reproduced the blocker at the backend boundary in an
isolated pod checkout:

```text
Torch-Spyre checkout: /tmp/torch-spyre-swiglu-ws-dxp-iter2
artifact root:        /tmp/torch-spyre-swiglu-ws-dxp-iter2-artifacts
torch/pytest:         /usr/bin/python3, torch 2.11.0+cpu, pytest 9.1.0
```

The focused on-chip movement unit tests pass in that checkout with a compatible
prebuilt `_C.so` copied into the isolated tree:

```text
PYTHONPATH=/tmp/torch-spyre-swiglu-ws-dxp-iter2 \
  python3 -m pytest tests/inductor/test_onchip_move.py -q

12 passed
```

The editable build was attempted first but the installed Spyre runtime headers
do not match this Torch-Spyre source revision:

```text
flex::DmaParams has no member named pipeline_barrier
flex::RuntimeStream has no member named launchOperationH2D
no operator<< for flex::CompositeAddress
```

A real MLP/SwiGLU compile was then run twice from the isolated checkout:

```text
baseline cache:
  /tmp/torchinductor_swiglu_ws_dxp_iter2_baseline/inductor-spyre/
    sdsc_fused_mm_mul_silu_0_q2qx72l6

mixed-STCDP cache:
  /tmp/torchinductor_swiglu_ws_dxp_iter2/inductor-spyre/
    sdsc_fused_mm_mul_silu_0_87wknxzu

archived comparison:
  /tmp/torch-spyre-swiglu-ws-dxp-iter2-artifacts/dxp_compare
  /tmp/torch-spyre-swiglu-ws-dxp-iter2-artifacts/dxp_compare.tar.gz
```

The baseline bundle compiled and the pytest smoke passed.  The mixed-STCDP
bundle emitted `sdsc_2.json` as `2_OnChipMoveMixedSTCDP`, then stock
`dxp_standalone --bundle` failed before DDC/DCG:

```text
baseline dxp exit: 0
on-chip dxp exit: 134

DtException: Datadsc not allowed, use dldsc,
file /project_src/deeptools/dxp/SdscTree.cpp line 152
```

The archived diff stats between the comparable fused bundles are:

```text
sdsc_1.json: +34 / -37
sdsc_2.json: +18829 / -39
sdsc_5.json: +34 / -37
```

Only the mixed bundle contains STCDP:

```text
baseline top ops:
  0_batchmatmul, 1_batchmatmul, 2_neg, 3_exp, 4_add, 5_realdiv, 6_mul

on-chip top ops:
  0_batchmatmul, 1_batchmatmul, 2_OnChipMoveMixedSTCDP,
  3_exp, 4_add, 5_realdiv, 6_mul
```

Running `diagnose_stcdp_output_layout_contiguity(...)` against the real
`sdsc_2.json` data-op reports the same scalable-placement failure:

```text
piece p1, layoutDimOrder_: ["mb", "out"]
start: {"mb": 0, "out": 0}
size:  {"mb": 8, "out": 512}

first mismatch:
  coord: {"mb": 0, "out": 1}
  contiguous STCDP element delta: 8
  required consumer-layout delta: 256
  contiguous STCDP byte delta: 16
  required consumer-layout byte delta: 512
```

This is the backend/runtime boundary proof for this branch: stock DXP does not
accept mixed `datadscs_`, and the real emitted STCDP output pieces still require
strided destination placement before a Deeptools physicalizer tries to make them
legal.

The non-hacky fixes are outside the current plain-STCDP encoding:

- a backend physicalizer that splits the transfer into layout-contiguous
  subpieces and proves the generated STCDP LX descriptors are legal; or
- a coordinate-aware LX-to-LX data-op that represents source and destination
  logical coordinates directly.

Splitting only this SwiGLU shape in torch-spyre would hide the problem rather
than making `STCDPOpLx` a general LX-to-LX carrier.

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
