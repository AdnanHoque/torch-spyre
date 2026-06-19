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

### Branch `swiglu-ws-input-fetch`

This branch adds a torch-spyre artifact-generation prototype behind:

```bash
SPYRE_ONCHIP_MOVE_REALIZE=1
SPYRE_ONCHIP_MOVE_CARRIER=input_fetch_neighbor
```

The carrier reuses the existing on-chip movement plan payload and emits a mixed
consumer SDSC with:

- producer output and consumer input patched to LX endpoints;
- one `datadscs_` entry carrying logical `mb/out/x` piece metadata;
- `coreIdToDscSchedule` rows with both indices set, for example
  `[0, 0, 0, 0]`, which is the artifact-level IFN trigger condition.

The emitted data DSC intentionally keeps logical dim names such as `mb`, `out`,
and collapsed size-one `x` instead of fabricating `i/j` aliases.  That means the
torch-spyre artifact no longer loses the non-IJ movement-domain metadata, but it
does not by itself prove Deeptools can run the helper: the known backend blocker
remains any hardcoded IFN code path that requires `i` and `j` keys instead of
using the artifact's actual layout dimension names.

The branch also makes the current one-neighbor-input limitation explicit.  If a
consumer has more than one planned on-chip neighbor input, the IFN carrier skips
realization with:

```text
input-fetch-neighbor-single-neighbor-input-only
```

That is a hard blocker for full SwiGLU fan-in when both gate and up activations
need neighbor handoff into the same multiply or SiLU/multiply consumer.  This
branch also does not provide the mixed carrier's later-consumer LX reuse path,
so non-adjacent fan-out remains unsupported here even when each consumer has
only one neighbor input.  IFN may still be useful for a single-edge smoke, but
it cannot cover the full SwiGLU fan-in/fan-out carrier requirement without a
Deeptools/backend semantic extension.

### Second Iteration Pod Result

The branch was tested in an isolated pod checkout:

```text
pod: adnan-cdx-spyre-dev-pf
checkout: /tmp/torch-spyre-swiglu-ws-input-fetch-iter2
artifact: artifacts/input_fetch_neighbor_real_mb_out_iter2/
```

A single-edge `mb/out` IFN artifact was built from full torch-spyre-generated
`batchmatmul` and `add` SDSCs.  The consumer SDSC contains one
`0_OnChipMoveIFNDataOpLx` and the IFN trigger schedule row
`[[0, 0, 0, 0]]`.

`dcg_inpfetch_standalone` reaches the Deeptools IFN path but fails before value
execution:

```text
DtException: mySDscMain.dscs_.at(0).primaryDsInfo_.count(DsTypes::INPUT)
file /project_src/deeptools/dcg/dcg_fe/pcfg_gen/inputNeighFetchOp.cpp line 16
```

Compatibility probes show the stock helper also assumes all checked tensors are
LX/ring-pinned, legacy `coreStateInit_` is present, and loop-order metadata is
non-empty.  The ordinary DXP bundle path rejects the artifact with:

```text
DtException: Datadsc not allowed, use dldsc
file /project_src/deeptools/dxp/SdscTree.cpp line 152
```

Status: blocked.  A value-correct single-edge smoke was not reached on the stock
pod backend.  Decomposing multi-neighbor fan-in in torch-spyre is not a safe
next step until Deeptools IFN accepts modern SuperDSC operand selection, LX
address metadata, and bundle data DSCs.

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
