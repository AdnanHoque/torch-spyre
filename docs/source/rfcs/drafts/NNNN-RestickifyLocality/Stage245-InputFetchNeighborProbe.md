# Stage 245: InputFetchNeighbor Probe

## Summary

After tightening the streaming PT-LX semantic guard, we checked whether
Deeptools' existing `InputFetchNeighbor` path can supply the missing
coordinate-remap/gather behavior for the tiled PT-LX design.

The answer is useful but limited:

- `InputFetchNeighbor` is real LX-neighbor movement machinery.
- It is not a restickify transform.
- It requires the producer output and consumer input stick definitions to
  match.

That means it can plausibly serve as a same-stick gather/scatter leg around a
local `ReStickifyOpWithPTLx`, but it cannot by itself replace
`ReStickifyOpHBM` for a layout-changing edge.

## Probe

Generated stock code for:

```text
adds_then_matmul, size 512
```

The stock bundle emitted:

```text
sdsc_0_ReStickifyOpHBM.json
sdsc_1_add.json
sdsc_2_add.json
sdsc_3_ReStickifyOpHBM.json
sdsc_0_batchmatmul.json
```

Then ran:

```text
dcg_inpfetch_standalone \
  -s \
  -initSdscMain sdsc_0_batchmatmul.json \
  -initSdscPre sdsc_3_ReStickifyOpHBM.json
```

Initial failure:

```text
DtException:
  lds.isLxPinned() || lds.isRingPinned() || lds.isSfpRingPinned()
  inputNeighFetchOp.cpp line 30
```

This is expected for a stock matmul SDSC because non-target operands are still
HBM-backed.

After patching the standalone copies to be LX-present only, the next failure
was:

```text
DtException:
  outLds.stickDimOrder_ == inpLds.stickDimOrder_
  inputNeighFetchOp.cpp line 248
```

## Interpretation

`InputFetchNeighbor` does not do the layout/stick conversion. It fetches
neighbor LX data when the main input and pre output are already compatible in
stick definition.

For the production-shaped streaming PT-LX path, this suggests the correct
structure is not:

```text
STCDPOpLx gather/remap -> ReStickifyOpWithPTLx -> STCDPOpLx scatter/remap
```

It is more likely:

```text
same-stick InputFetchNeighbor gather
  -> local ReStickifyOpWithPTLx
  -> same-stick InputFetchNeighbor/STCDP-style scatter
```

or a Deeptools-native single op that combines those contracts.

## Next Step

Build the next prototype around a stricter three-phase contract:

1. gather producer source tiles only where the gathered source view preserves
   the producer stick layout;
2. run `ReStickifyOpWithPTLx` on a local tile whose input/output piece metadata
   matches Deeptools' supported PT-LX restickify shape;
3. scatter/fetch the restickified output only where the output stick layout
   already matches the consumer.

This should replace the current uncertified STCDP global-to-compact remap.
