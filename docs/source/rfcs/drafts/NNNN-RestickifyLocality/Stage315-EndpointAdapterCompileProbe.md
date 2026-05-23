# Stage 315: Endpoint Adapter Compile Probe

## Summary

This stage added a probe hook for the native PT-LX endpoint adapter tile:

```sh
python tools/restickify_lx_dataop_probe.py \
  --streaming-ptlx-tile \
  --native-endpoint-adapter-tile \
  --size 512 \
  --tile-index 0 \
  --output-dir /tmp/stage315-native-endpoint-adapter-tile-retry \
  --run-dcg \
  --dcg-standalone /home/adnan-cdx/dt-inductor-mixed/sentient/deeptools/bin/dcg_standalone
```

The generated artifact is HBM-free and describes:

```text
input  layout = j_, i_, out_, mb_   stick = j_
output layout = out_, mb_           stick = mb_
op = ReStickifyOpWithPTLx
```

## Result

The artifact is generated successfully, but `dcg_standalone` rejects it:

```text
DtException:
op->outLds->dimToLayoutSize_.at(stickDimOut) <= elemPerSlice
file .../deeptools/dcg/dcg_fe/pcfg_gen/restickifyOp.cpp line 1930
```

Before switching the adapter to `ReStickifyOpWithPTLx`, a plain `STCDPOpLx`
adapter also failed:

```text
myOp->outLds->stickDimOrder_ == myOp->inpLds->stickDimOrder_
file .../transfer_compute.cpp line 450
```

That confirms the adapter cannot be a same-layout movement primitive when it
changes the stick dimension.

## Interpretation

The missing endpoint adapter is now concrete, but the direct descriptor:

```text
j_ stick -> mb_ stick
```

does not satisfy Deeptools' current `ReStickifyOpWithPTLx` contract.

Two diagnostic alternatives were checked:

- same-stick aliasing keeps the physical stick as `j_`, but Deeptools rejects
  it because `ReStickifyOpWithPTLx` expects an actual stick change;
- a valid-gap alias dimension gets closer to the older consumer-shaped probe,
  but the quick mutation still fails descriptor import with `map::at`.

## Current Status

The adapter remains fail-closed:

```text
status = static-codegen-only
semantic_transform_certified = false
fallback = ReStickifyOpHBM
```

## Next Step

The next implementation step is to make the valid-gap alias adapter a real
generated helper rather than a JSON mutation.  It should explicitly include the
alias dimension in root metadata, primary DS metadata, labeled DS metadata, and
piece metadata, then run the same DCG probe.  If that compiles, the next proof
is hardware value correctness for one tile.
