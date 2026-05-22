# Stage 310: Three-Stage PT-LX Sidecar

## Summary

This stage changed the LX-neighbor sidecar for kernel-to-output PT-LX layout
transforms from the direct diagnostic shape to the production-shaped sequence:

```text
STCDPOpLx gather producer LX fragments
ReStickifyOpWithPTLx local 64x64 tile transform
STCDPOpLx write/scatter the consumer-owned LX tile
```

The candidate remains non-executable and keeps `ReStickifyOpHBM` as fallback.
The change is about making the generated sidecar match the requested lowering
shape before we try to package it into the normal bundle.

## Code Changes

For LX-neighbor bridge candidates:

- same-layout ownership remaps still emit the certified `STCDPOpLx` remap
  sidecar;
- kernel-to-output PT-LX layout transforms now emit a three-stage
  gather/transform/scatter sidecar;
- output-to-kernel transforms keep the direct diagnostic sidecar until the
  three-stage helper supports that direction.

The candidate records the selected shape in:

```text
bridge_lowering
```

with values such as:

```text
same-layout-lx-ownership-remap
three-stage-gather-transform-scatter
direct-ptlx-diagnostic
```

## Current Gate

The three-stage sidecar still reports:

```text
production_valid = false
production_blocker = three-stage-ptlx-lacks-value-correct-transform-certificate
```

That is intentional.  It proves the compiler can emit the desired shape from
real producer/consumer ownership metadata, but it does not yet prove hardware
value correctness or insert the bridge into `bundle.mlir`.

## Validation

Pod validation:

```sh
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest \
  tests/inductor/test_restickify_lx_neighbor_descriptor.py \
  tests/inductor/test_restickify_lx_dataop.py \
  -q
```

Result:

```text
59 passed in 3.42s
```

## Next Step

The next implementation step is to make the three-stage sidecar executable for
a single-source, single-destination 64x64 tile in an isolated bundle, then add a
value-correctness certificate for that tile before attempting full-tensor
replacement.
