# Stage 311: Native PT-LX Sidecar

## Summary

Stage 310 made the LX-neighbor bridge emit a three-stage sidecar for
kernel-to-output PT-LX transforms:

```text
gather producer LX fragments
local PT-LX restickify
scatter/write consumer LX fragments
```

Stage 311 changes the local transform sidecar from the older 2D
`mb_/out_ -> out_/mb_` descriptor shape to the native PT-LX tile descriptor
shape:

```text
j_, i_, out_, mb_
```

The generated sidecar now records:

```text
coalescing = native-64x64-tiles
native_local_transform_contract = true
semantic_transform_certified = false
```

## Why This Matters

The production path needs more than LX endpoints.  It needs a local tile
descriptor whose dimensions match the PT-LX restickify contract.  The native
tile helper already expresses a 64x64 tile in the PT-LX form that Deeptools
expects for `ReStickifyOpWithPTLx`.

This still does not allow replacement of `ReStickifyOpHBM`.  The bridge remains
a sidecar.  The current blocker is now more precise:

```text
production_valid = false
production_blocker = native-ptlx-output-needs-consumer-endpoint-adapter
```

That is intentional.  The sidecar now uses the native local PT-LX tile
descriptor, but its final output descriptor is still native-shaped
`j_, i_, out_, mb_` rather than the consumer's final input endpoint.  The next
production step is an endpoint adapter or consumer descriptor override that
materializes the native tile output into a consumer-readable LX descriptor.

## Current Scope

- same-layout LX ownership remaps remain production-valid when certified;
- kernel-to-output PT-LX transforms use the native three-stage PT-LX sidecar;
- output-to-kernel PT-LX transforms remain on the direct diagnostic sidecar
  until native support is generalized for that direction;
- `bundle.mlir` remains unchanged and the stock `ReStickifyOpHBM` path remains
  runnable.

## Next Step

The next step is an isolated native-tile runtime probe for one
single-source/single-destination 64x64 tile with the same endpoint adapter that
the consumer would use.  Success means the native gather/transform/scatter
sidecar can be compiled, can present a consumer-readable LX endpoint, and can be
shown value-correct for that tile before we try full-tensor replacement.
