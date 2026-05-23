# Stage 314: Native Endpoint Adapter Tile

## Summary

This stage adds the first static lowering helper for the planned native PT-LX
consumer endpoint adapter:

```text
generate_native_ptlx_consumer_endpoint_adapter_tile_sdsc
```

The helper emits one codegen-only `ReStickifyOpWithPTLx` tile adapter:

```text
native PT-LX tile workspace -> consumer LX endpoint
```

for the current kernel-to-output direction.

## Descriptor Shape

The adapter input is the native PT-LX tile output:

```text
layout = j_, i_, out_, mb_
stick  = j_
```

The adapter output is the consumer-visible LX endpoint:

```text
layout = out_, mb_
stick  = mb_
```

The metadata records the intended coordinate map:

```text
destination_out = native_out
destination_mb  = native_j
drop native_i/native_mb singleton dimensions
```

## Status

This is still not a production certificate:

```text
status = static-codegen-only
semantic_transform_certified = false
fallback = ReStickifyOpHBM
```

The value of this stage is that the missing adapter is now a concrete SDSC-like
object with explicit LX-only input/output descriptors.  The next step is to
compile and value-check this one-tile adapter in isolation.
