# Stage 312: Native Endpoint Adapter Contract

## Summary

Stage 311 exposed the next exact blocker: the native PT-LX sidecar produces a
tile-local descriptor:

```text
j_, i_, out_, mb_
```

but the consumer still expects its normal LX endpoint, for example:

```text
layout = out_, mb_
stick  = mb_
```

Stage 312 records this required endpoint adapter explicitly in the
LX-neighbor bridge candidate.

## Contract

For the current kernel-to-output direction, the candidate records:

```text
consumer_endpoint_adapter.available = true
consumer_endpoint_adapter.executable = false
consumer_endpoint_adapter.coordinate_map = {
  destination_out: native_out,
  destination_mb: native_j,
}
consumer_endpoint_adapter.dropped_singleton_dims = [
  native_i,
  native_mb,
]
consumer_endpoint_adapter.required_stick_transform = {
  from: native_j,
  to: destination_mb,
}
```

The production gate remains closed:

```text
production_valid = false
production_blocker = native-ptlx-output-needs-consumer-endpoint-adapter
required_primitive = consumer-lx-endpoint-adapter
```

## Why This Matters

This separates two different remaining tasks:

1. The local PT-LX tile transform must be value-correct.
2. The final bridge output must be presented through the consumer's actual LX
   layout/stick descriptor.

Before this stage those two problems were collapsed into one vague
``needs certificate`` blocker.  Now the compiler metadata identifies the exact
coordinate map that the endpoint adapter must implement.

## Next Step

Lower the planned adapter for a single 64x64 tile.  The first executable probe
should be intentionally narrow:

```text
native tile workspace -> consumer LX endpoint
```

with no HBM allocation and no full graph replacement.  Only after that adapter
is value-checked should the full gather/transform/scatter sidecar be allowed to
advance toward bundle integration.
