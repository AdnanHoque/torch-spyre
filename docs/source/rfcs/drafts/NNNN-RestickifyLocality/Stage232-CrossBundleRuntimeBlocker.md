# Stage 232: Cross-Bundle PT-LX Runtime Blocker

## Summary

Stage 230-231 proved that the cross-bundle PT-LX bridge can be generated and
compiled without `ReStickifyOpHBM` for the in-graph edge. Hardware validation
showed the next blocker: the cross-bundle LX handoff is not value-correct.

This is a useful distinction:

- The bridge artifact is DXP/DCC accepted.
- The bridge and consumer launch without a stream hardware error.
- The final values are wrong, which strongly suggests the LX endpoint is not a
  valid persistent inter-bundle tensor boundary.

The older value-correct PT-LX result was same-bundle: the bridge and consumer
ran in one runtime bundle. This stage shows that moving the bridge into the
producer bundle and expecting the next bundle to read the same LX value is not a
production-safe contract.

## Hardware Runs

Common environment:

```sh
SPYRE_INDUCTOR_MAX_BUNDLE_TENSORS=7
SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E=1
SPYRE_RESTICKIFY_PTLX_STREAMING_E2E=1
SPYRE_RESTICKIFY_PTLX_CROSS_BUNDLE_E2E=1
SPYRE_RESTICKIFY_PTLX_FORCE_ENV_ENDPOINTS=1
SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE=0
SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE=1048576
SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT=1
```

512:

```text
error size=512 case=adds_then_matmul
Mismatched elements: 210465 / 262144 (80.3%)
Greatest absolute difference: 2.17578125
```

2048:

```text
error size=2048 case=adds_then_matmul
Mismatched elements: 3640383 / 4194304 (86.8%)
Greatest absolute difference: 6.0
```

Both runs launched these bundles:

```text
sdsc_fused_add_t_0:
  sdsc_0_ReStickifyOpHBM.json
  sdsc_1_add.json
  sdsc_3_CrossBundleProducerStreamingReStickifyOpWithPTLx.json

sdsc_fused_mm_1:
  sdsc_0_batchmatmul.json
```

The second bundle's matmul input was patched to LX, but the output did not
match CPU.

## One-Bundle Probe

Raising the bundle tensor cap to `20` made Inductor attempt a single
`sdsc_fused_add_mm_t_0` bundle. This is the shape we probably need for a
production PT-LX path, but it does not lower today.

First blocker:

```text
IndexError: list index out of range
SEGMENT_OFFSETS[arg.arg_index]
```

Temporarily extending segment offsets got past Python but hit a real DXP guard:

```text
DtException: Input/output tensor allocated in segment reserved for backend compiler
file .../deeptools/dxp/dxp.cpp line 372
```

So the valid next path is not to invent more HBM segment offsets. The fused
bundle needs fewer external argument segments or a different way to represent
internal buffers.

## Interpretation

The current production-shaped hypothesis should change:

```text
bad contract:
  producer bundle writes LX -> later consumer bundle reads LX

better contract:
  producer, PT-LX bridge, and consumer are one runtime value-flow unit
```

This lines up with the earlier value-correct result:

```text
MixedReStickifyOpWithPTLxConsumer
  bridge data ops
  consumer DL op
```

That path worked because the bridge and consumer were scheduled together. The
cross-bundle producer-mixed path is compile-clean but semantically unsafe.

## Next Step

Focus on same-bundle production lowering:

1. Preserve the existing stock HBM fallback.
2. Do not persist LX tensors across runtime bundle boundaries.
3. Teach the fusion/lowering path to form a producer/restickify/consumer runtime
   unit for eligible in-graph restickifies.
4. Avoid increasing HBM segment count beyond the backend-supported segment
   table.
5. Reuse the proven same-bundle PT-aware bridge contract, then broaden it with
   streaming/row-stripe lowering only after the same-bundle runtime contract is
   value-correct.
