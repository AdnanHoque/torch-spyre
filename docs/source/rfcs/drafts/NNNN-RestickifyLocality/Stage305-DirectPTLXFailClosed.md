# Stage 305: Direct PT-LX Split Fail-Closed Gate

## Summary

This stage tightened the direct 64x64 PT-LX split probe so it does not launch
an uncertified streaming sidecar by default.

The direct sidecar can still be generated and exported, and the producer and
data-op stages can retire cleanly on hardware. The remaining unsafe step is the
separate consumer bundle reading an internal LX value that was produced by a
previous bundle. That is not a production-shaped internal value-flow contract.

## Code Changes

- The split probe now derives the default consumer LX base from the emitted
  direct PT-LX sidecar destination endpoint instead of hard-coding `8192`.
- The split prepare cache signature was bumped so stale `8192` bundles are not
  reused.
- Direct PT-LX sidecar packaging now fails closed unless
  `streamingPTLXFull_.semantic_transform_certified` is true.
- Unsafe hardware probing is still possible with:

```sh
SPYRE_RESTICKIFY_LX_SPLIT_ALLOW_UNCERTIFIED_DIRECT=1
```

but that is diagnostic-only and not a candidate production gate.

## Hardware Isolation

Diagnostic run with uncertified direct sidecar explicitly allowed:

```sh
SPYRE_RESTICKIFY_LX_SPLIT_ALLOW_UNCERTIFIED_DIRECT=1
SPYRE_RESTICKIFY_LX_SPLIT_DATAOP_MODE=lx-neighbor-direct-ptlx
SPYRE_RESTICKIFY_LX_SPLIT_STAGES=producer,dataop,consumer
SPYRE_RESTICKIFY_LX_SPLIT_SYNC_EACH=1
SPYRE_RESTICKIFY_LX_SPLIT_HEALTH_EACH=1
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --skip-correctness \
  --lx-split-dataop-prototype
```

Observed stages:

```text
producer: launched and post-health passed
dataop: launched and post-health passed
consumer: launched, then post-health saw ComputeHardwareError/StreamInErrorState
```

The sidecar export was HBM-free and used the generated consumer endpoint:

```text
dataop_export_returncode: 0
dataop_sidecar_producer_base: 16384
dataop_sidecar_consumer_starts:
  [262144, 262272, 262400, 262528, 262656, 262784, 262912, 263040]
```

This narrows the failure: the direct PT-LX data-op sidecar itself is not the
hardware-error source. The split consumer bundle handoff is the unsafe contract.

## Fail-Closed Check

Without the unsafe override, the same probe now fails before launching the
sidecar or consumer:

```text
direct PT-LX split sidecar is not semantically certified; fall back to
ReStickifyOpHBM or set SPYRE_RESTICKIFY_LX_SPLIT_ALLOW_UNCERTIFIED_DIRECT=1
for diagnostic hardware probing
```

A fresh stock Spyre tensor smoke passed after the failed prepare, confirming
that the fail-closed path did not poison the device stream.

## Interpretation

The wide direct-tile PT-LX path is still not production-ready. The current
streaming sidecar reports:

```text
coalescing: direct-64x64-tiles
datadsc_count: 192
semantic_transform_certified: false
fallback: ReStickifyOpHBM
```

That is the correct compiler behavior: keep the stock HBM restickify fallback
unless the bridge can prove the logical coordinate transform and the
producer-bridge-consumer LX lifetime contract.

## Next Step

Continue from the proven Stage203 direction:

1. Use a same-bundle mixed schedule, not separate producer/dataop/consumer
   launches, for any internal LX value flow.
2. Teach the production lowering to emit the mixed bridge only when the
   producer, bridge, and consumer value-flow verifier passes.
3. Keep the streaming direct-tile path as a compile-only/fail-closed candidate
   until a primitive or schedule can certify gather/scatter coordinate remaps.

Artifacts:

```text
artifacts/stage305_direct_ptlx_failclosed/
```
