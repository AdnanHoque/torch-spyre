# Stage 297: Direct PT-LX Split Probe

## Summary

This stage connected the compact direct PT-LX sidecar to the existing
producer -> dataop -> consumer split-probe harness. The sidecar can now be
selected as:

```sh
SPYRE_RESTICKIFY_LX_SPLIT_DATAOP_MODE=lx-neighbor-direct-ptlx
```

The harness now:

1. finds the emitted `restickify_lx_neighbor_streaming_bridge_edge_*.json`
   sidecar,
2. patches the sidecar producer and consumer LX endpoint addresses to match the
   split producer/consumer contract,
3. exports the patched sidecar through the Deeprt data-op exporter,
4. stages `execute/*/init.txt` into the normal `launch_kernel` runtime shape.

This keeps the stock HBM restickify path as fallback and keeps the direct PT-LX
path explicitly probe-only.

## Endpoint Patching

The raw sidecar is emitted with its own internal endpoint choices, observed as:

```text
producer base: 0
consumer starts: [262144]
```

The split-probe path now patches these addresses to match the producer and
consumer bundles. With default split settings, the prepared sidecar reports:

```text
producer_lx_base: 16384
consumer_lx_base: 8192
dataop_sidecar_producer_base: 16384
dataop_sidecar_consumer_starts: [8192]
dataop_original_sidecar_producer_base: 0
dataop_original_sidecar_consumer_starts: [262144]
```

That avoids forcing the producer output to LX address zero just because the
sidecar was initially emitted that way.

## Prepare-Only Validation

Command shape:

```sh
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1 \
SPYRE_RESTICKIFY_LX_SPLIT_DATAOP_MODE=lx-neighbor-direct-ptlx \
SPYRE_RESTICKIFY_LX_SPLIT_PREPARE_ONLY=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --lx-split-dataop-prototype \
  --skip-correctness \
  --output-dir /tmp/stage297-direct-split-prepare-512 \
  --fail-on-error
```

Result:

```text
ok size=512 case=computed_transpose_adds_then_matmul_tuple restickifies=1 bytes=524288 byte_hops=0
dataop_export_returncode: 0
dataop_sidecar_producer_base: 16384
dataop_sidecar_consumer_starts: [8192]
```

## Hardware Value Run

Command shape:

```sh
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR_ALLOW_UNCERTIFIED=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_STREAMING_BRIDGE=1 \
SPYRE_RESTICKIFY_LX_SPLIT_DATAOP_MODE=lx-neighbor-direct-ptlx \
SPYRE_RESTICKIFY_LX_SPLIT_SYNC_EACH=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 \
  --lx-split-dataop-prototype \
  --output-dir /tmp/stage297-direct-split-run-512 \
  --fail-on-error
```

Result:

```text
RAS::RUNTIMESCHEDULER::ComputeHardwareError
RAS::RUNTIMESCHEDULER::StreamInErrorState
```

The kernel-launch log recorded producer, dataop, and consumer phases as
completed, then the next matmul launch failed because the stream had already
entered an error state. That exposed a probe bug: `_sync()` swallowed
accelerator synchronize exceptions unless `SPYRE_PROBE_STRICT_SYNC=1` was set.

The probe now treats `SPYRE_RESTICKIFY_LX_SPLIT_SYNC_EACH=1` as strict sync
mode, so the next hardware run should attribute the hardware error to the exact
producer, sidecar, or consumer phase.

A new-process stock Spyre smoke passed after the failure:

```text
post_stage297_stock_smoke_ok spyre:0
```

## Current Interpretation

What works now:

- compact direct PT-LX sidecar generation,
- sidecar endpoint rebasing to the split producer/consumer contract,
- Deeprt export of the patched sidecar,
- standalone no-argument runtime retirement of the HBM-free sidecar.

What is not solved:

- value-correct producer -> sidecar -> consumer execution,
- exact attribution of the hardware error within the split sequence.

The next run should use the stricter split sync behavior and stop at the first
failing stage. If the sidecar stage itself fails, inspect the patched sidecar's
LX source/destination address arithmetic. If the consumer stage fails, inspect
the consumer input descriptor and whether it expects a different physical LX
shape than the sidecar writes.

