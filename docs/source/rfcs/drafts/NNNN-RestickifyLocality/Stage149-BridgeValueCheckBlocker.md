# Stage 149: Bridge value-check blocker

## Summary

This stage attempted the first narrow hardware value check for the schema-v4
remote-LX materialization path:

```text
producer compute -> ReStickifyOpLx/STCDPOpLx data-op bridge -> consumer compute
```

The probe intentionally skipped the downstream matmul and compared only the first
tuple output, which is the materialized join tensor.  That isolates the
producer/bridge/consumer boundary from the later matmul bundle.

## Probe Controls Added

- `SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE=1`
  skips any later non-restickify bundles once the split bridge has handled one
  producer/restickify/consumer triplet.
- `--validate-tuple-prefix N`
  compares only the first `N` tuple outputs during correctness checking.  For
  `computed_transpose_adds_then_matmul_tuple`, `N=1` checks the bridge output
  without requiring the matmul output to be valid.
- The split bridge now rejects DeeRT data-op exports with nonzero return codes,
  even if partial `init.txt` files were emitted.

## Command

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_SPLIT_REQUIRE_MATERIALIZATION_CONTRACT=1 \
SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE=1 \
SPYRE_RESTICKIFY_LX_SPLIT_SYNC_EACH=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --copy-kernel-code \
  --kernel-launch-log \
  --lx-split-dataop-prototype \
  --validate-tuple-prefix 1 \
  --output-dir /tmp/stage149-stop-after-bridge-value-2048 \
  --fail-on-error
```

## Result

The split bridge reached:

```text
lx_split_dataop_after_producer
lx_split_dataop_after_dataop
lx_split_dataop_after_consumer
lx_split_dataop_launch_done
lx_split_dataop_stop_after_bridge_skip
```

So the downstream matmul was skipped as intended.  The generated data-op bridge
still had no HBM instructions:

```text
HBM=0, LXLU=64, LXSU=64
```

However, the value check failed when copying the first tuple output back to CPU:

```text
RAS::RUNTIMESCHEDULER::ComputeHardwareError
RAS::RUNTIMESCHEDULER::StreamInErrorState
OpType=D2H
```

This means the producer/data-op/consumer sequence is not yet hardware-clean at
the 2048 high-signal size.  The error is not caused by the matmul tail because
the log shows the matmul bundle was skipped before the D2H copy.

## Additional Clue

In the failing value run, the DeeRT data-op exporter returned `-11` while still
leaving an `init.txt` behind:

```text
dataop_export_returncode=-11
```

The old probe accepted that partial artifact because it only checked for
`execute/*/init.txt`.  That is now guarded: nonzero exporter return codes are
fatal and must not be launched.

## Current Blocker

The current blocker is now precise:

```text
Generate a schema-v4 LX materialization data-op that DeeRT exports cleanly
and that does not trigger a compute-control hardware error when run between
the producer and consumer at size 2048.
```

The most likely next experiments are compile/export-only:

- compare the clean prepare-only export with the later `-11` export and identify
  what changed in the generated SDSC or environment;
- reduce the data-op bridge shape while keeping a schema-v4 materialization edge,
  then sweep upward to find the first exporter/runtime failure size;
- split the launch stages with `SPYRE_RESTICKIFY_LX_SPLIT_STAGES=producer`,
  `producer,dataop`, and `producer,dataop,consumer` only after the exporter
  return-code guard is in place and the device is clean.

