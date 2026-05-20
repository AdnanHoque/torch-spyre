# Stage 153: Device Safety Gate

## Summary

We checked whether the Stage 152 LX bridge frame was safe to run on hardware.
The answer is:

```text
compile-only frame generation: safe and validated
current split-launch hardware validation path: not safe
```

The device itself was not obviously wedged before the experiment.  A tiny stock
Torch-Spyre smoke test passed.  But the experimental split path still triggered a
runtime scheduler compute hardware error.

## Pre-Run Checks

The Stage 152 bridge artifact remained valid:

```text
status=ok
contract=schema-v4-lx-materialization-contract
frame_bytes=17664
frame_flits_128b=138
HBM=0
L3LU=96
L3SU=96
LXLU=64
LXSU=64
```

`aiu-smi` was available from the profiler environment, but needed the profiler
venv site-packages on `PYTHONPATH`:

```sh
export PYTHON_CMD=/home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/python
export PYTHONPATH=/home/adnan-cdx/dt-inductor-profiler/.venv-py212/lib64/python3.12/site-packages:${PYTHONPATH:-}
/home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/aiu-smi -s -g A -d 1
```

The monitor reported an idle device, but power/temp and traffic were all zero in
this container path, so it is useful mostly as a lightweight guardrail rather
than a complete health proof.

A tiny stock kernel passed:

```text
plain_adds_then_matmul, size=64
status=ok
restickifies=0
device_events=0
```

## Experimental Run

We then ran the bounded bridge validation:

```sh
SPYRE_RESTICKIFY_RING_TELEMETRY=1 \
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
SPYRE_RESTICKIFY_LOCALITY_ASSERT=1 \
SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR=1 \
SPYRE_RESTICKIFY_LX_SPLIT_STOP_AFTER_BRIDGE=1 \
SPYRE_RESTICKIFY_LX_SPLIT_REQUIRE_MATERIALIZATION_CONTRACT=1 \
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 2048 \
  --ring-telemetry \
  --copy-kernel-code \
  --kernel-launch-log \
  --lx-split-dataop-prototype \
  --validate-tuple-prefix 1 \
  --output-dir /tmp/stage152-device-bridge-run-2048 \
  --fail-on-error
```

Result:

```text
RAS::RUNTIMESCHEDULER::ComputeHardwareError
RAS::RUNTIMESCHEDULER::StreamInErrorState
```

The launch log showed:

```text
lx_split_dataop_before_producer
lx_split_dataop_after_producer
lx_split_dataop_before_dataop
lx_split_dataop_after_dataop
lx_split_dataop_before_consumer
lx_split_dataop_after_consumer
lx_split_dataop_launch_done
sdsc_fused_mm_1: lx_split_dataop_stop_after_bridge_skip
```

The key detail is that the current "stop after bridge" split harness still
launches the split consumer inside the first fused add bundle before skipping the
later matmul.  That means this run is still exercising the consumer-side split
contract that Stage 151 already identified as invalid.

## Interpretation

This is not evidence that the Stage 152 bridge frame is wrong.  It is evidence
that the current split-launch validation harness is the wrong way to run it on
hardware.

The safe rule going forward is:

```text
Do not run the LX bridge through the split producer/data-op/consumer harness.
```

The next hardware attempt should happen only after same-artifact replacement,
where producer, replacement bridge frame, and consumer remain inside the normal
fused runtime bundle ordering.

## Next Step

Implement a compile-only same-artifact splice:

1. run DXP debug on the original fused code directory to recover frame sizes;
2. replace the `ReStickifyOpHBM` frame with the Stage 152 sentinel-cleared bridge
   frame;
3. update `segment_size.json` and `spyreCodeDir` size metadata;
4. verify the package without launching hardware;
5. launch only after the package inspection shows there is no split consumer
   harness involved.
