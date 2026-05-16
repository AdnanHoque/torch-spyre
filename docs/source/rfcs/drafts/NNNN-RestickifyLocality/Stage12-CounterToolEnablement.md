# Stage 12: Counter Tool Enablement

## Summary

This stage followed the third counter path from Stage 11: try to obtain or use
`aiu-monitor`, `aiu-smi`, or another installed counter sampler that can report
HBM and RIU traffic during isolated and fused restickify probes.

The result is partially useful but not yet sufficient for fabric attribution:

- `aiu-smi` and `aiu-monitor` are not installed in the current pod image.
- `aiu-monitor` appears to be distributed as an IBM Artifactory wheel, so it
  needs IBM package credentials or a predownloaded wheel.
- The pod does include lower-level `senlib` monitoring scripts and register
  definitions for RMI, RMO, MCI, and HMI counters.
- Direct `senlib` register access works, but holding the VFIO device open for
  sampling conflicts with running a workload in the same pod.
- AIUPTI advertises metric/event activity kinds and metric-path environment
  variables, but enabling them did not produce metric files or metric events in
  the current torch-profiler run.

So the current environment can still provide kernel timing and compiler
byte-hop telemetry, but it does not yet provide a noninvasive hardware counter
stream for RIU or HBM bytes.

## Tool Inventory

The current pod has:

```text
/opt/ibm/spyre/deeptools/bin/profiler_standalone
/opt/ibm/spyre/runtime/bin/flex_monitor
/opt/ibm/spyre/runtime/lib/libaiupti.so
/opt/ibm/spyre/runtime/include/libaiupti
/opt/ibm/spyre/senlib/bin/host_dma_r5_monitor
/opt/ibm/spyre/senlib/etc/scripts/monitoring
/opt/ibm/spyre/senlib/etc/senlib_config_monitoring.json
/opt/ibm/spyre/senlib/etc/senlib_config_monitoring_dd2.json
```

The current pod does not have these command-line tools on `PATH`:

```text
aiu-smi
aiu-monitor
aiupti-trace
acprof
```

`acelyzer` is installed only in the separate profiler venv:

```text
/home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/acelyzer
```

## aiu-monitor

The Torch-Spyre profiling docs describe `aiu-monitor` as an IBM Artifactory
wheel, for example an x86_64 Python 3.12 wheel under:

```text
https://na.artifactory.swg-devops.com/artifactory/sys-power-hpc-pypi-local/aiu-monitor/...
```

Trying to install from Artifactory inside the pod required interactive IBM
credentials. In the current noninteractive setup this blocks the install path.

This means `aiu-monitor` is still the most direct path if we can provide either:

- authenticated Artifactory access inside the pod, or
- a wheel copied into the pod and installed into the profiler venv.

## senlib Counter Scripts

The installed `senlib` monitoring scripts expose 54 configured counters after
combining the base monitoring config with the DD2 config. The useful families
for this project include:

```text
RMI::hci2rmi_write_lpddr
RMI::hci2rmi_write_valid
RMI::rmi2mci_write_req
RMI::rmi2mci_write_valid
RMO::hci2rmo_read_lpddr
RMO::rmo2mci_read_req
RMO::mci2rmo_read_resp_valid
RMO::mci2hci_read_resp_valid
MCI::req_read_hmi0
MCI::req_write_hmi0
MCI::pressure_read_any
MCI::pressure_writebus_any
```

The selector table also contains HMI SOC beat counters:

```text
HMI_SOC_PERF::wr_data_beats
HMI_SOC_PERF::rd_data_beats
```

Those would be especially useful because each beat is 128 bytes. However, the
current DD2 monitoring config selects HMI latency threshold counters rather than
the read/write beat counters, and the helper appears to map HMI SOC selections
through the generic `HMI` selector table. That makes this path a bring-up task,
not a ready-to-use sampler.

## Direct Register Probe

Direct Python access through `libsenlib.PfInterfaceWrapper` can open the device
and read DCR registers. The current RMI and RMO selector controls match the DD2
monitoring config:

```text
RMI ctrl = 0x50400601
RMO ctrl = 0x88804001
```

That corresponds to the configured write/read request and valid counters listed
above.

The blocker is concurrency. Keeping the `PfInterfaceWrapper` open while
launching a Spyre workload caused the workload to fail opening `/dev/vfio/85`
with "Device or resource busy". Reading counters from separate processes avoids
that device-open conflict, but opening a new wrapper can reinitialize or restore
counter configuration, so it is not a reliable before/after sampler for a
workload.

That makes direct `senlib` register reads useful for discovering counters, but
not yet useful as the production measurement path for restickify probes.

## AIUPTI Metric Path Probe

The installed AIUPTI headers define:

```text
AIUPTI_ACTIVITY_KIND_EVENT  = 9
AIUPTI_ACTIVITY_KIND_METRIC = 11
AIUPTI_ACTIVITY_METRIC_ID_HPM = 1
```

The installed `libaiupti.so` also contains metric-related strings and
environment variables:

```text
AIUPTI_ENABLE_METRICS
AIUSMI_ENABLE_METRICS
AIUPTI_METRIC_PATH
SPYRE_METRIC_PATH
AIUPTI_USE_NEW_FORMAT
AIUPTI_SAMPLER_INTERVAL
ENABLE_AIUPTI_ACTIVITY_KIND_EVENT
```

A tiny `adds_then_matmul` run was executed with:

```sh
ENABLE_AIUPTI_ACTIVITY_KIND_EVENT=1
ENABLE_AIUPTI_ACTIVITY_KIND_METRIC=1
AIUPTI_ENABLE_METRICS=1
AIUSMI_ENABLE_METRICS=1
AIUPTI_METRIC_PATH=/tmp/restickify-aiupti-metric-path
SPYRE_METRIC_PATH=/tmp/restickify-aiupti-spyre-metric-path
AIUPTI_SAMPLER_INTERVAL=1
```

The workload and torch profiler completed successfully, but no metric files were
created under either metric path, and the exported Chrome trace contained no
metric or HPM events.

This suggests that simply setting metric environment variables is not enough in
the current PR-1856 style profiler bridge. Either another runtime component
must start the sampler, or the bridge must explicitly enable and consume
`EVENT`/`METRIC` activity records.

## Current Conclusion

Path 3 is not fully available from the current pod image alone.

`aiu-monitor`/`aiu-smi` remains the fastest route to real counters if we can get
the internal package installed. Without that package, the next actionable
engineering path is to extend the AIUPTI profiler bridge so it enables and emits
`EVENT` and `METRIC` records into the Chrome trace.

Until one of those paths works, the restickify evidence should be framed as:

- compiler telemetry proves modeled byte-hop reductions;
- torch profiler proves fused-kernel timing changes;
- isolated `ReStickifyOpHBM` timing is consistent with data movement;
- direct RIU/HBM byte attribution is still unproven.

## Recommended Next Step

Use this decision tree:

1. If IBM Artifactory credentials or a local `aiu-monitor` wheel are available,
   install `aiu-monitor` into the profiler venv and sample HBM/RIU/MCI/HMI
   counters around `transpose_contiguous` and `adds_then_matmul_x`.
2. If the package is not available, implement the AIUPTI bridge extension for
   `AIUPTI_ACTIVITY_KIND_EVENT` and `AIUPTI_ACTIVITY_KIND_METRIC`.
3. Keep using the current kernel-timing profiler for ranking scenarios, but do
   not claim physical RIU/HBM bytes from timing alone.

