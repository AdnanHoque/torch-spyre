# Stage 13: AIU SMI Counter Smoke

## Summary

This stage installed the `aiu-monitor` wheel and confirmed that `aiu-smi` can
produce hardware memory-bandwidth samples while a restickify-heavy workload is
running. This is the first working path beyond timing-only evidence.

The important result is not yet a final fabric attribution, but it removes the
big blocker from Stage 12:

- `aiu-smi` is now installed in the profiler venv.
- The runtime emits a metric file when `SENLIB_DEVEL_CONFIG_FILE` points to the
  `aiu-monitor` config.
- `aiu-smi` reports nonzero device read/write memory bandwidth during the
  `adds_then_matmul_x` restickify probe.
- The default metric file path `/tmp/metrics.0000:aa:00.0` works; custom metric
  paths did not work in this smoke because the runtime-side config still points
  at the default pattern.

## Installation

The downloaded wheel was copied into the pod and installed into the separate
profiler environment:

```sh
oc cp \
  ~/Downloads/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl \
  adnan-cdx-spyre-dev-pf:/tmp/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl

source /home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/activate
pip install /tmp/ibm_aiu_monitor-1.2.1+torch.spyre-py312-none-linux_x86_64.whl
```

Verification:

```text
aiu-smi 1.2.1+torch-spyre
/home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/aiu-smi
```

`aiu-smi --help` exposes the useful groups:

```text
D = device Data rate
R = device Request rate
M = reserved memory
U = actual and peak device memory usage
P = pt_act
S = segment usage
A = all metrics
```

## Required Runtime Settings

The key setting was:

```sh
export SENLIB_DEVEL_CONFIG_FILE=/home/adnan-cdx/dt-inductor-profiler/.venv-py212/etc/senlib_config_aiusmi.json
```

Without that, `aiu-smi` can run but the workload does not create the metric file
and the samples stay at zero.

Useful environment for the smoke:

```sh
export AIUPTI_ENABLE_METRICS=1
export AIUSMI_ENABLE_METRICS=1
export ENABLE_AIUPTI_ACTIVITY_KIND_EVENT=1
export ENABLE_AIUPTI_ACTIVITY_KIND_METRIC=1
export AIUPTI_SAMPLER_INTERVAL=1
```

The metric file path that worked:

```text
/tmp/metrics.0000:aa:00.0
```

The installed `senlib_config_aiusmi.json` has:

```json
"METRICS": {
  "general": {
    "enable": true,
    "path": "/tmp/metrics.%BUSID",
    "collection_interval": {
      "sleep_time": 2,
      "unit": "ms"
    }
  }
}
```

Custom `AIUPTI_METRIC_PATH` / `SPYRE_METRIC_PATH` values did not redirect the
runtime-side metric file in this smoke. For now, use the default path and copy
or rename it after each mode.

## Counter Smoke

Probe:

```text
case: adds_then_matmul_x
size: 2048
iters: 300
correctness: skipped
```

Baseline:

```text
restickifies: 2
bytes moved: 16,777,216
modeled byte-hops: 67,108,864
median_ms: 1.7618
traffic samples: 6
peak rdmem: 76.207 GiB/s
peak wrmem: 26.869 GiB/s
avg nonzero rdmem: 61.006 GiB/s
avg nonzero wrmem: 21.524 GiB/s
peak n_rdmem: 639.268 Mreq/s
peak n_wrmem: 225.392 Mreq/s
```

Stage 3B:

```text
restickifies: 2
bytes moved: 16,777,216
modeled byte-hops: 0
median_ms: 1.7504
traffic samples: 7
peak rdmem: 72.354 GiB/s
peak wrmem: 27.007 GiB/s
avg nonzero rdmem: 49.323 GiB/s
avg nonzero wrmem: 18.455 GiB/s
peak n_rdmem: 606.950 Mreq/s
peak n_wrmem: 226.555 Mreq/s
```

This run is a counter-smoke, not a statistically stable benchmark. It confirms
that the counter pipeline is alive and that the workload is generating device
read/write memory traffic. It does not yet isolate which portion of the traffic
belongs to `ReStickifyOpHBM` versus matmul, pointwise, allocation, or other
runtime activity inside the same compiled run.

## Memory Hierarchy Connection

The Spyre Knowledgebase memory hierarchy page is currently a high-level index,
but its linked microarchitecture and AIU pages provide the constraints that
matter for restickification:

- AIU has 32 cores connected by a bidirectional ring.
- Each core has a private 2 MB LX scratchpad.
- Off-chip memory is shared by all cores.
- Off-chip memory to LX movement goes through the ring-facing L3LU/L3SU path.
- Cross-core LX-to-LX movement also travels over the ring.
- Software-visible tensor layout is expressed through stick/tiled tensor
  layouts, and the allocator ultimately backs tensor storage with device
  memory regions.

That means the three physical hypotheses for restickify are still the right
ones:

1. a global-memory materialization path;
2. a cross-core LX-to-LX path with poor physical locality;
3. a local LX ownership-preserving path.

The isolated `ReStickifyOpHBM` timing from Stage 10 was far too slow to be a
pure local-LX path and much closer to an HBM/RIU data-movement bound. This
Stage 13 smoke now adds a real counter signal: the fused restickify-heavy probe
causes nonzero read/write device memory bandwidth samples while it runs.

For the fused Stage 3B case, the interpretation remains conservative:

- Stage 3B removes modeled in-graph byte-hops.
- `aiu-smi` shows that the full workload still has device memory traffic.
- The counter sample does not yet separate restickify traffic from other fused
  work.
- Therefore the next measurement should pair `aiu-smi` with an isolated
  restickify-only bundle and then with the fused Stage 3B bundle.

## Next Measurements

1. Run `aiu-smi` around the isolated `transpose_contiguous` /
   `ReStickifyOpHBM` bundle from Stage 10.
2. Increase timed iterations or use a longer-running loop so `aiu-smi` captures
   more than a handful of nonzero samples.
3. Save per-mode metric files under a stable artifact directory after each run,
   because the runtime currently writes the default `/tmp/metrics.%BUSID` path.
4. Compare isolated restickify bytes against:
   - observed `rdmem` / `wrmem`,
   - expected tensor read/write bytes,
   - HBM lower bound,
   - RIU lower bound,
   - local-LX lower bound.
5. Only after isolated attribution works, rerun the fused Stage 3B comparison
   and decide whether the remaining traffic is restickify, matmul, or ordinary
   tensor/global-memory movement.

