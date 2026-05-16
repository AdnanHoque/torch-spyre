# Stage 15: Marker-Separated Fused Counters

## Summary

This stage added a marker-separated `aiu-smi` probe for the fused
`adds_then_matmul_x` Stage 3B case. The goal was to remove compile/allocation
noise from the counter stream by compiling and warming up first, then sampling
only the timed loop.

The main result:

- The marker-separated path now works.
- Stage 3B still removes the modeled in-graph byte-hops for the high-signal
  `2048` case.
- `aiu-smi` still reports substantial read/write memory traffic in both
  baseline and Stage 3B.
- Therefore Stage 3B should be interpreted as a locality improvement for one
  eligible in-graph restickify edge, not as eliminating all device-memory
  traffic in the fused workload.

## Tool Change

Added:

```text
tools/restickify_aiusmi_marker_probe.py
```

The probe:

1. enables the aiu-monitor metric environment;
2. clears the default metric file before any Spyre device initialization;
3. optionally starts a primer `aiu-smi`;
4. compiles the probe once;
5. runs warmup iterations;
6. starts a measured `aiu-smi`;
7. runs only the timed loop;
8. stops `aiu-smi`;
9. writes JSONL/CSV rows with compiler telemetry, timing, and counter summaries.

One important bug was found while building this:

- Deleting `/tmp/metrics.0000:aa:00.0` after a tensor had already been moved to
  `spyre` caused all later `aiu-smi` samples to stay at zero.
- The fix is to clear the metric file before any Spyre runtime/device touch, and
  never delete it while the runtime may still have it open.

## Measurement Setup

Required environment:

```sh
source /home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/activate
export SENLIB_DEVEL_CONFIG_FILE=/home/adnan-cdx/dt-inductor-profiler/.venv-py212/etc/senlib_config_aiusmi.json
export AIUPTI_ENABLE_METRICS=1
export AIUSMI_ENABLE_METRICS=1
export ENABLE_AIUPTI_ACTIVITY_KIND_EVENT=1
export ENABLE_AIUPTI_ACTIVITY_KIND_METRIC=1
export AIUPTI_SAMPLER_INTERVAL=1
export SENCORES=32
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
```

Probe command:

```sh
python3.12 -u tools/restickify_aiusmi_marker_probe.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --warmup 10 \
  --iters 5000 \
  --sample-interval 0.1 \
  --output-dir /tmp/restickify-aiusmi-marker-fused-2048-v2 \
  --fail-on-error
```

Artifacts were copied locally under:

```text
artifacts/restickify_aiusmi_marker/
```

## Results

### Size 512

| Mode | Restickifies | Bytes Moved | Byte-Hops | Avg Hops | Max Hops | Median ms | p10 ms | p90 ms | Avg rdmem GiB/s | Avg wrmem GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2 | 1,048,576 | 1,376,256 | 1.3125 | 7 | 0.161963 | 0.159014 | 0.165332 | 27.689 | 16.486 |
| Stage 3B | 2 | 1,048,576 | 655,360 | 0.6250 | 3 | 0.161989 | 0.159992 | 0.164805 | 24.682 | 14.694 |

Derived:

```text
byte-hop reduction: 52.38%
median speedup:     0.9998x
delta:              -0.000026 ms
```

Interpretation:

The compiler locality metric improves, but the runtime difference is below
measurement noise at this size. This matches earlier observations: small shapes
do not expose a meaningful Stage 3B latency win.

### Size 2048

| Mode | Restickifies | Bytes Moved | Byte-Hops | Avg Hops | Max Hops | Median ms | p10 ms | p90 ms | Avg rdmem GiB/s | Avg wrmem GiB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2 | 16,777,216 | 67,108,864 | 4.0 | 16 | 1.767253 | 1.750905 | 1.783172 | 73.842 | 26.054 |
| Stage 3B | 2 | 16,777,216 | 0 | 0.0 | 0 | 1.728434 | 1.712941 | 1.791421 | 71.343 | 26.694 |

Derived:

```text
byte-hop reduction: 100.00%
median speedup:     1.0225x
delta:              0.038818 ms
```

The 2048 row again proves the narrow Stage 3B claim:

- restickify count stays unchanged;
- total bytes moved stays unchanged;
- the eligible in-graph modeled byte-hops drop to zero;
- runtime improves directionally, though this run shows about `2.25%` rather
  than the earlier `4-5%` best case.

## Source Attribution

For `2048`, there are two restickifies:

| Source Kind | Producer | Restickify | Bytes | Baseline Splits | Stage 3B Result |
|---|---|---|---:|---|---|
| `in_graph_computed` | `buf2` | `buf5` | 8,388,608 | producer `d1:32`, restickify `d0:32` | aligned to zero modeled byte-hops |
| `graph_input_or_weight` | `<none>` | `buf4` | 8,388,608 | no in-graph producer | skipped, attribution only |

This explains why Stage 3B can remove all modeled in-graph byte-hops while the
fused workload still has memory traffic: half of the restickify bytes are from a
graph-input/weight boundary, and the compiled graph also includes pointwise,
matmul, input, output, and runtime/device-memory activity.

## Hardware Interpretation

Using the RIU data-ring bandwidth as a rough plausibility bound:

```text
67,108,864 byte-hops / 333 GB/s ~= 0.20 ms
67,108,864 byte-hops / 166 GB/s ~= 0.40 ms
```

The observed `2048` median delta was about `0.039 ms`, much smaller than those
direct byte-hop/bandwidth bounds. That is expected because compiler
`byte_hops` is a locality-weighted model, not a calibrated standalone hardware
traffic measurement. The fused kernel can overlap work, distribute traffic over
time, and continue to perform normal device-memory reads/writes unrelated to
the optimized restickify edge.

The `aiu-smi` counters support that conservative interpretation:

- baseline average read/write was about `73.8/26.1 GiB/s`;
- Stage 3B average read/write was about `71.3/26.7 GiB/s`;
- the counter signature stays broadly similar despite the modeled byte-hop
  removal.

So Stage 3B is not proving that all restickify data stayed in local LX. It is
proving that the compiler can remove a modeled cross-core ownership mismatch
for the eligible in-graph edge. Physical fabric attribution still needs either
per-op counters, trace correlation to the generated `ReStickifyOpHBM`, or a
single-op bundle whose counter traffic can be isolated.

## Conclusion

The marker-separated fused counter path is now usable. It strengthens the Stage
3B story but also narrows it:

- good: the compiler telemetry and runtime timing are directionally consistent;
- good: the high-signal case still has zero modeled byte-hops under Stage 3B;
- important: memory counters remain high in both modes, so the fused workload is
  still dominated by other device-memory activity;
- next: use this tool for repeated runs and for cases where the restickify op is
  either isolated or dominates the fused kernel enough to show a counter delta.

