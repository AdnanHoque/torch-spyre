# Stage 14: Isolated Restickify Counters

## Summary

This stage used the working `aiu-smi` path to measure an isolated
`transpose_contiguous` materialization and a pointwise control. The goal was to
separate a simple tensor read/write path from the fused Stage 3B workload.

The main result:

- `pointwise_control = a + b` reports roughly 2:1 read/write bandwidth, matching
  two input tensors and one output tensor.
- `transpose_contiguous = a.t().contiguous()` reports roughly 1:1 read/write
  bandwidth, matching one input tensor and one output tensor.
- The isolated materialization is therefore not consistent with a purely local
  LX-only path. It is behaving like device-memory read/write traffic.

This is stronger than the timing-only Stage 10 result because the hardware
counter stream now observes device read/write memory traffic during the probe.

## Probe Change

`tools/restickify_hierarchy_sweep.py` now includes two isolated materialization
cases:

```text
transpose_contiguous: a.t().contiguous()
transpose_clone:      a.t().clone()
```

These are labeled `isolated_restickify` because previous generated-bundle
inspection showed the same `sdsc_0_ReStickifyOpHBM.json` opfunc for
`transpose_contiguous`. The compiler restickify-plan telemetry reports zero
restickifies for this case because it is not the same `optimize_restickify` /
`insert_restickify` plan surface used by Stage 3B. For this case, the evidence
comes from generated opfunc inspection, kernel timing, and `aiu-smi` counters.

## Measurement Setup

Required counter environment:

```sh
source /home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/activate
export SENLIB_DEVEL_CONFIG_FILE=/home/adnan-cdx/dt-inductor-profiler/.venv-py212/etc/senlib_config_aiusmi.json
export AIUPTI_ENABLE_METRICS=1
export AIUSMI_ENABLE_METRICS=1
export ENABLE_AIUPTI_ACTIVITY_KIND_EVENT=1
export ENABLE_AIUPTI_ACTIVITY_KIND_METRIC=1
export AIUPTI_SAMPLER_INTERVAL=1
```

For each case, `aiu-smi` sampled at `0.1s` while the probe ran many iterations.
The runtime-side metric file used the default path:

```text
/tmp/metrics.0000:aa:00.0
```

Artifacts in the pod:

```text
/tmp/restickify-aiusmi-isolated
```

## Results

### Pointwise Control

Case:

```python
def pointwise_control(a, b):
    return a + b
```

Shape and run:

```text
size: 2048
iters: 5000
median_ms: 0.2714
```

Counter summary:

```text
traffic samples: 13
peak rdmem: 58.340 GiB/s
peak wrmem: 29.162 GiB/s
avg nonzero rdmem: 54.341 GiB/s
avg nonzero wrmem: 27.161 GiB/s
peak n_rdmem: 489.387 Mreq/s
peak n_wrmem: 244.627 Mreq/s
```

Interpretation:

The read/write ratio is about `2:1`, which matches `a + b`: two tensor reads
and one tensor write. This is a useful counter sanity check.

### Isolated Transpose Contiguous, 2048

Case:

```python
def transpose_contiguous(a):
    return a.t().contiguous()
```

Shape and run:

```text
size: 2048
iters: 5000
median_ms: 0.2039
tensor bytes: 8,388,608
```

Counter summary:

```text
traffic samples: 10
peak rdmem: 38.754 GiB/s
peak wrmem: 38.720 GiB/s
avg nonzero rdmem: 35.340 GiB/s
avg nonzero wrmem: 35.309 GiB/s
peak n_rdmem: 325.095 Mreq/s
peak n_wrmem: 324.804 Mreq/s
```

Interpretation:

The read/write ratio is essentially `1:1`, matching one input tensor read and
one output tensor write. This is not what a local-LX-only ownership-preserving
path would look like. It looks like a materialization path that reads and
writes device memory.

### Isolated Transpose Contiguous, 4096

Shape and run:

```text
size: 4096
iters: 2500
median_ms: 0.6267
tensor bytes: 33,554,432
```

Counter summary:

```text
traffic samples: 15
peak rdmem: 50.475 GiB/s
peak wrmem: 50.464 GiB/s
avg nonzero rdmem: 47.154 GiB/s
avg nonzero wrmem: 47.143 GiB/s
peak n_rdmem: 423.417 Mreq/s
peak n_wrmem: 423.322 Mreq/s
```

Interpretation:

The larger shape preserves the same `1:1` read/write signature. Runtime scales
with tensor size, and the counter stream again shows balanced read/write
traffic.

## Connection To Stage 3B

This result clarifies the two different restickify-like situations:

1. **Explicit materialization**, such as `a.t().contiguous()`.
   This lowers to `ReStickifyOpHBM` in generated DeepTools bundles and now has
   counter evidence of read/write device-memory traffic.

2. **In-graph producer-to-restickify locality**, such as `adds_then_matmul_x`.
   Stage 3B changes producer/restickify work ownership and removes modeled
   byte-hops. The fused workload still has memory traffic because it also
   contains matmul, pointwise, input/output, and runtime activity.

So the isolated result does not prove that every fused Stage 3B byte-hop is an
HBM round trip. Instead, it proves that at least one clean restickify
materialization path is real device-memory traffic, and it gives us a counter
method for separating physical paths.

## Current Conclusion

The hierarchy model is now better grounded:

- Local LX-only movement would be far below these timings and should not show
  this kind of sustained read/write memory bandwidth.
- Explicit restickify materialization shows balanced read/write device-memory
  traffic.
- Fused Stage 3B improvements remain a separate locality optimization: they
  reduce modeled in-graph byte-hops, but further counter work is needed to
  isolate how much of the fused memory traffic belongs specifically to the
  restickify opfunc.

## Next Step

Run longer, marker-separated fused probes:

1. compile once without `aiu-smi`;
2. start `aiu-smi`;
3. execute only the timed loop;
4. stop `aiu-smi`;
5. compare baseline and Stage 3B using the same number of timed iterations.

That will reduce compile/allocation noise in the fused comparison and make the
counter deltas more meaningful.

