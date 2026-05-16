# Stage 11: Profiler Counter Probe

## Summary

This stage tried to move from timing-only evidence to counter evidence for the
restickify memory path. The question was whether the current profiler stack can
distinguish these physical cases:

1. a round trip through HBM/global device memory,
2. suboptimal LX-to-LX cross-core movement over the RIU data ring, and
3. optimal local LX ownership where the same core keeps the same logical region.

The result is useful but not yet the final fabric proof:

- `torch.profiler` with `ProfilerActivity.PrivateUse1` gives repeatable
  per-bundle kernel timing.
- `profile_memory=True` adds device allocator events for AIU tensors.
- The current exported traces do not expose RIU/HBM byte counters.
- `aiu-trace-analyzer` installs successfully, but it cannot currently ingest the
  torch profiler traces produced by this prototype because the traces are
  missing the AIU `deviceProperties` metadata shape it expects.

So we can measure restickify-associated kernel time, and we can observe device
allocations, but we still cannot directly say "these N bytes went over RIU" or
"these N bytes went to HBM" from the current trace alone.

## Tool Inventory

The profiler pod has the runtime AIUPTI pieces installed:

```text
/opt/ibm/spyre/runtime/lib/libaiupti.so
/opt/ibm/spyre/runtime/include/libaiupti
/opt/ibm/spyre/runtime/include/flex/telemetry
```

The pod does not currently have these tools on `PATH`:

```text
aiu-smi
aiu-monitor
aiu-trace-analyzer / acelyzer
aiupti-trace
acprof
```

For this stage, `aiu-trace-analyzer` was installed into the separate profiler
venv:

```text
/home/adnan-cdx/dt-inductor-profiler/.venv-py212/bin/acelyzer
```

The installed AIUPTI headers define activity kinds beyond what we currently see
in the trace, including `EVENT` and `METRIC`. That is important because the
likely long-term answer is to enable or bridge those records, not to infer
fabric traffic only from kernel names.

## Current Torch Profiler Bridge

The PR-1856 style bridge exposes useful activity categories:

```text
CMPT        -> kernel events
RUNTIME     -> privateuse1_runtime events
DRIVER      -> privateuse1_driver events
MEMCPY      -> gpu_memcpy events
MEMSET      -> gpu_memset events
MEMORY      -> [memory] allocator instant events
```

In the traces captured here, `ReStickifyOpHBM` appears inside SDSC compute
bundles. It does not appear as a separate memcpy event. This means the profiler
can time the bundle, but it cannot currently split the bundle into HBM traffic,
RIU LX-LX traffic, and compute/control overhead.

## Fused Stage 3B Memory-Profile Run

Command shape:

```sh
python3.12 -u tools/restickify_hierarchy_sweep.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --time \
  --warmup 5 \
  --iters 20 \
  --skip-correctness \
  --torch-profiler \
  --torch-profiler-memory \
  --output-dir /tmp/restickify-profile-memory/fused-stage3b \
  --fail-on-error
```

High-level result:

```text
mode      restickifies  bytes moved  byte-hops   median_ms
baseline             2   16,777,216  67,108,864  1.7688
stage3b              2   16,777,216           0  1.7333
```

The short run shows a `35.5 us` fused median improvement, or about `1.020x`.
This is lower than the earlier 50-iteration runs, but directionally consistent.

Per-bundle profiler averages:

```text
bundle              baseline_avg_ms  stage3b_avg_ms  delta_us
fused_add_t_0              0.6825          0.6480      34.6
fused_add_mm_t_1           1.0187          1.0011      17.6
```

The memory-profile traces had the same event shape in both modes:

```text
trace events      519
kernel events      40
runtime events     40
memory events     240
AIU memory bytes   2,013,265,920 absolute bytes over 20 iterations
```

Those `[memory]` events are allocator instant events. They record allocation and
free activity such as `+8,388,608` and `-8,388,608` bytes for AIU device type
20. They are not HBM transaction counters and should not be interpreted as
physical memory traffic.

## Isolated Restickify Memory Trace

For the isolated `transpose_contiguous` restickify at `2048`, the memory-profile
trace contained:

```text
trace events       129
kernel events       10
runtime events      10
memory events       20
AIU memory bytes    167,772,160 absolute bytes over 10 iterations
```

Each iteration allocates and frees one `8,388,608` byte tensor. This matches the
expected output tensor size, but again it is allocator telemetry, not a direct
fabric or HBM byte counter.

## acelyzer Result

`acelyzer` installed successfully from the public repository, but ingestion of
the torch profiler traces failed before analysis:

```text
AssertionError:
Combining incoming metadata with multiple deviceProperties is not supported []
```

This happened for:

```text
/tmp/restickify-isolated-sweep/s2048/transpose_contiguous/trace.json
/tmp/restickify-profile-memory/fused-stage3b/baseline/.../torch_profiler_trace.json
/tmp/restickify-profile-memory/fused-stage3b/stage3b/.../torch_profiler_trace.json
```

Trying `acelyzer --tb` did not change the failure. The issue appears to be a
trace-format compatibility problem: the torch profiler trace produced by this
prototype does not contain the AIU `deviceProperties` metadata expected by
`aiu-trace-analyzer`.

## Interpretation

The evidence now separates into three tiers:

1. Strong timing evidence:
   `torch.profiler` measures the restickify-containing SDSC bundles and shows
   Stage 3B reducing fused runtime for `adds_then_matmul_x` at `2048`.

2. Strong allocator evidence:
   `profile_memory=True` confirms AIU device allocations at the expected tensor
   byte sizes, including isolated restickify output allocation.

3. Missing fabric evidence:
   the current trace does not report HBM bytes, RIU bytes, byte-hops, or
   per-fabric counters. The compiler byte-hop model remains a compiler model
   until AIUPTI metric/event records or another hardware counter source is
   wired through.

This means the current conclusion should stay conservative:

- isolated `ReStickifyOpHBM` timing is much closer to HBM/RIU movement than to
  a perfectly local LX-only bound;
- Stage 3B eliminates modeled in-graph RIU byte-hops for the high-signal case;
- direct physical confirmation of the fused path still requires counter support.

## Next Counter Work

The next profiler task should be one of:

1. extend the PR-1856 bridge to enable and emit AIUPTI `EVENT` and `METRIC`
   records into the Chrome trace;
2. patch or configure the trace export so `acelyzer` receives the AIU
   `deviceProperties` metadata it expects;
3. obtain `aiu-monitor` / `aiu-smi` or an internal AIUPTI counter sampler that
   can report HBM and RIU traffic during the isolated and fused probes.

The two recommended validation probes remain:

```text
isolated: a.t().contiguous() at 2048
fused:    adds_then_matmul_x at 2048, baseline vs Stage 3B
```

The isolated probe is the cleanest way to validate HBM/global movement. The
fused probe is the cleanest way to validate whether Stage 3B's zero byte-hop
certificate corresponds to reduced physical RIU traffic.

## Artifacts

Pod artifacts:

```text
/tmp/restickify-profile-memory/normal
/tmp/restickify-profile-memory/fused-stage3b
/tmp/restickify-acelyzer
/tmp/restickify-acelyzer-tb
```
