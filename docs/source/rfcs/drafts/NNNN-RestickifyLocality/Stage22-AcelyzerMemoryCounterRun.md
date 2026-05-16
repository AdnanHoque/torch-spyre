# Stage 22: Acelyzer And Memory Counter Run

## Summary

This stage combined three profiler threads:

- torch-profiler PrivateUse1 traces with the Stage 21 `deviceProperties` fix
- `acelyzer` ingestion for AIU kernel timing
- marker-separated `aiu-smi` memory counters for baseline vs Stage 3B

The useful result is that we now have kernel timing and memory-counter data in
the same measurement workflow. The counters still do not expose RIU/ring traffic
directly, but they do show that Stage 3B changes modeled in-graph locality while
leaving HBM/HMI-style read/write traffic essentially unchanged.

## Artifacts

Pod outputs:

```text
/tmp/restickify-acelyzer-memory/adds_then_matmul_2048
/tmp/restickify-aiusmi-marker-stage3b
```

Important files:

```text
/tmp/restickify-acelyzer-memory/adds_then_matmul_2048/torch_profiler/adds_then_matmul_2048/torch_profiler_trace.json
/tmp/restickify-acelyzer-memory/adds_then_matmul_2048/torch_profiler/adds_then_matmul_2048/torch_profiler_events.json
/tmp/restickify-acelyzer-memory/adds_then_matmul_2048/acelyzer/processed_trace.json
/tmp/restickify-acelyzer-memory/adds_then_matmul_2048/metrics.0000:aa:00.0
/tmp/restickify-aiusmi-marker-stage3b/aiusmi_marker_rows.csv
/tmp/restickify-aiusmi-marker-stage3b/aiusmi_marker_summary.html
```

## Acelyzer Kernel Timing

Workload:

```sh
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 2048 \
  --skip-correctness \
  --ring-telemetry \
  --torch-profiler \
  --time \
  --warmup 3 \
  --iters 10
```

`acelyzer` now ingests the trace:

```text
ACELYZER_RC=0
```

The Chrome trace contains the AIU device metadata added in Stage 21:

```json
[{"id": 0, "name": "AIU 0", "type": "AIU",
  "multiProcessorCount": 32, "computeCapability": "dd2", "coreCount": 32}]
```

Top torch-profiler device events:

| Event | Calls | Total device ms | Avg device ms |
|---|---:|---:|---:|
| `sdsc_fused_mm_1.../bundle.mlir` | 10 | 8.179 | 0.818 |
| `sdsc_fused_add_t_0.../bundle.mlir` | 10 | 6.696 | 0.670 |

This confirms the deviceProperties fix moved us forward: we can now use the
standard torch-profiler trace plus `acelyzer` path for per-kernel AIU timings.

## Marker-Separated Memory Counters

Workload:

```sh
python tools/restickify_aiusmi_marker_probe.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --warmup 5 \
  --iters 200 \
  --sample-interval 0.05
```

Results:

| Mode | Median ms | Ring bytes | Ring byte-hops | Avg hops | Peak read GiB/s | Peak write GiB/s |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1.764 | 16,777,216 | 67,108,864 | 4.0 | 76.022 | 26.861 |
| Stage 3B | 1.721 | 16,777,216 | 0 | 0.0 | 73.343 | 27.505 |

Stage 3B kept restickify count and bytes moved unchanged, but reduced modeled
in-graph RIU byte-hops from `67,108,864` to `0`. The median runtime improved by
about `0.043 ms`, or roughly `2.4%` in this run.

The memory counters did not move in the same direction as the byte-hop model:
read and write bandwidth peaks remained broadly similar. That is the expected
shape if Stage 3B is improving logical core ownership/locality for an in-graph
edge, not eliminating a DRAM round trip.

## Bandwidth Plausibility

Using the AIU fabric numbers:

- RIU data ring: `166 GB/s` per direction, `333 GB/s` aggregate biring
- HBM/HMI memory data path: roughly `166 GB/s`

The baseline modeled byte-hop quantity is `67,108,864 byte-hops`. If treated as
a simple serialized RIU bandwidth load, it corresponds to about:

| Model | Bandwidth | Time |
|---|---:|---:|
| optimistic biring | 333 GB/s | ~0.20 ms |
| one direction | 166 GB/s | ~0.40 ms |

The observed median delta was smaller, about `0.043 ms`. That means the
compiler byte-hop model should be treated as a locality cost proxy, not as a
direct wall-time predictor. Possible explanations are overlap with useful work,
non-critical-path transfer time, lower effective traffic than the byte-hop proxy,
or a different implementation path for part of the restickify movement.

The memory-counter observation is stronger: Stage 3B did not reduce global
read/write traffic in the sampled `aiu-smi` view.

## Deeper Metrics Writer Hack

We also tested whether the currently active `/tmp/metrics.<bus>` writer is the
patched `libaiupti` path.

Result: it is not.

The run log reports:

```text
[monitoring.cpp: 61] Opening Metrics File: /tmp/metrics.0000:aa:00.0
```

`strings` on the installed libraries points at:

```text
/project_src/senlib/senlib/1p0/monitoring.cpp
```

The local `libaiupti` checkout also has an `Opening Metrics File` path in
`src/aiupti/aiupti_metric.cpp`, but setting the debug env for that path did not
create any raw JSONL output. The active writer for these runs is the senlib
runtime object compiled into the installed image.

The source file `/project_src/senlib/senlib/1p0/monitoring.cpp` is not present
in the pod filesystem or current local checkouts. To patch lower than
`libaiupti`, we need the senlib source repository that produced
`/opt/ibm/spyre/senlib/lib/libsenlib-dd2.so`, or we need to reverse/decode the
existing `/tmp/metrics.<bus>` file as a sidecar.

## Conclusion

We improved the profiler stack compared to where we started:

- PrivateUse1 traces now include AIU `deviceProperties`.
- `acelyzer` can ingest the trace and preserve AIU kernel timing.
- `aiu-smi` gives marker-separated memory counter samples.
- The Stage 3B result is consistent with improved in-graph core locality, not
  reduced HBM traffic.

What still has not improved:

- no direct RIU/ring traffic counter is visible
- no AIUPTI metric records appear in the torch-profiler/Kineto buffer
- `/tmp/metrics.<bus>` is still produced by a senlib monitoring path outside
  the patched `libaiupti` source we currently have

The next best move is either to obtain/build the senlib source containing
`senlib/1p0/monitoring.cpp`, or build a decoder/sidecar around the existing
metrics file format and keep using torch-profiler plus `aiu-smi` for correlated
kernel and memory timelines.
