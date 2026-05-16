# Stage 17: Profiling Sources And Follow-Up Measurements

## Summary

This stage ran three parallel tracks:

1. continue the marker-separated Stage 15 measurement repeats;
2. query the Spyre Knowledgebase for profiling and memory-hierarchy context;
3. inspect the profiling/toolchain artifacts available in the pod because direct
   `github.ibm.com/ai-chip-toolchain` browsing requires IBM SSO from this
   session.

The important outcome is that Stage 3B's compiler-side locality signal remains
stable, but the measured runtime benefit is smaller and noisier than the first
single-run result suggested. The current tooling still cannot directly split
restickify traffic into HBM round-trip versus cross-core RIU LX-LX movement.

## Measurement Setup

The repeated run used:

```sh
tools/restickify_aiusmi_marker_probe.py \
  --case adds_then_matmul_x \
  --size 2048 \
  --mode baseline \
  --mode stage3b \
  --warmup 10 \
  --iters 5000 \
  --sample-interval 0.1
```

Five repeats alternated mode order. A size `512` run was used as a negative
control.

Artifacts were copied locally under:

```text
artifacts/restickify_stage15_followup/
```

Each repeat includes raw JSONL/CSV plus an interactive HTML report.

## Repeat Results

For `adds_then_matmul_x`, size `2048`:

| Mode | Repeats | Mean median latency | Modeled byte-hops | Avg rdmem | Avg wrmem |
|---|---:|---:|---:|---:|---:|
| baseline | 5 | `1.7768 ms` | `67,108,864` | `73.84 GiB/s` | `26.05 GiB/s` |
| Stage 3B | 5 | `1.7425 ms` | `0` | `70.88 GiB/s` | `26.52 GiB/s` |

Paired speedups:

| Repeat | Speedup | Delta |
|---|---:|---:|
| 0 | `1.0294x` | `0.0508 ms` |
| 1 | `1.0072x` | `0.0128 ms` |
| 2 | `1.0298x` | `0.0514 ms` |
| 3 | `1.0066x` | `0.0117 ms` |
| 4 | `1.0260x` | `0.0450 ms` |

Mean paired speedup was `1.0198x`, with a range of `1.0066x` to `1.0298x`.

For the `512` negative control:

| Mode | Median latency | Modeled byte-hops | Avg rdmem | Avg wrmem |
|---|---:|---:|---:|---:|
| baseline | `0.1602 ms` | `1,376,256` | `27.74 GiB/s` | `16.51 GiB/s` |
| Stage 3B | `0.1619 ms` | `655,360` | `27.74 GiB/s` | `16.51 GiB/s` |

The control behaves as expected: partial modeled byte-hop reduction without a
meaningful runtime win.

## Interpretation

Stage 3B continues to prove the narrow compiler claim:

- eligible in-graph restickify byte-hops can be removed by aligning physical
  core ownership;
- restickify count and bytes moved remain unchanged;
- graph-input/weight restickifies remain outside the Stage 3B scope.

The runtime result should now be stated as a small directional improvement, not
as a stable `4-5%` win. In this repeated measurement the high-signal `2048`
case averaged roughly `2%`, with repeat-to-repeat variation.

The `aiu-smi` aggregate read/write memory counters move only slightly:

- baseline average read bandwidth: `73.84 GiB/s`;
- Stage 3B average read bandwidth: `70.88 GiB/s`;
- write bandwidth is essentially unchanged.

That is consistent with the full fused workload still doing substantial device
memory traffic from inputs, weights, matmul, output, and runtime activity. It is
not enough to prove the restickify edge itself went HBM or LX-LX.

## Knowledgebase Findings

The Knowledgebase points to RFC 0601 as the design source for the profiling
stack. It describes a six-part toolkit:

- AIU SMI for device-level metrics;
- PyTorch profiler integration through `REGISTER_PRIVATEUSE1_PROFILER`;
- DDR and scratchpad memory profiling;
- Holistic Trace Analyzer for derived metrics;
- IR instrumentation for intra-kernel profiling;
- Inductor provenance tracking.

It also confirms the conceptual hardware model we have been using:

- 32-core AIU topology with a bidirectional ring;
- per-core LX scratchpad;
- off-chip memory and cross-core data movement both flow through ring-facing
  paths;
- profiling data should ultimately flow through `libaiupti` into Kineto/Chrome
  traces.

The KB does not currently give us fabric-specific counter names for separating
HBM/off-chip traffic from RIU cross-core LX-LX traffic.

## Installed Toolchain Findings

Direct browsing of `https://github.ibm.com/ai-chip-toolchain` is blocked by IBM
SSO from this session, and desktop `gh` is not logged in to
`github.ibm.com`. The pod-local installed artifacts still answer part of the
question.

The installed `libaiupti` headers expose activity records for:

- compute;
- memset;
- memory;
- memcpy and memcpy2;
- runtime and driver;
- synchronization;
- event;
- environment;
- metric.

The event IDs visible in the installed header are cycle and power oriented:

- `CYCLES`;
- `CYCLES_TS1` through `CYCLES_TS5`;
- `POWER`.

The only metric ID visible in the installed header is:

- `HPM`.

`AIUpti_ActivityCompute` includes `correlation_id`, timing fields, local-memory
total, and a fixed-size kernel `name`. That is the best current hook for stable
mapping from generated SDSC/opfunc names to profiler activity records.

The runtime callback IDs include useful windows:

- launch control block;
- launch compute;
- graph execute;
- super-node execute;
- node compute;
- prepare DMAs;
- host/device data transfer;
- compile graph.

`aiu-smi` exposes data-rate groups:

- `rdmem`;
- `wrmem`;
- `rxpci`;
- `txpci`;
- `rdrdma`;
- `wrrdma`.

It also exposes request-rate groups such as `n_rdmem` and `n_wrmem`. These are
useful aggregate counters, but they are not named as RIU-vs-HBM fabric counters.

`aiu-trace-analyzer` supports Chrome trace counter events and derived counters
such as power, bandwidth, collectives bandwidth, prep queue, and RCU
utilization. It expects `deviceProperties` metadata in torch traces, which
explains the current ingestion failure on traces missing that shape.

## What Is Still Missing

The missing items are now sharper:

1. A metric catalog for `AIUPTI_ACTIVITY_METRIC_ID_HPM`, or an internal header
   with named HBM/RIU/LX event IDs.
2. A clear definition of `aiu-smi` `rdmem`/`wrmem`: whether they count only
   off-chip memory-controller traffic or broader ring-facing memory movement.
3. Fabric-specific counters separating HBM/off-chip traffic from cross-core
   RIU LX-LX traffic.
4. `deviceProperties` metadata in torch profiler exports so
   `aiu-trace-analyzer` can ingest current traces cleanly.
5. A stable SDSC/opfunc-to-profiler mapping using activity `name`,
   `correlation_id`, and runtime callback windows.

## Recommendation

Keep Stage 3B framed as a narrow default-off compiler-locality prototype. The
repeat data is good enough to justify continued measurement and tooling work,
but not enough to claim a broad runtime optimization.

The next best work is not more Stage 3B tuning. It is profiler/counter
enablement:

1. get `deviceProperties` into the exported trace;
2. enable AIUPTI event/metric records in the PrivateUse1 bridge;
3. identify the HPM/fabric counter catalog;
4. bind generated restickify opfunc names to profiler events and counter
   windows;
5. then rerun isolated restickify and fused cases with fabric-specific evidence.

