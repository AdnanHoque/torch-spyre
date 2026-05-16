# Stage 20: AIUPTI Metric Activity Visibility Probe

## Summary

This stage tested a narrow profiler visibility hack: can the PR-1856
PrivateUse1 profiler bridge receive AIUPTI metric activity records while a
Spyre workload also enables the senlib/aiu-smi monitoring path?

Result: **not yet**. The profiler bridge can capture PrivateUse1 kernel timing,
and senlib monitoring can write the usual `/tmp/metrics.<bus>` file, but no
`AIUPTI_ACTIVITY_KIND_METRIC` records arrived in the Kineto activity buffer.
That means the currently exposed monitoring path still appears to bypass the
torch-profiler AIUPTI activity bridge.

## Branches And Artifacts

- Profiler prototype branch: `AdnanHoque/profiler-metric-visibility`
- Base: `torch-spyre` PR 1856 head
- Commit: `602a65f debug: expose aiupti metric activities`
- Pod worktree: `/home/adnan-cdx/dt-inductor-profiler/torch-spyre-pr1856`
- Smoke output:
  - `/tmp/restickify-profiler-metric-smoke2`
  - `/tmp/restickify-profiler-metric-smoke3`

The pod commit was also mirrored and pushed to the public fork branch because
the pod itself has no GitHub HTTPS credentials.

## What Changed In The Prototype

The profiler branch adds a default-off debug path:

- `AIUPTI_ENABLE_METRICS=1` or `AIUPTI_METRIC_TRACE=1` now asks libaiupti to
  enable `AIUPTI_ACTIVITY_KIND_METRIC`.
- The PR-1856 activity handler now accepts `AIUPTI_ACTIVITY_KIND_METRIC`.
- If metric activity records arrive, the handler writes one JSONL row per
  record to `AIUPTI_METRIC_ACTIVITY_JSONL`, falling back to
  `AIUPTI_RAW_METRIC_JSONL`.
- The handler also emits metric records as Kineto instant events named
  `[aiupti_metric]`.
- `setup.py` is compiled with `-std=c++20` for the PyTorch 2.12 profiler env.

## Smoke Results

The repaired profiler env is:

- Python 3.12
- PyTorch `2.12.0+cu130`
- `ProfilerActivity.PrivateUse1` available
- `torch_spyre._C.so` imports successfully
- `_C.so` links to the patched libaiupti:
  `/home/adnan-cdx/dt-inductor-profiler/raw-metric-install/libaiupti/lib/libaiupti.so`

Smoke command shape:

```sh
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 128 \
  --skip-correctness \
  --torch-profiler \
  --warmup 0 \
  --iters 1 \
  --output-dir /tmp/restickify-profiler-metric-smoke3 \
  --fail-on-error
```

Observed torch profiler events:

| Event | Device time |
|---|---:|
| `sdsc_fused_add_t_0.../bundle.mlir` | about `0.0037 ms` |
| `sdsc_fused_mm_1.../bundle.mlir` | about `0.0029 ms` |

The run with `SENLIB_DEVEL_CONFIG_FILE` set also created:

```text
/tmp/metrics.0000:aa:00.0
```

However, neither of these appeared:

```text
/tmp/restickify-profiler-metric-smoke3/metric_activities.jsonl
/tmp/restickify-profiler-metric-smoke3/raw_metric_terms.jsonl
```

The trace also did not contain `[aiupti_metric]` events.

## Interpretation

This separates three layers:

1. **PrivateUse1 kernel timing works.**
   The PR-1856 path can report generated SDSC kernel timings into torch
   profiler traces.

2. **Senlib monitoring works.**
   With the aiu-smi senlib config, the runtime writes a metrics file and logs
   HMI/MCI/DMA/RDMA metric initialization.

3. **Metric records are not reaching Kineto.**
   Enabling `AIUPTI_ACTIVITY_KIND_METRIC` in the PR-1856 bridge did not cause
   metric activity records to arrive. The evidence points to the monitoring
   path writing its metric file directly, outside the AIUPTI activity callback
   stream used by torch profiler.

This is useful even though it is a negative result: it says the next visibility
step is not just "handle metric records in torch-spyre." We either need to
bridge the senlib monitoring file into traces, identify a libaiupti API path
that actually emits metric records, or patch lower in Flex/senlib where the raw
monitoring samples are produced.

## Remaining Gaps

- `deviceProperties` is still empty in the exported Chrome trace:

```text
deviceProperties: []
ERROR: gpuGetDeviceCount failed with code 35
```

This is why `acelyzer` still cannot ingest the trace as-is.

- The exposed `/tmp/metrics.<bus>` file is still HMI/LPDDR-facing. It does not
give a named RIU data-ring counter.
- Direct RIU traffic remains unobserved by public/prototype tooling in this
stage.

## Recommended Next Step

Do not spend more time trying to make this exact PR-1856 metric handler produce
records. The next useful profiler task is one of:

1. Patch trace export/device metadata so `acelyzer` can ingest the existing
   PrivateUse1 trace.
2. Build a parser/visualizer for `/tmp/metrics.<bus>` and align those samples
   with marker-separated restickify kernel windows.
3. Inspect or patch the lower senlib/Flex monitoring writer that produces
   `/tmp/metrics.<bus>` if we want raw sample fields beyond `aiu-smi`.

For restickify optimization work, this reinforces the current stance: use
compiler byte-hop telemetry plus kernel timing for Stage 3B, and use
`aiu-smi`/senlib metrics for HMI/LPDDR evidence. Direct RIU proof is still a
tooling gap.
