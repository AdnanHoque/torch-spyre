# Stage 9: Kernel Timing Profiler Prototype

## Summary

This stage adds an experimental kernel-timing path for restickify probes using
`torch.profiler` with `ProfilerActivity.PrivateUse1`. The goal is to connect
compiler restickify telemetry with AIU kernel timings now that the profiler
environment can emit device activity.

This is not production-ready. It depends on the separate PyTorch 2.12 /
PR #1856 profiler environment and includes a narrow workaround for PyTorch
2.12 fake-tensor compilation.

## Prototype Changes

- `tools/restickify_scenario_probe.py` and `tools/restickify_hierarchy_sweep.py`
  accept `--torch-profiler`.
- Profiler output is written as Chrome trace JSON plus per-event JSON/CSV.
- CSV rows report profiler event count, device event count, total device time,
  and paths to the trace/event artifacts.
- `setup.py` uses C++20 for the profiler/PyTorch 2.12 build path.
- Spyre eager `_copy_from` skips the compiled `copy_from_d2d` path while Python
  dispatch is excluded, which avoids nested compilation during PyTorch 2.12
  fake-tensor handling.

## Current Result

In the profiler environment:

```text
/home/adnan-cdx/dt-inductor-profiler/.venv-py212
/home/adnan-cdx/dt-inductor-profiler/torch-spyre-pr1856
```

After the fake-copy workaround, a basic Spyre compile succeeds:

```python
a = torch.randn((8, 8), dtype=torch.float16).to("spyre")
torch.compile(lambda x, y: x + y, backend="inductor")(a, a)
```

The first restickify profiler smoke also succeeds:

```text
case=adds_then_matmul
size=2048
restickifies=2
total_bytes=16,777,216
profiler_device_event_count=2
profiler_total_device_ms=7.405 over 5 iterations
```

The profiler reports fused SDSC bundle timings:

```text
sdsc_fused_add_t_0.../bundle.mlir   0.666 ms average
sdsc_fused_mm_1.../bundle.mlir      0.815 ms average
```

No event is currently named `ReStickifyOpHBM`. The profiler therefore measures
fused kernel time, not isolated restickify op time.

## Stage 3B Kernel Timing Check

A disposable combined worktree was created in the pod from PR #1856 plus this
branch:

```text
/home/adnan-cdx/dt-inductor-profiler/torch-spyre-profiler-stage3b
```

The `adds_then_matmul_x` comparison produced the expected compiler telemetry:

```text
size=512   baseline byte-hops=1,376,256   stage3b byte-hops=655,360
size=2048  baseline byte-hops=67,108,864  stage3b byte-hops=0
```

The first short profiler run showed a 2048 fused-kernel improvement from
`1.734 ms` to `1.591 ms`, but a three-repeat run with 10 profiled iterations
per mode gave a more stable estimate:

```text
run  baseline_ms  stage3b_ms  delta_us  speedup
r1   1.6976       1.6532      44.4      1.0269x
r2   1.6943       1.6439      50.4      1.0307x
r3   1.6978       1.6442      53.6      1.0326x
```

The repeated 2048 result is therefore approximately `1.03x` fused-kernel
speedup when Stage 3B reduces modeled byte-hops to zero.

The per-kernel split suggests both fused bundles improve slightly:

```text
baseline add_t    ~0.678 ms
stage3b  add_t    ~0.646 ms
baseline add_mm   ~1.018 ms
stage3b  add_mm   ~1.000 ms
```

## Stage 3B Size Sweep

The combined profiler + Stage 3B worktree was then swept over larger square
sizes for `adds_then_matmul_x`, with three repeats per size and ten profiled
iterations per mode. The compiler telemetry and profiler device-time deltas
were:

```text
size  baseline_hops  stage3b_hops  saved_hops  RIU_agg_bound  observed_delta  speedup
1024    11,141,120     1,048,576   10,092,544      30.3 us        -2.8 us     0.9919x
1536    37,748,736    18,874,368   18,874,368      56.7 us       -12.9 us     0.9833x
2048    67,108,864             0   67,108,864     201.5 us        49.5 us     1.0301x
3072   150,994,944    75,497,472   75,497,472     226.7 us        43.8 us     1.0116x
```

`RIU_agg_bound` is `saved_hops / 333 GB/s`. Using one RIU direction would
double those lower-bound estimates.

The per-bundle split was mixed:

```text
size  add_t_delta  add_mm_delta
1024      4.5 us      -7.2 us
1536     -1.3 us     -11.6 us
2048    ~32.0 us     ~18.0 us
3072     69.9 us     -26.1 us
```

This is a useful correction to the early hypothesis. Reducing modeled
byte-hops is necessary for this optimization to matter, but it is not
sufficient to guarantee lower fused-kernel time. At `1024` and `1536`, Stage
3B improves the modeled locality but the fused `add_mm` bundle gets slightly
slower. At `2048`, both bundles improve and the result is stable. At `3072`,
the transpose/add bundle improves substantially, but the matmul bundle loses
some of that gain.

The current result should be stated narrowly: Stage 3B can produce a real,
repeatable fused-kernel speedup for an eligible in-graph restickify at the
right shape, but the profiler data does not support a blanket bandwidth-only
model where saved byte-hops directly convert to saved runtime.

## Baseline Restickify Family Profiler Sweep

A baseline profiler sweep over the current restickify probe families also ran
successfully for sizes `128`, `512`, and `2048`:

```text
pointwise_transpose_add
pointwise_three_mixed
matmul_lhs_wrong_stick
matmul_rhs_wrong_stick
adds_then_matmul
matmul_then_add
transpose_chain
fanout_diamond
linear_weight_transposed
```

At `2048`, the fused device time ranged from about `0.39 ms` for
`pointwise_transpose_add` to about `1.40 ms` for `adds_then_matmul`. These
events are still fused SDSC bundles, not isolated restickify kernels, so the
sweep is useful for ranking restickify-heavy scenarios but not for directly
classifying the physical memory path of an individual `ReStickifyOpHBM`.

The saved artifacts are in the pod under:

```text
/tmp/restickify-kernel-timing-sweep
/tmp/restickify-kernel-stage3b-comparison
/tmp/restickify-kernel-stage3b-repeat
/tmp/restickify-kernel-stage3b-size-sweep
```

## Next Measurements

1. Force or find a probe where restickify is isolated into its own measurable
   SDSC bundle, instead of being fused into `add_t` or `add_mm`.
2. Inspect generated SDSC/op-func metadata for `ReStickifyOpHBM` and determine
   whether the name means the op materializes through HBM, uses HBM-addressed
   tensors, or is simply the current DeepTools lowering name for layout
   materialization.
3. Rerun the high-signal `2048` and `3072` cases with AIUPTI/DeepTools counters
   if RIU/HBM traffic counters are exposed in the profiler environment.
4. Add a second high-signal producer-to-restickify pattern that is not
   `adds_then_matmul_x`, so the Stage 3B evidence is not anchored to a single
   synthetic graph.

## Interpretation

The profiler path is now useful for answering whether restickify-heavy graphs
have meaningful fused-kernel runtime. It does not yet prove the physical path of
`ReStickifyOpHBM`, and it cannot isolate restickify cost unless a future probe
forces restickify into its own measurable bundle or DeepTools/AIUPTI exposes
per-opfunc timing.

The bandwidth comparisons should therefore be treated as plausibility bounds.
For example, the 2048 case saves `67,108,864` modeled byte-hops. A literal,
serialized transfer over a `333 GB/s` bidirectional RIU would have a lower bound
of about `201.5 us`, while the observed fused-kernel improvement is about
`49.5 us`. That mismatch does not invalidate the locality signal, but it means
the compiler byte-hop model is not yet a hardware counter. Transfers may overlap
with compute, may not be fully serialized, may not map one-to-one to RIU bytes,
or may be hidden inside HBM-named DeepTools restickify opfuncs.
