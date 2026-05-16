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

The saved artifacts are in the pod under:

```text
/tmp/restickify-kernel-timing-sweep
/tmp/restickify-kernel-stage3b-comparison
/tmp/restickify-kernel-stage3b-repeat
```

## Next Measurements

1. Run a synthetic restickify family sweep with `--torch-profiler`:
   `pointwise_control`, `pointwise_transpose_add`, `matmul_control`,
   `matmul_lhs_wrong_stick`, `matmul_rhs_wrong_stick`, `adds_then_matmul`,
   `matmul_then_add`, `chain_transposed_intermediate`, and
   `matmul_both_inputs_upstream_conflict`.
2. Sweep sizes `128`, `512`, `1024`, and `2048`; add `3072` only if timing is
   stable.
3. Record restickify count, bytes moved, modeled byte-hops, per-kernel device
   time, total device time, wall time, and SDSC bundle names.
4. Extend the combined profiler + Stage 3B comparison to `1024`, `1536`, and
   `3072`, with at least three repeats per size.
5. Compare observed deltas against HBM, RIU, and LX lower bounds. Treat those
   as plausibility checks because the measured events are fused kernels.

## Interpretation

The profiler path is now useful for answering whether restickify-heavy graphs
have meaningful fused-kernel runtime. It does not yet prove the physical path of
`ReStickifyOpHBM`, and it cannot isolate restickify cost unless a future probe
forces restickify into its own measurable bundle or DeepTools/AIUPTI exposes
per-opfunc timing.
