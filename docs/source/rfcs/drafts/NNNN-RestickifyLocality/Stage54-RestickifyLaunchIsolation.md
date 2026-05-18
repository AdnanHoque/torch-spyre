# Stage 54: ReStickify Launch Isolation

## Summary

Stage 54 added a probe-only launch wrapper around generated SDSC bundle runs and
used it to separate three possible blockers:

1. the DDL bridge itself,
2. the transition from the add bundle to the mm bundle,
3. normal `ReStickifyOpHBM` bundle completion.

The result is clear: the current hardware blocker is not DDL-specific. A normal
baseline `ReStickifyOpHBM` bundle launches asynchronously but does not retire
under the tested pod/runtime state.

## Probe Change

`tools/restickify_scenario_probe.py` now has two default-off debug flags:

```sh
--kernel-launch-log
--sync-after-kernel
```

The wrapper patches only the Python probe process:

```text
SpyreSDSCKernelRunner.run()
  -> before_launch JSONL event
  -> original launch_kernel(...)
  -> after_launch JSONL event
  -> optional before_sync / after_sync JSONL events
```

No production runtime behavior changes unless the probe flag is used.

## Experiment 1: Baseline With Explicit Sync

Command shape:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=0 \
python3 tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 1280 \
  --ring-telemetry \
  --skip-correctness \
  --sync-after-kernel \
  --kernel-launch-log \
  --output-dir /tmp/stage54-baseline-sync-1280 \
  --fail-on-error
```

Result: timeout in `RuntimeStream::synchronize`.

Launch log:

```json
{"kernel_name":"sdsc_fused_add_t_0","phase":"before_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json","sdsc_2_add.json"]}
{"kernel_name":"sdsc_fused_add_t_0","phase":"after_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json","sdsc_2_add.json"]}
{"kernel_name":"sdsc_fused_add_t_0","phase":"before_sync","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json","sdsc_2_add.json"]}
```

Interpretation: the add bundle returns from `launch_kernel`, but explicit stream
sync after that bundle never completes. The mm bundle is not launched in this
variant.

The same explicit-sync behavior reproduced at sizes `128`, `512`, and `1280`.

## Experiment 2: Baseline With Launch Logging Only

Command shape:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=0 \
python3 tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 1280 \
  --ring-telemetry \
  --skip-correctness \
  --kernel-launch-log \
  --output-dir /tmp/stage54-baseline-logonly-1280 \
  --fail-on-error
```

Result: timeout in `PfRuntimeScheduler::issueBarrier` while loading the mm
bundle.

Launch log:

```json
{"kernel_name":"sdsc_fused_add_t_0","phase":"before_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json","sdsc_2_add.json"]}
{"kernel_name":"sdsc_fused_add_t_0","phase":"after_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json","sdsc_2_add.json"]}
{"kernel_name":"sdsc_fused_mm_1","phase":"before_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_batchmatmul.json"]}
```

Interpretation: the natural run also waits for the add bundle to retire before
the next program H2D. That wait happens inside the mm bundle launch, before
`launch_kernel` returns for the mm bundle.

## Experiment 3: DDL Bridge With Launch Logging Only

Command shape:

```sh
SPYRE_RESTICKIFY_DDL_BRIDGE_E2E=1 \
SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM=1 \
python3 tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 1280 \
  --ring-telemetry \
  --skip-correctness \
  --kernel-launch-log \
  --output-dir /tmp/stage54-ddl-logonly-1280 \
  --fail-on-error
```

Result: same timeout in `PfRuntimeScheduler::issueBarrier`.

Launch log:

```json
{"kernel_name":"sdsc_fused_add_t_0","phase":"after_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json","sdsc_2_add.json"]}
{"kernel_name":"sdsc_fused_mm_1","phase":"before_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM_ddl_bridge.json","sdsc_1_batchmatmul.json"]}
```

DDL audit:

```json
{"source_kind":"in_graph_computed","source_name":"buf1","status":"emitted"}
```

Interpretation: the DDL bridge is emitted, but the run does not reach mm bundle
completion. It blocks at the same boundary as baseline, before the DDL bridge can
be judged.

## Experiment 4: Single-Bundle Pointwise Restickify

Command shape:

```sh
python3 tools/restickify_scenario_probe.py \
  --case pointwise_transpose_add \
  --size 128 \
  --ring-telemetry \
  --skip-correctness \
  --kernel-launch-log \
  --output-dir /tmp/stage54-pointwise-logonly-128 \
  --fail-on-error
```

Result: timeout in final `RuntimeStream::synchronize`.

Launch log:

```json
{"kernel_name":"sdsc_fused_add_t_0","phase":"before_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json"]}
{"kernel_name":"sdsc_fused_add_t_0","phase":"after_launch","sdsc_files":["sdsc_0_ReStickifyOpHBM.json","sdsc_1_add.json"]}
```

Interpretation: a single normal HBM restickify plus add bundle launches, but the
stream never reports completion.

## Control: Plain Compiled Add

Control script:

```python
def fn(a, b):
    return a + b

x = torch.randn((128, 128), dtype=torch.float16, device="spyre")
y = torch.randn((128, 128), dtype=torch.float16, device="spyre")
compiled = torch.compile(fn, backend="inductor", dynamic=False)
z = compiled(x, y)
torch.accelerator.synchronize()
```

Result:

```text
ok torch.Size([128, 128])
```

Launch log:

```json
{"kernel_name":"sdsc_fused_add_0","phase":"before_launch","sdsc_files":["sdsc_0_add.json"]}
{"kernel_name":"sdsc_fused_add_0","phase":"after_launch","sdsc_files":["sdsc_0_add.json"]}
```

Interpretation: compiled Spyre compute and stream synchronize are not generally
broken in the pod. The observed hang is tied to bundles containing
`ReStickifyOpHBM`.

## Conclusion

The DDL bridge runtime blocker moved one level upstream:

- Stage 53 fixed the DDL bridge segment mapping bug.
- Stage 54 shows the current hardware run cannot validate DDL because even
  baseline `ReStickifyOpHBM` bundles do not retire in these probe runs.
- Plain compiled add runs and synchronizes successfully.

So the next investigation target is normal `ReStickifyOpHBM` execution, not the
DDL bridge.

## Next Step

Build a minimal hardware reproducer for normal HBM restickify:

1. generate a single `ReStickifyOpHBM` SDSC without fused add or matmul;
2. run it with explicit input/output tensors and launch logging;
3. compare against a single `add` SDSC control;
4. inspect the generated restickify `execute_dsg.txt`, `segment_size.json`, and
   senprog for missing completion/fence/status behavior;
5. only return to DDL once normal `ReStickifyOpHBM` retires cleanly.
