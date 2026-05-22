# Stage 217: PT-LX Benchmark Rerun

## Summary

Reran the `computed_transpose_adds_then_matmul_tuple` benchmark on the clean
Stage216 worktree at commit `4e760ef`. The run compared three modes:

- stock HBM restickify path,
- Stage3B HBM restickify mapping/alignment,
- PT-LX mixed schedule prototype.

Command shape:

```sh
python tools/restickify_scenario_probe.py \
  --case computed_transpose_adds_then_matmul_tuple \
  --size 512 --size 1024 --size 1536 --size 2048 --size 3072 --size 4096 \
  --ring-telemetry \
  --skip-correctness \
  --time --warmup 5 --iters 50 \
  --fail-on-error
```

Artifacts were copied locally under:

```text
artifacts/stage216_ptlx_bench_rerun/pod
```

## Timing Results

| Size | Stock ms | Stage3B ms | PT-LX ms | PT-LX vs Stock | PT-LX vs Stage3B | Status |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 0.114 | 0.111 | 0.110 | 1.036x | 1.009x | ok/ok/ok |
| 1024 | 0.308 | 0.305 | 0.306 | 1.007x | 0.997x | ok/ok/ok |
| 1536 | 0.612 | 0.606 | 0.604 | 1.013x | 1.003x | ok/ok/ok |
| 2048 | 1.345 | 1.347 | 1.014 | 1.326x | 1.328x | ok/ok/ok |
| 3072 | 2.880 | 2.884 | 2.999 | 0.960x | 0.962x | ok/ok/ok |
| 4096 | 6.938 | 6.954 | 6.819 | 1.017x | 1.020x | ok/ok/ok |

Representative PT-LX timing bands:

| Size | PT-LX p10 | PT-LX median | PT-LX p90 |
|---:|---:|---:|---:|
| 512 | 0.107 | 0.110 | 0.114 |
| 1024 | 0.301 | 0.306 | 0.313 |
| 1536 | 0.597 | 0.604 | 0.613 |
| 2048 | 1.007 | 1.014 | 1.026 |
| 3072 | 2.978 | 2.999 | 3.012 |
| 4096 | 6.798 | 6.819 | 6.852 |

## Interpretation

The result matches the previous story:

- `2048` is still the only patched size in this family and remains the clear
  win: about `1.33x` faster than both stock and Stage3B HBM.
- `512`, `1024`, and `1536` still do not patch to PT-LX. The timings are
  therefore effectively stock-path noise, not evidence that PT-LX helps those
  sizes yet.
- `3072` is still skipped and shows a small slowdown in this run. Treat that as
  noise plus prototype overhead, not a meaningful negative result.
- `4096` is also skipped but happened to time slightly faster. Because it did
  not patch, this is not evidence for PT-LX benefit.

The Stage216 streaming tiled planner is the right next step for broadening
coverage. The benchmark confirms that simply adding planner/audit logic did not
change the runtime picture: the current executable PT-LX implementation still
only materially improves the `2048` case.

## PT-LX Audit

The PT-LX audit showed:

- `2048`: patched as `1_MixedReStickifyOpWithPTLxConsumer`.
- all other sizes: skipped before mixed replacement because endpoints were not
  allocator-backed for the full bridge path.

This is consistent with the current prototype gate. Stage216's planner explains
how these skipped cases could be lowered later as streaming tiles, but it does
not yet emit the streaming data-op schedule.
