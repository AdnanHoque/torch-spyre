# AIU Kernel Profiler Skill

These notes capture the known-good profiler workflow on pod
`adnan-cdx-spyre-dev-pf` for the Torch-Spyre checkout at
`/tmp/torch-spyre-co-remap-native`, branch `swiglu-ws-co-remap`. Treat this as
a reusable runbook for coordinate-remap kernel profiling.

## Environment

Use the Python and environment from the clean Deeptools/Inductor checkout:

```bash
cd /tmp/torch-spyre-co-remap-native

export PY212=/home/adnan-cdx/dt-inductor-codex-clean/.venv-py212/bin/python
source /home/adnan-cdx/dt-inductor-codex-clean/env.sh
source /home/adnan-cdx/dt-inductor-codex-clean/matmul_gap_env.sh
use_py212_localflex_optdeeptools_spyre_runtime
```

When using the profiler overlay described below, library ordering matters.
Keep the local `libaiupti` and runtime libraries before Deeptools libraries in
`LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH}
```

If `_C.so` fails to import with an undefined `flex::AllocationDirective`
symbol, check `LD_LIBRARY_PATH` first. That failure has been caused by
Deeptools libraries taking precedence over the local flex/runtime libraries.
When invoking `tools/run_coordinate_remap_bench.py`, pass the same ordering
with `--env LD_LIBRARY_PATH=...`; wrapper `--env` values are applied after its
default Deeptools path construction.

## Profiler-enabled `_C.so` overlay

The branch build of `/tmp/torch-spyre-co-remap-native/torch_spyre/_C.so` is not
profiler-enabled. For profiler runs, temporarily overlay the profiler-enabled
binary from the clean checkout:

```bash
cd /tmp/torch-spyre-co-remap-native

backup=./torch_spyre/_C.so.pre_kernel_profiler_overlay_$(date +%Y%m%d_%H%M%S)
cp ./torch_spyre/_C.so "$backup"
cp /home/adnan-cdx/dt-inductor-codex-clean/torch-spyre/torch_spyre/_C.so ./torch_spyre/_C.so
```

This is a local runtime overlay only. Never commit the overlaid `_C.so`, the
backup file, or any `_C.so.pre_*_overlay_*` file. Restore the branch binary
after the run:

```bash
mv "$backup" ./torch_spyre/_C.so
```

## Benchmark command pattern

Use `tools/run_coordinate_remap_bench.py` to create per-variant run
directories, environment records, benchmark logs, and artifact summaries. Its
default benchmark command runs spyre-perf-suite with `--with-profiling` and
`--output "$run_dir/perf.txt"`.

Each variant run directory should contain the files needed to interpret the
run:

- `perf.txt` from spyre-perf-suite.
- `logs/...pt.trace.json` from the profiler trace.
- `artifacts/trace_summary.json` for trace-derived kernel timing.
- `artifacts/sdsc_summary.json` for SDSC structure.
- `onchip_move.jsonl` for coordinate-remap planner output.
- `env.json` and `commands.json` with SHAs, paths, flags, and exact commands.

Use at least 3 runs, usually 5, when profiling. A one-run perf-suite job can
hit `ZeroDivisionError` because there are no active profiling iterations.

## Small SwiGLU examples

Flat `mm` mode:

```bash
cd /tmp/torch-spyre-co-remap-native

$PY212 tools/run_coordinate_remap_bench.py \
  --output-root /tmp/small_swiglu_flat_mm_profile \
  --torch-root /tmp/torch-spyre-co-remap-native \
  --deeptools-root /tmp/deeptools-coordinate-remap-mainport-lean \
  --perf-suite-root /home/adnan-cdx/spyre-perf-suite \
  --runs 5 \
  --env LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH} \
  --env SPYRE_SMALL_SWIGLU_MODE=flat_mm \
  --command "$PY212" /home/adnan-cdx/spyre-perf-suite/benchmark.py \
    --stack torch-spyre \
    --op small_swiglu \
    --op-file /tmp/torch-spyre-co-remap-native/tools/perf_suite_small_swiglu_op.py \
    --shape 1 256 128 512 \
    --runs 5 \
    --without-compilation \
    --with-profiling \
    --output '{run_dir}/perf.txt'
```

Batch `bmm` mode:

```bash
cd /tmp/torch-spyre-co-remap-native

$PY212 tools/run_coordinate_remap_bench.py \
  --output-root /tmp/small_swiglu_bmm_profile \
  --torch-root /tmp/torch-spyre-co-remap-native \
  --deeptools-root /tmp/deeptools-coordinate-remap-mainport-lean \
  --perf-suite-root /home/adnan-cdx/spyre-perf-suite \
  --runs 5 \
  --env LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH} \
  --env SPYRE_SMALL_SWIGLU_MODE=bmm \
  --command "$PY212" /home/adnan-cdx/spyre-perf-suite/benchmark.py \
    --stack torch-spyre \
    --op small_swiglu \
    --op-file /tmp/torch-spyre-co-remap-native/tools/perf_suite_small_swiglu_op.py \
    --shape 2 256 128 512 \
    --runs 5 \
    --without-compilation \
    --with-profiling \
    --output '{run_dir}/perf.txt'
```

Because `--command` consumes the rest of the wrapper arguments, keep it last.
The wrapper still uses its own `--runs` value to compute active iterations for
artifact summarization, so keep the wrapper and perf-suite run counts in sync.

## Reading metrics

Use `artifacts/trace_summary.json` as the primary source for kernel timing.
The main comparison number is trace-derived `kernel_ms_per_iter`.

Keep these separate in notes and tables:

- `kernel_ms_per_iter` from trace kernel events.
- Wall time from `perf.txt`.
- Memory transfer time.
- `Spyre-kernel_times` from profiler-reported counters.

Do not claim a speedup from wall time alone. Wall time can include compile,
setup, synchronization, or profiler overhead that is not kernel execution.

## Known warnings

- A 60 second `RuntimeStream::synchronize()` warning may still complete. Check
  whether the benchmark continues and whether artifacts were written before
  treating it as a hard failure.
- One-run perf-suite profiling can fail with `ZeroDivisionError` because no
  active profiling iterations exist. Use at least 3 runs, and prefer 5 for
  comparison runs.

## Cleanup and recording

Before leaving the pod or handing off results:

- Restore `/tmp/torch-spyre-co-remap-native/torch_spyre/_C.so` from the backup.
- Remove any `_C.so.pre_*_overlay_*` files.
- Leave `codegen_dumps/` and benchmark output directories uncommitted.
- Record the Torch SHA, Deeptools SHA, and run directory. The wrapper writes
  these to each variant's `env.json`; copy the relevant values into any result
  summary.
