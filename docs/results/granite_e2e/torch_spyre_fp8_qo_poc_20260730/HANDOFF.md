# Torch-Spyre FP8 Q/O matmul PoC handoff

## Bottom line

The earlier SenDNN work on this branch established a measured DD2 standalone
baseline and showed that a compiler work-division change can take the Granite
Q/O scaled-FP8 pipeline from slower than FP16 to about `1.5-1.6x` faster at
large M. This continuation asks whether Torch-Spyre can reproduce the same
kind of result for:

```text
[M, 4096] @ [4096, 4096]
M = 1, 2, 4, ..., 2048
```

The answer is not yet. Torch-Spyre PR #2286 can execute a restricted FP8
matmul, but it does not implement the real scaled-matmul contract. The PoC
adds the DD2 minibatch-packed FP8 layout and obtains the desired 32-core
`M:8 x N:4 x K:1` outer work division for M=512. A narrow DeepTools change
also obtains the SenDNN-like M split across the two corelets. Compilation then
fails in a later compound-coordinate distribution step, so no optimized
Torch-Spyre timing should be reported.

Nothing here targets or depends on 1p5.

## Evidence boundary

### Measured and already accepted on this branch

The SenDNN DD2 measurements, raw result JSON, emitted artifacts, and analysis
are already committed here:

- [`../sendnn_scaled_fp8_vs_fp16_granite_linear_m_sweeps_20260729/`](../sendnn_scaled_fp8_vs_fp16_granite_linear_m_sweeps_20260729/)
- [`../sendnn_fp8_qo_weipreload_poc_20260729/`](../sendnn_fp8_qo_weipreload_poc_20260729/)

Those are SenDNN results. They are not Torch-Spyre results.

### Verified Torch-Spyre source and compile behavior

- The experimental source diff is pinned to PR #2286 head
  `a01c627d57ba18bc442d8b5f73086b2778fdc9d4`.
- The patch contains 16 modified files, 502 insertions, and 112 deletions.
- `git diff --check` and Python `compileall` pass.
- Stock QFP8CH M=512 compilation failures are packaged as SuperDSC JSON.
- The QFP8MB PoC reaches the intended outer work division.
- DeepTools initially chooses an output split across corelets.
- The narrow DeepTools PoC changes the corelet split to M and removes the
  original `dsc2.cpp:6379: TO DO: Loop split needed?` failure.
- Compilation then fails later at `dsc2.cpp:5862` while distributing the
  compound FP8 coordinates.

Syntax checks and deeper compiler progress are not device correctness or
performance validation.

### Archived restricted Torch-Spyre device results

The following M=512, K=N=4096 observations were retrieved from the isolated
device tree. Their result JSON files are in [`evidence/results/`](evidence/results/):

| Case | Archived observation | Interpretation |
|---|---:|---|
| FP16, 5 Kineto repetitions | 317.4604 us, 54.1166 TFLOP/s, relative L2 0.00452 | working reference |
| Stock FP8 forced to M:1 x N:32, 3 Kineto repetitions | 1392.735 us, 12.3353 TFLOP/s, relative L2 0.00248 | arithmetic path runs, but this is an inefficient fallback |

The FP8 observation includes activation and weight conversion, stops at the
raw FP16 matmul output, and does not apply output scales. It is not a
production scaled-FP8 measurement and not the optimized result requested by
the study. The result files contain the exact correctness, environment,
software, and timing metadata.

## What Torch-Spyre supports before this PoC

The PR #2286 snapshot has device-native building blocks:

```text
qfp8ch
qfp8wt
batchmatmulfp8
FMA8
```

Its `_scaled_mm` lowering is explicitly a placeholder. It accepts
prequantized E4M3 inputs and reaches an FP8 batch matmul, but it does not
implement the supplied activation scale, weight scale, bias, result scale, or
`use_fast_accum` semantics. Therefore it proves that a restricted standalone
FP8 arithmetic path exists; it does not provide the production FMS/SenDNN
scaled-matmul contract.

## What the Torch-Spyre PoC changes

The patch in
[`patches/torch_spyre_qfp8mb_poc.patch`](patches/torch_spyre_qfp8mb_poc.patch)
adds experimental DD2 support for:

- `QFP8MB` and the `qfp8mb` activation conversion;
- `batchmatmulfp8mb`;
- activation sticks physically arranged as `[K:8, M:2, K:8]`;
- weight sticks physically arranged as `[K:2, N:64]`;
- FP16 output sticks arranged as `[N:64]`;
- serialization of compound physical-stick coordinates;
- the even-M, two-dimensional, K/N-multiple-of-64 legality rules used by this
  path;
- a prohibition on splitting the K reduction; and
- FP8-aware work-division legality plus an experimental work-division hint.

For M=512, K=N=4096, the optimized outer division is:

```text
M split:       8
N split:       4
K split:       1
cores:         32
per-core work: M=64, N=1024, K=4096
```

This is a research patch, not a production-shaped implementation.

## Why the corelet choice was wrong

There are two levels of parallelism:

1. The outer work division assigns tiles to the 32 AIU cores.
2. Each core has two compute-side corelets that can divide that core's tile.

The direct Torch-Spyre SuperDSC path leaves the within-core split unspecified.
DeepTools then scans dimensions and greedily chooses the first split that
appears legal. For this compound FP8 layout, two problems were found:

- Input-stick legality was tested against unrelated candidate dimensions. A
  stick factor should constrain a split only when that factor belongs to the
  dimension being split.
- The direct path does not pass through the richer SenDNN/DSM policy that
  avoids splitting the stationary weight/output dimension for a large primary
  matmul and instead divides independent M rows.

At M=512, the bad choice divided each core's N=1024 output tile into
`N=512+512`. That crosses the compound `[K:2,N:64]` weight-coordinate series
and enters an unsupported loop-splitting path. The known-good SenDNN program
divides each core's M=64 rows into `M=32+32`: both corelets reuse the same
resident weight tile and produce different output rows.

The DeepTools experiment in
[`patches/deeptools_dd2_fp8mb_corelet_poc.patch`](patches/deeptools_dd2_fp8mb_corelet_poc.patch)
makes stick legality dimension-aware and narrowly prefers M for `QFP8MB` and
the matching FP8 matmul. With that change:

```text
qfp8mb corelet split:          M=8+8
batchmatmulfp8mb corelet split: M=32+32
```

The original output-loop-split failure disappears. The later
`dsc2.cpp:5862` failure proves that the corelet decision was one real blocker,
but not the last one. FP8 is not inherently restricted to one corelet.

## Precise planner gap

The claim is not that Torch-Spyre has no matmul planner. In this PoC the FP8
operation reaches the existing tuned matmul planner. The problem is that the
planner's model is calibrated for FP16:

- it assumes two-byte inputs;
- it uses the DL16 peak MAC rate;
- it assumes FP16 stick/feed behavior; and
- its tile and fanout coefficients were calibrated on FP16.

Applying that scorer to FP8 makes its ranking untrustworthy even when a
candidate is legal. The correct production order remains:

1. implement and validate the full scaled-operation semantics;
2. model FP8 conversion, matmul, output scaling, layout, and transport costs;
3. make the work-division scorer precision-aware; and
4. validate outside the Granite shapes.

## Benchmark contract

The prepared benchmark is
[`../../../../benchmarks/torch_spyre_fp8_matmul/bench_qo_fp8_poc.py`](../../../../benchmarks/torch_spyre_fp8_matmul/bench_qo_fp8_poc.py).
It collects CPU-reference error and Kineto `cat=="kernel"` time. Effective
throughput uses `2*M*K*N` as the numerator.

With `--prepack-weight`, static weight conversion is performed in a separate
compiled graph and excluded from the timed graph. The scaled FP8 variants
time:

```text
FP16 activation
  -> supplied-scale FP8 conversion and packing
  -> FP8 matmul
  -> explicit row-scale multiplication
  -> explicit column-scale multiplication
  -> FP16 output
```

This includes conversion and output rescaling overhead. It excludes dynamic
scale derivation, one-time weight preparation, compilation, and transfers.
The scales are unit-valued because the placeholder PR lowering does not
validate non-unit scale semantics.

Raw variants stop after the FP8 matmul's FP16 output. They are diagnostic only.

## Current blocker

The next blocker is not the outer work division or the selected corelet
dimension. With the DeepTools PoC preloaded, compilation advances to
`dsc2.cpp:5862` and fails while propagating/distributing the compound
QFP8MB/QFP8WT coordinate folds.

The next experiment should change one coordinate rule at a time and rerun the
M=512 raw case. Do not begin the full M sweep or Granite end-to-end run until
that single case compiles and passes correctness.

## Exact continuation

### 1. Verify access

OpenShift access was working when this package was finalized. On continuation,
first check identity and the target pod; reauthenticate only if `oc whoami`
itself fails:

```bash
oc whoami
oc get pod -n a6-quantization adnan-clc-spyre-dev-pf \
  -o name --request-timeout=8s
```

### 2. Recreate the Torch-Spyre source

Apply the archived patch only to its exact PR #2286 base:

```bash
git checkout --detach a01c627d57ba18bc442d8b5f73086b2778fdc9d4
git apply /path/to/patches/torch_spyre_qfp8mb_poc.patch
git diff --check
python3.12 -m compileall -q \
  torch_spyre tests/inductor/test_core_mapping.py
```

Do not apply it directly to this results branch's older source tree.

### 3. Recreate the DeepTools diagnostic

Apply the narrow patch to:

```text
ee2f97a86c609eeb20ea3ad2d48040259d67ded3
```

The old DeepTools snapshot did not complete a clean full rebuild against the
newer local LLVM source because of unrelated source drift. The investigation
therefore compiled the changed `SdscCoreletSplit.cpp` object and linked a
diagnostic preload library with unresolved symbols left for the running
process. Treat this as a diagnostic technique, not a distributable build.

Existing remote paths:

```text
source: /home/adnan/codex-isolated/torch_spyre_fp8_deeptools_ee2f97a_20260729
build:  /home/adnan/codex-isolated/torch_spyre_fp8_deeptools_ee2f97a_build_20260729
preload:/home/adnan/codex-isolated/torch_spyre_fp8_deeptools_ee2f97a_build_20260729/libdeeptools_fp8mb_corelet_poc.so
```

### 4. Reproduce the current M=512 boundary

On `adnan-clc-spyre-dev-pf`:

```bash
source /home/adnan/spyre-envs/image-913f394b4b3f/activate.sh
ROOT=/home/adnan/codex-isolated/torch_spyre_pr2286_a01c627d_20260729
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH}"
export SENCORES=32
export SENCORELETS=2
export LD_PRELOAD=/home/adnan/codex-isolated/torch_spyre_fp8_deeptools_ee2f97a_build_20260729/libdeeptools_fp8mb_corelet_poc.so
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache_qfp8mb_raw_opt_m512_8x4_resume"

python benchmarks/torch_spyre_fp8_matmul/bench_qo_fp8_poc.py \
  --variant fp8_raw_optimized \
  --m 512 --k 4096 --n 4096 \
  --m-split 8 --n-split 4 \
  --warmups 1 --reps 2 \
  --output-dir "$ROOT/results/qfp8mb_raw_opt_m512_8x4_resume"
```

The expected current result is the `dsc2.cpp:5862` compile failure. If it
passes, inspect emitted artifacts before timing.

### 5. Run the sweep only after M=512 passes

```bash
STUDY_ROOT="$ROOT/results/qo_m_sweep" \
FP8_CORELET_PRELOAD="$LD_PRELOAD" \
bash benchmarks/torch_spyre_fp8_matmul/run_qo_fp8_poc_sweep.sh
```

The runner serializes cases on one device. Different M subsets can be assigned
to different pods by setting `FP8_M_VALUES`, but never run two benchmarks
concurrently on one pod.

## Acceptance gates

1. M=512 compiles and passes the CPU-reference gate.
2. Emitted operations contain `qfp8mb`, `batchmatmulfp8mb`, and `FMA8`.
3. Physical sticks are activation `[K:8,M:2,K:8]`, weight `[K:2,N:64]`,
   and output `[N:64]`.
4. Outer matmul work division is `M:8 x N:4 x K:1` over 32 cores.
5. Final matmul corelet split is `M=32+32`, not `N=512+512`.
6. Static weight preparation is outside the timed graph.
7. Raw and scaled FP8 timing are reported separately.
8. A non-unit scale test validates real numerical semantics before this is
   called a scaled-matmul implementation.
9. The complete `M=1...2048` sweep records FP16, baseline FP8, optimized FP8,
   correctness, emitted artifacts, and full environment provenance.
10. Production work implements scale, bias, result-scale, and
    `use_fast_accum` semantics and replaces the FP16-calibrated cost model.
11. Granite end-to-end work starts only after the standalone matmul gates pass.

## Non-claims

- No accepted optimized Torch-Spyre FP8 timing exists yet.
- No Torch-Spyre M sweep has completed.
- No Torch-Spyre Granite end-to-end FP8 run has completed.
- The team-reported `1.5x` raw and `1.25x` model figures are not new
  measurements from this PoC.
- The DeepTools preload is an experiment, not a production fix.
