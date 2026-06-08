# pr-mlp-fix: Shared-Weight Unit-BMM Prefill Fix

## Status

Branch:

```text
pr-mlp-fix
base: upstream/main 8ee5682
HEAD: 824ad4b Bypass attention fake tensor setup for Spyre
```

Commits:

```text
824ad4b Bypass attention fake tensor setup for Spyre
94b3053 Improve shared-weight unit BMM planning
c2151bc Fix shared-weight unit BMM layout
```

Pure tests:

```text
python -m py_compile torch_spyre/_inductor/spyre_kernel.py \
  torch_spyre/_inductor/work_division.py \
  torch_spyre/_inductor/lowering.py \
  torch_spyre/_inductor/temp_passes.py \
  torch_spyre/_inductor/patches.py

python -m pytest tests/inductor/test_temp_passes.py \
  tests/inductor/test_coarse_tiling.py \
  tests/inductor/test_work_division.py -q

121 passed
```

Headline result:

```text
matmul [[1, 512, 4096], [4096, 12800]]
main:        3.749 ms kernel, 29.794% PT
pr-mlp-fix:  1.023 ms kernel, 72.799% PT
speedup:     3.66x kernel
```

IBM-facing `spyre-perf-suite/run_benchmark.py` reproduction on the PR:

```text
spyre-perf-suite: 7450624
torch-spyre:      824ad4b
kernel_ms:        1.016
spyre_ms:         3.165
PT:               73.297%
runroot:          /tmp/spyre-pr-mlp-fix-official-copy-20260606-141334
```

## Problem

The real branch target is the wide shared-weight prefill projection:

```text
activation: [1, 512, 4096]
weight:     [4096, 12800]
output:     [1, 512, 12800]
```

This is a standard transformer-style shared-weight projection. The weight is 2D
and reused across the token/prefill dimension. It is not the same as the
batched benchmark MLP form where the weight is 3D and each batch slice has a
distinct weight.

The bad prefill projection symptom on upstream main was low array utilization:

```text
main: 3.749 ms, 29.794% PT
```

The fixed branch uses the same mathematical operation, but preserves the
intended shared-weight unit-BMM representation and chooses a better work split:

```text
PR: 1.023 ms, 72.799% PT
```

## Jamie's temp_passes.py Clue

Jamie pointed at the BMM/unflatten path and the fact that torch-spyre was
flattening a dim-size-1 axis differently from sendnn:

```text
sendnn:      3D @ 2D
torch-spyre: 2D @ 2D
```

That was the useful clue. The issue was not "MLP" in the abstract. It was the
compiler representation for a shared-weight unit-BMM: logically there is still
a unit BMM dimension, even though the weight is shared and rank-2.

The correct SDSC primary layout for this path is:

```text
INPUT:  [mb, in, x]
KERNEL: [in, out]
OUTPUT: [mb, out, x]
```

If that logical unit-BMM axis is flattened away too early, codegen can still
emit a correct matmul, but it loses the layout/scheduling shape that gives good
Spyre utilization on the wide prefill projection.

## Root Cause

The fast prefill projection requires two things at once:

1. Preserve the shared-weight unit-BMM layout contract.
2. Pick a work division that exposes both token-side and wide-output-side work.

The important final split is:

```text
_spyre_bmm_unit = 1
mb              = 4
out             = 8
in              = 1
```

Interpretation:

```text
_spyre_bmm_unit=1  keeps the logical unit BMM dimension.
mb=4               splits the prefill/token dimension.
out=8              splits the wide MLP projection dimension.
in=1               avoids splitting the reduction dimension.
```

Earlier attempts that preserved only part of this were insufficient. Preserving
layout without the right split left PT utilization low. Picking the split
without preserving the shared-weight unit-BMM layout risked leaving the intended
SDSC path.

## Implementation

The PR has three parts.

### 1. Mark shared-weight unit BMMs

Files:

```text
torch_spyre/_inductor/constants.py
torch_spyre/_inductor/temp_passes.py
torch_spyre/_inductor/lowering.py
```

The branch adds metadata for the "static leading batch dim is 1" BMM case:

```text
_spyre_shared_weight_unit_bmm
shared_weight_unit_bmm
```

`temp_passes.py` tags the FX node when `_unflatten_mm_to_bmm` recognizes the
unit-BMM form. `lowering.py` carries that metadata into the `SpyreReduction`
`op_info` so later codegen can see it.

### 2. Preserve and infer the layout in codegen

File:

```text
torch_spyre/_inductor/spyre_kernel.py
```

The branch preserves the logical unit-BMM dimension in the iteration space and
uses the shared-weight layout:

```text
INPUT:  [mb, in, x]
KERNEL: [in, out]
OUTPUT: [mb, out, x]
```

It also infers the same `shared_weight_unit_bmm` info from TensorAccess sizes
when FX metadata is not available at codegen. That mattered because some
standalone focused projection runs reached codegen as a fused linear/reduction
with empty `op_info`, even though the logical size pattern was still
`[1, M, K] @ [K, N]`.

### 3. Improve the work-division tie-break

File:

```text
torch_spyre/_inductor/work_division.py
```

The cost model previously preferred a split that left the wide projection
underfilled. The branch adjusts the tie-break so M-side splits are penalized
only when the per-core M tile underfills PT. That allows the good candidate:

```text
mb=4, out=8, in=1
```

instead of the weaker wide-projection split.

### 4. Make the IBM-facing suite compile this path

File:

```text
torch_spyre/_inductor/patches.py
```

The ready `spyre-perf-suite` child process can enter PyTorch's attention/fake
tensor setup in a no-dispatch context when compiling Spyre graphs. Jamie's
suite had already worked around this. The PR now scopes the same behavior to
Spyre's `enable_spyre_context`:

```text
if input_device.type == "spyre":
    return None
```

The original `_sfdp_init` function is restored in `finally`, so the patch is
active only while the Spyre compile context is active.

This compatibility commit is separate from the performance fix. It exists so
IBM reviewers can use the ready `spyre-perf-suite/run_benchmark.py` path.

## Results

Shape-aware paired comparison:

```text
remote runroot: /tmp/spyre-main-vs-pr-mlp-fix-final-20260606-133103
```

| Case | Main kernel_ms | PR kernel_ms | Speedup | Main spyre_ms | PR spyre_ms | Main PT% | PR PT% | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matmul_prefill_kv | 0.127 | 0.091 | 1.40x | 0.331 | 0.336 | 46.800 | 65.306 | ok/ok |
| matmul_prefill_qo | 0.559 | 0.323 | 1.73x | 1.105 | 0.916 | 42.656 | 73.833 | ok/ok |
| matmul_prefill_mlp_proj | 3.749 | 1.023 | 3.66x | 5.790 | 3.159 | 29.794 | 72.799 | ok/ok |
| matmul_decode_kv | 0.058 | 0.058 | 1.00x | 0.231 | 0.274 | 0.803 | 0.806 | ok/ok |
| matmul_decode_qo | 0.215 | 0.216 | 1.00x | 0.717 | 0.648 | 0.864 | 0.863 | ok/ok |
| matmul_decode_mlp_proj | 0.718 | 0.722 | 0.99x | 2.587 | 2.635 | 0.810 | 0.806 | ok/ok |
| mlp_prefill |  |  |  |  |  |  |  | failed/failed |
| mlp_decode | 24.600 | 24.807 | 0.99x | 46.601 | 46.883 | 0.080 | 0.079 | ok/ok |

Main read:

```text
The PR is a prefill projection win. Decode matmuls are effectively unchanged.
```

## Full Prefill MLP Failure: Not Caused By This PR

The full `mlp --shape 1 512 4096` benchmark fails with default LX planning on
both upstream main and `pr-mlp-fix`.

Default config:

```text
torch_spyre/_inductor/config.py
lx_planning = os.environ.get("LX_PLANNING", "1") == "1"
allow_all_ops_in_lx_planning = False
```

Both branches fail with the same verifier signature:

```text
DtException: Program verification failed for core 6 node 3_exp
Register initialization out of boundary:
lxsu0 : LRF0 : 2457472
```

That makes it baseline-shared and consistent with the known Charlie/team
LX/DXP verifier bug. It is not evidence that `pr-mlp-fix` introduced the
failure.

The isolation run with `LX_PLANNING=0` confirms this:

```text
remote runroot: /tmp/spyre-main-vs-pr-mlp-fix-lxoff-20260606-135936
```

| Case | Main kernel_ms | PR kernel_ms | Speedup | Main spyre_ms | PR spyre_ms | Main PT% | PR PT% | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mlp_prefill, LX_PLANNING=0 | 23.389 | 13.421 | 1.74x | 28.717 | 18.683 | 11.291 | 22.456 | ok/ok |

Conclusion:

```text
Default full prefill MLP failure: LX/DXP verifier path, baseline-shared.
PR performance effect: still visible when LX planning is disabled.
```

## Relationship To Claude's Conclusion

Claude's decode-MLP artifact correction is right and should be kept separate
from this PR's claim.

The benchmark decode MLP path uses 3D batched weights and lowers to BMM. That
is not the same thing as a standard transformer MLP with shared 2D weights. A
real shared-weight decode MLP is already at peak on torch-spyre, so the PR
should not be sold as solving a decode standard-MLP gap.

What survives as real:

```text
Decode standard shared-weight MLP: no gap to fix here.
Decode batched/MoE BMM: real BMM kernel issue, separate from this PR.
Prefill shared-weight projection: real gap, fixed by this PR.
Full prefill MLP default failure: known LX/DXP verifier issue, not PR-caused.
```

This is why the PR claim should be narrow:

```text
Fix shared-weight unit-BMM layout/planning for wide prefill projection.
```

## Reproducing With Ready spyre-perf-suite

Ready suite checked on the pod:

```text
/home/adnan-cdx/dt-inductor-codex-clean/spyre-perf-suite
commit: 7450624
```

Important suite details:

1. The suite's `benchmark.py` sets `TORCH_DEVICE_BACKEND_AUTOLOAD` to `0` only
   if it is unset. Set it to `1` before launching.
2. `run_benchmark.py` invokes child `benchmark.py` with an output path under
   `perf/...` relative to the current suite directory. Use `--perf-dir "$PWD/perf"`
   or run from a temporary suite copy. Passing an unrelated absolute `--perf-dir`
   causes the benchmark to run but the report parser to miss the perf file.
3. Use `--runs 2` or more. With `--runs 1`, profiling discards the first run and
   the suite can divide by zero while summarizing active runs.

Minimal PR verification command. This keeps both the suite `perf/` directory
and the torch-spyre source tree temporary/clean, while using the ready suite
source unchanged:

```bash
ROOT=/home/adnan-cdx/dt-inductor-codex-clean
OFFICIAL_SUITE=$ROOT/spyre-perf-suite
BRANCH=/home/adnan-cdx/codex-worktrees/pr-mlp-fix/torch-spyre
RUNROOT=/tmp/pr-mlp-fix-official-check
SUITE=$RUNROOT/spyre-perf-suite
SRC=$RUNROOT/torch-spyre-pr-mlp-fix

rm -rf "$RUNROOT"
mkdir -p "$RUNROOT"
cp -a "$OFFICIAL_SUITE/." "$SUITE/"
cp -a "$BRANCH/." "$SRC/"
rm -f "$SRC/torch_spyre/_C.so" "$SRC/torch_spyre/_hooks.so"
cp "$ROOT/torch-spyre/torch_spyre/_C.so" "$SRC/torch_spyre/_C.so"
cp "$ROOT/torch-spyre/torch_spyre/_hooks.so" "$SRC/torch_spyre/_hooks.so"

source "$ROOT/env.sh"
source "$ROOT/matmul_gap_env.sh"
use_py212_localflex_optdeeptools_spyre_runtime >/dev/null

cd "$SUITE"
rm -rf "$PWD/perf"
mkdir -p "$PWD/perf" "$RUNROOT/export"

export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TORCHINDUCTOR_CACHE_DIR="$RUNROOT/cache"
export DTCOMPILER_EXPORT_DIR="$RUNROOT/export"
export DEEPRT_EXPORT_DIR="$RUNROOT/export"

PYTHONPATH="$SRC:$ROOT/foundation-model-stack:$ROOT/aiu-fms-testing-utils:$AIU_MONITOR_LIB:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/senlib/lib" \
PS_TORCH_SPYRE_PATH="$SRC" \
PS_SPYRE_PERF_SUITE_PATH="$SUITE" \
python run_benchmark.py \
  --op matmul \
  --shape 1 512 4096 \
  --shape 4096 12800 \
  --stacks torch-spyre \
  --runs 2 \
  --perf-dir "$PWD/perf" \
  --report "$RUNROOT/report.txt" \
  --kernel_report "$RUNROOT/kernel_report.txt"
```

Observed PR result from the ready suite:

```text
Op: matmul  Shape: [[1, 512, 4096], [4096, 12800]]
wall_clock_ms.mean_ms          60.834
spyre_ms.mean_ms                3.165
kernel_ms.mean_ms               1.016
memory_transfer_ms.mean_ms      2.149
pt_util%                       73.297
```

## Recommended PR Framing

Use this wording:

```text
This PR preserves shared-weight unit-BMM layout and improves work division for
wide prefill projection matmuls. It improves the key
[[1,512,4096],[4096,12800]] projection from 3.749 ms to 1.023 ms kernel time
in paired A/B testing, and reproduces at 1.016 ms using ready
spyre-perf-suite/run_benchmark.py.
```

Avoid this wording:

```text
This PR fixes all MLP gaps.
This PR fixes decode MLP.
This PR fixes the default full prefill MLP verifier failure.
```

Those are separate issues.
