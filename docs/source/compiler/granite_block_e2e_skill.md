# Granite Block E2E Skill

Use this runbook when you need to run a one-layer FMS Granite block end to end
on the `adnan-cdx-spyre-dev-pf` AIU pod with empty Spyre-resident weights.  It
is intended for compiler, SDSC, and coordinate-remap investigations where
parameter values are irrelevant.

## What This Proves

The known-good path compiles and executes a real FMS `GraniteBlock` prefill
shape:

- input: `[1, 512, 4096]`
- attention: `sdpa_causal`
- weights: fused, fp16, materialized with empty Spyre tensors
- output: `[1, 512, 4096]`
- returned KV cache: two tensors shaped `[1, 8, 512, 128]`

This is a wall-sync e2e smoke/probe.  It is not a replacement for Kineto
trace-derived `kernel_ms` benchmarking.

For kernel-time measurement, the same probe supports `--profile`.  Use one
warmup outside the profiler and at least five active iterations:

```bash
$PY212 benchmarks/granite_block_layer_probe.py \
  --fms-root "$FMS" \
  --run-root "$RUN" \
  --case prefill \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 5 \
  --warmups 1 \
  --profile \
  --no-profile-memory
```

Profiled runs write:

- `block_prefill/result.json`, with the compact trace summary embedded.
- `block_prefill/trace_summary.json`, the primary `kernel_ms_per_iter` source.
- `block_prefill/trace/*.pt.trace.json`, the raw Kineto trace.

The June 21, 2026 archived profile artifacts live at
`docs/source/compiler/lx_coordinate_remap_benchmarks/2026-06-20/granite_block_layer_profile/`.

## Required Stack

Use the Torch checkout and Deeptools build used for coordinate-remap work:

```bash
cd /tmp/torch-spyre-co-remap-native

export PY212=/home/adnan-cdx/dt-inductor-codex-clean/.venv-py212/bin/python
source /home/adnan-cdx/dt-inductor-codex-clean/env.sh
source /home/adnan-cdx/dt-inductor-codex-clean/matmul_gap_env.sh
use_py212_localflex_optdeeptools_spyre_runtime

DEE=/tmp/deeptools-coordinate-remap-mainport-lean
FMS=/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/decode_regression_rev_ab_20260610_163300/foundation-model-stack-eager_spyre

export PYTHONPATH=/tmp/torch-spyre-co-remap-native:/tmp/torch-spyre-co-remap-native/tests/inductor:$FMS:${PYTHONPATH:-}
export PATH="$DEE/build-swiglu-dxp-main-lean/dxp:$DEE/build/dxp:${PATH}"
export LD_LIBRARY_PATH=/home/adnan-cdx/dt-inductor-codex-clean/install/libaiupti/lib:/home/adnan-cdx/dt-inductor-codex-clean/install/runtime-localdt/lib:${LD_LIBRARY_PATH:-}
```

Do not prepend Deeptools build library directories to `LD_LIBRARY_PATH` for this
probe.  That can make `torch_spyre/_C.so` fail to load with Flex symbol
mismatches.

## Profiler Overlay

Temporarily overlay the profiler-enabled `_C.so` from the clean checkout:

```bash
backup=./torch_spyre/_C.so.pre_granite_block_$(date +%Y%m%d_%H%M%S)
cp ./torch_spyre/_C.so "$backup"
cp /home/adnan-cdx/dt-inductor-codex-clean/torch-spyre/torch_spyre/_C.so ./torch_spyre/_C.so
```

Restore it after the run:

```bash
cp "$backup" ./torch_spyre/_C.so
```

## Baseline Prefill

Use the branch-owned probe.  It avoids global `torch.manual_seed`, constructs
one FMS Granite layer, materializes empty Spyre parameters, compiles the block,
and runs one warmup followed by one measured wall-sync iteration.

```bash
RUN=/tmp/granite_block_layer_probe_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"

$PY212 benchmarks/granite_block_layer_probe.py \
  --fms-root "$FMS" \
  --run-root "$RUN" \
  --case prefill \
  --compile-block \
  --attn-name sdpa_causal \
  --iters 1 \
  --warmups 1
```

Expected success criteria:

- `RESULT_JSON=$RUN/block_prefill/result.json`
- `returncode` is `0`
- `error` is `null`
- `output_shape` is `[1, 512, 4096]`
- `cache_shape` is `[[1, 8, 512, 128], [1, 8, 512, 128]]`

Verified examples from June 21, 2026:

- `sdpa_causal`: `23.835897 ms` wall-sync measured iteration.

## Coordinate-Remap Variant

Enable the pass with the normal PR-1 flags:

```bash
export SPYRE_ONCHIP_MOVE_PLANNER=1
export SPYRE_ONCHIP_MOVE_REALIZE=1
export SPYRE_ONCHIP_MOVE_CARRIER=coordinate_remap
export SPYRE_ONCHIP_MOVE_COORDINATE_REMAP_CHUNK_CELLS=8192
export SPYRE_ONCHIP_MOVE_RANGE_ENCODING=1
export SPYRE_ONCHIP_MOVE_MAX_CELLS=131072
export SPYRE_ONCHIP_MOVE_JSONL="$RUN/onchip_move.jsonl"
export SPYRE_ONCHIP_MOVE_DEBUG_DIR="$RUN/onchip_move_debug"
```

Then run the same causal prefill probe command.  A June 21, 2026 profiled
causal run completed with `returncode=0`, a `21.236420 ms` wall-sync median,
and `13.819416 ms` trace-derived kernel time per iteration.  Its SDSC inventory
contained `OnChipMoveCoordinateRemap` rows in both attention and MLP kernels,
and `onchip_move.jsonl` reported:

- `10` planned coordinate-remap edges
- `44` skipped edges
- `87,556,096` planned bytes
- skipped reasons dominated by `same-per-core-view-owned-by-lx-planner`

The coordinate-remap full-block wall-sync probe is not currently a speedup
claim.  Use Kineto trace-derived `kernel_ms` before publishing performance.

## Failure Modes

If the probe fails with `_get_default_generator`, make sure you are using
`benchmarks/granite_block_layer_probe.py`.  Older golden probes call
`torch.manual_seed(0)`, which can hit missing PrivateUse1 generator hooks in
the profiler overlay.

If the probe fails with:

```text
Mismatch between index & stride dimensions: 4 vs 6
```

make sure the branch includes the split-multi trailing unflattening helper in
`torch_spyre/_inductor/split_multi_ops.py`.  The Granite RoPE path indexes a
consumer as `[B, S, H, 128]` while reading an intermediate view shaped
`[B, S, H, 2, 1, 64]`.

If the first call prints a `RuntimeStream::synchronize() still waiting after
60000ms` warning but the final `result.json` has `returncode=0`, treat the run
as passed.  That warning has appeared during the compile/startup call.
