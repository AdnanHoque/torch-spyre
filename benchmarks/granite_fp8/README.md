# Private Granite FP8 one-layer integration bridge

This directory contains a narrow DD2-only bridge from FMS-MO's real
`FP8Linear` checkpoint contract to Torch-Spyre's FP8 operations. It is an
integration experiment, not a production frontend.

For each Spyre `FP8Linear` invocation, the bridge:

1. flattens `[B,M,K]` to `[B*M,K]` before activation quantization;
2. computes the TorchAO 0.11 dynamic per-row FP32 scale
   `max(abs(row))/448`, clamped to FP32 epsilon;
3. calls `spyre.quantize_fp8_with_scale`, selecting QFP8MB for even prefill M;
4. reuses the checkpoint's E4M3 `[N,K]` weight and `[N,1]` static scale;
5. calls `aten._scaled_mm` with real scales, optional checkpoint bias,
   FP16 output, and `use_fast_accum=True`; and
6. restores the original leading activation dimensions.

The override is installed only when both
`TORCH_SPYRE_ENABLE_FMS_MO_FP8_BRIDGE=1` and
`TORCH_SPYRE_FP8_TARGET=dd2` are set. Non-Spyre inputs continue through the
original FMS-MO implementation. The launcher also rejects environment values
containing `1p5` or `1.5`.

## Exact dependency assumptions

- Torch `2.11.0+aiu.kineto.1.1.2`;
- Torch-Spyre branch containing the complete DD2 scaled-matmul contract;
- FMS `61bc991b175103e80cb8202b24a66ba7dbe79d1b`;
- FMS-MO `0418c190642acbe6530f93df30e45f31e5d8dd9a`
  (`fms-model-optimizer==0.8.5`);
- TorchAO `f34b473e56b2e406d5a0ce5a0cea8453aaf87cb3`
  (`torchao==0.11.0`);
- DeepTools `+1401` (`ee2f97a`);
- Flex `+388` (`81385a4`); and
- `/tmp/models/granite-3.3-8b-instruct-FP8`, whose compressed-tensors config
  specifies dynamic token activation scales and static channel weight scales.

FMS-MO v0.8.3 must not be used unchanged: its package metadata excludes Torch
2.11. Current FMS-MO main has the compatible Torch bound and the same FP8Linear
implementation used by this bridge.

## Prefill smoke

The canonical runner is kept outside this repository. On the cdx pod:

```bash
BASE=/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/\
latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724

ANTONI_RUNNER="$BASE/run_historical_98ac91e/antoni_inference_profile.py" \
FMS_ROOT="$BASE/test-spyre-scripts/granite/foundation-model-stack" \
AIU_FMS_UTILS_ROOT="$BASE/test-spyre-scripts/granite/aiu-fms-testing-utils" \
FP8_DEPS_ROOT=/tmp/codex-fp8-integration-deps-audit-20260731 \
STUDY_ROOT=/tmp/torch-spyre-granite-fp8-one-layer \
bash benchmarks/granite_fp8/run_granite_fp8_one_layer.sh
```

Set `FP8_BRIDGE_CHECK_ONLY=1` to validate imports and exact dependency versions
without loading Granite or executing on the device.

The smoke is deliberately prefill-only: one layer, batch 1, M=512, one output
token. It omits `--default_dtype fp16`, because the checkpoint already carries
FP8 weights, and uses `--cast_bf16_to_fp16` only for the remaining BF16 tensors.
It does not run the 40-layer model or claim end-to-end performance.

Acceptance requires emitted evidence of QFP8MB, FP8 matmul, both real scale
applications, and no `spyre.to_dtype_cpu` activation conversion. Decode M=1
and QFP8CH remain a separate validation gate.
