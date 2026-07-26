#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANTONI_NAMESPACE="${ANTONI_NAMESPACE:-a6-quantization}"
ANTONI_POD="${ANTONI_POD:-adnan-cdx-spyre-dev-pf}"
ANTONI_BASE="${ANTONI_BASE:-/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724}"
ANTONI_RUN_TAG="${ANTONI_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ANTONI_RUN_ROOT="${ANTONI_RUN_ROOT:-${ANTONI_BASE}/runs/one_layer_${ANTONI_RUN_TAG}}"
ANTONI_ITERS="${ANTONI_ITERS:-20}"

"${REPO_ROOT}/scripts/verify_pod_environment.sh"

oc exec -n "${ANTONI_NAMESPACE}" "${ANTONI_POD}" -- \
  env \
    ANTONI_BASE="${ANTONI_BASE}" \
    ANTONI_RUN_ROOT="${ANTONI_RUN_ROOT}" \
    ANTONI_ITERS="${ANTONI_ITERS}" \
  bash -lc '
set -euo pipefail

SCRIPTS="${ANTONI_BASE}/test-spyre-scripts"
RUNNER="${ANTONI_BASE}/run_historical_98ac91e/antoni_inference_profile.py"
PYTHON=/tmp/adnan-cdx-costmodel-kineto/bin/python
MODEL=/tmp/models/granite-3.3-8b-instruct

mkdir -p \
  "${ANTONI_RUN_ROOT}/cache" \
  "${ANTONI_RUN_ROOT}/export" \
  "${ANTONI_RUN_ROOT}/trace_warm"
cd "${ANTONI_RUN_ROOT}"

export PYTHONPATH="${ANTONI_BASE}/torch-spyre-98ac91e:${SCRIPTS}/granite/foundation-model-stack:${SCRIPTS}/granite/aiu-fms-testing-utils"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${ANTONI_RUN_ROOT}/cache"
export DTCOMPILER_EXPORT_DIR="${ANTONI_RUN_ROOT}/export"
export DEEPRT_EXPORT_DIR="${ANTONI_RUN_ROOT}/export"
export ANTONI_PROFILE_DIR="${ANTONI_RUN_ROOT}/trace_warm"
export ANTONI_LAYER_LIMIT=1
export DUMP_SPYRE_CODE=1
unset TORCH_SPYRE_DOWNCAST_WARN

"${PYTHON}" "${RUNNER}" \
  --architecture hf_pretrained \
  --model_path "${MODEL}" \
  --tokenizer "${MODEL}" \
  --unfuse_weights \
  --batch_size 1 \
  --max_new_tokens 4 \
  --fixed_prompt_length 512 \
  --iters "${ANTONI_ITERS}" \
  --device_type spyre \
  --default_dtype fp16 \
  --timing per-token \
  --attention_type sdpa

TRACE="$(find "${ANTONI_RUN_ROOT}/trace_warm" -maxdepth 1 -name "*.pt.trace.json" -type f -print -quit)"
test -n "${TRACE}"
printf "run_root=%s\n" "${ANTONI_RUN_ROOT}"
printf "trace=%s\n" "${TRACE}"
sha256sum "${TRACE}"
'

if test -n "${ANTONI_LOCAL_TRACE_DIR:-}"; then
  mkdir -p "${ANTONI_LOCAL_TRACE_DIR}"
  TRACE_NAME="antoni_equivalent_one_layer_${ANTONI_RUN_TAG}.pt.trace.json"
  REMOTE_TRACE="$(
    oc exec -n "${ANTONI_NAMESPACE}" "${ANTONI_POD}" -- \
      find "${ANTONI_RUN_ROOT}/trace_warm" -maxdepth 1 \
      -name '*.pt.trace.json' -type f -print -quit
  )"
  test -n "${REMOTE_TRACE}"
  oc cp \
    "${ANTONI_NAMESPACE}/${ANTONI_POD}:${REMOTE_TRACE}" \
    "${ANTONI_LOCAL_TRACE_DIR}/${TRACE_NAME}"
  printf 'local_trace=%s\n' "${ANTONI_LOCAL_TRACE_DIR}/${TRACE_NAME}"
fi
