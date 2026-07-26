#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SENDNN_NAMESPACE="${SENDNN_NAMESPACE:-a6-quantization}"
SENDNN_POD="${SENDNN_POD:-adnan-spyre-dev-pf}"
SENDNN_BASE="${SENDNN_BASE:-/home/adnan/codex-isolated/sendnn_granite_antoni_20260725}"
SENDNN_MODEL="${SENDNN_MODEL:-/home/adnan/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b}"
SENDNN_ITERS="${SENDNN_ITERS:-20}"
SENDNN_RUN_TAG="${SENDNN_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
SENDNN_RUN_ROOT="${SENDNN_RUN_ROOT:-${SENDNN_BASE}/runs/one_layer_b1_s512_20x4_${SENDNN_RUN_TAG}}"

"${REPO_ROOT}/scripts/verify_sendnn_pod_environment.sh"

oc exec -n "${SENDNN_NAMESPACE}" "${SENDNN_POD}" -- \
  env \
    SENDNN_BASE="${SENDNN_BASE}" \
    SENDNN_MODEL="${SENDNN_MODEL}" \
    SENDNN_ITERS="${SENDNN_ITERS}" \
    SENDNN_RUN_ROOT="${SENDNN_RUN_ROOT}" \
  bash -lc '
set -euo pipefail
PYTHON="${SENDNN_BASE}/venv/bin/python"
RUNNER="${SENDNN_BASE}/spyre-perf-suite/utils/inference_granite_os.py"

test ! -e "${SENDNN_RUN_ROOT}"
mkdir -p "${SENDNN_RUN_ROOT}/export" "${SENDNN_RUN_ROOT}/logs/granite"
cd "${SENDNN_RUN_ROOT}"

export PATH=/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/senlib/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="${SENDNN_BASE}/foundation-model-stack:${SENDNN_BASE}/aiu-fms-testing-utils:/opt/ibm/spyre/runtime/lib"
export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/spyre-comms/lib
export RUNTIME_INSTALL_DIR=/opt/ibm/spyre/runtime
export RUNTIME_FULL_INSTALL_DIR=/opt/ibm/spyre/runtime
export SENDNN_DIR=/opt/ibm/spyre/runtime
export SENDNN_INSTALL_DIR=/opt/ibm/spyre/runtime
export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
export DEEPTOOLS_PATH=/opt/ibm/spyre/deeptools/share
export SENLIB_INSTALL_DIR=/opt/ibm/spyre/senlib
export SPYRE_COMMS_INSTALL_DIR=/opt/ibm/spyre/spyre-comms
export SEN_COMMON_HEADERS=/opt/ibm/spyre/runtime/include
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export HF_HOME=/home/adnan
export PYTHONUNBUFFERED=1
export DT_DEEPRT_VERBOSE=0
export DT_OPT=autopilot=1
export DTCOMPILER_EXPORT_DIR="${SENDNN_RUN_ROOT}/export"
export DEEPRT_EXPORT_DIR="${SENDNN_RUN_ROOT}/export"
export DTCOMPILER_KEEP_EXPORT=true
export _PROMPT_LEN=512
export _MAX_DECODE_TOKENS=4
export _MAX_CONTEXT_LEN=515

printf "run_root=%s\n" "${SENDNN_RUN_ROOT}"
printf "compile_graph=%s\n" "$(command -v compile_graph)"

"${PYTHON}" "${RUNNER}" \
  --architecture hf_pretrained \
  --model_path "${SENDNN_MODEL}" \
  --tokenizer "${SENDNN_MODEL}" \
  --unfuse_weights \
  --fixed_prompt_length 512 \
  --prefill_only \
  --compile \
  --batch_size 1 \
  --compile_dynamic \
  --compile_dynamic_sendnn \
  --iters "${SENDNN_ITERS}" \
  --device_type aiu \
  --default_dtype fp16 \
  --timing=per-token \
  --with_profiling \
  --run_block 1 \
  --max_new_tokens 4 2>&1 | tee "${SENDNN_RUN_ROOT}/run.log"

TRACE="$(find "${SENDNN_RUN_ROOT}/logs/granite" -maxdepth 1 -name "*.pt.trace.json" -type f -print -quit)"
test -n "${TRACE}"
printf "trace=%s\n" "${TRACE}"
sha256sum "${TRACE}" "${SENDNN_RUN_ROOT}/run.log" \
  "${SENDNN_RUN_ROOT}/logs/granite/old_stack_compiler.log"
'

if test -n "${SENDNN_LOCAL_ARTIFACT_DIR:-}"; then
  mkdir -p "${SENDNN_LOCAL_ARTIFACT_DIR}"
  oc cp \
    "${SENDNN_NAMESPACE}/${SENDNN_POD}:${SENDNN_RUN_ROOT}" \
    "${SENDNN_LOCAL_ARTIFACT_DIR}/${SENDNN_RUN_TAG}"
fi
