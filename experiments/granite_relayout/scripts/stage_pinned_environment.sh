#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANTONI_NAMESPACE="${ANTONI_NAMESPACE:-a6-quantization}"
ANTONI_POD="${ANTONI_POD:-adnan-cdx-spyre-dev-pf}"
ANTONI_BASE="${ANTONI_BASE:-/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724}"

oc exec -n "${ANTONI_NAMESPACE}" "${ANTONI_POD}" -- \
  env ANTONI_BASE="${ANTONI_BASE}" bash -lc '
set -euo pipefail

ensure_checkout() {
  local url="$1"
  local revision="$2"
  local path="$3"
  if ! git -C "${path}" rev-parse --git-dir >/dev/null 2>&1; then
    git clone "${url}" "${path}"
  fi
  local actual
  actual="$(git -C "${path}" rev-parse HEAD)"
  if test "${actual}" != "${revision}"; then
    if test -n "$(git -C "${path}" status --porcelain)"; then
      printf "refusing to move dirty checkout %s\n" "${path}" >&2
      exit 1
    fi
    git -C "${path}" fetch origin "${revision}"
    git -C "${path}" checkout --detach "${revision}"
  fi
}

mkdir -p "${ANTONI_BASE}/run_historical_98ac91e"
ensure_checkout \
  git@github.ibm.com:aviros/test-spyre-scripts.git \
  afda166e58b23519d0b4ca871350b011b56d91a3 \
  "${ANTONI_BASE}/test-spyre-scripts"
mkdir -p "${ANTONI_BASE}/test-spyre-scripts/granite"
ensure_checkout \
  https://github.com/foundation-model-stack/foundation-model-stack.git \
  61bc991b175103e80cb8202b24a66ba7dbe79d1b \
  "${ANTONI_BASE}/test-spyre-scripts/granite/foundation-model-stack"
ensure_checkout \
  https://github.com/foundation-model-stack/aiu-fms-testing-utils.git \
  dbb1617525844651e7a2c5afcdec27fe163caa5f \
  "${ANTONI_BASE}/test-spyre-scripts/granite/aiu-fms-testing-utils"
ensure_checkout \
  https://github.com/torch-spyre/torch-spyre.git \
  98ac91e7823919e410b20dc2d0a1ee0ed6a620fa \
  "${ANTONI_BASE}/torch-spyre-98ac91e"
'

oc cp \
  "${REPO_ROOT}/implementation/antoni_inference_profile.py" \
  "${ANTONI_NAMESPACE}/${ANTONI_POD}:${ANTONI_BASE}/run_historical_98ac91e/antoni_inference_profile.py"
oc cp \
  "${REPO_ROOT}/implementation/torch_spyre_overlay/setup.py" \
  "${ANTONI_NAMESPACE}/${ANTONI_POD}:${ANTONI_BASE}/torch-spyre-98ac91e/setup.py"
oc cp \
  "${REPO_ROOT}/implementation/torch_spyre_overlay/execution/kernel_runner.py" \
  "${ANTONI_NAMESPACE}/${ANTONI_POD}:${ANTONI_BASE}/torch-spyre-98ac91e/torch_spyre/execution/kernel_runner.py"
oc cp \
  "${REPO_ROOT}/implementation/torch_spyre_overlay/csrc/profiler" \
  "${ANTONI_NAMESPACE}/${ANTONI_POD}:${ANTONI_BASE}/torch-spyre-98ac91e/torch_spyre/csrc"

oc exec -n "${ANTONI_NAMESPACE}" "${ANTONI_POD}" -- \
  env \
    ANTONI_BASE="${ANTONI_BASE}" \
    ANTONI_REBUILD_RUNTIME="${ANTONI_REBUILD_RUNTIME:-0}" \
  bash -lc '
set -euo pipefail

TORCH_SPYRE="${ANTONI_BASE}/torch-spyre-98ac91e"
if test ! -f "${TORCH_SPYRE}/torch_spyre/_C.so" || \
   test "${ANTONI_REBUILD_RUNTIME}" = 1; then
  cd "${TORCH_SPYRE}"
  export TORCH_DEVICE_BACKEND_AUTOLOAD=0
  export RUNTIME_INSTALL_DIR=/opt/ibm/spyre/runtime
  export SENLIB_INSTALL_DIR=/opt/ibm/spyre/senlib
  export DEEPTOOLS_INSTALL_DIR=/opt/ibm/spyre/deeptools
  export SPYRE_COMMS_INSTALL_DIR=/opt/ibm/spyre/spyre-comms
  export SEN_COMMON_HEADERS=/opt/ibm/spyre/runtime/include/flex
  export LIBAIUPTI_INSTALL_DIR=/opt/ibm/spyre/runtime
  export USE_SPYRE_PROFILER=1
  export MAX_JOBS="${MAX_JOBS:-8}"
  /tmp/adnan-cdx-costmodel-kineto/bin/python setup.py build_ext --inplace
fi
'

"${REPO_ROOT}/scripts/verify_pod_environment.sh"
