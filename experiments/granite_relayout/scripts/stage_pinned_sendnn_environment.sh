#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SENDNN_NAMESPACE="${SENDNN_NAMESPACE:-a6-quantization}"
SENDNN_POD="${SENDNN_POD:-adnan-spyre-dev-pf}"
SENDNN_BASE="${SENDNN_BASE:-/home/adnan/codex-isolated/sendnn_granite_antoni_20260725}"

SENDNN_SOURCE_REV=8cc4fe436f161f72e9bb4b76b8252d9bea981da6
PERF_SUITE_REV=cbde23adbe606775bf7fdad5c63e3ba32aa5e01d
FMS_REV=7a66f2f34ff95c2b9ad2e49b615e918f8aa85031
AIU_UTILS_REV=0325bd64662a8537803a2e3890294138ea17238a

oc exec -n "${SENDNN_NAMESPACE}" "${SENDNN_POD}" -- \
  env \
    SENDNN_BASE="${SENDNN_BASE}" \
    SENDNN_SOURCE_REV="${SENDNN_SOURCE_REV}" \
    PERF_SUITE_REV="${PERF_SUITE_REV}" \
    FMS_REV="${FMS_REV}" \
    AIU_UTILS_REV="${AIU_UTILS_REV}" \
  bash -lc '
set -euo pipefail

ensure_checkout() {
  local url="$1"
  local revision="$2"
  local path="$3"
  if ! git -C "${path}" rev-parse --git-dir >/dev/null 2>&1; then
    git clone "${url}" "${path}"
  fi
  if test -n "$(git -C "${path}" status --porcelain)"; then
    printf "refusing to move dirty checkout %s\n" "${path}" >&2
    exit 1
  fi
  if ! git -C "${path}" cat-file -e "${revision}^{commit}" 2>/dev/null; then
    git -C "${path}" fetch origin "${revision}"
  fi
  git -C "${path}" checkout --detach "${revision}"
}

mkdir -p "${SENDNN_BASE}/source"
ensure_checkout \
  https://github.ibm.com/ai-chip-toolchain/sendnn.git \
  "${SENDNN_SOURCE_REV}" \
  "${SENDNN_BASE}/source/sendnn"
ensure_checkout \
  https://github.ibm.com/ai-sw-acceleration/spyre-perf-suite.git \
  "${PERF_SUITE_REV}" \
  "${SENDNN_BASE}/spyre-perf-suite"
ensure_checkout \
  git@github.com:ppnaik1890/foundation-model-stack.git \
  "${FMS_REV}" \
  "${SENDNN_BASE}/foundation-model-stack"
ensure_checkout \
  https://github.com/foundation-model-stack/aiu-fms-testing-utils.git \
  "${AIU_UTILS_REV}" \
  "${SENDNN_BASE}/aiu-fms-testing-utils"

if test ! -x "${SENDNN_BASE}/venv/bin/python"; then
  /usr/bin/python3 -m venv --system-site-packages "${SENDNN_BASE}/venv"
fi
'

oc cp \
  "${REPO_ROOT}/results/2026-07-25/sendnn/requirements.freeze.txt" \
  "${SENDNN_NAMESPACE}/${SENDNN_POD}:${SENDNN_BASE}/requirements.freeze.txt"

oc exec -n "${SENDNN_NAMESPACE}" "${SENDNN_POD}" -- \
  env SENDNN_BASE="${SENDNN_BASE}" bash -lc '
set -euo pipefail
PYTHON="${SENDNN_BASE}/venv/bin/python"
"${PYTHON}" -m pip install -r "${SENDNN_BASE}/requirements.freeze.txt"
"${PYTHON}" -m pip install --no-deps -e "${SENDNN_BASE}/foundation-model-stack"
"${PYTHON}" -m pip install --no-deps -e "${SENDNN_BASE}/aiu-fms-testing-utils"
'

"${REPO_ROOT}/scripts/verify_sendnn_pod_environment.sh"
