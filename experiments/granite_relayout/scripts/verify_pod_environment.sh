#!/usr/bin/env bash
set -euo pipefail

ANTONI_NAMESPACE="${ANTONI_NAMESPACE:-a6-quantization}"
ANTONI_POD="${ANTONI_POD:-adnan-cdx-spyre-dev-pf}"
ANTONI_BASE="${ANTONI_BASE:-/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724}"

oc exec -n "${ANTONI_NAMESPACE}" "${ANTONI_POD}" -- \
  env ANTONI_BASE="${ANTONI_BASE}" bash -lc '
set -euo pipefail

PYTHON=/tmp/adnan-cdx-costmodel-kineto/bin/python
MODEL=/tmp/models/granite-3.3-8b-instruct
SCRIPTS="${ANTONI_BASE}/test-spyre-scripts"
FMS="${SCRIPTS}/granite/foundation-model-stack"
UTILS="${SCRIPTS}/granite/aiu-fms-testing-utils"
TORCH_SPYRE="${ANTONI_BASE}/torch-spyre-98ac91e"
RUNNER="${ANTONI_BASE}/run_historical_98ac91e/antoni_inference_profile.py"

check_revision() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(git -C "${path}" rev-parse HEAD)"
  test "${actual}" = "${expected}" || {
    printf "revision mismatch: %s\n  expected %s\n  actual   %s\n" \
      "${path}" "${expected}" "${actual}" >&2
    return 1
  }
}

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | cut -d " " -f 1)"
  test "${actual}" = "${expected}" || {
    printf "sha256 mismatch: %s\n  expected %s\n  actual   %s\n" \
      "${path}" "${expected}" "${actual}" >&2
    return 1
  }
}

test -x "${PYTHON}"
test -d "${MODEL}"
check_revision "${SCRIPTS}" afda166e58b23519d0b4ca871350b011b56d91a3
check_revision "${FMS}" 61bc991b175103e80cb8202b24a66ba7dbe79d1b
check_revision "${UTILS}" dbb1617525844651e7a2c5afcdec27fe163caa5f
check_revision "${TORCH_SPYRE}" 98ac91e7823919e410b20dc2d0a1ee0ed6a620fa
check_sha256 "${RUNNER}" 12191848dcb0c39f2b44e92c21d9a4dc41ae7b37f7dc021c53baa94742cbb366
check_sha256 "${TORCH_SPYRE}/torch_spyre/execution/kernel_runner.py" 1a5dc76ed73b75d649d7b6f035b584d5ffb02daff6129a7f4865c5941ed3e6ed
check_sha256 "${TORCH_SPYRE}/torch_spyre/_C.so" 1f4d328fbce73cb2e96b819437e4aebb7b760b6f8cbe361782a8ce1526f22db1
check_sha256 "${MODEL}/config.json" 1313edf0d39dcf7ed35a072d341ec11b516c12acc4267cfb6d248c6bdcdddcb7
check_sha256 "${MODEL}/model.safetensors.index.json" c3a88218300666c35343b129857d4e8583ee1b15bf68d90ab976f51744560379

TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="${TORCH_SPYRE}" \
  "${PYTHON}" - <<"PY"
import torch

expected = "2.11.0+aiu.kineto.1.1.2"
if torch.__version__ != expected:
    raise SystemExit(f"torch mismatch: expected {expected}, got {torch.__version__}")
print(f"torch={torch.__version__}")
PY

printf "verified_namespace=%s\n" "${ANTONI_NAMESPACE:-a6-quantization}"
printf "verified_pod=%s\n" "${HOSTNAME}"
printf "verified_base=%s\n" "${ANTONI_BASE}"
'
