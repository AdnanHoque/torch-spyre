#!/usr/bin/env bash
set -euo pipefail

SENDNN_NAMESPACE="${SENDNN_NAMESPACE:-a6-quantization}"
SENDNN_POD="${SENDNN_POD:-adnan-spyre-dev-pf}"
SENDNN_BASE="${SENDNN_BASE:-/home/adnan/codex-isolated/sendnn_granite_antoni_20260725}"
SENDNN_MODEL="${SENDNN_MODEL:-/home/adnan/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/51dd4bc2ade4059a6bd87649d68aa11e4fb2529b}"

oc exec -n "${SENDNN_NAMESPACE}" "${SENDNN_POD}" -- \
  env SENDNN_BASE="${SENDNN_BASE}" SENDNN_MODEL="${SENDNN_MODEL}" bash -lc '
set -euo pipefail

check_revision() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(git -C "${path}" rev-parse HEAD)"
  test "${actual}" = "${expected}" || {
    printf "revision mismatch: %s expected=%s actual=%s\n" \
      "${path}" "${expected}" "${actual}" >&2
    exit 1
  }
  test -z "$(git -C "${path}" status --porcelain)" || {
    printf "dirty checkout: %s\n" "${path}" >&2
    exit 1
  }
}

check_hash() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "${path}" | awk "{print \$1}")"
  test "${actual}" = "${expected}" || {
    printf "hash mismatch: %s expected=%s actual=%s\n" \
      "${path}" "${expected}" "${actual}" >&2
    exit 1
  }
}

check_revision "${SENDNN_BASE}/source/sendnn" \
  8cc4fe436f161f72e9bb4b76b8252d9bea981da6
check_revision "${SENDNN_BASE}/spyre-perf-suite" \
  cbde23adbe606775bf7fdad5c63e3ba32aa5e01d
check_revision "${SENDNN_BASE}/foundation-model-stack" \
  7a66f2f34ff95c2b9ad2e49b615e918f8aa85031
check_revision "${SENDNN_BASE}/aiu-fms-testing-utils" \
  0325bd64662a8537803a2e3890294138ea17238a

check_hash "${SENDNN_MODEL}/config.json" \
  1313edf0d39dcf7ed35a072d341ec11b516c12acc4267cfb6d248c6bdcdddcb7
check_hash "${SENDNN_MODEL}/model.safetensors.index.json" \
  c3a88218300666c35343b129857d4e8583ee1b15bf68d90ab976f51744560379
check_hash "${SENDNN_MODEL}/tokenizer.json" \
  91168e938f05796aa6dcca7e485e4b30ab52785320c7a6391ecef86e6c84681e

check_hash /opt/ibm/spyre/runtime/bin/compile_graph \
  805f83f452c4f12c42a28b6d6fa8c1a573102a22607bd74b62f1e44df655ebbf
check_hash /opt/ibm/spyre/runtime/lib/libsendnn.so \
  496da8a8f666a0f8f52501cd8535bd85a59c531f9292fdf831004ad582ba5eac
check_hash /opt/ibm/spyre/runtime/lib/sendnn.cpython-312-x86_64-linux-gnu.so \
  95f7348c99801d35ee92ffbdf7166439b689666ef1d657a56e1020d32a85ca4f
check_hash /usr/local/lib/python3.12/site-packages/torch_sendnn/torch_sendnn.py \
  b235eec921e26c40bb462c3a150567a1f4b6c99ee25dc9a272f181e76fe74658

test "$(rpm -q ibm-flex)" = \
  ibm-flex-2.0.0-0.main.1+388.81385a4_0.el10.x86_64
test -n "${AIU_WORLD_RANK_0:-}"

export PATH=/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/senlib/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="${SENDNN_BASE}/foundation-model-stack:${SENDNN_BASE}/aiu-fms-testing-utils:/opt/ibm/spyre/runtime/lib"
export LD_LIBRARY_PATH=/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/spyre-comms/lib
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
test "$(command -v compile_graph)" = /opt/ibm/spyre/runtime/bin/compile_graph

"${SENDNN_BASE}/venv/bin/python" - <<"PY"
import importlib.metadata as metadata
import os
import fms
import sendnn
import torch
import torch_sendnn

assert torch.__version__ == "2.10.0+aiu.kineto.1.1.1", torch.__version__
assert metadata.version("torch-sendnn") == "1.3.0+main.1.1bef083.0"
assert metadata.version("aiu-fms-testing-utils") == "0.9.0"
assert metadata.version("transformers") == "5.14.1"
expected_fms = os.path.join(os.environ["SENDNN_BASE"], "foundation-model-stack") + "/"
assert fms.__file__.startswith(expected_fms), (fms.__file__, expected_fms)
assert sendnn.__file__ == "/opt/ibm/spyre/runtime/lib/sendnn.cpython-312-x86_64-linux-gnu.so"
print("SenDNN pod environment verified")
PY

if ps -eo comm,args | awk \
  "\$1 ~ /^python/ && \$0 ~ /inference_granite_os[.]py/ { found = 1 } END { exit found ? 0 : 1 }"; then
  printf "another Granite SenDNN run is active on %s\n" "${HOSTNAME}" >&2
  exit 1
fi
'
