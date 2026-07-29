#!/usr/bin/env bash
set -uo pipefail

# Standalone SenDNN device sweep for one Granite 3 8B linear projection:
#   [M, K] @ [K, N], M in powers of two from 1 through 2048.
#
# Required:
#   SHAPE_LABEL  Stable filesystem label, for example qo or mlp_up.
#   K            Reduction dimension.
#   N            Output dimension.
#
# The FP8 case is the complete fixed-scale operation emitted by SenDNN:
#   Qfp8 -> FP8 BatchMatMul -> activation-scale recovery
#         -> weight-scale recovery
#
# Scale inputs are one FP32 value per activation row and per output channel.
# Their values are fixed at one; upstream scale derivation is out of scope.

: "${SHAPE_LABEL:?set SHAPE_LABEL, for example qo or mlp_up}"
: "${K:?set K}"
: "${N:?set N}"

if [[ ! "${SHAPE_LABEL}" =~ ^[a-z0-9_]+$ ]]; then
  printf 'invalid SHAPE_LABEL: %s\n' "${SHAPE_LABEL}" >&2
  exit 2
fi
if [[ ! "${K}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${N}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'K and N must be positive integers: K=%s N=%s\n' "${K}" "${N}" >&2
  exit 2
fi

STUDY_ROOT="${STUDY_ROOT:-/home/adnan/codex-isolated/fp8_sendnn_linear_sweeps_20260729}"
PRODUCTION_ROOT="${PRODUCTION_ROOT:-/home/adnan/codex-isolated/sendnn_granite_antoni_20260725}"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-${STUDY_ROOT}/direct_sendnn_fp16_fp8_pair_benchmark_per_axis.py}"
WRAPPER_SCRIPT="${WRAPPER_SCRIPT:-${STUDY_ROOT}/direct_sendnn_kineto_wrapper.py}"
PYTHON_BIN="${PYTHON_BIN:-${PRODUCTION_ROOT}/venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STUDY_ROOT}/runs/granite_${SHAPE_LABEL}_m_sweep_$(date +%Y%m%d_%H%M%S)}"
WARMUPS="${WARMUPS:-5}"
REPETITIONS="${REPETITIONS:-20}"

export PATH="/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/senlib/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${PRODUCTION_ROOT}/foundation-model-stack:${PRODUCTION_ROOT}/aiu-fms-testing-utils:/opt/ibm/spyre/runtime/lib"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/spyre-comms/lib"
export RUNTIME_INSTALL_DIR="/opt/ibm/spyre/runtime"
export RUNTIME_FULL_INSTALL_DIR="/opt/ibm/spyre/runtime"
export SENDNN_DIR="/opt/ibm/spyre/runtime"
export SENDNN_INSTALL_DIR="/opt/ibm/spyre/runtime"
export DEEPTOOLS_INSTALL_DIR="/opt/ibm/spyre/deeptools"
export DEEPTOOLS_PATH="/opt/ibm/spyre/deeptools/share"
export SENLIB_INSTALL_DIR="/opt/ibm/spyre/senlib"
export SPYRE_COMMS_INSTALL_DIR="/opt/ibm/spyre/spyre-comms"
export SEN_COMMON_HEADERS="/opt/ibm/spyre/runtime/include"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export DT_OPT=autopilot=1
export DTCOMPILER_KEEP_EXPORT=true

# Preserve the validated production environment. In particular, do not inherit
# experimental LX planning or architecture-selection overrides.
unset DXP_LX_FRAC_AVAIL LX_PLANNING PERFDSC_DEBUG PERFDSC_DUMP_DIR
unset SENTARGET SENARCH SENCORES SENCORELETS DATA_PREC

mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" \
  > "${STUDY_ROOT}/latest_granite_${SHAPE_LABEL}_m_sweep_root.txt"

{
  date -Ins
  uname -n
  printf 'run_root=%s\n' "${RUN_ROOT}"
  printf 'shape_label=%s\n' "${SHAPE_LABEL}"
  printf 'logical_shape=[M,%s]@[%s,%s]\n' "${K}" "${K}" "${N}"
  printf 'm_values=1,2,4,8,16,32,64,128,256,512,1024,2048\n'
  printf 'warmups=%s\n' "${WARMUPS}"
  printf 'repetitions=%s\n' "${REPETITIONS}"
  printf 'benchmark_sha256='
  sha256sum "${BENCHMARK_SCRIPT}"
  printf 'wrapper_sha256='
  sha256sum "${WRAPPER_SCRIPT}"
  printf 'runner_sha256='
  sha256sum "$0"
  rpm -qa | grep -E 'deeptools|flex|senlib|spyre-runtime' | sort || true
  "${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import sendnn
import torch
import torch_sendnn

print("python", __import__("sys").version.replace("\n", " "))
print("torch", torch.__version__, torch.__file__)
print(
    "torch_sendnn",
    importlib.metadata.version("torch_sendnn"),
    torch_sendnn.__file__,
)
print("sendnn", sendnn.__file__)
PY
  env | grep -E '^(PATH|PYTHONPATH|LD_LIBRARY_PATH|RUNTIME_|SENDNN_|DEEPTOOLS_|SENLIB_|SPYRE_COMMS_|SEN_COMMON_HEADERS|TORCH_DEVICE_BACKEND_AUTOLOAD|DT_OPT|DTCOMPILER_KEEP_EXPORT)=' | sort
} > "${RUN_ROOT}/provenance.txt" 2>&1

printf 'M\tmode\texit_status\tresult\n' > "${RUN_ROOT}/status.tsv"
M_VALUES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
failure_count=0

for index in "${!M_VALUES[@]}"; do
  m="${M_VALUES[$index]}"
  if (( index % 2 == 0 )); then
    modes=(fp16 fp8)
  else
    modes=(fp8 fp16)
  fi

  shape_root="${RUN_ROOT}/m${m}_k${K}_n${N}"
  mkdir -p "${shape_root}"

  for mode in "${modes[@]}"; do
    mode_root="${shape_root}/${mode}"
    mkdir -p "${mode_root}"
    export DTCOMPILER_EXPORT_DIR="${mode_root}/export"
    export DEEPRT_EXPORT_DIR="${mode_root}/export"
    printf 'starting shape=%s M=%s K=%s N=%s mode=%s at %s\n' \
      "${SHAPE_LABEL}" "${m}" "${K}" "${N}" "${mode}" "$(date -Ins)"

    FP8_BENCH_M="${m}" \
    FP8_BENCH_K="${K}" \
    FP8_BENCH_N="${N}" \
    "${PYTHON_BIN}" "${WRAPPER_SCRIPT}" \
      --mode "${mode}" \
      --benchmark-script "${BENCHMARK_SCRIPT}" \
      --run-dir "${mode_root}/profile" \
      --warmups "${WARMUPS}" \
      --repetitions "${REPETITIONS}" \
      > "${mode_root}/run.log" 2>&1
    status=$?

    if (( status == 0 )); then
      result="${mode_root}/profile/result.json"
    else
      result="${mode_root}/run.log"
      failure_count=$((failure_count + 1))
    fi
    printf '%s\t%s\t%s\t%s\n' "${m}" "${mode}" "${status}" "${result}" \
      >> "${RUN_ROOT}/status.tsv"
    printf 'finished shape=%s M=%s K=%s N=%s mode=%s status=%s at %s\n' \
      "${SHAPE_LABEL}" "${m}" "${K}" "${N}" "${mode}" "${status}" \
      "$(date -Ins)"
  done
done

printf 'failure_count=%s\n' "${failure_count}" >> "${RUN_ROOT}/provenance.txt"
printf '%s\n' "${RUN_ROOT}"
exit "${failure_count}"
