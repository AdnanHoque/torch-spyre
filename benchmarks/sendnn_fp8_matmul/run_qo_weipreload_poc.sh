#!/usr/bin/env bash
set -uo pipefail

STUDY_ROOT=/home/adnan/codex-isolated/fp8_sendnn_linear_sweeps_20260729
POC_ROOT=/home/adnan/codex-isolated/fp8_qo_planner_hack_20260729
PRODUCTION_ROOT=/home/adnan/codex-isolated/sendnn_granite_antoni_20260725
BENCHMARK_SCRIPT="${STUDY_ROOT}/direct_sendnn_fp16_fp8_pair_benchmark_per_axis.py"
WRAPPER_SCRIPT="${STUDY_ROOT}/direct_sendnn_kineto_wrapper.py"
PYTHON_BIN="${PRODUCTION_ROOT}/venv/bin/python"
RUN_ROOT="${RUN_ROOT:-${POC_ROOT}/runs/qo_weipreload_poc_$(date +%Y%m%d_%H%M%S)}"
WARMUPS="${WARMUPS:-5}"
REPETITIONS="${REPETITIONS:-20}"

export PATH="/opt/ibm/spyre/runtime/bin:/opt/ibm/spyre/deeptools/bin:/opt/ibm/spyre/senlib/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="${PRODUCTION_ROOT}/foundation-model-stack:${PRODUCTION_ROOT}/aiu-fms-testing-utils:/opt/ibm/spyre/runtime/lib"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/spyre-comms/lib"
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
export DTCOMPILER_KEEP_EXPORT=true

unset LD_PRELOAD
unset DXP_LX_FRAC_AVAIL LX_PLANNING PERFDSC_DEBUG PERFDSC_DUMP_DIR
unset SENTARGET SENARCH SENCORES SENCORELETS DATA_PREC SENSEARCH

mkdir -p "${RUN_ROOT}"
printf '%s\n' "${RUN_ROOT}" > "${POC_ROOT}/latest_qo_weipreload_poc_root.txt"

{
  date -Ins
  uname -n
  printf 'run_root=%s\n' "${RUN_ROOT}"
  printf 'logical_shape=[M,4096]@[4096,4096]\n'
  printf 'm_values=1,2,4,8,16,32,64,128,256,512,1024,2048\n'
  printf 'variants=fp16,fp8_baseline,fp8_weipreload0\n'
  printf 'fp8_baseline_dt_opt=autopilot=1\n'
  printf 'fp8_weipreload0_dt_opt=autopilot=1,weipreload=0\n'
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
print("torch_sendnn", importlib.metadata.version("torch_sendnn"), torch_sendnn.__file__)
print("sendnn", sendnn.__file__)
PY
} > "${RUN_ROOT}/provenance.txt" 2>&1

printf 'M\tvariant\tmode\tdt_opt\texit_status\tresult\n' > "${RUN_ROOT}/status.tsv"
M_VALUES=(1 2 4 8 16 32 64 128 256 512 1024 2048)
failure_count=0

for index in "${!M_VALUES[@]}"; do
  m="${M_VALUES[$index]}"
  case $((index % 3)) in
    0) variants=(fp16 fp8_baseline fp8_weipreload0) ;;
    1) variants=(fp8_weipreload0 fp16 fp8_baseline) ;;
    2) variants=(fp8_baseline fp8_weipreload0 fp16) ;;
  esac

  shape_root="${RUN_ROOT}/m${m}_k4096_n4096"
  mkdir -p "${shape_root}"

  for variant in "${variants[@]}"; do
    case "${variant}" in
      fp16)
        mode=fp16
        export DT_OPT=autopilot=1
        ;;
      fp8_baseline)
        mode=fp8
        export DT_OPT=autopilot=1
        ;;
      fp8_weipreload0)
        mode=fp8
        export DT_OPT=autopilot=1,weipreload=0
        ;;
      *)
        printf 'unknown variant: %s\n' "${variant}" >&2
        exit 2
        ;;
    esac

    variant_root="${shape_root}/${variant}"
    mkdir -p "${variant_root}"
    export DTCOMPILER_EXPORT_DIR="${variant_root}/export"
    export DEEPRT_EXPORT_DIR="${variant_root}/export"
    printf 'starting M=%s variant=%s mode=%s DT_OPT=%s at %s\n' \
      "${m}" "${variant}" "${mode}" "${DT_OPT}" "$(date -Ins)"

    FP8_BENCH_M="${m}" \
    FP8_BENCH_K=4096 \
    FP8_BENCH_N=4096 \
    "${PYTHON_BIN}" "${WRAPPER_SCRIPT}" \
      --mode "${mode}" \
      --benchmark-script "${BENCHMARK_SCRIPT}" \
      --run-dir "${variant_root}/profile" \
      --warmups "${WARMUPS}" \
      --repetitions "${REPETITIONS}" \
      > "${variant_root}/run.log" 2>&1
    run_status=$?

    if (( run_status == 0 )); then
      result="${variant_root}/profile/result.json"
    else
      result="${variant_root}/run.log"
      failure_count=$((failure_count + 1))
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${m}" "${variant}" "${mode}" "${DT_OPT}" "${run_status}" "${result}" \
      >> "${RUN_ROOT}/status.tsv"
    printf 'finished M=%s variant=%s status=%s at %s\n' \
      "${m}" "${variant}" "${run_status}" "$(date -Ins)"
  done
done

printf 'failure_count=%s\n' "${failure_count}" >> "${RUN_ROOT}/provenance.txt"
printf '%s\n' "${RUN_ROOT}"
exit "${failure_count}"
