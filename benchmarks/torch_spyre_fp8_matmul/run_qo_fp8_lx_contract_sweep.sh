#!/usr/bin/env bash
set -euo pipefail

# Serialized DD2 Q/O contract sweep. Split FP8_M_VALUES across device pods;
# never run two cases concurrently on one pod.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STUDY_ROOT="${STUDY_ROOT:?set STUDY_ROOT to an isolated output directory}"
FP8_M_VALUES="${FP8_M_VALUES:-512 1024 2048}"
FP8_WARMUPS="${FP8_WARMUPS:-5}"
FP8_REPS="${FP8_REPS:-20}"
FP8_K="${FP8_K:-4096}"
FP8_N="${FP8_N:-4096}"
FP8_DERIVE_ACTIVATION_SCALE="${FP8_DERIVE_ACTIVATION_SCALE:-0}"
FP8_FUSED_ACTIVATION_SCALE="${FP8_FUSED_ACTIVATION_SCALE:-0}"
FP8_LX_RELAYOUT_MIN_SOURCE_BYTES="${FP8_LX_RELAYOUT_MIN_SOURCE_BYTES:-65536}"
FP8_LX_RELAYOUT_MAX_SOURCE_BYTES="${FP8_LX_RELAYOUT_MAX_SOURCE_BYTES:-4194304}"

stack_text="${SENARCH:-} ${SENTARGET:-} ${DEEPTOOLS_PATH:-} ${LD_LIBRARY_PATH:-}"
shopt -s nocasematch
if [[ "$stack_text" == *1p5* ]]; then
  echo "Refusing to run: this experiment is DD2-only." >&2
  exit 2
fi
shopt -u nocasematch

mkdir -p "$STUDY_ROOT"
status_file="$STUDY_ROOT/status.tsv"
if [[ ! -f "$status_file" ]]; then
  printf 'M\tlabel\tstatus\tresult\n' >"$status_file"
fi

run_case() {
  local m="$1"
  local label="$2"
  local variant="$3"
  local case_root="$STUDY_ROOT/m${m}_k${FP8_K}_n${FP8_N}/$label"
  local -a extra_args=()

  mkdir -p "$case_root"
  export TORCHINDUCTOR_CACHE_DIR="$case_root/cache"

  unset SPYRE_LX_PLANNER_RELAYOUT
  unset TORCH_SPYRE_FP8_LX_POC_M_SPLIT
  unset TORCH_SPYRE_FP8_LX_POC_N_SPLIT
  unset TORCH_SPYRE_LX_RELAYOUT_MIN_SOURCE_BYTES
  unset TORCH_SPYRE_LX_RELAYOUT_MAX_SOURCE_BYTES
  unset SPYRE_CORE_ID_K_FAST_EMISSION

  if [[ "$variant" == fp8_* ]]; then
    extra_args+=(--prepack-weight --activation-packing minibatch)
    if [[ "$FP8_DERIVE_ACTIVATION_SCALE" == 1 ]]; then
      extra_args+=(--derive-activation-scale)
    fi
  fi

  if [[ "$label" == optimized_fp8 ]]; then
    export SPYRE_LX_PLANNER_RELAYOUT=1
    export TORCH_SPYRE_FP8_LX_POC_M_SPLIT=8
    export TORCH_SPYRE_FP8_LX_POC_N_SPLIT=4
    export TORCH_SPYRE_LX_RELAYOUT_MIN_SOURCE_BYTES="$FP8_LX_RELAYOUT_MIN_SOURCE_BYTES"
    export TORCH_SPYRE_LX_RELAYOUT_MAX_SOURCE_BYTES="$FP8_LX_RELAYOUT_MAX_SOURCE_BYTES"
    export SPYRE_CORE_ID_K_FAST_EMISSION=0
    extra_args+=(--m-split 8 --n-split 4)
    if [[ "$FP8_DERIVE_ACTIVATION_SCALE" == 1 && "$FP8_FUSED_ACTIVATION_SCALE" == 1 ]]; then
      extra_args+=(--fused-activation-scale)
    fi
  fi

  if "$PYTHON_BIN" "$SCRIPT_DIR/bench_qo_fp8_poc.py" \
      --variant "$variant" \
      --m "$m" --k "$FP8_K" --n "$FP8_N" \
      --warmups "$FP8_WARMUPS" --reps "$FP8_REPS" \
      --output-dir "$case_root/result" \
      "${extra_args[@]}" >"$case_root/run.log" 2>&1; then
    printf '%s\t%s\tPASS\t%s\n' \
      "$m" "$label" "$case_root/result/result.json" | tee -a "$status_file"
  else
    printf '%s\t%s\tFAIL\t%s\n' \
      "$m" "$label" "$case_root/run.log" | tee -a "$status_file"
    return 1
  fi
}

for m in $FP8_M_VALUES; do
  run_case "$m" fp16 fp16
  run_case "$m" baseline_fp8 fp8_baseline
  run_case "$m" optimized_fp8 fp8_optimized
done
