#!/usr/bin/env bash
set -euo pipefail

# Serialized DD2-only runner for the experimental torch-spyre Q/O FP8 path.
# Split FP8_M_VALUES across pods if desired, but never launch two cases on one
# device pod at the same time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
STUDY_ROOT="${STUDY_ROOT:-$REPO_ROOT/torch_spyre_fp8_qo_poc_runs}"
FP8_M_VALUES="${FP8_M_VALUES:-1 2 4 8 16 32 64 128 256 512 1024 2048}"
FP8_WARMUPS="${FP8_WARMUPS:-5}"
FP8_REPS="${FP8_REPS:-20}"
FP8_K="${FP8_K:-4096}"
FP8_N="${FP8_N:-4096}"
FP8_CORELET_PRELOAD="${FP8_CORELET_PRELOAD:-}"

stack_text="${SENARCH:-} ${SENTARGET:-} ${DEEPTOOLS_PATH:-} ${LD_LIBRARY_PATH:-} ${FP8_CORELET_PRELOAD}"
shopt -s nocasematch
if [[ "$stack_text" == *1p5* ]]; then
  echo "Refusing to run: this PoC is DD2-only and the environment contains 1p5." >&2
  exit 2
fi
shopt -u nocasematch

mkdir -p "$STUDY_ROOT"
status_file="$STUDY_ROOT/status.tsv"
if [[ ! -f "$status_file" ]]; then
  printf 'M\tvariant\tstatus\tresult\n' >"$status_file"
fi

if [[ -n "$FP8_CORELET_PRELOAD" ]]; then
  if [[ ! -f "$FP8_CORELET_PRELOAD" ]]; then
    echo "FP8_CORELET_PRELOAD does not exist: $FP8_CORELET_PRELOAD" >&2
    exit 2
  fi
  export LD_PRELOAD="$FP8_CORELET_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
fi

export SENCORES="${SENCORES:-32}"
export SENCORELETS="${SENCORELETS:-2}"
unset TORCH_SPYRE_DOWNCAST_WARN

run_case() {
  local m="$1"
  local variant="$2"
  local case_root="$STUDY_ROOT/m${m}_k${FP8_K}_n${FP8_N}/$variant"
  local cache_root="$case_root/torchinductor_cache"
  local -a extra_args=()

  if [[ "$variant" == fp8_* ]]; then
    extra_args+=(--prepack-weight)
  fi

  mkdir -p "$case_root"
  export TORCHINDUCTOR_CACHE_DIR="$cache_root"

  if python "$SCRIPT_DIR/bench_qo_fp8_poc.py" \
      --variant "$variant" \
      --m "$m" \
      --k "$FP8_K" \
      --n "$FP8_N" \
      --warmups "$FP8_WARMUPS" \
      --reps "$FP8_REPS" \
      --output-dir "$case_root/profile" \
      "${extra_args[@]}" \
      >"$case_root/run.log" 2>&1; then
    printf '%s\t%s\tPASS\t%s\n' \
      "$m" "$variant" "$case_root/profile/result.json" | tee -a "$status_file"
  else
    printf '%s\t%s\tFAIL\t%s\n' \
      "$m" "$variant" "$case_root/run.log" | tee -a "$status_file"
    return 1
  fi
}

for m in $FP8_M_VALUES; do
  run_case "$m" fp16
  run_case "$m" fp8_baseline
  run_case "$m" fp8_optimized
done
