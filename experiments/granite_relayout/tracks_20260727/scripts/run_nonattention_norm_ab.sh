#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 5 ]]; then
  echo "usage: $0 <run-name> <planner:0|1> [disabled-sources] [iters] [layers|full]" >&2
  exit 2
fi

ROOT=/home/adnan/codex-isolated/device_parity_pr2939_20260725
RUN="$ROOT/runs/$1"
PLANNER="$2"
DISABLED_SOURCES="${3:-buf11,buf15,buf19,buf66,buf29,buf30,buf43,buf44,buf45,buf53,buf55,buf56,buf57}"
ITERS="${4:-1}"
LAYERS="${5:-1}"
MLP_NORM_ORACLE="${SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_NORM:-1}"
MLP_INPUTS_ORACLE="${SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS:-0}"
MLP_DOWN_PROJ_ORACLE="${SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ:-0}"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUNNER="$ROOT/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct

test "$PLANNER" = 0 -o "$PLANNER" = 1
test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export" "$RUN/logits" "$RUN/trace"
cd "$RUN"

source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$ROOT/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/torch-spyre:$ROOT/foundation-model-stack:$ROOT/aiu-fms-testing-utils"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export DTCOMPILER_EXPORT_DIR="$RUN/export"
export DEEPRT_EXPORT_DIR="$RUN/export"
export ANTONI_PROFILE_DIR="$RUN/trace"
export ANTONI_LOGIT_DUMP_DIR="$RUN/logits"
if [[ "$LAYERS" = full ]]; then
  unset ANTONI_LAYER_LIMIT
else
  export ANTONI_LAYER_LIMIT="$LAYERS"
fi
export DUMP_SPYRE_CODE=1

export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS="$MLP_INPUTS_ORACLE"
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ="$MLP_DOWN_PROJ_ORACLE"
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_NORM="$MLP_NORM_ORACLE"
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0
export SPYRE_LX_PLANNER_RELAYOUT="$PLANNER"
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_to_all,all_gather,broadcast
export SPYRE_LX_RELAYOUT_ALL_TO_ALL_MAX_BYTES=-1
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES="$DISABLED_SOURCES"
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"
unset TORCH_SPYRE_DOWNCAST_WARN

"$PYTHON" "$RUNNER" \
  --architecture hf_pretrained \
  --model_path "$MODEL" \
  --tokenizer "$MODEL" \
  --unfuse_weights \
  --batch_size 1 \
  --max_new_tokens 1 \
  --fixed_prompt_length 512 \
  --iters "$ITERS" \
  --device_type spyre \
  --default_dtype fp16 \
  --timing per-token \
  --attention_type sdpa \
  2>&1 | tee "$RUN/run.log"
