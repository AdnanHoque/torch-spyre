#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?run name required}"
PLANNER="${2:?planner flag required}"
ITERS="${3:-1}"
ROOT=/home/adnan/codex-isolated/device_parity_tracks_20260726/p09
LEGACY=/home/adnan/codex-isolated/device_parity_pr2939_20260725
RUN="$ROOT/runs/$RUN_NAME"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUNNER="$LEGACY/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct

test "$PLANNER" = 0 -o "$PLANNER" = 1
test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export" "$RUN/logits" "$RUN/trace"
cd "$RUN"

source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$LEGACY/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/torch-spyre:$LEGACY/foundation-model-stack:$LEGACY/aiu-fms-testing-utils"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export DTCOMPILER_EXPORT_DIR="$RUN/export"
export DEEPRT_EXPORT_DIR="$RUN/export"
export ANTONI_PROFILE_DIR="$RUN/trace"
export ANTONI_LOGIT_DUMP_DIR="$RUN/logits"
export ANTONI_LAYER_LIMIT=1
export DUMP_SPYRE_CODE=1

export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT="$PLANNER"
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_gather
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf11,buf15,buf17,buf18,buf19,buf29,buf30,buf43,buf44,buf45,buf46,buf47,buf48,buf49,buf50,buf51,buf52,buf53,buf55,buf56,buf57,buf66
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"
export SPYRE_RELAYOUT_ORACLE_PREFILL_QKV_INPUTS=1

export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS=0
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0
export SPYRE_RELAYOUT_ORACLE_NO_INPUT_PINNING=1

if [[ "$PLANNER" = 1 ]]; then
  export STCDP_DUMP_TRANSFERS=1
else
  unset STCDP_DUMP_TRANSFERS
fi
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
