#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <run-name> [iters]" >&2
  exit 2
fi

BASE=/home/adnan/codex-isolated/device_parity_tracks_20260726/p14
LEGACY=/home/adnan/codex-isolated/device_parity_pr2939_20260725
RUN="$BASE/runs/$1"
ITERS="${2:-5}"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUNNER="$LEGACY/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct

test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export" "$RUN/logits" "$RUN/trace"
cd "$RUN"

source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$BASE/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BASE/torch-spyre:$BASE/foundation-model-stack:$LEGACY/aiu-fms-testing-utils"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export DTCOMPILER_EXPORT_DIR="$RUN/export"
export DEEPRT_EXPORT_DIR="$RUN/export"
export ANTONI_PROFILE_DIR="$RUN/trace"
export ANTONI_LOGIT_DUMP_DIR="$RUN/logits"
unset ANTONI_LAYER_LIMIT
export DUMP_SPYRE_CODE=1

export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_to_all,all_gather,broadcast
export SPYRE_LX_RELAYOUT_ALL_TO_ALL_MAX_BYTES=-1
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf11,buf15,buf17,buf19,buf30,buf44,buf46,buf47,buf48,buf49,buf51,buf53,buf55,buf56,buf57,buf66
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"
export STCDP_DUMP_TRANSFERS=1

# P01/P02: compact K/V GQA attention routes.
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=1
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA_BUFFERS=buf18,buf29

# P03/P04: shared MLP input and attention-output projection routes.
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=1

# P12: final repeated residual handoff.
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_SOURCE=buf45
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_CONSUMER=buf46

# P13/P14: final-token permutation and 28-owner LM head.
export SPYRE_ENABLE_LAST_N_TOKENS=1
export SPYRE_FUSE_GRANITE_LAST_TOKEN_STAGE=0
export SPYRE_FUSE_GRANITE_FINAL_NORM_LAST_TOKEN_STAGE=1
export SPYRE_GRANITE_LAST_TOKEN_HEAD_28_OWNER=1
export SPYRE_WORK_DIV_ORACLE_GRANITE_LAST_TOKEN_HEAD=1
export SPYRE_RELAYOUT_ORACLE_GRANITE_P14=1

# Keep all unfinished routes disabled.
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_NORM=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0
unset DXP_RELAYOUT_PRESERVE_SINGLETON_Y
unset DXP_RELAYOUT_DIRECT_COPY_NOP
unset STCDP_FORCE_UNICAST_SPLIT
unset STCDP_FORCE_NOOPT
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
