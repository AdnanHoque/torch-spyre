#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 7 ]]; then
  echo "usage: $0 <run-name> <planner:0|1> <p03:0|1> [iters] [layers|full] [max-new-tokens] [root]" >&2
  exit 2
fi

RUN_NAME="$1"
PLANNER="$2"
P03="$3"
ITERS="${4:-1}"
LAYERS="${5:-full}"
MAX_NEW_TOKENS="${6:-1}"
ROOT="${7:-/home/adnan/codex-isolated/device_parity_tracks_20260726/p14}"
LEGACY=/home/adnan/codex-isolated/device_parity_pr2939_20260725
RUN="$ROOT/runs/$RUN_NAME"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUNNER="$LEGACY/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct

test "$PLANNER" = 0 -o "$PLANNER" = 1
test "$P03" = 0 -o "$P03" = 1
test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export" "$RUN/logits" "$RUN/trace"
cd "$RUN"

source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$ROOT/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/torch-spyre:$ROOT/foundation-model-stack:$LEGACY/aiu-fms-testing-utils"
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

# Preserve the production LX budget while selecting only the non-attention
# Granite edges that have already passed isolated structural gates.
export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT="$PLANNER"
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_to_all,all_gather
export SPYRE_LX_RELAYOUT_ALL_TO_ALL_MAX_BYTES=-1
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf11,buf15,buf17,buf18,buf19,buf29,buf30,buf43,buf44,buf46,buf47,buf48,buf49,buf50,buf51,buf53,buf55,buf56,buf57,buf66
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"

# P12: post-attention residual add. This changes transport/work placement only;
# it does not alter any attention operation.
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_SOURCE=buf45
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_CONSUMER=buf46

# P03: shared MLP input into gate/up. Keep switchable because production-size
# dense BMM accumulation is token-correct but not bit-exact under LX transport.
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS="$P03"
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_NORM=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=0

# P14 + P13: fuse final RMSNorm/slice/head, repartition the last token to 32
# hidden shards, then gather those shards to the 28 LM-head output owners.
export SPYRE_ENABLE_LAST_N_TOKENS=1
export SPYRE_FUSE_GRANITE_LAST_TOKEN_STAGE=0
export SPYRE_FUSE_GRANITE_FINAL_NORM_LAST_TOKEN_STAGE=1
export SPYRE_GRANITE_LAST_TOKEN_HEAD_28_OWNER=1
export SPYRE_WORK_DIV_ORACLE_GRANITE_LAST_TOKEN_HEAD=1
export SPYRE_RELAYOUT_ORACLE_GRANITE_P14=1

# Park attention completely.
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0

if [[ "$PLANNER" = 1 ]]; then
  export STCDP_DUMP_TRANSFERS=1
else
  unset STCDP_DUMP_TRANSFERS
fi
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
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --fixed_prompt_length 512 \
  --iters "$ITERS" \
  --device_type spyre \
  --default_dtype fp16 \
  --timing per-token \
  --attention_type sdpa \
  2>&1 | tee "$RUN/run.log"
