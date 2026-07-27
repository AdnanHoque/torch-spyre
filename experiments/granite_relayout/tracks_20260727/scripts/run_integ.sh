#!/usr/bin/env bash
# Independent reproduction of the P08 native-src -> SenDNN-dst bridge result.
# Reads Codex's pinned env / deeptools / FMS / bench read-only; writes only under
# /home/adnan/claude-isolated/granite_parity_20260727.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 11 ]]; then
  echo "usage: $0 <run-name> [iters] [mlp-down-proj:0|1] [mb-split] [out-split] [p08:0|1] [p07:0|1]" >&2
  exit 2
fi

SELF=/home/adnan/claude-isolated/p07_integrated_20260727
TRACK=/home/adnan/codex-isolated/device_parity_tracks_20260726
BASE="$TRACK/p14"
LEGACY=/home/adnan/codex-isolated/device_parity_pr2939_20260725
RUN="$SELF/runs/$1"
ITERS="${2:-1}"
MLP_DOWN_PROJ="${3:-1}"
MLP_DOWN_PROJ_MB="${4:-16}"
MLP_DOWN_PROJ_OUT="${5:-2}"
P08="${6:-1}"
P07="${7:-0}"
P06="${8:-1}"
P09="${9:-0}"
LXFRAC="${10:-0.2}"
RESTICK_LX="${11:-0}"
if [[ "$MLP_DOWN_PROJ" = 1 && $((MLP_DOWN_PROJ_MB * MLP_DOWN_PROJ_OUT)) -ne 32 ]]; then
  echo "mlp-down-proj mb-split * out-split must equal 32" >&2; exit 2
fi

PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUNNER="$LEGACY/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct

test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export" "$RUN/logits" "$RUN/trace"
cd "$RUN"

source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$BASE/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SELF/torch-spyre:$BASE/foundation-model-stack:$LEGACY/aiu-fms-testing-utils"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export DTCOMPILER_EXPORT_DIR="$RUN/export"
export DEEPRT_EXPORT_DIR="$RUN/export"
export ANTONI_PROFILE_DIR="$RUN/trace"
export ANTONI_LOGIT_DUMP_DIR="$RUN/logits"
unset ANTONI_LAYER_LIMIT
export DUMP_SPYRE_CODE=1

export DXP_LX_FRAC_AVAIL="$LXFRAC"
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_to_all,all_gather,broadcast
export SPYRE_LX_RELAYOUT_ALL_TO_ALL_MAX_BYTES=-1
if [[ "$MLP_DOWN_PROJ" = 1 ]]; then
  export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf11,buf15,buf17,buf19,buf30,buf39,buf44,buf46,buf47,buf48,buf49,buf51,buf53,buf55,buf57,buf66
else
  export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf11,buf15,buf17,buf19,buf30,buf39,buf44,buf46,buf47,buf48,buf49,buf51,buf53,buf55,buf56,buf57,buf66
fi
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"
export STCDP_DUMP_TRANSFERS=1
export SPYRE_LX_ALLOW_RESTICKIFY_READ="$RESTICK_LX"

# Accepted working stack: P01/P02/P03/P04/P12/P13/P14.
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=1
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA_BUFFERS=buf18,buf29
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_SOURCE=buf45
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_CONSUMER=buf46
export SPYRE_ENABLE_LAST_N_TOKENS=1
export SPYRE_FUSE_GRANITE_LAST_TOKEN_STAGE=0
export SPYRE_FUSE_GRANITE_FINAL_NORM_LAST_TOKEN_STAGE=1
export SPYRE_GRANITE_LAST_TOKEN_HEAD_28_OWNER=1
export SPYRE_WORK_DIV_ORACLE_GRANITE_LAST_TOKEN_HEAD=1
export SPYRE_RELAYOUT_ORACLE_GRANITE_P14=1

# P06: preserve 8 token x 4 query-head cohorts through rotary, gather at the QK consumer.
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY="$P06"
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY_BUFFERS=buf11,buf12,buf13,buf14
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_FAST_EMISSION=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_AXIS_BRIDGE=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0

# P08: final normalized attention output, head-major 4x8 grid to token-major.
export SPYRE_RELAYOUT_ORACLE_PREFILL_ATTN_PERMUTATION="$P08"

export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ="$MLP_DOWN_PROJ"
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ_MB="$MLP_DOWN_PROJ_MB"
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ_OUT="$MLP_DOWN_PROJ_OUT"
export SPYRE_RELAYOUT_ORACLE_PREFILL_ROPE_INPUT="$P07"
export SPYRE_RELAYOUT_ORACLE_PREFILL_QKV_INPUTS="$P09"
export SPYRE_RELAYOUT_ORACLE_NO_INPUT_PINNING="$P09"
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_NORM=0
unset DXP_RELAYOUT_PRESERVE_SINGLETON_Y
unset DXP_RELAYOUT_DIRECT_COPY_NOP
unset STCDP_FORCE_UNICAST_SPLIT
unset STCDP_FORCE_NOOPT
unset TORCH_SPYRE_DOWNCAST_WARN

printf 'tree=accepted+p07lane(merged)\nstack=P01,P02,P03,P04,P06,P08,P12,P13,P14\niters=%s\nmlp_down_proj=%s\nmlp_down_proj_mb=%s\nmlp_down_proj_out=%s\np08=%s\ntorch_spyre=%s\n' \
  "$ITERS" "$MLP_DOWN_PROJ" "$MLP_DOWN_PROJ_MB" "$MLP_DOWN_PROJ_OUT" "$P08" "$SELF/torch-spyre" > "$RUN/contract.txt"
printf 'p07=%s\np06=%s\np09=%s\nlxfrac=%s\nrestickify_lx=%s\n' "$P07" "$P06" "$P09" "$LXFRAC" "$RESTICK_LX" >> "$RUN/contract.txt"

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
