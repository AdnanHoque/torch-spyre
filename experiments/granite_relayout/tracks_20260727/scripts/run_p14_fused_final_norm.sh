#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/adnan/codex-isolated/device_parity_tracks_20260726/p14
LEGACY=/home/adnan/codex-isolated/device_parity_pr2939_20260725
RUN_NAME="${1:?run name required}"
PLANNER="${2:-0}"
ITERS="${3:-1}"
RUN="$ROOT/runs/$RUN_NAME"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUNNER="$LEGACY/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct

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
export ANTONI_LAYER_LIMIT=1
export DUMP_SPYRE_CODE=1

export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_ENABLE_LAST_N_TOKENS=1
export SPYRE_FUSE_GRANITE_LAST_TOKEN_STAGE=0
export SPYRE_FUSE_GRANITE_FINAL_NORM_LAST_TOKEN_STAGE=1
export SPYRE_GRANITE_LAST_TOKEN_HEAD_28_OWNER=1
export SPYRE_WORK_DIV_ORACLE_GRANITE_LAST_TOKEN_HEAD=1
export SPYRE_RELAYOUT_ORACLE_GRANITE_P14=1

export SPYRE_LX_PLANNER_RELAYOUT="$PLANNER"
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_to_all
export SPYRE_LX_RELAYOUT_ALL_TO_ALL_MAX_BYTES=4194304
# Isolate the final-stage P14 candidate from the decoder block's generic
# all-to-all candidates.  These six source names are the complete candidate
# set observed in the one-layer block with every replay oracle disabled.
# Keep the final RMSNorm input relayout (buf4 -> buf5) out of this P14 gate.
# P14 itself is the last-token slice handoff sourced by buf5.
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf11,buf15,buf19,buf29,buf44,buf57
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"

export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_NORM=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD=0
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0
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
