#!/usr/bin/env bash
set -euo pipefail

BASE=/home/adnan/codex-isolated/device_parity_tracks_20260726/p14
LEGACY=/home/adnan/codex-isolated/device_parity_pr2939_20260725
FMS=/home/adnan/codex-isolated/device_parity_tracks_20260726/prefill_edges_unmodified_decode_20260726/foundation-model-stack
RUN="$BASE/runs/prefill_edges_unmodified_decode_20260726_b"
PYTHON=/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/kineto-venv/bin/python
RUNNER="$LEGACY/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct

test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export" "$RUN/trace"
cd "$RUN"

source /home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh
export PATH="$BASE/deeptools-build/dxp:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BASE/torch-spyre:$FMS:$LEGACY/aiu-fms-testing-utils"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export DTCOMPILER_EXPORT_DIR="$RUN/export"
export DEEPRT_EXPORT_DIR="$RUN/export"
export ANTONI_PROFILE_DIR="$RUN/trace"
unset ANTONI_LAYER_LIMIT
export DUMP_SPYRE_CODE=1

export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_to_all,all_gather,broadcast
export SPYRE_LX_RELAYOUT_ALL_TO_ALL_MAX_BYTES=-1
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf5,buf6,buf11,buf15,buf17,buf19,buf30,buf33,buf44,buf46,buf47,buf48,buf49,buf50,buf51,buf53,buf55,buf56,buf57,buf59,buf63,buf66
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"

# Working prefill routes only.
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=1
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA_BUFFERS=buf18,buf29
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_SOURCE=buf45
export SPYRE_RELAYOUT_ORACLE_PREFILL_RESIDUAL_ADD_CONSUMER=buf46

# Restore the ordinary generation/decode path.
export SPYRE_ENABLE_LAST_N_TOKENS=0
export SPYRE_FUSE_GRANITE_LAST_TOKEN_STAGE=0
export SPYRE_FUSE_GRANITE_FINAL_NORM_LAST_TOKEN_STAGE=0
export SPYRE_GRANITE_LAST_TOKEN_HEAD_28_OWNER=0
export SPYRE_WORK_DIV_ORACLE_GRANITE_LAST_TOKEN_HEAD=0
export SPYRE_RELAYOUT_ORACLE_GRANITE_P14=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_DOWN_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_NORM=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0
unset STCDP_DUMP_TRANSFERS
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
  --max_new_tokens 4 \
  --fixed_prompt_length 512 \
  --iters 1 \
  --device_type spyre \
  --default_dtype fp16 \
  --timing per-token \
  --attention_type sdpa \
  2>&1 | tee "$RUN/run.log"
