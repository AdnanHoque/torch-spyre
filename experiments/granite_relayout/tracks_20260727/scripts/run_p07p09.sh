#!/usr/bin/env bash
# Isolated continuation of the P07/P09 attention-edge lane.
# Reads the cdx pinned env / dxp wrapper / reference model tree read-only;
# writes only under /home/adnan-cdx/claude-isolated/p07p09_20260727.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 <run-name> <p07|p09|both> [layers|full] [iters]" >&2; exit 2
fi
SELF=/home/adnan-cdx/claude-isolated/p07p09_20260727
SRC=/home/adnan-cdx/codex-isolated/p07_p09_completion_20260726
REPO="$SELF/torch-spyre"
RUN="$SELF/runs/$1"
ARM="$2"; LAYERS="${3:-1}"; ITERS="${4:-1}"
REF=/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724/test-spyre-scripts/granite
PYTHON=/tmp/adnan-cdx-costmodel-kineto/bin/python
RUNNER="$REPO/experiments/granite_relayout/reference/bench_harness/implementation/antoni_inference_profile.py"
MODEL=/tmp/models/granite-3.3-8b-instruct
case "$ARM" in
  p07) P07=1; P09=0 ;;
  p09) P07=0; P09=1 ;;
  both) P07=1; P09=1 ;;
  *) echo "arm must be p07, p09, or both" >&2; exit 2 ;;
esac
test ! -e "$RUN"
mkdir -p "$RUN/cache" "$RUN/export" "$RUN/logits" "$RUN/trace"
cd "$RUN"
source /home/adnan-cdx/spyre-envs/pr2/activate.sh
export PATH="$SRC/dxp_wrapper:$PATH"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:$REF/foundation-model-stack:$REF/aiu-fms-testing-utils"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache"
export DTCOMPILER_EXPORT_DIR="$RUN/export"
export DEEPRT_EXPORT_DIR="$RUN/export"
export ANTONI_PROFILE_DIR="$RUN/trace"
export ANTONI_LOGIT_DUMP_DIR="$RUN/logits"
if [[ "$LAYERS" = full ]]; then unset ANTONI_LAYER_LIMIT; else export ANTONI_LAYER_LIMIT="$LAYERS"; fi
export DUMP_SPYRE_CODE=1
export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_LX_RELAYOUT_COLLECTIVES=all_gather
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf4,buf11,buf15,buf17,buf18,buf19,buf29,buf30,buf43,buf44,buf45,buf46,buf47,buf48,buf49,buf50,buf51,buf52,buf53,buf55,buf56,buf57,buf66
export SPYRE_LX_RELAYOUT_DUMP_PLANS="$RUN/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="$RUN/allocations.jsonl"
export STCDP_DUMP_TRANSFERS=1
export SPYRE_RELAYOUT_ORACLE_PREFILL_ROPE_INPUT="$P07"
export SPYRE_RELAYOUT_ORACLE_PREFILL_QKV_INPUTS="$P09"
export SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS=0
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0
export SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
export SPYRE_RELAYOUT_ORACLE_P06_RAMP=0
export SPYRE_RELAYOUT_ORACLE_NO_INPUT_PINNING="$P09"
unset TORCH_SPYRE_DOWNCAST_WARN
printf 'arm=%s\nlayers=%s\niters=%s\nDXP_LX_FRAC_AVAIL=%s\nrepo=%s\n' \
  "$ARM" "$LAYERS" "$ITERS" "$DXP_LX_FRAC_AVAIL" "$REPO" > "$RUN/contract.txt"
"$PYTHON" "$RUNNER" \
  --architecture hf_pretrained --model_path "$MODEL" --tokenizer "$MODEL" \
  --unfuse_weights --batch_size 1 --max_new_tokens 1 --fixed_prompt_length 512 \
  --iters "$ITERS" --device_type spyre --default_dtype fp16 \
  --timing per-token --attention_type sdpa 2>&1 | tee "$RUN/run.log"
