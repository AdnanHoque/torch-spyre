#!/usr/bin/env bash
# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 UNIQUE_RUN_NAME" >&2
  exit 2
fi

TRACK_ROOT="${TRACK_ROOT:-/home/adnan-cdx/codex-isolated/torch_spyre_granite_relayout_20260726}"
ANTONI_BASE="${ANTONI_BASE:-/home/adnan-cdx/dt-inductor-codex-clean/profiler_runs/latest_cost_model_granite_block_20260724_202708/antoni_exact_repro_20260724}"
PYTHON="${PYTHON:-/tmp/adnan-cdx-costmodel-kineto/bin/python}"
DXP_BUILD_ROOT="${DXP_BUILD_ROOT:-/home/adnan-cdx/codex-isolated/p04-deeptools-build-shared-20260726}"
RUN_ROOT="${DECODE_DISCOVERY_RUNS_ROOT:-/home/adnan-cdx/codex-isolated/granite_decode_relayout_20260726}"
RUN="${RUN_ROOT}/$1"

if [[ -e "${RUN}" ]]; then
  echo "refusing to reuse existing run directory: ${RUN}" >&2
  exit 2
fi

mkdir -p "${RUN}/cache" "${RUN}/export" "${RUN}/trace" "${RUN}/logits"

DXP_LIBRARY_PATH="$(${FIND:-find} "${DXP_BUILD_ROOT}" -type f -name '*.so*' -printf '%h\n' | sort -u | paste -sd:)"

export PATH="${DXP_BUILD_ROOT}/dxp:/opt/ibm/spyre/runtime/bin:${PATH}"
export PYTHONPATH="${TRACK_ROOT}:${ANTONI_BASE}/test-spyre-scripts/granite/foundation-model-stack:${ANTONI_BASE}/test-spyre-scripts/granite/aiu-fms-testing-utils"
export LD_LIBRARY_PATH="${DXP_LIBRARY_PATH}:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib:/opt/ibm/spyre/spyre-comms/lib:/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/sentinyexec/lib:/tmp/adnan-cdx-costmodel-kineto/lib64/python3.12/site-packages/torch/lib"
export DEEPTOOLS_PATH="${DECODE_DISCOVERY_DEEPTOOLS_PATH:-${TRACK_ROOT}/../granite_relayout_p04_deeptools_b_20260726}"
export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${RUN}/cache"
export DTCOMPILER_EXPORT_DIR="${RUN}/export"
export DEEPRT_EXPORT_DIR="${RUN}/export"
export ANTONI_PROFILE_DIR="${RUN}/trace"
export ANTONI_LOGIT_DUMP_DIR="${RUN}/logits"
export ANTONI_LAYER_LIMIT="${DECODE_DISCOVERY_LAYER_LIMIT:-1}"
export DUMP_SPYRE_CODE=1
export DXP_LX_FRAC_AVAIL=0.2
export SPYRE_LX_PLANNER_RELAYOUT="${DECODE_DISCOVERY_PLANNER_ENABLE:-1}"
export SPYRE_LX_RELAYOUT_COLLECTIVES="${DECODE_DISCOVERY_COLLECTIVES:-all_to_all,all_gather,broadcast}"
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES="${DECODE_DISCOVERY_DISABLED_SOURCES:-buf11,buf14,buf15,buf18,buf20,buf29,buf43,buf44,buf51,buf52,buf53,buf55,buf56,buf57,buf66}"
export SPYRE_LX_RELAYOUT_DUMP_PLANS="${RUN}/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="${RUN}/allocations.jsonl"
unset SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ
unset SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ_CHAIN
unset SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA="${DECODE_DISCOVERY_COMPACT_GQA_ENABLE:-0}"
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA_BUFFERS="${DECODE_DISCOVERY_COMPACT_GQA_BUFFERS:-}"
export SPYRE_RELAYOUT_ORACLE_DECODE_MLP_DOWN_INPUT="${DECODE_DISCOVERY_MLP_DOWN_INPUT_ENABLE:-0}"
unset SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY
unset SPYRE_RELAYOUT_ORACLE_P06_RAMP
unset TORCH_SPYRE_DOWNCAST_WARN

cd "${RUN}"
"${PYTHON}" \
  "${TRACK_ROOT}/experiments/granite_relayout/reference/bench_harness/implementation/antoni_inference_profile.py" \
  --architecture hf_pretrained \
  --model_path /tmp/models/granite-3.3-8b-instruct \
  --tokenizer /tmp/models/granite-3.3-8b-instruct \
  --unfuse_weights \
  --batch_size 1 \
  --max_new_tokens "${DECODE_DISCOVERY_MAX_NEW_TOKENS:-4}" \
  --fixed_prompt_length 512 \
  --iters "${DECODE_DISCOVERY_ITERS:-1}" \
  --device_type spyre \
  --default_dtype fp16 \
  --timing per-token \
  --attention_type sdpa 2>&1 | tee "${RUN}/run.log"
