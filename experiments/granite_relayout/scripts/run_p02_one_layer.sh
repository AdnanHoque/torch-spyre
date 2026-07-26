#!/usr/bin/env bash
# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/activate_device_parity_env.sh"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 UNIQUE_RUN_NAME" >&2
  exit 2
fi

RUN="${GRANITE_PARITY_ROOT}/runs/$1"
if [[ -e "${RUN}" ]]; then
  echo "refusing to reuse existing run directory: ${RUN}" >&2
  exit 2
fi

mkdir -p "${RUN}/cache" "${RUN}/export" "${RUN}/trace" "${RUN}/logits"

export HF_HOME=/tmp/models/hf_cache
export PYTHONUNBUFFERED=1
export TORCHINDUCTOR_CACHE_DIR="${RUN}/cache"
export DTCOMPILER_EXPORT_DIR="${RUN}/export"
export DEEPRT_EXPORT_DIR="${RUN}/export"
export ANTONI_PROFILE_DIR="${RUN}/trace"
export ANTONI_LOGIT_DUMP_DIR="${RUN}/logits"
export ANTONI_LAYER_LIMIT=1
export DUMP_SPYRE_CODE=1
export SPYRE_LX_PLANNER_RELAYOUT=1
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA=1
export SPYRE_RELAYOUT_ORACLE_COMPACT_GQA_BUFFERS=buf29
export SPYRE_RELAYOUT_ORACLE_REINTERPRET_OUTPUT_CLONE_BUFFERS=buf29
export SPYRE_LX_RELAYOUT_DISABLED_SOURCES=buf11,buf14,buf15,buf18,buf20,buf43,buf44,buf51,buf52,buf53,buf55,buf56,buf57,buf66
export SPYRE_LX_RELAYOUT_DUMP_PLANS="${RUN}/relayout_plans.jsonl"
export SPYRE_LX_RELAYOUT_DUMP_ALLOCATIONS="${RUN}/allocations.jsonl"

cd "${RUN}"
python "${GRANITE_PARITY_ROOT}/spyre-granite-e2e-bench/implementation/antoni_inference_profile.py" \
  --architecture hf_pretrained \
  --model_path /tmp/models/granite-3.3-8b-instruct \
  --tokenizer /tmp/models/granite-3.3-8b-instruct \
  --unfuse_weights \
  --batch_size 1 \
  --max_new_tokens 1 \
  --fixed_prompt_length 512 \
  --iters 1 \
  --device_type spyre \
  --default_dtype fp16 \
  --timing per-token \
  --attention_type sdpa 2>&1 | tee "${RUN}/run.log"

