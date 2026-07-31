#!/usr/bin/env bash
# Copyright 2026 IBM Corporation
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

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
torch_spyre_root=$(cd "$script_dir/../.." && pwd)

python_bin=${PYTHON_BIN:-/tmp/adnan-cdx-costmodel-kineto/bin/python}
model_path=${MODEL_PATH:-/tmp/models/granite-3.3-8b-instruct-FP8}
run_root=${STUDY_ROOT:-/tmp/torch-spyre-granite-fp8-one-layer}

: "${ANTONI_RUNNER:?set ANTONI_RUNNER to antoni_inference_profile.py}"
: "${FMS_ROOT:?set FMS_ROOT to the pinned foundation-model-stack checkout}"
: "${AIU_FMS_UTILS_ROOT:?set AIU_FMS_UTILS_ROOT to aiu-fms-testing-utils}"
: "${FP8_DEPS_ROOT:?set FP8_DEPS_ROOT to the isolated FMS-MO/TorchAO install}"

for required_path in \
    "$python_bin" \
    "$model_path/config.json" \
    "$ANTONI_RUNNER" \
    "$FMS_ROOT/fms" \
    "$AIU_FMS_UTILS_ROOT/aiu_fms_testing_utils" \
    "$FP8_DEPS_ROOT/fms_mo" \
    "$FP8_DEPS_ROOT/torchao"; do
    if [[ ! -e "$required_path" ]]; then
        echo "required path does not exist: $required_path" >&2
        exit 2
    fi
done

for target_var in AIU_ARCH DATA_PREC_CONFIG PREC_CONFIG SENARCH SENTARGET SPYRE_ARCH; do
    target_value=${!target_var-}
    target_value=${target_value,,}
    if [[ "$target_value" == *1p5* || "$target_value" == *1.5* ]]; then
        echo "refusing non-DD2 target from $target_var=${!target_var}" >&2
        exit 2
    fi
done

mkdir -p "$run_root"/{cache,export,outputs,trace_warm}

bridge_pythonpath="$script_dir:$FP8_DEPS_ROOT:$torch_spyre_root"
bridge_pythonpath="$bridge_pythonpath:$FMS_ROOT:$AIU_FMS_UTILS_ROOT"
export PYTHONPATH="$bridge_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_SPYRE_ENABLE_FMS_MO_FP8_BRIDGE=1
export TORCH_SPYRE_FP8_TARGET=dd2
export TORCHINDUCTOR_CACHE_DIR="$run_root/cache"
export DTCOMPILER_EXPORT_DIR="$run_root/export"
export DEEPRT_EXPORT_DIR="$run_root/export"
export ANTONI_PROFILE_DIR="$run_root/trace_warm"
export ANTONI_LAYER_LIMIT=${ANTONI_LAYER_LIMIT:-1}
export DUMP_SPYRE_CODE=1
unset TORCH_SPYRE_DOWNCAST_WARN || true

bridge_runner="$script_dir/run_antoni_with_fms_mo_fp8_bridge.py"

if [[ ${FP8_BRIDGE_CHECK_ONLY:-0} == 1 ]]; then
    exec "$python_bin" "$bridge_runner" \
        --runner "$ANTONI_RUNNER" \
        --check-only
fi

exec "$python_bin" "$bridge_runner" \
    --runner "$ANTONI_RUNNER" \
    -- \
    --architecture hf_pretrained \
    --model_path "$model_path" \
    --tokenizer "$model_path" \
    --unfuse_weights \
    --cast_bf16_to_fp16 \
    --compile \
    --batch_size 1 \
    --max_new_tokens 1 \
    --fixed_prompt_length 512 \
    --iters 1 \
    --device_type spyre \
    --timing per-token \
    --attention_type sdpa \
    --output_path "$run_root/outputs"
