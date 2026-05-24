#!/bin/bash
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
#
# Device baselines for the full-block workloads on Spyre. Run SOLO. Captures the
# baseline timing AND any unsupported-op / compile failures (the transformer
# block + MoE full block do NOT compile on the current stack -- see
# PerformanceResults.md; MoE expert FFN does, ~125.85 ms). Paths from env.sh;
# workload scripts live in ../workloads/.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"
PATCHED_DXP_DIR="$(dirname "$PATCHED_DXP")"
W="$HERE/../workloads"
COMMON="PATH=$PATCHED_DXP_DIR:$PATH PYTHONPATH=$VAL_BOOT TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }
run() { # $1=label $2=script $3=cachedir
  echo "############ $1 ############"
  rm -rf "$3"
  eval "env $COMMON TORCHINDUCTOR_CACHE_DIR=$3 $PYTHON $2" 2>&1 | clean | \
    grep -iE 'BENCH|max_err|VALIDATE|Error|Traceback|Exception|not (supported|implemented)|fallback|Unsupported|RAS::|Compute CB' | tail -12
  echo
}
run "TRANSFORMER BLOCK" "$W/transformer_block/transformer_block_workload.py" "$WORK_DIR/bl-xfmr"
run "MoE EXPERT FFN"    "$W/moe_block/moe_ffn_workload.py"                    "$WORK_DIR/bl-moeffn"
run "MoE FULL BLOCK"    "$W/moe_block/moe_block_workload.py"                  "$WORK_DIR/bl-moeblk"
echo "### DONE"
