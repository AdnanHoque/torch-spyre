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
# Negative control for Microbench B: a spliced on-chip bundle with its senprog
# removed MUST hard-fail to load on device (proves the device really executes the
# spliced senprog, not a stale/fallback path). SOLO device use.
set +e
PY=/home/adnan/dt-inductor/.venv/bin/python
MOE=/tmp/ab_moe_routing
DIR=/tmp/ab_bw_micro
COMMON="PYTHONPATH=/tmp/val-boot TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204|likely passed to kineto|address stability|report this to|collection.cpp'; }

SPL=$DIR/Bspl_E8_T512_H2048
NEG=$DIR/Bneg_E8_T512_H2048
rm -rf "$NEG"; cp -r "$SPL" "$NEG"
# Remove the spliced senprog -> device load must fail.
rm -f "$NEG"/loadprogram_to_device/*/init.txt
echo "=== negative control: senprog removed from $NEG ==="
echo "init.txt present after removal: $(ls $NEG/loadprogram_to_device/*/init.txt 2>/dev/null | wc -l)"
rm -rf /tmp/cB_neg
timeout 180 env $COMMON MOE_E=8 MOE_T=512 MOE_H=2048 ONCHIP_DIR=$NEG ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/cB_neg stdbuf -oL $PY $MOE/devval_moe.py 2>&1 | clean | tail -25
rc=${PIPESTATUS[0]}
echo "=== negative control devval exit code = $rc (nonzero / no DIRECT_VALIDATE_OK == PASS) ==="
