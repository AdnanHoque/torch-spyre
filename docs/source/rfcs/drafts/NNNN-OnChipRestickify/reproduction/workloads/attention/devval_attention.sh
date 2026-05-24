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
# Device A/B for the spliced SDPA QK^T -> softmax on-chip bundle. Run SOLO
# (single shared accelerator -- never in parallel with other device agents).
#
#   1. POSITIVE   : redirect attention runner -> spliced bundle; must be
#                   VALUE-CORRECT (no Compute-CB / RAS hardware error).
#   2. NEGATIVE   : remove the spliced senprog; rerun -> must FAIL (proves the
#                   device loaded OUR spliced program, not a cached/baseline one).
#   3. A/B TIMING : baseline_HBM (stock bundle) vs spliced on-chip, median ms.
#
# The orchestrator runs this; this script never runs concurrently with peers.
set +e

PY=/home/adnan/dt-inductor/.venv/bin/python
SCRIPT=/tmp/ab_attention/devval_attention.py
DIR=/tmp/ab_attention/spliced-attn-qk
SP=$DIR/loadprogram_to_device/spliced-attn-qk-SenProgSend/init.txt

COMMON="PYTHONPATH=/tmp/val-boot TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }

echo "spliced senprog present: $(wc -l < "$SP" 2>/dev/null) lines"

echo "### POSITIVE: redirect runner -> spliced (must load + be VALUE-CORRECT)"
rm -rf /tmp/ab-attn-pos-cache
eval "env $COMMON ONCHIP_DIR=$DIR ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn-pos-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'REDIRECT|DIRECT_VALIDATE_OK|max_err|Error|assert|Mismatch|Traceback|Exception|No such|Compute CB|ComputeHardware|RAS::' | tail -8

echo ""
echo "### NEGATIVE CONTROL: remove senprog -> must FAIL"
mv "$SP" "${SP}.bak"; rm -rf /tmp/ab-attn-pos-cache
eval "env $COMMON ONCHIP_DIR=$DIR ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn-pos-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'DIRECT_VALIDATE_OK|Error|No such|RuntimeError' | tail -6
mv "${SP}.bak" "$SP"; echo "### restored."

echo ""
echo "### A/B TIMING: baseline_HBM vs spliced on-chip (one config per process)"
echo "--- A: baseline_HBM ---"
rm -rf /tmp/ab-attn-base-cache
eval "env $COMMON ONCHIP_BASELINE=1 ONCHIP_MODE=bench \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn-base-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'BENCH|DIRECT_VALIDATE_OK|Error|Compute CB|RAS::' | tail -4
echo "--- B: spliced on-chip ---"
rm -rf /tmp/ab-attn-bench-cache
eval "env $COMMON ONCHIP_DIR=$DIR ONCHIP_MODE=bench \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn-bench-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'BENCH|DIRECT_VALIDATE_OK|Error|Compute CB|RAS::' | tail -4

echo "### DONE"
