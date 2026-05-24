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
# Device A/B for the spliced seq=512 SDPA QK^T -> softmax on-chip bundle. Run
# SOLO (single shared accelerator -- never in parallel with other device agents).
#
#   1. POSITIVE   : redirect attention runner -> spliced bundle; must be
#                   VALUE-CORRECT (no Compute-CB / RAS hardware error).
#   2. NEGATIVE   : remove the spliced senprog; rerun -> must FAIL (proves the
#                   device loaded OUR spliced program, not a cached/baseline one).
#   3. A/B TIMING : baseline_HBM (stock bundle, ref 2.5998 ms) vs spliced on-chip.
#
# Shapes: ATTN_BH=32 ATTN_SEQ=512 ATTN_HEAD_DIM=128 (match the spliced bundle).
# The orchestrator runs this; this script never runs concurrently with peers.
set +e

PY=/home/adnan/dt-inductor/.venv/bin/python
SCRIPT=/tmp/ab_attention_512/devval_attention_512.py
DIR=/tmp/ab_attention_512/spliced-attn-512
SP=$DIR/loadprogram_to_device/spliced-attn-512-SenProgSend/init.txt

SHAPES="ATTN_BH=32 ATTN_SEQ=512 ATTN_HEAD_DIM=128"
COMMON="PYTHONPATH=/tmp/val-boot TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }

echo "baseline reference: 2.5998 ms (bh=32 seq=512 head_dim=128)"
echo "spliced senprog present: $(wc -l < "$SP" 2>/dev/null) lines"

echo "### POSITIVE: redirect runner -> spliced (must load + be VALUE-CORRECT)"
rm -rf /tmp/ab-attn512-pos-cache
eval "env $COMMON $SHAPES ONCHIP_DIR=$DIR ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn512-pos-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'REDIRECT|DIRECT_VALIDATE_OK|max_err|Error|assert|Mismatch|Traceback|Exception|No such|Compute CB|ComputeHardware|RAS::' | tail -8

echo ""
echo "### NEGATIVE CONTROL: remove senprog -> must FAIL"
mv "$SP" "${SP}.bak"; rm -rf /tmp/ab-attn512-pos-cache
eval "env $COMMON $SHAPES ONCHIP_DIR=$DIR ONCHIP_MODE=validate \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn512-pos-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'DIRECT_VALIDATE_OK|Error|No such|RuntimeError' | tail -6
mv "${SP}.bak" "$SP"; echo "### restored."

echo ""
echo "### A/B TIMING: baseline_HBM vs spliced on-chip (one config per process)"
echo "--- A: baseline_HBM (ref 2.5998 ms) ---"
rm -rf /tmp/ab-attn512-base-cache
eval "env $COMMON $SHAPES ONCHIP_BASELINE=1 ONCHIP_MODE=bench \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn512-base-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'BENCH|DIRECT_VALIDATE_OK|Error|Compute CB|RAS::' | tail -4
echo "--- B: spliced on-chip ---"
rm -rf /tmp/ab-attn512-bench-cache
eval "env $COMMON $SHAPES ONCHIP_DIR=$DIR ONCHIP_MODE=bench \
  TORCHINDUCTOR_CACHE_DIR=/tmp/ab-attn512-bench-cache $PY $SCRIPT" 2>&1 | clean | \
  grep -E 'BENCH|DIRECT_VALIDATE_OK|Error|Compute CB|RAS::' | tail -4

echo "### DONE"
