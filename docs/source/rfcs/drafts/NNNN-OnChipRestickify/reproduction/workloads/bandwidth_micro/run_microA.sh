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
# Microbench A driver: one fresh process per size (isolates runtime DMA stalls).
# SOLO device use. Stock dxp on PATH (HBM variant). Per-size timeout guard.
set +e
PY=/home/adnan/dt-inductor/.venv/bin/python
DIR=/tmp/ab_bw_micro
COMMON="PYTHONPATH=/tmp/val-boot TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204|likely passed to kineto|address stability|report this to|collection.cpp'; }
OUT=$DIR/microA_results.txt
: > "$OUT"
SIZES="${BW_SIZES_MB:-1 2 4 8 16 32}"
W="${BENCH_WARMUP:-12}"
N="${BENCH_ITERS:-50}"
for s in $SIZES; do
  echo "### size_mb=$s (W=$W N=$N)"
  rm -rf /tmp/c_bwa_$s
  timeout 240 env $COMMON BW_SIZES_MB=$s BENCH_WARMUP=$W BENCH_ITERS=$N \
    TORCHINDUCTOR_CACHE_DIR=/tmp/c_bwa_$s \
    stdbuf -oL -eL $PY $DIR/bw_hbm_micro.py 2>&1 | clean | grep -E 'BWA|Error|Signal|Fence|RAS' | tee -a "$OUT"
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 124 ]; then echo "TIMEOUT size=$s" | tee -a "$OUT"; fi
done
echo "### MICROBENCH A DONE"
