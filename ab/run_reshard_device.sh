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
#
# ============================================================================
#  # PARENT RUNS THIS ON DEVICE  --  Claude does NOT execute this script.
# ============================================================================
# Device run of the live-spliced A2 asymmetric reshard. Sources the locked
# profiler stack (ab/profenv.sh), THEN overrides the env so the live
# async_compile.generate_bundle monkeypatch's `dxp_standalone` (resolved by name
# from PATH, async_compile.py:63) is the §5-PATCHED build that admits the mixed
# SuperDSC -- not the harvest stock dxp (which rejects it at SdscTree.cpp:152).
#
# `run_ab.py --lever reshard` always runs with profiling on (run_tsp_stack
# with_profiling=True) for kernel time. The profiler build triggers the flex
# profiling-in-streams thread-lock (~60s stall per sync, see
# CORE_TO_CORE_SWIGLU_BASELINE.md "Device caveat") -- hence the long timeout.
# Run SOLO (single shared accelerator) and check max_err vs CPU + kernel time
# vs the A0 baseline (fused 19.8 ms).
set -euo pipefail

WT=/tmp/core-to-core-wt
PDXP=/home/adnan/dt-inductor/build/deeptools-onchip/dxp
LLVM_LIB=/home/adnan/dt-inductor/build/llvm/lib

# 1. Locked profiler stack (harvest libs + USE_SPYRE_PROFILER _C.so + torch 2.11).
source "$WT/ab/profenv.sh"

# 2. Patched-dxp overrides (prepended so they win the name/lib resolution).
export PATH="$PDXP:$PATH"
export LD_LIBRARY_PATH="$PDXP:$LLVM_LIB:$LD_LIBRARY_PATH"
export DEEPTOOLS_PATH=/home/adnan/dt-inductor/deeptools-onchip

# Sanity: the patched dxp must win `which`.
echo "[run_reshard_device] which dxp_standalone -> $(which dxp_standalone)"

OUT="${OUT:-$WT/ab/results/reshard_swiglu_1x512x4096.txt}"

# 3. The reshard run (long timeout for the flex profiling-in-streams stall).
timeout 1800 /home/adnan/dt-inductor/.venv/bin/python "$WT/ab/run_ab.py" \
  --lever reshard \
  --op fms_granite_micro.swiglu \
  --shape 1 512 4096 \
  --runs 3 \
  --out "$OUT"
