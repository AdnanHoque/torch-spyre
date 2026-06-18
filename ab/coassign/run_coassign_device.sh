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
# Device value-capture for co-assignment. Stock dxp (co-assign emits no data-op,
# so no patched-dxp override) + the locked harvest stack (profenv.sh). No
# profiling (single forward, value only) -> no flex profiling-in-streams stall.
# Single forward (leaner than check_outputs' warmup+measured). Run SOLO.
set -uo pipefail

WT=/tmp/core-to-core-wt
source "$WT/ab/profenv.sh"

OUT="${OUT:-/tmp/c2c-out/spyre_coassign_unfused_out.pt}"
LOG="${LOG:-/tmp/c2c-out/coassign_save.log}"

echo "[run_coassign_device] which dxp_standalone -> $(which dxp_standalone)"
echo "[run_coassign_device] OUT=$OUT LOG=$LOG"

timeout 3600 /home/adnan/dt-inductor/.venv/bin/python \
  "$WT/ab/coassign/save_coassign_out.py" \
  granite_micro_bench.swiglu_unfused 0 "$OUT" 1 512 4096 2>&1 | tee "$LOG"
echo "[run_coassign_device] exit ${PIPESTATUS[0]}"
