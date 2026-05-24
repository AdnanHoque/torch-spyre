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
# Compiler-driven on-chip handoff E2E: torch.compile itself emits the mixed
# bundle (no splice, no runner redirect). Flag OFF = baseline (no datadscs_);
# flag ON = compiler emits the mixed SuperDSC, the PATCHED dxp accepts it
# (success proves the gate was exercised), and the device runs value-correct.
# Run SOLO (single shared accelerator). Paths come from reproduction/env.sh.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=../env.sh
source "$HERE/../env.sh"
PATCHED_DXP_DIR="$(dirname "$PATCHED_DXP")"
BASEENV="PATH=$PATCHED_DXP_DIR:$PATH PYTHONPATH=$VAL_BOOT TORCH_DISABLE_ADDR2LINE=1 TORCHINDUCTOR_COMPILE_THREADS=1"
clean() { grep -vE 'symbolizing C\+\+|TORCH_DISABLE|Module.cpp:204'; }

echo "dxp_standalone resolves to: $(PATH=$PATCHED_DXP_DIR:$PATH which dxp_standalone)"
echo
echo "### NEGATIVE CONTROL: flag OFF -> baseline (no datadscs_), value-correct"
rm -rf "$WORK_DIR/e2e-base"
env $BASEENV TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/e2e-base" SPYRE_ONCHIP_HANDOFF_REALIZE=0 \
  "$PYTHON" "$HERE/e2e_onchip.py" 2>&1 | clean | \
  grep -E 'E2E|Error|Traceback|assert|Datadsc|CalledProcess' | tail -6
echo
echo "### E2E ON-CHIP: flag ON -> COMPILER emits mixed bundle, patched dxp accepts it"
rm -rf "$WORK_DIR/e2e-onchip"
env $BASEENV TORCHINDUCTOR_CACHE_DIR="$WORK_DIR/e2e-onchip" SPYRE_ONCHIP_HANDOFF_REALIZE=1 \
  "$PYTHON" "$HERE/e2e_onchip.py" 2>&1 | clean | \
  grep -E 'E2E|Error|Traceback|assert|Datadsc|CalledProcess' | tail -6
echo "### DONE"
