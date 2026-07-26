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

GRANITE_PARITY_ROOT="${GRANITE_PARITY_ROOT:-/home/adnan/codex-isolated/device_parity_pr2939_20260725}"
GRANITE_PARITY_RUNTIME="${GRANITE_PARITY_RUNTIME:-/home/adnan/spyre-envs/main-e3a79c56-hints-pr2/activate.sh}"

# shellcheck disable=SC1090
source "${GRANITE_PARITY_RUNTIME}"

export PATH="${GRANITE_PARITY_ROOT}/deeptools-build/dxp:${PATH}"
export LD_LIBRARY_PATH="/opt/ibm/spyre/runtime/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${GRANITE_PARITY_ROOT}/torch-spyre:${GRANITE_PARITY_ROOT}/foundation-model-stack:${GRANITE_PARITY_ROOT}/aiu-fms-testing-utils"
export DXP_LX_FRAC_AVAIL=0.2
unset TORCH_SPYRE_DOWNCAST_WARN

