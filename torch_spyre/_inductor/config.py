# Copyright 2025 The Torch-Spyre Authors.
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

import os
import sys

from torch.utils._config_module import install_config_module

lx_planning: bool = os.environ.get("LX_PLANNING", "0") == "1"

dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

sencores: int = int(os.getenv("SENCORES", "32"))

# Phase 2.0 element-count priority heuristic for matmul output-dim ranking.
# When True, output dims are ranked by element count instead of stick-adjusted
# iteration-space units. Default off — calibrated against Phase 1.0 split-gap
# measurements (tests/diag_split_gap_results.md) which showed the planner's
# default pick of (32, 1, 1) is 15-40% slower than (1, 32, 1) on most prefill
# matmul shapes because the stick-adjusted N is artificially deflated relative
# to non-stick M.
output_element_priority: bool = (
    os.environ.get("OUTPUT_ELEMENT_PRIORITY", "0") == "1"
)

install_config_module(sys.modules[__name__])
