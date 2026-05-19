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

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

allow_all_ops_in_lx_planning: bool = False

# Fraction of LX (scratchpad memory) reserved for the deeptools backend (DXP).
# Higher = more LX for DDC's tile pipelining = faster matmul kernels.
# Empirical sweep (matmul m=512, k=4096, n in {1024, 4096, 12800}) shows
# 0.8 closes the matmul kernel_ms gap with sendnn by 1.8-2.4x; the residual
# 20% LX is the "user" scratchpad (only used when LX_PLANNING=1, off by
# default). At LX_FRAC=0.2 (prior default) inductor's scratchpad allocator
# would get 80% but was never invoked, leaving the LX underutilised.
dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.8"))

sencores: int = int(os.getenv("SENCORES", "32"))

# k_fast: a two-layer optimisation for K-split matmul work-divisions.
#   Layer 1 (planner, core_division.py): picks (1, n, k>1) over pure-M
#     for narrow-N small-M matmul shapes that would otherwise leave the
#     PT array under-utilised.
#   Layer 2 (SDSC emitter, codegen/compute_ops.py): permutes physical
#     core IDs so K-collaborators land on adjacent ring positions,
#     reducing PSUM chain hops from m*n to 1.
# Set SPYRE_CORE_ID_K_FAST_EMISSION=0 to disable both layers.
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)

# When set, replace a matmul's pure-m work-split with a 2D m x n co-split
# (see tests/diag_hbm_bank_aware_findings.md). Off by default.
two_d_mn_split: bool = os.environ.get("SPYRE_2D_MN_SPLIT", "0") == "1"

install_config_module(sys.modules[__name__])
