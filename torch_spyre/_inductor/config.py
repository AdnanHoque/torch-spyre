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
chunk_large_tensors: bool = os.environ.get("CHUNK_LARGE_TENSORS", "0") == "1"

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

allow_all_ops_in_lx_planning: bool = False

# Fraction of LX (scratchpad memory) reserved for the deeptools backend (DXP).
# Higher = more LX for DDC's tile pipelining = faster matmul kernels.
# Empirical sweep (matmul m=512, k=4096, n in {1024, 4096, 12800}) shows
# 0.8 closes the matmul kernel_ms gap with sendnn by 1.8-2.4x; the residual
# 20% LX is the "user" scratchpad (only used when LX_PLANNING=1, off by
# default). At LX_FRAC=0.2 (prior default) inductor's scratchpad allocator
# would get 80% but was never invoked, leaving the LX underutilised.
# NB: DDC reads this via getenv("DXP_LX_FRAC_AVAIL") in dxp.cpp -- the python
# config value alone does NOT propagate. To opt in to the larger DDC tile
# budget, set DXP_LX_FRAC_AVAIL=0.8 in the environment; the LX-aware 2D split
# in work_division.py keys off this value (>=0.5) to pick deeper PT passes.
dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

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

# When set, override the default per-dim work split for matmul/bmm ops
# with the lowest-cost (b, m, n, k) split chosen by the cost-model
# planner in work_division.py. Off by default.
cost_model_matmul_planner: bool = (
    os.environ.get("SPYRE_COST_MODEL_MATMUL_PLANNER", "0") == "1"
)

# When set, override the default per-dim work split for simple reduction
# ops (sum, mean, max, min, amax, amin, exx2) with the lowest-cost
# (d_splits, r_splits) choice from the reduction sibling cost-model
# planner in work_division.py. Off by default.
cost_model_reduction_planner: bool = (
    os.environ.get("SPYRE_COST_MODEL_REDUCTION_PLANNER", "0") == "1"
)

install_config_module(sys.modules[__name__])
