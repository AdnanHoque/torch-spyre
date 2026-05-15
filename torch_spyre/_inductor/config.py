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

# Default-off producer-aligned physical core mapping for compatible
# compiler-inserted restickify ops. This does not change restickify placement
# or selected tensor layouts.
align_restickify_core_mapping: bool = (
    os.environ.get("SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING", "0") == "1"
)

# Default-off restickify work-distribution steering. When enabled, restickify
# ops with an unambiguous in-graph producer prefer the output dimension that
# corresponds to the producer's dominant split dimension.
align_restickify_work_distribution: bool = (
    os.environ.get("SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION", "0") == "1"
)

# Default-off exact byte-hop telemetry for compiler-inserted restickify ops.
restickify_ring_telemetry: bool = (
    os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY", "0") == "1"
)

restickify_ring_telemetry_jsonl: str = os.environ.get(
    "SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL",
    "",
)

# Default-off prototype guard for Stage 3B restickify locality. When enabled,
# producer-aligned restickify core mapping overrides must certify zero modeled
# RIU byte-hops before reaching codegen.
restickify_locality_assert: bool = (
    os.environ.get("SPYRE_RESTICKIFY_LOCALITY_ASSERT", "0") == "1"
)

install_config_module(sys.modules[__name__])
