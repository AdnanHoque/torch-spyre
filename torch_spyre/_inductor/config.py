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

# Experimental Inductor-level SDPA prefill decomposition.  This keeps the
# normal decomposition as the default and, when enabled, emits the blockwise
# online-softmax form used by the Flash Attention building-block tests.
flash_attention_prefill: bool = (
    os.environ.get("SPYRE_FLASH_ATTENTION_PREFILL", "0") == "1"
)
flash_attention_prefill_block_size: int = int(
    os.environ.get("SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE", "128")
)
# Default-off proof path for a mixed-SDSC, double-buffered flash-attention
# prefill pipeline.  The first implementation only builds descriptor/scheduler
# proof artifacts; compiler promotion remains gated until device overlap is
# proven.
flash_attention_mixed_pipeline: bool = (
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE", "0") == "1"
)
# Conservative default is serial double buffering.  Set only when validating a
# Foundation/DXP build that accepts rows containing both data-op and DL-op work.
flash_attention_mixed_pipeline_overlap: bool = (
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP", "0") == "1"
)

# --- Tier 0: ring-aware restickify (telemetry + producer-aligned work division) ---
# Default-off ring byte-hop telemetry for compiler-inserted restickifies.
restickify_ring_telemetry: bool = (
    os.environ.get("SPYRE_RESTICKIFY_RING_TELEMETRY", "0") == "1"
)
restickify_ring_telemetry_jsonl: str = os.environ.get(
    "SPYRE_RESTICKIFY_RING_TELEMETRY_JSONL", ""
)
# Stage 2: align a restickify's physical core mapping to its producer's.
align_restickify_core_mapping: bool = (
    os.environ.get("SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING", "0") == "1"
)
# Stage 3B: steer a restickify's work-division split to the producer's dim.
align_restickify_work_distribution: bool = (
    os.environ.get("SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION", "0") == "1"
)
# Assert (rather than skip) when a locality override cannot be certified.
restickify_locality_assert: bool = (
    os.environ.get("SPYRE_RESTICKIFY_LOCALITY_ASSERT", "0") == "1"
)

# --- Tier 1: general same-layout cross-core on-chip handoff planner ---
# Default-off planner: detect same-stick producer->consumer edges that re-partition
# across cores and would otherwise spill to HBM. Plans + telemetry only; realizing
# the on-chip transfer needs the deeptools Foundation contract, so this fail-closes.
onchip_handoff_planner: bool = (
    os.environ.get("SPYRE_ONCHIP_HANDOFF_PLANNER", "0") == "1"
)
onchip_handoff_telemetry_jsonl: str = os.environ.get(
    "SPYRE_ONCHIP_HANDOFF_TELEMETRY_JSONL", ""
)
# Realize eligible same-layout handoffs as a mixed DL+data-op SuperDSC (the
# producer/consumer LX value flow) instead of fail-closing. Default off; the
# planner stays fail-closed unless this is set.
onchip_handoff_realize: bool = (
    os.environ.get("SPYRE_ONCHIP_HANDOFF_REALIZE", "0") == "1"
)
onchip_attention_score_handoff: bool = (
    os.environ.get("SPYRE_ONCHIP_ATTENTION_SCORE_HANDOFF", "0") == "1"
)
onchip_static_matmul_handoff: bool = (
    os.environ.get("SPYRE_ONCHIP_STATIC_MATMUL_HANDOFF", "0") == "1"
)
onchip_handoff_min_bytes: int = int(
    os.environ.get("SPYRE_ONCHIP_HANDOFF_MIN_BYTES", str(1 << 20))
)

install_config_module(sys.modules[__name__])
