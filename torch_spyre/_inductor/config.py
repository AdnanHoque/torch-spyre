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
# Production-candidate umbrella for the certified on-chip SDPA path.  This
# enables the generated flash-prefill decomposition plus fail-closed same-stick
# handoffs inside that graph.  It intentionally does not enable overlap,
# sidecar artifact emission, or tile replacement by default; those are still
# individual probe/debug gates.
flash_attention_onchip_sdpa: bool = (
    os.environ.get("SPYRE_FLASH_ATTENTION_ONCHIP_SDPA", "0") == "1"
)
flash_attention_onchip_sdpa_layout_xform: bool = (
    flash_attention_onchip_sdpa
    and os.environ.get("SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM", "0")
    == "1"
)
flash_attention_prefill_block_size: int = int(
    os.environ.get(
        "SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE",
        "512" if flash_attention_onchip_sdpa else "128",
    )
)
# Default-off proof path for a mixed-SDSC, double-buffered flash-attention
# prefill pipeline.  The first implementation only builds descriptor/scheduler
# proof artifacts; compiler promotion remains gated until device overlap is
# proven.
flash_attention_mixed_pipeline: bool = (
    flash_attention_onchip_sdpa
    or os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE", "0") == "1"
)
# Conservative default is serial double buffering. Set only for Foundation/DXP
# contract probes that intentionally emit rows containing both data-op and DL-op
# work; production execution keeps this off until that row shape is certified.
flash_attention_mixed_pipeline_overlap: bool = (
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_OVERLAP", "0") == "1"
)
# Emit a compiler-produced mixed-SDSC flash pipeline proof artifact next to the
# normal SDSCs, but do not add it to bundle.mlir execution. This lets DXP/senprog
# validation target the real generated flash-prefill graph before the production
# path replaces any executed SDSCs.
flash_attention_mixed_pipeline_artifact: bool = (
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_ARTIFACT", "0") == "1"
)
# Execute one generated flash-prefill batchmatmul tile through its mixed sidecar
# instead of the original SDSC.  -1 keeps sidecars non-executed.  0 executes the
# first batchmatmul tile in each generated flash-prefill bundle, etc.
flash_attention_mixed_pipeline_execute_tile: int = int(
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_EXECUTE_TILE", "-1")
)
# Stronger diagnostic than EXECUTE_TILE: flip eligible single-consumer producer
# outputs and matching flash-tile batchmatmul inputs to LX, then execute the
# mixed tile sidecar with real STCDPOpLx value flow. -1 disables.
flash_attention_mixed_pipeline_value_flow_tile: int = int(
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_VALUE_FLOW_TILE", "-1")
)
# Strongest Stage039 value-flow experiment: replace a strict
# producer->single-consumer, same-physical-layout edge with two ordered sidecar
# SDSCs.  The producer leaves its output in LX, and the consumer runs an explicit
# STCDPOpLx copy into its input LX before compute. -1 disables; layout-changing
# edges remain fail-closed.
flash_attention_mixed_pipeline_ifn_pair_tile: int = int(
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PAIR_TILE", "-1")
)
# Default-off diagnostic gate for the IFN-attached overlap-prefix tile.  The
# artifact is normally emitted but not executed because it has no real
# predecessor-backed input yet; this flag forces execution so pod probes can
# expose the next Foundation/DXP blocker for AIU warp-specialized prefill.
flash_attention_mixed_pipeline_ifn_prefix_force: bool = (
    os.environ.get("SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_IFN_PREFIX_FORCE", "0")
    == "1"
)
# Experimental Stage039 follow-up for real SDPA edges that have a strict
# producer->single-consumer relation but require a same-dim layout transform
# before the consumer can read the predecessor LX payload. -1 disables; -2 scans
# for the first eligible tile; non-negative values request a concrete tile.  The
# production-shaped master adjunct selects auto mode only when both
# SPYRE_FLASH_ATTENTION_ONCHIP_SDPA=1 and
# SPYRE_FLASH_ATTENTION_ONCHIP_SDPA_LAYOUT_XFORM=1 are set; the explicit tile env
# remains the lower-level override for probe work.
flash_attention_mixed_pipeline_layout_xform_pair_tile: int = int(
    os.environ.get(
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_TILE",
        "-2" if flash_attention_onchip_sdpa_layout_xform else "-1",
    )
)
# Default-off diagnostic follow-up for the layout-transform pair: schedule the
# predecessor-backed STCDPOpLx copy in the same row as the consumer DL compute.
# This is the closest current Torch-side shape to AIU warp-overlap, but remains
# probe-only until value correctness is proven.
flash_attention_mixed_pipeline_layout_xform_pair_overlap: bool = (
    os.environ.get(
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_PAIR_OVERLAP", "0"
    )
    == "1"
)
# Default-off value-plausible successor to the same-input overlap diagnostic:
# copy the current layout-transform input before compute, then overlap current
# compute with a prefetch for a different future input whose producer is already
# available in bundle order. ``-2`` means auto-select the first legal current
# tile/future tile pair.
flash_attention_mixed_pipeline_layout_xform_lookahead_tile: int = int(
    os.environ.get(
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_LOOKAHEAD_TILE", "-1"
    )
)
# Default-off real-graph probe for the common case where a future layout-transform
# producer appears after the current batchmatmul but depends only on external or
# already-available inputs.  The future producer is hoisted into the current
# sidecar as a prologue, then its LX output is copied into the future consumer's
# input buffer while the current batchmatmul computes.
flash_attention_mixed_pipeline_layout_xform_hoist_tile: int = int(
    os.environ.get(
        "SPYRE_FLASH_ATTENTION_MIXED_PIPELINE_LAYOUT_XFORM_HOIST_TILE", "-1"
    )
)
# Default-off production-shaped bridge for same-stick pointwise edges that appear
# inside the flash-prefill graph. This keeps the attention experiment off the
# generic add/add handoff flag while reusing the same fail-closed Tier 1 realizer.
flash_attention_pointwise_handoff: bool = (
    flash_attention_onchip_sdpa
    or os.environ.get("SPYRE_FLASH_ATTENTION_POINTWISE_HANDOFF", "0") == "1"
)
# Experimental production-candidate gate for the flash score-scale edge:
#     PT batchmatmul score output -> scalar SFP mul input.
# Stage 020 certifies the PT endpoint contract through 128-wide score blocks.
# Wider blocks still fail closed in the realizer until value correctness is
# proven.
flash_attention_score_scale_handoff: bool = (
    flash_attention_onchip_sdpa
    or os.environ.get("SPYRE_FLASH_ATTENTION_SCORE_SCALE_HANDOFF", "0") == "1"
)
# Default-off debug artifact for the causal SDPA mask bring-up. When enabled,
# bundle generation writes a non-executed IdxToMask+where3 candidate plan next
# to generated SDSCs for causal_score_bias_like. It does not alter bundle.mlir.
causal_idx_to_mask_plan_artifact: bool = (
    os.environ.get("SPYRE_CAUSAL_IDX_TO_MASK_PLAN_ARTIFACT", "0") == "1"
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
