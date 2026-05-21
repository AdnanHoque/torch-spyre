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

allow_all_ops_in_lx_planning: bool = (
    os.environ.get("SPYRE_ALLOW_ALL_OPS_IN_LX_PLANNING", "0") == "1"
)

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

# Default-off diagnostic prototype for emitting restickify-like movement as a
# Deeptools data-op DSC (`datadscs_`) instead of the normal compute-op SDSC.
# This is not used by production lowering; it exists to probe STCDPOpLx /
# ReStickifyOpLx / ReStickifyOpHBM contracts in isolation.
restickify_lx_dataop: bool = (
    os.environ.get("SPYRE_RESTICKIFY_LX_DATAOP", "0") == "1"
)

# Default-off diagnostic e2e lowering prototype. When enabled, bundle emission
# may replace a certified local, LX-resident `ReStickifyOpHBM` compute SDSC with
# a `ReStickifyOpLx` data-op SDSC. This is deliberately narrower than the
# standalone probe and skips anything HBM-backed or uncertified.
restickify_lx_dataop_e2e: bool = (
    os.environ.get("SPYRE_RESTICKIFY_LX_DATAOP_E2E", "0") == "1"
)

restickify_lx_dataop_audit_jsonl: str = os.environ.get(
    "SPYRE_RESTICKIFY_LX_DATAOP_AUDIT_JSONL",
    "",
)

# Default-off normal-bundle prototype for replacing an adjacent
# producer -> ReStickifyOpHBM -> consumer edge with the PT-aware LX data-op
# bridge proven by the same-artifact splice. Unlike the splice hook, this runs
# before bundle files and runtime frames are generated.
restickify_ptlx_bridge_e2e: bool = (
    os.environ.get("SPYRE_RESTICKIFY_PTLX_BRIDGE_E2E", "0") == "1"
)

restickify_ptlx_bridge_audit_jsonl: str = os.environ.get(
    "SPYRE_RESTICKIFY_PTLX_BRIDGE_AUDIT_JSONL",
    "",
)

# Default-off mixed-schedule prototype for the PT-aware LX restickify bridge.
# When enabled, eligible adjacent `ReStickifyOpHBM -> consumer` pairs are
# emitted as one mixed SuperDsc containing the bridge data ops and the consumer
# DL op with an explicit coreIdToDscSchedule. This is the production-shaped
# follow-up to the Stage198 probe, but it still requires Deeptools/DXP mixed
# SuperDsc bundle support before it can launch.
restickify_ptlx_mixed_schedule_e2e: bool = (
    os.environ.get("SPYRE_RESTICKIFY_PTLX_MIXED_SCHEDULE_E2E", "0") == "1"
)

# Default-off verifier for the PT-aware LX mixed bridge. When enabled, the
# generated producer output, bridge endpoints, and consumer input must agree on
# the same per-core LX addresses before bundle files are written.
restickify_ptlx_value_flow_assert: bool = (
    os.environ.get("SPYRE_RESTICKIFY_PTLX_VALUE_FLOW_ASSERT", "0") == "1"
)

# Default-off diagnostic e2e lowering prototype for the Stage42 DDL bridge.
# When enabled, a small, compile-proven subset of ReStickifyOpHBM SDSCs may be
# emitted in the compact restickify DDL input form instead of the normal HBM
# compute SDSC. Unsupported restickifies stay on the existing path.
restickify_ddl_bridge_e2e: bool = (
    os.environ.get("SPYRE_RESTICKIFY_DDL_BRIDGE_E2E", "0") == "1"
)

restickify_ddl_bridge_audit_jsonl: str = os.environ.get(
    "SPYRE_RESTICKIFY_DDL_BRIDGE_AUDIT_JSONL",
    "",
)

# Default-off boundary prototype for the DDL bridge. When enabled, adjacent
# producer -> ReStickifyOpHBM_ddl_bridge -> consumer triples are patched as one
# internal LX edge: producer output, bridge endpoints, and consumer input are
# marked LX-only so the consumer no longer lowers that logical input as an HBM
# reload. This is still experimental and intentionally requires the DDL bridge
# flag above.
restickify_ddl_bridge_boundary_patch: bool = (
    os.environ.get("SPYRE_RESTICKIFY_DDL_BRIDGE_BOUNDARY_PATCH", "0") == "1"
)

# Prototype-only runtime hook for the DDL bridge. The installed DXP currently
# runs generic corelet splitting and L3 scheduling before DDC, which rejects the
# compact restickify DDL input. When this is enabled, Torch-Spyre applies a
# selective preload shim for bundles containing a DDL-bridge restickify. The
# shim bypasses those two pre-DDC steps only for the bridge SDSC and delegates
# normal SDSCs back to Deeptools.
restickify_ddl_bridge_preddc_shim: bool = (
    os.environ.get("SPYRE_RESTICKIFY_DDL_BRIDGE_PREDDC_SHIM", "1") == "1"
)

# Default-off integration prototype for Stage 121. When enabled, bundle
# generation emits a sidecar descriptor for eligible adjacent
# producer/restickify/consumer triples that can be handed to the
# InputFetchNeighbor LX-to-LX packaging path.
restickify_lx_neighbor_descriptor: bool = (
    os.environ.get("SPYRE_RESTICKIFY_LX_NEIGHBOR_DESCRIPTOR", "0") == "1"
)

# Default-off producer-consumer ownership telemetry. This generalizes the
# restickify byte-hop estimator to all exact in-graph tensor edges, without
# changing core division or codegen behavior.
core_continuity_telemetry: bool = (
    os.environ.get("SPYRE_CORE_CONTINUITY_TELEMETRY", "0") == "1"
)

core_continuity_telemetry_jsonl: str = os.environ.get(
    "SPYRE_CORE_CONTINUITY_TELEMETRY_JSONL",
    "",
)

# Default-off source fanout telemetry for read-only graph inputs, weights,
# constants, and other external sources. This is attribution-only; it does not
# emit GTR/multicast metadata.
input_fanout_telemetry: bool = (
    os.environ.get("SPYRE_INPUT_FANOUT_TELEMETRY", "0") == "1"
)

input_fanout_telemetry_jsonl: str = os.environ.get(
    "SPYRE_INPUT_FANOUT_TELEMETRY_JSONL",
    "",
)

# Default-off prototype for preserving producer-consumer core ownership across
# exact in-graph pointwise edges. This is intentionally narrower than the
# telemetry and does not change matmul/reduction distribution.
align_core_division_continuity: bool = (
    os.environ.get("SPYRE_ALIGN_CORE_DIVISION_CONTINUITY", "0") == "1"
)

# Default-off mapping-only producer-consumer continuity prototype. This keeps
# work split factors unchanged and only attaches certified core mapping
# overrides when producer/consumer split factors already match.
align_core_mapping_continuity: bool = (
    os.environ.get("SPYRE_ALIGN_CORE_MAPPING_CONTINUITY", "0") == "1"
)

install_config_module(sys.modules[__name__])
