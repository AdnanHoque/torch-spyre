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
from typing import Literal

from torch.utils._config_module import install_config_module
from .logging_utils import _get_env_bool

lx_planning: bool = os.environ.get("LX_PLANNING", "1") == "1"
co_optimizing_lx_planning: bool = (
    os.environ.get("CO_OPTIMIZING_LX_PLANNING", "0") == "1"
)
hbm_planning: bool = _get_env_bool("SPYRE_INDUCTOR_MEMORY_PLAN", True)

global_stick_optimizer: bool = os.environ.get("GLOBAL_STICK_OPTIMIZER", "1") == "1"

allow_all_ops_in_lx_planning: bool = False

# Opt-in extension to LX planning: materialize compatible producer and
# consumer per-core views as an explicit S1 -> SHUFFLE -> S2 sequence in LX.
lx_planner_relayout: bool = os.environ.get("SPYRE_LX_PLANNER_RELAYOUT", "0") == "1"

# Comma-separated collective kinds accepted by the LX relayout planner.  This
# is primarily an A/B and cost-model-development control; all supported kinds
# remain enabled by default.
lx_relayout_collectives: str = os.environ.get(
    "SPYRE_LX_RELAYOUT_COLLECTIVES", "all_to_all,all_gather,broadcast"
)

# Optional comma-separated buffer names whose producer -> consumer transport
# is excluded from LX relayout planning.  This is an edge-isolation control:
# unlike SPYRE_LX_RELAYOUT_COLLECTIVES it leaves unrelated collectives enabled.
lx_relayout_disabled_sources: str = os.environ.get(
    "SPYRE_LX_RELAYOUT_DISABLED_SOURCES", ""
)

# Reject dense all-to-all candidates above this total producer size when set
# to a non-negative byte count.  The default preserves existing behavior.
lx_relayout_all_to_all_max_bytes: int = int(
    os.environ.get("SPYRE_LX_RELAYOUT_ALL_TO_ALL_MAX_BYTES", "-1")
)

# Test-only SenDNN replay oracle for Granite B1/S512.  This deliberately
# hard-codes one graph-local producer/consumer edge; it is not production
# policy and exists only to measure an exact counterfactual topology.
relayout_oracle_prefill_output_projection: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_PREFILL_OUTPUT_PROJ", "0") == "1"
)

# Replay the shared RMSNorm activation -> SwiGLU gate/up projections. SenDNN
# uses an 8x4 (mb x hidden/output) producer layout, then a four-core grouped
# all-gather into two 8x4 BMM consumers.
relayout_oracle_prefill_mlp_inputs: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS", "0") == "1"
)

# Test-only SenDNN replay oracle that preserves Granite GQA K/V in compact
# [KV-head, query-group] form through the attention BMMs.
relayout_oracle_compact_gqa: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_COMPACT_GQA", "0") == "1"
)

# Test-only SenDNN P06 replay oracle.  Preserve the Granite prefill query in
# 8 token cohorts x 4 query-head cohorts through projection and rotary, then
# let the QK consumer gather the four head fragments for each 16-token shard.
relayout_oracle_prefill_qk_query: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY", "0") == "1"
)

# Optional comma-separated subset used to isolate the P06 producer chain.
# The default keeps the complete replay behavior.
relayout_oracle_prefill_qk_query_buffers: str = os.environ.get(
    "SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY_BUFFERS",
    "buf11,buf12,buf13,buf14",
)

relayout_oracle_prefill_qk_head_fast_emission: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_FAST_EMISSION", "1")
    == "1"
)

relayout_oracle_prefill_qk_axis_bridge: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_PREFILL_QK_AXIS_BRIDGE", "1") == "1"
)

# Test-only local QK schedule probe.  Divide the exact Granite B1/S512
# compact-GQA score BMM by KV head x query group (8 x 4) instead of by query
# token (32).  Each four-core GQA cohort then shares only its own compact K
# head, avoiding full-K replication on every core without changing the math.
relayout_oracle_prefill_qk_head_owned: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED", "0") == "1"
)

# Test-only value probe for the P06 transport geometry.  The named source is
# forced to 8 token cohorts x 4 head cohorts, while the named consumer is
# forced to 32 token shards so an identity BMM can expose the relayout output.
relayout_oracle_p06_ramp: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_P06_RAMP", "0") == "1"
)
relayout_oracle_p06_ramp_source: str = os.environ.get(
    "SPYRE_RELAYOUT_ORACLE_P06_RAMP_SOURCE", "buf0"
)
relayout_oracle_p06_ramp_consumer: str = os.environ.get(
    "SPYRE_RELAYOUT_ORACLE_P06_RAMP_CONSUMER", "buf1"
)

# Diagnostic isolation for relayout replay experiments.  Keep the allocator's
# solved LX address map (including relayout S1/S2 storage), but do not realize
# graph-input LX placements as clone operations.  This distinguishes relayout
# transport failures from unrelated input-pinning side effects.
relayout_oracle_no_input_pinning: bool = (
    os.environ.get("SPYRE_RELAYOUT_ORACLE_NO_INPUT_PINNING", "0") == "1"
)

dxp_lx_frac_avail: float = float(os.environ.get("DXP_LX_FRAC_AVAIL", "0.2"))

sencores: int = int(os.getenv("SENCORES", "32"))

# Symbolic-dim knobs consumed by compute_granularity in pass_utils.py.
# The pointwise work-division PR (#2499) wires that helper into the
# compilation pipeline; until then these knobs are read only by the
# helper and its unit tests. See #2284, #2287 for the design.

# Cap on bucket count (= max_size / granularity).
# TODO: confirm the default with the Deeptools team.
max_buckets: int = int(os.getenv("MAX_BUCKETS", "32"))

# Soft floor on the auto-derived granularity when mark_dynamic(min=...)
# is not provided. Keeps the picked granularity from collapsing to a
# very small divisor when max_size has many of them.
min_default_granularity: int = int(os.getenv("MIN_DEFAULT_GRANULARITY", "4"))

ignore_work_division_hints: bool = (
    os.environ.get("SPYRE_INDUCTOR_IGNORE_HINTS", "0") == "1"
)

ignore_wsr_hints: bool = os.environ.get("SPYRE_INDUCTOR_IGNORE_HINTS", "0") == "1"

# Per-pass operation logging for CustomPreSchedulingPasses.
# Set to "all" or "1" to log after every pass, or a comma-separated list of
# pass function names (e.g., "split_multi_ops,insert_restickify") to log only
# after specific passes. Set via SPYRE_LOG_PASSES env var or programmatically.
log_passes: str = os.environ.get("SPYRE_LOG_PASSES", "")

# Disable compiler-generated span-overflow coarse-tiling hints.  The global
# SPYRE_INDUCTOR_IGNORE_HINTS flag also disables these so one switch can still
# suppress all WSR/coarse-tiling hint paths.
#
# Defaults to disabled (opt-in): span-overflow auto-tiling can synchronize
# compatible contiguous pointwise groups, but incompatible producer/consumer
# groups and reduction-dim tiling still need broader support. Set
# SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS=0 to opt in;
# tests exercising this path directly should override via
# config.patch({"ignore_span_overflow_hints": False}).
ignore_span_overflow_hints: bool = (
    ignore_wsr_hints
    or os.environ.get("SPYRE_INDUCTOR_IGNORE_SPAN_OVERFLOW_HINTS", "1") == "1"
)

# For K-split matmuls, permute physical core IDs so the cores collaborating on a
# K reduction land on adjacent ring positions, cutting PSUM chain hops from m*n
# to 1. The split itself is chosen by the cost-model planner; this only reorders
# cores at SDSC emission. Set SPYRE_CORE_ID_K_FAST_EMISSION=0 to disable.
core_id_k_fast_emission: bool = (
    os.environ.get("SPYRE_CORE_ID_K_FAST_EMISSION", "1") == "1"
)

# When True (default), HBM tensor addresses are emitted as runtime symbols
# with !sdscbundle.input_arg<index> parameters and input_arg_extract ops
# in the bundle.mlir.
# When False, HBM tensor addresses are baked as concrete integers
# into the SDSC JSON and bundle.mlir emits sdsc_execute with no operands.
bundle_symbolic_args: bool = os.environ.get("BUNDLE_SYMBOLIC_ARGS", "1") == "1"

# Layout solver class used by default in scratchpad.allocator.ScratchpadAllocator.
# Options:
#  "greedy":       GreedyLayoutSolver (default),
#  "bestfit":      BestFitLayoutSolver,
#  "firstfit":     FirstFitLayoutSolver,
#  "simulated_annealing":  SimulatedAnnealingLayoutSolver,
#  "cpsat":    CpSatLayoutSolver (OR-Tools CP-SAT joint core-division +
#              LX placement, minimizing HBM transfer traffic).

# TODO(isuruf): Change to firstfit when deeptools PR4298 lands
layout_solver: Literal[
    "greedy", "bestfit", "firstfit", "cpsat", "simulated_annealing"
] = os.environ.get("LAYOUT_SOLVER", "greedy")  # type: ignore[assignment]

install_config_module(sys.modules[__name__])
