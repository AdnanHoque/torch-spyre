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

# Phase 0 telemetry for ring-aware restickify (RFC draft at
# docs/source/rfcs/drafts/NNNN-RingAwareRestickify). When True, a
# diagnostic pass runs after work_distribution and logs each restickify
# with its producer/consumer split mapping. Read-only; no behavior change.
restickify_telemetry: bool = os.environ.get("SPYRE_RESTICKIFY_TELEMETRY", "0") == "1"

# Phase 1 alignment for ring-aware restickify. When True, work_distribution
# re-prioritises an op's output_dims so iteration symbols that physically
# index the same buffer dim as a producer's split symbol come first —
# biasing the consumer's split to land on the same physical axis as the
# producer's, which eliminates inter-core ring traffic in the intervening
# restickify. Off by default while we validate.
align_consumer_splits: bool = os.environ.get("SPYRE_ALIGN_CONSUMER_SPLITS", "0") == "1"

# Alternative 1.5 in the v2 Ring-Aware Restickify RFC: when True, the
# codegen-time op-func name for an explicit restickify is swapped from
# `ReStickifyOpHBM` to `STCDPOpLx` (on-chip RIU BiRing shuffle) when the
# restickify classifier (torch_spyre/_inductor/restickify_classify.py)
# returns FUNDAMENTAL. Today STCDPOpLx silently no-ops in the deeptools
# bundle pipeline (DDL template missing in DDC); the gate is therefore a
# pre-staging mechanism for the moment deeptools ships the primitive.
# Off by default -- enabling without deeptools support produces silent
# wrong output for affected kernels.
emit_stcdp_oplx: bool = os.environ.get("SPYRE_EMIT_STCDP_OPLX", "0") == "1"

install_config_module(sys.modules[__name__])
