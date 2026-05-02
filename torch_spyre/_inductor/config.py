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

# Phase 2 K-split heuristic — see tests/splitk_phase1_findings.md.
# When enabled, prioritize_dimensions rotates the reduction (K) dim ahead of
# output dims for matmul iteration spaces that satisfy all of:
#   - product of output dim iter-space sizes < k_split_max_output_iter_units
#   - max reduction dim iter-space size >= k_split_min_k_iter_units
#   - reduction dim iter-space size is divisible by sencores (clean num_cores
#     -way K-split on stick boundaries)
# Iter-space units are post-stick-adjustment: stick dims are in stick counts
# (e.g. fp16 stick = 64 elements). Defaults are calibrated for fp16 matmul
# from Phase 1 measurements:
#   - max_output 32768 = M_elems x N_sticks bound matching the M*N <= 1M-elem
#     forceK-wins regime (e.g. (128, 8192, K) -> 128 * 128 sticks = 16K).
#   - min_k 64 sticks = 4096 elements, the K threshold below which
#     K-parallelism gain is too small to offset cross-core reduction.
k_split_heuristic: bool = (
    os.environ.get("TORCH_SPYRE_K_SPLIT_HEURISTIC", "0") == "1"
)
k_split_max_output_iter_units: int = int(
    os.environ.get("TORCH_SPYRE_K_SPLIT_MAX_OUTPUT_ITER", "32768")
)
k_split_min_k_iter_units: int = int(
    os.environ.get("TORCH_SPYRE_K_SPLIT_MIN_K_ITER", "64")
)

install_config_module(sys.modules[__name__])
