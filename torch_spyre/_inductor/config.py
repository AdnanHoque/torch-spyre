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

# When enabled, mark graph-input tensors that look like nn.Parameters
# (placeholder tensor_meta.requires_grad=True) as static via the SDSC
# JSON `labeledDs_.isStatic_` field. This is the codegen half of the
# cross-call weight preload pipeline; runtime separation is a separate
# concern. Off by default — Phase 3 diagnostic spike, not production.
preload_static: bool = os.environ.get("SPYRE_PRELOAD_STATIC", "0") == "1"

install_config_module(sys.modules[__name__])
