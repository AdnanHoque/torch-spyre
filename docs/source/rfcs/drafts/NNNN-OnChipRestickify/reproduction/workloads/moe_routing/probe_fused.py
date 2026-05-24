# Copyright 2026 The Torch-Spyre Authors.
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
"""Probe: does a fused router->dispatch (or dispatch->consumer) graph create a
multi-SDSC bundle with a real producer->consumer HBM-handoff edge?"""

import os
import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch_spyre  # noqa: F401

DEVICE = "spyre"
E = int(os.environ.get("MOE_E", "8"))
T = int(os.environ.get("MOE_T", "512"))
H = int(os.environ.get("MOE_H", "2048"))
CAP = max(1, (T + E - 1) // E)
EC = E * CAP


def build():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(T, H, dtype=torch.float16, generator=g) * 0.1
    perm = torch.zeros(EC, T, dtype=torch.float16)
    for r in range(EC):
        perm[r, (r * 7 + 3) % T] = 1.0
    wexp = torch.randn(H, H, dtype=torch.float16, generator=g) * 0.02
    return x, perm, wexp


# dispatch then a per-expert-ish linear: (perm @ x) @ wexp -- output feeds a matmul
def f_dispatch_then_linear(perm, x, wexp):
    return (perm @ x) @ wexp


def main():
    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    x, perm, wexp = build()
    args = [perm.to(DEVICE), x.to(DEVICE), wexp.to(DEVICE)]
    cf = torch.compile(f_dispatch_then_linear, backend="inductor")
    out = cf(*args).cpu().float()
    ref = f_dispatch_then_linear(perm, x, wexp).float()
    print("max_err", (out - ref).abs().max().item(), flush=True)


if __name__ == "__main__":
    main()
