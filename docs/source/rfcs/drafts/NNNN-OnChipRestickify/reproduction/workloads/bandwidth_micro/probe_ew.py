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

"""Probe: compile the elementwise producer->consumer graph and report its bundle.

f(x) = (x * 2) * 3, fp16. The intermediate m = x*2 is the producer->consumer
handoff. We compile it and let the caller inspect the generated SDSC bundle dir
(set TORCHINDUCTOR_CACHE_DIR). Prints whether it is one fused kernel or two.
"""

import os

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch_spyre  # noqa: F401

ROWS = int(os.environ.get("EW_ROWS", "512"))
COLS = int(os.environ.get("EW_COLS", "2048"))
DEVICE = "spyre"


def f(x):
    m = x * 2.0
    return m * 3.0


def main():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    cpu = torch.randn(ROWS, COLS, dtype=torch.float16, generator=g) * 0.1
    dev = cpu.to(DEVICE)
    ref = (cpu.float() * 2.0) * 3.0

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(f, backend="inductor")
    out = compiled(dev).cpu().float()
    max_err = (out - ref).abs().max().item()
    print(f"PROBE_EW rows={ROWS} cols={COLS} max_err={max_err:.6f}", flush=True)


if __name__ == "__main__":
    main()
