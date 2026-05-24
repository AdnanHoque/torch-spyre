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

"""Force the runtime to load the SPLICED on-chip bundle by redirecting the
fused kernel's runner to $WORK_DIR/spliced-roundtrip (a fresh code_dir the
per-process g_artifact_cache has never seen -> real disk load of the spliced
senprog). 2048 cross-core round trip."""

import os

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr

_WORK_DIR = os.environ.get("WORK_DIR", "/tmp")
SPLICED = os.environ.get("ONCHIP_DIR", f"{_WORK_DIR}/spliced-roundtrip")
_orig = kr.SpyreSDSCKernelRunner.__init__


def _patched(self, name, code_dir):
    _orig(self, name, code_dir)
    if "mm" in name.lower():  # the (a+b.t+c.t)@d fused add+matmul kernel
        print(f"[REDIRECT] {name}: {code_dir} -> {SPLICED}", flush=True)
        self.code_dir = SPLICED


kr.SpyreSDSCKernelRunner.__init__ = _patched

DEVICE = "spyre"
S = 2048


def f(a, b, c, d):
    return (a + b.t() + c.t()) @ d


torch.manual_seed(0)
cpu = [torch.randn(S, S, dtype=torch.float16) * 0.1 for _ in range(4)]
ref = f(*cpu).float()
dev = [t.to(DEVICE) for t in cpu]
torch._dynamo.reset()
_ind.fx_graph_cache = False
cf = torch.compile(f, backend="inductor")
out = cf(*dev).cpu().float()
torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
print("DIRECT_VALIDATE_OK max_err", (out - ref).abs().max().item())
