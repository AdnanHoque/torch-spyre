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

"""End-to-end COMPILER-DRIVEN on-chip handoff test (no splicing, no redirect).

With SPYRE_ONCHIP_HANDOFF_REALIZE=1 the inductor backend itself emits the mixed
DL+data-op SuperDSC during generate_bundle; torch_spyre's own async_compile then
invokes dxp_standalone (the PATCHED gate-relaxed binary, put first on PATH) and
the runtime loads the compiler's own code_dir. We validate value-correctness AND
confirm the COMPILER emitted the mixed bundle (datadscs_ in the consumer SDSC).
Flag off => byte-identical baseline (no datadscs_), the negative control.

Requires (set by e2e_onchip.sh): the patched dxp first on PATH, PYTHONPATH at the
val-boot shim (imports torch_spyre from the tier0-tier1-onchip worktree), and a
fresh TORCHINDUCTOR_CACHE_DIR. See reproduction/env.sh.
"""

import glob
import json
import os

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch_spyre  # noqa: F401

S = int(os.environ.get("E2E_SIZE", "2048"))
CACHE = os.environ["TORCHINDUCTOR_CACHE_DIR"]
MODE = "REALIZE" if os.environ.get("SPYRE_ONCHIP_HANDOFF_REALIZE") == "1" else "BASE"


def f(a, b, c, d):
    return (a + b.t() + c.t()) @ d


def main():
    torch.manual_seed(0)
    cpu = [torch.randn(S, S, dtype=torch.float16) * 0.1 for _ in range(4)]
    ref = f(*cpu).float()
    dev = [t.to("spyre") for t in cpu]
    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    cf = torch.compile(f, backend="inductor")
    out = cf(*dev).cpu().float()
    err = (out - ref).abs().max().item()

    # Inspect the compiler's own code_dir: did generate_bundle emit datadscs_?
    c2 = glob.glob(f"{CACHE}/**/sdsc_2_add.json", recursive=True)
    emitted = False
    opfuncs = None
    if c2:
        doc = json.load(open(c2[0]))
        body = doc[next(iter(doc))]
        emitted = "datadscs_" in body and bool(body["datadscs_"])
        opfuncs = body.get("opFuncsUsed_")
    print(
        f"E2E mode={MODE} max_err={err:.6f} compiler_emitted_mixed={emitted} "
        f"opFuncsUsed_={opfuncs}",
        flush=True,
    )


if __name__ == "__main__":
    main()
