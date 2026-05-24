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

"""Compile (a+b.t()+c.t())@d at BENCH_SIZE, run once to populate the code_dir,
then report the code_dir path + the add->add edge sharding so the splice can be
parameterized per size."""

import glob
import json
import os

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch_spyre  # noqa: F401

S = int(os.environ["BENCH_SIZE"])
CACHE = os.environ["TORCHINDUCTOR_CACHE_DIR"]


def f(a, b, c, d):
    return (a + b.t() + c.t()) @ d


torch.manual_seed(0)
cpu = [torch.randn(S, S, dtype=torch.float16) * 0.1 for _ in range(4)]
ref = f(*cpu).float()
dev = [t.to("spyre") for t in cpu]
torch._dynamo.reset()
_ind.fx_graph_cache = False
cf = torch.compile(f, backend="inductor")
out = cf(*dev).cpu().float()
err = (out - ref).abs().max().item()

bm = glob.glob(f"{CACHE}/**/bundle.mlir", recursive=True)
cd = os.path.dirname(bm[0]) if bm else "NONE"
sdscs = sorted(os.path.basename(p) for p in glob.glob(f"{cd}/sdsc_*.json"))
info = {"size": S, "max_err": err, "code_dir": cd, "sdscs": sdscs}
# inspect the consumer add (sdsc_2_add) sharding + iter sizes
c2 = os.path.join(cd, "sdsc_2_add.json")
if os.path.exists(c2):
    doc = json.load(open(c2))
    body = doc[list(doc)[0]]
    info["sdsc_2_numWkSlicesPerDim_"] = body.get("numWkSlicesPerDim_")
    dl = body["dscs_"][0]
    dl = dl[list(dl)[0]]
    # layout sizes from the first labeledDs
    lds0 = dl["labeledDs_"][0]
    info["dimToLayoutSize_"] = lds0.get("dimToLayoutSize_")
    info["layoutDimOrder_"] = lds0.get("layoutDimOrder_")
print("GENBASE " + json.dumps(info), flush=True)
