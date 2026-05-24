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

"""Microbench: scan -> gate -> gated-RMSNorm -> out-proj elementwise chain.

This is the bandwidth-bound tail. scan*silu(z), then RMSNorm, are all
elementwise / residual ops sharing stick=['out'] and shard=mb across the
chain -> same-stick SAME-shard handoffs = degenerate same-core LX->LX copy
(HBM-elim, no ring, fits at every size incl. seq=1 decode). The RMSNorm tail
has one out<->x reshape (needs-transpose, blocked). Targets the decode regime
(SEQLEN=1) where activation round-trips dominate. ONE config/process.

A/B via SPLICED_DIR; decode regime: SEQLEN=1.
"""

import os

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr

from _bench_common import bench

DEV = "spyre"
SEQ = int(os.environ.get("SEQLEN", "1"))  # decode default
DMODEL = int(os.environ.get("D_MODEL", "2048"))
SPLICED = os.environ.get("SPLICED_DIR", "").strip()
DT = torch.float16

if SPLICED:
    _orig = kr.SpyreSDSCKernelRunner.__init__

    def _patched(self, name, code_dir):
        _orig(self, name, code_dir)
        if "mm" in name.lower():
            self.code_dir = SPLICED

    kr.SpyreSDSCKernelRunner.__init__ = _patched


def tail(scan, z, rms_w, w_out):
    gate = scan * F.silu(z)  # scan -> gate (same-stick same-shard)
    n = gate * torch.rsqrt((gate * gate).mean(-1, keepdim=True) + 1e-6) * rms_w
    return n @ w_out  # norm -> out-proj


def main():
    torch.manual_seed(0)
    s = 0.05
    scan = torch.randn(SEQ, DMODEL, dtype=DT) * s
    z = torch.randn(SEQ, DMODEL, dtype=DT) * s
    rms_w = torch.randn(DMODEL, dtype=DT) * s
    w_out = torch.randn(DMODEL, DMODEL, dtype=DT) * s
    cpu = [scan, z, rms_w, w_out]
    ref = tail(*cpu).float()
    dev = [t.to(DEV) for t in cpu]
    lbl = f"spliced={SPLICED}" if SPLICED else "baseline_HBM"
    bench(tail, dev, ref, f"gate_norm_out seq={SEQ} d_model={DMODEL} {lbl}")


if __name__ == "__main__":
    main()
