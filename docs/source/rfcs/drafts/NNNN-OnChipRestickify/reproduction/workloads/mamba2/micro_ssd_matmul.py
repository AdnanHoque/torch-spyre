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

"""Microbench: the SSD chunked-matmul -> matmul handoff (C@B -> decay -> @x).

The Mamba-2 SSD dual is two stacked matmuls with an elementwise mask between:
scores = (C @ B.t) * decay  (bmm output, stick=['out']); scan = scores @ x.
The bmm-output -> mul -> bmm chain crosses an SDSC boundary same-stick but
re-sharded (bmm shards out/in, mul reshards mb) -> genuine cross-core ring,
exactly the proven STCDP-today case. Highest-recurrence matmul handoff.

A/B via SPLICED_DIR (like bench_onchip.py): empty -> baseline HBM bundle;
a dir -> redirect mm runner to spliced on-chip bundle. ONE config/process.
"""

import os

import torch
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr

from _bench_common import bench

DEV = "spyre"
SEQ = int(os.environ.get("SEQLEN", "256"))
D_STATE = int(os.environ.get("D_STATE", "64"))
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


def ssd(c, b, x, decay):
    scores = (c @ b.t()) * decay  # bmm out -> mul (same-stick, cross-core)
    return scores @ x  # mul -> bmm scan


def main():
    torch.manual_seed(0)
    s = 0.05
    c = torch.randn(SEQ, D_STATE, dtype=DT) * s
    b = torch.randn(SEQ, D_STATE, dtype=DT) * s
    x = torch.randn(SEQ, DMODEL, dtype=DT) * s
    decay = torch.tril(torch.ones(SEQ, SEQ, dtype=DT))
    cpu = [c, b, x, decay]
    ref = ssd(*cpu).float()
    dev = [t.to(DEV) for t in cpu]
    lbl = f"spliced={SPLICED}" if SPLICED else "baseline_HBM"
    bench(ssd, dev, ref, f"ssd_matmul seq={SEQ} d_state={D_STATE} {lbl}")


if __name__ == "__main__":
    main()
