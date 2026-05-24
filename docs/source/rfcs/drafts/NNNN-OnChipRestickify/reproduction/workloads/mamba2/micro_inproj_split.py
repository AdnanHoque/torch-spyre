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

"""Microbench: in-proj fused output -> x/B/C/z/dt slice consumers.

In-proj is one big linear; its output is split into 5 tensors. Each slice is a
producer-output -> consumer handoff. Slices on the d_model axis preserve
stick=['out'] (same-stick); each feeds an elementwise/conv consumer. This is
the in-proj -> {conv, scan} fan-out handoff. We model the split as 4 mm gathers
keeping the handoff matmul-shaped (clean compile). ONE config/process.

A/B via SPLICED_DIR. Tunable: SEQLEN (1=decode), D_MODEL.
"""

import os

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr

from _bench_common import bench

DEV = "spyre"
SEQ = int(os.environ.get("SEQLEN", "256"))
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


def inproj(x, w_in, w_x, w_z):
    proj = x @ w_in  # in-proj fused output
    xs = F.silu(proj @ w_x)  # x slice -> conv/activation consumer
    z = proj @ w_z  # z gating slice consumer
    return xs * z  # gate handoff


def main():
    torch.manual_seed(0)
    s = 0.05
    x = torch.randn(SEQ, DMODEL, dtype=DT) * s
    w_in = torch.randn(DMODEL, 4 * DMODEL, dtype=DT) * s
    w_x = torch.randn(4 * DMODEL, DMODEL, dtype=DT) * s
    w_z = torch.randn(4 * DMODEL, DMODEL, dtype=DT) * s
    cpu = [x, w_in, w_x, w_z]
    ref = inproj(*cpu).float()
    dev = [t.to(DEV) for t in cpu]
    lbl = f"spliced={SPLICED}" if SPLICED else "baseline_HBM"
    bench(inproj, dev, ref, f"inproj_split seq={SEQ} d_model={DMODEL} {lbl}")


if __name__ == "__main__":
    main()
