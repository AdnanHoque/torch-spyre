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

"""Representative Mamba-2 (SSD) block, torch.compile(backend="inductor").

Baseline timing + CPU-correctness for one block: in-proj -> causal conv1d ->
SiLU -> chunked-matmul SSD scan -> gated RMSNorm -> out-proj. The scan is the
chunked / segment-sum "dual" (matmul-heavy) form, which is the regime where the
same-stick on-chip handoff matters. ONE config/process, OFFLINE-validated only.

SIMPLIFICATIONS (see projection.md):
- conv1d uses F.conv1d depthwise. If conv1d does not lower cleanly on Spyre,
  set MAMBA_SKIP_CONV=1 to drop the conv and exercise the rest of the chain.
- segment-sum uses a single chunk (chunk == seqlen) intra-chunk matmul; the
  inter-chunk recurrence is dropped (handoff-equivalent, not full SSD math).
- dt/A discretization folded into a precomputed L decay matrix to keep the
  graph matmul-shaped. Correctness is vs the same-decomposed CPU ref.

Env: D_MODEL, SEQLEN, HEADDIM, D_STATE, BENCH_SIZE unused (params below).
"""

import os

import torch
import torch.nn.functional as F
import torch_spyre  # noqa: F401

from _bench_common import bench

DEV = "spyre"
D_MODEL = int(os.environ.get("D_MODEL", "2048"))
SEQ = int(os.environ.get("SEQLEN", "256"))
HEADDIM = int(os.environ.get("HEADDIM", "64"))
D_STATE = int(os.environ.get("D_STATE", "64"))
SKIP_CONV = os.environ.get("MAMBA_SKIP_CONV", "0") == "1"
NHEADS = D_MODEL // HEADDIM
DT = torch.float16


def block(x, w_in, conv_w, w_x, w_b, w_c, w_z, decay, rms_w, w_out):
    # in-proj: produces fused activation, sliced into x/B/C/z (dt folded -> decay)
    proj = x @ w_in  # [S, 4*D_MODEL]
    xs = proj @ w_x  # [S, D_MODEL] gather slice (matmul keeps it a clean handoff)
    z = proj @ w_z
    bmat = proj @ w_b  # [S, D_STATE]
    cmat = proj @ w_c  # [S, D_STATE]
    if not SKIP_CONV:
        xc = F.conv1d(xs.t().unsqueeze(0), conv_w, groups=D_MODEL, padding=3)
        xc = xc[..., :SEQ].squeeze(0).t()
    else:
        xc = xs
    xc = F.silu(xc)  # conv -> activation
    cb = (cmat @ bmat.t()) * decay  # [S,S] segment-sum (single-chunk SSD dual)
    scan = cb @ xc  # [S, D_MODEL] scan-out
    gate = scan * F.silu(z)  # scan -> gate
    n = gate * torch.rsqrt((gate * gate).mean(-1, keepdim=True) + 1e-6) * rms_w
    return n @ w_out  # gated RMSNorm -> out-proj


def main():
    torch.manual_seed(0)
    s = 0.05
    x = torch.randn(SEQ, D_MODEL, dtype=DT) * s
    w_in = torch.randn(D_MODEL, 4 * D_MODEL, dtype=DT) * s
    conv_w = torch.randn(D_MODEL, 1, 4, dtype=DT) * s
    w_x = torch.randn(4 * D_MODEL, D_MODEL, dtype=DT) * s
    w_b = torch.randn(4 * D_MODEL, D_STATE, dtype=DT) * s
    w_c = torch.randn(4 * D_MODEL, D_STATE, dtype=DT) * s
    w_z = torch.randn(4 * D_MODEL, D_MODEL, dtype=DT) * s
    decay = torch.tril(torch.ones(SEQ, SEQ, dtype=DT))
    rms_w = torch.randn(D_MODEL, dtype=DT) * s
    w_out = torch.randn(D_MODEL, D_MODEL, dtype=DT) * s
    cpu = [x, w_in, conv_w, w_x, w_b, w_c, w_z, decay, rms_w, w_out]
    ref = block(*cpu).float()
    dev = [t.to(DEV) for t in cpu]
    label = f"d_model={D_MODEL} seq={SEQ} headdim={HEADDIM} d_state={D_STATE}"
    bench(block, dev, ref, f"mamba2_block {label}")


if __name__ == "__main__":
    main()
