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

"""Standalone decoder-style transformer block + A/B baseline harness (Spyre).

Structure (pre-norm decoder block, the Llama/Mistral/Granite shape):

    h  = x + Attn(RMSNorm(x))          # RMSNorm -> SDPA attention -> residual
    y  = h + MLP(RMSNorm(h))           # RMSNorm -> gate/up/down MLP -> residual

Attention uses ``torch.nn.functional.scaled_dot_product_attention`` (the fused
SDPA path the Spyre backend lowers to a single attention kernel). The MLP is the
SwiGLU shape (gate/up/down linears + SiLU).

This script is the BASELINE leg of the A/B benchmark: it times the stock
``torch.compile(backend="inductor")`` block on Spyre and checks correctness
against CPU. The on-chip leg reuses the same module + harness; the orchestrator
swaps in the spliced on-chip bundle via the kernel-runner redirect trick (see
``/tmp/bench_onchip.py``) and re-times. Differencing the two median latencies
gives the measured whole-block on-chip speedup that ``projection.md`` predicts.

Run (orchestrator only -- do NOT run on device here):

    PYTHONPATH=/home/adnan/dt-inductor/torch-spyre \\
      /home/adnan/dt-inductor/.venv/bin/python transformer_block_workload.py

Config via env (defaults = the measured mid-range sweet spot, hidden 2048):

    HIDDEN          hidden / model dim          (default 2048)
    N_HEADS         attention heads             (default 16)
    SEQ             sequence length             (default 512)
    INTERMEDIATE    MLP intermediate dim        (default 5504)
    BATCH           batch size                  (default 1)
    BENCH_WARMUP    warmup iters                (default 15)
    BENCH_ITERS     timed iters                 (default 60)
"""

import os
import statistics
import time

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc
import torch.nn as nn
import torch.nn.functional as F
import torch_spyre  # noqa: F401

HIDDEN = int(os.environ.get("HIDDEN", "2048"))
N_HEADS = int(os.environ.get("N_HEADS", "16"))
SEQ = int(os.environ.get("SEQ", "512"))
INTERMEDIATE = int(os.environ.get("INTERMEDIATE", "5504"))
BATCH = int(os.environ.get("BATCH", "1"))
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))
EPS = 1e-5


def rms_norm(x, weight, eps):
    """RMSNorm over the last (hidden) dim, fp16 throughout.

    The mean-square reduction stays in fp16 -- the Spyre backend does not
    support a mean reduction on fp32 (it lowers granite's RMSNorm the same
    fp16 way). Slight precision delta vs an fp32 reference, within tolerance.
    """
    var = (x * x).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(var + eps)
    return x_normed * weight


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: RMSNorm -> SDPA -> add -> RMSNorm -> MLP -> add."""

    def __init__(self, hidden, n_heads, intermediate, eps=EPS):
        super().__init__()
        assert hidden % n_heads == 0, "hidden must be divisible by n_heads"
        self.hidden = hidden
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.eps = eps

        self.attn_norm_w = nn.Parameter(torch.ones(hidden, dtype=torch.float16))
        self.mlp_norm_w = nn.Parameter(torch.ones(hidden, dtype=torch.float16))

        self.q_proj = nn.Linear(hidden, hidden, bias=False, dtype=torch.float16)
        self.k_proj = nn.Linear(hidden, hidden, bias=False, dtype=torch.float16)
        self.v_proj = nn.Linear(hidden, hidden, bias=False, dtype=torch.float16)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=torch.float16)

        self.gate_proj = nn.Linear(
            hidden, intermediate, bias=False, dtype=torch.float16
        )
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(
            intermediate, hidden, bias=False, dtype=torch.float16
        )

    def attn(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(b, s, self.hidden)
        return self.o_proj(out)

    def mlp(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def forward(self, x):
        h = x + self.attn(rms_norm(x, self.attn_norm_w, self.eps))
        y = h + self.mlp(rms_norm(h, self.mlp_norm_w, self.eps))
        return y


def main():
    label = f"hidden={HIDDEN} heads={N_HEADS} seq={SEQ} inter={INTERMEDIATE}"
    torch.manual_seed(0)

    block_cpu = TransformerBlock(HIDDEN, N_HEADS, INTERMEDIATE).eval()
    # Small init so fp16 stays well-conditioned through two residual adds.
    with torch.no_grad():
        for p in block_cpu.parameters():
            if p.dim() == 2:
                p.mul_(0.02)
    x_cpu = torch.randn(BATCH, SEQ, HIDDEN, dtype=torch.float16) * 0.1

    with torch.no_grad():
        ref = block_cpu(x_cpu).float()

    block_dev = block_cpu.to("spyre")
    x_dev = x_cpu.to("spyre")

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    cblock = torch.compile(block_dev, backend="inductor")

    with torch.no_grad():
        out0 = cblock(x_dev).cpu().float()
    max_err = (out0 - ref).abs().max().item()

    with torch.no_grad():
        for _ in range(W):
            cblock(x_dev)
        acc.synchronize()
        s = []
        for _ in range(N):
            t0 = time.perf_counter()
            cblock(x_dev)
            acc.synchronize()
            s.append((time.perf_counter() - t0) * 1000.0)

    print(
        f"BENCH block {label} median_ms={statistics.median(s):.4f} "
        f"min_ms={min(s):.4f} max_err={max_err:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
