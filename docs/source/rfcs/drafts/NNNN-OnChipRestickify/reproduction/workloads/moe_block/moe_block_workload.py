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

"""Full MoE transformer block (attention + MoE MLP) baseline workload for Spyre.

Structure (one decoder block, pre-norm):

  h  = x + sdpa( rmsnorm(x) projected to Q,K,V )      # attention sub-block
  y  = h + moe_mlp( rmsnorm(h) )                       # MoE MLP sub-block

The MoE MLP uses a top-k router with a *dense, capacity-free, mask-based*
dispatch so it compiles cleanly on Spyre (no data-dependent shapes, no dynamic
gather): the dispatch and combine are expressed as matmuls/masks over the full
token set. The per-expert FFN is a true batched matmul (bmm) over E experts,
matching project_bmm_aware_split (MoE expert FFNs are the real bmm case).

  router_logits = norm_h @ w_router          [Tk, E]
  gates, idx    = topk(softmax(router_logits), k)        top-k expert weights
  dispatch_mask = scatter top-k into [Tk, E] (0/gate)    combine = dispatch_mask
  xe            = einsum dispatch -> [E, Tk, H]          per-expert token stack
  ye            = expert_ffn_bmm(xe)         [E, Tk, H]   gate/up/down SwiGLU bmm
  moe_out       = combine ye -> [Tk, H]      weighted sum back to token order

Shapes (fp16), parameterized via env:
  MOE_BATCH      B     batch                              (default 1)
  MOE_SEQ        Sq    sequence length                    (default 128)
  MOE_HIDDEN     H     model hidden dim                   (default 2048)
  MOE_INTER      INTER per-expert intermediate dim          (default 8192)
  MOE_EXPERTS    E     number of experts                  (default 8)
  MOE_TOPK       k     experts selected per token         (default 2)
  MOE_HEADS      nh    attention heads                    (default 16)
  BENCH_WARMUP         warmup iters                       (default 15)
  BENCH_ITERS          timed iters                        (default 60)

Run (offline orchestrator does the device step):
  /home/adnan/dt-inductor/.venv/bin/python3 moe_block_workload.py
"""

import os
import statistics
import time

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc
import torch.nn.functional as F
import torch_spyre  # noqa: F401

B = int(os.environ.get("MOE_BATCH", "1"))
Sq = int(os.environ.get("MOE_SEQ", "128"))
H = int(os.environ.get("MOE_HIDDEN", "2048"))
INTER = int(os.environ.get("MOE_INTER", "8192"))
E = int(os.environ.get("MOE_EXPERTS", "8"))
K = int(os.environ.get("MOE_TOPK", "2"))
NH = int(os.environ.get("MOE_HEADS", "16"))
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))

HD = H // NH  # head dim


def rmsnorm(x, weight, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def attention(x, wq, wk, wv, wo):
    """Standard multi-head attention via fused SDPA."""
    b, s, _ = x.shape
    q = (x @ wq).view(b, s, NH, HD).transpose(1, 2)
    k = (x @ wk).view(b, s, NH, HD).transpose(1, 2)
    v = (x @ wv).view(b, s, NH, HD).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.transpose(1, 2).reshape(b, s, H)
    return o @ wo


def moe_mlp(x, w_router, w_gate, w_up, w_down):
    """Top-k routed MoE MLP with mask-based dispatch and bmm expert FFNs.

    x: [B, Sq, H] -> flatten to [Tk, H], Tk = B*Sq tokens.
    """
    b, s, h = x.shape
    tk = b * s
    xf = x.reshape(tk, h)

    # Router: softmax over experts, then top-k gating weights per token.
    logits = xf @ w_router  # [Tk, E]
    probs = F.softmax(logits, dim=-1)
    gates, idx = torch.topk(probs, K, dim=-1)  # [Tk, K]
    gates = gates / gates.sum(dim=-1, keepdim=True)

    # Build a dense [Tk, E] combine/dispatch weight matrix from the top-k result.
    combine = torch.zeros(tk, E, dtype=x.dtype, device=x.device)
    combine = combine.scatter(1, idx, gates)  # [Tk, E], 0 except chosen experts

    # Dispatch: per-expert token stack. dispatch[e] selects all tokens, the
    # combine weight (0 for non-routed tokens) zeroes the unrouted ones after
    # the FFN. xe[e] = xf (every expert sees the full token set; mask applied on
    # combine). This is the dense MoE formulation (compiles to static bmm).
    xe = xf.unsqueeze(0).expand(E, tk, h)  # [E, Tk, H]

    # Per-expert SwiGLU FFN as batched matmuls (the real MoE bmm case).
    gate = torch.bmm(xe, w_gate)  # [E, Tk, INTER]
    up = torch.bmm(xe, w_up)  # [E, Tk, INTER]
    act = F.silu(gate) * up  # [E, Tk, INTER]
    ye = torch.bmm(act, w_down)  # [E, Tk, H]

    # Combine: weighted sum of expert outputs back to token order.
    # combine.t() is [E, Tk]; weight each expert's output per token then sum.
    cw = combine.t().unsqueeze(-1)  # [E, Tk, 1]
    moe_out = (ye * cw).sum(dim=0)  # [Tk, H]
    return moe_out.reshape(b, s, h)


def moe_block(
    x,
    norm1_w,
    wq,
    wk,
    wv,
    wo,
    norm2_w,
    w_router,
    w_gate,
    w_up,
    w_down,
):
    """One pre-norm decoder block: attention sub-block then MoE MLP sub-block."""
    h = x + attention(rmsnorm(x, norm1_w), wq, wk, wv, wo)
    y = h + moe_mlp(rmsnorm(h, norm2_w), w_router, w_gate, w_up, w_down)
    return y


def main():
    label = f"B={B} Sq={Sq} H={H} INTER={INTER} E={E} K={K} NH={NH}"
    torch.manual_seed(0)
    f16 = torch.float16

    def r(*shape):
        return torch.randn(*shape, dtype=f16) * 0.1

    x = r(B, Sq, H)
    norm1_w = r(H)
    wq, wk, wv, wo = r(H, H), r(H, H), r(H, H), r(H, H)
    norm2_w = r(H)
    w_router = r(H, E)
    w_gate = r(E, H, INTER)
    w_up = r(E, H, INTER)
    w_down = r(E, INTER, H)

    args = (x, norm1_w, wq, wk, wv, wo, norm2_w, w_router, w_gate, w_up, w_down)
    ref = moe_block(*args).float()

    dev = [t.to("spyre") for t in args]
    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    cf = torch.compile(moe_block, backend="inductor")

    out0 = cf(*dev).cpu().float()
    max_err = (out0 - ref).abs().max().item()

    for _ in range(W):
        cf(*dev)
    acc.synchronize()
    s = []
    for _ in range(N):
        t0 = time.perf_counter()
        cf(*dev)
        acc.synchronize()
        s.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"BENCH moe_block {label} median_ms={statistics.median(s):.4f} "
        f"min_ms={min(s):.4f} max_err={max_err:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
