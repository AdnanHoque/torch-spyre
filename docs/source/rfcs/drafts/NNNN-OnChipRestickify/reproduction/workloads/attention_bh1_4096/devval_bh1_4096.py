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

"""Device A/B for spliced B*H=1 seq_q=512 seq_k=4096 SDPA on-chip bundle.

Redirects the fused attention kernel runner to a FRESH spliced code_dir so
g_artifact_cache cannot shadow the load (recipe 7a). validate / bench modes.
ONCHIP_BASELINE=1 = stock HBM (A side). Separate q/kv seq lengths.
"""

import os
import statistics
import time

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc
import torch.nn.functional as functional
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr

BH = int(os.environ.get("ATTN_BH", "1"))
SEQ_Q = int(os.environ.get("ATTN_SEQ_Q", "512"))
SEQ_K = int(os.environ.get("ATTN_SEQ_K", "4096"))
HEAD_DIM = int(os.environ.get("ATTN_HEAD_DIM", "128"))
MODE = os.environ.get("ONCHIP_MODE", "validate").strip()
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))
BASELINE = os.environ.get("ONCHIP_BASELINE", "").strip() in ("1", "true", "yes")
SPLICED = os.environ.get("ONCHIP_DIR", "/tmp/ab_attn_bh1_4096/spliced-attn-bh1-4096")
DEVICE = "spyre"

if not BASELINE:
    _orig = kr.SpyreSDSCKernelRunner.__init__

    def _patched(self, name, code_dir):
        _orig(self, name, code_dir)
        if "attention" in name.lower() or "scaled_dot_product" in name.lower():
            print(f"[REDIRECT] {name}: {code_dir} -> {SPLICED}", flush=True)
            self.code_dir = SPLICED

    kr.SpyreSDSCKernelRunner.__init__ = _patched


def attention(q, k, v):
    return functional.scaled_dot_product_attention(q, k, v)


def main():
    torch.manual_seed(0)
    q = torch.randn((1, BH, SEQ_Q, HEAD_DIM), dtype=torch.float16) * 0.1
    k = torch.randn((1, BH, SEQ_K, HEAD_DIM), dtype=torch.float16) * 0.1
    v = torch.randn((1, BH, SEQ_K, HEAD_DIM), dtype=torch.float16) * 0.1
    ref = attention(q, k, v).float()
    dq, dk, dv = q.to(DEVICE), k.to(DEVICE), v.to(DEVICE)

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(attention, backend="inductor")

    out = compiled(dq, dk, dv).cpu().float()
    max_err = (out - ref).abs().max().item()
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=3e-2)
    side = "baseline_HBM" if BASELINE else f"spliced={SPLICED}"
    print(f"DIRECT_VALIDATE_OK {side} max_err {max_err}", flush=True)

    if MODE == "bench":
        for _ in range(W):
            compiled(dq, dk, dv)
        acc.synchronize()
        s = []
        for _ in range(N):
            t0 = time.perf_counter()
            compiled(dq, dk, dv)
            acc.synchronize()
            s.append((time.perf_counter() - t0) * 1000.0)
        print(
            f"BENCH bh1 q={SEQ_Q} k={SEQ_K} {side} "
            f"median_ms={statistics.median(s):.4f} min_ms={min(s):.4f} "
            f"max_err={max_err:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
