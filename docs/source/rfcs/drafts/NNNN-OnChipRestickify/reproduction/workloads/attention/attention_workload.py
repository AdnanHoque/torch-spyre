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

"""Standalone flash / SDPA attention workload + baseline timing harness (Spyre).

This is the "A" side of the on-chip-vs-HBM A/B for attention. It builds a single
scaled-dot-product-attention block at representative shapes and times it on the
Spyre device via ``torch.compile(backend="inductor")`` using the proven
``bench_onchip.py`` methodology (warm-up + median + min + max_err vs CPU).

The compiled SDPA lowers to exactly the cached attention bundle this A/B targets
(``sdsc_fused__scaled_dot_product_fused_attention_overrideable_0_*``), whose
QK^T -> softmax(max/sub) handoff is the same-stick cross-core edge spliced by
``splice_attention_qk_softmax.py``. The cached bundle was compiled at
batch*heads=32, seq=64, head_dim=128 -- the defaults below reproduce it.

Shapes are env-parameterized:
  ATTN_BH       batch*heads (the bmm batch axis 'x')           default 32
  ATTN_SEQ      sequence length (query==key seq)               default 64
  ATTN_HEAD_DIM head dimension (the QK^T contraction 'in')     default 128
  BENCH_WARMUP  warm-up iterations                             default 15
  BENCH_ITERS   timed iterations                               default 60
  SPLICED_DIR   if set, redirect the attention kernel runner to this fresh
                code_dir (the on-chip spliced bundle) -- the "B" side. Unset =
                baseline_HBM.

DO NOT run this here; the orchestrator runs all device steps serially (single
shared accelerator). This file only needs to be correct + offline-validated.
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

BH = int(os.environ.get("ATTN_BH", "32"))
SEQ = int(os.environ.get("ATTN_SEQ", "64"))
HEAD_DIM = int(os.environ.get("ATTN_HEAD_DIM", "128"))
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))
SPLICED = os.environ.get("SPLICED_DIR", "").strip()

DEVICE = "spyre"

# Redirect the attention kernel's runner to a FRESH code_dir the process has
# never seen, so g_artifact_cache cannot shadow the real load (recipe section 7a).
# The fused SDPA kernel name carries "scaled_dot_product"/"attention".
if SPLICED:
    _orig = kr.SpyreSDSCKernelRunner.__init__

    def _patched(self, name, code_dir):
        _orig(self, name, code_dir)
        if "attention" in name.lower() or "scaled_dot_product" in name.lower():
            self.code_dir = SPLICED

    kr.SpyreSDSCKernelRunner.__init__ = _patched


def attention(query, key, value):
    """Single SDPA block: softmax(QK^T) @ V, no causal mask, no scaling tweak."""
    return functional.scaled_dot_product_attention(query, key, value)


def main():
    label = f"spliced={SPLICED}" if SPLICED else "baseline_HBM"
    torch.manual_seed(0)
    shape = (1, BH, SEQ, HEAD_DIM)  # (batch, heads, seq, head_dim)
    cpu = [torch.randn(shape, dtype=torch.float16) * 0.1 for _ in range(3)]
    ref = attention(*cpu).float()
    dev = [t.to(DEVICE) for t in cpu]

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(attention, backend="inductor")

    out0 = compiled(*dev).cpu().float()
    max_err = (out0 - ref).abs().max().item()

    for _ in range(W):
        compiled(*dev)
    acc.synchronize()
    samples = []
    for _ in range(N):
        t0 = time.perf_counter()
        compiled(*dev)
        acc.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"BENCH attn bh={BH} seq={SEQ} head_dim={HEAD_DIM} {label} "
        f"median_ms={statistics.median(samples):.4f} min_ms={min(samples):.4f} "
        f"max_err={max_err:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
