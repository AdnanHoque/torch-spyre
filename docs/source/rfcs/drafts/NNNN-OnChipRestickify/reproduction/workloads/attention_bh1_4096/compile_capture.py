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

"""Compile + baseline-time SDPA at B*H=1, seq_q=512, seq_k=4096, head_dim=128.

Separate q/kv seq lengths (the report.txt:119 shape). Captures the compiled
attention code_dir + baseline wall/device timing. Validates vs CPU SDPA.
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
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))
DEVICE = "spyre"

_seen = []
_orig = kr.SpyreSDSCKernelRunner.__init__


def _patched(self, name, code_dir):
    _orig(self, name, code_dir)
    _seen.append((name, code_dir))
    print(f"[KERNEL] {name}: {code_dir}", flush=True)


kr.SpyreSDSCKernelRunner.__init__ = _patched


def attention(query, key, value):
    return functional.scaled_dot_product_attention(query, key, value)


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
    print(
        f"COMPILE_OK shape q={list(q.shape)} kv={list(k.shape)} max_err {max_err}",
        flush=True,
    )

    for _ in range(W):
        compiled(dq, dk, dv)
    acc.synchronize()
    samples = []
    for _ in range(N):
        t0 = time.perf_counter()
        compiled(dq, dk, dv)
        acc.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"BENCH baseline_HBM median_ms={statistics.median(samples):.4f} "
        f"min_ms={min(samples):.4f} max_err={max_err:.6f}",
        flush=True,
    )
    for nm, cd in _seen:
        if "attention" in nm.lower() or "scaled_dot" in nm.lower():
            print(f"ATTN_CODEDIR {cd}", flush=True)


if __name__ == "__main__":
    main()
