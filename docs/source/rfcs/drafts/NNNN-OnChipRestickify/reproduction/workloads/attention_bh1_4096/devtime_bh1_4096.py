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

"""Device-time (spyre_ms + kernel_ms) for the report.txt:119 attention shape.

Asymmetric SDPA: Q=[1,1,512,128], K=V=[1,1,4096,128] (B*H=1, seq_q=512,
seq_k=4096, head_dim=128). Uses torch.profiler PrivateUse1 device_time_total
(same method as the 3-way helper; no USE_SPYRE_PROFILER needed). Reports
spyre_ms (all device events) and kernel_ms (compute only, excluding
Memcpy/Memset) to line up with report.txt columns. Redirects the attention
kernel runner to the spliced on-chip bundle unless ONCHIP_BASELINE=1.
"""

import os

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.nn.functional as functional
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr
from torch.profiler import ProfilerActivity

Q_SEQ = int(os.environ.get("ATTN_Q_SEQ", "512"))
KV_SEQ = int(os.environ.get("ATTN_KV_SEQ", "4096"))
HEAD_DIM = int(os.environ.get("ATTN_HEAD_DIM", "128"))
W = int(os.environ.get("BENCH_WARMUP", "8"))
N = int(os.environ.get("BENCH_ITERS", "10"))
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


def attention(query, key, value):
    return functional.scaled_dot_product_attention(query, key, value)


def main():
    torch.manual_seed(0)
    q = torch.randn(1, 1, Q_SEQ, HEAD_DIM, dtype=torch.float16) * 0.1
    k = torch.randn(1, 1, KV_SEQ, HEAD_DIM, dtype=torch.float16) * 0.1
    v = torch.randn(1, 1, KV_SEQ, HEAD_DIM, dtype=torch.float16) * 0.1
    ref = attention(q, k, v).float()
    dev = [t.to(DEVICE) for t in (q, k, v)]

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(attention, backend="inductor")

    out = compiled(*dev).cpu().float()
    max_err = (out - ref).abs().max().item()
    for _ in range(W):
        compiled(*dev)

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    )
    prof.start()
    for _ in range(N):
        compiled(*dev)
        prof.step()
    prof.stop()

    total = 0.0
    kernel = 0.0
    for e in prof.events():
        dt = e.device_time_total
        if dt <= 0:
            continue
        total += dt
        nm = e.key.lower()
        if "memcpy" not in nm and "memset" not in nm:
            kernel += dt
    label = "baseline_HBM" if BASELINE else f"onchip={SPLICED}"
    print(
        f"DEVTIME {label} q={Q_SEQ} kv={KV_SEQ} hd={HEAD_DIM} "
        f"spyre_ms={total / 1000.0 / N:.4f} kernel_ms={kernel / 1000.0 / N:.4f} "
        f"max_err={max_err:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
