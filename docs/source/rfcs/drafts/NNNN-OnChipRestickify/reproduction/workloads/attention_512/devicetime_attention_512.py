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

"""Device-time (spyre_ms) for torch-spyre attention at the common shape.

Mirrors the perf-suite sendnn measurement (torch.profiler PrivateUse1
device_time_total / runs) so the torch-spyre device time is apples-to-apples
with sendnn's spyre_ms. Redirects the fused attention kernel runner to the
spliced on-chip bundle when ONCHIP_DIR is set (same patch as devval).
"""

import os

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.nn.functional as functional
import torch_spyre  # noqa: F401
import torch_spyre.execution.kernel_runner as kr
from torch.profiler import ProfilerActivity

BH = int(os.environ.get("ATTN_BH", "32"))
SEQ = int(os.environ.get("ATTN_SEQ", "512"))
HEAD_DIM = int(os.environ.get("ATTN_HEAD_DIM", "128"))
W = int(os.environ.get("BENCH_WARMUP", "8"))
N = int(os.environ.get("BENCH_ITERS", "8"))
BASELINE = os.environ.get("ONCHIP_BASELINE", "").strip() in ("1", "true", "yes")
SPLICED = os.environ.get("ONCHIP_DIR", "/tmp/ab_attention_512/spliced-attn-512")
DEVICE = "spyre"

if not BASELINE:
    _orig = kr.SpyreSDSCKernelRunner.__init__

    def _patched(self, name, code_dir):
        _orig(self, name, code_dir)
        if "attention" in name.lower() or "scaled_dot_product" in name.lower():
            self.code_dir = SPLICED

    kr.SpyreSDSCKernelRunner.__init__ = _patched


def attention(query, key, value):
    return functional.scaled_dot_product_attention(query, key, value)


def main():
    torch.manual_seed(0)
    shape = (1, BH, SEQ, HEAD_DIM)
    cpu = [torch.randn(shape, dtype=torch.float16) * 0.1 for _ in range(3)]
    dev = [t.to(DEVICE) for t in cpu]

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(attention, backend="inductor")

    # warm-up / compile (not timed)
    compiled(*dev)
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
    total_spyre = sum(e.device_time_total for e in prof.events()) / 1000.0
    side = "baseline_HBM" if BASELINE else f"spliced={SPLICED}"
    print(
        f"DEVTIME attn bh={BH} seq={SEQ} head_dim={HEAD_DIM} {side} "
        f"spyre_ms={total_spyre / N:.4f}",
        flush=True,
    )
    # Per-kernel device time (mirrors perf-suite kernel_times: device_time/count).
    for ev in prof.key_averages():
        if ev.device_time_total > 0:
            print(
                f"  KERNEL {ev.key[-60:]} ms={ev.device_time_total / ev.count / 1000.0:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
