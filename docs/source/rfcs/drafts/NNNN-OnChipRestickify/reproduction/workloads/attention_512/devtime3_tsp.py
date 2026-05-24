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

"""Clean 3-way prefill attention device-time for torch-spyre (baseline + on-chip).

Resident device tensors (NO per-iter host copies). torch.profiler PrivateUse1
device_time_total. Reports:
  spyre_ms  = total device time / N  (all device events: compute + memcpy/memset)
  kernel_ms = compute device time / N (EXCLUDING Memcpy/Memset events)
  max_err   = vs CPU SDPA reference

acc_events=True so device_time_total aggregates across all N profiled steps
(without it the profiler clears per cycle and only the last step survives).

When ONCHIP_BASELINE is unset, the fused attention kernel runner is redirected
to the spliced on-chip bundle (ONCHIP_DIR), same patch as devval.
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
N = int(os.environ.get("BENCH_ITERS", "20"))
BASELINE = os.environ.get("ONCHIP_BASELINE", "").strip() in ("1", "true", "yes")
SPLICED = os.environ.get("ONCHIP_DIR", "/tmp/ab_attention_512/spliced-attn-512")
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


def _is_mem(key: str) -> bool:
    k = key.lower()
    return "memcpy" in k or "memset" in k


def main():
    torch.manual_seed(0)
    shape = (1, BH, SEQ, HEAD_DIM)
    cpu = [torch.randn(shape, dtype=torch.float16) * 0.1 for _ in range(3)]
    ref = attention(*cpu).float()
    dev = [t.to(DEVICE) for t in cpu]  # resident; copied ONCE, before timing

    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    compiled = torch.compile(attention, backend="inductor")

    out = compiled(*dev)
    max_err = (out.cpu().float() - ref).abs().max().item()

    # warm-up / compile (not timed); tensors stay resident on device
    for _ in range(W):
        compiled(*dev)

    prof = torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        acc_events=True,
    )
    prof.start()
    for _ in range(N):
        compiled(*dev)
        prof.step()
    prof.stop()

    total = 0.0
    kernel = 0.0
    rows = []
    for ev in prof.key_averages():
        dt = ev.device_time_total
        if dt <= 0:
            continue
        total += dt
        if not _is_mem(ev.key):
            kernel += dt
        rows.append((ev.key, dt, ev.count))

    side = "baseline_HBM" if BASELINE else f"spliced={SPLICED}"
    print(
        f"DEVTIME3 attn bh={BH} seq={SEQ} head_dim={HEAD_DIM} {side} "
        f"spyre_ms={total / 1000.0 / N:.4f} kernel_ms={kernel / 1000.0 / N:.4f} "
        f"max_err={max_err:.6f} N={N}",
        flush=True,
    )
    for key, dt, count in sorted(rows, key=lambda r: -r[1]):
        tag = "MEM" if _is_mem(key) else "CMP"
        print(
            f"  {tag} {key[-58:]:<58} total_ms={dt / 1000.0:.4f} "
            f"per_iter_ms={dt / 1000.0 / N:.4f} count={count}",
            flush=True,
        )


if __name__ == "__main__":
    main()
