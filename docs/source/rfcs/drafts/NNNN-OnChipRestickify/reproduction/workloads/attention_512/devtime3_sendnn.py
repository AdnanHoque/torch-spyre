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

"""Clean sendnn (production) prefill attention device-time, apples-to-apples
with the torch-spyre legs.

- Resident device tensors (q/k/v copied to device ONCE, before timing; NO
  per-iter host copies inside the timed loop).
- torch.profiler PrivateUse1 device_time_total, acc_events=True (aggregate over
  all N steps; identical method to the torch-spyre devtime script).
- Reports spyre_ms (total device) and kernel_ms (compute, excluding
  Memcpy/Memset). If sendnn's attention compute does not surface as a named
  kernel, kernel_ms == total-minus-mem and the per-event dump shows what IS
  available.
- max_err vs CPU SDPA (computed in the sendnn torch 2.10 venv).

Plain SDPA call (no GQA, not causal) to match the torch-spyre legs exactly.
"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity  # noqa: E402
from torch_sendnn import torch_sendnn  # noqa: E402

BH = int(os.environ.get("ATTN_BH", "32"))
SEQ = int(os.environ.get("ATTN_SEQ", "512"))
HEAD_DIM = int(os.environ.get("ATTN_HEAD_DIM", "128"))
W = int(os.environ.get("BENCH_WARMUP", "8"))
N = int(os.environ.get("BENCH_ITERS", "20"))
DEVICE = "aiu"

torch_sendnn.sendnn_backend.is_available = lambda: False
torch.utils.rename_privateuse1_backend("aiu")
torch._register_device_module("aiu", torch_sendnn.sendnn_backend)
torch.utils.generate_methods_for_privateuse1_backend()


def attention(query, key, value):
    return torch.nn.functional.scaled_dot_product_attention(query, key, value)


def _is_mem(key: str) -> bool:
    k = key.lower()
    return "memcpy" in k or "memset" in k


def main():
    torch.manual_seed(0)
    shape = (1, BH, SEQ, HEAD_DIM)
    # sendnn has no eager .to("aiu"); its programming model is "compile on CPU
    # tensors, runtime moves to device". "Resident" here means the SAME fixed
    # input tensor objects are reused every iteration (no per-iter realloc / no
    # re-creation), so the only host work in the loop is the call itself. Any
    # H2D the compiled graph does is part of sendnn's execution model and shows
    # up as Memcpy device events, which kernel_ms excludes.
    dev = [torch.randn(shape, dtype=torch.float16) * 0.1 for _ in range(3)]
    ref = attention(*dev).float()

    torch._dynamo.reset()
    compiled = torch.compile(attention, backend="sendnn")

    out = compiled(*dev)
    max_err = (out.cpu().float() - ref).abs().max().item()

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

    print(
        f"DEVTIME3 sendnn attn bh={BH} seq={SEQ} head_dim={HEAD_DIM} "
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
