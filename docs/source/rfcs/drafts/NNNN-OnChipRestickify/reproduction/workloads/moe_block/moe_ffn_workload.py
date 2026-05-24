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

"""MoE expert-FFN microbenchmark (gate/up/down) as a batched matmul over experts.

This isolates the per-expert SwiGLU FFN of a Mixture-of-Experts MLP and runs it
as a true batched matmul (bmm): batch dim = number of experts E, so each expert
applies its own weight matrices to its own (capacity) token slice. This is the
real bmm case for MoE on Spyre (see project_bmm_aware_split: per-expert weights =
true bmm, batch = num experts); it is what the M x N co-split planner targets.

Shapes (fp16), parameterized via env:
  MOE_EXPERTS  E      number of experts                   (default 8)
  MOE_HIDDEN   H      model hidden dim                    (default 2048)
  MOE_INTER    INTER  per-expert intermediate (FFN) dim   (default 8192)
  MOE_TOKENS   T      tokens routed per expert (capacity) (default 128)
  BENCH_WARMUP        warmup iters                        (default 15)
  BENCH_ITERS         timed iters                         (default 60)

Tensors:
  x       [E, T, H]        tokens dispatched to each expert
  w_gate  [E, H, INTER]    gate projection (per expert)
  w_up    [E, H, INTER]    up   projection (per expert)
  w_down  [E, INTER, H]    down projection (per expert)

  gate = bmm(x, w_gate)         [E, T, INTER]
  up   = bmm(x, w_up)           [E, T, INTER]
  act  = silu(gate) * up        [E, T, INTER]   (SwiGLU)
  out  = bmm(act, w_down)       [E, T, H]

Run (offline orchestrator does the device step):
  /home/adnan/dt-inductor/.venv/bin/python3 moe_ffn_workload.py

The harness mirrors /tmp/bench_onchip.py: warm + median, one config per process,
max-abs-error vs CPU as the correctness sanity check.
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

E = int(os.environ.get("MOE_EXPERTS", "8"))
H = int(os.environ.get("MOE_HIDDEN", "2048"))
INTER = int(os.environ.get("MOE_INTER", "8192"))
T = int(os.environ.get("MOE_TOKENS", "128"))
W = int(os.environ.get("BENCH_WARMUP", "15"))
N = int(os.environ.get("BENCH_ITERS", "60"))


def expert_ffn(x, w_gate, w_up, w_down):
    """Per-expert SwiGLU FFN over E experts as batched matmuls."""
    gate = torch.bmm(x, w_gate)
    up = torch.bmm(x, w_up)
    act = F.silu(gate) * up
    return torch.bmm(act, w_down)


def main():
    label = f"E={E} H={H} INTER={INTER} T={T}"
    torch.manual_seed(0)
    # Scale down to keep fp16 accumulation well-conditioned for the error check.
    x = torch.randn(E, T, H, dtype=torch.float16) * 0.1
    w_gate = torch.randn(E, H, INTER, dtype=torch.float16) * 0.1
    w_up = torch.randn(E, H, INTER, dtype=torch.float16) * 0.1
    w_down = torch.randn(E, INTER, H, dtype=torch.float16) * 0.1

    ref = expert_ffn(x, w_gate, w_up, w_down).float()

    dev = [t.to("spyre") for t in (x, w_gate, w_up, w_down)]
    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    cf = torch.compile(expert_ffn, backend="inductor")

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
        f"BENCH moe_ffn {label} median_ms={statistics.median(s):.4f} "
        f"min_ms={min(s):.4f} max_err={max_err:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
