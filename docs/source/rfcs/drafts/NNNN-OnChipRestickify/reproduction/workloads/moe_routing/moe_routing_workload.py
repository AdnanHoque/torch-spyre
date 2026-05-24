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

"""Isolated microbenchmarks for the DATA-MOVEMENT pieces of an MoE layer.

This benchmarks ONLY the token routing data movement -- dispatch (gather) and
combine (scatter-add) -- as standalone ops at representative MoE shapes, plus the
small router GEMM. The expert FFNs themselves are out of scope here (a separate
agent covers them). The goal is an offline baseline + correctness harness the
orchestrator can run on the Spyre device; THIS SCRIPT DOES NOT RUN ANYTHING ITSELF
beyond importing -- the orchestrator invokes it serially (single shared
accelerator).

Why the gather/scatter are expressed as one-hot PERMUTATION MATMULs rather than
``index_select`` / ``scatter_add``
================================================================================
On Spyre, ``aten.index_select`` is not natively lowered and ``aten.embedding``
falls back to CPU (see ``torch_spyre/ops/fallbacks.py``); a literal index_select
gather would not stay on device. The standard, Spyre-friendly way to express MoE
token routing is as a **one-hot permutation matmul**:

  dispatch:  dispatched = P @ x        # P is [E*C, T] one-hot, x is [T, H]
  combine:   combined   = Pw.T @ y     # Pw is the weighted one-hot, y is [E*C, H]

where T = tokens, H = hidden, E = experts, C = per-expert capacity, E*C the padded
dispatch buffer. This is exactly how device-friendly MoE dispatch/combine is built
(e.g. GShard/Switch-style capacity routing): the permutation is a 0/1 (or weighted)
matrix and gather/scatter become GEMMs the Spyre matmul path already handles.

This formulation is also what makes the layout reasoning in ``eligibility.md``
concrete: the activation operand ``x`` / ``y`` keeps its hidden (stick) dim intact;
only the token (mb) dim is permuted -- a whole-stick re-placement.

Parameterization (all via env, no CLI)
======================================
  MOE_E        n experts            (default 8;  try 64)
  MOE_T        n tokens             (default 2048)
  MOE_H        hidden dim           (default 2048; try 4096)
  MOE_TOPK     top-k routing        (default 1;   try 2)
  MOE_CAP_FAC  capacity factor      (default 1.0) -> C = ceil(TOPK*T/E*fac)
  MOE_BENCH    which bench to run   one of: dispatch | combine | router | all
  MOE_WARMUP   warmup iters         (default 15)
  MOE_ITERS    timed iters          (default 60)
  MOE_BACKEND  torch.compile backend (default "inductor")

Each bench prints a single ``BENCH ...`` line with median/min ms and the max abs
error vs the CPU reference (the correctness sanity check), matching the
``/tmp/bench_onchip.py`` template.
"""

import math
import os
import statistics
import time

import torch
import torch._dynamo
import torch._inductor.config as _ind
import torch.accelerator as acc
import torch_spyre  # noqa: F401

DEVICE = "spyre"

E = int(os.environ.get("MOE_E", "8"))
T = int(os.environ.get("MOE_T", "2048"))
H = int(os.environ.get("MOE_H", "2048"))
TOPK = int(os.environ.get("MOE_TOPK", "1"))
CAP_FAC = float(os.environ.get("MOE_CAP_FAC", "1.0"))
WHICH = os.environ.get("MOE_BENCH", "all").strip().lower()
WARMUP = int(os.environ.get("MOE_WARMUP", "15"))
ITERS = int(os.environ.get("MOE_ITERS", "60"))
BACKEND = os.environ.get("MOE_BACKEND", "inductor").strip()

# Per-expert capacity (padded buffer rows per expert) and total dispatch rows.
CAP = max(1, math.ceil(TOPK * T / E * CAP_FAC))
EC = E * CAP  # rows of the dispatch buffer = sum of per-expert capacities.


def _build_routing(seed: int = 0):
    """Build a fixed (deterministic) one-hot dispatch/combine permutation.

    Returns CPU fp16 tensors:
      x         [T, H]    input activations (token-major).
      perm      [EC, T]   one-hot dispatch matrix (row r selects one token).
      perm_w    [T, EC]   weighted combine matrix (combine = perm_w @ y).
      gate      [EC, 1]   per-slot combine weight (router softmax prob).

    The assignment is deterministic (round-robin top-k) so the bench is
    reproducible offline; the data-movement cost does not depend on WHICH tokens
    map where, only on the shapes, so a fixed assignment is representative.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, dtype=torch.float16, generator=g) * 0.1

    # Deterministic round-robin assignment: slot r (= expert e, cap c) gets a
    # token. This fills the capacity buffer densely (no dropped/empty slots) so
    # the moved-bytes count matches the worst case the projection assumes.
    perm = torch.zeros(EC, T, dtype=torch.float16)
    gate = torch.zeros(EC, 1, dtype=torch.float16)
    for r in range(EC):
        tok = (r * 7 + 3) % T  # spread tokens across slots
        perm[r, tok] = 1.0
        # A plausible softmax-ish combine weight in (0, 1].
        gate[r, 0] = 0.5 + 0.5 * ((r % 4) / 4.0)

    # Combine maps EC expert-slot outputs back to T tokens, weighted by gate.
    # combined[t] = sum_r perm[r, t] * gate[r] * y[r]  ==>  perm_w = (perm*gate).T
    perm_w = (perm * gate).t().contiguous()  # [T, EC]
    return x, perm, perm_w, gate


def f_dispatch(perm, x):
    """Dispatch gather as a one-hot permutation matmul: [EC,T] @ [T,H] -> [EC,H]."""
    return perm @ x


def f_combine(perm_w, y):
    """Combine scatter-add as weighted permutation matmul: [T,EC] @ [EC,H]."""
    return perm_w @ y


def f_router(x, wg):
    """Router GEMM (token->expert logits): [T,H] @ [H,E] -> [T,E]."""
    return x @ wg


def _time(label, fn, args, ref):
    """Warm, time (median/min ms), and report max abs error vs CPU ref."""
    torch._dynamo.reset()
    _ind.fx_graph_cache = False
    dev_args = [a.to(DEVICE) for a in args]
    cf = torch.compile(fn, backend=BACKEND)

    out0 = cf(*dev_args).cpu().float()
    max_err = (out0 - ref).abs().max().item()

    for _ in range(WARMUP):
        cf(*dev_args)
    acc.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        cf(*dev_args)
        acc.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    print(
        f"BENCH moe_routing op={label} E={E} T={T} H={H} topk={TOPK} cap={CAP} "
        f"EC={EC} median_ms={statistics.median(samples):.4f} "
        f"min_ms={min(samples):.4f} max_err={max_err:.6f}",
        flush=True,
    )


def main():
    torch.manual_seed(0)
    x, perm, perm_w, _gate = _build_routing()

    if WHICH in ("dispatch", "all"):
        ref = f_dispatch(perm, x).float()
        _time("dispatch_gather", f_dispatch, (perm, x), ref)

    if WHICH in ("combine", "all"):
        # Expert outputs y are [EC, H]; here we feed the dispatched activations
        # back through combine (the data-movement shape is what matters, not the
        # FFN in between, which a separate agent benches).
        y = f_dispatch(perm, x)
        ref = f_combine(perm_w, y).float()
        _time("combine_scatter", f_combine, (perm_w, y), ref)

    if WHICH in ("router", "all"):
        g = torch.Generator().manual_seed(1)
        wg = torch.randn(H, E, dtype=torch.float16, generator=g) * 0.02
        ref = f_router(x, wg).float()
        _time("router_gemm", f_router, (x, wg), ref)


if __name__ == "__main__":
    main()
