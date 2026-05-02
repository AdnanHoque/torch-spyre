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

"""Phase 0a of the MoE grouped-GEMM project.

Measures the cost of running an MoE FFN layer on Spyre via the naive
"E separate matmul calls" path. The core question is: **how does decode
latency scale with the number of active experts?** If it grows linearly in
top_k, kernel-launch overhead is the bottleneck and a grouped-GEMM op
helps. If it's flatter, Spyre already amortizes well and the lever is
smaller than expected.

Three baselines per (hidden, intermediate) configuration:

1. **Single expert** — top_k=1, one (gate, up, down) trio. Reference floor.
2. **K active experts (K = 1, 2, 4, 8)** — sequential per-expert calls,
   each token routed to K experts. Naive MoE.
3. **Dense fallback** — one fat (hidden → E·intermediate) matmul replacing
   all expert gates fused, similar for up and down. Computes all expert
   outputs even if only top_k are kept. Establishes the ceiling cost of
   "no per-expert dispatch."

For decode we fix M=1 (one token per step). Real Mixtral hidden=4096
intermediate=14336 — we run with downsized dims (1024 / 2048) to keep
compile time bounded for this probe. Real-dim measurements come in
Phase 0b once Phase 0a justifies the project scope.

Run:  python tests/diag_moe_baseline.py
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import torch

# Same compile configuration as the SplitK / DDR-traffic diagnostics.
import torch._inductor.config as _icfg

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: F401
from torch_spyre import streams as _ts


# ---- model + experts --------------------------------------------------------

@dataclass
class _Cfg:
    hidden: int
    intermediate: int
    num_experts: int

    def label(self) -> str:
        return f"H={self.hidden}, I={self.intermediate}, E={self.num_experts}"


def _make_experts(cfg: _Cfg, dtype=torch.float16, device="spyre"):
    """E experts × (gate, up, down) weight matrices on device."""
    return [
        {
            "gate": torch.randn(cfg.hidden, cfg.intermediate, dtype=dtype, device=device),
            "up": torch.randn(cfg.hidden, cfg.intermediate, dtype=dtype, device=device),
            "down": torch.randn(cfg.intermediate, cfg.hidden, dtype=dtype, device=device),
        }
        for _ in range(cfg.num_experts)
    ]


def _make_dense_fused(cfg: _Cfg, dtype=torch.float16, device="spyre"):
    """Fused 'all-experts' weights for the dense fallback. Each is the
    column-stacked concatenation of every expert's matrix."""
    return {
        "gate_all": torch.randn(
            cfg.hidden, cfg.intermediate * cfg.num_experts, dtype=dtype, device=device
        ),
        "up_all": torch.randn(
            cfg.hidden, cfg.intermediate * cfg.num_experts, dtype=dtype, device=device
        ),
        "down_all": torch.randn(
            cfg.intermediate * cfg.num_experts, cfg.hidden, dtype=dtype, device=device
        ),
    }


# ---- kernels ----------------------------------------------------------------

def _expert_forward_eager(x, gate, up, down):
    """SwiGLU expert: silu(x @ gate) * (x @ up), then @ down."""
    h = torch.nn.functional.silu(x @ gate) * (x @ up)
    return h @ down


def _dense_forward_eager(x, gate_all, up_all, down_all):
    """The dense fallback: one fat matmul per stage. Computes all E
    experts' outputs. Same compute as if every expert were active."""
    h = torch.nn.functional.silu(x @ gate_all) * (x @ up_all)
    return h @ down_all


# ---- bench loop -------------------------------------------------------------

WARMUP = 5
ITERS = 30


def _bench(fn, *args) -> float:
    """Returns median per-iteration ms with per-iter device sync."""
    for _ in range(WARMUP):
        fn(*args)
    _ts.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn(*args)
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def _bench_top_k(cfg: _Cfg, K: int, expert_fn, x, experts) -> float:
    """K active experts, sequential calls, weighted sum into output.
    Models naive MoE decode where top_k experts each process the same token."""
    active = list(range(K))  # active expert indices
    weights = [1.0 / K] * K  # uniform routing weights for the bench

    def step():
        out = torch.zeros_like(x)
        for eid, w in zip(active, weights):
            e = experts[eid]
            y = expert_fn(x, e["gate"], e["up"], e["down"])
            out = out + y * w
        return out

    return _bench(step)


def _bench_dense_fallback(cfg: _Cfg, dense_fn, x, dense_w) -> float:
    def step():
        return dense_fn(x, dense_w["gate_all"], dense_w["up_all"], dense_w["down_all"])
    return _bench(step)


# ---- sweeps ----------------------------------------------------------------

CFGS = [
    # Downsized Mixtral-style: H=1024 I=2048 (vs real 4096/14336).
    # Compiles fast, gives meaningful kernel timings at decode M=1.
    _Cfg(hidden=1024, intermediate=2048, num_experts=8),
    # A second config with a larger intermediate to stress B-side traffic.
    _Cfg(hidden=1024, intermediate=4096, num_experts=8),
]

K_VALUES = [1, 2, 4, 8]


@dataclass
class _Row:
    cfg: _Cfg
    label: str
    median_ms: float
    note: str = ""


def _print_table(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# MoE naive-baseline diagnostic — Phase 0a")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w(f"decode M:       1")
    w(f"per-iter sync:  torch_spyre.streams.synchronize() inside the timed loop")
    w("")
    w("**Naive MoE step**: K active experts run as separate (gate, up, down) "
      "matmul calls in a Python loop, output is the weighted sum. Each "
      "expert's call is `silu(x @ W_gate) * (x @ W_up) @ W_down` — three "
      "matmul launches per active expert.")
    w("")
    w("**Dense fallback**: column-stacked weights `(hidden -> E*intermediate)` "
      "for gate/up and `(E*intermediate -> hidden)` for down. ONE matmul per "
      "stage regardless of E, but computes all E experts' outputs (so "
      "compute is `E×` more than what an oracle MoE would need).")
    w("")

    # Group by config.
    by_cfg: dict[str, list[_Row]] = {}
    for r in rows:
        by_cfg.setdefault(r.cfg.label(), []).append(r)

    for cfg_label, group in by_cfg.items():
        w(f"## {cfg_label}")
        w("")
        w("| variant | median ms | per-active-expert ms | vs single-expert |")
        w("|---|---:|---:|---:|")

        # Find single-expert baseline within this config.
        single = next((r.median_ms for r in group if r.label == "naive K=1"), None)

        for r in group:
            ms = r.median_ms
            if r.label.startswith("naive K="):
                K = int(r.label.split("=")[-1])
                per = ms / K if K > 0 else float("nan")
                ratio = (ms / single) if (single and single > 0) else float("nan")
                w(f"| {r.label} | {ms:.2f} | {per:.2f} | {ratio:.2f}× |")
            else:
                ratio = (ms / single) if (single and single > 0) else float("nan")
                w(f"| {r.label} | {ms:.2f} | — | {ratio:.2f}× |")
        w("")


def main() -> int:
    all_rows: list[_Row] = []

    for cfg in CFGS:
        print(f"\n# config {cfg.label()}", flush=True)

        # Build experts + dense weights once.
        x = torch.randn(1, cfg.hidden, dtype=torch.float16, device="spyre")
        experts = _make_experts(cfg)
        dense_w = _make_dense_fused(cfg)

        # Compile both expert and dense paths. Identical-shape experts will
        # all hit the same compiled artifact after the first.
        torch._dynamo.reset()
        compiled_expert = torch.compile(_expert_forward_eager, dynamic=False)
        compiled_dense = torch.compile(_dense_forward_eager, dynamic=False)

        # Trigger expert compile via a single warm call.
        compiled_expert(x, experts[0]["gate"], experts[0]["up"], experts[0]["down"])
        _ts.synchronize()
        compiled_dense(x, dense_w["gate_all"], dense_w["up_all"], dense_w["down_all"])
        _ts.synchronize()

        # Bench naive top-k for K = 1, 2, 4, 8.
        for K in K_VALUES:
            if K > cfg.num_experts:
                continue
            ms = _bench_top_k(cfg, K, compiled_expert, x, experts)
            all_rows.append(_Row(cfg=cfg, label=f"naive K={K}", median_ms=ms))
            print(f"  naive K={K}: {ms:.2f} ms", flush=True)

        # Bench dense fallback.
        ms = _bench_dense_fallback(cfg, compiled_dense, x, dense_w)
        all_rows.append(_Row(
            cfg=cfg, label="dense fallback (E experts always run)", median_ms=ms,
        ))
        print(f"  dense:    {ms:.2f} ms", flush=True)

    _print_table(all_rows)

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_moe_baseline_results.md",
    )
    with open(results_path, "w") as f:
        _print_table(all_rows, file=f)
    print(f"\n# results written to {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
