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


def _bench_permute(hidden: int = 4096) -> list[_Row]:
    """Time a (M, hidden) gather and (M, hidden) scatter on Spyre at decode-
    relevant batch sizes M ∈ {1, 4, 8, 16, 64}. Tells us whether permuted-
    token grouped-GEMM (which needs gather pre-pass + scatter post-pass) is
    viable: if permute is cheap, the format is fine; if it's >5ms per
    side, block-sparse format is the better op design even though it's
    more invasive on the Spyre backend side.
    """
    rows: list[_Row] = []
    permute_cfg = _Cfg(hidden=hidden, intermediate=0, num_experts=0)

    # Spyre op coverage of indexed gather/scatter is incomplete as of this
    # writing. We probe both the eager path (`aten::index.Tensor_out`,
    # `aten::index_put`) and the compiled path; both can be missing or hit
    # symbolic-shape issues (see #1372). The cells where Spyre can't run
    # the op natively are reported as "n/a" so the table still surfaces
    # what works and what doesn't.

    def _try_bench(fn, *args) -> tuple[float | None, str | None]:
        try:
            return _bench(fn, *args), None
        except NotImplementedError as e:
            return None, f"NotImplementedError: {str(e)[:80]}"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {str(e)[:80]}"

    for M in (1, 4, 8, 16, 64):
        x = torch.randn(M, hidden, dtype=torch.float16, device="spyre")
        idx = torch.randperm(M).to("spyre")
        out = torch.zeros_like(x)

        ms_g, err_g = _try_bench(_gather_eager, x, idx)
        ms_s, err_s = _try_bench(_scatter_eager, out, idx, x)

        g_label = f"gather M={M}, H={hidden}"
        s_label = f"scatter M={M}, H={hidden}"
        if ms_g is not None:
            rows.append(_Row(cfg=permute_cfg, label=g_label, median_ms=ms_g))
            g_str = f"{ms_g:.2f} ms"
        else:
            rows.append(_Row(cfg=permute_cfg, label=g_label, median_ms=float("nan"),
                             note=err_g or ""))
            g_str = "n/a"
        if ms_s is not None:
            rows.append(_Row(cfg=permute_cfg, label=s_label, median_ms=ms_s))
            s_str = f"{ms_s:.2f} ms"
        else:
            rows.append(_Row(cfg=permute_cfg, label=s_label, median_ms=float("nan"),
                             note=err_s or ""))
            s_str = "n/a"

        print(f"  M={M:>2}  gather: {g_str}  scatter: {s_str}", flush=True)
    return rows


# ---- sweeps ----------------------------------------------------------------

CFGS = [
    # Downsized Mixtral-style: H=1024 I=2048 (vs real 4096/14336).
    # Compiles fast, gives meaningful kernel timings at decode M=1.
    _Cfg(hidden=1024, intermediate=2048, num_experts=8),
    # A second config with a larger intermediate to stress B-side traffic.
    _Cfg(hidden=1024, intermediate=4096, num_experts=8),
    # Real Mixtral 8x7B dims (Phase 0b). Confirms the launch-overhead-
    # dominance story holds at production scale.
    _Cfg(hidden=4096, intermediate=14336, num_experts=8),
]

K_VALUES = [1, 2, 4, 8]


# ---- Phase 0b: framework-overhead isolation + permute cost ------------------

def _empty_step_eager(x):
    """Pure framework overhead per outer step: zeros_like + sync.
    No matmul. Tells us how much of 'naive K=1' is bookkeeping vs kernel."""
    return torch.zeros_like(x)


def _single_mm_eager(x, W):
    """One matmul, no SwiGLU pointwise. Tells us per-mm launch cost."""
    return x @ W


def _gather_eager(x, idx):
    """Token gather — reorder rows of x by idx. The pre-pass for permuted-
    token grouped-GEMM."""
    return x[idx]


def _scatter_eager(out, idx, src):
    """Token scatter — write src rows back into out at positions idx.
    Post-pass for permuted-token grouped-GEMM."""
    out[idx] = src
    return out


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

    # Group by config; permute rows (intermediate=0) go to their own section.
    by_cfg: dict[str, list[_Row]] = {}
    permute_rows: list[_Row] = []
    for r in rows:
        if r.cfg.intermediate == 0 and r.cfg.num_experts == 0:
            permute_rows.append(r)
        else:
            by_cfg.setdefault(r.cfg.label(), []).append(r)

    for cfg_label, group in by_cfg.items():
        w(f"## {cfg_label}")
        w("")
        w("| variant | median ms | per-active-expert ms | vs single-expert |")
        w("|---|---:|---:|---:|")

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

    if permute_rows:
        w("## Token permute cost (H=4096)")
        w("")
        w("Decode-relevant batch sizes. Permuted-token grouped-GEMM needs a "
          "gather pre-pass + a scatter post-pass; if either is >5ms it eats "
          "into the win, and if either is unsupported on the Spyre op set "
          "(`n/a` below) the format is fully blocked until the op is "
          "registered.")
        w("")
        w("| op | median ms | note |")
        w("|---|---:|---|")
        for r in permute_rows:
            import math as _math
            if _math.isnan(r.median_ms):
                w(f"| {r.label} | n/a | {r.note} |")
            else:
                w(f"| {r.label} | {r.median_ms:.2f} | |")
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
        compiled_empty = torch.compile(_empty_step_eager, dynamic=False)
        compiled_single_mm = torch.compile(_single_mm_eager, dynamic=False)

        # Trigger compiles via warm calls.
        compiled_expert(x, experts[0]["gate"], experts[0]["up"], experts[0]["down"])
        _ts.synchronize()
        compiled_dense(x, dense_w["gate_all"], dense_w["up_all"], dense_w["down_all"])
        _ts.synchronize()
        compiled_empty(x)
        _ts.synchronize()
        compiled_single_mm(x, experts[0]["gate"])
        _ts.synchronize()

        # Phase 0b — framework-overhead isolation BEFORE the K-sweep so the
        # per-K numbers are easier to interpret in context.
        ms = _bench(compiled_empty, x)
        all_rows.append(_Row(
            cfg=cfg, label="empty step (zeros_like + sync only)", median_ms=ms,
        ))
        print(f"  empty step:           {ms:.2f} ms", flush=True)

        ms = _bench(compiled_single_mm, x, experts[0]["gate"])
        all_rows.append(_Row(
            cfg=cfg, label="single mm (no SwiGLU pointwise)", median_ms=ms,
        ))
        print(f"  single mm (gate):     {ms:.2f} ms", flush=True)

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
        print(f"  dense:                {ms:.2f} ms", flush=True)

    # Phase 0b — token-permute cost probe. Standalone, not per-config, since
    # it depends only on (M, hidden) and we want to characterize the curve
    # vs M for a representative hidden dim.
    print(f"\n# token-permute cost probe (H=4096)", flush=True)
    permute_rows = _bench_permute(hidden=4096)
    all_rows.extend(permute_rows)

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
