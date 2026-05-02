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

"""Cross-kernel weight residency probe.

Tests whether Spyre's device-side runtime / scratchpad management caches
weight data across consecutive kernel calls. If yes, we can leverage this
for MoE-style "expert weight residency" optimizations at the planner
level. If no, weight residency is firmly an inference-stack-level concern
that won't be addressed in Inductor.

Method: at a fixed shape, run N back-to-back compiled mm calls in two
configurations:

1. **same-W**: every iteration uses the SAME W tensor (same memory
   address). If anything caches weights between launches, this benefits.
2. **different-W**: every iteration uses a different W tensor of the same
   shape. Each launch sees a fresh weight matrix. Caching can't help.

Compare per-iter wall time. If same-W is faster, weight residency is real.
If they're identical, kernel boundaries reset the device-side state and
weight residency at the planner level isn't a leverage point.

Phase 0a / 0b context: per-launch overhead is ~3 ms FLAT, so for tiny
matmul shapes the test is dominated by launch overhead and any reuse win
would be invisible. We pick a shape with non-trivial compute so reuse
effects can show up.

Run: python tests/diag_weight_residency.py
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: F401
from torch_spyre import streams as _ts


WARMUP = 5
ITERS = 30


def _bench(fn) -> float:
    for _ in range(WARMUP):
        fn()
    _ts.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


@dataclass
class _Shape:
    label: str
    M: int
    N: int
    K: int


# Shapes chosen to span: per-launch-dominated (small) → compute-visible (medium)
# → bandwidth-shaped (large with big weight matrices). If cross-kernel
# weight reuse exists, we'd see it most clearly at large shapes where the
# weight DMA cost is non-trivial.
SHAPES = [
    _Shape("small (per-launch dominated)", M=64, N=128, K=128),
    _Shape("medium (compute visible)", M=128, N=4096, K=4096),
    _Shape("large weight (BW potentially visible)", M=128, N=8192, K=14336),
]


N_BACK_TO_BACK = 8


@dataclass
class _Row:
    shape_label: str
    config: str
    median_ms: float
    per_call_ms: float


def main() -> int:
    rows: list[_Row] = []

    for sh in SHAPES:
        print(f"\n# shape: {sh.label} ({sh.M}×{sh.N}×{sh.K})", flush=True)

        a = torch.randn(sh.M, sh.K, dtype=torch.float16, device="spyre")
        W_single = torch.randn(sh.K, sh.N, dtype=torch.float16, device="spyre")
        W_list = [
            torch.randn(sh.K, sh.N, dtype=torch.float16, device="spyre")
            for _ in range(N_BACK_TO_BACK)
        ]

        torch._dynamo.reset()

        @torch.compile(dynamic=False)
        def mm(x, y):
            return x @ y

        # Trigger compile.
        mm(a, W_single)
        _ts.synchronize()

        # Single call as baseline.
        single_ms = _bench(lambda: mm(a, W_single))
        rows.append(_Row(sh.label, "single (baseline)", single_ms, single_ms))
        print(f"  single:        {single_ms:6.2f} ms", flush=True)

        # N same-W back-to-back.
        def step_same():
            for _ in range(N_BACK_TO_BACK):
                mm(a, W_single)

        same_ms = _bench(step_same)
        per_same = same_ms / N_BACK_TO_BACK
        rows.append(_Row(sh.label, f"same-W ×{N_BACK_TO_BACK}", same_ms, per_same))
        print(f"  same-W ×{N_BACK_TO_BACK}: total {same_ms:6.2f} ms, per call {per_same:.3f} ms",
              flush=True)

        # N different-W back-to-back.
        def step_diff():
            for i in range(N_BACK_TO_BACK):
                mm(a, W_list[i])

        diff_ms = _bench(step_diff)
        per_diff = diff_ms / N_BACK_TO_BACK
        rows.append(_Row(sh.label, f"different-W ×{N_BACK_TO_BACK}", diff_ms, per_diff))
        print(f"  diff-W ×{N_BACK_TO_BACK}: total {diff_ms:6.2f} ms, per call {per_diff:.3f} ms",
              flush=True)

        # Per-call ratio
        if per_diff > 0:
            ratio = per_same / per_diff
            print(f"  same-W / different-W per-call ratio: {ratio:.3f}× "
                  f"(<1 means same-W faster → weight reuse evidence)",
                  flush=True)

    _print_table(rows)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_weight_residency_results.md",
    )
    with open(out_path, "w") as f:
        _print_table(rows, file=f)
    print(f"\n# results written to {out_path}", flush=True)
    return 0


def _print_table(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# Cross-kernel weight residency probe")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w(f"N back-to-back: {N_BACK_TO_BACK}")
    w("")
    w("**Hypothesis under test**: does Spyre's device-side runtime / "
      "scratchpad reuse weight data across consecutive kernel calls when "
      "the same W tensor is referenced? If yes, MoE-style expert weight "
      "residency is a real planner-level lever.")
    w("")
    w("**Method**: bench N back-to-back `mm(a, W)` calls with the SAME W "
      "vs. with N DIFFERENT Ws. Per-call ratio < 1 indicates same-W is "
      "faster (caching evidence); ratio = 1 indicates per-launch overhead "
      "and DDR streaming reset between every kernel.")
    w("")

    # Group by shape
    by_shape: dict[str, list[_Row]] = {}
    for r in rows:
        by_shape.setdefault(r.shape_label, []).append(r)

    w("| shape | config | total ms | per-call ms |")
    w("|---|---|---:|---:|")
    for shape_label, group in by_shape.items():
        for r in group:
            w(f"| {shape_label} | {r.config} | {r.median_ms:.2f} | "
              f"{r.per_call_ms:.3f} |")
    w("")
    w("### Per-call ratio (same-W vs different-W)")
    w("")
    w("| shape | same-W per call | different-W per call | ratio | verdict |")
    w("|---|---:|---:|---:|---|")
    for shape_label, group in by_shape.items():
        same = next((r.per_call_ms for r in group if r.config.startswith("same-W")), None)
        diff = next((r.per_call_ms for r in group if r.config.startswith("different-W")), None)
        if same is None or diff is None:
            continue
        ratio = same / diff if diff > 0 else float("inf")
        if ratio < 0.92:
            verdict = "**same-W FASTER** (reuse evidence)"
        elif ratio > 1.08:
            verdict = "*same-W slower* (anti-reuse — surprising)"
        else:
            verdict = "tied (no reuse at kernel-boundary granularity)"
        w(f"| {shape_label} | {same:.3f} | {diff:.3f} | {ratio:.3f}× | {verdict} |")


if __name__ == "__main__":
    raise SystemExit(main())
