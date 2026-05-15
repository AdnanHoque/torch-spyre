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

"""FUNDAMENTAL restickify wall-clock cost probe.

Measures the latency cost of today's HBM-bound FUNDAMENTAL restickify by
comparing two graphs that differ only in whether a restickify fires on
the additive operand. Same FLOPs, same weight reads, same output shape.

  Graph A: (X @ W) + Y         — Y already aligned, no restickify
  Graph B: (X @ W) + Yt.t()    — Yt transposed, restickify fires at sc=32

The wall-clock delta is the upper-bound on what `STCDPOpLx` (if it shipped)
could save us. Cost-model prediction is `2 * |Y| / 107 GB/s` (HBM round
trip at measured effective bandwidth).

The probe also wraps `SpyreAsyncCompile.sdsc` to count SDSC emissions per
graph; if A and B emit the same number of kernels, the global restickify
optimizer has likely eliminated the FUNDAMENTAL in B and the timing
comparison is degenerate (which is itself a useful finding).

Run:  SENCORES=32 LX_PLANNING=1 python3 tests/diag_fundamental_restickify_cost.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import Counter
from unittest.mock import patch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_spyre

torch_spyre._autoload()

from torch._inductor import config as t_inductor_config
from torch_spyre._inductor import config as ts_config
from torch_spyre.execution import async_compile as ac


HD = 4096           # H * D = 32 * 128 (granite/llama-style)
M_VALUES = [128, 512, 2048, 8192]
DTYPE = torch.float16
DEVICE = "spyre"
WARMUP = 5
ITERS = 50

# Effective HBM bw, measured single-shot on ReStickifyOpHBM in prior probes.
HBM_BW = 107e9


def predicted_delta_ms(M: int) -> float:
    """Cost model: T_hbm = 2 * |Y| / HBM_BW."""
    y_bytes = M * HD * 2
    return 2.0 * y_bytes / HBM_BW * 1e3


def time_compiled(fn, args) -> list[float]:
    """Wall-clock a compiled callable; force sync via .sum().item()."""
    for _ in range(WARMUP):
        out = fn(*args)
        _ = out.sum().item()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn(*args)
        _ = out.sum().item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return times


def main():
    sdsc_calls: Counter[str] = Counter()
    orig_sdsc = ac.SpyreAsyncCompile.sdsc
    current = {"label": None}

    def wrapped_sdsc(self, kernel_name, specs):
        if current["label"] is not None:
            sdsc_calls[current["label"]] += 1
        return orig_sdsc(self, kernel_name, specs)

    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("lx_planning", True),
        ts_config.patch("allow_all_ops_in_lx_planning", True),
        ts_config.patch("sencores", 32),
        patch.object(ac.SpyreAsyncCompile, "sdsc", wrapped_sdsc),
    ]
    for p in patchers:
        p.__enter__()
    torch.compiler.reset()

    rows = []
    try:
        def fn_a(X, W, Y):
            return (X @ W) + Y

        def fn_b(X, W, Yt):
            return (X @ W) + Yt.t()

        compiled_a = torch.compile(fn_a, fullgraph=True)
        compiled_b = torch.compile(fn_b, fullgraph=True)

        for M in M_VALUES:
            X = torch.rand((M, HD), dtype=DTYPE, device=DEVICE)
            W = torch.rand((HD, HD), dtype=DTYPE, device=DEVICE)
            Y = torch.rand((M, HD), dtype=DTYPE, device=DEVICE)
            Yt = torch.rand((HD, M), dtype=DTYPE, device=DEVICE)

            current["label"] = f"A_M{M}"
            try:
                times_a = time_compiled(compiled_a, (X, W, Y))
            except Exception as e:
                print(f"  M={M} graph A failed: {type(e).__name__}: {e}",
                      flush=True)
                current["label"] = None
                continue

            current["label"] = f"B_M{M}"
            try:
                times_b = time_compiled(compiled_b, (X, W, Yt))
            except Exception as e:
                print(f"  M={M} graph B failed: {type(e).__name__}: {e}",
                      flush=True)
                current["label"] = None
                continue
            current["label"] = None

            ta = statistics.median(times_a)
            tb = statistics.median(times_b)
            delta = tb - ta
            pred = predicted_delta_ms(M)
            ratio = delta / pred if pred > 0 else 0.0
            rows.append((M, M * HD * 2 / 1e6, ta, tb, delta, pred, ratio))
            print(f"  done M={M}: T_A={ta:.3f} T_B={tb:.3f} Δ={delta:.3f}",
                  flush=True)
    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)

    print()
    print(f"FUNDAMENTAL restickify cost probe — HD={HD}, dtype={DTYPE}")
    print(f"  WARMUP={WARMUP}, ITERS={ITERS}")
    print(f"  SENCORES={os.environ.get('SENCORES', '32')}, "
          f"LX_PLANNING={os.environ.get('LX_PLANNING', '?')}")
    print()
    print(f"  {'M':>6} {'|Y|MB':>7} {'T_A(ms)':>10} {'T_B(ms)':>10} "
          f"{'Δ(ms)':>9} {'Δ_pred(ms)':>12} {'Δ/Δ_pred':>10}")
    print("  " + "-" * 72)
    for M, mb, ta, tb, delta, pred, ratio in rows:
        print(f"  {M:>6} {mb:>7.1f} {ta:>10.3f} {tb:>10.3f} "
              f"{delta:>9.3f} {pred:>12.3f} {ratio:>10.2f}x")

    print()
    print("SDSC emissions per (graph, M):")
    for label in sorted(sdsc_calls):
        print(f"  {label}: {sdsc_calls[label]} kernels")

    print()
    print("Interpretation:")
    print("  Δ/Δ_pred ≈ 1.0    → cost model validated; ring would save ~96% of Δ")
    print("  Δ/Δ_pred >> 1.0   → restickify has extra overhead; bigger ring win")
    print("  Δ/Δ_pred << 1.0   → restickify partly streamed; smaller win")
    print("  A_M{i} == B_M{i}  → global optimizer eliminated restickify;")
    print("                       comparison degenerate; ring opportunity")
    print("                       already partially captured at IR level")


if __name__ == "__main__":
    main()
