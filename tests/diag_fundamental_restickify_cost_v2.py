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

"""FUNDAMENTAL restickify cost probe v2 — matmul-forced restickify.

Probe v1 (pointwise consumer `(X@W) + Y.t()`) found the global restickify
optimizer absorbs the FUNDAMENTAL by picking a matmul output STL that
aligns with the transposed input. Pointwise ops are AllSameNode in the
optimizer's cost graph (free output STL) so it has degrees of freedom to
avoid restickifying.

Matmul is a FixedInOutNode (see optimize_restickify.py): fixed input and
output STL requirements per matmul plan. A transposed matmul input forces
a restickify regardless of optimizer choices. This is the same mechanism
tests/inductor/test_restickify.py::test_matmul_xt_y verifies — it asserts
`optimal_cost = x.numel()` (one full restickify of x).

  Graph A: torch.matmul(X1, Y)        — X1:(M,HD) contiguous, no restickify
  Graph B: torch.matmul(X2.t(), Y)    — X2:(HD,M) contiguous, .t() forces restickify

Y:(HD,HD) in both. Output (M,HD) in both. Same FLOPs M*HD^2. Same input
element count M*HD. The only difference is the storage layout of the
first input.

Verification: SPYRE_CAPTURE_RESTICKIFY_PLAN=1 captures the plan; we
assert plan_cost_A == 0 and plan_cost_B == M*HD (matches the test).
Predicted timing delta: 2 * M * HD * 2 bytes / 107 GB/s.

Run:  SENCORES=32 LX_PLANNING=1 .venv/bin/python tests/diag_fundamental_restickify_cost_v2.py
"""

from __future__ import annotations

import math
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

import torch_spyre._inductor.insert_restickify as _insert_restickify
from torch._inductor import config as t_inductor_config
from torch_spyre._inductor import config as ts_config
from torch_spyre.execution import async_compile as ac


HD = 4096
M_VALUES = [128, 512, 2048, 8192]
DTYPE = torch.float16
DEVICE = "spyre"
WARMUP = 5
ITERS = 50

# Effective HBM bw, measured single-shot on ReStickifyOpHBM in prior probes.
HBM_BW = 107e9


def predicted_delta_ms(M: int) -> float:
    """Cost model: T_hbm = 2 * |X| / HBM_BW where |X| = M*HD*2 bytes."""
    x_bytes = M * HD * 2
    return 2.0 * x_bytes / HBM_BW * 1e3


def plan_cost(plan: dict) -> int:
    """Total elements across all entries in a restickify plan."""
    return sum(
        math.prod(int(s) for s in entry["target_layout"].size)
        for entries in plan.values()
        for entry in entries
    )


def time_compiled(fn, args) -> list[float]:
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
    kernel_names: Counter = Counter()
    orig_sdsc = ac.SpyreAsyncCompile.sdsc
    current = {"label": None}

    def wrapped_sdsc(self, kernel_name, specs):
        if current["label"] is not None:
            kernel_names[(current["label"], kernel_name)] += 1
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
    plan_costs: dict[str, int] = {}

    def fn_a(X1, Y):
        return torch.matmul(X1, Y)

    def fn_b(X2, Y):
        return torch.matmul(X2.t(), Y)

    def fresh_compile(fn):
        # Mirror tests/inductor/utils_inductor.py::_compile_and_run: clear all
        # caches so each M gets a fresh static compile (no dynamic-shape leak).
        torch._dynamo.reset_code_caches()
        torch._inductor.codecache.FxGraphCache.clear()
        torch.compiler.reset()
        return torch.compile(fn, fullgraph=True)

    try:
        for M in M_VALUES:
            X1 = torch.rand((M, HD), dtype=DTYPE, device=DEVICE)
            X2 = torch.rand((HD, M), dtype=DTYPE, device=DEVICE)
            Y = torch.rand((HD, HD), dtype=DTYPE, device=DEVICE)

            # ---- Graph A: fresh compile, capture plan, time ----
            compiled_a = fresh_compile(fn_a)
            current["label"] = f"A_M{M}"
            _insert_restickify.restickify_plan = {}
            os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = "1"
            try:
                times_a = time_compiled(compiled_a, (X1, Y))
            except Exception as e:
                print(f"  M={M} graph A failed: {type(e).__name__}: {e}",
                      flush=True)
                os.environ.pop("SPYRE_CAPTURE_RESTICKIFY_PLAN", None)
                current["label"] = None
                continue
            os.environ.pop("SPYRE_CAPTURE_RESTICKIFY_PLAN", None)
            plan_costs[f"A_M{M}"] = plan_cost(_insert_restickify.restickify_plan)

            # ---- Graph B: fresh compile, capture plan, time ----
            compiled_b = fresh_compile(fn_b)
            current["label"] = f"B_M{M}"
            _insert_restickify.restickify_plan = {}
            os.environ["SPYRE_CAPTURE_RESTICKIFY_PLAN"] = "1"
            try:
                times_b = time_compiled(compiled_b, (X2, Y))
            except Exception as e:
                print(f"  M={M} graph B failed: {type(e).__name__}: {e}",
                      flush=True)
                os.environ.pop("SPYRE_CAPTURE_RESTICKIFY_PLAN", None)
                current["label"] = None
                continue
            os.environ.pop("SPYRE_CAPTURE_RESTICKIFY_PLAN", None)
            plan_costs[f"B_M{M}"] = plan_cost(_insert_restickify.restickify_plan)
            current["label"] = None

            ta = statistics.median(times_a)
            tb = statistics.median(times_b)
            delta = tb - ta
            pred = predicted_delta_ms(M)
            ratio = delta / pred if pred > 0 else 0.0
            rows.append((M, M * HD * 2 / 1e6, ta, tb, delta, pred, ratio))
            print(
                f"  done M={M}: T_A={ta:.3f} T_B={tb:.3f} Δ={delta:.3f} "
                f"planA={plan_costs[f'A_M{M}']} planB={plan_costs[f'B_M{M}']}",
                flush=True,
            )

    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)

    print()
    print(f"FUNDAMENTAL restickify cost probe v2 — HD={HD}, dtype={DTYPE}")
    print(f"  pattern A: torch.matmul(X1:(M,HD) , Y:(HD,HD))      "
          f"# expected plan_cost == 0")
    print(f"  pattern B: torch.matmul(X2:(HD,M).t(), Y:(HD,HD))   "
          f"# expected plan_cost == M*HD (FUNDAMENTAL)")
    print(f"  WARMUP={WARMUP}, ITERS={ITERS}, "
          f"SENCORES={os.environ.get('SENCORES', '32')}, "
          f"LX_PLANNING={os.environ.get('LX_PLANNING', '?')}")
    print()
    print(f"  {'M':>6} {'|X|MB':>7} {'T_A(ms)':>10} {'T_B(ms)':>10} "
          f"{'Δ(ms)':>9} {'Δ_pred(ms)':>12} {'Δ/Δ_pred':>10}")
    print("  " + "-" * 72)
    for M, mb, ta, tb, delta, pred, ratio in rows:
        print(
            f"  {M:>6} {mb:>7.1f} {ta:>10.3f} {tb:>10.3f} "
            f"{delta:>9.3f} {pred:>12.3f} {ratio:>10.2f}x"
        )

    print()
    print("Restickify plan verification (elements):")
    print(f"  {'label':<10} {'cost':>12} {'expected':>12}  status")
    for label in sorted(plan_costs):
        m_label = int(label.split("M")[1])
        expected = m_label * HD if label.startswith("B") else 0
        cost = plan_costs[label]
        status = "OK" if cost == expected else "MISMATCH"
        print(f"  {label:<10} {cost:>12} {expected:>12}  {status}")

    print()
    print("Kernel names per (graph, M):")
    for (label, kname), count in sorted(kernel_names.items()):
        print(f"  {label:<10} {kname}: {count}")

    print()
    print("Interpretation:")
    print("  plan_B == M*HD       → matmul forced restickify, FUNDAMENTAL confirmed")
    print("  Δ/Δ_pred ≈ 1.0       → cost model validated; ring would save ~96% of Δ")
    print("  Δ/Δ_pred >> 1.0      → restickify carries extra overhead; bigger ring win")
    print("  Δ/Δ_pred << 1.0      → restickify partly streamed; smaller win")


if __name__ == "__main__":
    main()
