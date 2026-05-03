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

"""Phase 0 broadcast-topology probe.

Goal: determine how Spyre's cross-core operand sharing scales with the
number of cores receiving the same operand. Three candidate models:

  Ring/chain : t_share = (n - 1) * t_hop      → linear in n
  Tree       : t_share = log2(n) * t_hop      → log in n
  Bus        : t_share = constant             → flat in n

Method: hold per-core work (compute + per-core unique B + per-core C)
constant while varying `n`, the number of cores broadcasting the same A.

  matmul: C[M, n*N_per] = A[M, K] @ B[K, n*N_per]
  forced split: (m=1, n, k=1)
  SENCORES = n         — only n cores active per run

Per-core slice for every n:
  - compute  = M * N_per * K     (constant)
  - B unique = K * N_per * 2     (constant)
  - C unique = M * N_per * 2     (constant)
  - A shared = M * K * 2         (this is what gets broadcast across n)

So: wall(n) - wall(1) ≈ broadcast cost for fanning A out to n cores.
Plot it. Linear → ring. Log → tree. Flat → bus.

Run: python tests/diag_broadcast_topology.py
"""

from __future__ import annotations

import math
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre
torch_spyre._autoload()
from torch_spyre import streams as _ts
from torch_spyre._inductor import config as ts_config
from torch_spyre._inductor import core_division as _core_div


# ---- shape design --------------------------------------------------------

# Fixed per-core work. Tuned so that:
#   1. per-core compute is well above launch floor (~3 ms) for the smallest
#      n, so we're not just measuring launch overhead;
#   2. shared A is large enough (1+ MB) that broadcast cost is observable;
#   3. valid stick-alignment for all n in {1, 2, 4, 8, 16, 32}.

M = 128
K = 8192
N_PER = 256       # per-core N-band width (must be stick-aligned for fp16)

NS = [1, 2, 4, 8, 16, 32]
WARMUP = 3
ITERS = 20

DTYPE = torch.float16
DTYPE_BYTES = 2

A_SHARED_BYTES = M * K * DTYPE_BYTES                # 2 MB at K=8192
B_PER_CORE_BYTES = K * N_PER * DTYPE_BYTES          # 4 MB at K=8192,N=256
C_PER_CORE_BYTES = M * N_PER * DTYPE_BYTES          # 64 KB
PER_CORE_FLOPS = 2 * M * N_PER * K                  # ~134 MFLOPs


# ---- force-split machinery ---------------------------------------------

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target: tuple[int, int, int]):
    """Force the planner to return `target` for any matmul iteration space
    whose product equals max_cores."""
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        prod = target[0] * target[1] * target[2]
        if prod != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target: tuple[int, int, int]):
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


# ---- bench primitive ---------------------------------------------------

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


def _bench_at_n(n: int) -> tuple[float, str]:
    """Compile and bench (M, n*N_PER, K) with forced (1, n, 1) split using
    n active cores. Returns (median_ms, error_or_empty_str)."""
    ts_config.sencores = n
    N_total = n * N_PER

    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N_total, dtype=DTYPE, device="spyre")

    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split((1, n, 1)):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split((1, n, 1)):
                mm(a, b)

        ms = _bench(step)
        return ms, ""
    except Exception as e:  # noqa: BLE001
        return float("nan"), f"{type(e).__name__}: {str(e)[:80]}"


# ---- model fits ---------------------------------------------------------

def _fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares fit y = a + b*x. Returns (a, b, RMSE)."""
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return ys[0], 0.0, 0.0
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    rmse = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rmse


# ---- main --------------------------------------------------------------

@dataclass
class _Row:
    n: int
    wall_ms: float
    error: str = ""


def main() -> int:
    print(f"# Broadcast-topology probe")
    print(f"# matmul (M={M}, n*N_per={N_PER}*n, K={K}), forced (1, n, 1)")
    print(f"# per-core compute = {PER_CORE_FLOPS:,} flops "
          f"(constant across n)")
    print(f"# per-core unique B = {B_PER_CORE_BYTES//1024} KB; "
          f"shared A = {A_SHARED_BYTES//1024} KB")
    print(f"# warmup={WARMUP} iters={ITERS}")
    print()

    rows: list[_Row] = []
    for n in NS:
        print(f"# n={n} (SENCORES={n}, N_total={n*N_PER}) "
              f"…", end="", flush=True)
        ms, err = _bench_at_n(n)
        if err:
            print(f"  ERR {err}")
        else:
            print(f"  {ms:.3f} ms")
        rows.append(_Row(n=n, wall_ms=ms, error=err))

    # Restore SENCORES for any later code in the same process.
    ts_config.sencores = 32

    print()
    print("## Results table\n")
    print("| n cores | wall ms | Δ vs n=1 |")
    print("|---:|---:|---:|")
    valid = [r for r in rows if not r.error and not math.isnan(r.wall_ms)]
    if not valid:
        print("(no valid measurements)")
        return 1
    base = valid[0].wall_ms
    for r in rows:
        if r.error or math.isnan(r.wall_ms):
            print(f"| {r.n} | err | — |")
            continue
        delta = r.wall_ms - base
        print(f"| {r.n} | {r.wall_ms:.3f} | {delta:+.3f} ms |")

    # Fit linear (ring) and log (tree) models to wall vs n, normalized
    # by subtracting wall(n=1) so both fits start at 0.
    ns = [r.n for r in valid]
    ds = [r.wall_ms - base for r in valid]
    log_ns = [math.log2(n) for n in ns]

    a_lin, b_lin, rmse_lin = _fit_linear(ns, ds)
    a_log, b_log, rmse_log = _fit_linear(log_ns, ds)

    print()
    print("## Model fits to (Δ wall) vs n\n")
    print(f"  Ring model   (Δ ≈ {a_lin:+.3f} + {b_lin:+.4f} * n) "
          f"RMSE = {rmse_lin:.3f} ms")
    print(f"  Tree model   (Δ ≈ {a_log:+.3f} + {b_log:+.4f} * log2(n)) "
          f"RMSE = {rmse_log:.3f} ms")

    print()
    print("## Verdict\n")
    if max(ds) < 0.3:
        print("  Δ across the full sweep is < 0.3 ms — broadcast cost is "
              "either negligible or fully hidden by compute pipelining. "
              "Consistent with bus-broadcast or aggressive overlap. "
              "Cannot distinguish ring/tree from this data.")
    elif rmse_lin < rmse_log * 0.7:
        print(f"  Linear fit is materially better (RMSE {rmse_lin:.3f} vs "
              f"{rmse_log:.3f}). Consistent with **ring/chain** broadcast "
              f"with t_hop ≈ {b_lin*1000:.1f} μs per A-broadcast.")
    elif rmse_log < rmse_lin * 0.7:
        print(f"  Log fit is materially better (RMSE {rmse_log:.3f} vs "
              f"{rmse_lin:.3f}). Consistent with **tree** broadcast with "
              f"t_hop ≈ {b_log*1000:.1f} μs.")
    else:
        print(f"  Linear and log fits within 30% of each other "
              f"(RMSE {rmse_lin:.3f} vs {rmse_log:.3f}). Inconclusive — "
              f"need more data points or a different probe design.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
