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

"""Pure ring-share probe — separates ring-share cost from HMI contention.

Background: the original broadcast-topology probe measured ring-linear
behaviour and reported ~30 us per MB of broadcast operand at ~67 GB/s
per-link effective bandwidth. The IBM AIU architecture doc confirms the
ring is real (dual counter-rotating rings, 128 B each), AND that HMI
(DRAM interface) is a node on the same data ring. So the original
67 GB/s figure is a *combined* ring-share + HMI-stream cost — the same
ring that broadcasts A across cores is also moving per-core unique B
chunks from DRAM concurrently.

This probe runs two side-by-side sweeps:

  Phase A (DRAM-bound, same shape as original probe):
    M=128, K=8192, N_per=256
    per-core unique B = 4 MB (way over 2 MB LX scratchpad → must stream)

  Phase B (LX-fit, all per-core data fits in 2 MB scratchpad):
    M=128, K=2048, N_per=128
    shared A  = 512 KB
    per-core B = 512 KB
    per-core C = 32 KB
    per-core total = ~1 MB

In both phases we sweep n in {1, 2, 4, 8, 16, 32} with forced (1, n, 1)
split, holding per-core compute and per-core unique data constant
within each phase. Cross-phase comparison: if Phase B's broadcast slope
(us / MB / hop) is much smaller than Phase A's, that confirms HMI
contention dominated the original number and gives us the *true* pure
ring-share cost for any future cost-model term.

Run: python tests/diag_broadcast_lx_resident.py
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


WARMUP = 5
ITERS = 20
DTYPE = torch.float16
DTYPE_BYTES = 2

NS = [1, 2, 4, 8, 16, 32]


@dataclass
class _Phase:
    name: str
    M: int
    K: int
    N_per: int

    @property
    def shared_a_bytes(self) -> int:
        return self.M * self.K * DTYPE_BYTES

    @property
    def per_core_b_bytes(self) -> int:
        return self.K * self.N_per * DTYPE_BYTES

    @property
    def per_core_c_bytes(self) -> int:
        return self.M * self.N_per * DTYPE_BYTES

    @property
    def per_core_total_bytes(self) -> int:
        return (self.shared_a_bytes
                + self.per_core_b_bytes
                + self.per_core_c_bytes)

    @property
    def per_core_flops(self) -> int:
        return 2 * self.M * self.N_per * self.K


PHASES = [
    _Phase(name="DRAM-bound (original sizing)",
           M=128, K=8192, N_per=256),
    _Phase(name="LX-fit (small operands)",
           M=128, K=2048, N_per=128),
]


# ---- force-split machinery (same as original probe) ---------------------

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target):
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
def _force_split(target):
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


def _bench_at_n(phase: _Phase, n: int) -> tuple[float, str]:
    ts_config.sencores = n
    N_total = n * phase.N_per
    a = torch.randn(phase.M, phase.K, dtype=DTYPE, device="spyre")
    b = torch.randn(phase.K, N_total, dtype=DTYPE, device="spyre")
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
        return float("nan"), f"{type(e).__name__}: {str(e)[:60]}"


# ---- model fits ---------------------------------------------------------

def _fit_linear(xs, ys):
    n = len(xs)
    if n < 2:
        return ys[0] if ys else 0.0, 0.0, 0.0
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

def main() -> int:
    print("# Pure ring-share probe (LX-resident vs DRAM-bound)\n")

    all_results: dict[str, list[tuple[int, float]]] = {}

    for phase in PHASES:
        print(f"## Phase: {phase.name}")
        print(f"  M={phase.M}, K={phase.K}, N_per={phase.N_per}")
        print(f"  shared A     = {phase.shared_a_bytes // 1024} KB")
        print(f"  per-core B   = {phase.per_core_b_bytes // 1024} KB")
        print(f"  per-core C   = {phase.per_core_c_bytes // 1024} KB")
        print(f"  per-core sum = {phase.per_core_total_bytes // 1024} KB "
              f"(scratchpad limit: 2048 KB)")
        print(f"  per-core flops = {phase.per_core_flops:,}")
        print()

        rows: list[tuple[int, float]] = []
        for n in NS:
            print(f"  n={n} (SENCORES={n}, N_total={n*phase.N_per}) ...",
                  end="", flush=True)
            ms, err = _bench_at_n(phase, n)
            if err:
                print(f"  ERR {err}")
            else:
                print(f"  {ms:.3f} ms")
            rows.append((n, ms))
        all_results[phase.name] = rows
        ts_config.sencores = 32  # restore between phases
        print()

    # --- emit comparison table ---
    print("\n## Side-by-side comparison\n")
    header = "| n | " + " | ".join(p.name for p in PHASES) + " |"
    sep = "|---:" + "|---:" * len(PHASES) + "|"
    print(header)
    print(sep)
    for i, n in enumerate(NS):
        cells = [str(n)]
        for p in PHASES:
            ms = all_results[p.name][i][1]
            cells.append(f"{ms:.3f}" if not math.isnan(ms) else "err")
        print("| " + " | ".join(cells) + " |")

    # --- per-phase model fits ---
    print("\n## Per-phase ring-fit (Δ wall vs n=1)\n")
    for p in PHASES:
        rows = all_results[p.name]
        valid = [(n, ms) for (n, ms) in rows
                 if not math.isnan(ms)]
        if len(valid) < 3:
            print(f"  {p.name}: insufficient data")
            continue
        base = valid[0][1]
        ns = [r[0] for r in valid]
        ds = [r[1] - base for r in valid]
        a_lin, b_lin, rmse_lin = _fit_linear(ns, ds)
        a_log, b_log, rmse_log = _fit_linear(
            [math.log2(n) for n in ns], ds)
        # bytes broadcast = shared A
        bytes_mb = p.shared_a_bytes / (1024 * 1024)
        per_hop_us_per_mb = (b_lin * 1000) / max(bytes_mb, 1e-6)
        print(f"  {p.name}:")
        print(f"    Δ ≈ {a_lin:+.3f} + {b_lin:+.4f}·n ms  "
              f"(RMSE {rmse_lin:.3f})")
        print(f"    Δ ≈ {a_log:+.3f} + {b_log:+.4f}·log2(n) ms  "
              f"(RMSE {rmse_log:.3f})")
        print(f"    per-hop cost = {b_lin*1000:.1f} us  "
              f"(operand size = {bytes_mb:.2f} MB)")
        print(f"    per-hop per MB = {per_hop_us_per_mb:.1f} us/MB")
        print()

    # --- verdict ---
    print("## Verdict\n")
    if len(PHASES) >= 2:
        a_rows = all_results[PHASES[0].name]
        b_rows = all_results[PHASES[1].name]
        a_valid = [r for r in a_rows if not math.isnan(r[1])]
        b_valid = [r for r in b_rows if not math.isnan(r[1])]
        if a_valid and b_valid:
            a_base = a_valid[0][1]
            b_base = b_valid[0][1]
            a_max_delta = max(r[1] - a_base for r in a_valid)
            b_max_delta = max(r[1] - b_base for r in b_valid)
            print(f"  DRAM-bound max Δ at n=32:  "
                  f"{a_max_delta:.3f} ms")
            print(f"  LX-fit max Δ at n=32:      "
                  f"{b_max_delta:.3f} ms")
            if b_max_delta < a_max_delta * 0.5:
                print()
                print("  LX-fit broadcast cost is <50% of DRAM-bound. "
                      "HMI contention contributed substantially to the "
                      "original number — pure ring-share is faster than "
                      "the 30 us/MB combined cost suggested.")
            elif b_max_delta > a_max_delta * 0.8:
                print()
                print("  LX-fit broadcast cost is similar to DRAM-bound. "
                      "Ring-share dominates regardless of HMI traffic — "
                      "the original 30 us/MB IS approximately the pure "
                      "ring-share cost.")
            else:
                print()
                print("  LX-fit cost is moderately lower than DRAM-bound. "
                      "Ring-share and HMI contention BOTH contribute to "
                      "the original number; pure ring-share is somewhat "
                      "faster but not negligible.")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_broadcast_lx_resident_results.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
