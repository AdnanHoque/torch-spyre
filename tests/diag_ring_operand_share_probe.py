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

"""Pure ring-share probe — clean rebuild reverification.

Adapted from `diag_broadcast_lx_resident.py` on
`AdnanHoque/diag-cost-model-planner`. Measures the marginal wall-time
cost of adding more cores to a *pure operand-multicast* matmul:

  shape: M × K × N_per
  split: (1, n, 1)  with SENCORES = n
  → every core multiplies the SAME M×K of A against its own K×N_per
    slice of B
  → so A is shared across all n cores (broadcast); B is per-core unique
  → no K-cohort PSUM reduction (k=1)

Holding per-core compute and per-core unique B constant, the wall
delta from n=1 → n=N captures the *pure operand-multicast cost on
the RIU data ring* — separated from per-core HMI bandwidth (which
is constant per core).

Two phases by data-residency regime:
  Phase A: B is too big for LX (DRAM-bound)
  Phase B: everything fits in LX (LX-fit)

Cross-phase comparison: if Phase B's slope is much smaller than
Phase A's, that's evidence the original measurement was contaminated
by HMI traffic; Phase B isolates pure ring-share.

Usage:
    python /tmp/ring_share_probe.py
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
import torch._inductor.config as _icfg  # noqa: E402

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, "/home/adnan/dt-inductor/torch-spyre")

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402

try:
    from torch_spyre._inductor import work_division as _planner  # noqa: E402
except ImportError:
    from torch_spyre._inductor import core_division as _planner  # noqa: E402


WARMUP = 3
ITERS = 8
DTYPE = torch.float16
DTYPE_BYTES = 2

# 5-shape sweep — original probe used 6 (n in {1,2,4,8,16,32}); we keep
# all 6 for the regression but report 5 hop-deltas (n=2..32 vs n=1).
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


PHASES = [
    _Phase(name="DRAM-bound (orig sizing)", M=128, K=8192, N_per=256),
    _Phase(name="LX-fit (small operands)",  M=128, K=2048, N_per=128),
]


_orig_multi = _planner.multi_dim_iteration_space_split


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
    _planner.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _planner.multi_dim_iteration_space_split = _orig_multi


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


def main() -> int:
    print("# Pure ring-share probe — clean rebuild reverification\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16\n")

    all_results: dict[str, list[tuple[int, float]]] = {}

    for phase in PHASES:
        print(f"## Phase: {phase.name}")
        print(f"  M={phase.M}, K={phase.K}, N_per={phase.N_per}")
        print(f"  shared A     = {phase.shared_a_bytes // 1024} KB")
        print(f"  per-core B   = {phase.per_core_b_bytes // 1024} KB")
        print(f"  per-core C   = {phase.per_core_c_bytes // 1024} KB")
        print(f"  per-core sum = {phase.per_core_total_bytes // 1024} KB "
              f"(scratchpad limit: 2048 KB)")
        print()

        rows: list[tuple[int, float]] = []
        for n in NS:
            print(f"  n={n} (SENCORES={n}, N_total={n*phase.N_per}) ...",
                  end="", flush=True)
            ms, err = _bench_at_n(phase, n)
            if err:
                print(f"  ERR {err}")
                rows.append((n, float("nan")))
            else:
                print(f"  {ms:.3f} ms")
                rows.append((n, ms))
        all_results[phase.name] = rows
        ts_config.sencores = 32  # restore between phases
        print()

    # --- side-by-side comparison ---
    print("\n## Side-by-side comparison")
    print()
    print("| n | " + " | ".join(p.name for p in PHASES) + " |")
    print("|---:" + "|---:" * len(PHASES) + "|")
    for i, n in enumerate(NS):
        cells = [str(n)]
        for p in PHASES:
            ms = all_results[p.name][i][1]
            cells.append(f"{ms:.3f}" if not math.isnan(ms) else "err")
        print("| " + " | ".join(cells) + " |")

    # --- per-phase ring-fits ---
    print("\n## Per-phase ring-fit (Δ wall vs n=1)\n")
    for phase in PHASES:
        print(f"### {phase.name}")
        rows = all_results[phase.name]
        if math.isnan(rows[0][1]):
            print("  (n=1 baseline missing; skipping)")
            continue
        baseline = rows[0][1]
        deltas = [(n, ms - baseline) for n, ms in rows[1:]
                  if not math.isnan(ms)]
        if not deltas:
            print("  (no valid deltas)")
            continue
        # Fit Δ = a + b·n  (linear in n)
        a, b, rmse = _fit_linear([n for n, _ in deltas],
                                  [d for _, d in deltas])
        print(f"  Δ ≈ {a:+.3f} + {b:+.4f}·n ms (RMSE {rmse:.3f})")
        # Also fit Δ = a + b·log2(n)
        a2, b2, rmse2 = _fit_linear([math.log2(n) for n, _ in deltas],
                                     [d for _, d in deltas])
        print(f"  Δ ≈ {a2:+.3f} + {b2:+.4f}·log2(n) ms (RMSE {rmse2:.3f})")
        # Per-hop cost (us per added core)
        if b > 0:
            shared_mb = phase.shared_a_bytes / 1024 / 1024
            us_per_hop = b * 1000  # ms→µs
            print(f"  per-hop cost = {us_per_hop:.1f} us  "
                  f"(shared-A size = {shared_mb:.2f} MB)")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
