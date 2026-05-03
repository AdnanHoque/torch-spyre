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

"""Pairwise-distance probe — directly verify monotonic core_id ↔ ring position.

The kv_proj (1, 16, 2) result (block_cyclic giving 2.76× over identity)
is consistent with the linear ring-distance model — but doesn't prove
it. This probe varies the K-pair ring distance d ∈ {1, 2, 4, 8, 16}
on the SAME shape and split, by constructing permutations that put
EVERY K-pair at exactly distance d apart.

If wall time scales linearly with d → core_id 0..31 is monotonically
adjacent on the physical ring (the assumption every prior measurement
relied on). Direct verification.

If non-monotonic / plateaus → core_id ↔ physical mapping is more
complex, and the prior chain-distance arguments need revisiting.

Predicted (under linear model):
  d=1:  ~3.94 ms  (block_cyclic baseline)
  d=2:  ~4.4 ms
  d=4:  ~5.4 ms
  d=8:  ~7.4 ms
  d=16: ~10.9 ms (identity baseline)
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
import sys

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402
from torch_spyre._inductor import core_division as _core_div  # noqa: E402


WARMUP = 5
ITERS = 30
DTYPE = torch.float16

# kv_proj-style shape: M=2048, N=1024 (16 sticks), K=8192. Split (1, 16, 2).
M, N, K = 2048, 1024, 8192
SPLIT = (1, 16, 2)

# Test d ∈ {1, 2, 4, 8, 16}. d=1 should match block_cyclic; d=16 = identity.
DISTANCES = [1, 2, 4, 8, 16]
PERMUTATIONS = [f"ring_pair_d{d}" for d in DISTANCES]


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


def _bench(fn) -> tuple[float, float, float]:
    """Returns (median_ms, iqr_ms, min_ms)."""
    for _ in range(WARMUP):
        fn()
    _ts.synchronize()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        fn()
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    samples_ms = sorted(s * 1e3 for s in samples)
    q1 = samples_ms[len(samples_ms) // 4]
    q3 = samples_ms[3 * len(samples_ms) // 4]
    return statistics.median(samples_ms), q3 - q1, min(samples_ms)


def _compile_and_bench(perm: str):
    ts_config.core_id_permutation = perm
    ts_config.core_emission_reverse = False
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(SPLIT):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(SPLIT):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return ((None, None, None), f"{type(e).__name__}: {str(e)[:100]}")


def main() -> int:
    print("# Pairwise-distance probe — wall time vs K-pair ring distance d\n")
    print(f"# shape: M={M} N={N} K={K}  split={SPLIT}  fp16  SENCORES=32")
    print(f"# warmup={WARMUP} iters={ITERS}\n")

    # Two trial orders: forward sweep and reverse sweep.
    results: dict[tuple[int, str], tuple[float, float, float]] = {}
    for tname, ordered in (("trial1", DISTANCES),
                            ("trial2", list(reversed(DISTANCES)))):
        print(f"## {tname} (order: d={ordered})\n")
        for d in ordered:
            perm = f"ring_pair_d{d}"
            (med, iqr, mn), err = _compile_and_bench(perm)
            if err:
                print(f"  d={d:2d}: ERR {err[:80]}")
                results[(d, tname)] = (None, None, None)
            else:
                print(f"  d={d:2d}: median={med:.3f}  iqr={iqr:.3f}  "
                      f"min={mn:.3f}  ms")
                results[(d, tname)] = (med, iqr, mn)
        print()

    # --- summary table ---
    print("\n## Summary — wall time vs K-pair ring distance\n")
    print("| d | trial1 median | trial2 median | mean median | trial1 min | trial2 min |")
    print("|---:|---:|---:|---:|---:|---:|")
    for d in DISTANCES:
        m1 = results[(d, "trial1")][0]
        m2 = results[(d, "trial2")][0]
        if m1 is None or m2 is None:
            print(f"| {d} | ERR | ERR | — | — | — |")
            continue
        avg = (m1 + m2) / 2
        mn1 = results[(d, "trial1")][2]
        mn2 = results[(d, "trial2")][2]
        print(f"| {d} | {m1:.3f} | {m2:.3f} | **{avg:.3f}** | "
              f"{mn1:.3f} | {mn2:.3f} |")
    print()

    # --- linearity check ---
    print("## Linearity check — fit y = a + b·d to mean medians\n")
    pts = []
    for d in DISTANCES:
        m1 = results[(d, "trial1")][0]
        m2 = results[(d, "trial2")][0]
        if m1 is None or m2 is None:
            continue
        pts.append((d, (m1 + m2) / 2))

    if len(pts) >= 3:
        # least-squares fit
        n = len(pts)
        sum_x = sum(p[0] for p in pts)
        sum_y = sum(p[1] for p in pts)
        sum_xx = sum(p[0] ** 2 for p in pts)
        sum_xy = sum(p[0] * p[1] for p in pts)
        b = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
        a = (sum_y - b * sum_x) / n
        # residuals
        rmse = (
            sum((p[1] - (a + b * p[0])) ** 2 for p in pts) / n
        ) ** 0.5

        print(f"  Linear fit: wall_ms ≈ {a:.3f} + {b:.4f} · d")
        print(f"  RMSE: {rmse:.4f} ms over {n} points")
        print()
        print("  Per-point residuals:")
        for d, y in pts:
            pred = a + b * d
            res = y - pred
            print(f"    d={d:2d}: actual={y:.3f}  predicted={pred:.3f}  "
                  f"residual={res:+.3f}")

        # Quality of fit
        max_res = max(abs(p[1] - (a + b * p[0])) for p in pts)
        if max_res < 0.1:
            print(
                "\n  VERDICT: Wall time is highly linear in K-pair distance "
                "(max residual <0.1 ms). Direct verification of monotonic "
                "core_id ↔ physical ring position. Sequential placement is "
                "optimal for K-chain-shortening modulo runtime constraints."
            )
        elif max_res < 0.5:
            print(
                "\n  VERDICT: Approximately linear (max residual <0.5 ms). "
                "Linear ring-distance model is a good first-order description; "
                "small deviations may come from per-hop fixed costs or other "
                "secondary effects."
            )
        else:
            print(
                "\n  VERDICT: NOT linear. core_id ↔ physical mapping is "
                "more complex. The chain-distance arguments used elsewhere "
                "in this project may need revisiting."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
