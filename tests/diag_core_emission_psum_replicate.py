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

"""Replication of the strongest K-split reorder signals.

Initial probe (`diag_core_emission_psum_chain.py`) found:
  L3-70B q_proj prefill (4, 1, 8): 1.038x reverse-win (max signal)
  L3-70B MLP down (2, 16, 1): 1.028x reverse-win (Part B)
  L3-70B MLP down (16, 2, 1): 1.018x reverse-win (Part B)
  L3-8B MLP down prefill (8, 1, 4): 1.013x reverse-win

To separate signal from noise: rerun the four configs with
  - 2× the iters (30 vs 15)
  - alternating mode order (def-first then rev-first) to control for
    in-process warmup effects
  - report median + iqr per trial so we can see dispersion

If signal holds across both trial orders, it's real and we have at
least one ring-aware reorder lever worth shipping. If it collapses
into noise, the original probe overfit a single run.
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
from torch_spyre._inductor.codegen import superdsc as _superdsc  # noqa: E402


WARMUP = 5
ITERS = 30
DTYPE = torch.float16

# (label, M, N, K, split)
TARGETS = [
    ("L3-70B q_proj prefill",   128, 8192, 8192,  (4, 1, 8)),
    ("L3-70B MLP down prefill", 128, 8192, 28672, (2, 16, 1)),
    ("L3-70B MLP down prefill", 128, 8192, 28672, (16, 2, 1)),
    ("L3-8B  MLP down prefill", 128, 4096, 14336, (8, 1, 4)),
]


# ---- machinery (same shape as the original probe) ---------------------

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
    samples_ms = [s * 1e3 for s in samples]
    samples_ms.sort()
    q1 = samples_ms[len(samples_ms) // 4]
    q3 = samples_ms[3 * len(samples_ms) // 4]
    return statistics.median(samples_ms), q3 - q1, min(samples_ms)


def _compile_and_bench(M: int, N: int, K: int, target, reverse: bool):
    ts_config.core_emission_reverse = reverse
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(target):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(target):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return ((None, None, None), f"{type(e).__name__}: {str(e)[:80]}")


def _run_one_trial(label: str, M, N, K, target, mode_order):
    """Run both modes in `mode_order` and return (def_med, rev_med, dispersion)."""
    results = {}
    dispersion = {}
    mins = {}
    for mode in mode_order:
        rev = (mode == "rev")
        (med, iqr, mn), err = _compile_and_bench(M, N, K, target, rev)
        if err:
            print(f"    {mode}: ERR {err}")
            results[mode] = None
            continue
        results[mode] = med
        dispersion[mode] = iqr
        mins[mode] = mn
        print(f"    {mode}:  median={med:.3f}  iqr={iqr:.3f}  min={mn:.3f}  ms")
    return results, dispersion, mins


def main() -> int:
    print("# K-split / output reorder REPLICATION\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32, fp16\n")
    print(
        "## Method: each config runs as TWO trials, def-first then "
        "rev-first, to control for in-process warm state.\n"
    )

    summary_rows = []
    for label, M, N, K, target in TARGETS:
        print(f"### {label} (M={M}, N={N}, K={K})  split={target}\n")
        print("  Trial 1 (default-first):")
        r1, _, _ = _run_one_trial(label, M, N, K, target,
                                   mode_order=["def", "rev"])
        print("  Trial 2 (reverse-first):")
        r2, _, _ = _run_one_trial(label, M, N, K, target,
                                   mode_order=["rev", "def"])
        if all(v is not None for v in r1.values()) and all(
            v is not None for v in r2.values()
        ):
            sp1 = r1["def"] / r1["rev"]
            sp2 = r2["def"] / r2["rev"]
            print(
                f"  Trial 1 speedup (def/rev): {sp1:.3f}x"
                f" → {('reverse wins' if sp1 > 1 else 'default wins')}"
            )
            print(
                f"  Trial 2 speedup (def/rev): {sp2:.3f}x"
                f" → {('reverse wins' if sp2 > 1 else 'default wins')}"
            )
            avg_sp = (sp1 + sp2) / 2
            print(f"  Mean speedup (avg of trials): {avg_sp:.3f}x")
            summary_rows.append(
                (label, target, r1["def"], r1["rev"], r2["def"], r2["rev"],
                 sp1, sp2, avg_sp)
            )
        print()

    print("## Replication summary\n")
    print("| shape | split | trial1 def | trial1 rev | trial1 sp | "
          "trial2 def | trial2 rev | trial2 sp | mean sp | consistent? |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        label, target, t1d, t1r, t2d, t2r, sp1, sp2, avg = row
        consistent = (
            "✓ same direction" if (sp1 - 1) * (sp2 - 1) > 0 else
            "✗ flipped"
        )
        print(
            f"| {label} | {target} | {t1d:.3f} | {t1r:.3f} | {sp1:.3f}x | "
            f"{t2d:.3f} | {t2r:.3f} | {sp2:.3f}x | {avg:.3f}x | "
            f"{consistent} |"
        )
    print()

    # Real-signal vs noise verdict
    print("## Verdict\n")
    real_signals = [
        r for r in summary_rows
        if (r[6] - 1) * (r[7] - 1) > 0 and abs((r[8] - 1)) >= 0.02
    ]
    if not summary_rows:
        print("  No valid measurements.")
        return 1
    if real_signals:
        print(f"  {len(real_signals)} of {len(summary_rows)} configs show "
              "consistent ≥2% reorder effect across both trial orders. "
              "Signal is real, not noise.")
        for r in real_signals:
            label, target = r[0], r[1]
            avg = r[8]
            print(f"    - {label} {target}: mean speedup {avg:.3f}x "
                  f"({'reverse' if avg > 1 else 'default'} wins)")
    else:
        print("  No config shows a consistent ≥2% reorder effect across "
              "both trial orders. The single-run signals from the initial "
              "probe were within noise. Reorder lever is dead.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
