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

"""Project A — PSUM-via-SFP-ring K-split probe.

The IBM AIU architecture doc (slide 30) says cross-core partial-sum
reduction uses a *dedicated* 32 B SFP ring, separate from the 128 B
data rings that carry operand streaming. Today's planner avoids
K-split because the cost-model accounting treats all cross-core
traffic as competing for the same bandwidth — but PSUM doesn't
compete with data movement, so K-split's "extra cost" might be much
cheaper in reality than the cost model thinks.

We already have hints from Phase 1.0 measurements that K-split wins
on at least two production shapes:

  L3-8B MLP down (128, 4096, 14336): (2, 1, 16) at 4.20 ms
                                vs (1, 32, 1) at 4.64 ms (+10%)
  Mixtral down (same shape): same pattern

This probe forces a representative set of splits per shape (mixing
N-split, K-split, and combinations) and tabulates wall time. Goal:
identify the shape characteristics that favor K-split so a planner
heuristic can pick correctly.

Run: python tests/diag_psum_split_probe.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

WARMUP = 3
ITERS = 20


# ---- child-mode bench --------------------------------------------------

def _run_child(M: int, N: int, K: int, target: tuple[int, int, int]) -> list[float]:
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

    import torch._inductor.config as _icfg
    _icfg.compile_threads = 1
    _icfg.worker_start_method = "fork"
    _icfg.fx_graph_cache = False
    _icfg.fx_graph_remote_cache = False

    import torch
    import torch_spyre
    torch_spyre._autoload()
    from torch_spyre import streams as _ts
    from torch_spyre._inductor import core_division as _core_div

    _orig_multi = _core_div.multi_dim_iteration_space_split

    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        prod = target[0] * target[1] * target[2]
        if prod != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}

    _core_div.multi_dim_iteration_space_split = _forced  # type: ignore[assignment]

    a = torch.randn(M, K, dtype=torch.float16, device="spyre")
    b = torch.randn(K, N, dtype=torch.float16, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    for _ in range(WARMUP):
        mm(a, b)
    _ts.synchronize()

    samples = []
    import time as _time
    for _ in range(ITERS):
        t0 = _time.perf_counter()
        mm(a, b)
        _ts.synchronize()
        samples.append(_time.perf_counter() - t0)
    return samples


def _child_mode():
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])
    target = tuple(int(x) for x in sys.argv[5].split(","))
    samples = _run_child(M, N, K, target)  # type: ignore[arg-type]
    print("__RESULTS__" + json.dumps(samples))


# ---- shapes + per-shape split lists ------------------------------------

# Each entry: (label, M, N, K, [splits])
# Splits chosen per shape to cover pure-N, slight-m, K-split variants,
# and pure-K. Stick alignment limits which splits are valid.

PROBE = [
    (
        "L3-8B MLP down prefill",
        128, 4096, 14336,
        [
            (1, 32, 1),   # pure N — what element_priority picks today
            (2, 16, 1),   # slight m, mostly N
            (2, 1, 16),   # Phase 1.0 empirical best — K-split with slight m
            (1, 4, 8),    # mixed N+K
            (1, 1, 32),   # pure K — full reduction-axis parallelism
        ],
    ),
    (
        "L3-70B q_proj prefill",
        128, 8192, 8192,
        [
            (1, 32, 1),   # pure N — current best after element_priority
            (2, 16, 1),   # Phase 1.0 best
            (1, 4, 8),    # mixed
            (1, 1, 32),   # pure K
        ],
    ),
    (
        "L3-8B q_proj prefill",
        128, 4096, 4096,
        [
            (1, 32, 1),   # current best (element_priority)
            (1, 8, 4),    # mixed
            (1, 4, 8),    # more K
            (1, 1, 32),   # pure K
        ],
    ),
    (
        "L3-70B GQA kv_proj prefill",
        128, 1024, 8192,
        [
            (1, 16, 2),   # pure-N max + slight K
            (2, 16, 1),   # Phase 1.0 best
            (1, 8, 4),    # more K
            (1, 4, 8),    # even more K
            (1, 1, 32),   # pure K
        ],
    ),
    (
        "L3-70B GQA TP=8 prefill",
        128, 128, 8192,
        [
            (1, 2, 16),   # N max + K
            (1, 1, 32),   # pure K — Phase 1.0 (32,1,1) was best, this is close
            (2, 2, 8),    # mixed
            (16, 2, 1),   # high m
            (32, 1, 1),   # pure M (Phase 1.0 best)
        ],
    ),
    (
        "Synthetic small-N (128, 512, 32768)",
        128, 512, 32768,
        [
            (1, 8, 4),    # pure-N max
            (1, 4, 8),
            (1, 2, 16),
            (1, 1, 32),   # pure K — predicted sweet spot
            (2, 4, 4),    # mixed
        ],
    ),
]


def _run_parent():
    print("# Project A — PSUM-via-SFP-ring K-split probe\n")
    n_total = sum(len(splits) for _, _, _, _, splits in PROBE)
    print(f"# {len(PROBE)} shapes, {n_total} (shape, split) measurements")
    print(f"# warmup={WARMUP} iters={ITERS}\n")

    results: dict[tuple[str, tuple[int, int, int]], list[float]] = {}
    n_done = 0
    t_start = time.time()

    for label, M, N, K, splits in PROBE:
        print(f"\n## {label} ({M}, {N}, {K})")
        for target in splits:
            n_done += 1
            elapsed = time.time() - t_start
            print(f"  [{n_done}/{n_total} t={elapsed:.0f}s] {target} ...",
                  end="", flush=True)
            env = os.environ.copy()
            target_str = ",".join(str(x) for x in target)
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--child",
                     str(M), str(N), str(K), target_str],
                    env=env, capture_output=True, text=True, timeout=300,
                )
            except subprocess.TimeoutExpired:
                print("  TIMEOUT")
                continue

            samples = None
            for line in proc.stdout.split("\n"):
                if line.startswith("__RESULTS__"):
                    samples = json.loads(line[len("__RESULTS__"):])
                    break

            if samples is None:
                tail = proc.stderr[-200:] if proc.stderr else "(empty)"
                print(f"  FAIL ...{tail}")
                continue

            ms = statistics.median(samples) * 1e3
            print(f"  {ms:.3f} ms")
            results[(label, target)] = samples

    # --- per-shape table ---
    print("\n\n## Per-shape wall time and speedup vs pure-N pick\n")
    for label, M, N, K, splits in PROBE:
        print(f"\n### {label} ({M}, {N}, {K})\n")
        # Use the FIRST split as the baseline (it's pure-N or N-heavy by design)
        baseline = splits[0]
        baseline_samples = results.get((label, baseline))
        if baseline_samples is None:
            print(f"  baseline {baseline} failed; skipping table")
            continue
        baseline_ms = statistics.median(baseline_samples) * 1e3
        print(f"  Baseline: {baseline} = {baseline_ms:.3f} ms\n")
        print("  | split | wall ms | speedup vs baseline | k? |")
        print("  |---|---:|---:|---|")
        for target in splits:
            samples = results.get((label, target))
            if samples is None:
                print(f"  | {target} | err | — | — |")
                continue
            ms = statistics.median(samples) * 1e3
            speedup = baseline_ms / ms
            is_ksplit = "✓" if target[2] > 1 else ""
            flag = ""
            if speedup >= 1.05:
                flag = " ✓"
            elif speedup <= 0.95:
                flag = " ✗"
            print(f"  | {target} | {ms:.3f} | {speedup:.3f}x{flag} | {is_ksplit} |")

    # --- summary ---
    print("\n\n## Summary: best split per shape\n")
    print("| shape | best split | best ms | baseline (pure-N) | speedup | "
          "k-split wins? |")
    print("|---|---|---:|---:|---:|---|")
    for label, M, N, K, splits in PROBE:
        baseline = splits[0]
        baseline_samples = results.get((label, baseline))
        if baseline_samples is None:
            print(f"| {label} | err | err | err | err | err |")
            continue
        baseline_ms = statistics.median(baseline_samples) * 1e3
        best_split = baseline
        best_ms = baseline_ms
        for target in splits:
            samples = results.get((label, target))
            if samples is None:
                continue
            ms = statistics.median(samples) * 1e3
            if ms < best_ms:
                best_ms = ms
                best_split = target
        speedup = baseline_ms / best_ms
        is_ksplit_win = "**YES**" if best_split[2] > 1 else "no"
        print(f"| {label} | {best_split} | {best_ms:.3f} | "
              f"{baseline_ms:.3f} | {speedup:.3f}x | {is_ksplit_win} |")

    print(f"\n\n# Total wall time: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_mode()
    else:
        sys.exit(_run_parent())
