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

"""K/N ratio sweep — find where the K-split heuristic should fire.

Phase 1.0 + the Project A probe established that for L3-8B MLP down
(K/N = 3.5), mixed `(m=2, n=1, k=large)` and balanced `(8, 4, 1)`
both beat element_priority's `(1, 32, 1)` by ~10%, and pure-K
`(1, 1, 32)` actually regresses vs pure-N. We need to find:

  - The K/N threshold above which K-split (or balanced) wins
  - Whether pure-K is ever the best (so far: no)
  - The shape-class boundary for a refined heuristic

Method: hold M=128 fixed, vary N and K to sweep K/N from 1.0 to 16.0.
For each shape, force a representative set of splits that bracket the
heuristic decision (pure-N / balanced m+n / mixed m+K / pure-K).

Run: python tests/diag_psum_kn_ratio_probe.py
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

# Each shape: (label, M, N, K, K/N ratio, [splits to test])
# Splits are picked to bracket the heuristic decision space:
#   - "pure-N": what element_priority picks today
#   - "pure-K": what existing k_split_heuristic picks
#   - "mixed (m, 1, k)": the predicted refined-heuristic pick
#   - "balanced (m, n, 1)": the alternative non-K-split winner
# Per-shape constraint: m·n·k = 32, all dims stick-aligned at fp16.

PROBE = [
    (
        "K/N=1.0",
        128, 4096, 4096,
        [
            (1, 32, 1),    # pure-N (EP)
            (1, 1, 32),    # pure-K (existing)
            (2, 1, 16),    # mixed (refined)
            (8, 4, 1),     # balanced
        ],
    ),
    (
        "K/N=2.0",
        128, 4096, 8192,
        [
            (1, 32, 1),
            (1, 1, 32),
            (2, 1, 16),
            (8, 4, 1),
        ],
    ),
    (
        "K/N=3.5 (L3-8B MLP down)",
        128, 4096, 14336,
        [
            (1, 32, 1),
            (1, 1, 32),
            (2, 1, 16),
            (8, 4, 1),
        ],
    ),
    (
        "K/N=8.0",
        128, 2048, 16384,
        [
            (1, 32, 1),    # pure-N (n=32 valid: 2048/32=64 sticks)
            (1, 1, 32),    # pure-K
            (2, 1, 16),    # mixed
            (8, 4, 1),     # balanced
        ],
    ),
    (
        "K/N=16.0",
        128, 1024, 16384,
        [
            # N=1024: max n = 16. So "pure-N" needs m or k > 1.
            (2, 16, 1),    # pure-N analog with slight m
            (1, 1, 32),    # pure-K
            (2, 1, 16),    # mixed
            (8, 4, 1),     # balanced (8*4=32, no K)
        ],
    ),
]


def _run_parent():
    print("# K/N ratio sweep — heuristic fire boundary\n")
    n_total = sum(len(splits) for _, _, _, _, splits in PROBE)
    print(f"# {len(PROBE)} shapes (M=128 fixed, varying N and K), "
          f"{n_total} measurements")
    print(f"# warmup={WARMUP} iters={ITERS}\n")

    results: dict[tuple[str, tuple[int, int, int]], list[float]] = {}
    n_done = 0
    t_start = time.time()

    for label, M, N, K, splits in PROBE:
        print(f"\n## {label}: ({M}, {N}, {K})")
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
    print("\n\n## Per-shape comparison\n")
    print("| K/N ratio | shape | pure-N | pure-K | mixed (2,1,16) | "
          "balanced (8,4,1) | best |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for label, M, N, K, splits in PROBE:
        cells = [label, f"({M}, {N}, {K})"]
        # Find the four "canonical" picks
        pure_n = None
        for s in splits:
            if s[2] == 1 and s[0] in (1, 2):  # pure-N or near-pure-N
                pure_n = s
                break
        pure_k = (1, 1, 32) if (1, 1, 32) in splits else None
        mixed = (2, 1, 16) if (2, 1, 16) in splits else None
        balanced = (8, 4, 1) if (8, 4, 1) in splits else None

        # Get wall ms for each
        def _ms_or_dash(s):
            if s is None:
                return "—"
            samples = results.get((label, s))
            if samples is None:
                return "err"
            return f"{statistics.median(samples) * 1e3:.3f}"

        cells.append(_ms_or_dash(pure_n))
        cells.append(_ms_or_dash(pure_k))
        cells.append(_ms_or_dash(mixed))
        cells.append(_ms_or_dash(balanced))

        # Identify best
        best_split = None
        best_ms = float("inf")
        for s in splits:
            samples = results.get((label, s))
            if samples is None:
                continue
            ms = statistics.median(samples) * 1e3
            if ms < best_ms:
                best_ms = ms
                best_split = s
        cells.append(f"{best_split} = {best_ms:.3f}" if best_split else "err")
        print("| " + " | ".join(cells) + " |")

    # --- speedup table (vs pure-N) ---
    print("\n## Speedup vs pure-N (current EP pick)\n")
    print("| K/N ratio | pure-K | mixed (2,1,16) | balanced (8,4,1) | "
          "best speedup | which? |")
    print("|---|---:|---:|---:|---:|---|")
    for label, M, N, K, splits in PROBE:
        pure_n_split = None
        for s in splits:
            if s[2] == 1 and s[0] in (1, 2):
                pure_n_split = s
                break
        if pure_n_split is None:
            continue
        pn_samples = results.get((label, pure_n_split))
        if pn_samples is None:
            continue
        pn_ms = statistics.median(pn_samples) * 1e3

        cells = [label]
        for s in [(1, 1, 32), (2, 1, 16), (8, 4, 1)]:
            if s not in splits:
                cells.append("—")
                continue
            samples = results.get((label, s))
            if samples is None:
                cells.append("err")
                continue
            ms = statistics.median(samples) * 1e3
            speedup = pn_ms / ms
            flag = " ✓" if speedup >= 1.05 else (" ✗" if speedup <= 0.95 else "")
            cells.append(f"{speedup:.3f}x{flag}")

        # Best speedup overall
        best_speedup = 1.0
        best_split = pure_n_split
        for s in splits:
            samples = results.get((label, s))
            if samples is None:
                continue
            ms = statistics.median(samples) * 1e3
            sp = pn_ms / ms
            if sp > best_speedup:
                best_speedup = sp
                best_split = s
        cells.append(f"{best_speedup:.3f}x")
        cells.append(str(best_split))
        print("| " + " | ".join(cells) + " |")

    print(f"\n\n# Total wall time: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_mode()
    else:
        sys.exit(_run_parent())
