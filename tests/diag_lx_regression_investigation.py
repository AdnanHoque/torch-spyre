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

"""Investigate the L3-8B MLP gate/up regression at high LX frac.

Phase 1 catalog sweep showed L3-8B MLP gate/up (128, 14336, 4096)
regresses 16% at any DXP_LX_FRAC_AVAIL > 0.2. Hypothesis: per-core B
under (1, 32, 1) split is 3.5 MB, way over the 2 MB scratchpad. At
high frac the LX planner over-commits weight-buffer space, starving
activation/output staging and forcing inefficient pipelining.

Test: hold M=128, K=4096 constant. Force (1, 32, 1) split for all
variants so we isolate the LX-budget effect from the split choice.
Vary N so per-core B sweeps from comfortably-fits to over-by-2x.
At each N, sweep DXP_LX_FRAC_AVAIL.

Predicted per-core operand total at (1, 32, 1) for K=4096:

  N=2048:  per-core A=1MB, B=512KB, C=16KB → 1.53 MB ✓ fits
  N=4096:  per-core A=1MB, B=1MB, C=32KB   → 2.03 MB borderline
  N=8192:  per-core A=1MB, B=2MB, C=64KB   → 3.06 MB over
  N=14336: per-core A=1MB, B=3.5MB, C=112KB → 4.61 MB way over
                                              (the regressing case)

Hypothesis prediction: regression appears around N=4096-8192 and
worsens with N.

Run: python tests/diag_lx_regression_investigation.py
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

M = 128
K = 4096
NS = [2048, 4096, 8192, 14336]
FORCED_SPLIT = (1, 32, 1)


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

    # Force the planner to use the target split.
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
    M_ = int(sys.argv[2])
    N_ = int(sys.argv[3])
    K_ = int(sys.argv[4])
    target = tuple(int(x) for x in sys.argv[5].split(","))
    samples = _run_child(M_, N_, K_, target)  # type: ignore[arg-type]
    print("__RESULTS__" + json.dumps(samples))


# ---- parent-mode orchestration ----------------------------------------

CONFIGS = [
    ("control", {"LX_PLANNING": "0"}),
    ("frac=0.2", {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.2"}),
    ("frac=0.4", {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.4"}),
    ("frac=0.8", {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.8"}),
    ("frac=0.95", {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.95"}),
]


def _per_core_total(N: int) -> int:
    """Bytes per core under (1, 32, 1) split, fp16."""
    a = M * K * 2
    b = K * (N // 32) * 2
    c = M * (N // 32) * 2
    return a + b + c


def _run_parent():
    print("# LX-budget regression investigation\n")
    print(f"# Forced split: {FORCED_SPLIT}")
    print(f"# Fixed: M={M}, K={K}; varied: N in {NS}")
    print(f"# warmup={WARMUP} iters={ITERS}\n")

    print("## Per-core operand total at (1, 32, 1) split, fp16\n")
    print("| N | per-core A | per-core B | per-core C | total | "
          "fits 2 MB? |")
    print("|---|---:|---:|---:|---:|---|")
    for N in NS:
        a = M * K * 2
        b = K * (N // 32) * 2
        c = M * (N // 32) * 2
        total = a + b + c
        fits = "✓" if total <= 2 * 1024 * 1024 else "✗"
        print(f"| {N} | {a // 1024} KB | {b // 1024} KB | "
              f"{c // 1024} KB | {total // 1024} KB | {fits} |")
    print()

    results: dict[tuple[int, str], list[float]] = {}
    n_done = 0
    n_total = len(NS) * len(CONFIGS)
    t_start = time.time()

    for N in NS:
        print(f"\n## N={N} (per-core total = {_per_core_total(N) // 1024} KB)")
        for cfg_name, cfg_env in CONFIGS:
            n_done += 1
            elapsed = time.time() - t_start
            print(f"  [{n_done}/{n_total} t={elapsed:.0f}s] "
                  f"{cfg_name} ...", end="", flush=True)
            env = os.environ.copy()
            env.update(cfg_env)
            target_str = ",".join(str(x) for x in FORCED_SPLIT)
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
            results[(N, cfg_name)] = samples

    # --- table ---
    print("\n\n## Median wall time per (N, config) — ms\n")
    cfg_names = [name for name, _ in CONFIGS]
    print("| N | per-core total | fits | " + " | ".join(cfg_names) + " |")
    print("|---|---:|---|" + "|".join(["---:"] * len(cfg_names)) + "|")
    for N in NS:
        total_kb = _per_core_total(N) // 1024
        fits = "✓" if total_kb <= 2048 else "✗"
        cells = [str(N), f"{total_kb} KB", fits]
        for cfg_name in cfg_names:
            samples = results.get((N, cfg_name))
            if samples is None:
                cells.append("err")
            else:
                cells.append(f"{statistics.median(samples)*1e3:.3f}")
        print("| " + " | ".join(cells) + " |")

    # --- speedup vs control ---
    print("\n## Speedup vs control (LX_PLANNING=0)\n")
    test_names = [name for name, _ in CONFIGS[1:]]
    print("| N | per-core total | fits | " + " | ".join(test_names) + " |")
    print("|---|---:|---|" + "|".join(["---:"] * len(test_names)) + "|")
    for N in NS:
        total_kb = _per_core_total(N) // 1024
        fits = "✓" if total_kb <= 2048 else "✗"
        ctrl = results.get((N, "control"))
        cells = [str(N), f"{total_kb} KB", fits]
        for cfg_name in test_names:
            samples = results.get((N, cfg_name))
            if samples is None or ctrl is None:
                cells.append("err")
            else:
                ctrl_ms = statistics.median(ctrl) * 1e3
                test_ms = statistics.median(samples) * 1e3
                speedup = ctrl_ms / test_ms
                flag = ""
                if speedup >= 1.05:
                    flag = " ✓"
                elif speedup <= 0.95:
                    flag = " ✗"
                cells.append(f"{speedup:.3f}x{flag}")
        print("| " + " | ".join(cells) + " |")

    # --- verdict ---
    print("\n## Verdict\n")
    fits_speedups = []
    nofits_speedups = []
    for N in NS:
        total_kb = _per_core_total(N) // 1024
        ctrl = results.get((N, "control"))
        if ctrl is None:
            continue
        ctrl_ms = statistics.median(ctrl) * 1e3
        # Use frac=0.8 as a representative high-frac data point
        high = results.get((N, "frac=0.8"))
        if high is None:
            continue
        high_ms = statistics.median(high) * 1e3
        speedup = ctrl_ms / high_ms
        if total_kb <= 2048:
            fits_speedups.append((N, speedup))
        else:
            nofits_speedups.append((N, speedup))

    if fits_speedups:
        avg_fits = sum(s for _, s in fits_speedups) / len(fits_speedups)
        print(f"  Shapes that FIT 2MB scratchpad ({len(fits_speedups)}): "
              f"average frac=0.8 speedup = {avg_fits:.3f}×")
        for N, s in fits_speedups:
            print(f"    N={N}: {s:.3f}×")
    if nofits_speedups:
        avg_no = sum(s for _, s in nofits_speedups) / len(nofits_speedups)
        print(f"  Shapes that DON'T fit ({len(nofits_speedups)}): "
              f"average frac=0.8 speedup = {avg_no:.3f}×")
        for N, s in nofits_speedups:
            print(f"    N={N}: {s:.3f}×")

    if fits_speedups and nofits_speedups:
        print()
        if avg_fits > 1.0 and avg_no < 0.95:
            print("  Hypothesis CONFIRMED: high frac helps shapes that "
                  "fit LX, hurts shapes that don't. The regression "
                  "correlates with per-core operand size exceeding 2 MB.")
        elif avg_fits > 1.0 and avg_no > 1.0:
            print("  Hypothesis NOT supported: high frac helps both "
                  "regimes. The regression on L3-8B MLP gate/up may be "
                  "shape-specific (split choice, K size, …) rather than "
                  "an over-commit issue.")
        elif avg_fits < 1.0 and avg_no < 1.0:
            print("  Hypothesis MIXED: high frac hurts both regimes. "
                  "Something else is going on with this forced split — "
                  "may be that (1,32,1) at K=4096 is just bad with "
                  "LX_PLANNING regardless of N.")
        else:
            print("  Result inconclusive — re-examine raw data.")

    print(f"\n# Total wall time: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_mode()
    else:
        sys.exit(_run_parent())
