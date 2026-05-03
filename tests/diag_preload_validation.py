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

"""LX scratchpad preload validation.

Question: Does enabling LX_PLANNING + DXP_LX_FRAC_AVAIL meaningfully
keep weights resident in the on-chip scratchpad across kernel calls?

The IBM AIU architecture doc (slides 86-94) describes a static-tensor
preload mechanism that can keep frequently-reused weights in LX
across inferences. An earlier Phase 0 weight-residency probe (FA
branch) found "no cross-kernel weight persistence" — but that probe
ran without LX_PLANNING enabled. This probe re-tests with the flag
on, varying DXP_LX_FRAC_AVAIL.

Method: re-invokes itself via subprocess for each (shape, config)
pair. Both env vars must be set BEFORE torch_spyre import, so we
need a fresh process per config. Each child compiles a matmul,
benches per-iteration wall time, returns the samples as JSON.

We compare:
  - LX_PLANNING=0  (control: no LX management, no preload)
  - LX_PLANNING=1 with DXP_LX_FRAC_AVAIL=0.2 (default)
  - LX_PLANNING=1 with DXP_LX_FRAC_AVAIL=0.5 (more preload budget)

If preload is firing and helping, the LX_PLANNING=1 cases should be
faster than the control. If they're similar, either preload isn't
firing for our usage pattern (lazy torch.compile) or it doesn't help
on these shapes.

Run: python tests/diag_preload_validation.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

WARMUP = 3
ITERS = 25


# ---- child-mode bench --------------------------------------------------

def _run_child(M: int, N: int, K: int) -> list[float]:
    """Compile + bench matmul (M, N, K), return per-iter wall samples
    in seconds. Reads LX_PLANNING / DXP_LX_FRAC_AVAIL from env at
    import time."""
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
    for _ in range(ITERS):
        t0 = time.perf_counter()
        mm(a, b)
        _ts.synchronize()
        samples.append(time.perf_counter() - t0)
    return samples


def _child_mode():
    M = int(sys.argv[2])
    N = int(sys.argv[3])
    K = int(sys.argv[4])
    samples = _run_child(M, N, K)
    print("__RESULTS__" + json.dumps(samples))


# ---- parent-mode orchestration ----------------------------------------

CONFIGS = [
    # Control: LX_PLANNING off, default scratchpad budget. This is what
    # most users get out of the box if they don't set the env var.
    ("LX_PLANNING=0 (control)",
     {"LX_PLANNING": "0"}),
    ("LX_PLANNING=1, frac=0.2 (default)",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.2"}),
    ("LX_PLANNING=1, frac=0.5",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.5"}),
    ("LX_PLANNING=1, frac=0.8",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.8"}),
]

SHAPES = [
    ("L3-8B q_proj prefill", 128, 4096, 4096),
    ("L3-70B q_proj prefill", 128, 8192, 8192),
    ("L3-8B GQA kv_proj prefill", 128, 1024, 4096),
]


def _run_parent():
    print("# LX scratchpad preload validation\n")
    print(f"# Shapes: {[(label, M, N, K) for (label, M, N, K) in SHAPES]}")
    print(f"# Configs:")
    for name, env in CONFIGS:
        print(f"#   {name}: {env}")
    print(f"# warmup={WARMUP} iters={ITERS}\n")

    results: dict[tuple[str, str], list[float]] = {}

    for label, M, N, K in SHAPES:
        print(f"\n## {label} ({M}, {N}, {K})")
        for cfg_name, cfg_env in CONFIGS:
            env = os.environ.copy()
            env.update(cfg_env)

            print(f"  {cfg_name} ...", end="", flush=True)
            try:
                proc = subprocess.run(
                    [sys.executable, __file__, "--child",
                     str(M), str(N), str(K)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
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
                stderr_tail = proc.stderr[-300:] if proc.stderr else "(empty)"
                print(f"  FAIL stderr: ...{stderr_tail}")
                continue

            median_ms = statistics.median(samples) * 1e3
            first_ms = samples[0] * 1e3
            min_ms = min(samples) * 1e3
            print(f"  median={median_ms:.3f}ms first={first_ms:.3f}ms "
                  f"min={min_ms:.3f}ms")
            results[(label, cfg_name)] = samples

    # --- summary table ---
    print("\n\n## Summary: median wall time per (shape, config)\n")
    cfg_names = [name for name, _ in CONFIGS]
    print("| shape | " + " | ".join(cfg_names) + " |")
    print("|---|" + "|".join(["---:"] * len(cfg_names)) + "|")
    for label, M, N, K in SHAPES:
        cells = [f"{label} ({M}, {N}, {K})"]
        for cfg_name in cfg_names:
            samples = results.get((label, cfg_name))
            if samples is None:
                cells.append("err")
            else:
                cells.append(f"{statistics.median(samples)*1e3:.3f}")
        print("| " + " | ".join(cells) + " |")

    # --- speedup table ---
    print("\n## Speedup vs control (LX_PLANNING=0)\n")
    control_name = CONFIGS[0][0]
    test_names = [name for name, _ in CONFIGS[1:]]
    print("| shape | " + " | ".join(test_names) + " |")
    print("|---|" + "|".join(["---:"] * len(test_names)) + "|")
    for label, M, N, K in SHAPES:
        ctrl = results.get((label, control_name))
        cells = [f"{label}"]
        for cfg_name in test_names:
            samples = results.get((label, cfg_name))
            if samples is None or ctrl is None:
                cells.append("err")
            else:
                ctrl_ms = statistics.median(ctrl) * 1e3
                test_ms = statistics.median(samples) * 1e3
                speedup = ctrl_ms / test_ms
                flag = " ✓" if speedup >= 1.05 else ""
                cells.append(f"{speedup:.3f}x{flag}")
        print("| " + " | ".join(cells) + " |")

    # --- first vs median (within-process cache warmup) ---
    print("\n## First-iter vs median (within-process cache warmup)\n")
    print("If the first iteration is much slower than the median, "
          "weights are being fetched from DRAM on iter 0 and cached "
          "for subsequent iters. A small first/median gap means "
          "either DRAM fetch is fast or there's no caching.\n")
    print("| shape | config | first ms | median ms | first/median |")
    print("|---|---|---:|---:|---:|")
    for label, M, N, K in SHAPES:
        for cfg_name in cfg_names:
            samples = results.get((label, cfg_name))
            if samples is None:
                continue
            first_ms = samples[0] * 1e3
            median_ms = statistics.median(samples) * 1e3
            ratio = first_ms / median_ms
            print(f"| {label} | {cfg_name} | "
                  f"{first_ms:.3f} | {median_ms:.3f} | {ratio:.2f}x |")

    # --- verdict ---
    print("\n## Verdict\n")
    speedups = []
    for label, M, N, K in SHAPES:
        ctrl = results.get((label, control_name))
        if ctrl is None:
            continue
        ctrl_ms = statistics.median(ctrl) * 1e3
        for cfg_name in test_names:
            samples = results.get((label, cfg_name))
            if samples is None:
                continue
            test_ms = statistics.median(samples) * 1e3
            speedups.append(ctrl_ms / test_ms)
    if not speedups:
        print("  No valid speedup data.")
        return 1
    max_speedup = max(speedups)
    if max_speedup >= 1.10:
        print(f"  Max speedup {max_speedup:.2f}x — preload is firing "
              "and helping on at least one shape. Worth pursuing as a "
              "tuning project.")
    elif max_speedup >= 1.03:
        print(f"  Max speedup {max_speedup:.2f}x — preload provides a "
              "small benefit. Marginal, but real. Project might be "
              "worth pursuing if the benefit is consistent across "
              "production shapes.")
    else:
        print(f"  Max speedup {max_speedup:.2f}x — preload is either "
              "not firing for torch.compile-driven matmul, or it "
              "isn't helping on these shapes. Likely the lazy-compile "
              "path doesn't trigger the loadmodel_to_spad dsengraph "
              "where preload nodes live.")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_mode()
    else:
        sys.exit(_run_parent())
