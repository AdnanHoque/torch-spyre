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

"""Phase 1 — full catalog sweep of LX_PLANNING + DXP_LX_FRAC_AVAIL.

Three questions to answer:

  1. Which shapes benefit most from increasing the LX scratchpad
     budget? (Hypothesis: large-weight prefill — q_proj, MLP-down.)
  2. Is there a single best `frac` value, or does the optimum vary
     by shape?
  3. Does the LX-budget gain compound with `output_element_priority`?

Method: same subprocess pattern as `diag_preload_validation.py`. Each
(shape, config) pair runs in a fresh child process so the env-time
configs (LX_PLANNING, DXP_LX_FRAC_AVAIL, OUTPUT_ELEMENT_PRIORITY) take
effect.

Configurations per shape (7 total):
  - control: LX_PLANNING=0
  - frac=0.2 (current default), 0.4, 0.6, 0.8, 0.95 with LX_PLANNING=1
  - compound: LX_PLANNING=1, frac=0.8, OUTPUT_ELEMENT_PRIORITY=1

Shapes: the 13 Phase 1.0 shapes
(see tests/diag_split_gap_results.md for the original measurements).

Total: 13 shapes × 7 configs = 91 runs. ~30-40 s per run on the card,
so the full sweep is ~45-60 minutes.

Run: python tests/diag_lx_budget_catalog_sweep.py
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

def _run_child(M: int, N: int, K: int) -> list[float]:
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
    ("control",
     {"LX_PLANNING": "0"}),
    ("frac=0.2 (default)",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.2"}),
    ("frac=0.4",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.4"}),
    ("frac=0.6",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.6"}),
    ("frac=0.8",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.8"}),
    ("frac=0.95",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.95"}),
    ("compound: ep + frac=0.8",
     {"LX_PLANNING": "1", "DXP_LX_FRAC_AVAIL": "0.8",
      "OUTPUT_ELEMENT_PRIORITY": "1"}),
]

SHAPES = [
    ("L3-8B q_proj prefill",        128, 4096, 4096),
    ("L3-8B GQA kv_proj prefill",   128, 1024, 4096),
    ("L3-8B MLP gate/up prefill",   128, 14336, 4096),
    ("L3-8B MLP down prefill",      128, 4096, 14336),
    ("L3-70B q_proj prefill",       128, 8192, 8192),
    ("L3-70B GQA kv_proj prefill",  128, 1024, 8192),
    ("L3-70B GQA TP=8 kv prefill",  128, 128, 8192),
    ("L3-70B MLP down prefill",     128, 8192, 28672),
    ("Mixtral down per-expert",     128, 4096, 14336),
    ("Qwen3-MoE gate per-expert",   128, 1536, 2048),
    ("DeepSeek-MoE gate (M=192)",   192, 1408, 2048),
    ("L3-8B q_proj decode",         1, 4096, 4096),
    ("L3-70B GQA TP=8 kv decode",   1, 128, 8192),
]


def _run_parent():
    print("# LX scratchpad catalog sweep — Phase 1\n")
    print(f"# Shapes: {len(SHAPES)}, Configs: {len(CONFIGS)}, "
          f"Total runs: {len(SHAPES) * len(CONFIGS)}")
    print(f"# warmup={WARMUP} iters={ITERS}\n")

    results: dict[tuple[str, str], list[float]] = {}
    n_done = 0
    n_total = len(SHAPES) * len(CONFIGS)
    t_start = time.time()

    for label, M, N, K in SHAPES:
        print(f"\n## {label} ({M}, {N}, {K})")
        for cfg_name, cfg_env in CONFIGS:
            n_done += 1
            elapsed = time.time() - t_start
            print(f"  [{n_done}/{n_total} t={elapsed:.0f}s] "
                  f"{cfg_name} ...", end="", flush=True)
            env = os.environ.copy()
            env.update(cfg_env)
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
                tail = proc.stderr[-200:] if proc.stderr else "(empty)"
                print(f"  FAIL ...{tail}")
                continue

            ms = statistics.median(samples) * 1e3
            print(f"  {ms:.3f} ms")
            results[(label, cfg_name)] = samples

    # --- per-shape table ---
    print("\n\n## Median wall time per (shape, config) — ms\n")
    cfg_names = [name for name, _ in CONFIGS]
    print("| shape | " + " | ".join(cfg_names) + " |")
    print("|---|" + "|".join(["---:"] * len(cfg_names)) + "|")
    for label, M, N, K in SHAPES:
        cells = [label]
        for cfg_name in cfg_names:
            samples = results.get((label, cfg_name))
            if samples is None:
                cells.append("err")
            else:
                cells.append(f"{statistics.median(samples)*1e3:.3f}")
        print("| " + " | ".join(cells) + " |")

    # --- speedup vs control ---
    print("\n## Speedup vs control (LX_PLANNING=0)\n")
    test_names = [name for name, _ in CONFIGS[1:]]
    print("| shape | " + " | ".join(test_names) + " |")
    print("|---|" + "|".join(["---:"] * len(test_names)) + "|")
    for label, M, N, K in SHAPES:
        ctrl = results.get((label, "control"))
        cells = [label]
        for cfg_name in test_names:
            samples = results.get((label, cfg_name))
            if samples is None or ctrl is None:
                cells.append("err")
            else:
                ctrl_ms = statistics.median(ctrl) * 1e3
                test_ms = statistics.median(samples) * 1e3
                speedup = ctrl_ms / test_ms
                flag = ""
                if speedup >= 1.10:
                    flag = " ✓✓"
                elif speedup >= 1.05:
                    flag = " ✓"
                elif speedup <= 0.95:
                    flag = " ✗"
                cells.append(f"{speedup:.3f}x{flag}")
        print("| " + " | ".join(cells) + " |")

    # --- best frac per shape ---
    print("\n## Best LX frac per shape (no element_priority)\n")
    print("| shape | best frac | speedup vs control |")
    print("|---|---|---:|")
    frac_names = ["frac=0.2 (default)", "frac=0.4", "frac=0.6",
                  "frac=0.8", "frac=0.95"]
    for label, M, N, K in SHAPES:
        ctrl = results.get((label, "control"))
        if ctrl is None:
            print(f"| {label} | err | err |")
            continue
        ctrl_ms = statistics.median(ctrl) * 1e3
        best_ms = ctrl_ms
        best_name = "control"
        for f_name in frac_names:
            samples = results.get((label, f_name))
            if samples is None:
                continue
            ms = statistics.median(samples) * 1e3
            if ms < best_ms:
                best_ms = ms
                best_name = f_name
        speedup = ctrl_ms / best_ms
        print(f"| {label} | {best_name} | {speedup:.3f}x |")

    # --- compound stack ---
    print("\n## Compound: element_priority + frac=0.8 vs each lever alone\n")
    print("| shape | control | LX frac=0.8 alone | "
          "element_priority alone* | compound (both) | "
          "compound speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    print("\n*element_priority alone numbers from previous "
          "compare results (committed `0ff598a`).\n")
    # Ship previous element-priority alone numbers from
    # diag_element_priority_compare_results.md (median of 15 iters)
    EP_ALONE = {
        "L3-8B q_proj prefill": 3.24,
        "L3-8B GQA kv_proj prefill": 3.04,
        "L3-8B MLP gate/up prefill": 3.78,
        "L3-8B MLP down prefill": 4.64,
        "L3-70B q_proj prefill": 4.05,
        "L3-70B GQA kv_proj prefill": 3.13,
        "L3-70B GQA TP=8 kv prefill": 3.00,
        "L3-70B MLP down prefill": 8.03,
        "Mixtral down per-expert": 4.65,
        "Qwen3-MoE gate per-expert": 3.05,
        "DeepSeek-MoE gate (M=192)": 3.00,
        "L3-8B q_proj decode": 3.15,
        "L3-70B GQA TP=8 kv decode": 3.00,
    }
    for label, M, N, K in SHAPES:
        ctrl = results.get((label, "control"))
        f08 = results.get((label, "frac=0.8"))
        comp = results.get((label, "compound: ep + frac=0.8"))
        if not (ctrl and f08 and comp):
            print(f"| {label} | err | err | err | err | err |")
            continue
        ctrl_ms = statistics.median(ctrl) * 1e3
        f08_ms = statistics.median(f08) * 1e3
        comp_ms = statistics.median(comp) * 1e3
        ep_ms = EP_ALONE.get(label, float("nan"))
        compound_speedup = ctrl_ms / comp_ms if comp_ms else 0
        print(f"| {label} | {ctrl_ms:.2f} | {f08_ms:.2f} | "
              f"{ep_ms:.2f} | {comp_ms:.2f} | {compound_speedup:.3f}x |")

    print(f"\n\n# Total wall time: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_mode()
    else:
        sys.exit(_run_parent())
