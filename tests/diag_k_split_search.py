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

"""Search for shapes where (1, 16, 2) + k_fast beats the planner.

The focused planner-correctness check showed pure-M wins on every
production-relevant shape we tested at M=2048. But maybe the K-split
sweet spot lives elsewhere:

  - Smaller M (where pure-M's elements/core drops, hurting compute
    utilization)
  - Even narrower N (MQA-style, TP-sharded kv)
  - Wide K (where pure-M's "everyone needs full B" cost grows)

For each shape, compare four configurations:
  A. natural   — whatever the planner picks
  B. pure-M    — (32, 1, 1) if valid
  C. pure-N    — (1, 32, 1) if valid (sanity, often invalid here)
  D. K-split   — (1, 16, 2) + k_fast (k_fast at its best)

Looking for any (M, N, K) where D < A.
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


WARMUP = 3
ITERS = 12
DTYPE = torch.float16

# Shapes designed to give K-split its best shot. Focus on narrow-N
# (where pure-N is invalid) at varying M and K.
TARGETS = [
    # ─── kv_proj family (N=1024) at varying M ────────────────────────
    # As M shrinks, pure-M's elements/core shrinks; K-split might catch up.
    ("kv N=1024 M=8",       8, 1024, 8192),
    ("kv N=1024 M=32",     32, 1024, 8192),
    ("kv N=1024 M=64",     64, 1024, 8192),
    ("kv N=1024 M=128",   128, 1024, 8192),
    ("kv N=1024 M=256",   256, 1024, 8192),
    ("kv N=1024 M=512",   512, 1024, 8192),
    ("kv N=1024 M=1024", 1024, 1024, 8192),
    ("kv N=1024 M=2048", 2048, 1024, 8192),

    # ─── Even narrower N (MQA-style or TP-sharded) ───────────────────
    ("MQA N=128 M=128",   128,  128, 8192),
    ("MQA N=256 M=128",   128,  256, 8192),
    ("MQA N=512 M=128",   128,  512, 8192),
    ("MQA N=128 M=2048", 2048,  128, 8192),
    ("MQA N=512 M=2048", 2048,  512, 8192),

    # ─── Wide-K narrow-N ─────────────────────────────────────────────
    ("kv wide-K M=128 K=16384",  128, 1024, 16384),
    ("kv wide-K M=128 K=32768",  128, 1024, 32768),
    ("kv wide-K M=2048 K=16384", 2048, 1024, 16384),

    # ─── Narrow-output FFN (TP-sharded) ──────────────────────────────
    ("FFN narrow M=128",   128, 1024, 14336),
    ("FFN narrow M=512",   512, 1024, 14336),
    ("FFN narrow M=2048", 2048, 1024, 14336),
]


# ---- machinery --------------------------------------------------------

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
    if target is None:
        yield
        return
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


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


def _compile_and_bench(M, N, K, force_target, perm_name: str):
    ts_config.core_id_permutation = perm_name
    ts_config.core_emission_reverse = False
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(force_target):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(force_target):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:90]}"


def main() -> int:
    print("# Search for shapes where (1,16,2)+k_fast beats the planner\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")
    print("Configs per shape:")
    print("  A. natural-pick")
    print("  B. pure-M (32, 1, 1)  [skip if invalid]")
    print("  C. pure-N (1, 32, 1)  [skip if invalid; often is]")
    print("  D. (1, 16, 2) + k_fast [skip if invalid]\n")

    CONFIGS = [
        ("A:natural",       None,        "identity"),
        ("B:pure-M",        (32, 1, 1),  "identity"),
        ("C:pure-N",        (1, 32, 1),  "identity"),
        ("D:(1,16,2)+kf",   (1, 16, 2),  "k_fast"),
    ]

    rows = []
    for label, M, N, K in TARGETS:
        n_st = N // 64
        k_st = K // 64
        m_per_core = M // 32 if M >= 32 else "<1"
        print(f"### {label}  M={M} N={N}({n_st}st) K={K}({k_st}st)")
        config_results = {}
        for cfg_name, force, perm in CONFIGS:
            ms, err = _compile_and_bench(M, N, K, force, perm)
            if err:
                print(f"  {cfg_name:18s}: ERR")
                config_results[cfg_name] = None
            else:
                print(f"  {cfg_name:18s}: {ms:.3f} ms")
                config_results[cfg_name] = ms
        rows.append((label, M, N, K, config_results))
        print()

    # --- summary table ---
    print("\n## Summary — wall time (ms) per config\n")
    print("| shape | M | N | K | A:natural | B:pure-M | C:pure-N | D:K-split+kf | best | "
          "D vs best |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for label, M, N, K, cfgs in rows:
        cells = []
        valid = {}
        for k in ("A:natural", "B:pure-M", "C:pure-N", "D:(1,16,2)+kf"):
            v = cfgs.get(k)
            if v is None:
                cells.append("err")
            else:
                cells.append(f"{v:.3f}")
                valid[k] = v
        if not valid:
            print(f"| {label} | {M} | {N} | {K} | "
                  + " | ".join(cells) + " | — | — |")
            continue
        best_cfg = min(valid, key=valid.get)
        best_v = valid[best_cfg]
        d_v = cfgs.get("D:(1,16,2)+kf")
        if d_v is None:
            d_vs_best = "—"
        else:
            d_vs_best = (
                f"BEST" if best_cfg == "D:(1,16,2)+kf"
                else f"{d_v / best_v:.3f}x best"
            )
        print(f"| {label} | {M} | {N} | {K} | "
              + " | ".join(cells)
              + f" | {best_cfg} ({best_v:.3f}) | {d_vs_best} |")
    print()

    # --- find K-split wins ---
    print("## K-split wins (D beats A AND D is the global best)\n")
    found = False
    for label, M, N, K, cfgs in rows:
        a = cfgs.get("A:natural")
        d = cfgs.get("D:(1,16,2)+kf")
        if a is None or d is None:
            continue
        if d < a:
            others = [v for k, v in cfgs.items()
                      if v is not None and k != "D:(1,16,2)+kf"]
            d_is_best = all(d <= o + 0.01 for o in others)
            note = " (and D is global best)" if d_is_best else " (but B or C is best)"
            print(f"  {label} M={M} N={N} K={K}: "
                  f"natural={a:.3f}  D={d:.3f}  speedup {a/d:.3f}x{note}")
            found = True
    if not found:
        print("  None. Pure-M (or natural) wins on every shape tested.")
    print()
    print("## D-vs-pure-M comparison on shapes where pure-M is valid\n")
    for label, M, N, K, cfgs in rows:
        b = cfgs.get("B:pure-M")
        d = cfgs.get("D:(1,16,2)+kf")
        if b is None or d is None:
            continue
        ratio = b / d
        marker = " ← D WINS" if ratio > 1.02 else ""
        print(f"  {label}: pure-M={b:.3f}ms, D={d:.3f}ms, "
              f"D/B={d/b:.3f}x{marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
