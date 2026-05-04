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

"""Is the planner's pure-M pick actually correct, or could (1, 16, 2)+k_fast win?

Earlier verification showed the planner picks (32, 1, 1) — pure-M — for
every shape we measured k_fast wins on. So our forced (1, 16, 2) splits
don't compose with production. The remaining question:

  Does (1, 16, 2) + k_fast beat (32, 1, 1) on the same hardware?

If yes: the planner is wrong and we should pair k_fast with a planner
change (Option C from the analysis).

If no: the planner is correct; (32, 1, 1) is genuinely faster than
even our best K-split + k_fast configuration. k_fast wins are
unreachable from production. Close the project.

For each shape, three configurations:
  A. NATURAL pure-M (32, 1, 1) — what the planner picks today
  B. FORCED  (1, 16, 2) + identity emission — what we measured originally
  C. FORCED  (1, 16, 2) + k_fast emission   — k_fast at its best

Two trial orders for replication confidence.
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
ITERS = 25
DTYPE = torch.float16

# (label, M, N, K, our_forced_split)
TARGETS = [
    ("L3-70B kv_proj M=2048",          2048,  1024,  8192, (1, 16, 2)),
    ("Mixtral 8x7B kv_proj M=2048",    2048,  1024,  4096, (1, 16, 2)),
    ("DSv3 o_proj M=2048",             2048,  7168, 16384, (1, 16, 2)),
    ("DSv3 down_proj M=2048 (dense)",  2048,  7168,  2048, (1, 16, 2)),
]


# ---- force-split machinery --------------------------------------------

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
        return None, f"{type(e).__name__}: {str(e)[:120]}"


def _config_label(force_target, perm_name):
    if force_target is None:
        return f"natural+{perm_name}"
    return f"{force_target}+{perm_name}"


def main() -> int:
    print("# Is (1, 16, 2)+k_fast actually faster than the planner's pure-M pick?\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")
    print("Three configs per shape:")
    print("  A. natural+identity   (planner picks; today's production)")
    print("  B. (1, 16, 2)+identity (our original forced measurement)")
    print("  C. (1, 16, 2)+k_fast   (k_fast at its best)\n")
    print("If C < A: pair k_fast with a planner change (Option C in writeup).")
    print("If C >= A: planner is right; k_fast wins are unreachable in prod.\n")

    # CONFIGS: (label, force_split, perm)
    CONFIGS = [
        ("A: natural+identity",   None,        "identity"),
        ("B: (1,16,2)+identity",  (1, 16, 2),  "identity"),
        ("C: (1,16,2)+k_fast",    (1, 16, 2),  "k_fast"),
    ]

    results = {}  # (label, config_label, trial) -> ms
    for label, M, N, K, _forced in TARGETS:
        print(f"### {label}  (M={M}, N={N}, K={K})\n")
        for tname, ordered in (
            ("trial1", CONFIGS),
            ("trial2", list(reversed(CONFIGS))),
        ):
            print(f"  {tname} (order: {[c[0] for c in ordered]}):")
            for cfg_label, force, perm in ordered:
                ms, err = _compile_and_bench(M, N, K, force, perm)
                if err:
                    print(f"    {cfg_label:25s}: ERR {err[:80]}")
                    results[(label, cfg_label, tname)] = None
                else:
                    print(f"    {cfg_label:25s}: {ms:.3f} ms")
                    results[(label, cfg_label, tname)] = ms
            print()

    # --- summary ---
    print("\n## Summary — does (1,16,2)+k_fast beat planner-natural pure-M?\n")
    print("| shape | A: pure-M | B: (1,16,2)+id | C: (1,16,2)+k_fast | "
          "C/A speedup | C beats A? |")
    print("|---|---:|---:|---:|---:|---|")
    verdicts = []
    for label, _M, _N, _K, _f in TARGETS:
        a1 = results.get((label, "A: natural+identity", "trial1"))
        a2 = results.get((label, "A: natural+identity", "trial2"))
        b1 = results.get((label, "B: (1,16,2)+identity", "trial1"))
        b2 = results.get((label, "B: (1,16,2)+identity", "trial2"))
        c1 = results.get((label, "C: (1,16,2)+k_fast", "trial1"))
        c2 = results.get((label, "C: (1,16,2)+k_fast", "trial2"))
        if any(x is None for x in (a1, a2, b1, b2, c1, c2)):
            print(f"| {label} | ERR | ERR | ERR | — | — |")
            continue
        a_med = (a1 + a2) / 2
        b_med = (b1 + b2) / 2
        c_med = (c1 + c2) / 2
        speedup_ca = a_med / c_med
        beats = "✓ YES" if c_med < a_med * 0.98 else (
            "≈ tie" if abs(c_med - a_med) / a_med < 0.02 else "✗ NO"
        )
        verdicts.append((label, a_med, c_med, speedup_ca, beats))
        print(
            f"| {label} | {a_med:.3f} | {b_med:.3f} | {c_med:.3f} | "
            f"{speedup_ca:.3f}x | {beats} |"
        )

    # --- final verdict ---
    print("\n## Verdict\n")
    wins = [v for v in verdicts if v[4] == "✓ YES"]
    losses = [v for v in verdicts if v[4] == "✗ NO"]
    ties = [v for v in verdicts if v[4] == "≈ tie"]

    if wins:
        print(f"  {len(wins)} of {len(verdicts)} shapes: (1,16,2)+k_fast BEATS pure-M:")
        for v in wins:
            print(f"    - {v[0]}: pure-M {v[1]:.2f}ms vs k_fast {v[2]:.2f}ms "
                  f"({v[3]:.3f}x speedup)")
        print(
            "\n  → OPTION C: pair k_fast with a planner change to pick (1,16,2)\n"
            "    for these shapes. The planner is sub-optimal today; k_fast\n"
            "    + planner heuristic together unlock real production wins."
        )
    if losses:
        print(f"\n  {len(losses)} of {len(verdicts)} shapes: pure-M BEATS (1,16,2)+k_fast:")
        for v in losses:
            print(f"    - {v[0]}: pure-M {v[1]:.2f}ms vs k_fast {v[2]:.2f}ms "
                  f"(planner wins by {1/v[3]:.3f}x)")
        print(
            "\n  → For these shapes, the planner is right. (1,16,2)+k_fast\n"
            "    cannot beat pure-M even at its best. Don't pursue planner\n"
            "    change for these."
        )
    if ties:
        print(f"\n  {len(ties)} of {len(verdicts)} shapes: tied (within 2%):")
        for v in ties:
            print(f"    - {v[0]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
