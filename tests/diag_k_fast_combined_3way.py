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

"""3-way measurement campaign for the combined k_fast PR (1932 + 1933).

For each shape in a representative production suite, measures three
configurations:

  A — main baseline: pure-M (32, 1, 1), identity emission
  B — K-split + id: heuristic-picked split (1, n, k>1), identity emission
  C — K-split + kf: heuristic-picked split (1, n, k>1), k_fast emission

The deltas isolate two distinct contributions:
  A → B : gain from picking K-split alone (better PT util, per-cluster bytes)
  B → C : gain from k_fast emission (PSUM hops m*n → 1)

Either delta can be negative; the C/A ratio is the headline number for
the combined PR.

The probe runs the planner heuristic from PR 1933 on each shape to
pick the (1, n, k) split — so it tests exactly the configuration the
PR ships, not a hand-picked candidate. Configurations B and C use
that same split with identity / k_fast emission respectively.

Usage:
    python tests/diag_k_fast_combined_3way.py
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
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
from torch_spyre._inductor import core_division as _core_div  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16
ELEMS_PER_STICK = 64


# ---- shape suite ---------------------------------------------------
# Mix of shapes from production transformer blocks plus the verified
# small-M wide-N rows we want the extended heuristic to capture.

@dataclass
class Shape:
    label: str
    M: int
    N: int
    K: int
    in_pr1933_band: bool   # does the current heuristic fire?


SHAPES: list[Shape] = [
    # PR 1933's existing target band (narrow-N small-M)
    Shape("L3-70B kv_proj M=32",   32, 1024, 8192, True),
    Shape("L3-70B kv_proj M=128",  128, 1024, 8192, True),
    Shape("L3-70B kv_proj M=512",  512, 1024, 8192, True),
    Shape("Mixtral kv_proj M=128", 128, 1024, 4096, True),
    Shape("DSv3 kv_proj M=128",    128, 1536, 7168, True),
    Shape("DSv3 q_a_proj M=128",   128, 1536, 7168, True),

    # Outside the band (wider N) — heuristic currently skips, but
    # verification showed (1, 4, 8)+kf wins at small M:
    Shape("L3-70B q_proj M=32",      32, 8192, 8192, False),
    Shape("DSv3 gate_proj M=32",     32, 18432, 7168, False),
    Shape("L3-70B q_proj M=128",    128, 8192, 8192, False),
    Shape("L3-70B q_proj M=512",    512, 8192, 8192, False),
    Shape("DSv3 down_proj M=128",   128, 7168, 18432, False),

    # M out-of-band (current heuristic gates at M < 32 or M > 512)
    Shape("L3-70B kv_proj M=2048", 2048, 1024, 8192, False),
]


# ---- planner heuristic mirror (PR 1933) ----------------------------
# Same logic as torch_spyre._inductor.core_division._try_k_fast_split,
# kept here so the probe can also explore extensions before committing
# to a code change.

def heuristic_split_pr1933(M, N, K, max_cores=32):
    """Mirror of the combined-branch _try_k_fast_split heuristic.

    Matches torch_spyre._inductor.core_division._try_k_fast_split as of
    the n_sticks-gate-relaxation extension: at M ≤ 128 the n_sticks ≥ 32
    skip is dropped to capture wins on small-M wide-N shapes.
    """
    if max_cores != 32:
        return None
    if M < 32 or M > 512:
        return None
    n_sticks = N // ELEMS_PER_STICK
    k_sticks = K // ELEMS_PER_STICK
    # Extended gate: n_sticks ≥ 32 only skips when M > 128.
    if M > 128 and n_sticks >= 32:
        return None
    if k_sticks < 32:
        return None
    for n in (16, 8, 4, 2):
        if max_cores % n != 0 or n_sticks % n != 0:
            continue
        k = max_cores // n
        if k_sticks < k or k_sticks % k != 0:
            continue
        return (1, n, k)
    return None


# ---- forced-split + emission machinery -----------------------------

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


@contextmanager
def _kfast_emission(enabled: bool):
    """Toggle k_fast emission via the production flag."""
    prev = ts_config.core_id_k_fast_emission
    ts_config.core_id_k_fast_emission = enabled
    try:
        yield
    finally:
        ts_config.core_id_k_fast_emission = prev


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


def _compile_and_bench(M, N, K, split, kfast):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _kfast_emission(kfast), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _kfast_emission(kfast), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:50]}"


# ---- main ----------------------------------------------------------

def main() -> int:
    print("# k_fast combined PR (1932 + 1933) — 3-way measurement campaign\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32\n")
    print("Configurations:")
    print("  A — main baseline:  pure-M (32, 1, 1), identity emission")
    print("  B — K-split + id:   heuristic split, identity emission")
    print("  C — K-split + kf:   heuristic split, k_fast emission\n")

    print("| shape | (M, N, K) | h-split | "
          "A ms | B ms | C ms | A→B | B→C | A→C | combined PR | %% from kf |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    n_pr_speedup = 0
    n_pr_regress = 0
    sum_a = 0.0
    sum_c = 0.0

    for s in SHAPES:
        h_split = heuristic_split_pr1933(s.M, s.N, s.K)
        if h_split is None:
            split_str = "—"
        else:
            split_str = f"({h_split[0]},{h_split[1]},{h_split[2]})"

        # A: pure-M, identity
        a_ms, a_err = _compile_and_bench(s.M, s.N, s.K, (32, 1, 1), False)

        if h_split is not None:
            # B: forced K-split, identity
            b_ms, b_err = _compile_and_bench(s.M, s.N, s.K, h_split, False)
            # C: forced K-split, k_fast
            c_ms, c_err = _compile_and_bench(s.M, s.N, s.K, h_split, True)
        else:
            b_ms, c_ms = None, None
            b_err = c_err = ""

        def _f(x):
            return f"{x:.2f}" if x is not None else "—"

        if a_ms is not None and b_ms is not None and c_ms is not None:
            ab = a_ms / b_ms
            bc = b_ms / c_ms
            ac = a_ms / c_ms
            kf_share = (b_ms - c_ms) / (a_ms - c_ms) * 100 if (a_ms - c_ms) > 0 else 0
            kf_share_str = f"{kf_share:.0f}%"
            if ac > 1.05:
                pr_status = f"WIN {ac:.2f}×"
                n_pr_speedup += 1
            elif ac < 0.95:
                pr_status = f"REGRESS {ac:.2f}×"
                n_pr_regress += 1
            else:
                pr_status = "neutral"
            sum_a += a_ms
            sum_c += c_ms
        else:
            ab = bc = ac = None
            kf_share_str = "—"
            pr_status = "n/a (heuristic skip)" if h_split is None else "ERR"

        print(f"| {s.label} | ({s.M},{s.N},{s.K}) | {split_str} | "
              f"{_f(a_ms)} | {_f(b_ms)} | {_f(c_ms)} | "
              f"{f'{ab:.2f}×' if ab else '—'} | "
              f"{f'{bc:.2f}×' if bc else '—'} | "
              f"{f'{ac:.2f}×' if ac else '—'} | "
              f"{pr_status} | {kf_share_str} |")

    # Aggregate
    print()
    print("## Aggregate\n")
    fired = sum(1 for s in SHAPES if heuristic_split_pr1933(s.M, s.N, s.K) is not None)
    print(f"  shapes in suite: {len(SHAPES)}")
    print(f"  PR 1933 heuristic fires on: {fired}/{len(SHAPES)}")
    print(f"  combined PR (A→C): {n_pr_speedup} wins, {n_pr_regress} regressions")
    if sum_a > 0 and sum_c > 0:
        agg = sum_a / sum_c
        saved = sum_a - sum_c
        print(f"  total A→C wall change on shapes where PR fires: "
              f"{agg:.2f}× (saved {saved:.1f} ms total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
