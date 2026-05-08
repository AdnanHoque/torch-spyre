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

"""Granite 3-way measurement campaign for combined k_fast PR.

Mirror of diag_k_fast_combined_3way.py with the shape suite replaced by
IBM Granite 3.x dense linear-layer shapes. Provides a Granite-specific
companion to the Llama / Mixtral / DSv3 evidence already on the PR.

For each shape in the suite, measures three configurations:

  A — main baseline:  pure-M (32, 1, 1), identity emission
  B — split-k + id:   PR 1933 heuristic split, identity emission
  C — split-k + kf:   PR 1933 heuristic split, k_fast emission

Reports the same A→B / B→C / A→C / combined columns as the main
campaign, suitable for inclusion in the PR description.

Shape suite covers Granite 3.x 2B and 8B (the dense
production-deployed members) at decode + prefill batch sizes.

Usage:
    python tests/diag_k_fast_granite_3way.py
"""

from __future__ import annotations

import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

sys.stdout.reconfigure(line_buffering=True)

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402

try:
    from torch_spyre._inductor import work_division as _core_div  # noqa: E402
except ImportError:
    from torch_spyre._inductor import core_division as _core_div  # noqa: E402

from torch_spyre._inductor import config as ts_config  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16
ELEMS_PER_STICK = 64


# ---- Granite 3.x shape suite ---------------------------------------
# Dense models in production deployment. Each block has q/kv/o/gate/up/down.
# kv_proj here is the combined K+V output (GQA: 2 * n_kv_heads * head_dim).

@dataclass(frozen=True)
class GraniteConfig:
    name: str
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def kv_proj_out(self) -> int:
        return 2 * self.n_kv_heads * self.head_dim

    @property
    def q_proj_out(self) -> int:
        return self.n_heads * self.head_dim


GRANITE_MODELS = [
    GraniteConfig("Granite 3 2B", 2048, 8192, 32, 8, 64),
    GraniteConfig("Granite 3 8B", 4096, 12800, 32, 8, 128),
]

# M values matching the main 3-way campaign.
M_VALUES = (32, 128, 512, 2048)


@dataclass
class Shape:
    label: str
    M: int
    N: int
    K: int


def build_shape_suite() -> list[Shape]:
    out: list[Shape] = []
    for cfg in GRANITE_MODELS:
        H, I = cfg.hidden, cfg.intermediate
        Nq = cfg.q_proj_out
        Nkv = cfg.kv_proj_out
        for M in M_VALUES:
            out.append(Shape(f"{cfg.name} kv_proj M={M}",   M, Nkv, H))
            out.append(Shape(f"{cfg.name} q_proj M={M}",    M, Nq, H))
            out.append(Shape(f"{cfg.name} o_proj M={M}",    M, H, Nq))
            out.append(Shape(f"{cfg.name} gate_proj M={M}", M, I, H))
            out.append(Shape(f"{cfg.name} down_proj M={M}", M, H, I))
    return out


SHAPES = build_shape_suite()


# ---- planner heuristic mirror (matches torch_spyre/_inductor/work_division.py)
# Reflects the small-M wide-N extension we landed on AdnanHoque/pr-k-fast.

def heuristic_split(M, N, K, max_cores=32):
    if max_cores != 32:
        return None
    if M < 32 or M > 512:
        return None
    n_sticks = N // ELEMS_PER_STICK
    k_sticks = K // ELEMS_PER_STICK
    # Extension: drop n_sticks gate at M ≤ 128
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


# ---- machinery (mirrors main 3-way campaign) -----------------------

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        if target[0] * target[1] * target[2] != max_cores:
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


def main() -> int:
    print("# Granite 3-way measurement campaign\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32\n")
    print("All wall times normalized to A (A = 1.00). Speedup ratios > 1× = improvement.\n")
    print(f"Shape suite: {len(SHAPES)} shapes across {len(GRANITE_MODELS)} models × {len(M_VALUES)} M values.\n")

    print("| shape | (M, N, K) | h-split | A→B | B→C | A→C | combined |")
    print("|---|---|---|---:|---:|---:|---|")

    n_pr_speedup = 0
    n_pr_regress = 0
    sum_a = 0.0
    sum_c = 0.0

    for s in SHAPES:
        h_split = heuristic_split(s.M, s.N, s.K)
        split_str = f"({h_split[0]},{h_split[1]},{h_split[2]})" if h_split else "—"

        a_ms, _ = _compile_and_bench(s.M, s.N, s.K, (32, 1, 1), False)

        if h_split is not None:
            b_ms, _ = _compile_and_bench(s.M, s.N, s.K, h_split, False)
            c_ms, _ = _compile_and_bench(s.M, s.N, s.K, h_split, True)
        else:
            b_ms, c_ms = None, None

        if a_ms is not None and b_ms is not None and c_ms is not None:
            ab = a_ms / b_ms
            bc = b_ms / c_ms
            ac = a_ms / c_ms
            if ac > 1.05:
                pr_status = f"win {ac:.2f}×"
                n_pr_speedup += 1
            elif ac < 0.95:
                pr_status = f"REGRESS {ac:.2f}×"
                n_pr_regress += 1
            else:
                pr_status = "neutral"
            sum_a += a_ms
            sum_c += c_ms
            ab_s = f"{ab:.2f}×"
            bc_s = f"{bc:.2f}×"
            ac_s = f"{ac:.2f}×"
        else:
            ab_s = bc_s = ac_s = "—"
            pr_status = "(skipped, correct)" if h_split is None else "ERR"

        print(f"| {s.label} | ({s.M},{s.N},{s.K}) | {split_str} | "
              f"{ab_s} | {bc_s} | {ac_s} | {pr_status} |")

    # Aggregate
    print()
    print("## Aggregate\n")
    fired = sum(1 for s in SHAPES if heuristic_split(s.M, s.N, s.K) is not None)
    print(f"  shapes in suite: {len(SHAPES)}")
    print(f"  PR 1933 heuristic fires on: {fired}/{len(SHAPES)}")
    print(f"  combined PR (A→C): {n_pr_speedup} wins, {n_pr_regress} regressions")
    if sum_a > 0 and sum_c > 0:
        agg = sum_a / sum_c
        print(f"  geomean A→C on shapes where PR fires: {agg:.2f}×")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
