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

"""Multicast core_id permutation — broader sweep.

Phase 0 (`diag_multicast_core_perm.py`) tested 7 (shape, split)
combinations and found permutation spreads ≤1.4% on structured
permutations. This sweep broadens coverage to confirm before
declaring the lever closed:

  - Real Llama 70B / DSv3 matmul shapes (q_proj, kv_proj, o_proj,
    gate_proj, up_proj, down_proj, q_a_proj)
  - M ∈ {128, 512, 2048} (decode, decode-batching, prefill)
  - Multiple m·n splits

If the spread stays ≤2% across this broader sweep, the lever is
confidently closed.

Usage:
    python tests/diag_multicast_core_perm_sweep.py
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from pathlib import Path
import sys

import torch
from sympy import Symbol

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
from torch_spyre._inductor.codegen import compute_ops as _co  # noqa: E402


WARMUP = 2
ITERS = 5
DTYPE = torch.float16


# ---- split + perm forcing (same machinery as Phase 0) ----------------

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
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


_orig_generate_sdsc = _co.generate_sdsc


@contextmanager
def _force_perm(perm):
    def _patched(sdsc_spec):
        result = _orig_generate_sdsc(sdsc_spec)
        new_mapping = {
            str(c): {
                str(dim): int(expr.subs({Symbol("core_id"): perm[c]}))
                for dim, expr in sdsc_spec.core_id_to_work_slice.items()
            }
            for c in range(sdsc_spec.num_cores)
        }
        result[sdsc_spec.opfunc]["coreIdToWkSlice_"] = new_mapping
        return result

    _co.generate_sdsc = _patched
    try:
        yield
    finally:
        _co.generate_sdsc = _orig_generate_sdsc


def perm_identity(num_cores: int) -> list[int]:
    return list(range(num_cores))


def perm_m_adjacent(m: int, n: int, num_cores: int = 32) -> list[int]:
    return [(c % n) * m + (c // n) for c in range(num_cores)]


def perm_reversed(num_cores: int) -> list[int]:
    return [(num_cores - 1) - c for c in range(num_cores)]


def perm_random(num_cores: int, seed: int = 42) -> list[int]:
    import random
    rng = random.Random(seed)
    p = list(range(num_cores))
    rng.shuffle(p)
    return p


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


def _compile_and_bench(M, N, K, split, perm):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(split), _force_perm(perm):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(split), _force_perm(perm):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:50]}"


# Real LLM matmul shapes. (label, N, K). Trimmed to 4 representative shapes:
# - q_proj: square, most common pattern
# - kv_proj: narrow N (1024 ÷ 64 = 16 sticks) — boundary case
# - gate: wide N (28672)
# - DSv3 o_proj: wide K (16384), HMI-heavy
LLM_SHAPES = [
    ("L3-70B q_proj",   8192,  8192),
    ("L3-70B kv_proj",  1024,  8192),
    ("L3-70B gate",    28672,  8192),
    ("DSv3 o_proj",     7168, 16384),
]
M_VALUES = [128, 512, 2048]

# Splits to try — pure-M was already established by HMI BW probe as
# slowest, so focus on m·n splits where the multicast permutation
# might matter.
SPLITS = [
    ((8, 4, 1), "(8,4,1)"),
    ((4, 8, 1), "(4,8,1)"),
]


def main() -> int:
    print("# Multicast core_id permutation — broader sweep\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    print("| shape | M | split | id ms | m_adj ms | rev ms | rand ms | spread (struct) | spread (incl rand) |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|")

    results = []
    for label, N, K in LLM_SHAPES:
        for M in M_VALUES:
            for split, split_label in SPLITS:
                m, n, _ = split
                walls = {}
                perms = [
                    ("id",    perm_identity(32)),
                    ("m_adj", perm_m_adjacent(m, n)),
                    ("rev",   perm_reversed(32)),
                    ("rand",  perm_random(32)),
                ]
                row_walls = []
                for pname, perm in perms:
                    ms, err = _compile_and_bench(M, N, K, split, perm)
                    walls[pname] = ms if ms is not None else None
                    row_walls.append((pname, ms, err))

                cells = []
                struct_walls = []
                all_walls = []
                for pname, ms, err in row_walls:
                    if err:
                        cells.append("ERR")
                    else:
                        cells.append(f"{ms:.2f}")
                        all_walls.append(ms)
                        if pname != "rand":
                            struct_walls.append(ms)

                if struct_walls:
                    spread_struct = (max(struct_walls) - min(struct_walls)) / min(struct_walls) * 100
                else:
                    spread_struct = float('nan')
                if all_walls:
                    spread_all = (max(all_walls) - min(all_walls)) / min(all_walls) * 100
                else:
                    spread_all = float('nan')

                print(f"| {label} | {M} | {split_label} | "
                      f"{cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | "
                      f"{spread_struct:.1f}% | {spread_all:.1f}% |", flush=True)
                results.append((label, M, split_label, walls,
                                spread_struct, spread_all))
    print()

    # Aggregate
    print("## Aggregate stats\n")
    structured = [r[4] for r in results if not (r[4] != r[4])]  # filter NaN
    inc_random = [r[5] for r in results if not (r[5] != r[5])]
    if structured:
        print(f"  structured-perm spread:")
        print(f"    rows: {len(structured)}")
        print(f"    median: {statistics.median(structured):.2f}%")
        print(f"    max:    {max(structured):.2f}%")
        over_2pct = sum(1 for s in structured if s > 2.0)
        over_5pct = sum(1 for s in structured if s > 5.0)
        print(f"    > 2% spread: {over_2pct}/{len(structured)}")
        print(f"    > 5% spread: {over_5pct}/{len(structured)}")
    if inc_random:
        print(f"  including-random spread:")
        print(f"    median: {statistics.median(inc_random):.2f}%")
        print(f"    max:    {max(inc_random):.2f}%")
    print()

    print("## Verdict\n")
    if structured and max(structured) <= 2.0:
        print("  CONFIRMED: structured permutations within 2% across all shapes/splits.")
        print("  HMI multicast does NOT depend on physical core placement on AIU.")
        print("  Multicast core_id permutation is closed as a torch_spyre lever.")
    elif structured and max(structured) <= 5.0:
        print("  MOSTLY CLOSED: spread within 5% but with some outliers worth")
        print("  investigating before fully closing.")
    else:
        print("  NOT CLOSED: significant spread found. Worth pursuing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
