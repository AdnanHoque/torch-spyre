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

"""SFP ring 2D-direction sensitivity probe.

The k_fast PR (1932) packs k-collaborators on adjacent core_ids,
reducing 1D distance from m·n to 1. The chip is 8×4 (per AIU 1.0
ISA spec), so this 1D adjacency maps to ROW-direction adjacency in
2D physical layout (assuming row-major core_id assignment).

This probe tests whether the SFP ring traversal cost depends on
2D direction by comparing three permutations under split (1, 16, 2):

  default          — k-collaborators 16 apart in 1D = 2 hops in COLUMN dir
  k_fast (PR 1932) — k-collaborators 1 apart in 1D  = 1 hop in ROW dir
  column-direction — k-collaborators 8 apart in 1D  = 1 hop in COLUMN dir

If walls go (default >> k_fast ≈ column): hop count is all that
matters, SFP ring is direction-symmetric → no new lever.

If walls differ between k_fast and column: SFP ring has a preferred
direction → new lever (pick the better direction per shape).

Usage:
    python tests/diag_2d_direction_probe.py
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


WARMUP = 3
ITERS = 8
DTYPE = torch.float16

_orig_multi = _core_div.multi_dim_iteration_space_split
_orig_generate_sdsc = _co.generate_sdsc


@contextmanager
def _force_split(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        if len(it_space) != 3 or target[0] * target[1] * target[2] != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(it_space.keys())}
    _core_div.multi_dim_iteration_space_split = _forced
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


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


# ---- permutations specific to (1, 16, 2) split -----------------------
# Encoding: m_slice = c % 1 = 0; n_slice = c % 16; k_slice = c // 16
# (When m=1, the m_slice term drops; n_slice is the inner-radix.)
# So logical c=0..15 are k=0 group; c=16..31 are k=1 group.
# k-collaborator pairs: (0, 16), (1, 17), ..., (15, 31).
#
# The permutation perm[p] specifies what logical core physical position p
# should execute. Physical p = row * 8 + col under 8×4 row-major.

def perm_default():
    """Identity. k-collaborators 16 apart in 1D = 2 hops column dir."""
    return list(range(32))


def perm_k_fast():
    """k_fast (PR 1932). perm[c] = (c % 2) * 16 + (c // 2).

    k-collaborators end up at physical positions (0,1), (2,3), ...
    1 hop in row direction.
    """
    return [(c % 2) * 16 + (c // 2) for c in range(32)]


def perm_column_dir():
    """Column-direction packing for (1, 16, 2): k-collaborators 8 apart.

    Physical (row 0, col c) and (row 1, col c) are k-collaborators.
    1 hop in column direction.

    Mapping:
      p in 0..7  (row 0): execute logical (n=p, k=0) = p
      p in 8..15 (row 1): execute logical (n=p-8, k=1) = p + 8
      p in 16..23 (row 2): execute logical (n=p-8, k=0) = p - 8
      p in 24..31 (row 3): execute logical (n=p-16, k=1) = p
    """
    return [
        c if c < 8 else
        c + 8 if c < 16 else
        c - 8 if c < 24 else
        c
        for c in range(32)
    ]


def perm_diagonal():
    """Pack k-collaborators 9 apart in 1D = (1 row, 1 col) diagonal in 2D.

    perm[p] = p XOR 9 (i.e. flip bits to map p to k-partner)
    Actually need to be careful — let me build it explicitly.

    p in 0..7  (row 0, col c=p): execute logical (n=p, k=0) = p
    p in 9..15 (row 1, col 1..7): execute logical (n=col, k=1) = col + 16
    Hmm this won't form a clean permutation. Skip diagonal for now.
    """
    raise NotImplementedError


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


def main() -> int:
    print("# 2D-direction probe — does SFP ring care about row vs column?\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32, split=(1,16,2)\n")

    # Verify permutations are valid + show what they pack
    print("## Permutation sanity check\n")
    for name, fn in [("default", perm_default),
                     ("k_fast", perm_k_fast),
                     ("col_dir", perm_column_dir)]:
        p = fn()
        # k-collaborator of logical 0 is logical 16. Find their physical positions.
        phys_of_logical_0 = p.index(0)
        phys_of_logical_16 = p.index(16)
        row_0, col_0 = divmod(phys_of_logical_0, 8)
        row_16, col_16 = divmod(phys_of_logical_16, 8)
        print(f"  {name}: logical 0 at physical ({row_0},{col_0}), "
              f"logical 16 at physical ({row_16},{col_16}) — "
              f"|Δrow|={abs(row_0-row_16)}, |Δcol|={abs(col_0-col_16)}")
    print()

    # Llama-style narrow-N small-M shape (the kv_proj win-band shape)
    SHAPES = [
        ("L3-70B kv_proj M=128",  128, 1024,  8192),
        ("L3-70B kv_proj M=512",  512, 1024,  8192),
        ("L3-70B kv_proj M=2048", 2048, 1024, 8192),
        # Test with a different N for variety
        ("Mixtral kv_proj M=128", 128, 1024,  4096),
    ]

    SPLIT = (1, 16, 2)
    PERMS = [
        ("default",  perm_default()),
        ("k_fast",   perm_k_fast()),
        ("col_dir",  perm_column_dir()),
    ]

    print("## Wall measurements\n")
    print("| shape | default ms | k_fast ms | col_dir ms | "
          "col vs k_fast | k_fast vs default |")
    print("|---|---:|---:|---:|---:|---:|")

    rows = []
    for label, M, N, K in SHAPES:
        walls = {}
        for perm_name, perm in PERMS:
            ms, err = _compile_and_bench(M, N, K, SPLIT, perm)
            if err:
                walls[perm_name] = None
                print(f"| {label} | ERR ({perm_name}): {err[:40]} | — | — | — | — |")
                break
            walls[perm_name] = ms
        else:
            d, kf, cd = walls['default'], walls['k_fast'], walls['col_dir']
            col_vs_kf = (cd / kf - 1) * 100 if kf else 0
            kf_vs_def = (kf / d - 1) * 100 if d else 0
            print(f"| {label} | {d:.3f} | {kf:.3f} | {cd:.3f} | "
                  f"{col_vs_kf:+.1f}% | {kf_vs_def:+.1f}% |")
            rows.append((label, M, N, K, d, kf, cd))
    print()

    if rows:
        # Aggregate
        deltas_kf_vs_default = [(r[5] - r[4]) / r[4] * 100 for r in rows]
        deltas_col_vs_kfast = [(r[6] - r[5]) / r[5] * 100 for r in rows]
        deltas_col_vs_default = [(r[6] - r[4]) / r[4] * 100 for r in rows]

        print("## Aggregate\n")
        print(f"  k_fast vs default (median): {statistics.median(deltas_kf_vs_default):+.1f}%")
        print(f"  col_dir vs default (median): {statistics.median(deltas_col_vs_default):+.1f}%")
        print(f"  col_dir vs k_fast (median): {statistics.median(deltas_col_vs_kfast):+.1f}%")
        print()

        print("## Verdict\n")
        max_col_vs_kf = max(abs(d) for d in deltas_col_vs_kfast)
        if max_col_vs_kf < 3:
            print("  HOP COUNT IS ALL THAT MATTERS: col_dir and k_fast within 3% of")
            print("  each other. SFP ring is direction-symmetric on the 8×4 layout.")
            print("  k_fast captures the win; no new direction-aware lever needed.")
        elif max_col_vs_kf < 10:
            print("  WEAK DIRECTION DEPENDENCE: col_dir and k_fast differ by 3-10%.")
            print("  May be worth picking direction per shape; small effect.")
        else:
            print("  STRONG DIRECTION DEPENDENCE: col_dir and k_fast differ by >10%.")
            print("  SFP ring has a preferred traversal direction. New torch_spyre")
            print("  lever: pick the best direction per workload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
