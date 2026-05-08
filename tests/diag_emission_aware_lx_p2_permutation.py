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

"""Probe 2 — permutation discriminator at (1, 16, 2).

After Phase 0 refuted the chain-cooperative LX hypothesis (M1), the
remaining puzzle in the +id-vs-+kf gap at (m·n=16, k=2) is:

  Is the +id penalty driven *only* by K-collaborator distance on
  the SFP ring (which would make every "K-collab 16-hops-apart"
  permutation equally slow), or does the larger cohort/cell
  arrangement matter too?

For the (1, 16, 2) split:
  - K-collaborators of cell c are logical IDs c and c+16
  - Same-k cohort (same k_idx, different cell) is 16 contiguous
    logical IDs (0..15 for k=0; 16..31 for k=1)

Permutations tested:

| name | K-collab physical distance | k-cohort layout |
|---|---|---|
| identity     | 16 | contiguous block      |
| reversed     | 16 | contiguous block      |
| stride2      | 16 | contiguous (every-other physical) |
| antipodal    | 16 | swapped halves        |
| bit_reverse  | varies | scrambled         |
| random_42    | random | random            |
| random_7     | random | random            |
| k_fast       | 1  | interleaved (every-other) |
| block_cyclic | 1  | interleaved (every-other) — equals k_fast at this split |

If the +id penalty is purely about K-collab distance:
  - identity, reversed, stride2, antipodal walls all ≈ 11 ms
  - k_fast, block_cyclic walls all ≈ 4 ms (or 31 ms for o_proj)
  - random walls intermediate, scaled with average K-collab distance

If cohort arrangement matters too:
  - identity, reversed, stride2, antipodal differ in walls despite
    same K-collab distance

Usage:
    python tests/diag_emission_aware_lx_p2_permutation.py
    python tests/diag_emission_aware_lx_p2_permutation.py --shape l3_70b_kv
"""

from __future__ import annotations

import argparse
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
from torch_spyre._inductor import core_division as _core_div  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16


SHAPES = {
    "dsv3_o_proj": ("DSv3 o_proj M=2048", 2048, 7168, 16384),
    "l3_70b_kv":   ("L3-70B kv_proj M=2048", 2048, 1024, 8192),
}

PERMUTATIONS = [
    "identity",
    "reversed",
    "stride2",
    "antipodal",
    "bit_reverse",
    "block_cyclic",   # equals k_fast at (1, 16, 2)
    "k_fast",
    "random_42",
    "random_7",
]

SPLIT = (1, 16, 2)


# ---- machinery (mirrors Probe 1) -----------------------------------

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
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


@contextmanager
def _permutation(name: str):
    prev = ts_config.core_id_permutation
    ts_config.core_id_permutation = name
    try:
        yield
    finally:
        ts_config.core_id_permutation = prev


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


def _compile_and_bench(M, N, K, split, perm_name):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _permutation(perm_name), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _permutation(perm_name), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


def _kcollab_distance(perm_name: str, num_cores: int = 32,
                      m: int = 1, n: int = 16, k: int = 2) -> float:
    """Compute average physical distance between K-collaborators
    under the named permutation, modulo the ring length.

    Returns avg over all cells of |perm[c] - perm[c+m*n]| in ring metric.
    """
    perm = _materialize_perm(perm_name, num_cores, m, n, k)
    # inverse perm: physical_pos_of_logical[L] = i where perm[i] = L
    phys_of_log = [0] * num_cores
    for i, L in enumerate(perm):
        phys_of_log[L] = i
    distances = []
    for cell in range(m * n):
        # K-collaborators: logical cell, cell+m*n
        l0 = cell
        l1 = cell + m * n
        p0 = phys_of_log[l0]
        p1 = phys_of_log[l1]
        d = min(abs(p0 - p1), num_cores - abs(p0 - p1))
        distances.append(d)
    return statistics.mean(distances)


def _materialize_perm(name, num_cores, m, n, k):
    """Recreate the permutation list — must match compute_ops._get_core_id_permutation."""
    if name == "identity":
        return list(range(num_cores))
    if name == "reversed":
        return list(range(num_cores - 1, -1, -1))
    if name == "stride2":
        return list(range(0, num_cores, 2)) + list(range(1, num_cores, 2))
    if name == "block_cyclic":
        half = num_cores // 2
        out = []
        for i in range(half):
            out.append(i)
            out.append(half + i)
        return out
    if name == "antipodal":
        half = num_cores // 2
        return list(range(half, num_cores)) + list(range(half))
    if name == "bit_reverse":
        n_bits = (num_cores - 1).bit_length()
        out = []
        for c in range(num_cores):
            r = 0
            for b in range(n_bits):
                if c & (1 << b):
                    r |= 1 << (n_bits - 1 - b)
            out.append(r)
        return out
    if name.startswith("random_"):
        seed = int(name.split("_", 1)[1])
        import random as _random
        rng = _random.Random(seed)
        out = list(range(num_cores))
        rng.shuffle(out)
        return out
    if name == "k_fast":
        # Generalized — we mirror compute_ops behaviour for our split
        out = [(c % k) * (m * n) + (c // k) for c in range(num_cores)]
        return out
    raise ValueError(f"unknown permutation: {name}")


# ---- main -----------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default="dsv3_o_proj",
                        choices=list(SHAPES.keys()))
    args = parser.parse_args()

    label, M, N, K = SHAPES[args.shape]
    m, n, k = SPLIT

    print("# Probe 2 — permutation discriminator at (1, 16, 2)\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, "
          f"split={SPLIT} (m={m}, n={n}, k={k})\n")
    print(f"## {label}  shape=({M}, {N}, {K})\n")
    print("| permutation | avg K-collab distance | wall (ms) |")
    print("|---|---:|---:|")

    for perm in PERMUTATIONS:
        dist = _kcollab_distance(perm, 32, m, n, k)
        ms, err = _compile_and_bench(M, N, K, SPLIT, perm)
        if ms is None:
            print(f"| {perm} | {dist:.1f} | ERR ({err[:30]}) |")
        else:
            print(f"| {perm} | {dist:.1f} | {ms:.3f} |")

    print()
    print("## Reading guide\n")
    print("If +id penalty is purely K-collab-distance-driven:")
    print("  identity, reversed, stride2, antipodal walls ≈ each other (all 16 hops)")
    print("  k_fast, block_cyclic walls fastest (1 hop)")
    print("  random walls scale with avg distance")
    print()
    print("If cohort arrangement matters too:")
    print("  identity, reversed, stride2, antipodal differ despite same K-collab distance")
    print("  some non-obvious permutation might be even faster than k_fast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
