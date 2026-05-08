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

"""Probe 5 — generality of the n=1 streaming-output fast path.

Probe 4 confirmed on DSv3 o_proj M=2048: when n=1, the kernel
template emits a streaming-output path that absorbs C_psum overage
that's catastrophic at n>1. Smoking gun: (8, 1, 4)+kf is 18 ms
while (1, 8, 4)+kf is 125 ms — same shape, same C_psum, n=1 is 7×
faster.

This probe tests whether the fast path is general to wide-N shapes,
or specific to DSv3 o_proj. We measure four shapes the LX-Phase-1
diagnostic flagged as production overflow cases (pure-M C_psum > LX
at M=2048):

| shape | (M, N, K) | pure-M C_psum |
|---|---|---:|
| L3-70B gate_proj M=2048    | (2048, 28672, 8192)  | 7.00 MB |
| DSv3 gate_proj M=2048      | (2048, 18432, 7168)  | 4.50 MB |
| Mixtral gate_proj M=2048   | (2048, 14336, 4096)  | 3.50 MB |
| L3-70B down_proj M=2048    | (2048, 8192, 28672)  | 2.00 MB |

For each shape, we compare:
  - (32, 1, 1) — pure-M planner default
  - (16, 1, 2) + kf — streaming pipeline regime
  - (8, 1, 4)  + kf — streaming pipeline regime
  - (1, 1, 32) + kf — pure-K allreduce regime
  - (1, 8, 4)  + kf — n>1 catastrophic control

If the n=1 fast path is general, the (m, 1, k)+kf walls should be
below catastrophic (~60 ms) on all shapes. If it's specific to
o_proj, some shapes will show catastrophe even at n=1.

Usage:
    python tests/diag_emission_aware_lx_p5_n1_generality.py
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
from torch_spyre._inductor import core_division as _core_div  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16


# Wide-N shapes the LX-Phase-1 diagnostic flagged as pure-M overflow.
SHAPES = [
    ("L3-70B gate_proj M=2048", 2048, 28672, 8192),
    ("DSv3 gate_proj M=2048",   2048, 18432, 7168),
    ("Mixtral gate_proj M=2048", 2048, 14336, 4096),
    ("L3-70B down_proj M=2048", 2048, 8192, 28672),
]


# Splits to test on each shape. Same family as Probe 4, narrowed.
CONFIGS = [
    ((32, 1, 1), "identity", "pure-M"),
    ((16, 1, 2), "k_fast",   "n=1 chain=2 (pipeline)"),
    ((8,  1, 4), "k_fast",   "n=1 chain=4 (pipeline)"),
    ((1,  1, 32), "k_fast",  "pure-K (allreduce)"),
    ((1,  8, 4),  "k_fast",  "n>1 catastrophic control"),
]


# ---- machinery ----------------------------------------------------

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


def _compile_and_bench(M, N, K, split, perm):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _permutation(perm), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _permutation(perm), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


def _is_valid(M, N, K, split):
    """Divisibility + stick alignment check."""
    m, n, k = split
    if M % m or N % n or K % k:
        return False, "divisibility"
    if (N // n) % 64 != 0:
        return False, "stick-align"
    return True, ""


def _per_core_psum_mb(M, N, split):
    m, n, _ = split
    return (M // m) * (N // n) * 4 / (1024 * 1024)


# ---- main ----------------------------------------------------------

def main() -> int:
    print("# Probe 5 — generality of the n=1 streaming-output fast path\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16\n")
    print("Hypothesis: (m, 1, k)+kf walls << (1, n>1, k)+kf walls "
          "for any wide-N shape with C_psum overage.\n")

    for label, M, N, K in SHAPES:
        print(f"## {label}  shape=({M}, {N}, {K})\n")
        print("| split | description | C_psum/core | overage | wall ms |")
        print("|---|---|---:|---:|---:|")
        for split, perm, desc in CONFIGS:
            ok, reason = _is_valid(M, N, K, split)
            if not ok:
                print(f"| {split} | {desc} | — | — | SKIP ({reason}) |")
                continue
            cpsum = _per_core_psum_mb(M, N, split)
            overage = cpsum / 2.0   # LX = 2 MB
            ms, err = _compile_and_bench(M, N, K, split, perm)
            wall = f"{ms:.3f}" if ms is not None else f"ERR ({err[:30]})"
            print(f"| {split} | {desc} | {cpsum:.2f} MB | "
                  f"{overage:.2f}× | {wall} |")
        print()

    print("## Reading guide\n")
    print("Per-shape pattern to look for:")
    print("  pure-M wall ≈ HMI/compute baseline")
    print("  (16, 1, 2)+kf and (8, 1, 4)+kf walls ≤ ~2× pure-M  →  fast path active")
    print("  (1, 1, 32)+kf wall ≈ allreduce regime")
    print("  (1, 8, 4)+kf wall >> 2× pure-M (catastrophic)  →  confirms n>1 penalty")
    print()
    print("Generality verdict:")
    print("  All 4 shapes show pattern  →  n=1 fast path is general")
    print("  Pattern only on o_proj-like shapes  →  shape-specific")
    print("  Mixed  →  characterise the trigger condition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
