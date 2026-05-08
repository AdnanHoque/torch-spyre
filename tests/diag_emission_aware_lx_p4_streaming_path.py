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

"""Probe 4 — characterise the (1, 1, 32) streaming-output fast path.

Probe 1 found that DSv3 o_proj M=2048 under (1, 1, 32)+kf runs at
30 ms despite C_psum = 58.7 MB (29× LX) — the only catastrophic-
overage case that's fast. Hypothesis: the kernel template detects
some structural condition and emits a streaming-output path that
writes accumulator to HMI as the chain reduces, bypassing LX
residency.

Two competing hypotheses for the trigger condition:

  H1 (single-chain streaming): m·n = 1. With one cell across all
       cores, the kernel reduces along K and streams output. The
       fast path requires a single chain head holding the running
       sum to a single output destination.

  H2 (n=1 column structure): n = 1 alone triggers it. The kernel
       template handles "single output column" splits via streaming
       regardless of m. Multiple chain heads, one per M-tile, all
       streaming.

Discriminator: the (m, 1, k) family at DSv3 o_proj M=2048. All have
n = 1 (so H2 predicts all fast). Only m = n = 1 has m·n = 1 (so H1
predicts only the last is fast).

| split | m·n cells | chain length | n=1? |
|---|---:|---:|:-:|
| (32, 1, 1) | 32 | 1 | ✓ (no chain, pure-M) |
| (16, 1, 2)+kf | 16 | 2 | ✓ |
| (8, 1, 4)+kf | 8 | 4 | ✓ |
| (4, 1, 8)+kf | 4 | 8 | ✓ |
| (2, 1, 16)+kf | 2 | 16 | ✓ |
| (1, 1, 32)+kf | 1 | 32 | ✓ |

Pure-M (32, 1, 1) is included as the n=1 baseline (no K-chain).
The (1, n=8, k=4)+kf catastrophic case is included as the n>1
control (we measured 127 ms in Probe 1).

If the (m, 1, k) middle splits are all fast (~30 ms or under): H2
wins — n=1 alone triggers the streaming path. This is the
production-relevant finding because it means the planner has many
LX-fitting options for wide-N prefill.

If only (1, 1, 32)+kf is fast and others are catastrophic: H1 wins.
The fast path is single-chain-only. Then the lever is much narrower
(only fully-reduced shapes).

Usage:
    python tests/diag_emission_aware_lx_p4_streaming_path.py
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
LX_BYTES = 2 * 1024 * 1024


# DSv3 o_proj M=2048 — the catastrophic-regime shape from Probe 1.
SHAPE = (2048, 7168, 16384)


# Splits to test. All n = 1 except the last control row at n > 1.
SPLITS = [
    # (m, n, k)
    ((32, 1, 1), "n=1, no chain (pure-M baseline)"),
    ((16, 1, 2), "n=1, chain=2"),
    ((8,  1, 4), "n=1, chain=4"),
    ((4,  1, 8), "n=1, chain=8"),
    ((2,  1, 16), "n=1, chain=16"),
    ((1,  1, 32), "n=1, chain=32 (pure-K, single-chain — known fast)"),
    ((1,  8, 4),  "n>1 control (chain=4, 8 cells — known catastrophic)"),
]


# ---- machinery (mirrors prior probes) ------------------------------

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


def _per_core_psum(M, N, split):
    m, n, k = split
    return (M // m) * (N // n) * 4


# ---- main ----------------------------------------------------------

def main() -> int:
    M, N, K = SHAPE
    print("# Probe 4 — characterise the (1, 1, 32) streaming-output fast path\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, "
          f"shape=DSv3 o_proj ({M}, {N}, {K})\n")
    print("Testing the (m, 1, k) family to discriminate:")
    print("  H1: only (1, 1, 32) is fast (single chain triggers streaming)")
    print("  H2: every n=1 split is fast (n=1 column triggers streaming)\n")

    print("| split | description | C_psum/core | overage(LX) | "
          "perm | wall ms |")
    print("|---|---|---:|---:|---|---:|")

    for split, desc in SPLITS:
        m, n, k = split
        c_psum = _per_core_psum(M, N, split)
        overage = c_psum / LX_BYTES
        perm = "k_fast" if k > 1 else "identity"
        ms, err = _compile_and_bench(M, N, K, split, perm)
        c_str = f"{c_psum / 1024 / 1024:.2f} MB"
        if ms is None:
            wall = f"ERR ({err[:30]})"
        else:
            wall = f"{ms:.3f}"
        print(f"| {split} | {desc} | {c_str} | {overage:.2f}× | "
              f"{perm} | {wall} |")

    print()
    print("## Reading guide\n")
    print("  H1 (single-chain) confirmed if: only (1, 1, 32) is < 60 ms")
    print("                                  and (m, 1, k) for m>1 are catastrophic")
    print("  H2 (n=1 trigger)  confirmed if: all (m, 1, k) walls are < 60 ms")
    print("                                  and only the n>1 control is catastrophic")
    print("  Mixed: kernel template behaviour depends on more than n alone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
