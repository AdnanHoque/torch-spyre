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

"""Probe 6 verification — chain-length regime structure on clean rebuild.

Replicates the May 2026 Probe 6 measurement: full (m, 1, k) sweep on
three production shapes, plus an n=8 control to verify the
catastrophic regime exists.

Original finding to verify:
  Pipeline regime  (chain ≤ 4):    +3 ms regime cost
  Sync regime      (chain 8-16):   +23-55 ms regime cost
  Allreduce regime (chain = 32):   +14-15 ms regime cost
  chain=4 → chain=8 boundary:      sharp and universal

Measure walls for each (split, shape):
  (32, 1, 1) identity        — pure-M baseline
  (16, 1, 2)+kf chain=2      — pipeline
  (8, 1, 4)+kf chain=4       — pipeline
  (4, 1, 8)+kf chain=8       — sync (regime jump expected here)
  (2, 1, 16)+kf chain=16     — sync
  (1, 1, 32)+kf chain=32     — allreduce
  (1, 8, 4)+kf n=8 control   — catastrophic if C_psum > LX

Regime cost = wall − max(compute_bound, hmi_bound).
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch  # noqa: E402
import torch._inductor.config as _icfg  # noqa: E402

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, "/home/adnan/dt-inductor/torch-spyre")

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402

try:
    from torch_spyre._inductor import work_division as _planner  # noqa: E402
except ImportError:
    from torch_spyre._inductor import core_division as _planner  # noqa: E402

from torch_spyre._inductor import config as ts_config  # noqa: E402

WARMUP = 3
ITERS = 8
DTYPE = torch.float16


@dataclass(frozen=True)
class Shape:
    label: str
    M: int
    N: int
    K: int


SHAPES = [
    Shape("DSv3 o_proj M=2048",     2048,  7168, 16384),
    Shape("DSv3 gate_proj M=2048",  2048, 18432,  7168),
    Shape("Mixtral gate_proj M=2048", 2048, 14336, 4096),
]


SPLITS = [
    # (m, n, k, label, kfast)
    (32, 1,  1, "(32,1,1) identity", False),
    (16, 1,  2, "(16,1,2)+kf chain=2", True),
    ( 8, 1,  4, "(8,1,4)+kf chain=4", True),
    ( 4, 1,  8, "(4,1,8)+kf chain=8", True),
    ( 2, 1, 16, "(2,1,16)+kf chain=16", True),
    ( 1, 1, 32, "(1,1,32)+kf chain=32", True),
    ( 1, 8,  4, "(1,8,4)+kf n=8 ctrl", True),
]


_orig_multi = _planner.multi_dim_iteration_space_split


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
    _planner.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _planner.multi_dim_iteration_space_split = _orig_multi


@contextmanager
def _kfast_emission(enabled):
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


def measure(M, N, K, m, n, k, kfast):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _kfast_emission(kfast), _force_split((m, n, k)):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _kfast_emission(kfast), _force_split((m, n, k)):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    print("# Probe 6 verification — chain-length regime structure on clean rebuild")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32")
    print()

    results = []
    for s in SHAPES:
        # Compute baseline bounds for "regime cost" computation
        # (no Spyre HMI floor here, so we just report walls and let the user
        # compute regime cost from baseline)
        flops = 2 * s.M * s.N * s.K
        compute_bound_ms = flops / 72.1e9  # 72.1 TFLOPS at fp16
        # Naive HMI bound: pure-M loads B 32× (M·N·K·2 / 166e9 / 1e-3)
        # but more useful is the (32,1,1) measured baseline
        print(f"\n## {s.label}: ({s.M}, {s.N}, {s.K})")
        print(f"   compute bound = {compute_bound_ms:.2f} ms")
        print()
        print("| split | wall ms |")
        print("|---|---:|")
        for (m, n, k, label, kfast) in SPLITS:
            ms, err = measure(s.M, s.N, s.K, m, n, k, kfast)
            wall_str = f"{ms:.2f}" if ms is not None else f"ERR ({err})"
            print(f"| {label} | {wall_str} |")
            sys.stdout.flush()
            results.append((s, m, n, k, label, kfast, ms))

    # Summary: regime cost comparison vs (32,1,1) baseline per shape
    print("\n\n## Regime cost summary")
    print()
    print("Regime cost = wall − pure-M baseline. Highlights any regime jump.")
    print()
    print("| shape | chain=2 | chain=4 | chain=8 | chain=16 | chain=32 | n=8 ctrl |")
    print("|---|---:|---:|---:|---:|---:|---:|")

    by_shape = {}
    for (s, m, n, k, label, kfast, ms) in results:
        by_shape.setdefault(s.label, {})[(m, n, k)] = ms

    for shape_label, pts in by_shape.items():
        baseline = pts.get((32, 1, 1))
        if baseline is None:
            continue
        def diff(key):
            v = pts.get(key)
            if v is None:
                return "ERR"
            return f"{v - baseline:+.2f}"
        row = " | ".join([
            shape_label,
            diff((16, 1, 2)),
            diff((8, 1, 4)),
            diff((4, 1, 8)),
            diff((2, 1, 16)),
            diff((1, 1, 32)),
            diff((1, 8, 4)),
        ])
        print(f"| {row} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
