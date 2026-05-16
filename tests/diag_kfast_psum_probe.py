# Copyright 2026 The Torch-Spyre Authors.
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

"""Does k_fast Layer 2 actually emit reduce-sum-ring instructions?

The inductor-side `_k_fast_core_to_slice_mapping` permutes the
core-id-to-slice mapping so K-cohort members sit at adjacent physical
core IDs. The design intent (per the docstring) is that the PSUM
reduction then traverses 1 hop on the reduce-sum ring instead of m*n.

This probe runs the same narrow-N small-M matmul twice -- with
`core_id_k_fast_emission` on and off -- captures the emitted SDSC
bundle dirs, and lists their contents so we can diff what actually
changed and check whether a ring-based PSUM algorithm shows up.

Triggering k_fast: matmul with M=1, K=4096, N=32 -- decode-step shape,
narrow-N, planner is expected to pick (1, n, k>1) split which makes
the dim_splits[K] > 1 condition fire.

Run: SENCORES=32 .venv/bin/python tests/diag_kfast_psum_probe.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from unittest.mock import patch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_spyre

torch_spyre._autoload()

from torch._inductor import config as t_inductor_config
from torch_spyre._inductor import config as ts_config
from torch_spyre.execution import async_compile as ac


M, K, N = 128, 8192, 8192
DTYPE = torch.float16
DEVICE = "spyre"
WARMUP = 5
ITERS = 20


def time_compiled(fn, args):
    for _ in range(WARMUP):
        out = fn(*args)
        _ = out.sum().item()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn(*args)
        _ = out.sum().item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds
    return times


def run(k_fast_on: bool):
    captured: list[tuple[str, str | None]] = []
    orig_sdsc = ac.SpyreAsyncCompile.sdsc

    def wrapped(self, kernel_name, specs):
        result = orig_sdsc(self, kernel_name, specs)
        output_dir = (
            getattr(result, "output_dir", None)
            or getattr(result, "_output_dir", None)
        )
        captured.append((kernel_name, output_dir))
        return result

    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("sencores", 32),
        ts_config.patch("core_id_k_fast_emission", k_fast_on),
        patch.object(ac.SpyreAsyncCompile, "sdsc", wrapped),
    ]
    for p in patchers:
        p.__enter__()
    torch.compiler.reset()
    try:
        X = torch.randn((M, K), dtype=DTYPE, device=DEVICE)
        W = torch.randn((K, N), dtype=DTYPE, device=DEVICE)

        def fn(x, w):
            return torch.matmul(x, w)

        compiled = torch.compile(fn, fullgraph=True)
        try:
            times = time_compiled(compiled, (X, W))
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            times = []
    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)
    return times, captured


def list_bundle(d: str | None):
    if d is None or not os.path.isdir(d):
        return None
    return sorted(os.listdir(d))


def main():
    print(f"k_fast PSUM-ring probe -- matmul(X:({M},{K}), W:({K},{N}))")
    print(f"  SENCORES={os.environ.get('SENCORES', '32')}, WARMUP={WARMUP}, "
          f"ITERS={ITERS}\n")

    print("=== k_fast OFF (SPYRE_CORE_ID_K_FAST_EMISSION=0) ===")
    times_off, caps_off = run(k_fast_on=False)
    if times_off:
        print(f"  median: {statistics.median(times_off):.1f} us  "
              f"min: {min(times_off):.1f} us  max: {max(times_off):.1f} us")
    print(f"  kernels emitted ({len(caps_off)}):")
    for name, d in caps_off:
        files = list_bundle(d)
        print(f"    {name}: {d}")
        if files:
            print(f"      files: {files}")

    print()
    print("=== k_fast ON (SPYRE_CORE_ID_K_FAST_EMISSION=1) ===")
    times_on, caps_on = run(k_fast_on=True)
    if times_on:
        print(f"  median: {statistics.median(times_on):.1f} us  "
              f"min: {min(times_on):.1f} us  max: {max(times_on):.1f} us")
    print(f"  kernels emitted ({len(caps_on)}):")
    for name, d in caps_on:
        files = list_bundle(d)
        print(f"    {name}: {d}")
        if files:
            print(f"      files: {files}")

    if times_off and times_on:
        m_off = statistics.median(times_off)
        m_on = statistics.median(times_on)
        speedup = m_off / m_on if m_on > 0 else 0
        print(f"\n=== Wall-clock comparison ===")
        print(f"  k_fast OFF: {m_off:.1f} us median")
        print(f"  k_fast ON:  {m_on:.1f} us median")
        print(f"  speedup:    {speedup:.2f}x")

    print(f"\n=== Manual next step ===")
    print(f"  diff the bundle contents to see what changed under k_fast.")
    print(f"  grep for 'psum', 'ring', 'unichain', 'bichain', 'singleshot' in")
    print(f"  init.txt / addr.txt of each bundle.")


if __name__ == "__main__":
    main()
