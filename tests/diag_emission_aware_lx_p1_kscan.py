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

"""Emission-aware LX scheduling — Probe 1: chain-LX scaling under +kf.

Sweeps the K-split factor `k` at fixed shape under k_fast emission
to test whether per-core LX overage is absorbed by chain-cooperative
residency. The hypothesis (M1 in the Phase 0 scope doc):

  Same-k-cohort cores under k_fast sit in clustered ring positions,
  so the Data QuadRing can multicast operand A from one HMI fetch
  to the cohort. Effective per-chain LX = k × LX_per_core. A split
  whose A_per_core overflows LX may run without re-fetch penalty
  if k * 2 MB > A_per_core.

For each shape, sweeps k ∈ {1, 2, 4, 8, 16, 32} (with n = 32/k,
m = 1) and records wall under +kf and +id. Also records pure-M
control (32, 1, 1) for context.

The probe is designed to discriminate three mechanisms:

  M1 (chain-cooperative LX): wall plateaus past k where chain-LX
                             > A_per_core
  M2 (kernel-template fast path): wall flat regardless of k
  M3 (PSUM forward-pipelining): wall increases at high k as PSUM
                                payload grows

This script must run on a branch with the k_fast emission
infrastructure (i.e., descendant of `feat-k-fast-emission`). On
hmi-cost-model-simulator and earlier, the `core_id_permutation`
config flag does not exist.

Usage:
    python tests/diag_emission_aware_lx_p1_kscan.py
    python tests/diag_emission_aware_lx_p1_kscan.py --shape dsv3_o_proj
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


# ---- bench params ---------------------------------------------------

WARMUP = 3
ITERS = 12
DTYPE = torch.float16
LX_BYTES_PER_CORE = 2 * 1024 * 1024


# ---- shape matrix (chosen so chain-LX threshold sits inside the k sweep)

SHAPES = {
    "dsv3_o_proj":   ("DSv3 o_proj M=2048",   2048, 7168, 16384),
    "l3_70b_kv":     ("L3-70B kv_proj M=2048", 2048, 1024, 8192),
    "mixtral_kv":    ("Mixtral kv_proj M=2048", 2048, 1024, 4096),
    "dsv3_qa_ctrl":  ("DSv3 q_a_proj M=128 (control, no overflow)",
                      128, 1536, 7168),
}

# k = factor of 32; m=1 means whole-M dim per-cohort. Skip splits whose
# divisibility doesn't hold for a given shape (handled per-shape below).
K_VALUES = (1, 2, 4, 8, 16, 32)


# ---- planner override + emission toggle (mirrors diag-branch pattern)

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
def _emission(k_fast: bool):
    prev_perm = ts_config.core_id_permutation
    prev_rev = ts_config.core_emission_reverse
    ts_config.core_id_permutation = "k_fast" if k_fast else "identity"
    ts_config.core_emission_reverse = False
    try:
        yield
    finally:
        ts_config.core_id_permutation = prev_perm
        ts_config.core_emission_reverse = prev_rev


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


def _compile_and_bench(M, N, K, split, k_fast):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _emission(k_fast), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _emission(k_fast), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


# ---- helpers --------------------------------------------------------

def _candidate_splits(M, N, K):
    """For each k ∈ K_VALUES, build (m, n, k) = (1, 32/k, k) if valid."""
    out = []
    for k in K_VALUES:
        if 32 % k != 0:
            continue
        n = 32 // k
        # divisibility on shape dims
        if N % n != 0 or K % k != 0:
            continue
        N_per = N // n
        K_per = K // k
        # stick alignment on N (fp16 stick = 64 elems)
        if N_per % 64 != 0:
            continue
        out.append((1, n, k))
    # always include pure-M as control
    if (32, 1, 1) not in out and M % 32 == 0:
        out.insert(0, (32, 1, 1))
    return out


def _a_per_core_mb(M, K, split):
    m, _, k = split
    return (M // m) * (K // k) * 2 / (1024 * 1024)


def _chain_lx_mb(split):
    return split[2] * (LX_BYTES_PER_CORE / (1024 * 1024))


# ---- main -----------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", default=None,
                        choices=list(SHAPES.keys()),
                        help="run only this shape (default: all)")
    parser.add_argument("--no-id", action="store_true",
                        help="skip identity-emission rows (kf only)")
    args = parser.parse_args()

    keys = [args.shape] if args.shape else list(SHAPES.keys())

    print("# Probe 1 — chain-LX scaling under +kf\n")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, "
          f"LX={LX_BYTES_PER_CORE//1024//1024} MB/core\n")
    print("Hypothesis (M1): wall flat once chain_LX > A_per "
          "(chain-cooperative residency absorbs overflow).\n")

    for key in keys:
        label, M, N, K = SHAPES[key]
        splits = _candidate_splits(M, N, K)
        print(f"## {label}  shape=({M}, {N}, {K})\n")
        print("| split | A_per (MB) | chain_LX (MB) | overage | "
              "kf wall (ms) | id wall (ms) | id − kf |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for split in splits:
            a_per = _a_per_core_mb(M, K, split)
            chain_lx = _chain_lx_mb(split)
            overage = a_per / chain_lx
            kf_ms, kf_err = _compile_and_bench(M, N, K, split, k_fast=True)
            kf_str = f"{kf_ms:.3f}" if kf_ms is not None else f"ERR ({kf_err[:25]})"
            if args.no_id or split == (32, 1, 1):
                id_str = "—"
                delta_str = "—"
            else:
                id_ms, id_err = _compile_and_bench(M, N, K, split, k_fast=False)
                id_str = f"{id_ms:.3f}" if id_ms is not None else f"ERR ({id_err[:25]})"
                if kf_ms is not None and id_ms is not None:
                    delta_str = f"{id_ms - kf_ms:+.3f}"
                else:
                    delta_str = "—"
            print(f"| {split} | {a_per:.2f} | {chain_lx:.0f} | "
                  f"{overage:.2f}× | {kf_str} | {id_str} | {delta_str} |")
        print()

    print("## Reading guide\n")
    print("M1 (chain-coop LX) signature:")
    print("  - wall(kf) approximately FLAT across k once overage < 1.0")
    print("  - wall(kf) climbs ABOVE that line where overage > 1.0")
    print("  - the inflection point predicts k-threshold for chain-LX")
    print()
    print("M2 (kernel fast path) signature:")
    print("  - wall(kf) FLAT across all k regardless of overage")
    print("  - id − kf gap is the *only* mode-dependent term")
    print()
    print("M3 (PSUM forward-pipeline) signature:")
    print("  - wall(kf) INCREASES at high k due to growing PSUM payload")
    print("  - id − kf gap GROWS with k (more hops × larger payload)")
    print()
    print("Mixed: probe 2 (permutation discriminator) needed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
