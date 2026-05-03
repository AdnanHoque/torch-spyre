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

"""Core-emission reorder probe — K-split PSUM-chain regime + clean M↔N retest.

The earlier `diag_core_emission_lx_fit.py` and the production-shape
sweep both exhausted (m, n, 1) splits and concluded the lever was dead
because kernel-template overlap hides ring sharing. Both probes
deliberately set k=1, so the dedicated SFP psum ring was never
exercised.

This probe revisits the question with two lenses, neither of which the
earlier work covered:

Part A — K-split PSUM-chain reorder
-----------------------------------
For mixed (m, 1, k) splits where k > 1, partial-sum reduction across
the k psum-collaborating cores travels the SFP ring (32 B/cycle,
dedicated). Default emission walks M first → for (2, 1, 16) the K=0
band gets cores {0, 2, 4, ..., 30} (every other physical position,
30-hop chain). Reverse emission walks K first → K=0 band gets
cores {0, 1, ..., 15} (contiguous, 15-hop chain). PSUM is on the
critical path after compute, so the kernel-template overlap argument
that defeats input-fetch reordering does NOT apply here.

Part B — Pure M↔N output reorder, focused
-----------------------------------------
A clean re-test of (m, n, 1) splits on shapes where ring-share could
plausibly matter. The earlier sweep used many shapes; this part picks
two and runs both modes back-to-back to give the cleanest possible
read on whether M-vs-N as the fast core-ID dim ever matters at all.

Run: python tests/diag_core_emission_psum_chain.py
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

# Allow direct invocation from repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402
from torch_spyre._inductor import config as ts_config  # noqa: E402
from torch_spyre._inductor import core_division as _core_div  # noqa: E402
from torch_spyre._inductor.codegen import superdsc as _superdsc  # noqa: E402


WARMUP = 3
ITERS = 15
DTYPE = torch.float16
DTYPE_BYTES = 2

# (label, M, N, K, [(forced_split, note), ...])
PROBES_A_KSPLIT = [
    (
        "L3-8B MLP down prefill (K-heavy)",
        128, 4096, 14336,
        [
            ((1, 1, 32), "pure-K (reverse is no-op, sanity)"),
            ((2, 1, 16), "mixed K — chain 30→15 hops predicted (~2x)"),
            ((4, 1, 8),  "mixed K — chain 28→7 hops predicted (~4x)"),
            ((8, 1, 4),  "mixed K — chain 24→3 hops predicted (~8x)"),
        ],
    ),
    (
        "L3-70B q_proj prefill (square)",
        128, 8192, 8192,
        [
            ((2, 1, 16), "mixed K — chain 30→15 hops predicted (~2x)"),
            ((4, 1, 8),  "mixed K — chain 28→7 hops predicted (~4x)"),
        ],
    ),
    (
        "Synthetic K-extreme (small N, big K)",
        128, 512, 32768,
        [
            ((1, 1, 32), "pure-K (reverse is no-op)"),
            ((2, 1, 16), "mixed K"),
            ((4, 1, 8),  "mixed K"),
        ],
    ),
]

PROBES_B_OUTPUT = [
    (
        "L3-8B q_proj prefill",
        128, 4096, 4096,
        [
            ((4, 8, 1),  "moderate mixed"),
            ((16, 2, 1), "extreme M-fast"),
            ((2, 16, 1), "extreme N-fast"),
        ],
    ),
    (
        "L3-70B MLP down prefill",
        128, 8192, 28672,
        [
            ((16, 2, 1), "extreme M-fast (Phase 1.0 best)"),
            ((2, 16, 1), "extreme N-fast"),
        ],
    ),
]


# ---- planner-pick capture ----------------------------------------------

_captured: list = []
_orig_parse = _superdsc.parse_op_spec


def _hook(op_spec):
    sdsc = _orig_parse(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        _captured.append(op_spec)
    return sdsc


_superdsc.parse_op_spec = _hook  # type: ignore[assignment]


# ---- force-split machinery ---------------------------------------------

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


# ---- chain-distance prediction ----------------------------------------

def _chain_distance(target: tuple[int, int, int], reverse: bool) -> tuple[int, int]:
    """Return (k_chain_hops, n_chain_hops) for a (m, n, k) split.

    Each chain's hop count assumes physical core IDs sit in sequential
    ring order. The chain spans the ring positions of the cores that
    share a constant value of the OTHER output coordinates. A 32-core
    ring has worst-case 31 hops.

    For default (M-fast) emission of (m, n, k):
      core_id = m_slice + m * (n_slice + n * k_slice)
      → varying k holds (m, n) fixed; chain spans (k-1)*m*n positions
      → varying n holds (m, k) fixed; chain spans (n-1)*m positions

    For reverse (K-fast) emission of (m, n, k):
      core_id = k_slice + k * (n_slice + n * m_slice)
      → varying k holds (n, m) fixed; chain spans (k-1) positions
      → varying n holds (m, k) fixed; chain spans (n-1)*k positions
    """
    m, n, k = target
    if not reverse:
        k_chain = (k - 1) * m * n if k > 1 else 0
        n_chain = (n - 1) * m if n > 1 else 0
    else:
        k_chain = (k - 1) if k > 1 else 0
        n_chain = (n - 1) * k if n > 1 else 0
    return k_chain, n_chain


# ---- bench primitive --------------------------------------------------

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


def _compile_and_bench(M: int, N: int, K: int, target):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(target):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _force_split(target):
                mm(a, b)
        ms = _bench(step)
        return ms, ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:80]}"


# ---- run a probe-section ----------------------------------------------

def _run_section(title: str, probes, predict_psum: bool):
    print(f"## {title}\n")
    rows = []
    for label, M, N, K, configs in probes:
        print(f"### {label}  (M={M}, N={N}, K={K})\n")
        for target, note in configs:
            kd_def, nd_def = _chain_distance(target, reverse=False)
            kd_rev, nd_rev = _chain_distance(target, reverse=True)
            print(f"#### split={target}  ({note})")
            print(
                f"  Predicted ring-chain hops: "
                f"K default={kd_def} reverse={kd_rev}  | "
                f"N default={nd_def} reverse={nd_rev}"
            )

            ts_config.core_emission_reverse = False
            ms_def, err_def = _compile_and_bench(M, N, K, target)
            if err_def:
                print(f"  default:  ERR {err_def}")
            else:
                print(f"  default:  {ms_def:.3f} ms")

            ts_config.core_emission_reverse = True
            ms_rev, err_rev = _compile_and_bench(M, N, K, target)
            if err_rev:
                print(f"  reverse:  ERR {err_rev}")
            else:
                print(f"  reverse:  {ms_rev:.3f} ms")

            if ms_def is not None and ms_rev is not None:
                delta = ms_def - ms_rev
                speed = ms_def / ms_rev
                print(
                    f"  delta:    {delta:+.3f} ms  (speedup {speed:.3f}x)"
                )
                rows.append((label, target, ms_def, ms_rev, kd_def, kd_rev,
                             nd_def, nd_rev))
            print()
    return rows


def main() -> int:
    print("# Core-emission reorder probe — K-split PSUM + M↔N output\n")
    print(f"# warmup={WARMUP} iters={ITERS}, fp16, SENCORES=32\n")

    print("## Hypothesis recap\n")
    print(
        "- For (m, 1, k>1): default M-fast scatters K-chain across the\n"
        "  ring (cores {0, m, 2m, ...}); reverse K-fast packs K-chain\n"
        "  contiguously (cores {0..k-1}). Predicted hop reduction\n"
        "  scales with `m*n - 1` per chain step → reverse should win\n"
        "  iff PSUM is on the critical path.\n"
        "- For (m, n, 1): no PSUM chain. Default vs reverse swaps which\n"
        "  output dim is the fast-changing core-ID dim. Earlier probes\n"
        "  showed null result; we re-confirm on focused shapes.\n"
    )

    rows_a = _run_section(
        "Part A — K-split PSUM-chain reorder", PROBES_A_KSPLIT,
        predict_psum=True,
    )

    rows_b = _run_section(
        "Part B — Pure M↔N output reorder (k=1, focused)", PROBES_B_OUTPUT,
        predict_psum=False,
    )

    # --- summary table ---
    print("## Summary — all valid (def, rev) measurements\n")
    print("| section | shape | split | def ms | rev ms | delta | speedup | "
          "K-chain (def→rev) |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    for label, target, d, r, kd_def, kd_rev, _nd, _nd2 in rows_a:
        delta = d - r
        speed = d / r
        flag = " ✓" if speed >= 1.05 else ""
        print(
            f"| A | {label} | {target} | {d:.3f} | {r:.3f} | "
            f"{delta:+.3f} | {speed:.3f}x{flag} | {kd_def}→{kd_rev} |"
        )
    for label, target, d, r, kd_def, kd_rev, _nd, _nd2 in rows_b:
        delta = d - r
        speed = d / r
        flag = " ✓" if speed >= 1.05 else ""
        print(
            f"| B | {label} | {target} | {d:.3f} | {r:.3f} | "
            f"{delta:+.3f} | {speed:.3f}x{flag} | (k=1, no PSUM) |"
        )
    print()

    # --- verdict ---
    print("## Verdict\n")
    if not rows_a and not rows_b:
        print("  No valid measurements.")
        return 1
    a_speedups = [d / r for (_, _, d, r, *_) in rows_a]
    b_speedups = [d / r for (_, _, d, r, *_) in rows_b]
    if a_speedups:
        print(
            f"  Part A (K-split): max speedup {max(a_speedups):.3f}x, "
            f"median {statistics.median(a_speedups):.3f}x"
        )
    if b_speedups:
        print(
            f"  Part B (output reorder): max speedup "
            f"{max(b_speedups):.3f}x, median "
            f"{statistics.median(b_speedups):.3f}x"
        )
    if max(a_speedups + b_speedups, default=0.0) >= 1.05:
        print(
            "\n  Reorder shows ≥5% movement somewhere. The ring lever "
            "is alive in at least one regime — drill into which row "
            "moved and design a planner heuristic."
        )
    elif max(a_speedups + b_speedups, default=0.0) >= 1.02:
        print("\n  Marginal (2-5%) movement. Worth one more replication.")
    else:
        print(
            "\n  Reorder is flat (<2%) even on K-split shapes where PSUM "
            "is on the critical path. Combined with the earlier (m, n, 1) "
            "null, this closes core-ID reordering as a practical lever — "
            "the runtime is hiding ring topology effects in BOTH the data-"
            "ring (input fetch overlap) and SFP-ring (PSUM serialization "
            "latency) regimes. Time to shift to root-cause writeup."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
