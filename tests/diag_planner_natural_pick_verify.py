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

"""Verify which split the planner naturally picks for our k_fast targets.

Every k_fast wall-time measurement so far used ``_force_split`` to pin a
specific ``(m, n, k)``. That tells us "k_fast helps when this split is
chosen," but NOT "the planner would have chosen this split in
production." This probe closes that gap.

For each shape we measured, compile a torch.compile matmul WITHOUT
forcing a split, hook ``parse_op_spec`` to capture the planner's
choice, and compare to the split we forced.

Run twice:
  - OUTPUT_ELEMENT_PRIORITY=0 (default in production today)
  - OUTPUT_ELEMENT_PRIORITY=1 (the shipped opt-in)

If the natural pick matches the forced split → the wall-time win
composes with production planner. If not → the win is contingent on
shipping a separate planner change to make this split get picked.
"""

from __future__ import annotations

import os
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
from torch_spyre._inductor import config as ts_config  # noqa: E402
from torch_spyre._inductor.codegen import superdsc as _superdsc  # noqa: E402


DTYPE = torch.float16

# (label, M, N, K, forced_split_we_used)
TARGETS = [
    # SHOULD survive verification (planner forced into k>1 by stick math)
    ("L3-70B kv_proj M=2048",         2048,  1024,  8192, (1, 16, 2)),
    ("Mixtral 8x7B kv_proj M=2048",   2048,  1024,  4096, (1, 16, 2)),

    # NEEDS verification (planner has a real choice)
    ("DSv3 o_proj M=2048",            2048,  7168, 16384, (1, 16, 2)),
    ("DSv3 down_proj M=2048 (dense)", 2048,  7168,  2048, (1, 16, 2)),
    ("DSv3 q_a_proj M=2048",          2048,  1536,  7168, (1, 8, 4)),

    # Likely DOESN'T survive (planner naturally picks pure-N)
    ("L3-70B q_proj M=128",            128,  8192,  8192, (4, 1, 8)),
    ("L3-8B  MLP down M=128",          128,  4096, 14336, (4, 1, 8)),
]


# ---- planner-pick capture via parse_op_spec hook -----------------------

_captured: list = []
_orig_parse = _superdsc.parse_op_spec


def _hook(op_spec):
    sdsc = _orig_parse(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        _captured.append(op_spec)
    return sdsc


_superdsc.parse_op_spec = _hook  # type: ignore[assignment]


def _extract_split(op_spec) -> tuple:
    """Pull the (m, n, k) split out of the captured op_spec."""
    parts = []
    for sym, value in op_spec.iteration_space.items():
        # iteration_space[dim] = (size, num_cores) where num_cores is the split
        try:
            size, ncores = value
            parts.append(int(ncores))
        except (TypeError, ValueError):
            parts.append(-1)
    return tuple(parts)


def _split_str(op_spec) -> str:
    pieces = []
    for sym, value in op_spec.iteration_space.items():
        try:
            size, ncores = value
            pieces.append(f"{int(size)}x{int(ncores)}c")
        except (TypeError, ValueError):
            pieces.append(f"?x{value}c")
    return "[" + ", ".join(pieces) + "]"


# ---- compile-and-capture ----------------------------------------------

def _compile_and_capture(M, N, K) -> tuple:
    """Compile a matmul; return (split_tuple, full_iter_space_str)."""
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    cap_start = len(_captured)

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        mm(a, b)
        _ts.synchronize()
    except Exception as e:  # noqa: BLE001
        return None, f"COMPILE_ERR: {type(e).__name__}: {str(e)[:80]}"

    if len(_captured) <= cap_start:
        return None, "no matmul OpSpec captured"

    op_spec = _captured[cap_start]
    return _extract_split(op_spec), _split_str(op_spec)


# ---- run for one priority mode ----------------------------------------

def _run_mode(label_mode: str, element_priority: bool):
    print(f"\n## Mode: {label_mode}  (output_element_priority={element_priority})\n")
    ts_config.output_element_priority = element_priority
    print(f"| shape | M | N | K | forced split | natural pick | match? |")
    print(f"|---|---:|---:|---:|---|---|---|")
    rows = []
    for label, M, N, K, forced in TARGETS:
        natural, info = _compile_and_capture(M, N, K)
        if natural is None:
            match_marker = "—"
            row = (label, M, N, K, forced, None, info, match_marker)
            print(
                f"| {label} | {M} | {N} | {K} | {forced} | "
                f"ERR: {info} | {match_marker} |"
            )
        else:
            match = "✓ MATCH" if tuple(natural) == tuple(forced) else "✗ DIFFERS"
            # Compute whether k > 1 either way
            forced_has_k = forced[-1] > 1 if len(forced) >= 3 else False
            natural_has_k = natural[-1] > 1 if len(natural) >= 3 else False
            kfast_relevant = (
                "(k_fast applies)" if natural_has_k else
                "(k=1, k_fast=identity)"
            )
            row = (label, M, N, K, forced, natural, info, match)
            print(
                f"| {label} | {M} | {N} | {K} | {forced} | "
                f"{tuple(natural)} {kfast_relevant} | {match} |"
            )
        rows.append(row)
    return rows


def main() -> int:
    print("# Planner natural-pick verification — does k_fast compose with production?\n")
    print("Each row compares (a) the split we FORCED in earlier wall-time")
    print("measurements vs (b) the split the planner picks NATURALLY for")
    print("the same shape. Match → measured win composes with production.")
    print("Differs → measured win is contingent on a separate planner change.\n")

    # Reset between modes (different env / config)
    rows_off = _run_mode("OUTPUT_ELEMENT_PRIORITY = 0 (production default)", False)
    rows_on  = _run_mode("OUTPUT_ELEMENT_PRIORITY = 1 (shipped opt-in)", True)

    print("\n## Verdict per shape\n")
    for (label, M, N, K, forced, _n_off, _info_off, m_off), \
        (_l, _M, _N, _K, _f, n_on, _info_on, m_on) in zip(rows_off, rows_on):
        natural_default = _n_off
        natural_priority = n_on
        if natural_default is None and natural_priority is None:
            print(f"  {label}: ERR in both modes")
            continue

        survives_default = (
            tuple(natural_default) == tuple(forced)
            if natural_default else None
        )
        survives_priority = (
            tuple(natural_priority) == tuple(forced)
            if natural_priority else None
        )

        # k_fast helps if natural pick has k > 1
        natural_default_k = (
            natural_default[-1] if natural_default and len(natural_default) >= 3 else None
        )
        natural_priority_k = (
            natural_priority[-1] if natural_priority and len(natural_priority) >= 3 else None
        )

        print(f"\n  {label}  (forced was {forced}):")
        print(f"    OUTPUT_ELEMENT_PRIORITY=0 (production today):  "
              f"natural={tuple(natural_default) if natural_default else 'ERR'}  "
              f"matches_forced={survives_default}  "
              f"k_fast_helps={(natural_default_k or 1) > 1}")
        print(f"    OUTPUT_ELEMENT_PRIORITY=1:  "
              f"natural={tuple(natural_priority) if natural_priority else 'ERR'}  "
              f"matches_forced={survives_priority}  "
              f"k_fast_helps={(natural_priority_k or 1) > 1}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
