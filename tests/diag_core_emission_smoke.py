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

"""End-to-end smoke test for core_emission_reverse.

Compiles a matmul forced into a (2, 16, 1) mixed split twice — once with
the default core emitter and once with config.core_emission_reverse=True
— and reads back the actual `core_id_to_work_slice` map produced by
`parse_op_spec`. Materializes the map (evaluates each dim's expression
at every core_id) and asserts:

  - default ordering: M is fast — adjacent cores walk M, share N
  - reversed ordering: N is fast — adjacent cores walk N, share M

Pure planner / codegen check; does NOT bench wall time. The ordering
sweep is a separate script.

Run: python tests/diag_core_emission_smoke.py
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre
torch_spyre._autoload()
from torch_spyre import streams as _ts
from torch_spyre._inductor import config as ts_config
from torch_spyre._inductor import core_division as _core_div
from torch_spyre._inductor.codegen import superdsc as _superdsc


# ---- planner force-split (same mechanism as diag_split_gap) ------------

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


# ---- SDSCSpec capture --------------------------------------------------

_captured: list = []
_orig_parse_op_spec = _superdsc.parse_op_spec


def _hook(op_spec):
    sdsc = _orig_parse_op_spec(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        _captured.append(sdsc)
    return sdsc


_superdsc.parse_op_spec = _hook  # type: ignore[assignment]


# ---- harness -----------------------------------------------------------

def _compile_and_capture(M, N, K, target):
    a = torch.randn(M, K, dtype=torch.float16, device="spyre")
    b = torch.randn(K, N, dtype=torch.float16, device="spyre")
    torch._dynamo.reset()
    cap_start = len(_captured)

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    with _force_split(target):
        mm(a, b)
    _ts.synchronize()

    sdsc = _captured[cap_start]
    return sdsc


def _materialize(sdsc):
    """Evaluate `core_id_to_work_slice` at every core_id and return a list
    of dicts (one per core), each mapping dim name → slice index."""
    from sympy import Symbol
    core_id = Symbol("core_id")

    expr_map = sdsc.core_id_to_work_slice
    n_cores = sdsc.num_cores
    out = []
    for c in range(n_cores):
        coords = {
            dim_name: int(expr.subs(core_id, c))
            for dim_name, expr in expr_map.items()
        }
        out.append(coords)
    return out


def _format_table(materialized, dim_order):
    rows = ["  core | " + " | ".join(d for d in dim_order)]
    for c, coords in enumerate(materialized):
        rows.append(
            f"  {c:>4} | "
            + " | ".join(f"{coords[d]:>1}" for d in dim_order)
        )
    return "\n".join(rows)


def main() -> int:
    # Shape that supports (2, 16, 1) split exactly (M%2==0, N/16=64 elem
    # = stick aligned, K/1 stick aligned).
    M, N, K = 128, 1024, 4096
    TARGET = (2, 16, 1)

    print(f"# core_emission smoke test on ({M}, {N}, {K}) "
          f"forced split {TARGET}\n")

    # --- default emitter ---
    ts_config.core_emission_reverse = False
    sdsc_def = _compile_and_capture(M, N, K, TARGET)
    mat_def = _materialize(sdsc_def)
    dim_order = list(sdsc_def.core_id_to_work_slice.keys())
    print("## default emitter (M-fast)\n")
    print(_format_table(mat_def[:8], dim_order))
    print("  ... (showing first 8 of 32 cores)\n")

    # --- reversed emitter ---
    ts_config.core_emission_reverse = True
    sdsc_rev = _compile_and_capture(M, N, K, TARGET)
    mat_rev = _materialize(sdsc_rev)
    print("## reversed emitter (N-fast)\n")
    print(_format_table(mat_rev[:8], dim_order))
    print("  ... (showing first 8 of 32 cores)\n")

    # --- check default = M-fast pattern ---
    # core 0 = (M=0, N=0); core 1 = (M=1, N=0). Each pair of adjacent
    # cores shares N, walks M.
    ok_default = True
    for c in range(32):
        expected_m = c % 2
        expected_n = c // 2
        if mat_def[c][dim_order[0]] != expected_m:
            print(f"  default core {c}: expected M={expected_m}, got "
                  f"{mat_def[c][dim_order[0]]}")
            ok_default = False
        if mat_def[c][dim_order[1]] != expected_n:
            print(f"  default core {c}: expected N={expected_n}, got "
                  f"{mat_def[c][dim_order[1]]}")
            ok_default = False

    # --- check reversed = N-fast pattern ---
    # K=1 so K dim is no-op. With (m=2, n=16, k=1) reversed iteration
    # walks K (skip), then N (split=16, becomes fast), then M.
    # core 0 = (M=0, N=0); core 1 = (M=0, N=1); core 16 = (M=1, N=0).
    ok_reverse = True
    for c in range(32):
        expected_n = c % 16
        expected_m = c // 16
        if mat_rev[c][dim_order[0]] != expected_m:
            print(f"  reversed core {c}: expected M={expected_m}, got "
                  f"{mat_rev[c][dim_order[0]]}")
            ok_reverse = False
        if mat_rev[c][dim_order[1]] != expected_n:
            print(f"  reversed core {c}: expected N={expected_n}, got "
                  f"{mat_rev[c][dim_order[1]]}")
            ok_reverse = False

    # --- verdict ---
    print()
    if ok_default and ok_reverse:
        print("PASS: default emits M-fast, reversed emits N-fast — "
              "config flag flips the mapping end-to-end through "
              "parse_op_spec.")
        return 0
    else:
        print("FAIL: emitted mapping does not match the expected pattern.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
