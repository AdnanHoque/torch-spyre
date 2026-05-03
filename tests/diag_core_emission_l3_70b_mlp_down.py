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

"""Targeted core-emission probe: L3-70B MLP down prefill.

Single-shape benchmark to test the algebraic prediction that reversing
the core emission order should help the L3-70B MLP down case. This
shape has |B| = 470 MB which forces the planner to span-pre-split N to
2; remaining cores fall to M, giving the (16, 2, 1) split. Under the
default M-fast emitter, ring-adjacent cores 0..15 all share an
N-band — meaning they all need the same B-column slice of size
~235 MB. Under the reversed N-fast emitter, ring-adjacent pairs (0,1),
(2,3), ... share an M-band — they need the same tiny A-row slice of
~458 KB.

Cheaper neighbor-shared operand → less ring traffic → predicted faster.

Run: python tests/diag_core_emission_l3_70b_mlp_down.py
"""

from __future__ import annotations

import statistics
import time

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
from torch_spyre._inductor.codegen import superdsc as _superdsc


WARMUP = 3
ITERS = 20

M = 128
N = 8192
K = 28672

# These are the predicted per-neighbor-broadcast operand sizes under
# each emitter, computed from the (16, 2, 1) span-forced split.
DTYPE_BYTES = 2
B_NEIGHBOR_SLICE_BYTES = K * (N // 2) * DTYPE_BYTES   # M-fast neighbor share
A_NEIGHBOR_SLICE_BYTES = (M // 16) * K * DTYPE_BYTES  # N-fast neighbor share


# ---- planner-pick capture ----------------------------------------------

_captured: list = []
_orig_parse = _superdsc.parse_op_spec


def _hook(op_spec):
    sdsc = _orig_parse(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        _captured.append(op_spec)
    return sdsc


_superdsc.parse_op_spec = _hook  # type: ignore[assignment]


def _split_str(op_spec) -> str:
    parts = []
    for sym, (sz, nc) in op_spec.iteration_space.items():
        try:
            parts.append(f"{int(sz)}x{int(nc)}c")
        except (TypeError, ValueError):
            parts.append(f"?x{nc}c")
    return "[" + ", ".join(parts) + "]"


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


def _compile_and_bench():
    a = torch.randn(M, K, dtype=torch.float16, device="spyre")
    b = torch.randn(K, N, dtype=torch.float16, device="spyre")
    torch._dynamo.reset()
    cap_start = len(_captured)

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    mm(a, b)
    _ts.synchronize()
    cap = _captured[cap_start]

    ms = _bench(lambda: mm(a, b))
    return ms, cap


def main() -> int:
    print(f"# Core-emission probe: L3-70B MLP down ({M}x{N}x{K})")
    print(f"# warmup={WARMUP} iters={ITERS}")
    print(f"# predicted (16,2,1) neighbor-broadcast operand sizes:")
    print(f"#   M-fast (default): each neighbor pair shares "
          f"{B_NEIGHBOR_SLICE_BYTES / 1024 / 1024:.1f} MB of B")
    print(f"#   N-fast (reverse): each neighbor pair shares "
          f"{A_NEIGHBOR_SLICE_BYTES / 1024:.1f} KB of A")
    print()

    ts_config.core_emission_reverse = False
    ms_def, cap_def = _compile_and_bench()
    print(f"  default  emitter: split {_split_str(cap_def)}  "
          f"wall {ms_def:.3f} ms")

    ts_config.core_emission_reverse = True
    ms_rev, cap_rev = _compile_and_bench()
    print(f"  reversed emitter: split {_split_str(cap_rev)}  "
          f"wall {ms_rev:.3f} ms")

    print()
    speedup = ms_def / ms_rev
    delta_ms = ms_def - ms_rev
    print(f"  speedup: {speedup:.3f}x   "
          f"(delta {delta_ms:+.3f} ms)")

    if speedup >= 1.05:
        print("\n  Reversed emitter wins by >=5%. Algebraic prediction "
              "confirmed: smaller neighbor-shared operand reduces ring "
              "traffic.")
    elif speedup <= 0.95:
        print("\n  Reversed emitter LOSES by >=5%. Theory is wrong "
              "or there's another effect dominating.")
    else:
        print("\n  Within noise. Topology effect smaller than predicted, "
              "or compute / launch overhead is masking it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
