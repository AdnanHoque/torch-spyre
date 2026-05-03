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

"""End-to-end smoke test for output_element_priority heuristic.

Compiles L3-8B q_proj (M=128, N=4096, K=4096) twice — once with the
heuristic off and once with it on. Captures the planner's actual split
both times via parse_op_spec hook. Asserts the heuristic flips the
default pick from M-priority (m=32, n=1, k=1) to N-priority
(m=1, n=32, k=1).

Pure planner check — does NOT bench wall time. The full Phase 1.0
sweep with heuristic on is a separate script.

Run: python tests/diag_element_priority_smoke.py
"""

from __future__ import annotations

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


_captured: list[tuple[str, list[tuple[str, int, int]]]] = []
_orig_parse_op_spec = _superdsc.parse_op_spec


def _to_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def _hook_parse(op_spec):
    sdsc = _orig_parse_op_spec(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        dims = [
            (str(s), int(_to_int(sz)), int(nc))
            for s, (sz, nc) in op_spec.iteration_space.items()
        ]
        _captured.append((op_spec.op, dims))
    return sdsc


_superdsc.parse_op_spec = _hook_parse  # type: ignore[assignment]


def _compile_and_capture(M: int, N: int, K: int) -> list[tuple[str, int, int]]:
    a = torch.randn(M, K, dtype=torch.float16, device="spyre")
    b = torch.randn(K, N, dtype=torch.float16, device="spyre")
    torch._dynamo.reset()
    cap_start = len(_captured)

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    mm(a, b)
    _ts.synchronize()
    captures = _captured[cap_start:]
    assert captures, "no matmul captured"
    return captures[0][1]


def main() -> int:
    M, N, K = 128, 4096, 4096

    print(f"# element-priority smoke test on ({M}, {N}, {K})\n")

    ts_config.output_element_priority = False
    default_dims = _compile_and_capture(M, N, K)
    print(f"default planner: {default_dims}")

    ts_config.output_element_priority = True
    heuristic_dims = _compile_and_capture(M, N, K)
    print(f"heuristic on:    {heuristic_dims}")

    # iteration_space order is (M, N, K) by index. Take cores per position
    # rather than per size — N and K are both 4096 so a size-keyed dict
    # would collapse them.
    def _cores_at(dims, idx):
        return dims[idx][2]

    print()
    print(f"default cores per dim:   M={_cores_at(default_dims, 0)}, "
          f"N={_cores_at(default_dims, 1)}, K={_cores_at(default_dims, 2)}")
    print(f"heuristic cores per dim: M={_cores_at(heuristic_dims, 0)}, "
          f"N={_cores_at(heuristic_dims, 1)}, K={_cores_at(heuristic_dims, 2)}")

    default_pure_m = (
        _cores_at(default_dims, 0) == 32
        and _cores_at(default_dims, 1) == 1
        and _cores_at(default_dims, 2) == 1
    )
    heuristic_pure_n = (
        _cores_at(heuristic_dims, 0) == 1
        and _cores_at(heuristic_dims, 1) == 32
        and _cores_at(heuristic_dims, 2) == 1
    )
    if default_pure_m and heuristic_pure_n:
        print("\nPASS: default picks (32,1,1); heuristic flips to (1,32,1).")
        return 0
    else:
        print("\nFAIL: did not see the expected flip.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
