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

"""Default vs output_element_priority planner comparison.

For each Phase 1.0 production shape, compiles and benches the matmul
twice — once with the default planner, once with the
`output_element_priority` heuristic on. Captures the planner's pick
both times and reports wall-time speedup.

Output: a single comparison table suitable for a PR description or
presentation. Apples-to-apples: both passes use identical warmup/iter
counts in the same run, eliminating day-to-day card variance.

Run: python tests/diag_element_priority_compare.py
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

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
ITERS = 15


# ---- planner-pick capture (same hook as diag_split_gap) -----------------

_captured: list[list[tuple[str, int, int]]] = []
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
        _captured.append(dims)
    return sdsc


_superdsc.parse_op_spec = _hook_parse  # type: ignore[assignment]


# ---- bench primitive ----------------------------------------------------

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


def _compile_and_bench(M: int, N: int, K: int) -> tuple[float, list[tuple[str, int, int]]]:
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
    pick = captures[0] if captures else []

    ms = _bench(lambda: mm(a, b))
    return ms, pick


# ---- shapes (same as Phase 1.0) -----------------------------------------

@dataclass
class _Shape:
    label: str
    M: int
    N: int
    K: int


SHAPES: list[_Shape] = [
    _Shape("L3-8B q_proj prefill",       128, 4096, 4096),
    _Shape("L3-8B GQA kv_proj prefill",  128, 1024, 4096),
    _Shape("L3-8B MLP gate/up prefill",  128, 14336, 4096),
    _Shape("L3-8B MLP down prefill",     128, 4096, 14336),
    _Shape("L3-70B q_proj prefill",      128, 8192, 8192),
    _Shape("L3-70B GQA kv_proj prefill", 128, 1024, 8192),
    _Shape("L3-70B GQA TP=8 kv prefill", 128, 128, 8192),
    _Shape("L3-70B MLP down prefill",    128, 8192, 28672),
    _Shape("Mixtral down per-expert",    128, 4096, 14336),
    _Shape("Qwen3-MoE gate per-expert",  128, 1536, 2048),
    _Shape("DeepSeek-MoE gate (M=192)",  192, 1408, 2048),
    _Shape("L3-8B q_proj decode",        1, 4096, 4096),
    _Shape("L3-70B GQA TP=8 kv decode",  1, 128, 8192),
]


def _split_str(pick: list[tuple[str, int, int]]) -> str:
    """Collapse iteration-space dims to a (cores_per_dim) tuple in the
    order the iteration space was emitted (typically M, N, K)."""
    if not pick:
        return "(no capture)"
    return "(" + ", ".join(str(nc) for _, _, nc in pick) + ")"


@dataclass
class _ShapeRow:
    shape: _Shape
    default_pick: str
    default_ms: float
    heuristic_pick: str
    heuristic_ms: float
    error: str = ""


def _measure_shape(sh: _Shape) -> _ShapeRow:
    print(f"\n# {sh.label} ({sh.M}x{sh.N}x{sh.K})", flush=True)
    try:
        ts_config.output_element_priority = False
        ms_def, pick_def = _compile_and_bench(sh.M, sh.N, sh.K)
        print(f"  default:   {_split_str(pick_def):>14}  {ms_def:.2f} ms",
              flush=True)
    except Exception as e:  # noqa: BLE001
        return _ShapeRow(sh, "err", 0.0, "err", 0.0, f"default err: {e!s:.80}")

    try:
        ts_config.output_element_priority = True
        ms_heu, pick_heu = _compile_and_bench(sh.M, sh.N, sh.K)
        print(f"  heuristic: {_split_str(pick_heu):>14}  {ms_heu:.2f} ms",
              flush=True)
    except Exception as e:  # noqa: BLE001
        return _ShapeRow(
            sh, _split_str(pick_def), ms_def, "err", 0.0,
            f"heuristic err: {e!s:.80}",
        )

    return _ShapeRow(
        sh, _split_str(pick_def), ms_def, _split_str(pick_heu), ms_heu,
    )


def _emit_table(rows: list[_ShapeRow], file=None) -> None:
    def w(s):
        print(s, file=file)

    w("# output_element_priority comparison\n")
    w(f"PyTorch {torch.__version__}, "
      f"SENCORES={os.environ.get('SENCORES', '32 (default)')}, "
      f"warmup={WARMUP}, iters={ITERS}\n")
    w("| shape | default split | default ms | heuristic split | "
      "heuristic ms | speedup |")
    w("|---|---|---:|---|---:|---:|")
    speedups: list[float] = []
    for r in rows:
        if r.error:
            w(f"| {r.shape.label} | {r.default_pick} | {r.default_ms:.2f} | "
              f"{r.heuristic_pick} | err | err |")
            continue
        speedup = r.default_ms / r.heuristic_ms
        speedups.append(speedup)
        flag = ""
        if speedup >= 1.05:
            flag = " ✓"
        elif speedup <= 0.95:
            flag = " ✗"
        w(f"| {r.shape.label} | `{r.default_pick}` | {r.default_ms:.2f} | "
          f"`{r.heuristic_pick}` | {r.heuristic_ms:.2f} | "
          f"{speedup:.2f}x{flag} |")
    w("")
    if speedups:
        w(f"**Geometric mean speedup**: "
          f"{statistics.geometric_mean(speedups):.3f}x")
        w(f"**Best**: {max(speedups):.2f}x   **Worst**: {min(speedups):.2f}x")
        wins = sum(1 for s in speedups if s >= 1.05)
        regr = sum(1 for s in speedups if s <= 0.95)
        w(f"**>= 5% faster**: {wins}/{len(speedups)} shapes   "
          f"**>= 5% regression**: {regr}/{len(speedups)}")


def main() -> int:
    rows: list[_ShapeRow] = []
    for sh in SHAPES:
        rows.append(_measure_shape(sh))

    print()
    _emit_table(rows)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_element_priority_compare_results.md",
    )
    with open(out_path, "w") as f:
        _emit_table(rows, file=f)
    print(f"\nresults written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
