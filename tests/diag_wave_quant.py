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

"""Phase 0 of the Stream-K-on-Spyre project.

Characterizes wave-quantization losses on Spyre across production matmul
shapes. The default planner chooses an `(m_split, n_split, k_split)`
factorization within stick-alignment + divisibility constraints; for
shapes whose dimensions don't factor cleanly into 32 cores, some cores
end up idle.

For each shape, capture:

1. The split factors actually chosen by the default planner (via the
   parse_op_spec hook from the SplitK Phase 0 work).
2. The product = num_cores actually used.
3. Idle cores = 32 - num_cores_used.
4. Wall time per call.

Sample is biased toward shapes that production LLM serving stresses:
LoRA, MQA / GQA-TP=8 kv_proj, MoE per-expert, dynamic-shape prefill.

The deliverable is a catalog: which production workloads actually leave
cores idle today, and by how much. That bounds the Stream-K project's
addressable perf headroom.

Run: python tests/diag_wave_quant.py
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

import torch_spyre  # noqa: F401
from torch_spyre import streams as _ts
from torch_spyre._inductor.codegen import superdsc as _superdsc


# ---- SDSC capture ----------------------------------------------------------

@dataclass
class _Capture:
    op: str
    dims: list[tuple[str, int, int]]   # (sym, size, n_cores)
    num_cores: int


_captured: list[_Capture] = []
_orig_parse_op_spec = _superdsc.parse_op_spec


def _hook(op_spec):
    sdsc = _orig_parse_op_spec(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        dims = [
            (str(s), int(_to_int(sz)), int(nc))
            for s, (sz, nc) in op_spec.iteration_space.items()
        ]
        _captured.append(_Capture(
            op=op_spec.op, dims=dims, num_cores=int(sdsc.num_cores),
        ))
    return sdsc


def _to_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


_superdsc.parse_op_spec = _hook  # type: ignore[assignment]


# ---- bench primitive --------------------------------------------------------

WARMUP = 3
ITERS = 15


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


# ---- shapes -----------------------------------------------------------------

@dataclass
class _Shape:
    label: str
    M: int
    N: int
    K: int
    use_case: str


SHAPES = [
    # LoRA r=16 down-proj (decode + prefill batch)
    _Shape("LoRA r=16 down decode",      1, 16, 4096,    "LoRA adapter"),
    _Shape("LoRA r=16 down prefill",   128, 16, 4096,    "LoRA adapter"),
    _Shape("LoRA r=64 down prefill",   128, 64, 4096,    "LoRA adapter"),

    # MQA / GQA at TP=8 — N is small after sharding
    _Shape("L3-70B GQA TP=8 kv decode",   1, 128, 8192, "Llama-70B GQA TP=8"),
    _Shape("L3-70B GQA TP=8 kv prefill", 128, 128, 8192, "Llama-70B GQA TP=8"),
    _Shape("L3-8B GQA TP=4 kv prefill",  128, 256, 4096, "Llama-8B GQA TP=4"),

    # MoE per-expert at varying intermediate dims
    _Shape("DeepSeek-MoE inter=1408 prefill", 192, 1408, 2048, "DeepSeek-MoE per-expert"),
    _Shape("Qwen3-MoE inter=1536 prefill",    128, 1536, 2048, "Qwen3-MoE per-expert"),

    # Awkward M (prime) — dynamic-shape prefill simulation
    _Shape("Prime M=257 prefill",           257, 4096, 4096, "dynamic prefill"),
    _Shape("Prime M=521 prefill",           521, 4096, 4096, "dynamic prefill"),

    # Reference: well-aligned shapes that should saturate cleanly
    _Shape("L3-8B q_proj prefill (aligned)",  128, 4096, 4096, "reference / aligned"),
    _Shape("L3-70B q_proj prefill (aligned)", 128, 8192, 8192, "reference / aligned"),
]


# ---- main -----------------------------------------------------------------

@dataclass
class _Row:
    shape: _Shape
    splits_str: str
    num_cores_used: int
    wall_ms: float
    error: str | None = None


def main() -> int:
    rows: list[_Row] = []

    for sh in SHAPES:
        print(f"\n# {sh.label} ({sh.M}, {sh.N}, {sh.K}) — {sh.use_case}", flush=True)
        try:
            a = torch.randn(sh.M, sh.K, dtype=torch.float16, device="spyre")
            b = torch.randn(sh.K, sh.N, dtype=torch.float16, device="spyre")

            torch._dynamo.reset()

            @torch.compile(dynamic=False)
            def mm(x, y):
                return x @ y

            _captured.clear()
            mm(a, b)
            _ts.synchronize()

            cap = _captured[0] if _captured else None
            if cap is None:
                rows.append(_Row(shape=sh, splits_str="(no capture)",
                                 num_cores_used=0, wall_ms=0.0,
                                 error="no matmul kernel captured"))
                print(f"  no capture", flush=True)
                continue

            splits_str = "[" + ", ".join(
                f"{sz}×{nc}c" for _, sz, nc in cap.dims
            ) + "]"

            ms = _bench(lambda: mm(a, b))
            rows.append(_Row(
                shape=sh, splits_str=splits_str,
                num_cores_used=cap.num_cores, wall_ms=ms,
            ))
            idle = 32 - cap.num_cores
            print(f"  splits: {splits_str}  cores: {cap.num_cores}/32 "
                  f"({idle} idle)  wall: {ms:.2f} ms",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:120]}"
            rows.append(_Row(shape=sh, splits_str="(error)", num_cores_used=0,
                             wall_ms=0.0, error=err))
            print(f"  ERR: {err}", flush=True)

    _print_table(rows)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_wave_quant_results.md",
    )
    with open(out_path, "w") as f:
        _print_table(rows, file=f)
    print(f"\n# results written to {out_path}", flush=True)
    return 0


def _print_table(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# Wave-quantization diagnostic — Stream-K Phase 0")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"SENCORES:       {os.environ.get('SENCORES', '32 (default)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w("")
    w("**Method**: at each shape, default planner picks an `(m_split, "
      "n_split, k_split)` factorization. Splits are captured via "
      "`parse_op_spec` hook. Cores used = product of factors; idle = "
      "32 - cores_used. Wall time measured with per-iter sync.")
    w("")
    w("**Stream-K hypothesis**: shapes with idle cores are leaving perf on "
      "the table. A planner that does 1D linearized work assignment "
      "(Stream-K-style) could activate idle cores at the cost of cross-"
      "core partial reductions or padding overhead.")
    w("")
    w("| shape | use case | M, N, K | splits (size×cores) | cores/32 | idle | wall ms |")
    w("|---|---|---|---|---:|---:|---:|")
    for r in rows:
        sh = r.shape
        if r.error:
            w(f"| {sh.label} | {sh.use_case} | {sh.M}×{sh.N}×{sh.K} | "
              f"{r.splits_str} | — | — | err: {r.error} |")
            continue
        idle = 32 - r.num_cores_used
        w(f"| {sh.label} | {sh.use_case} | {sh.M}×{sh.N}×{sh.K} | "
          f"`{r.splits_str}` | {r.num_cores_used}/32 | {idle} | "
          f"{r.wall_ms:.2f} |")
    w("")

    # Summary: how many shapes leave cores idle?
    valid = [r for r in rows if not r.error]
    if valid:
        idle_shapes = [r for r in valid if (32 - r.num_cores_used) > 0]
        w(f"**Summary**: {len(idle_shapes)} of {len(valid)} measured shapes "
          f"leave at least one core idle under the default planner.")
        if idle_shapes:
            avg_idle = sum(32 - r.num_cores_used for r in idle_shapes) / len(idle_shapes)
            w(f"Mean idle cores across affected shapes: {avg_idle:.1f}/32 "
              f"({100*avg_idle/32:.0f}% of capacity).")


if __name__ == "__main__":
    raise SystemExit(main())
