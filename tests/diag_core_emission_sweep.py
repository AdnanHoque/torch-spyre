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

"""Core-emission-order sweep across production + forced mixed splits.

Two parts:

  Phase A — natural picks. For each Phase 1.0 production shape, runs
  with output_element_priority=True (so the planner naturally lands on
  mixed splits like (2, 16, 1) where the topology flag actually
  matters), then benches default emitter vs reversed emitter. Captures
  the planner pick to confirm both modes pick the same split.

  Phase B — forced mixed splits. For three hot shapes, force every valid
  (m, n, 1) split with m>1 AND n>1 so the topology flag is non-trivial,
  bench default vs reversed for each. This isolates the topology effect
  at varying m/n ratios without confounding from priority logic.

Output: comparison table with speedup column for each (shape, split)
data point.

Run: python tests/diag_core_emission_sweep.py
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager
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
from torch_spyre._inductor import core_division as _core_div
from torch_spyre._inductor.codegen import superdsc as _superdsc


WARMUP = 3
ITERS = 15
NUM_CORES = 32
STICK_ELEMS = 64


# ---- planner-pick capture --------------------------------------------

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


def _split_tuple(op_spec) -> tuple[int, ...]:
    return tuple(int(nc) for _, (_, nc) in op_spec.iteration_space.items())


# ---- force-split machinery ------------------------------------------

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        prod = 1
        for f in target:
            prod *= f
        if prod != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target):
    if target is None:
        yield
        return
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi


# ---- bench primitive ------------------------------------------------

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


def _compile_and_bench(M: int, N: int, K: int, target: tuple | None):
    a = torch.randn(M, K, dtype=torch.float16, device="spyre")
    b = torch.randn(K, N, dtype=torch.float16, device="spyre")
    torch._dynamo.reset()
    cap_start = len(_captured)

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _force_split(target):
            mm(a, b)
        _ts.synchronize()
        cap = _captured[cap_start]

        if target is None:
            ms = _bench(lambda: mm(a, b))
        else:
            def step():
                with _force_split(target):
                    mm(a, b)
            ms = _bench(step)
        return ms, cap, ""
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {str(e)[:60]}"


# ---- shape catalog -------------------------------------------------

@dataclass
class _Shape:
    label: str
    M: int
    N: int
    K: int


PHASE_1_0_SHAPES: list[_Shape] = [
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

# Hot shapes for the forced-mixed-split sweep — chosen to escape launch
# floor (large total work) and to admit several valid mixed (m, n, 1)
# factorizations of 32.
HOT_SHAPES: list[_Shape] = [
    _Shape("L3-8B q_proj prefill",   128, 4096, 4096),
    _Shape("L3-70B q_proj prefill",  128, 8192, 8192),
    _Shape("L3-70B MLP down prefill", 128, 8192, 28672),
]


def _is_valid_mn1(M: int, N: int, K: int, m: int, n: int) -> bool:
    if M % m != 0:
        return False
    n_per = N // n
    if n_per < STICK_ELEMS or n_per % STICK_ELEMS != 0:
        return False
    if K < STICK_ELEMS or K % STICK_ELEMS != 0:
        return False
    return True


def _mixed_splits(M: int, N: int, K: int) -> list[tuple[int, int, int]]:
    out = []
    for m in range(2, NUM_CORES // 2 + 1):
        if NUM_CORES % m != 0:
            continue
        n = NUM_CORES // m
        if n < 2:
            continue
        if not _is_valid_mn1(M, N, K, m, n):
            continue
        out.append((m, n, 1))
    return out


# ---- core sweep --------------------------------------------------------

@dataclass
class _Result:
    label: str
    M: int
    N: int
    K: int
    target: tuple | None
    default_split: str
    default_ms: float | None
    reversed_split: str
    reversed_ms: float | None
    error: str = ""


def _measure(label: str, M: int, N: int, K: int,
             target: tuple | None,
             use_element_priority: bool) -> _Result:
    ts_config.output_element_priority = use_element_priority

    ts_config.core_emission_reverse = False
    ms_def, cap_def, err_def = _compile_and_bench(M, N, K, target)
    if err_def:
        return _Result(label, M, N, K, target, "err", None, "—", None,
                       error=f"default: {err_def}")

    ts_config.core_emission_reverse = True
    ms_rev, cap_rev, err_rev = _compile_and_bench(M, N, K, target)
    if err_rev:
        return _Result(label, M, N, K, target,
                       _split_str(cap_def), ms_def, "err", None,
                       error=f"reversed: {err_rev}")

    return _Result(
        label, M, N, K, target,
        _split_str(cap_def), ms_def,
        _split_str(cap_rev), ms_rev,
    )


# ---- emit ---------------------------------------------------------

def _emit_table(rows: list[_Result], title: str, file=None) -> None:
    def w(s: str): print(s, file=file)

    w(f"\n## {title}\n")
    w("| shape | forced split | default split | default ms | "
      "reversed split | reversed ms | speedup |")
    w("|---|---|---|---:|---|---:|---:|")
    speedups = []
    for r in rows:
        force_str = "—" if r.target is None else f"{r.target}"
        if r.error:
            w(f"| {r.label} | {force_str} | err | err | err | err | "
              f"err ({r.error[:30]}) |")
            continue
        if r.default_ms is None or r.reversed_ms is None:
            w(f"| {r.label} | {force_str} | {r.default_split} | "
              f"{r.default_ms or 'err'} | {r.reversed_split} | "
              f"{r.reversed_ms or 'err'} | — |")
            continue
        speedup = r.default_ms / r.reversed_ms
        speedups.append(speedup)
        flag = ""
        if speedup >= 1.05:
            flag = " ✓"
        elif speedup <= 0.95:
            flag = " ✗"
        w(f"| {r.label} | {force_str} | `{r.default_split}` | "
          f"{r.default_ms:.2f} | `{r.reversed_split}` | "
          f"{r.reversed_ms:.2f} | {speedup:.3f}x{flag} |")
    w("")
    if speedups:
        w(f"**Geomean**: {statistics.geometric_mean(speedups):.3f}x   "
          f"**Best**: {max(speedups):.3f}x   "
          f"**Worst**: {min(speedups):.3f}x")
        wins = sum(1 for s in speedups if s >= 1.05)
        regr = sum(1 for s in speedups if s <= 0.95)
        w(f"**>=5% wins**: {wins}/{len(speedups)}   "
          f"**>=5% regressions**: {regr}/{len(speedups)}")


def main() -> int:
    print(f"# Core-emission sweep")
    print(f"# PyTorch {torch.__version__}, "
          f"SENCORES={os.environ.get('SENCORES', '32 (default)')}, "
          f"warmup={WARMUP}, iters={ITERS}\n")

    # --- Phase A: natural picks under element_priority ---
    print("# Phase A — natural planner picks (element_priority=True), "
          "default vs reversed emitter\n")
    phase_a: list[_Result] = []
    for sh in PHASE_1_0_SHAPES:
        print(f"# {sh.label} ({sh.M}x{sh.N}x{sh.K})", flush=True)
        r = _measure(sh.label, sh.M, sh.N, sh.K,
                     target=None, use_element_priority=True)
        if r.error:
            print(f"  ERR {r.error}", flush=True)
        else:
            speedup = (r.default_ms / r.reversed_ms
                       if r.default_ms and r.reversed_ms else float("nan"))
            print(f"  default {r.default_ms:.2f} ms  "
                  f"reversed {r.reversed_ms:.2f} ms  "
                  f"speedup {speedup:.3f}x", flush=True)
        phase_a.append(r)

    # --- Phase B: forced mixed splits on hot shapes ---
    print("\n# Phase B — forced mixed (m, n, 1) splits on hot shapes, "
          "default vs reversed emitter\n")
    phase_b: list[_Result] = []
    for sh in HOT_SHAPES:
        splits = _mixed_splits(sh.M, sh.N, sh.K)
        print(f"# {sh.label} — {len(splits)} valid mixed splits",
              flush=True)
        for target in splits:
            r = _measure(sh.label, sh.M, sh.N, sh.K,
                         target=target, use_element_priority=False)
            if r.error:
                print(f"  {target}: ERR {r.error[:50]}", flush=True)
            else:
                speedup = (r.default_ms / r.reversed_ms
                           if r.default_ms and r.reversed_ms
                           else float("nan"))
                print(f"  {target}: default {r.default_ms:.2f}  "
                      f"reversed {r.reversed_ms:.2f}  "
                      f"speedup {speedup:.3f}x", flush=True)
            phase_b.append(r)

    # --- emit final table ---
    print()
    _emit_table(phase_a,
                "Phase A — natural picks (element_priority=True)")
    _emit_table(phase_b,
                "Phase B — forced mixed splits on hot shapes")

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_core_emission_sweep_results.md",
    )
    with open(out_path, "w") as f:
        print(f"# Core-emission sweep results", file=f)
        print(f"# PyTorch {torch.__version__}, "
              f"SENCORES={os.environ.get('SENCORES', '32 (default)')}, "
              f"warmup={WARMUP}, iters={ITERS}", file=f)
        _emit_table(phase_a,
                    "Phase A — natural picks (element_priority=True)",
                    file=f)
        _emit_table(phase_b,
                    "Phase B — forced mixed splits on hot shapes",
                    file=f)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
