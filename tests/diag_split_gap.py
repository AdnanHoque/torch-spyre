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

"""Phase 1.0 of the cost-model planner project.

For each production matmul shape, enumerate all valid `(m_split, n_split,
k_split)` factorizations where `m·n·k = num_cores`, force each via the
`multi_dim_iteration_space_split` monkey-patch (same mechanism as the
DDR-traffic diagnostic), and measure wall-time. Compare to the default
planner's actual choice.

Output: per-shape gap = (default_wall_ms - best_forced_wall_ms) /
default_wall_ms. Aggregate across the production shape catalog.

This is the gating measurement for the project: if the average gap is
small (<10%), the cost model has limited headroom and we'd narrow scope.
If the gap is meaningful (>15-20%), the cost model is justified as a
real perf lever beyond the M=521-class anomalies.

Run: python tests/diag_split_gap.py
"""

from __future__ import annotations

import math
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

import torch._inductor.config as _icfg
_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: F401
from torch_spyre import streams as _ts
from torch_spyre._inductor import core_division as _core_div
from torch_spyre._inductor.codegen import superdsc as _superdsc


# ---- SDSC capture (records the planner's actual split per compile) -------

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


# ---- Force-split monkey-patch -------------------------------------------

_orig_multi = _core_div.multi_dim_iteration_space_split


def _force_split_factory(target: tuple[int, int, int]):
    """Force `multi_dim_iteration_space_split` to return `target` for any
    iteration space whose length matches len(target). Other iteration
    spaces fall through unchanged."""
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
def _force_split(target: tuple[int, int, int]):
    _core_div.multi_dim_iteration_space_split = _force_split_factory(target)  # type: ignore[assignment]
    try:
        yield
    finally:
        _core_div.multi_dim_iteration_space_split = _orig_multi  # type: ignore[assignment]


# ---- factorization enumeration -------------------------------------------

NUM_CORES = 32
STICK_ELEMS = 64  # fp16


def _divisors_le(n: int, limit: int) -> list[int]:
    return [d for d in range(1, limit + 1) if n % d == 0 and d <= n]


def _factorizations_of(total: int) -> list[tuple[int, int, int]]:
    """All (m, n, k) ordered triples with m*n*k = total."""
    out = []
    for m in range(1, total + 1):
        if total % m != 0:
            continue
        rem = total // m
        for n in range(1, rem + 1):
            if rem % n != 0:
                continue
            k = rem // n
            out.append((m, n, k))
    return out


def _is_valid_split(M: int, N: int, K: int, m: int, n: int, k: int) -> tuple[bool, str]:
    """Check basic constraints. Returns (valid, reason_if_invalid)."""
    # m: M is non-stick (in elements). M/m must be integer.
    if M % m != 0:
        return False, f"M={M} not divisible by m={m}"
    # n: N is stick. N/n must be ≥ stick_size and stick-aligned.
    n_per_core_elems = N // n
    if n_per_core_elems < STICK_ELEMS:
        return False, f"N/n={n_per_core_elems} < stick={STICK_ELEMS}"
    if n_per_core_elems % STICK_ELEMS != 0:
        return False, f"N/n={n_per_core_elems} not stick-aligned"
    # k: K is stick. Same check.
    k_per_core_elems = K // k
    if k_per_core_elems < STICK_ELEMS:
        return False, f"K/k={k_per_core_elems} < stick={STICK_ELEMS}"
    if k_per_core_elems % STICK_ELEMS != 0:
        return False, f"K/k={k_per_core_elems} not stick-aligned"
    return True, ""


# ---- bench primitive ----------------------------------------------------

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


def _run_mm(M: int, N: int, K: int, target: tuple[int, int, int] | None):
    """Compile + bench one matmul under the given split (or default if None).
    Returns (median_ms, captured_split_str, error_or_None)."""
    a = torch.randn(M, K, dtype=torch.float16, device="spyre")
    b = torch.randn(K, N, dtype=torch.float16, device="spyre")

    torch._dynamo.reset()
    cap_start = len(_captured)

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        if target is None:
            mm(a, b)
        else:
            with _force_split(target):
                mm(a, b)
        _ts.synchronize()

        captures = _captured[cap_start:]
        cap_str = ""
        if captures:
            _, dims = captures[0]
            cap_str = "[" + ", ".join(f"{sz}×{nc}c" for _, sz, nc in dims) + "]"

        if target is None:
            ms = _bench(lambda: mm(a, b))
        else:
            def step():
                with _force_split(target):
                    mm(a, b)
            ms = _bench(step)
        return ms, cap_str, None
    except Exception as e:  # noqa: BLE001
        return None, "", f"{type(e).__name__}: {str(e)[:100]}"


# ---- shapes -----------------------------------------------------------

@dataclass
class _Shape:
    label: str
    M: int
    N: int
    K: int
    use_case: str


SHAPES: list[_Shape] = [
    # Llama-3-8B prefill at typical batch (M=128 tokens)
    _Shape("L3-8B q_proj prefill",       128, 4096, 4096,   "Llama-3-8B prefill"),
    _Shape("L3-8B GQA kv_proj prefill",  128, 1024, 4096,   "Llama-3-8B GQA"),
    _Shape("L3-8B MLP gate/up prefill",  128, 14336, 4096,  "Llama-3-8B MLP"),
    _Shape("L3-8B MLP down prefill",     128, 4096, 14336,  "Llama-3-8B MLP"),

    # Llama-3-70B prefill
    _Shape("L3-70B q_proj prefill",      128, 8192, 8192,   "Llama-3-70B prefill"),
    _Shape("L3-70B GQA kv_proj prefill", 128, 1024, 8192,   "Llama-3-70B GQA"),
    _Shape("L3-70B GQA TP=8 kv prefill", 128, 128, 8192,    "Llama-3-70B GQA TP=8"),
    _Shape("L3-70B MLP down prefill",    128, 8192, 28672,  "Llama-3-70B MLP"),

    # MoE per-expert at typical prefill (top_k×batch / num_experts ≈ 128-192 tokens/expert)
    _Shape("Mixtral down per-expert",    128, 4096, 14336,  "Mixtral 8x7B per-expert"),
    _Shape("Qwen3-MoE gate per-expert",  128, 1536, 2048,   "Qwen3-MoE per-expert"),
    _Shape("DeepSeek-MoE gate (M=192)",  192, 1408, 2048,   "DeepSeek-MoE per-expert"),

    # Decode (M=1) — should be launch-overhead-bound; gap likely small
    _Shape("L3-8B q_proj decode",        1, 4096, 4096,     "Llama-3-8B decode"),
    _Shape("L3-70B GQA TP=8 kv decode",  1, 128, 8192,      "Llama-3-70B decode"),
]


# ---- sweep + report ----------------------------------------------------

@dataclass
class _Trial:
    target: tuple[int, int, int] | None  # None = default planner
    captured: str
    median_ms: float | None
    error: str | None


@dataclass
class _ShapeResult:
    shape: _Shape
    default: _Trial
    forced: list[_Trial] = field(default_factory=list)


def _measure_shape(sh: _Shape) -> _ShapeResult:
    print(f"\n# {sh.label} ({sh.M}×{sh.N}×{sh.K}) — {sh.use_case}", flush=True)

    # 1. default planner
    ms, cap, err = _run_mm(sh.M, sh.N, sh.K, target=None)
    default = _Trial(target=None, captured=cap, median_ms=ms, error=err)
    if err:
        print(f"  default: ERR {err}", flush=True)
    else:
        print(f"  default: {cap} → {ms:.2f} ms", flush=True)

    # 2. enumerate valid factorizations
    factorings = _factorizations_of(NUM_CORES)
    valid = [
        (m, n, k) for (m, n, k) in factorings
        if _is_valid_split(sh.M, sh.N, sh.K, m, n, k)[0]
    ]
    print(f"  {len(valid)} valid factorizations of {NUM_CORES}", flush=True)

    forced_results: list[_Trial] = []
    for (m, n, k) in valid:
        ms, cap, err = _run_mm(sh.M, sh.N, sh.K, target=(m, n, k))
        trial = _Trial(target=(m, n, k), captured=cap, median_ms=ms, error=err)
        forced_results.append(trial)
        if err:
            print(f"  ({m:>2},{n:>2},{k:>2}): ERR {err[:60]}", flush=True)
        else:
            tag = " ←DEFAULT" if cap == default.captured else ""
            print(f"  ({m:>2},{n:>2},{k:>2}): {ms:.2f} ms{tag}", flush=True)

    return _ShapeResult(shape=sh, default=default, forced=forced_results)


def _print_table(results: list[_ShapeResult], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# Cost-model gap probe — Phase 1.0")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"SENCORES:       {os.environ.get('SENCORES', f'{NUM_CORES} (default)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w("")
    w("**Method**: for each shape, force every valid `(m, n, k)` "
      f"factorization of {NUM_CORES} via `multi_dim_iteration_space_split` "
      "monkey-patch, measure wall-time. Compare to default planner's choice. "
      "**Gap = how much perf the planner is leaving on the table at this "
      "shape.**")
    w("")
    w("Validity: (M / m), (N / n), (K / k) all integer, (N / n) and (K / k) "
      f"≥ {STICK_ELEMS} (stick) and stick-aligned. Decode shapes (M=1) have "
      f"limited valid factorizations because m must = 1.")
    w("")

    # Summary table — one row per shape
    w("## Summary: per-shape gap")
    w("")
    w("| shape | use case | default split | default ms | best forced | best ms | gap |")
    w("|---|---|---|---:|---|---:|---:|")
    overall_gaps: list[float] = []
    for r in results:
        sh = r.shape
        if r.default.error or r.default.median_ms is None:
            w(f"| {sh.label} | {sh.use_case} | err | err | — | — | err |")
            continue
        ok_forced = [t for t in r.forced if t.median_ms is not None]
        if not ok_forced:
            w(f"| {sh.label} | {sh.use_case} | {r.default.captured} | "
              f"{r.default.median_ms:.2f} | (no valid) | — | — |")
            continue
        best = min(ok_forced, key=lambda t: t.median_ms)
        gap = (r.default.median_ms - best.median_ms) / r.default.median_ms
        overall_gaps.append(gap)
        best_str = f"{best.target}"
        marker = "" if r.default.captured else ""
        w(f"| {sh.label} | {sh.use_case} | `{r.default.captured}` | "
          f"{r.default.median_ms:.2f} | `{best_str}` | "
          f"{best.median_ms:.2f} | {gap*100:+.1f}% |")
    w("")

    if overall_gaps:
        avg = sum(overall_gaps) / len(overall_gaps)
        max_gap = max(overall_gaps)
        positive = [g for g in overall_gaps if g > 0]
        w(f"**Across {len(overall_gaps)} measurable shapes**: "
          f"average gap **{avg*100:.1f}%**, max **{max_gap*100:.1f}%**, "
          f"{len(positive)} shapes have positive gap (planner is suboptimal).")
        w("")

    # Detail tables — one per shape with all factorizations
    for r in results:
        sh = r.shape
        w(f"## {sh.label} — `({sh.M}, {sh.N}, {sh.K})`")
        w("")
        w("| (m, n, k) | wall ms | vs default | note |")
        w("|---|---:|---:|---|")
        d_ms = r.default.median_ms
        for t in r.forced:
            tag = ""
            if t.captured == r.default.captured:
                tag = "← DEFAULT"
            if t.error:
                w(f"| {t.target} | err | — | {t.error[:60]} |")
                continue
            ms = t.median_ms
            ratio = (ms / d_ms) if (d_ms and ms) else float("nan")
            w(f"| {t.target} | {ms:.2f} | {ratio:.3f}× | {tag} |")
        w("")


def main() -> int:
    results: list[_ShapeResult] = []
    for sh in SHAPES:
        results.append(_measure_shape(sh))

    _print_table(results)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_split_gap_results.md",
    )
    with open(out_path, "w") as f:
        _print_table(results, file=f)
    print(f"\n# results written to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
