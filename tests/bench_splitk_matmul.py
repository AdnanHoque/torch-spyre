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

"""Phase 1 perf+accuracy bench for SplitK matmul on Spyre.

For each (M, N, K) shape and mode (`default` | `forceK`):

- Compile mm with the configured planner mode
- Warmup, then time `ITERS` iterations with per-iter `ts.synchronize()`
- Compute TFLOPs/s = 2*M*N*K / median_time
- Cross-check correctness vs fp32 CPU reference, report abs/rel drift

Two sweeps:

1. **Decode-skinny K-crossover**: M=1, N=4096, K ∈ a sweep — characterizes
   the LLM-decode path where M=1 prevents M-split entirely.
2. **Balanced-square K-crossover**: M=N=1024, K ∈ same sweep — characterizes
   the prefill / training-step path where M and N can absorb cores.

The goal is to find the K threshold where forceK starts winning on perf
(end-to-end TFLOPs/s) and to confirm Phase 0's accuracy advantage holds at
proper measurement.

Run:  python tests/bench_splitk_matmul.py
"""

from __future__ import annotations

import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

# Same four config knobs as the Phase 0 diagnostic — see
# tests/diag_splitk_matmul.py for the rationale. Without all four the
# planner-stage monkey-patches silently no-op.
import torch._inductor.config as _icfg

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: F401  -- ensure backend is registered
from torch_spyre import streams as _ts

from torch_spyre._inductor import core_division as _core_div
from torch_spyre._inductor.codegen import superdsc as _superdsc


# ---- SDSC capture (lifted from diag) -----------------------------------------

@dataclass
class _DimSplit:
    sym: str
    size: int
    n_cores: int

    def fmt(self) -> str:
        return f"{self.size}×{self.n_cores}c"


@dataclass
class _MatmulCapture:
    op: str
    dims: list[_DimSplit]
    num_cores: int

    def splits_str(self) -> str:
        return "[" + ", ".join(d.fmt() for d in self.dims) + "]"

    def k_split(self) -> int:
        return self.dims[-1].n_cores if self.dims else 1


_captured: list[_MatmulCapture] = []
_orig_parse_op_spec = _superdsc.parse_op_spec


def _to_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def _wrapped_parse_op_spec(op_spec):
    sdsc_spec = _orig_parse_op_spec(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        dims = [
            _DimSplit(sym=str(sym), size=_to_int(size), n_cores=int(n_cores))
            for sym, (size, n_cores) in op_spec.iteration_space.items()
        ]
        _captured.append(
            _MatmulCapture(
                op=op_spec.op, dims=dims, num_cores=int(sdsc_spec.num_cores)
            )
        )
    return sdsc_spec


_superdsc.parse_op_spec = _wrapped_parse_op_spec  # type: ignore[assignment]


# ---- forceK mode -------------------------------------------------------------

_orig_prioritize_dimensions = _core_div.prioritize_dimensions


def _force_k_prioritize(output_td, it_space_remaining, exclude_reduction=False):
    orig = _orig_prioritize_dimensions(
        output_td, it_space_remaining, exclude_reduction=exclude_reduction
    )
    if exclude_reduction:
        return orig
    coord_vars = {
        v for e in output_td.device_coords[:-1] for v in e.free_symbols
    }
    output_syms = [s for s in orig if s in coord_vars]
    reduction_syms = [s for s in orig if s not in coord_vars]
    return reduction_syms + output_syms


@contextmanager
def _patch_for_mode(mode: str):
    if mode == "default":
        prio = _orig_prioritize_dimensions
    elif mode == "forceK":
        prio = _force_k_prioritize
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    _core_div.prioritize_dimensions = prio  # type: ignore[assignment]
    try:
        yield
    finally:
        _core_div.prioritize_dimensions = _orig_prioritize_dimensions  # type: ignore[assignment]


# ---- Drift -------------------------------------------------------------------

@dataclass
class _Drift:
    mean_abs: float
    p99_abs: float
    max_abs: float
    p99_rel: float


def _drift(out: torch.Tensor, ref: torch.Tensor, rel_eps: float = 1e-3) -> _Drift:
    out_f = out.detach().to(torch.float32).cpu().reshape(-1)
    ref_f = ref.detach().to(torch.float32).cpu().reshape(-1)
    abs_err = (out_f - ref_f).abs()
    mask = ref_f.abs() > rel_eps
    rel_err = torch.zeros_like(abs_err)
    rel_err[mask] = abs_err[mask] / ref_f[mask].abs()
    return _Drift(
        mean_abs=float(abs_err.mean().item()),
        p99_abs=float(torch.quantile(abs_err, 0.99).item()),
        max_abs=float(abs_err.max().item()),
        p99_rel=float(torch.quantile(rel_err, 0.99).item()),
    )


# ---- Bench loop --------------------------------------------------------------

WARMUP = 5
ITERS = 20


def _compile_mm():
    def _mm(a, b):
        return a @ b
    return torch.compile(_mm, dynamic=False)


def _bench_one(M: int, N: int, K: int, mode: str):
    """Returns (median_kernel_ms, drift, capture, error)."""
    a_cpu = torch.randn(M, K, dtype=torch.float16)
    b_cpu = torch.randn(K, N, dtype=torch.float16)
    a_dev = a_cpu.to("spyre")
    b_dev = b_cpu.to("spyre")
    ref = a_cpu.to(torch.float32) @ b_cpu.to(torch.float32)

    torch._dynamo.reset()
    cap_start = len(_captured)

    try:
        with _patch_for_mode(mode):
            mm_fn = _compile_mm()
            # First call triggers compile; warmup on top of that for cache /
            # initialization effects.
            for _ in range(WARMUP):
                mm_fn(a_dev, b_dev)
            _ts.synchronize()

            samples = []
            for _ in range(ITERS):
                t0 = time.perf_counter()
                out_dev = mm_fn(a_dev, b_dev)
                _ts.synchronize()
                samples.append(time.perf_counter() - t0)

        median_ms = statistics.median(samples) * 1e3
        # One D2H copy for correctness (NOT in the timed region).
        out_cpu = out_dev.to("cpu").to(torch.float32)
        drift = _drift(out_cpu, ref)
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"{type(e).__name__}: {e}"

    captures = list(_captured[cap_start:])
    cap = captures[0] if captures else None
    return median_ms, drift, cap, None


def _tflops(M: int, N: int, K: int, ms: float) -> float:
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


# ---- Sweeps ------------------------------------------------------------------

K_SWEEP = [1024, 2048, 4096, 8192, 12288, 16384]

SHAPES_DECODE = [(1, 4096, K) for K in K_SWEEP]      # M=1: decode-time
SHAPES_BALANCED = [(1024, 1024, K) for K in K_SWEEP] # M=N=1024: balanced

# Small-N decode sweep: M=1, K fixed at 8192 (Llama-70B-class kv_proj scale),
# N ∈ {128..4096}. Spyre's stick is 64 fp16 elements; default planner gives
# N/64 cores to N. So N=128 leaves 30 cores idle (saturation only at N=2048).
# This is the regime where forceK should win on perf — if the cross-core
# reduction overhead doesn't eat the win.
SHAPES_SMALL_N = [(1, N, 8192) for N in (128, 256, 512, 1024, 2048, 4096)]

# Prefill-M scaling at (N=4096, K=8192). Tests whether the forceK perf gap
# closes as M grows and the matmul becomes compute-dominated rather than
# fixed-overhead-dominated.
SHAPES_M_SCALING = [(M, 4096, 8192) for M in (128, 512, 2048)]

# Real Llama prefill shapes — q_proj and MLP-down for 8B and 70B variants.
# MLP-down is the canonical large-K matmul in modern LLMs (intermediate is
# 3.5x hidden in Llama-3) and is the most likely place for K-split to pay off.
SHAPES_LLAMA_PREFILL = [
    (128, 4096, 4096),   # Llama-3-8B q_proj prefill
    (128, 4096, 14336),  # Llama-3-8B MLP down-proj (large K)
    (128, 8192, 8192),   # Llama-3-70B q_proj prefill
    (128, 8192, 28672),  # Llama-3-70B MLP down-proj (huge K, ~470MB B; may
                         # force span-required K-split in default)
]

MODES = ("default", "forceK")


@dataclass
class _Row:
    shape: tuple[int, int, int]
    mode: str
    median_ms: float | None = None
    tflops: float | None = None
    drift: _Drift | None = None
    capture: _MatmulCapture | None = None
    error: str | None = None

    def k_split(self) -> str:
        if self.error:
            return "err"
        if self.capture:
            return str(self.capture.k_split())
        return "—"

    def cores(self) -> str:
        if self.capture:
            return str(self.capture.num_cores)
        return "—"


def _run_section(shapes: list) -> list[_Row]:
    """Run the bench for one section, streaming a one-liner per (shape,mode).
    Returns collected rows; does not emit tables."""
    rows: list[_Row] = []
    for M, N, K in shapes:
        for mode in MODES:
            ms, drift, cap, err = _bench_one(M, N, K, mode)
            tflops = _tflops(M, N, K, ms) if ms else None
            row = _Row(
                shape=(M, N, K),
                mode=mode,
                median_ms=ms,
                tflops=tflops,
                drift=drift,
                capture=cap,
                error=err,
            )
            rows.append(row)
            tag = "ERR" if err else f"K={row.k_split()}"
            tflops_s = "—" if tflops is None else f"{tflops:6.2f}"
            ms_s = "—" if ms is None else f"{ms:7.2f}"
            print(
                f"# {M:>5}×{N:>5}×{K:>5}  {mode:<7}  cores={row.cores():<3}  "
                f"{tag:<8}  {ms_s} ms  {tflops_s} TFLOPs/s",
                flush=True,
            )
    return rows


def _print_header(file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# SplitK matmul perf + accuracy bench — Phase 1")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"SENCORES:       {os.environ.get('SENCORES', '32 (default)')}")
    w(f"warmup iters:   {WARMUP}")
    w(f"measure iters:  {ITERS}")
    w("")
    w("**Timing**: per-iter `time.perf_counter()` around `mm_fn` + "
      "`torch_spyre.streams.synchronize()`. `.to('cpu')` happens AFTER the "
      "timed region for correctness check only — D2H is not in the kernel "
      "wall-time.")
    w("")
    w("**TFLOPs/s** = 2·M·N·K / median_time. End-to-end including any "
      "cross-core reduction (the dxp_standalone backend handles K-split "
      "partial-sum reduction; we observe its cost in the wall-time).")
    w("")
    w("**Drift** is vs fp32 CPU reference (`a.to(fp32) @ b.to(fp32)`). "
      "Relative error masks |ref| ≤ 1e-3.")
    w("")


def main() -> int:
    print("# starting decode-skinny sweep", flush=True)
    decode_rows = _run_section(SHAPES_DECODE)
    print("# starting balanced-square sweep", flush=True)
    balanced_rows = _run_section(SHAPES_BALANCED)
    print("# starting small-N decode sweep", flush=True)
    small_n_rows = _run_section(SHAPES_SMALL_N)
    print("# starting M-scaling sweep", flush=True)
    m_scaling_rows = _run_section(SHAPES_M_SCALING)
    print("# starting Llama prefill sweep", flush=True)
    llama_rows = _run_section(SHAPES_LLAMA_PREFILL)

    sections = [
        ("Decode-skinny sweep (M=1, N=4096)", decode_rows),
        ("Balanced-square sweep (M=N=1024)", balanced_rows),
        ("Small-N decode sweep (M=1, K=8192)", small_n_rows),
        ("M-scaling at (N=4096, K=8192)", m_scaling_rows),
        ("Llama prefill shapes", llama_rows),
    ]

    # Emit to stdout + sidecar file, both from already-collected rows.
    _print_header()
    for name, rows in sections:
        _emit_tables(None, name, rows)

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "bench_splitk_matmul_results.md",
    )
    with open(results_path, "w") as f:
        _print_header(file=f)
        for name, rows in sections:
            _emit_tables(f, name, rows)
    print(f"\n# results written to {results_path}", flush=True)
    return 0


def _emit_tables(file, name: str, rows: list[_Row]) -> None:
    """Emit the per-section tables to `file` from already-collected rows.
    Mirrors the table formatting in `_print_section` but without re-running
    the bench."""
    def w(s: str) -> None:
        print(s, file=file)

    w("")
    w(f"## {name}")
    w("")
    w("| shape (M×N×K) | mode | cores | splits | median ms | TFLOPs/s | "
      "abs(p99) | rel(p99) | note |")
    w("|---|---|---:|---|---:|---:|---:|---:|---|")
    for r in rows:
        M, N, K = r.shape
        if r.error:
            w(f"| {M}×{N}×{K} | {r.mode} | — | — | — | — | — | — | "
              f"{r.error} |")
            continue
        cap = r.capture
        d = r.drift
        splits = cap.splits_str() if cap else "—"
        ms = f"{r.median_ms:.2f}" if r.median_ms else "—"
        tf = f"{r.tflops:.2f}" if r.tflops else "—"
        absp = f"{d.p99_abs:.2e}" if d else "—"
        relp = f"{d.p99_rel:.2e}" if d else "—"
        w(
            f"| {M}×{N}×{K} | {r.mode} | {r.cores()} | {splits} | {ms} | "
            f"{tf} | {absp} | {relp} | |"
        )

    w("")
    w(f"### {name} — forceK vs default")
    w("")
    w("| shape | default TFLOPs/s | forceK TFLOPs/s | speedup | "
      "default abs(p99) | forceK abs(p99) | drift Δ |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    by_shape: dict[tuple, dict[str, _Row]] = {}
    for r in rows:
        by_shape.setdefault(r.shape, {})[r.mode] = r
    for shape, modes in by_shape.items():
        d = modes.get("default")
        f = modes.get("forceK")
        if not (d and f and d.tflops and f.tflops and d.drift and f.drift):
            continue
        speedup = f.tflops / d.tflops
        drift_delta = f.drift.p99_abs - d.drift.p99_abs
        w(
            f"| {shape[0]}×{shape[1]}×{shape[2]} | {d.tflops:.2f} | "
            f"{f.tflops:.2f} | {speedup:.2f}× | "
            f"{d.drift.p99_abs:.2e} | {f.drift.p99_abs:.2e} | "
            f"{drift_delta:+.2e} |"
        )


if __name__ == "__main__":
    raise SystemExit(main())
