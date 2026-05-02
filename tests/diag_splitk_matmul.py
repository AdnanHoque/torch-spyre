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

"""Phase 0 diagnostic for SplitK matmul on Spyre — v2.

Goal: characterize how the work-division planner splits matmul iteration
spaces across cores, and whether forcing K-first priority changes drift
versus the default M/N-greedy behavior. Output is per-shape drift stats
plus the *raw* per-dim split factors captured from the SDSC layer.

Three modes per shape:

1. **default**  — Spyre with the current planner (output dims first, K last).
2. **noK**      — `exclude_reduction=True` forced. Planner cannot K-split
                  even when the matmul reduction-op path would normally allow
                  it. May raise `Unsupported`; we log and continue.
3. **forceK**   — `prioritize_dimensions` monkey-patched to put reduction
                  dims (K) BEFORE output dims (M, N). Tests whether K-split
                  fires at all on Spyre and what its drift looks like.

We compare each Spyre output against an fp32 CPU reference (a.fp32 @ b.fp32)
and report abs/rel error (mean, p99, max). The relative-error mask drops
ref entries with |x| ≤ 1e-3 to keep div-by-near-zero from dominating.

Hook-fire counters confirm each monkey-patch actually reaches the call site
during compile — without these, a silent fallthrough (eager fallback,
late-binding miss, etc.) would be invisible.

Run:  python tests/diag_splitk_matmul.py
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

# Inductor's default compile pool (compile_threads=32, worker_start_method=
# subprocess) offloads IR-pass work to subprocess workers. Parent-process
# monkey-patches on plan_splits / prioritize_dimensions never reach the
# subprocess, so the noK / forceK modes silently no-op without this fix.
# Single-threaded fork-based compile keeps everything in the parent so our
# patches actually bite. Diagnostic-only — slows compile but irrelevant here.
import torch._inductor.config as _icfg

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"

# Inductor's FX-graph cache reuses compiled artifacts across processes when a
# graph hashes to a cached entry. That skips the entire scheduler pipeline —
# parse_op_spec still fires (artifact restoration) but plan_splits does not,
# so our patches silently no-op. Disable both caches so every compile runs
# the IR pipeline end-to-end.
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

import torch_spyre  # noqa: F401  -- ensure backend is registered

from torch_spyre._inductor import core_division as _core_div
from torch_spyre._inductor.codegen import superdsc as _superdsc


# ---- Hook-fire counters ------------------------------------------------------
# Reset between runs via _reset_counters() so each shape×mode reports its own.

_counters: dict[str, int] = {
    "parse_op_spec": 0,        # every parse_op_spec call (matmul or not)
    "matmul_capture": 0,       # only when the parsed op is a matmul
    "plan_splits": 0,          # every plan_splits call (forced or not)
    "no_ksplit_active": 0,     # plan_splits calls that went through noK wrapper
    "prioritize_dims": 0,      # every prioritize_dimensions call
    "force_k_active": 0,       # prioritize calls that went through forceK wrapper
}


def _reset_counters() -> None:
    for k in _counters:
        _counters[k] = 0


# ---- SDSC capture ------------------------------------------------------------

@dataclass
class _DimSplit:
    """One dim's record from the captured iteration_space."""
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

    def last_dim_split(self) -> int:
        """K-split factor by convention (last dim in iteration order is the
        reduction dim per Inductor's construction)."""
        return self.dims[-1].n_cores if self.dims else 1


_captured: list[_MatmulCapture] = []
_orig_parse_op_spec = _superdsc.parse_op_spec


def _to_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(v.subs([]))  # type: ignore[union-attr]
        except Exception:
            return -1


def _wrapped_parse_op_spec(op_spec):
    _counters["parse_op_spec"] += 1
    sdsc_spec = _orig_parse_op_spec(op_spec)
    if _superdsc._is_matmul(op_spec.op):
        _counters["matmul_capture"] += 1
        dims = [
            _DimSplit(sym=str(sym), size=_to_int(size), n_cores=int(n_cores))
            for sym, (size, n_cores) in op_spec.iteration_space.items()
        ]
        _captured.append(
            _MatmulCapture(
                op=op_spec.op,
                dims=dims,
                num_cores=int(sdsc_spec.num_cores),
            )
        )
    return sdsc_spec


def _install_capture_hook() -> None:
    _superdsc.parse_op_spec = _wrapped_parse_op_spec  # type: ignore[assignment]


def _uninstall_capture_hook() -> None:
    _superdsc.parse_op_spec = _orig_parse_op_spec  # type: ignore[assignment]


# ---- noK mode: force exclude_reduction=True for all plan_splits calls --------

_orig_plan_splits = _core_div.plan_splits


def _wrapped_plan_splits_default(*args, **kwargs):
    """Default-mode passthrough — increments plan_splits counter only."""
    _counters["plan_splits"] += 1
    return _orig_plan_splits(*args, **kwargs)


def _wrapped_plan_splits_no_ksplit(*args, **kwargs):
    """noK mode — force exclude_reduction=True even for matmul. Will raise
    core_division.Unsupported if the planner needs K-split to satisfy the
    256MB memory-span limit."""
    _counters["plan_splits"] += 1
    _counters["no_ksplit_active"] += 1
    kwargs["exclude_reduction"] = True
    return _orig_plan_splits(*args, **kwargs)


# ---- forceK mode: rotate reduction dims to the front of priority -------------

_orig_prioritize_dimensions = _core_div.prioritize_dimensions


def _wrapped_prioritize_default(
    output_td,
    it_space_remaining,
    exclude_reduction=False,
    min_splits=None,
):
    """Default-mode passthrough — increments prioritize counter only."""
    _counters["prioritize_dims"] += 1
    return _orig_prioritize_dimensions(
        output_td,
        it_space_remaining,
        exclude_reduction=exclude_reduction,
        min_splits=min_splits,
    )


def _wrapped_prioritize_force_k(
    output_td,
    it_space_remaining,
    exclude_reduction=False,
    min_splits=None,
):
    """forceK mode — call original, then rotate reduction dims to the front
    so the planner consumes cores along K before M/N."""
    _counters["prioritize_dims"] += 1
    _counters["force_k_active"] += 1
    orig = _orig_prioritize_dimensions(
        output_td,
        it_space_remaining,
        exclude_reduction=exclude_reduction,
        min_splits=min_splits,
    )
    if exclude_reduction:
        # Reduction dims are excluded — nothing to rotate.
        return orig
    # Identify which symbols in `orig` are reduction dims by checking against
    # the output's coordinate variables. Output dims appear in
    # output_td.device_coords[:-1] (the trailing entry is the stick); a
    # reduction dim does not.
    coord_vars = {
        v for e in output_td.device_coords[:-1] for v in e.free_symbols
    }
    output_syms = [s for s in orig if s in coord_vars]
    reduction_syms = [s for s in orig if s not in coord_vars]
    return reduction_syms + output_syms


# ---- Mode dispatch -----------------------------------------------------------

@contextmanager
def _patch_for_mode(mode: str):
    """Install plan_splits + prioritize_dimensions wrappers for a given mode."""
    if mode == "default":
        plan_patch = _wrapped_plan_splits_default
        prio_patch = _wrapped_prioritize_default
    elif mode == "noK":
        plan_patch = _wrapped_plan_splits_no_ksplit
        prio_patch = _wrapped_prioritize_default
    elif mode == "forceK":
        plan_patch = _wrapped_plan_splits_default
        prio_patch = _wrapped_prioritize_force_k
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    _core_div.plan_splits = plan_patch  # type: ignore[assignment]
    _core_div.prioritize_dimensions = prio_patch  # type: ignore[assignment]
    try:
        yield
    finally:
        _core_div.plan_splits = _orig_plan_splits  # type: ignore[assignment]
        _core_div.prioritize_dimensions = _orig_prioritize_dimensions  # type: ignore[assignment]


# ---- Drift stats -------------------------------------------------------------

@dataclass
class _DriftStats:
    n: int
    mean_abs: float
    p99_abs: float
    max_abs: float
    mean_rel: float
    p99_rel: float
    max_rel: float


def _drift(out: torch.Tensor, ref: torch.Tensor, rel_eps: float = 1e-3) -> _DriftStats:
    out_f = out.detach().to(torch.float32).cpu().reshape(-1)
    ref_f = ref.detach().to(torch.float32).cpu().reshape(-1)
    abs_err = (out_f - ref_f).abs()
    mask = ref_f.abs() > rel_eps
    rel_err = torch.zeros_like(abs_err)
    rel_err[mask] = abs_err[mask] / ref_f[mask].abs()

    def _pct(x: torch.Tensor, q: float) -> float:
        return float(torch.quantile(x, q).item())

    return _DriftStats(
        n=int(abs_err.numel()),
        mean_abs=float(abs_err.mean().item()),
        p99_abs=_pct(abs_err, 0.99),
        max_abs=float(abs_err.max().item()),
        mean_rel=float(rel_err.mean().item()),
        p99_rel=_pct(rel_err, 0.99),
        max_rel=float(rel_err.max().item()),
    )


# ---- Compile + run -----------------------------------------------------------

def _compile_mm():
    """torch.compile a tiny mm wrapper. No `backend=` — torch_spyre's
    compile_fx wrapper routes Spyre-tensor graphs automatically."""

    def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    return torch.compile(_mm, dynamic=False)


def _run_spyre(M: int, N: int, K: int, mode: str):
    """Compile + run mm(M,K) @ mm(K,N) on Spyre under the given mode.
    Returns (out_fp32 or None, ref_fp32, captures, hook_counters, error_or_None).
    """
    a_cpu = torch.randn(M, K, dtype=torch.float16)
    b_cpu = torch.randn(K, N, dtype=torch.float16)
    a_dev = a_cpu.to("spyre")
    b_dev = b_cpu.to("spyre")
    ref = a_cpu.to(torch.float32) @ b_cpu.to(torch.float32)

    # Reset dynamo's in-memory compile cache so each mode actually re-runs the
    # compile pipeline. Without this, noK/forceK on a shape already compiled
    # under default would cache-hit and skip the patched IR passes entirely.
    torch._dynamo.reset()

    capture_start = len(_captured)
    _reset_counters()

    err: str | None = None
    out: torch.Tensor | None = None
    try:
        with _patch_for_mode(mode):
            mm_fn = _compile_mm()
            out_dev = mm_fn(a_dev, b_dev)
        out = out_dev.to("cpu").to(torch.float32)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"

    captures_this_run = list(_captured[capture_start:])
    counters_snapshot = dict(_counters)
    return out, ref, captures_this_run, counters_snapshot, err


# ---- Sweep -------------------------------------------------------------------

@dataclass
class _Row:
    shape: tuple[int, int, int]
    mode: str
    captures: list[_MatmulCapture] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    drift: _DriftStats | None = None
    error: str | None = None
    wall_ms: float = 0.0


SHAPES: list[tuple[int, int, int]] = [
    (2048, 2048, 2048),    # balanced
    (1, 4096, 4096),       # decode skinny — K-split candidate
    (16, 4096, 4096),      # transitional
    (512, 512, 8192),      # large-K square
    (1024, 1024, 16384),   # large-K wider
]

MODES = ("default", "noK", "forceK")


def _bench_one(M: int, N: int, K: int, mode: str) -> _Row:
    t0 = time.perf_counter()
    out, ref, captures, counters, err = _run_spyre(M, N, K, mode)
    wall = (time.perf_counter() - t0) * 1e3
    drift = _drift(out, ref) if (out is not None and err is None) else None
    return _Row(
        shape=(M, N, K),
        mode=mode,
        captures=captures,
        counters=counters,
        drift=drift,
        error=err,
        wall_ms=wall,
    )


def _hook_str(c: dict[str, int]) -> str:
    return (
        f"parse={c.get('parse_op_spec', 0)}/"
        f"mm={c.get('matmul_capture', 0)} "
        f"plan={c.get('plan_splits', 0)} "
        f"noK={c.get('no_ksplit_active', 0)} "
        f"forceK={c.get('force_k_active', 0)}"
    )


def _print_header(file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("# SplitK matmul diagnostic — Phase 0 (v2)")
    w("")
    w(f"PyTorch:        {torch.__version__}")
    w(f"torch_spyre:    {getattr(torch_spyre, '__version__', '(editable)')}")
    w(f"SENCORES:       {os.environ.get('SENCORES', '32 (default)')}")
    w(f"shapes:         {len(SHAPES)}")
    w(f"modes:          {', '.join(MODES)}")
    w("")
    w("**default**: planner runs unmodified (output dims first, K last).")
    w("**noK**: `exclude_reduction=True` forced for every plan_splits call.")
    w("**forceK**: prioritize_dimensions monkey-patched to put K first.")
    w("")
    w("Splits column shows the captured iteration_space as `[size×ncores, ...]` "
      "in dict order. Last entry is the reduction (K) dim by convention.")
    w("")
    w("Drift is vs. fp32 CPU reference (a.to(fp32) @ b.to(fp32)).")
    w("")
    w("Hooks shows fire counts: parse=N/mm=N (parse_op_spec total / matmul "
      "captures), plan=N (plan_splits total), noK=N (noK wrapper hits), "
      "forceK=N (forceK wrapper hits).")
    w("")


def _print_table(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("| shape | mode | cores | splits | abs(p99) | abs(max) | rel(p99) | "
      "rel(max) | wall ms | hooks | note |")
    w("|---|---|---:|---|---:|---:|---:|---:|---:|---|---|")
    for r in rows:
        M, N, K = r.shape
        shape_s = f"{M}×{N}×{K}"
        hooks = _hook_str(r.counters)
        if r.error:
            w(
                f"| {shape_s} | {r.mode} | — | — | — | — | — | — | "
                f"{r.wall_ms:.1f} | {hooks} | {r.error} |"
            )
            continue
        if not r.captures:
            note = "no matmul kernel captured"
            w(
                f"| {shape_s} | {r.mode} | — | — | — | — | — | — | "
                f"{r.wall_ms:.1f} | {hooks} | {note} |"
            )
            continue
        cap = r.captures[0]
        d = r.drift
        if d is None:
            w(
                f"| {shape_s} | {r.mode} | {cap.num_cores} | "
                f"{cap.splits_str()} | — | — | — | — | "
                f"{r.wall_ms:.1f} | {hooks} | no drift |"
            )
            continue
        w(
            f"| {shape_s} | {r.mode} | {cap.num_cores} | "
            f"{cap.splits_str()} | "
            f"{d.p99_abs:.2e} | {d.max_abs:.2e} | "
            f"{d.p99_rel:.2e} | {d.max_rel:.2e} | "
            f"{r.wall_ms:.1f} | {hooks} | |"
        )


def _print_summary(rows: list[_Row], file=None) -> None:
    def w(s: str) -> None:
        print(s, file=file)

    w("")
    w("## Summary")
    w("")

    # Group by shape; for each, report whether K-split fired in each mode.
    by_shape: dict[tuple, dict[str, _Row]] = {}
    for r in rows:
        by_shape.setdefault(r.shape, {})[r.mode] = r

    w("### K-split (last-dim split factor) per shape × mode")
    w("")
    w("| shape | default K | noK K | forceK K |")
    w("|---|---:|---:|---:|")
    for shape, modes in by_shape.items():
        row = []
        for m in MODES:
            r = modes.get(m)
            if r and r.captures:
                row.append(str(r.captures[0].last_dim_split()))
            elif r and r.error:
                row.append("err")
            elif r:
                row.append("no-cap")
            else:
                row.append("—")
        w(f"| {shape[0]}×{shape[1]}×{shape[2]} | {row[0]} | {row[1]} | "
          f"{row[2]} |")
    w("")

    # If forceK actually changed K-split for any shape, report drift delta.
    w("### forceK-vs-default drift delta (where both ran with captures)")
    w("")
    w("| shape | default K | forceK K | abs(p99) Δ | rel(p99) Δ |")
    w("|---|---:|---:|---:|---:|")
    for shape, modes in by_shape.items():
        d = modes.get("default")
        f = modes.get("forceK")
        if not (d and f and d.drift and f.drift and d.captures and f.captures):
            continue
        dd, ff = d.drift, f.drift
        dk = d.captures[0].last_dim_split()
        fk = f.captures[0].last_dim_split()
        w(
            f"| {shape[0]}×{shape[1]}×{shape[2]} | {dk} | {fk} | "
            f"{ff.p99_abs - dd.p99_abs:+.2e} | "
            f"{ff.p99_rel - dd.p99_rel:+.2e} |"
        )


def main() -> int:
    _install_capture_hook()
    rows: list[_Row] = []
    try:
        for M, N, K in SHAPES:
            for mode in MODES:
                row = _bench_one(M, N, K, mode)
                rows.append(row)
                cap_tag = (
                    f"K={row.captures[0].last_dim_split()}"
                    if row.captures else "no-cap"
                )
                tag = "ERR" if row.error else cap_tag
                cores = row.captures[0].num_cores if row.captures else None
                core_s = "—" if cores is None else f"{cores}"
                print(
                    f"# {M:>5}×{N:>5}×{K:>5}  {mode:<7}  cores={core_s:<3}  "
                    f"{tag:<10}  {row.wall_ms:.1f}ms  "
                    f"[{_hook_str(row.counters)}]",
                    flush=True,
                )
    finally:
        _uninstall_capture_hook()

    _print_header()
    _print_table(rows)
    _print_summary(rows)

    results_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "diag_splitk_matmul_results.md",
    )
    with open(results_path, "w") as f:
        _print_header(file=f)
        _print_table(rows, file=f)
        _print_summary(rows, file=f)
    print(f"\n# results written to {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
