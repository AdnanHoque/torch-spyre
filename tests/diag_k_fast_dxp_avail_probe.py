"""Two-condition 3-way campaign for the combined k_fast PR.

Runs the standard A/B/C measurement (pure-M id / K-split id / K-split kf)
under two LX availability conditions:

  default   — DXP_LX_FRAC_AVAIL=0.2 (20% reserved for backend)
  full-rsvd — DXP_LX_FRAC_AVAIL=1.0 (100% reserved for backend = 0 LX
              available to inductor; expected to error on shapes that
              need scratchpad)

Shape suite: the 12-shape cross-vendor cohort from
diag_k_fast_combined_findings_normalized.md plus 3 representative
Granite shapes covering the main regimes (A→B-dominated, B→C-dominated,
mixed).

Each condition is run in a subprocess so the env var takes effect
before torch_spyre imports.

Usage:
    python /tmp/dxp_avail_probe.py default
    python /tmp/dxp_avail_probe.py full-rsvd
"""

from __future__ import annotations

import os
import sys

# CONDITION must be set BEFORE importing torch_spyre.
if len(sys.argv) > 1:
    cond = sys.argv[1]
    if cond == "full-rsvd":
        os.environ["DXP_LX_FRAC_AVAIL"] = "1.0"
    elif cond == "default":
        os.environ["DXP_LX_FRAC_AVAIL"] = "0.2"
    else:
        raise SystemExit(f"unknown condition: {cond}")
else:
    cond = "default"
    os.environ["DXP_LX_FRAC_AVAIL"] = "0.2"

# Quiet down autoload noise.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import statistics  # noqa: E402
import time  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

import torch._inductor.config as _icfg  # noqa: E402

_icfg.compile_threads = 1
_icfg.worker_start_method = "fork"
_icfg.fx_graph_cache = False
_icfg.fx_graph_remote_cache = False

sys.stdout.reconfigure(line_buffering=True)

_REPO = Path("/home/adnan/dt-inductor/torch-spyre")
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch_spyre  # noqa: E402

torch_spyre._autoload()

from torch_spyre import streams as _ts  # noqa: E402

try:
    from torch_spyre._inductor import work_division as _planner  # noqa: E402
except ImportError:
    from torch_spyre._inductor import core_division as _planner  # noqa: E402

from torch_spyre._inductor import config as ts_config  # noqa: E402


WARMUP = 3
ITERS = 12
DTYPE = torch.float16
ELEMS_PER_STICK = 64


@dataclass
class Shape:
    label: str
    M: int
    N: int
    K: int


SHAPES: list[Shape] = [
    # Cross-vendor cohort (matches diag_k_fast_combined_findings_normalized.md)
    Shape("L3-70B kv_proj M=32",      32,  1024,  8192),
    Shape("L3-70B kv_proj M=128",    128,  1024,  8192),
    Shape("L3-70B kv_proj M=512",    512,  1024,  8192),
    Shape("Mixtral kv_proj M=128",   128,  1024,  4096),
    Shape("DSv3 kv_proj M=128",      128,  1536,  7168),
    Shape("DSv3 q_a_proj M=128",     128,  1536,  7168),
    Shape("L3-70B q_proj M=32",       32,  8192,  8192),
    Shape("DSv3 gate_proj M=32",      32, 18432,  7168),
    Shape("L3-70B q_proj M=128",     128,  8192,  8192),
    Shape("L3-70B q_proj M=512",     512,  8192,  8192),
    Shape("DSv3 down_proj M=128",    128,  7168, 18432),
    Shape("L3-70B kv_proj M=2048",  2048,  1024,  8192),
    # Granite — representative from each regime in granite findings
    Shape("Granite 8B q_proj M=128",     128,  4096,  4096),    # B→C-dominated
    Shape("Granite 8B gate_proj M=32",    32, 12800,  4096),    # A→B-dominated, (1,8,4)
    Shape("Granite 8B down_proj M=128",  128,  4096, 12800),    # B→C-dominated
]


def heuristic_split(M, N, K, max_cores=32):
    """Mirror of _try_k_fast_split (post-refactor) for split selection."""
    PT_ROWS = 8
    if max_cores < 2:
        return None
    n_sticks = N // ELEMS_PER_STICK
    k_sticks = K // ELEMS_PER_STICK
    rows_per_core = M / max_cores
    if rows_per_core < 1 or rows_per_core > 2 * PT_ROWS:
        return None
    if rows_per_core > PT_ROWS / 2 and n_sticks >= max_cores:
        return None
    if k_sticks < max_cores:
        return None
    from sympy import divisors
    candidates = sorted(
        (int(n) for n in divisors(max_cores) if 1 < n < max_cores), reverse=True
    )
    for n_split in candidates:
        if n_sticks % n_split != 0:
            continue
        k_split = max_cores // n_split
        if k_sticks < k_split or k_sticks % k_split != 0:
            continue
        return (1, n_split, k_split)
    return None


_orig_multi = _planner.multi_dim_iteration_space_split


def _force_split_factory(target):
    def _forced(it_space, max_cores, priorities, min_splits=None):
        syms = list(it_space.keys())
        if len(syms) != len(target):
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        if target[0] * target[1] * target[2] != max_cores:
            return _orig_multi(it_space, max_cores, priorities, min_splits)
        return {sym: target[i] for i, sym in enumerate(syms)}
    return _forced


@contextmanager
def _force_split(target):
    if target is None:
        yield
        return
    _planner.multi_dim_iteration_space_split = _force_split_factory(target)
    try:
        yield
    finally:
        _planner.multi_dim_iteration_space_split = _orig_multi


@contextmanager
def _kfast_emission(enabled: bool):
    prev = ts_config.core_id_k_fast_emission
    ts_config.core_id_k_fast_emission = enabled
    try:
        yield
    finally:
        ts_config.core_id_k_fast_emission = prev


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


def _compile_and_bench(M, N, K, split, kfast):
    a = torch.randn(M, K, dtype=DTYPE, device="spyre")
    b = torch.randn(K, N, dtype=DTYPE, device="spyre")
    torch._dynamo.reset()

    @torch.compile(dynamic=False)
    def mm(x, y):
        return x @ y

    try:
        with _kfast_emission(kfast), _force_split(split):
            mm(a, b)
        _ts.synchronize()

        def step():
            with _kfast_emission(kfast), _force_split(split):
                mm(a, b)
        return _bench(step), ""
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main() -> int:
    print(f"# DXP_LX_FRAC_AVAIL probe — condition={cond}, "
          f"DXP_LX_FRAC_AVAIL={os.environ['DXP_LX_FRAC_AVAIL']}")
    print(f"WARMUP={WARMUP}, ITERS={ITERS}, dtype=fp16, SENCORES=32\n")
    print("| shape | (M, N, K) | h-split | A→B | B→C | A→C | combined |")
    print("|---|---|---|---:|---:|---:|---|")

    n_win = 0
    n_reg = 0
    sum_a = 0.0
    sum_c = 0.0

    for s in SHAPES:
        h_split = heuristic_split(s.M, s.N, s.K)
        split_str = f"({h_split[0]},{h_split[1]},{h_split[2]})" if h_split else "—"

        a_ms, a_err = _compile_and_bench(s.M, s.N, s.K, (32, 1, 1), False)
        if h_split is not None:
            b_ms, b_err = _compile_and_bench(s.M, s.N, s.K, h_split, False)
            c_ms, c_err = _compile_and_bench(s.M, s.N, s.K, h_split, True)
        else:
            b_ms, c_ms = None, None
            b_err = c_err = ""

        if a_ms is not None and b_ms is not None and c_ms is not None:
            ab = a_ms / b_ms
            bc = b_ms / c_ms
            ac = a_ms / c_ms
            if ac > 1.05:
                pr_status = f"win {ac:.2f}×"
                n_win += 1
            elif ac < 0.95:
                pr_status = f"REGRESS {ac:.2f}×"
                n_reg += 1
            else:
                pr_status = "neutral"
            sum_a += a_ms
            sum_c += c_ms
            ab_s = f"{ab:.2f}×"
            bc_s = f"{bc:.2f}×"
            ac_s = f"{ac:.2f}×"
        elif h_split is None and a_ms is not None:
            ab_s = bc_s = ac_s = "—"
            pr_status = "(skipped, correct)"
        else:
            ab_s = bc_s = ac_s = "—"
            errs = [e for e in (a_err, b_err, c_err) if e]
            pr_status = f"ERR ({errs[0]})" if errs else "ERR"

        print(f"| {s.label} | ({s.M},{s.N},{s.K}) | {split_str} | "
              f"{ab_s} | {bc_s} | {ac_s} | {pr_status} |")

    print()
    print("## Aggregate\n")
    fired = sum(1 for s in SHAPES if heuristic_split(s.M, s.N, s.K) is not None)
    print(f"  shapes in suite: {len(SHAPES)}")
    print(f"  heuristic fires on: {fired}/{len(SHAPES)}")
    print(f"  A→C: {n_win} wins, {n_reg} regressions")
    if sum_a > 0 and sum_c > 0:
        print(f"  geomean A→C on shapes where heuristic fires: {sum_a / sum_c:.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
