# Copyright 2026 The Torch-Spyre Authors.
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

"""First-principles probe: which memory does today's restickify touch?

Forgets the STCDPOpLx framing. Captures three independent signals about
where the restickified data actually moves, on a single compiled graph:

  Signal 1 - allocation:
    For each restickify ComputedBuffer, log the input and output
    buffers' `layout.allocation` (LX-resident vs HBM-resident) after
    `scratchpad_planning` has decided placement. Static signal.

  Signal 2 - kernel choice:
    Wrap `SpyreAsyncCompile.sdsc` to capture the op-func string the
    codegen emitted for each kernel. `ReStickifyOpHBM` implies HBM
    path; anything else would imply a different path. Static signal.

  Signal 3 - bandwidth fingerprint:
    Time the compiled graph with and without the restickify (graphs B
    and A, isomorphic FLOPs/output-shape). Compute effective bandwidth
    under each hypothesis:
       BW_hbm  = 2 * B / dt   (HBM round-trip: B write + B read)
       BW_lx   = B / dt       (one-way cross-core LX-LX hop)
    Compare to architecture ceilings:
       HBM bus              ~107 GB/s  (MNI single-node, effective)
       Cross-core LX-LX     ~1328 GB/s (RIU ring, uniform all-to-all)
       Per-core LX bus     ~4500 GB/s aggregate (no inter-core movement)

  Signal 4 - bytes-moved sanity check:
    `expected_bytes = numel * dtype_size`. Time-derived bytes at the
    HBM ceiling = `dt * 107e9`. Ratio > 1 means the kernel moves more
    HBM bytes than the tensor's notional size (alignment, double
    buffering, spill).

If Signals 1, 2, and 3 all converge on the HBM path:
  - allocation: HBM/HBM
  - op-func:    ReStickifyOpHBM
  - effective BW_hbm: ~107 GB/s
then today's restickify is unambiguously an HBM round-trip. If any
signal disagrees, we've discovered a subtlety the cost model missed.

Run: SENCORES=32 LX_PLANNING=1 .venv/bin/python tests/diag_restickify_memory_path.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import Counter
from unittest.mock import patch

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_spyre

torch_spyre._autoload()

from torch._inductor import config as t_inductor_config
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer
from torch_spyre._inductor import config as ts_config
from torch_spyre._inductor import passes as ts_passes
from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor.restickify_classify import (
    RestickifyVerdict,
    _is_restickify,
    classify_all_restickifies,
)
from torch_spyre.execution import async_compile as ac


HD = 4096
M_VALUES = [128, 512, 2048, 8192]
DTYPE = torch.float16
DEVICE = "spyre"
WARMUP = 5
ITERS = 50

# Spec / measured architecture ceilings on AIU 1.0.
HBM_BW = 107e9              # MNI single-node, measured effective
CROSS_CORE_LX_BW = 1328e9   # RIU ring, uniform all-to-all (32 * 2 * 166 / 8)
PER_CORE_LX_BW = 4500e9     # 32 cores * 140 GB/s aggregate


def _location(buf) -> str:
    """LX iff the FixedTiledLayout.allocation dict has any 'lx' entry."""
    alloc = getattr(buf.get_layout(), "allocation", None) or {}
    return "LX" if any("lx" in str(k).lower() for k in alloc) else "HBM"


def _restickify_alloc_info(operations) -> list[dict]:
    """For each restickify in operations, return its input/output allocation
    + verdict + tensor numel. Reads buffer state after scratchpad_planning."""
    from torch._inductor.virtualized import V

    verdicts = classify_all_restickifies(operations)
    info: list[dict] = []
    for op in operations:
        if not (isinstance(op, ComputedBuffer) and _is_restickify(op)):
            continue
        out_name = op.get_name()
        try:
            in_reads = [
                d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)
            ]
        except Exception:
            in_reads = []
        in_name = in_reads[0].name if in_reads else None
        in_buf = V.graph.get_buffer(in_name) if in_name else None
        out_buf = V.graph.get_buffer(out_name)
        info.append(
            {
                "name": out_name,
                "in_name": in_name,
                "in_loc": _location(in_buf) if in_buf is not None else "?",
                "out_loc": _location(out_buf),
                "verdict": verdicts.get(out_name),
                "out_numel": int(
                    1 if not out_buf.get_layout().size
                    else _prod_int(out_buf.get_layout().size)
                ),
            }
        )
    return info


def _prod_int(seq) -> int:
    p = 1
    for s in seq:
        try:
            p *= int(s)
        except (TypeError, ValueError):
            return 0
    return p


def time_compiled(fn, args) -> list[float]:
    for _ in range(WARMUP):
        out = fn(*args)
        _ = out.sum().item()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn(*args)
        _ = out.sum().item()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
    return times


def main():
    captured_alloc: dict = {}      # M -> list[alloc_info]
    captured_opfunc: Counter = Counter()
    current = {"label": None}

    class _HookedPasses(CustomPreSchedulingPasses):
        def __call__(self, operations):
            super().__call__(operations)
            if current["label"] is not None and current["label"].startswith("B"):
                captured_alloc[current["label"]] = _restickify_alloc_info(operations)

    orig_sdsc = ac.SpyreAsyncCompile.sdsc

    def wrapped_sdsc(self, kernel_name, specs):
        if current["label"] is not None:
            for spec in getattr(specs, "op_specs", specs) or []:
                opfunc = getattr(spec, "op", None)
                if opfunc is not None:
                    captured_opfunc[(current["label"], opfunc)] += 1
        return orig_sdsc(self, kernel_name, specs)

    def fn_a(X1, Y):
        return torch.matmul(X1, Y)

    def fn_b(X2, Y):
        return torch.matmul(X2.t(), Y)

    def fresh_compile(fn):
        torch._dynamo.reset_code_caches()
        torch._inductor.codecache.FxGraphCache.clear()
        torch.compiler.reset()
        return torch.compile(fn, fullgraph=True)

    rows = []
    try:
        patchers = [
            t_inductor_config.patch("force_disable_caches", True),
            ts_config.patch("lx_planning", True),
            ts_config.patch("allow_all_ops_in_lx_planning", True),
            ts_config.patch("sencores", 32),
            patch.object(ts_passes, "CustomPreSchedulingPasses", _HookedPasses),
            patch.object(ac.SpyreAsyncCompile, "sdsc", wrapped_sdsc),
        ]
        for p in patchers:
            p.__enter__()
        torch.compiler.reset()

        try:
            for M in M_VALUES:
                X1 = torch.rand((M, HD), dtype=DTYPE, device=DEVICE)
                X2 = torch.rand((HD, M), dtype=DTYPE, device=DEVICE)
                Y = torch.rand((HD, HD), dtype=DTYPE, device=DEVICE)

                current["label"] = f"A_M{M}"
                compiled_a = fresh_compile(fn_a)
                try:
                    times_a = time_compiled(compiled_a, (X1, Y))
                except Exception as e:
                    print(f"  M={M} A failed: {type(e).__name__}: {e}", flush=True)
                    current["label"] = None
                    continue

                current["label"] = f"B_M{M}"
                compiled_b = fresh_compile(fn_b)
                try:
                    times_b = time_compiled(compiled_b, (X2, Y))
                except Exception as e:
                    print(f"  M={M} B failed: {type(e).__name__}: {e}", flush=True)
                    current["label"] = None
                    continue
                current["label"] = None

                ta = statistics.median(times_a)
                tb = statistics.median(times_b)
                dt = tb - ta  # restickify-only wall-clock, ms
                bytes_moved = M * HD * 2  # |X| in bytes (one tensor of restickify)
                # Effective bandwidth interpretations (GB/s):
                #   HBM round-trip: 2B / dt
                #   cross-core LX one-way hop: B / dt
                dt_s = dt * 1e-3
                bw_hbm_eff = (2 * bytes_moved / dt_s) / 1e9 if dt_s > 0 else 0
                bw_lx_eff = (bytes_moved / dt_s) / 1e9 if dt_s > 0 else 0
                # How many HBM bytes the kernel implicitly moves if dt is at the
                # measured 107 GB/s HBM ceiling. Ratio > 2 means more-than-round-trip.
                implied_hbm_bytes = dt_s * HBM_BW
                bytes_ratio = implied_hbm_bytes / bytes_moved if bytes_moved > 0 else 0
                rows.append((M, bytes_moved, dt, bw_hbm_eff, bw_lx_eff, bytes_ratio))
                print(
                    f"  done M={M}: dt={dt:.3f}ms "
                    f"|X|={bytes_moved/1e6:.1f}MB "
                    f"bw_hbm_eff={bw_hbm_eff:.1f}GB/s "
                    f"bw_lx_eff={bw_lx_eff:.1f}GB/s "
                    f"ratio={bytes_ratio:.2f}",
                    flush=True,
                )
        finally:
            for p in reversed(patchers):
                p.__exit__(None, None, None)
            torch.compiler.reset()
    except Exception as e:
        print(f"top-level failure: {type(e).__name__}: {e}", flush=True)
        raise

    # ----- report -----
    print()
    print(f"Restickify memory-path probe -- HD={HD}, dtype={DTYPE}")
    print(f"  pattern A: torch.matmul(X1:(M,HD) , Y:(HD,HD))           no restickify")
    print(f"  pattern B: torch.matmul(X2:(HD,M).t(), Y:(HD,HD))         restickify on X2")
    print(f"  WARMUP={WARMUP}, ITERS={ITERS}, "
          f"SENCORES={os.environ.get('SENCORES', '32')}, "
          f"LX_PLANNING={os.environ.get('LX_PLANNING', '?')}")
    print()
    print("== Signal 3 + 4: bandwidth fingerprint & bytes-moved ==")
    print(f"  {'M':>6} {'|X|MB':>7} {'dt(ms)':>8} {'BW_hbm':>9} {'BW_lx':>9} "
          f"{'HBM/|X|':>8}")
    print("  " + "-" * 50)
    for M, _b, dt, bw_h, bw_l, ratio in rows:
        print(
            f"  {M:>6} {_b/1e6:>7.1f} {dt:>8.3f} {bw_h:>8.1f}G {bw_l:>8.1f}G "
            f"{ratio:>8.2f}"
        )
    print()
    print(f"  Architecture ceilings: HBM={HBM_BW/1e9:.0f} GB/s | "
          f"cross-core LX-LX={CROSS_CORE_LX_BW/1e9:.0f} GB/s | "
          f"per-core LX={PER_CORE_LX_BW/1e9:.0f} GB/s")
    print(f"  Interpretation:")
    print(f"    BW_hbm near {HBM_BW/1e9:.0f} GB/s   -> HBM round-trip "
          f"(2B HBM traffic)")
    print(f"    BW_lx near {CROSS_CORE_LX_BW/1e9:.0f} GB/s -> cross-core LX-LX "
          f"(B one-way ring traffic)")
    print(f"    HBM/|X| near 2.0          -> kernel moves exactly tensor "
          f"size in each direction")
    print(f"    HBM/|X| > 2.0             -> kernel moves more bytes than "
          f"tensor (spill, double-buffer)")

    print()
    print("== Signal 1: allocation per restickify (where buffers live) ==")
    print(f"  (read from layout.allocation after scratchpad_planning)")
    if captured_alloc:
        for label in sorted(captured_alloc):
            entries = captured_alloc[label]
            if not entries:
                print(f"  {label}: no restickifies inserted")
                continue
            for e in entries:
                v = e.get("verdict")
                v_str = v.value if v is not None else "?"
                print(
                    f"  {label} {e['name']:<8} "
                    f"in={e['in_loc']}/out={e['out_loc']}  "
                    f"verdict={v_str:<11}  numel={e['out_numel']:>10}"
                )
    else:
        print("  (no allocation info captured)")

    print()
    print("== Signal 2: SDSC op-func names emitted ==")
    if captured_opfunc:
        for (label, opfunc), n in sorted(captured_opfunc.items()):
            print(f"  {label:<10} {opfunc:<30} x{n}")
    else:
        print("  (no op-funcs captured)")

    print()
    print("== Verdict (does today's restickify do an HBM round-trip?) ==")
    print(f"  HBM round-trip == YES if all three are true:")
    print(f"    Signal 1: allocations are HBM/HBM (or HBM-leaning) at sc=32")
    print(f"    Signal 2: op-func is ReStickifyOpHBM")
    print(f"    Signal 3: BW_hbm ~= {HBM_BW/1e9:.0f} GB/s (within ~15%)")


if __name__ == "__main__":
    main()
