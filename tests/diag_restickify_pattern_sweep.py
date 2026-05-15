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

"""Sweep through guaranteed-restickify patterns from test_restickify.py,
comparing measured wall-clock cost to theoretical fabric predictions.

Patterns are sourced from `tests/inductor/test_restickify.py` -- only
those that assert `optimal_cost > 0`, meaning a restickify is guaranteed
to be in the plan (or, for the codegen-emission path discovered by
probe v3, in the emitted SDSC).

For each pattern, we time a paired (A, B) graph where A has no
restickify and B has one, holding compute volume fixed. The delta is
the restickify cost. We compare it to four theoretical predictions:

* `T_hbm_spec  = 2|B|  / 166 GB/s`   HBM round-trip at spec
* `T_hbm_eff   = 2|B|  / 107 GB/s`   HBM round-trip at measured effective
* `T_ring_spec = |B|   / 1328 GB/s`  Cross-core LX-LX, uniform all-to-all spec
* `T_ring_eff  = |B|   /  850 GB/s`  Same with 64% efficiency (HBM ratio)

`|B|` is the bytes-restickified for the pattern, derived from the
`optimal_cost` assertions in test_restickify.py converted to bytes
(elements * 2 for fp16).

Fabric ceilings from the AIU 1.0 spec:
* HBM is one ring node (MNI) at 166 GB/s aggregate.
* RIU BiRing is 32 nodes at 166 GB/s/dir bidirectional each link.
* Cross-core LX-LX uses 32 nodes in parallel; HBM uses one.

The honest speedup bracket is 10-25x. Spec-to-spec is 16x. Effective-
to-spec is 24.8x. Effective-to-effective is ~16x.

Run: SENCORES=32 LX_PLANNING=1 .venv/bin/python tests/diag_restickify_pattern_sweep.py
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
from torch_spyre._inductor import config as ts_config
from torch_spyre.execution import async_compile as ac


DTYPE = torch.float16
DEVICE = "spyre"
WARMUP = 5
ITERS = 30

HBM_SPEC = 166e9
HBM_EFF = 107e9
RING_SPEC = 1328e9
RING_EFF = 850e9


def fresh_compile(fn):
    torch._dynamo.reset_code_caches()
    torch._inductor.codecache.FxGraphCache.clear()
    torch.compiler.reset()
    return torch.compile(fn, fullgraph=True)


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


# ---- pattern definitions ------------------------------------------------
# Each entry returns (fn_a, args_a, fn_b, args_b, restick_bytes, label).
# fn_a is the same computation without the restickify; fn_b is with it.
# Restickify bytes = expected size of the restickified tensor in bytes.


def pat_at_plus_x(S):
    """test_2arg_a_plus_xt: x + a.t(); restickify on a."""
    a_a = torch.randn((S, S), dtype=DTYPE)
    x_a = torch.randn((S, S), dtype=DTYPE)
    a_b = torch.randn((S, S), dtype=DTYPE)
    x_b = torch.randn((S, S), dtype=DTYPE)

    def fn_a(a, x):
        return a + x

    def fn_b(a, x):
        return a.t() + x

    return fn_a, (a_a, x_a), fn_b, (a_b, x_b), S * S * 2, f"at_plus_x  S={S}"


def pat_matmul_xt_y(S):
    """test_matmul_xt_y: matmul(x.t(), y); restickify on x. Use square."""
    x_a = torch.randn((S, S), dtype=DTYPE)
    y_a = torch.randn((S, S), dtype=DTYPE)
    x_b = torch.randn((S, S), dtype=DTYPE)
    y_b = torch.randn((S, S), dtype=DTYPE)

    def fn_a(x, y):
        return torch.matmul(x, y)

    def fn_b(x, y):
        return torch.matmul(x.t(), y)

    return fn_a, (x_a, y_a), fn_b, (x_b, y_b), S * S * 2, f"matmul_xt_y  S={S}"


def pat_matmul_x_yt(S):
    """test_matmul_x_yt: matmul(x, y.t()); restickify on y."""
    x_a = torch.randn((S, S), dtype=DTYPE)
    y_a = torch.randn((S, S), dtype=DTYPE)
    x_b = torch.randn((S, S), dtype=DTYPE)
    y_b = torch.randn((S, S), dtype=DTYPE)

    def fn_a(x, y):
        return torch.matmul(x, y)

    def fn_b(x, y):
        return torch.matmul(x, y.t())

    return fn_a, (x_a, y_a), fn_b, (x_b, y_b), S * S * 2, f"matmul_x_yt  S={S}"


def pat_matmul_then_add(S):
    """test_opt_matmul_then_adds: (a @ b) + c.t(); restickify on c."""
    a_a = torch.randn((S, S), dtype=DTYPE)
    b_a = torch.randn((S, S), dtype=DTYPE)
    c_a = torch.randn((S, S), dtype=DTYPE)
    a_b = torch.randn((S, S), dtype=DTYPE)
    b_b = torch.randn((S, S), dtype=DTYPE)
    c_b = torch.randn((S, S), dtype=DTYPE)

    def fn_a(a, b, c):
        return (a @ b) + c

    def fn_b(a, b, c):
        return (a @ b) + c.t()

    return (
        fn_a,
        (a_a, b_a, c_a),
        fn_b,
        (a_b, b_b, c_b),
        S * S * 2,
        f"(a@b)+c.t()  S={S}",
    )


def pat_chained_intermediate(S):
    """test_opt_chain_transposed_intermediate: (a.t() + b).t() + c.
    Optimal cost asserted as S*S; one restickify on the chained intermediate."""
    a_a = torch.randn((S, S), dtype=DTYPE)
    b_a = torch.randn((S, S), dtype=DTYPE)
    c_a = torch.randn((S, S), dtype=DTYPE)
    a_b = torch.randn((S, S), dtype=DTYPE)
    b_b = torch.randn((S, S), dtype=DTYPE)
    c_b = torch.randn((S, S), dtype=DTYPE)

    def fn_a(a, b, c):
        return (a + b) + c

    def fn_b(a, b, c):
        return (a.t() + b).t() + c

    return (
        fn_a,
        (a_a, b_a, c_a),
        fn_b,
        (a_b, b_b, c_b),
        S * S * 2,
        f"(a.t()+b).t()+c  S={S}",
    )


def pat_mtm(S):
    """matmul-transpose-matmul: (a@b).t() @ c; FUNDAMENTAL pattern from probe v3."""
    a_a = torch.randn((S, S), dtype=DTYPE)
    b_a = torch.randn((S, S), dtype=DTYPE)
    c_a = torch.randn((S, S), dtype=DTYPE)
    a_b = torch.randn((S, S), dtype=DTYPE)
    b_b = torch.randn((S, S), dtype=DTYPE)
    c_b = torch.randn((S, S), dtype=DTYPE)

    def fn_a(a, b, c):
        return (a @ b) @ c

    def fn_b(a, b, c):
        return (a @ b).t() @ c

    return (
        fn_a,
        (a_a, b_a, c_a),
        fn_b,
        (a_b, b_b, c_b),
        S * S * 2,
        f"(a@b).t()@c  S={S}",
    )


PATTERN_FNS = [
    pat_at_plus_x,
    pat_matmul_xt_y,
    pat_matmul_x_yt,
    pat_matmul_then_add,
    pat_chained_intermediate,
    pat_mtm,
]
SCALES = [256, 2048]


def main():
    captured_opfunc: Counter = Counter()
    current = {"label": None}
    orig_sdsc = ac.SpyreAsyncCompile.sdsc

    def wrapped_sdsc(self, kernel_name, specs):
        if current["label"] is not None:
            for spec in getattr(specs, "op_specs", specs) or []:
                opfunc = getattr(spec, "op", None)
                if opfunc is not None:
                    captured_opfunc[(current["label"], opfunc)] += 1
        return orig_sdsc(self, kernel_name, specs)

    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("lx_planning", True),
        ts_config.patch("allow_all_ops_in_lx_planning", True),
        ts_config.patch("sencores", 32),
        patch.object(ac.SpyreAsyncCompile, "sdsc", wrapped_sdsc),
    ]
    for p in patchers:
        p.__enter__()

    rows = []
    try:
        for pat_fn in PATTERN_FNS:
            for S in SCALES:
                fn_a, args_a, fn_b, args_b, restick_bytes, label = pat_fn(S)
                args_a = tuple(a.to(DEVICE) for a in args_a)
                args_b = tuple(a.to(DEVICE) for a in args_b)

                current["label"] = f"A {label}"
                try:
                    ca = fresh_compile(fn_a)
                    times_a = time_compiled(ca, args_a)
                except Exception as e:
                    print(
                        f"  {label} A failed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    current["label"] = None
                    continue

                current["label"] = f"B {label}"
                try:
                    cb = fresh_compile(fn_b)
                    times_b = time_compiled(cb, args_b)
                except Exception as e:
                    print(
                        f"  {label} B failed: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    current["label"] = None
                    continue
                current["label"] = None

                ta = statistics.median(times_a)
                tb = statistics.median(times_b)
                dt = tb - ta  # restickify cost (ms)
                rows.append({
                    "label": label,
                    "S": S,
                    "bytes": restick_bytes,
                    "ta": ta,
                    "tb": tb,
                    "dt": dt,
                })
                print(
                    f"  {label}: T_A={ta:.3f}  T_B={tb:.3f}  dt={dt:.3f} ms",
                    flush=True,
                )

    finally:
        for p in reversed(patchers):
            p.__exit__(None, None, None)

    # ----- report -----
    print()
    print("Pattern sweep -- guaranteed-restickify ops from test_restickify.py")
    print(f"  fabric ceilings (spec): HBM={HBM_SPEC/1e9:.0f}, "
          f"ring={RING_SPEC/1e9:.0f} GB/s | "
          f"effective: HBM={HBM_EFF/1e9:.0f}, ring={RING_EFF/1e9:.0f}")
    print(
        f"  speedup bracket: spec/spec={2*RING_SPEC/HBM_SPEC:.1f}x  "
        f"eff/spec={2*RING_SPEC/HBM_EFF:.1f}x  "
        f"spec/eff={2*RING_EFF/HBM_SPEC:.1f}x  "
        f"eff/eff={2*RING_EFF/HBM_EFF:.1f}x"
    )
    print()
    print(f"  {'pattern':<25} {'|B|MB':>6} {'dt(ms)':>7} "
          f"{'T_hbm_spec':>11} {'T_hbm_eff':>10} {'T_ring_spec':>12} "
          f"{'T_ring_eff':>11} {'BW_dt_eff':>10}")
    print("  " + "-" * 100)
    for r in rows:
        B = r["bytes"]
        mb = B / 1e6
        dt = r["dt"]
        # Theoretical predictions in ms:
        t_hbm_spec = 2 * B / HBM_SPEC * 1e3
        t_hbm_eff = 2 * B / HBM_EFF * 1e3
        t_ring_spec = B / RING_SPEC * 1e3
        t_ring_eff = B / RING_EFF * 1e3
        bw_dt_eff = (2 * B / (dt * 1e-3)) / 1e9 if dt > 0 else 0  # 2B/dt
        print(
            f"  {r['label']:<25} {mb:>6.2f} {dt:>7.3f} "
            f"{t_hbm_spec:>10.3f}m {t_hbm_eff:>9.3f}m "
            f"{t_ring_spec:>11.3f}m {t_ring_eff:>10.3f}m "
            f"{bw_dt_eff:>9.1f}G"
        )

    print()
    print("Columns:")
    print("  |B|MB        restickify bytes (from test optimal_cost)")
    print("  dt(ms)       measured T_B - T_A (restickify-isolated wall-clock)")
    print("  T_hbm_spec   theoretical at 166 GB/s HBM round-trip")
    print("  T_hbm_eff    theoretical at 107 GB/s effective HBM round-trip")
    print("  T_ring_spec  theoretical at 1328 GB/s ring all-to-all (spec)")
    print("  T_ring_eff   theoretical at  850 GB/s ring (64% efficiency)")
    print("  BW_dt_eff    2|B|/dt -- effective HBM bandwidth if HBM round-trip")

    print()
    print("== Op-func emissions (verifies restickify presence per pattern) ==")
    # For each B-graph label, list which op-funcs were emitted.
    by_label: dict = {}
    for (label, opfunc), n in captured_opfunc.items():
        by_label.setdefault(label, []).append((opfunc, n))
    for label in sorted(by_label):
        ops = sorted(by_label[label])
        names = ", ".join(f"{op}x{n}" for op, n in ops)
        marker = " <-- restickify" if any("Restickify" in op for op, _ in ops) else ""
        print(f"  {label:<35} {names}{marker}")


if __name__ == "__main__":
    main()
