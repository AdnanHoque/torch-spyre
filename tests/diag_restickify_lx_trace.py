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

"""Diagnostic probe: are restickify cross-core movements *fundamental* or
*incidental* under LX_PLANNING=1 at sencores=32?

Model. A restickify lowers to a Pointwise whose read index == write index
(`loader(index)`); the relayout lives entirely in the device SpyreTensorLayout,
and the input/output buffers share one host layout. So a restickify induces the
*same* host-stride partition on both its input and its output. Therefore both
edges can be cross-core-free iff producer and consumer partition the buffer the
same way -- then work_distribution can give the restickify that partition too.

Per restickify (sc=32) we capture, by host stride:
  prod_part : how the producer partitioned the buffer across cores
  rs_part   : how the restickify (currently) partitions it
  cons_part : how each consumer reads it
Verdict:
  HBM-LOAD    -> producer is a graph input (HBM anyway; not a ring case).
  INCIDENTAL  -> producer & consumers agree; work_distribution just picked a
                 non-matching rs_part. Fixable by split alignment; no ring.
  FUNDAMENTAL -> producer & consumer partition differently; no restickify split
                 bridges them. Needs an on-chip cross-core shuffle (STCDPOpLx).
"""

import os
import sys

# torch_spyre is imported from the repo-root source tree (not pip-installed), and
# its device-backend autoload entrypoint hits a circular import. Disable torch's
# autoload and register the backend explicitly via _autoload() below instead.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch

import torch
import torch_spyre

torch_spyre._autoload()  # registers the "spyre" device

from torch._inductor.virtualized import V
from torch._inductor import config as t_inductor_config
from torch._inductor.ir import ComputedBuffer
from torch._inductor.dependencies import MemoryDep

from torch_spyre._inductor.passes import CustomPreSchedulingPasses
from torch_spyre._inductor import passes
from torch_spyre._inductor import config as ts_config
from torch_spyre._inductor.pass_utils import (
    apply_splits_from_index_coeff,
    iteration_space_from_op,
)

RESTICKIFY = torch.ops.spyre.restickify.default
_RESULTS: dict = {}


def _is_restickify(op) -> bool:
    origins = getattr(op, "origins", None) or []
    return any(getattr(n, "target", None) is RESTICKIFY for n in origins)


def _location(buf) -> str:
    alloc = getattr(buf.get_layout(), "allocation", None) or {}
    return "LX" if "lx" in alloc else "HBM"


def _op_kind(op) -> str:
    if op is None:
        return "<graph_input>"
    if _is_restickify(op):
        return "restickify"
    data = getattr(op, "data", None)
    rt = getattr(data, "reduction_type", None)
    if rt:
        return f"reduction:{rt}"
    return type(data).__name__ if data is not None else type(op).__name__


def _indices(op):
    """(write_index, first_read_index) for an op, or (None, None)."""
    try:
        rw = op.get_read_writes()
    except Exception:
        return None, None
    writes = [d for d in rw.writes if isinstance(d, MemoryDep)]
    reads = [d for d in rw.reads if isinstance(d, MemoryDep)]
    w = writes[0].index if writes else None
    r = reads[0].index if reads else w
    return w, r


def _coeff(index, sym):
    if index is None:
        return 0
    try:
        c = index.coeff(sym)
        return int(c) if c.is_number else c
    except Exception:
        return "?"


def _sym_splits(op):
    """dict[sym -> split] for an op after work_distribution, or {}."""
    raw = getattr(op, "op_it_space_splits", None)
    if not raw:
        return {}
    w, r = _indices(op)
    if w is None:
        return {}
    try:
        it_space = iteration_space_from_op(op)
        return apply_splits_from_index_coeff(raw, w, r, it_space)
    except Exception:
        return {}


def _partition_by_stride(op, index, sym_splits):
    """How `op` partitions the buffer reached by `index`, keyed by host stride."""
    part = {}
    for sym, split in sym_splits.items():
        if split <= 1:
            continue
        c = _coeff(index, sym)
        if c not in (0, "?"):
            part[c] = part.get(c, 1) * split
    return part


def _read_index_for_buffer(op, buf_name):
    try:
        rw = op.get_read_writes()
    except Exception:
        return None
    for d in rw.reads:
        if isinstance(d, MemoryDep) and d.name == buf_name:
            return d.index
    return None


def _analyze(operations) -> list:
    writes_by_buf, reads_by_buf = {}, {}
    for op in operations:
        try:
            rw = op.get_read_writes()
        except Exception:
            continue
        for dep in rw.writes:
            if isinstance(dep, MemoryDep):
                writes_by_buf[dep.name] = op
        for dep in rw.reads:
            if isinstance(dep, MemoryDep):
                reads_by_buf.setdefault(dep.name, []).append(op)

    rows = []
    for op in operations:
        if not isinstance(op, ComputedBuffer) or not _is_restickify(op):
            continue
        out_name = op.get_name()
        rs_w, rs_r = _indices(op)
        rs_splits = _sym_splits(op)
        try:
            rs_it = iteration_space_from_op(op)
        except Exception:
            rs_it = {}

        var_rows = []
        for sym, rng in rs_it.items():
            try:
                rng_i = int(rng)
            except Exception:
                rng_i = rng
            var_rows.append(
                {
                    "sym": str(sym),
                    "range": rng_i,
                    "stride": _coeff(rs_w, sym),
                    "split": rs_splits.get(sym, 1),
                }
            )

        in_name = None
        try:
            in_reads = [
                d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)
            ]
            in_name = in_reads[0].name if in_reads else None
        except Exception:
            pass
        producer = writes_by_buf.get(in_name) if in_name else None

        rs_in_part = _partition_by_stride(op, rs_r, rs_splits)
        rs_out_part = _partition_by_stride(op, rs_w, rs_splits)

        prod_part = {}
        if producer is not None:
            p_w, _ = _indices(producer)
            prod_part = _partition_by_stride(producer, p_w, _sym_splits(producer))

        cons_rows = []
        for c in reads_by_buf.get(out_name, []):
            c_idx = _read_index_for_buffer(c, out_name)
            c_part = _partition_by_stride(c, c_idx, _sym_splits(c))
            cons_rows.append(
                {
                    "kind": _op_kind(c),
                    "part": c_part,
                    "aligned": c_part == rs_out_part,
                }
            )

        # Verdict. A restickify induces one host-stride partition on both ends,
        # so both edges are alignable iff producer and all consumers agree.
        cons_parts = [c["part"] for c in cons_rows]
        if producer is None:
            verdict = "HBM-LOAD (producer is a graph input -- not a ring case)"
        elif not cons_parts:
            verdict = "n/a (no consumer)"
        elif all(cp == prod_part for cp in cons_parts):
            verdict = "INCIDENTAL (prod==cons; work_distribution can align)"
        else:
            verdict = "FUNDAMENTAL (prod != cons; needs cross-core shuffle)"

        rows.append(
            {
                "restickify": out_name,
                "in_buf": in_name,
                "in_loc": _location(V.graph.get_buffer(in_name))
                if in_name
                else "?",
                "out_loc": _location(op),
                "in_producer": _op_kind(producer),
                "vars": var_rows,
                "verdict": verdict,
                "rs_in_part": rs_in_part,
                "prod_part": prod_part,
                "in_aligned": (producer is not None and rs_in_part == prod_part),
                "rs_out_part": rs_out_part,
                "consumers": cons_rows,
            }
        )
    return rows


def _probe_class(label):
    class _Probe(CustomPreSchedulingPasses):
        def __call__(self, operations):
            super().__call__(operations)
            _RESULTS[label] = _analyze(operations)

    return _Probe


def run_case(label, fn, args, sencores):
    key = f"{label}@sc{sencores}"
    patchers = [
        t_inductor_config.patch("force_disable_caches", True),
        ts_config.patch("lx_planning", True),
        ts_config.patch("allow_all_ops_in_lx_planning", True),
        ts_config.patch("sencores", sencores),
        patch.object(passes, "CustomPreSchedulingPasses", _probe_class(key)),
    ]
    for p in patchers:
        p.__enter__()
    torch.compiler.reset()
    try:
        compiled = torch.compile(fn, fullgraph=True)
        try:
            compiled(*args)
        except Exception as e:  # passes already ran; device exec is not needed
            print(f"[{key}] post-compile exec raised (ok): {type(e).__name__}")
    except Exception as e:
        print(f"[{key}] COMPILE FAILED: {type(e).__name__}: {e}")
    finally:
        torch.compiler.reset()
        for p in reversed(patchers):
            p.__exit__(None, None, None)


def dev(*shape, dtype=torch.float16):
    return torch.rand(shape, dtype=dtype, device="spyre")


# ---- cases -----------------------------------------------------------------

def case_linear_weight_restickify():
    x = dev(1, 4096)
    W = dev(6144, 4096)
    return "linear_x_Wt_decode", (lambda x, W: x @ W.t()), (x, W)


def case_transposed_computed_intermediate():
    S = 128
    a, b, c = dev(S, S), dev(S, S), dev(S, S)
    return (
        "transposed_computed_intermediate",
        (lambda a, b, c: (a + b).t() + c),
        (a, b, c),
    )


def case_matmul_then_transposed_add():
    S = 128
    a, b, c = dev(S, S), dev(S, S), dev(S, S)
    return (
        "matmul_then_transposed_add",
        (lambda a, b, c: (a @ b) + c.t()),
        (a, b, c),
    )


def case_chained_matmul_transposed():
    M, K, N = 256, 512, 512
    a, b, c = dev(M, K), dev(K, N), dev(N, M)
    return (
        "chained_matmul_transposed",
        (lambda a, b, c: (a @ b).t() + c),
        (a, b, c),
    )


def case_matmul_transposed_matmul():
    # Both producer and consumer are matmuls -> the cleanest "both ends on-chip"
    # test. (a@b) feeds, transposed, into another matmul.
    S = 256
    a, b, c = dev(S, S), dev(S, S), dev(S, S)
    return (
        "matmul_transposed_matmul",
        (lambda a, b, c: (a @ b).t() @ c),
        (a, b, c),
    )


CASES = [
    case_linear_weight_restickify,
    case_transposed_computed_intermediate,
    case_matmul_then_transposed_add,
    case_chained_matmul_transposed,
    case_matmul_transposed_matmul,
]


def _fmt_part(p):
    if not p:
        return "{}"
    return "{" + ", ".join(
        f"s{k}:x{v}" for k, v in sorted(p.items(), key=lambda kv: str(kv[0]))
    ) + "}"


def main():
    for make in CASES:
        label, fn, args = make()
        for sc in (1, 32):
            print(f"=== compiling: {label} @ sencores={sc} ===")
            run_case(label, fn, args, sc)

    print("\n\n############  RESTICKIFY CROSS-CORE STRUCTURE (sc=32)  ############")
    tally = {}
    for key, rows in _RESULTS.items():
        if not key.endswith("@sc32"):
            continue
        print(f"\n--- {key} : {len(rows)} restickify ---")
        if not rows:
            print("  (no restickify ops inserted)")
        for r in rows:
            tag = r["verdict"].split()[0]
            tally[tag] = tally.get(tag, 0) + 1
            print(
                f"  restickify {r['restickify']}: "
                f"producer={r['in_producer']}[{r['in_buf']} {r['in_loc']}] "
                f"out={r['out_loc']}"
            )
            print(f"    VERDICT: {r['verdict']}")
            print(
                f"    {'var':>5} {'range':>7} {'host_stride':>12} {'split':>6}"
            )
            for v in r["vars"]:
                mark = "  <-- split" if v["split"] > 1 else ""
                print(
                    f"    {v['sym']:>5} {str(v['range']):>7} "
                    f"{str(v['stride']):>12} {v['split']:>6}{mark}"
                )
            ia = "ALIGNED" if r["in_aligned"] else "MISMATCH"
            print(
                f"    input  edge: producer {_fmt_part(r['prod_part'])} | "
                f"restickify {_fmt_part(r['rs_in_part'])}  -> {ia}"
            )
            for c in r["consumers"]:
                oa = "ALIGNED" if c["aligned"] else "MISMATCH"
                print(
                    f"    output edge: restickify {_fmt_part(r['rs_out_part'])} | "
                    f"{c['kind']} {_fmt_part(c['part'])}  -> {oa}"
                )
    print(f"\n  TALLY: {tally}")
    print("###################################################################")


if __name__ == "__main__":
    main()
