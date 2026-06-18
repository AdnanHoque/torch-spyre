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
"""Work-division co-assignment — element-wise consumers inherit the matmul split.

After the matmul cost model picks ``(m4,n8)``, propagate that split to the
downstream element-wise (``Pointwise``) consumers so the matmul->pointwise edge is
**same-division same-core**: each consumer core reads exactly the tile its own
core produced. Value-correct (element-wise ops are split-agnostic), no data-op, no
dxp gate. This is Stage 1 (the split propagation -> same-shard HBM hand-off, no
cross-core re-read). Stage 2 would add ``onchip_softmax_chain.apply_lx_flip`` to
make the same-core edge LX-resident.

Installed as a monkeypatch on ``passes.cost_model_matmul_division`` (the name the
pass orchestration resolves): we call the original (matmuls get their split), then
BFS the element-wise consumer chain and commit the mapped split to each, returning
them as preassigned so ``work_distribution`` honors it.
"""

import torch_spyre._inductor.passes as passes
import torch_spyre._inductor.work_division as wd
from torch._inductor.ir import Pointwise
from torch_spyre._inductor.pass_utils import apply_splits_from_index_coeff


def _bufs(op, kind):
    return {d.name for d in getattr(op.get_read_writes(), kind)}


def _recover_split(op):
    """Reconstruct the {symbol: count} split committed on op + its iter space."""
    it = wd.iteration_space_from_op(op)
    rw = op.get_read_writes()
    wi = next(iter(rw.writes)).index
    ri = next((d.index for d in rw.reads), wi)
    split = apply_splits_from_index_coeff(op.op_it_space_splits, wi, ri, it)
    return {s: v for s, v in split.items() if v > 1}, it


def _map_split_by_extent(src_split, src_it, dst_it):
    """Map a producer split onto a consumer iter space by matching dim extents."""
    out, used = {}, set()
    for ssym, cnt in src_split.items():
        ext = wd.concretize_expr(src_it[ssym])
        for dsym, dext in dst_it.items():
            if dsym in used:
                continue
            if wd.concretize_expr(dext) == ext:
                out[dsym] = cnt
                used.add(dsym)
                break
    return out


def _commit_split(op, split):
    rw = op.get_read_writes()
    args = wd.get_mem_deps_from_rw(rw)
    _, output_td = wd.collect_tensor_deps(op, args)
    wd.apply_splits(op, split, output_td)


def apply_coassign():
    """Monkeypatch cost_model_matmul_division to co-assign element-wise consumers."""
    orig = passes.cost_model_matmul_division

    def _patched(graph):
        mm_ops = orig(graph)
        if not mm_ops:
            return mm_ops
        ops = list(wd._iter_computed_buffers(graph.operations))
        seen = {id(m) for m in mm_ops}
        extra = []
        # BFS: matmul -> element-wise consumers -> their element-wise consumers.
        frontier = []
        for m in mm_ops:
            if getattr(m, "op_it_space_splits", None):
                frontier.append((m, _recover_split(m)))
        for prod, (psplit, pit) in frontier:
            if not psplit:
                continue
            pbufs = _bufs(prod, "writes")
            for op in ops:
                if id(op) in seen or not isinstance(op.data, Pointwise):
                    continue
                if not (_bufs(op, "reads") & pbufs):
                    continue
                dit = wd.iteration_space_from_op(op)
                csplit = _map_split_by_extent(psplit, pit, dit)
                if not csplit:
                    continue
                _commit_split(op, csplit)
                seen.add(id(op))
                extra.append(op)
                frontier.append((op, (csplit, dit)))
                print(f"[COASSIGN] {op.get_name()} <- {csplit}", flush=True)
        return mm_ops + extra

    passes.cost_model_matmul_division = _patched
    print("[COASSIGN] patched passes.cost_model_matmul_division", flush=True)
