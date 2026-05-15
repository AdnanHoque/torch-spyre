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

"""Classifier for restickify origins on the AIU.

Three verdicts distinguish where the cost of a restickify can be
addressed:

  HBM_LOAD     -- producer is a graph input. Data was in HBM anyway;
                  the ring cannot help because the source is off-chip.
  INCIDENTAL   -- producer and all consumers agree on the host-stride
                  partition of the buffer. The restickify only exists
                  because work_distribution gave it a non-matching split.
                  Fixable by aligning splits; ring is the wrong tool.
  FUNDAMENTAL  -- producer and at least one consumer want different
                  partitions. No split alignment bridges them; the
                  cross-core shuffle is structurally required. This is
                  the case `STCDPOpLx` would address.

Decision rule (matches tests/diag_restickify_lx_trace.py):

  if producer is graph input          -> HBM_LOAD
  elif all consumers' partitions match the producer's partition -> INCIDENTAL
  else                                 -> FUNDAMENTAL

Must be called after `work_distribution` has set `op.op_it_space_splits`,
i.e. as a hook on `CustomPreSchedulingPasses` that calls super() first.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    InputBuffer,
    Operation,
    StorageBox,
    TensorBox,
)

from .pass_utils import (
    apply_splits_from_index_coeff,
    iteration_space_from_op,
)


class RestickifyVerdict(str, Enum):
    HBM_LOAD = "HBM_LOAD"
    INCIDENTAL = "INCIDENTAL"
    FUNDAMENTAL = "FUNDAMENTAL"


_RESTICKIFY_OP_TARGET = torch.ops.spyre.restickify.default


def _is_restickify(op: Operation) -> bool:
    """True iff op's origins include the spyre.restickify FX op target."""
    origins = getattr(op, "origins", None) or []
    return any(getattr(n, "target", None) is _RESTICKIFY_OP_TARGET for n in origins)


def _is_graph_input(buf) -> bool:
    """True iff buf wraps an InputBuffer (graph input, HBM-resident at load)."""
    if isinstance(buf, TensorBox):
        buf = buf.data
    if isinstance(buf, StorageBox):
        buf = buf.data
    return isinstance(buf, InputBuffer)


def _indices(op):
    """(write_index, first_read_index) for op, or (None, None) on failure."""
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
    """Coefficient of sym in index, or 0 if absent, '?' on failure."""
    if index is None:
        return 0
    try:
        c = index.coeff(sym)
        return int(c) if c.is_number else c
    except Exception:
        return "?"


def _sym_splits(op) -> dict:
    """dict[sym -> split] for op after work_distribution, or {} if unavailable."""
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


def _partition_by_stride(op, index, sym_splits: dict) -> dict:
    """How `op` partitions a buffer reached via `index`, keyed by host stride.

    Returns dict[stride -> total_split_count] over the symbols with split > 1
    that participate in this index. Empty if no non-trivial splits, indicating
    the buffer is held whole (not core-partitioned) on this access.
    """
    part: dict = {}
    for sym, split in sym_splits.items():
        if split <= 1:
            continue
        c = _coeff(index, sym)
        if c not in (0, "?"):
            part[c] = part.get(c, 1) * split
    return part


def _read_index_for_buffer(op, buf_name: str):
    """Return the sympy index `op` uses to read `buf_name`, or None."""
    try:
        rw = op.get_read_writes()
    except Exception:
        return None
    for d in rw.reads:
        if isinstance(d, MemoryDep) and d.name == buf_name:
            return d.index
    return None


def _build_buffer_maps(operations: list[Operation]) -> tuple[dict, dict]:
    """Build (writes_by_buf, reads_by_buf) lookups over the operations list."""
    writes_by_buf: dict = {}
    reads_by_buf: dict = {}
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
    return writes_by_buf, reads_by_buf


def classify_inserted_restickify(
    op: Operation, operations: list[Operation]
) -> RestickifyVerdict:
    """Classify a restickify ComputedBuffer post-insert_restickify + work_distribution.

    Raises ValueError if op is not a restickify.
    """
    if not _is_restickify(op):
        raise ValueError(f"op {op.get_name()} is not a restickify")

    out_name = op.get_name()
    writes_by_buf, reads_by_buf = _build_buffer_maps(operations)

    # Locate producer via the restickify's input buffer.
    try:
        in_reads = [d for d in op.get_read_writes().reads if isinstance(d, MemoryDep)]
    except Exception:
        in_reads = []
    in_name = in_reads[0].name if in_reads else None
    producer = writes_by_buf.get(in_name) if in_name else None

    # HBM_LOAD: producer is absent from the ops list. This catches the
    # graph-input case (the input wasn't written by any op in this graph)
    # plus other "producer not in ops" edge cases that are equally
    # ring-ineligible for the same reason -- data sourced from HBM.
    if producer is None:
        return RestickifyVerdict.HBM_LOAD

    # Build producer's host-stride partition of the buffer.
    p_w, _ = _indices(producer)
    prod_part = _partition_by_stride(producer, p_w, _sym_splits(producer))

    # Build each consumer's host-stride partition (as it reads the restickify's
    # output buffer).
    cons_parts = []
    for c in reads_by_buf.get(out_name, []):
        c_idx = _read_index_for_buffer(c, out_name)
        c_part = _partition_by_stride(c, c_idx, _sym_splits(c))
        cons_parts.append(c_part)

    if not cons_parts:
        # No consumers -- treat as INCIDENTAL; no ring benefit either way.
        return RestickifyVerdict.INCIDENTAL

    if all(cp == prod_part for cp in cons_parts):
        return RestickifyVerdict.INCIDENTAL
    return RestickifyVerdict.FUNDAMENTAL


def classify_all_restickifies(
    operations: list[Operation],
) -> dict[str, RestickifyVerdict]:
    """Return {restickify_op_name: verdict} for every restickify in `operations`.

    Walks the operations list once; must be called after the full
    `CustomPreSchedulingPasses` pipeline has run (so that `op_it_space_splits`
    is populated by `work_distribution`).
    """
    out: dict[str, RestickifyVerdict] = {}
    for op in operations:
        if isinstance(op, ComputedBuffer) and _is_restickify(op):
            out[op.get_name()] = classify_inserted_restickify(op, operations)
    return out


def is_ring_eligible_producer(producer_buffer) -> bool:
    """Optimizer-time predicate: is the producer NOT a graph input?

    Useful at `optimize_restickify` time where `op.op_it_space_splits` is not
    yet populated (work_distribution hasn't run), so the full FUNDAMENTAL vs
    INCIDENTAL distinction cannot be made. The graph-input check is the only
    part of the classifier that can be answered pre-split.
    """
    return not _is_graph_input(producer_buffer)


def annotate_restickify_verdicts(operations: list[Operation]) -> None:
    """Attach a `_spyre_restickify_verdict` attribute to every restickify
    ComputedBuffer in `operations`, set to its FUNDAMENTAL/INCIDENTAL/HBM_LOAD
    verdict.

    Intended to run as a pre-scheduling pass step *after* `work_distribution`
    (so splits are populated) and *before* codegen (so spyre_kernel.store can
    read the verdict back and decide whether to swap RESTICKIFY_OP for
    STCDPOpLx under `config.emit_stcdp_oplx`). Uses object.__setattr__ to
    accommodate buffer classes that may be frozen.
    """
    verdicts = classify_all_restickifies(operations)
    for op in operations:
        if isinstance(op, ComputedBuffer) and _is_restickify(op):
            v = verdicts.get(op.get_name())
            if v is not None:
                try:
                    op._spyre_restickify_verdict = v  # type: ignore[attr-defined]
                except AttributeError:
                    object.__setattr__(op, "_spyre_restickify_verdict", v)
