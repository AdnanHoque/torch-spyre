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

from collections import defaultdict

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import (
    ComputedBuffer,
    MutationLayoutSHOULDREMOVE,
    Operation,
)
from torch._inductor.virtualized import V

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("elide_copy_to_input")


def _single_mem_dep(deps) -> "MemoryDep | None":
    mem = [d for d in deps if isinstance(d, MemoryDep)]
    return mem[0] if len(mem) == 1 else None


def _host_layout_matches(a, b) -> bool:
    """True if two host FixedLayouts are element-for-element identical."""
    return (
        a.device == b.device
        and a.dtype == b.dtype
        and tuple(a.size) == tuple(b.size)
        and tuple(a.stride) == tuple(b.stride)
        and a.offset == b.offset
    )


def elide_copy_to_input(operations: list[Operation]) -> None:
    """Fold an ``out=``/``copy_`` epilogue into its producer.

    ``torch.mm(x, y, out=z)`` (z a graph input) functionalises to
    ``mm = aten.mm(x, y); aten.copy_(z, mm)``.  Inductor's ``mutate_to`` refuses
    to alias an op's output onto a graph-input buffer, so it materialises ``mm``
    into a fresh buffer and emits the ``copy_`` as a standalone identity op.
    That fresh buffer is dead weight: the producer could write straight into the
    mutated input.

    This pass detects ``copy_(graph_input, producer_output)`` epilogues that are
    pure identities and retargets the producer to write into the graph input
    directly, then deletes the copy.  Other readers of the producer's output are
    handled by the scheduler's mutation rename (producer buffer -> target).

    Must run after ``propagate_spyre_tensor_layouts`` (so the producer keeps its
    ``layouts``/``restick_cost_fn`` for input restickify analysis) and before
    ``finalize_layouts``.  The retargeted producer is left with a
    ``MutationLayoutSHOULDREMOVE`` layout; ``propagate_mutation_layouts``
    resolves it to the target's concrete layout after scheduler init.
    """
    producer: dict[str, Operation] = {}
    write_count: dict[str, int] = defaultdict(int)
    read_names: set[str] = set()
    for op in operations:
        rw = op.get_read_writes()
        for dep in rw.writes:
            producer[dep.name] = op
            write_count[dep.name] += 1
        for dep in rw.reads:
            if isinstance(dep, MemoryDep):
                read_names.add(dep.name)

    output_names = set(V.graph.get_output_names())
    input_names = set(V.graph.graph_input_names)

    removed: list[Operation] = []
    claimed_targets: set[str] = set()
    for op in operations:
        if not (
            isinstance(op, ComputedBuffer)
            and isinstance(op.layout, MutationLayoutSHOULDREMOVE)
        ):
            continue

        rw = op.get_read_writes()
        src_dep = _single_mem_dep(rw.reads)
        dst_dep = _single_mem_dep(rw.writes)
        if src_dep is None or dst_dep is None:
            continue
        # Identity copy only: same indexing in and out (no slice/broadcast/restickify).
        if src_dep.index != dst_dep.index:
            continue

        target = op.layout.get_buffer()
        target_name = target.get_name()
        # Target must be a graph input whose prior value is dead, and which is
        # not itself a graph output (returning it would observe the alias).
        if target_name not in input_names or target_name in output_names:
            continue
        if target_name in read_names:
            continue
        # Only one producer may be retargeted onto a given input.
        if target_name in claimed_targets:
            continue

        prod = producer.get(src_dep.name)
        if prod is None or prod is op:
            continue
        # Producer must own its output outright and not already mutate something.
        if not isinstance(prod, ComputedBuffer):
            continue
        if isinstance(prod.layout, MutationLayoutSHOULDREMOVE):
            continue
        if write_count[src_dep.name] != 1:
            continue
        # The producer's output must not be a graph output: the scheduler will
        # rename it to the target, which would corrupt a returned buffer.
        if src_dep.name in output_names:
            continue
        # Identity requires the producer's output and the target to have the
        # exact same host layout, otherwise the copy was doing a real relayout.
        if not _host_layout_matches(prod.get_layout(), target.get_layout()):
            continue

        # Retarget: producer now writes into the graph input; drop the copy.
        prod.layout = op.layout
        # optimize_restickify won't commit an STL for a mutation op, but
        # finalize_layouts needs one to derive pointwise input requirements
        # (FixedInOutNode/matmul ignores it; AllSameNode/pointwise uses it).
        # The producer writes into the target, so its committed STL is the
        # target's STL (set on the graph input by propagate_spyre_tensor_layouts).
        target_tb = V.graph.graph_inputs[target_name]
        prod.committed_stl = next(iter(target_tb.layouts))
        # Mark so finalize_layouts' pure-copy mutation handling skips this op:
        # it is a real compute op, not a copy whose input must match the target.
        prod._spyre_elided_copy_producer = True
        claimed_targets.add(target_name)
        removed.append(op)
        logger.info(
            "elided copy_ %s: %s now writes graph input %s directly",
            op.get_name(),
            prod.get_name(),
            target_name,
        )

    for op in removed:
        for dep in op.get_read_writes().writes:
            V.graph.removed_buffers.add(dep.name)
        operations.remove(op)
