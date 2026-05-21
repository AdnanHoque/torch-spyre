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

from collections import Counter

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, MutationLayoutSHOULDREMOVE, Operation
from torch._inductor.virtualized import V

from .ir import FixedTiledLayout
from .logging_utils import get_inductor_logger


logger = get_inductor_logger("copy_back_elision")


ELIDED_COPY_BACK_ATTR = "_spyre_writes_copy_back_target"


def _one_mem_dep(deps) -> MemoryDep | None:
    mem_deps = [dep for dep in deps if isinstance(dep, MemoryDep)]
    if len(mem_deps) != 1:
        return None
    return mem_deps[0]


def _same_host_layout(lhs, rhs) -> bool:
    return (
        lhs.device == rhs.device
        and lhs.dtype == rhs.dtype
        and tuple(lhs.size) == tuple(rhs.size)
        and tuple(lhs.stride) == tuple(rhs.stride)
        and lhs.offset == rhs.offset
    )


def _target_device_layout(target, name: str):
    layout = target.get_layout()
    if isinstance(layout, FixedTiledLayout):
        return layout.device_layout

    graph_input = V.graph.graph_inputs.get(name)
    layouts = getattr(graph_input, "layouts", None)
    if not layouts:
        return None
    return next(iter(layouts))


def elide_identity_copy_back(operations: list[Operation]) -> None:
    """Retarget safe copy-back epilogues to write directly into graph inputs.

    Functionalization represents ``out=`` updates on graph inputs as a pure
    producer followed by an identity ``copy_`` into the user-visible input
    buffer.  For Spyre that otherwise becomes an extra temp plus a standalone
    copy kernel.  This pass removes only the copy-back cases that are trivial
    aliases: the old destination is dead, the copy has identical indexing, and
    producer/destination host layouts already match.
    """

    writer_by_name: dict[str, Operation] = {}
    write_counts: Counter[str] = Counter()
    names_read: set[str] = set()

    for op in operations:
        read_writes = op.get_read_writes()
        for write in read_writes.writes:
            writer_by_name[write.name] = op
            write_counts[write.name] += 1
        for read in read_writes.reads:
            if isinstance(read, MemoryDep):
                names_read.add(read.name)

    graph_inputs = set(V.graph.graph_input_names)
    graph_outputs = set(V.graph.get_output_names())
    removed_ops: list[Operation] = []
    mutated_inputs: set[str] = set()

    for copy_op in operations:
        if not (
            isinstance(copy_op, ComputedBuffer)
            and isinstance(copy_op.layout, MutationLayoutSHOULDREMOVE)
        ):
            continue

        read_writes = copy_op.get_read_writes()
        source = _one_mem_dep(read_writes.reads)
        destination = _one_mem_dep(read_writes.writes)
        if source is None or destination is None:
            continue
        if source.index != destination.index:
            continue

        target = copy_op.layout.get_buffer()
        target_name = target.get_name()
        if target_name not in graph_inputs:
            continue
        if target_name in graph_outputs or source.name in graph_outputs:
            continue
        if target_name in names_read or target_name in mutated_inputs:
            continue

        producer = writer_by_name.get(source.name)
        if producer is None or producer is copy_op:
            continue
        if not isinstance(producer, ComputedBuffer):
            continue
        if isinstance(producer.layout, MutationLayoutSHOULDREMOVE):
            continue
        if write_counts[source.name] != 1:
            continue
        if not _same_host_layout(producer.get_layout(), target.get_layout()):
            continue

        target_stl = _target_device_layout(target, target_name)
        if target_stl is None:
            continue

        producer.layout = copy_op.layout
        producer.committed_stl = target_stl
        setattr(producer, ELIDED_COPY_BACK_ATTR, True)
        mutated_inputs.add(target_name)
        removed_ops.append(copy_op)
        logger.info(
            "elided copy-back %s; %s now writes %s",
            copy_op.get_name(),
            producer.get_name(),
            target_name,
        )

    for op in removed_ops:
        for write in op.get_read_writes().writes:
            V.graph.removed_buffers.add(write.name)
        operations.remove(op)
