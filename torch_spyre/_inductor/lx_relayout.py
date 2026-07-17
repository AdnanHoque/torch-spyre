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

"""Plan LX relayouts from producer and consumer per-core views.

The regular LX planner handles same-core scratchpad persistence. This module
finds edges where a producer and consumer use different per-core views of the
same tensor. Accepted edges are materialized as an explicit, frontend-allocated
S1 -> SHUFFLE -> S2 sequence.
"""

from __future__ import annotations

import dataclasses

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise

from torch_spyre._inductor import config
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    op_read_writes,
)
from torch_spyre._inductor.scratchpad.utils import _op_num_cores, _op_short_name

LX_RELAYOUT_ATTR = "_spyre_lx_relayout_inputs"
LX_RELAYOUT_DESTINATION_PREFIX = "__spyre_lx_relayout_destination__"


def make_lx_relayout_destination_name(source_name: str) -> str:
    return f"{LX_RELAYOUT_DESTINATION_PREFIX}:{source_name}"


def is_lx_relayout_destination(name: str) -> bool:
    return name.startswith(f"{LX_RELAYOUT_DESTINATION_PREFIX}:")


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    """Names and exact per-core geometry for one LX shuffle."""

    source_name: str
    consumer_name: str
    source_core_id_to_device_slice: dict[str, dict[str, int]]
    destination_core_id_to_device_slice: dict[str, dict[str, int]]
    source_device_dim_splits: dict[str, int]
    destination_device_dim_splits: dict[str, int]
    destination_size_ratio: int
    destination_lx_address: int | None = None

    @property
    def destination_name(self) -> str:
        return make_lx_relayout_destination_name(self.source_name)


def _intervals_overlap(
    lhs_slot: int, lhs_split: int, rhs_slot: int, rhs_split: int
) -> bool:
    return lhs_slot * rhs_split < (rhs_slot + 1) * lhs_split and (
        rhs_slot * lhs_split < (lhs_slot + 1) * rhs_split
    )


def _destination_size_ratio(
    producer_map: dict[str, dict[str, int]],
    producer_splits: dict[str, int],
    consumer_map: dict[str, dict[str, int]],
    consumer_splits: dict[str, int],
) -> int | None:
    """Return S2:S1 per-core size ratio for exact uniform shuffle geometry."""

    def slice_count(core_map: dict[str, dict[str, int]]) -> int:
        return len(
            {
                tuple(sorted((dim, int(slot)) for dim, slot in per_core.items()))
                for per_core in core_map.values()
            }
        )

    fanout = {core: 0 for core in producer_map}
    fanin = {core: 0 for core in consumer_map}
    transfer_count = 0
    dims = set(producer_splits) | set(consumer_splits)
    for producer_core, producer_slice in producer_map.items():
        for consumer_core, consumer_slice in consumer_map.items():
            if all(
                _intervals_overlap(
                    int(producer_slice.get(dim, 0)),
                    int(producer_splits.get(dim, 1)),
                    int(consumer_slice.get(dim, 0)),
                    int(consumer_splits.get(dim, 1)),
                )
                for dim in dims
            ):
                fanout[producer_core] += 1
                fanin[consumer_core] += 1
                transfer_count += 1

    fanout_values = set(fanout.values())
    fanin_values = set(fanin.values())
    uniform = len(fanout_values) == 1 and len(fanin_values) == 1
    max_fanout = max(fanout_values, default=0)
    max_fanin = max(fanin_values, default=0)
    covers_all_cores = (
        min(fanout_values, default=0) > 0 and min(fanin_values, default=0) > 0
    )
    producer_is_partitioned = slice_count(producer_map) == len(producer_map)
    consumer_is_partitioned = slice_count(consumer_map) == len(consumer_map)
    if transfer_count == 0 or not covers_all_cores or not uniform:
        return None
    if (
        producer_is_partitioned
        and consumer_is_partitioned
        and max_fanout <= 1
        and max_fanin <= 1
    ):
        return 1
    if (
        producer_is_partitioned
        and not consumer_is_partitioned
        and max_fanout > 1
        and max_fanin > 1
    ):
        return max_fanin
    return None


def _single_write_dep(op: ComputedBuffer, buf_name: str) -> MemoryDep | None:
    matches = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and dep.name == buf_name
    ]
    return matches[0] if len(matches) == 1 else None


def _operations_by_name(graph: GraphLowering) -> dict[str, Operation]:
    """Index the current graph, including operations inserted by graph editing."""

    return {op.get_name(): op for op in graph.operations}


def _restickify_reads_computed_input(
    operations: dict[str, Operation], op: Operation
) -> bool:
    if _op_short_name(op) != "restickify":
        return False
    return any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def _matmul_operand_source_good_for_lx_relayout(
    operations: dict[str, Operation], op: Operation
) -> bool:
    """Exclude graph-input and weight restickifies from activation relayout."""

    return _op_short_name(op) != "restickify" or _restickify_reads_computed_input(
        operations, op
    )


def _core_id_to_device_slice(
    view: PerCoreView,
    core_count: int,
) -> dict[str, dict[str, int]] | None:
    """Return ownership as ``core -> device-dim -> slice-index``."""

    core_id = sympy.Symbol("core_id")
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    split_dims = {int(dim): int(split) for dim, split in view.work_slice_dims}
    result: dict[str, dict[str, int]] = {}

    for core in range(core_count):
        per_core: dict[str, int] = {}
        for dim, split in split_dims.items():
            expr = sympy.sympify(expr_by_dim.get(dim, 0))
            slot = sympy.simplify(expr.subs(core_id, core))
            if getattr(slot, "free_symbols", None):
                return None
            try:
                slot_int = int(slot)
            except TypeError:
                return None
            if slot_int < 0 or slot_int >= split:
                return None
            per_core[str(dim)] = slot_int
        result[str(core)] = per_core

    return result


def _work_slice_dims(view: PerCoreView) -> dict[str, int]:
    return {str(int(dim)): int(split) for dim, split in view.work_slice_dims}


def _dense_view(
    core_map: dict[str, dict[str, int]],
    splits: dict[str, int],
    dims: set[str],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    ordered_dims = sorted(dims, key=int)
    dense_map = {
        core: {dim: int(per_core.get(dim, 0)) for dim in ordered_dims}
        for core, per_core in core_map.items()
    }
    dense_splits = {dim: int(splits.get(dim, 1)) for dim in ordered_dims}
    return dense_map, dense_splits


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_ATTR):
            delattr(op, LX_RELAYOUT_ATTR)


def collect_lx_relayout_plans(
    graph: GraphLowering, cache: dict | None = None
) -> list[LXRelayoutPlan]:
    """Plan bounded LX relayouts from producer and consumer coordinates.

    V1 only records movement for single-writer intermediate tensors whose
    producer output is final (not K-split partials) and whose producer and
    consumer PerCoreViews differ.  Same-view edges remain owned by the existing
    LX planner.
    """

    if not config.lx_planner_relayout:
        return []

    operations = _operations_by_name(graph)
    read_counts: dict[str, int] = {}
    for op in graph.operations:
        for dep in op_read_writes(op).reads:
            if isinstance(dep, MemoryDep):
                read_counts[dep.name] = read_counts.get(dep.name, 0) + 1
    planned: dict[str, list[LXRelayoutPlan]] = {}

    for consumer in graph.operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        is_matmul_consumer = _is_matmul_op(consumer)
        if not is_matmul_consumer and not isinstance(consumer.data, Pointwise):
            continue
        reads = (
            dep for dep in op_read_writes(consumer).reads if isinstance(dep, MemoryDep)
        )
        for read_index, dep in enumerate(reads):
            producer = operations.get(dep.name)
            if not isinstance(producer, ComputedBuffer) or producer is consumer:
                continue

            write_dep = _single_write_dep(producer, dep.name)
            if write_dep is None:
                continue

            producer_view, producer_has_partial, producer_representable = (
                _per_core_view_on_buf(producer, write_dep, dep.name, cache)
            )
            if producer_has_partial or not producer_representable:
                continue

            consumer_view, consumer_has_partial, consumer_representable = (
                _per_core_view_on_buf(consumer, dep, dep.name, cache)
            )
            if consumer_has_partial or not consumer_representable:
                continue
            if producer_view == consumer_view:
                continue

            producer_core_count = _op_num_cores(producer)
            consumer_core_count = _op_num_cores(consumer)
            producer_core_slices = _core_id_to_device_slice(
                producer_view, producer_core_count
            )
            if producer_core_slices is None:
                continue

            consumer_work_slice_dims = _work_slice_dims(consumer_view)
            consumer_core_slices = _core_id_to_device_slice(
                consumer_view, consumer_core_count
            )
            if consumer_core_slices is None:
                continue

            producer_work_slice_dims = _work_slice_dims(producer_view)
            relayout_dims = set(producer_work_slice_dims) | set(
                consumer_work_slice_dims
            )
            producer_core_slices, producer_work_slice_dims = _dense_view(
                producer_core_slices, producer_work_slice_dims, relayout_dims
            )
            consumer_core_slices, consumer_work_slice_dims = _dense_view(
                consumer_core_slices, consumer_work_slice_dims, relayout_dims
            )
            destination_size_ratio = _destination_size_ratio(
                producer_core_slices,
                producer_work_slice_dims,
                consumer_core_slices,
                consumer_work_slice_dims,
            )
            if destination_size_ratio is None:
                continue

            if is_matmul_consumer and (
                not _matmul_operand_source_good_for_lx_relayout(operations, producer)
                or read_index not in (0, 1)
            ):
                continue

            plan = LXRelayoutPlan(
                source_name=dep.name,
                consumer_name=consumer.get_name(),
                source_core_id_to_device_slice=producer_core_slices,
                destination_core_id_to_device_slice=consumer_core_slices,
                source_device_dim_splits=producer_work_slice_dims,
                destination_device_dim_splits=consumer_work_slice_dims,
                destination_size_ratio=destination_size_ratio,
            )
            planned.setdefault(plan.source_name, []).append(plan)

    # V1 deliberately materializes one consumer view per source. Sharing one
    # S2 allocation across independently scheduled consumers requires a wider
    # lifetime and scheduling contract.
    return [
        plans[0]
        for source_name, plans in planned.items()
        if len(plans) == 1 and read_counts.get(source_name, 0) == 1
    ]


def record_lx_relayout_plan(graph: GraphLowering, plan: LXRelayoutPlan) -> None:
    """Stamp a relayout plan after the source buffer is placed in LX."""

    consumer = _operations_by_name(graph).get(plan.consumer_name)
    if consumer is None:
        return
    plans = getattr(consumer, LX_RELAYOUT_ATTR, None)
    if not isinstance(plans, dict):
        plans = {}
        setattr(consumer, LX_RELAYOUT_ATTR, plans)
    plans[plan.source_name] = plan
