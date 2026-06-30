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

"""LX relayout planning metadata for Deeptools dl-dsc relayout insertion.

The regular LX planner handles same-core scratchpad persistence.  This module
classifies edges where a producer and consumer use different per-core views of
the same LX-resident tensor.  It does not emit movement operations.  Instead,
it records the producer tensor distribution so SDSC codegen can populate
``allocateCoordinates_.coreIdToWkSlice_`` on the consumer input; Deeptools then
derives and lowers the physical movement.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation

from torch_spyre._inductor import config
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
)

logger = get_inductor_logger("lx_relayout")

LX_RELAYOUT_ATTR = "_spyre_lx_relayout_inputs"
LX_RELAYOUT_CLASSIFICATION_ATTR = "_spyre_lx_relayout_classifications"
LX_RELAYOUT_SOURCE_ATTR = "_spyre_lx_relayout_source"
LX_RELAYOUT_RESERVE_PREFIX = "__spyre_lx_relayout_reserve__"


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    """A logical producer-to-consumer LX relayout edge."""

    source_name: str
    producer_name: str
    consumer_name: str
    kind: str
    producer_core_count: int
    consumer_core_count: int
    producer_core_id_to_device_slice: dict[str, dict[str, int]]
    producer_work_slice_dims: dict[str, int]
    consumer_work_slice_dims: dict[str, int]
    read_index: int | None = None
    realized: bool = True
    communication_pattern: str = ""
    unsupported_reason: str = ""


def _op_num_cores(op: Operation) -> int:
    splits: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    factors = [int(factor) for per_dim in splits for factor in per_dim.values()]
    return math.prod(factors) if factors else 1


def _op_name(op: Operation) -> str:
    target = getattr(getattr(op, "origin_node", None), "target", None)
    return (
        getattr(target, "_opname", None)
        or getattr(target, "__name__", None)
        or getattr(target, "name", None)
        or str(target)
    )


def _is_restickify_op(op: Operation) -> bool:
    return _op_name(op) == "restickify"


def _restickify_reads_only_graph_inputs(graph: GraphLowering, op: Operation) -> bool:
    if not _is_restickify_op(op):
        return False
    read_names = [
        dep.name for dep in op.get_read_writes().reads if isinstance(dep, MemoryDep)
    ]
    return bool(read_names) and all(
        name in graph.graph_input_names for name in read_names
    )


def _restickify_reads_computed_input(graph: GraphLowering, op: Operation) -> bool:
    if not _is_restickify_op(op):
        return False
    return any(
        isinstance(graph.name_to_buffer.get(dep.name), ComputedBuffer)
        for dep in op.get_read_writes().reads
        if isinstance(dep, MemoryDep)
    )


def _single_write_dep(op: ComputedBuffer, buf_name: str) -> MemoryDep | None:
    matches = [
        dep
        for dep in op.get_read_writes().writes
        if isinstance(dep, MemoryDep) and dep.name == buf_name
    ]
    return matches[0] if len(matches) == 1 else None


def _core_id_to_device_slice(
    view: PerCoreView,
    core_count: int,
) -> dict[str, dict[str, int]] | None:
    """Return producer ownership as ``core -> device-dim -> slice-index``."""

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


def _memory_read_index(op: ComputedBuffer, dep: MemoryDep) -> int | None:
    """Return the zero-based MemoryDep read position for ``dep``."""

    for idx, read_dep in enumerate(
        read_dep
        for read_dep in op.get_read_writes().reads
        if isinstance(read_dep, MemoryDep)
    ):
        if read_dep is dep:
            return idx
        if read_dep.name == dep.name:
            return idx
    return None


def _producer_ops(graph: GraphLowering) -> dict[str, ComputedBuffer]:
    return {
        op.get_name(): op for op in graph.operations if isinstance(op, ComputedBuffer)
    }


def _record_plan(consumer: Operation, plan: LXRelayoutPlan) -> None:
    classifications = getattr(consumer, LX_RELAYOUT_CLASSIFICATION_ATTR, None)
    if not isinstance(classifications, dict):
        classifications = {}
        setattr(consumer, LX_RELAYOUT_CLASSIFICATION_ATTR, classifications)
    classifications[plan.source_name] = dataclasses.asdict(plan)

    if not plan.realized:
        return

    plans = getattr(consumer, LX_RELAYOUT_ATTR, None)
    if not isinstance(plans, dict):
        plans = {}
        setattr(consumer, LX_RELAYOUT_ATTR, plans)
    plans[plan.source_name] = dataclasses.asdict(plan)


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_ATTR):
            delattr(op, LX_RELAYOUT_ATTR)
        if hasattr(op, LX_RELAYOUT_CLASSIFICATION_ATTR):
            delattr(op, LX_RELAYOUT_CLASSIFICATION_ATTR)
        if hasattr(op, LX_RELAYOUT_SOURCE_ATTR):
            delattr(op, LX_RELAYOUT_SOURCE_ATTR)


def make_lx_relayout_reservation_name(consumer_name: str, source_name: str) -> str:
    return f"{LX_RELAYOUT_RESERVE_PREFIX}:{consumer_name}:{source_name}"


def is_lx_relayout_reservation(name: str) -> bool:
    return name.startswith(f"{LX_RELAYOUT_RESERVE_PREFIX}:")


def parse_lx_relayout_reservation_name(name: str) -> tuple[str, str] | None:
    prefix = f"{LX_RELAYOUT_RESERVE_PREFIX}:"
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix) :]
    try:
        consumer_name, source_name = rest.split(":", 1)
    except ValueError:
        return None
    if not consumer_name or not source_name:
        return None
    return consumer_name, source_name


def drop_lx_relayout_reservations(
    graph: GraphLowering, reservation_names: list[str]
) -> int:
    """Disable realized relayout plans whose scratchpad reservations failed."""

    failed_pairs = {
        parsed
        for name in reservation_names
        if (parsed := parse_lx_relayout_reservation_name(name)) is not None
    }
    if not failed_pairs:
        return 0

    removed = 0
    for op in graph.operations:
        plans = getattr(op, LX_RELAYOUT_ATTR, None)
        classifications = getattr(op, LX_RELAYOUT_CLASSIFICATION_ATTR, None)
        for consumer_name, source_name in failed_pairs:
            if op.get_name() != consumer_name:
                continue
            if isinstance(plans, dict) and source_name in plans:
                del plans[source_name]
                removed += 1
            if isinstance(classifications, dict) and source_name in classifications:
                classifications[source_name] = {
                    **classifications[source_name],
                    "realized": False,
                    "unsupported_reason": (
                        "backend relayout reservation did not fit in scratchpad"
                    ),
                }

    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_SOURCE_ATTR):
            delattr(op, LX_RELAYOUT_SOURCE_ATTR)

    producers = _producer_ops(graph)
    for op in graph.operations:
        plans = getattr(op, LX_RELAYOUT_ATTR, None)
        if not isinstance(plans, dict):
            continue
        for plan in plans.values():
            producer = producers.get(plan.get("producer_name", ""))
            if producer is not None:
                setattr(producer, LX_RELAYOUT_SOURCE_ATTR, True)

    return removed


def relayout_source_names(graph: GraphLowering) -> set[str]:
    if not config.lx_planner_relayout:
        return set()
    return {
        op.name
        for op in graph.operations
        if getattr(op, LX_RELAYOUT_SOURCE_ATTR, False)
    }


def get_lx_relayout_inputs(op: Operation) -> dict[str, Any]:
    plans = getattr(op, LX_RELAYOUT_ATTR, None)
    return plans if isinstance(plans, dict) else {}


def get_lx_relayout_classifications(op: Operation) -> dict[str, Any]:
    plans = getattr(op, LX_RELAYOUT_CLASSIFICATION_ATTR, None)
    return plans if isinstance(plans, dict) else {}


def _is_loop_scoped_relayout(plan: dict[str, Any]) -> bool:
    return plan.get("kind") == "matmul_operand_broadcast" or (
        plan.get("kind") == "layout_restickify_activation"
        and plan.get("communication_pattern")
        == "layout_transform_then_operand_broadcast"
    )


def lx_relayout_needs_resident_reservation(plan: dict[str, Any]) -> bool:
    """True when Deeptools is expected to materialize a full resident output view."""

    return not _is_loop_scoped_relayout(plan)


def plan_lx_relayouts(
    graph: GraphLowering, cache: dict | None = None
) -> list[LXRelayoutPlan]:
    """Classify scatter-capable producer/consumer LX relayout edges.

    V1 only records movement for single-writer intermediate tensors whose
    producer output is final (not K-split partials) and whose producer and
    consumer PerCoreViews differ.  Same-view edges remain owned by the existing
    LX planner.
    """

    if not config.lx_planner_relayout:
        return []

    clear_lx_relayout_metadata(graph)
    producers = _producer_ops(graph)
    planned: list[LXRelayoutPlan] = []

    for consumer in graph.operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        is_matmul_consumer = _is_matmul_op(consumer)
        for dep in consumer.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            producer = producers.get(dep.name)
            if producer is None or producer is consumer:
                continue

            write_dep = _single_write_dep(producer, dep.name)
            if write_dep is None:
                continue

            producer_view, producer_has_partial = _per_core_view_on_buf(
                producer, write_dep, dep.name, cache
            )
            if producer_has_partial:
                logger.debug(
                    "lx relayout skip: %s -> %s has partial reduction output",
                    producer.name,
                    consumer.name,
                )
                continue

            consumer_view, _consumer_has_partial = _per_core_view_on_buf(
                consumer, dep, dep.name, cache
            )
            if producer_view == consumer_view:
                continue

            producer_core_count = _op_num_cores(producer)
            consumer_core_count = _op_num_cores(consumer)
            producer_core_slices = _core_id_to_device_slice(
                producer_view, producer_core_count
            )
            if producer_core_slices is None:
                logger.debug(
                    "lx relayout skip: %s -> %s has non-static producer slices",
                    producer.name,
                    consumer.name,
                )
                continue

            producer_work_slice_dims = _work_slice_dims(producer_view)
            consumer_work_slice_dims = _work_slice_dims(consumer_view)
            read_index = _memory_read_index(consumer, dep)
            if _restickify_reads_only_graph_inputs(graph, producer):
                plan = LXRelayoutPlan(
                    source_name=dep.name,
                    producer_name=producer.get_name(),
                    consumer_name=consumer.get_name(),
                    kind="layout_restickify_weight",
                    producer_core_count=producer_core_count,
                    consumer_core_count=consumer_core_count,
                    producer_core_id_to_device_slice=producer_core_slices,
                    producer_work_slice_dims=producer_work_slice_dims,
                    consumer_work_slice_dims=consumer_work_slice_dims,
                    read_index=read_index,
                    realized=False,
                    communication_pattern="offline_weight_prelayout",
                    unsupported_reason=(
                        "graph-input/parameter restickify is owned by offline "
                        "weight prelayout, not runtime LX relayout"
                    ),
                )
                _record_plan(consumer, plan)
                planned.append(plan)
                continue

            if _restickify_reads_computed_input(graph, producer):
                is_matmul_operand = is_matmul_consumer and read_index not in (0, None)
                realize_collective = (
                    config.lx_planner_relayout_collectives and is_matmul_operand
                )
                plan = LXRelayoutPlan(
                    source_name=dep.name,
                    producer_name=producer.get_name(),
                    consumer_name=consumer.get_name(),
                    kind="layout_restickify_activation",
                    producer_core_count=producer_core_count,
                    consumer_core_count=consumer_core_count,
                    producer_core_id_to_device_slice=producer_core_slices,
                    producer_work_slice_dims=producer_work_slice_dims,
                    consumer_work_slice_dims=consumer_work_slice_dims,
                    read_index=read_index,
                    realized=realize_collective,
                    communication_pattern=(
                        "layout_transform_then_operand_broadcast"
                        if is_matmul_operand
                        else "layout_transform"
                    ),
                    unsupported_reason=""
                    if realize_collective
                    else (
                        "computed activation restickify needs an LX layout "
                        "restickify contract"
                        + (
                            " plus loop-scoped matmul operand lowering"
                            if is_matmul_operand
                            else ""
                        )
                    ),
                )
                _record_plan(consumer, plan)
                if plan.realized:
                    setattr(producer, LX_RELAYOUT_SOURCE_ATTR, True)
                planned.append(plan)
                continue

            if is_matmul_consumer and read_index not in (0, None):
                realize_collective = config.lx_planner_relayout_collectives
                plan = LXRelayoutPlan(
                    source_name=dep.name,
                    producer_name=producer.get_name(),
                    consumer_name=consumer.get_name(),
                    kind="matmul_operand_broadcast",
                    producer_core_count=producer_core_count,
                    consumer_core_count=consumer_core_count,
                    producer_core_id_to_device_slice=producer_core_slices,
                    producer_work_slice_dims=producer_work_slice_dims,
                    consumer_work_slice_dims=consumer_work_slice_dims,
                    read_index=read_index,
                    realized=realize_collective,
                    communication_pattern="all_gather_replicate",
                    unsupported_reason=""
                    if realize_collective
                    else (
                        "non-primary matmul operands need loop-scoped "
                        "collective lowering, not resident scatter materialization"
                    ),
                )
                _record_plan(consumer, plan)
                if plan.realized:
                    setattr(producer, LX_RELAYOUT_SOURCE_ATTR, True)
                planned.append(plan)
                continue

            plan = LXRelayoutPlan(
                source_name=dep.name,
                producer_name=producer.get_name(),
                consumer_name=consumer.get_name(),
                kind="scatter",
                producer_core_count=producer_core_count,
                consumer_core_count=consumer_core_count,
                producer_core_id_to_device_slice=producer_core_slices,
                producer_work_slice_dims=producer_work_slice_dims,
                consumer_work_slice_dims=consumer_work_slice_dims,
                read_index=read_index,
            )
            _record_plan(consumer, plan)
            setattr(producer, LX_RELAYOUT_SOURCE_ATTR, True)
            planned.append(plan)

    if planned:
        logger.debug("planned %d LX relayout edge(s)", len(planned))
    return planned
