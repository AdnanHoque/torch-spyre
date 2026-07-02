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
from torch_spyre._inductor.layout_allgather_restickify import (
    COMM_CLASS_ALL_GATHER,
    LAYOUT_ALLGATHER_RESTICKIFY,
    RESTICKIFY_LX_OP,
    make_layout_allgather_restickify_contract,
)

logger = get_inductor_logger("lx_relayout")

LX_RELAYOUT_ATTR = "_spyre_lx_relayout_inputs"
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
    consumer_core_id_to_device_slice: dict[str, dict[str, int]] | None = None
    read_index: int | None = None
    realized: bool = True
    communication_class: str = "scatter"
    communication_pattern: str = "scatter"
    max_fanout: int = 0
    max_fanin: int = 0
    transfer_count: int = 0
    requires_staged_realization: bool = False
    layout_contract: dict[str, Any] | None = None
    unsupported_reason: str = ""


@dataclasses.dataclass(frozen=True)
class LXRelayoutTopology:
    """Coordinate-overlap communication class for a producer/consumer edge."""

    communication_class: str
    communication_pattern: str
    max_fanout: int
    max_fanin: int
    transfer_count: int


def _int_keyed_map(
    value: dict[str, dict[str, int]] | dict[int, dict[int, int]],
) -> dict[int, dict[int, int]]:
    return {
        int(core): {int(dim): int(slot) for dim, slot in per_core.items()}
        for core, per_core in value.items()
    }


def _int_keyed_dims(value: dict[str, int] | dict[int, int]) -> dict[int, int]:
    return {int(dim): int(split) for dim, split in value.items()}


def _intervals_overlap(
    a_slot: int, a_split: int, b_slot: int, b_split: int
) -> bool:
    # Compare [slot/split, (slot+1)/split) without floating point.
    return a_slot * b_split < (b_slot + 1) * a_split and b_slot * a_split < (
        a_slot + 1
    ) * b_split


def _core_slices_overlap(
    producer_slice: dict[int, int],
    producer_splits: dict[int, int],
    consumer_slice: dict[int, int],
    consumer_splits: dict[int, int],
) -> bool:
    for dim in set(producer_splits) | set(consumer_splits):
        producer_split = producer_splits.get(dim, 1)
        consumer_split = consumer_splits.get(dim, 1)
        producer_slot = producer_slice.get(dim, 0)
        consumer_slot = consumer_slice.get(dim, 0)
        if not _intervals_overlap(
            producer_slot, producer_split, consumer_slot, consumer_split
        ):
            return False
    return True


def _classify_coordinate_topology(
    producer_core_id_to_device_slice: dict[str, dict[str, int]],
    producer_work_slice_dims: dict[str, int],
    consumer_core_id_to_device_slice: dict[str, dict[str, int]],
    consumer_work_slice_dims: dict[str, int],
) -> LXRelayoutTopology:
    """Classify movement cardinality from producer/consumer coordinate maps.

    The backend owns physical realization. This helper only names the logical
    overlap class so artifacts distinguish one-to-one scatter from multicast,
    gather, and all-gather shaped coordinate mismatches.
    """

    producer_map = _int_keyed_map(producer_core_id_to_device_slice)
    consumer_map = _int_keyed_map(consumer_core_id_to_device_slice)
    producer_splits = _int_keyed_dims(producer_work_slice_dims)
    consumer_splits = _int_keyed_dims(consumer_work_slice_dims)

    fanout = {core: 0 for core in producer_map}
    fanin = {core: 0 for core in consumer_map}
    transfer_count = 0
    for producer_core, producer_slice in producer_map.items():
        for consumer_core, consumer_slice in consumer_map.items():
            if _core_slices_overlap(
                producer_slice, producer_splits, consumer_slice, consumer_splits
            ):
                fanout[producer_core] += 1
                fanin[consumer_core] += 1
                transfer_count += 1

    max_fanout = max(fanout.values(), default=0)
    max_fanin = max(fanin.values(), default=0)
    if transfer_count == 0:
        return LXRelayoutTopology("unsupported", "no_coordinate_overlap", 0, 0, 0)
    if max_fanout <= 1 and max_fanin <= 1:
        return LXRelayoutTopology(
            "scatter", "one_to_one", max_fanout, max_fanin, transfer_count
        )
    if max_fanout > 1 and max_fanin <= 1:
        communication_class = (
            "broadcast" if max_fanout == len(consumer_map) else "multicast"
        )
        return LXRelayoutTopology(
            communication_class,
            "one_to_many",
            max_fanout,
            max_fanin,
            transfer_count,
        )
    if max_fanout <= 1 and max_fanin > 1:
        return LXRelayoutTopology(
            "gather", "many_to_one", max_fanout, max_fanin, transfer_count
        )
    return LXRelayoutTopology(
        COMM_CLASS_ALL_GATHER, "many_to_many", max_fanout, max_fanin, transfer_count
    )


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


def _restickify_reads_computed_input(graph: GraphLowering, op: Operation) -> bool:
    if _op_name(op) != "restickify":
        return False
    return any(
        isinstance(graph.name_to_buffer.get(dep.name), ComputedBuffer)
        for dep in op.get_read_writes().reads
        if isinstance(dep, MemoryDep)
    )


def _producer_ops(graph: GraphLowering) -> dict[str, ComputedBuffer]:
    return {
        op.get_name(): op for op in graph.operations if isinstance(op, ComputedBuffer)
    }


def _plan_payload(plan: LXRelayoutPlan) -> dict[str, Any]:
    payload = dataclasses.asdict(plan)
    contract = payload.pop("layout_contract", None)
    if isinstance(contract, dict):
        payload.update(contract)
    return {key: value for key, value in payload.items() if value not in (None, "")}


def _record_plan(consumer: Operation, plan: LXRelayoutPlan) -> None:
    plans = getattr(consumer, LX_RELAYOUT_ATTR, None)
    if not isinstance(plans, dict):
        plans = {}
        setattr(consumer, LX_RELAYOUT_ATTR, plans)
    plans[plan.source_name] = _plan_payload(plan)


def clear_lx_relayout_metadata(
    graph: GraphLowering, *, preserve_unrealized: bool = False
) -> None:
    kept_sources: set[str] = set()
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_SOURCE_ATTR):
            delattr(op, LX_RELAYOUT_SOURCE_ATTR)

        plans = getattr(op, LX_RELAYOUT_ATTR, None)
        if not isinstance(plans, dict):
            continue
        if not preserve_unrealized:
            delattr(op, LX_RELAYOUT_ATTR)
            continue

        kept = {
            name: plan
            for name, plan in plans.items()
            if isinstance(plan, dict)
            and (
                plan.get("realized") is False
                or plan.get("requires_staged_realization") is True
            )
        }
        if kept:
            setattr(op, LX_RELAYOUT_ATTR, kept)
            kept_sources.update(kept)
        else:
            delattr(op, LX_RELAYOUT_ATTR)

    if preserve_unrealized and kept_sources:
        for op in graph.operations:
            if op.get_name() in kept_sources:
                setattr(op, LX_RELAYOUT_SOURCE_ATTR, True)


def make_lx_relayout_reservation_name(consumer_name: str, source_name: str) -> str:
    return f"{LX_RELAYOUT_RESERVE_PREFIX}:{consumer_name}:{source_name}"


def is_lx_relayout_reservation(name: str) -> bool:
    return name.startswith(f"{LX_RELAYOUT_RESERVE_PREFIX}:")


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
            consumer_core_slices = _core_id_to_device_slice(
                consumer_view, consumer_core_count
            )
            if consumer_core_slices is None:
                logger.debug(
                    "lx relayout skip: %s -> %s has non-static consumer slices",
                    producer.name,
                    consumer.name,
                )
                continue
            topology = _classify_coordinate_topology(
                producer_core_slices,
                producer_work_slice_dims,
                consumer_core_slices,
                consumer_work_slice_dims,
            )
            if topology.communication_class == "unsupported":
                logger.debug(
                    "lx relayout skip: %s -> %s has unsupported coordinate topology: %s",
                    producer.name,
                    consumer.name,
                    topology.communication_pattern,
                )
                continue
            read_index = _memory_read_index(consumer, dep)
            if is_matmul_consumer and read_index not in (0, None):
                if (
                    config.lx_planner_relayout_layout_allgather_restickify
                    and _restickify_reads_computed_input(graph, producer)
                ):
                    layout_contract = make_layout_allgather_restickify_contract(
                        producer_op="mul",
                        restickify_op=RESTICKIFY_LX_OP,
                        consumer_op="batchmatmul",
                        producer_work_slice_dims=producer_work_slice_dims,
                        restickify_work_slice_dims=producer_work_slice_dims,
                        consumer_work_slice_dims=consumer_work_slice_dims,
                    )
                    plan = LXRelayoutPlan(
                        source_name=dep.name,
                        producer_name=producer.get_name(),
                        consumer_name=consumer.get_name(),
                        kind=LAYOUT_ALLGATHER_RESTICKIFY,
                        producer_core_count=producer_core_count,
                        consumer_core_count=consumer_core_count,
                        producer_core_id_to_device_slice=producer_core_slices,
                        producer_work_slice_dims=producer_work_slice_dims,
                        consumer_work_slice_dims=consumer_work_slice_dims,
                        consumer_core_id_to_device_slice=consumer_core_slices,
                        read_index=read_index,
                        realized=False,
                        communication_class=COMM_CLASS_ALL_GATHER,
                        communication_pattern=LAYOUT_ALLGATHER_RESTICKIFY,
                        max_fanout=topology.max_fanout,
                        max_fanin=topology.max_fanin,
                        transfer_count=topology.transfer_count,
                        requires_staged_realization=True,
                        layout_contract=layout_contract,
                        unsupported_reason="staged layout all-gather restickify metadata only",
                    )
                    _record_plan(consumer, plan)
                    setattr(producer, LX_RELAYOUT_SOURCE_ATTR, True)
                    planned.append(plan)
                    continue

                if not config.lx_planner_relayout_collectives:
                    continue

                plan = LXRelayoutPlan(
                    source_name=dep.name,
                    producer_name=producer.get_name(),
                    consumer_name=consumer.get_name(),
                    kind=topology.communication_class,
                    producer_core_count=producer_core_count,
                    consumer_core_count=consumer_core_count,
                    producer_core_id_to_device_slice=producer_core_slices,
                    producer_work_slice_dims=producer_work_slice_dims,
                    consumer_work_slice_dims=consumer_work_slice_dims,
                    consumer_core_id_to_device_slice=consumer_core_slices,
                    read_index=read_index,
                    communication_class=topology.communication_class,
                    communication_pattern=topology.communication_pattern,
                    max_fanout=topology.max_fanout,
                    max_fanin=topology.max_fanin,
                    transfer_count=topology.transfer_count,
                )
                _record_plan(consumer, plan)
                setattr(producer, LX_RELAYOUT_SOURCE_ATTR, True)
                planned.append(plan)
                continue

            plan = LXRelayoutPlan(
                source_name=dep.name,
                producer_name=producer.get_name(),
                consumer_name=consumer.get_name(),
                kind=topology.communication_class,
                producer_core_count=producer_core_count,
                consumer_core_count=consumer_core_count,
                producer_core_id_to_device_slice=producer_core_slices,
                producer_work_slice_dims=producer_work_slice_dims,
                consumer_work_slice_dims=consumer_work_slice_dims,
                consumer_core_id_to_device_slice=consumer_core_slices,
                read_index=read_index,
                communication_class=topology.communication_class,
                communication_pattern=topology.communication_pattern,
                max_fanout=topology.max_fanout,
                max_fanin=topology.max_fanin,
                transfer_count=topology.transfer_count,
            )
            _record_plan(consumer, plan)
            setattr(producer, LX_RELAYOUT_SOURCE_ATTR, True)
            planned.append(plan)

    if planned:
        logger.debug("planned %d LX relayout edge(s)", len(planned))
    return planned
