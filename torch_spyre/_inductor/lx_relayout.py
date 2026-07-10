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
ALL_TO_ALL_SHUFFLE = "all_to_all_shuffle"
ALL_GATHER = "all_gather"


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    """A logical producer-to-consumer LX relayout edge."""

    source_name: str
    producer_name: str
    consumer_name: str
    kind: str
    consumer_core_count: int
    producer_core_id_to_device_slice: dict[str, dict[str, int]]
    consumer_work_slice_dims: dict[str, int]


@dataclasses.dataclass(frozen=True)
class LXRelayoutTopology:
    """Communication cardinality implied by two per-core coordinate maps."""

    kind: str
    max_fanout: int
    max_fanin: int
    transfer_count: int


def _intervals_overlap(
    lhs_slot: int, lhs_split: int, rhs_slot: int, rhs_split: int
) -> bool:
    return lhs_slot * rhs_split < (rhs_slot + 1) * lhs_split and (
        rhs_slot * lhs_split < (lhs_slot + 1) * rhs_split
    )


def _classify_coordinate_topology(
    producer_map: dict[str, dict[str, int]],
    producer_splits: dict[str, int],
    consumer_map: dict[str, dict[str, int]],
    consumer_splits: dict[str, int],
) -> LXRelayoutTopology:
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

    max_fanout = max(fanout.values(), default=0)
    max_fanin = max(fanin.values(), default=0)
    if transfer_count == 0:
        kind = "unsupported"
    elif max_fanout <= 1 and max_fanin <= 1:
        kind = ALL_TO_ALL_SHUFFLE
    elif max_fanout > 1 and max_fanin > 1:
        kind = ALL_GATHER
    else:
        kind = "unsupported"
    return LXRelayoutTopology(kind, max_fanout, max_fanin, transfer_count)


def _op_num_cores(op: Operation) -> int:
    splits: tuple[dict, dict] = getattr(op, "op_it_space_splits", ({}, {}))
    factors = [int(factor) for per_dim in splits for factor in per_dim.values()]
    return math.prod(factors) if factors else 1


def _single_write_dep(op: ComputedBuffer, buf_name: str) -> MemoryDep | None:
    matches = [
        dep
        for dep in op.get_read_writes().writes
        if isinstance(dep, MemoryDep) and dep.name == buf_name
    ]
    return matches[0] if len(matches) == 1 else None


def _op_name(op: Operation) -> str:
    for fx_node in (getattr(op, "origin_node", None), *getattr(op, "origins", ())):
        target = getattr(fx_node, "target", None)
        name = (
            getattr(target, "_opname", None)
            or getattr(target, "__name__", None)
            or getattr(target, "name", None)
        )
        if name is not None:
            return str(name)
    return "None"


def _restickify_reads_computed_input(graph: GraphLowering, op: Operation) -> bool:
    if _op_name(op) != "restickify":
        return False
    return any(
        isinstance(graph.name_to_buffer.get(dep.name), ComputedBuffer)
        for dep in op.get_read_writes().reads
        if isinstance(dep, MemoryDep)
    )


def _matmul_operand_source_good_for_lx_relayout(
    graph: GraphLowering, op: Operation
) -> bool:
    """Exclude graph-input and weight restickifies from activation relayout."""

    return _op_name(op) != "restickify" or _restickify_reads_computed_input(graph, op)


def _op_nbytes(graph: GraphLowering, op: Operation) -> int | None:
    try:
        sizes = list(op.get_size())
        dtype = op.get_dtype()
    except Exception:
        return None

    numel = 1
    for dim in sizes:
        try:
            extent = int(dim)
        except (TypeError, ValueError):
            try:
                extent = int(graph.sizevars.size_hint(dim))
            except Exception:
                return None
        numel *= extent

    itemsize = getattr(dtype, "itemsize", None)
    return None if itemsize is None else numel * int(itemsize)


def _core_id_to_device_slice(
    view: PerCoreView,
    core_count: int,
) -> dict[str, dict[str, int]] | None:
    """Return producer ownership as ``core -> device-dim -> slice-index``."""

    core_id = sympy.Symbol("core_id")
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    split_dims = {
        int(dim): _work_div_factor(split) for dim, split in view.work_slice_dims
    }
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


def _work_div_factor(split: int | tuple[int, int]) -> int:
    return int(split[0] if isinstance(split, tuple) else split)


def _work_slice_dims(view: PerCoreView) -> dict[str, int]:
    return {
        str(int(dim)): _work_div_factor(split) for dim, split in view.work_slice_dims
    }


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


def _memory_read_index(op: ComputedBuffer, dep: MemoryDep) -> int | None:
    """Return the zero-based MemoryDep read position for ``dep``."""

    for idx, read_dep in enumerate(
        read_dep
        for read_dep in op.get_read_writes().reads
        if isinstance(read_dep, MemoryDep)
    ):
        if read_dep.name == dep.name:
            return idx
    return None


def _producer_ops(graph: GraphLowering) -> dict[str, ComputedBuffer]:
    return {
        op.get_name(): op for op in graph.operations if isinstance(op, ComputedBuffer)
    }


def _record_plan(consumer: Operation, plan: LXRelayoutPlan) -> None:
    plans = getattr(consumer, LX_RELAYOUT_ATTR, None)
    if not isinstance(plans, dict):
        plans = {}
        setattr(consumer, LX_RELAYOUT_ATTR, plans)
    plans[plan.source_name] = dataclasses.asdict(plan)


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_ATTR):
            delattr(op, LX_RELAYOUT_ATTR)


def get_lx_relayout_inputs(op: Operation) -> dict[str, Any]:
    plans = getattr(op, LX_RELAYOUT_ATTR, None)
    return plans if isinstance(plans, dict) else {}


def collect_lx_relayout_plans(
    graph: GraphLowering, cache: dict | None = None
) -> list[LXRelayoutPlan]:
    """Classify bounded LX all-to-all-shuffle and all-gather edges.

    V1 only records movement for single-writer intermediate tensors whose
    producer output is final (not K-split partials) and whose producer and
    consumer PerCoreViews differ.  Same-view edges remain owned by the existing
    LX planner.
    """

    if not config.lx_planner_relayout:
        return []

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

            consumer_view, consumer_has_partial = _per_core_view_on_buf(
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
            topology = _classify_coordinate_topology(
                producer_core_slices,
                producer_work_slice_dims,
                consumer_core_slices,
                consumer_work_slice_dims,
            )

            read_index = _memory_read_index(consumer, dep)
            if is_matmul_consumer and topology.kind == ALL_TO_ALL_SHUFFLE:
                if read_index not in (0, None):
                    continue
            elif is_matmul_consumer and topology.kind == ALL_GATHER:
                if (
                    read_index not in (0, 1)
                    or consumer_has_partial
                    or not _matmul_operand_source_good_for_lx_relayout(graph, producer)
                ):
                    continue
                operand_nbytes = _op_nbytes(graph, producer)
                if (
                    operand_nbytes is None
                    or operand_nbytes
                    > config.lx_planner_relayout_max_matmul_operand_bytes
                ):
                    logger.debug(
                        "lx relayout skip: %s -> %s all-gather operand is %s "
                        "bytes (limit %s)",
                        producer.name,
                        consumer.name,
                        operand_nbytes,
                        config.lx_planner_relayout_max_matmul_operand_bytes,
                    )
                    continue
            elif topology.kind != ALL_TO_ALL_SHUFFLE:
                continue

            plan = LXRelayoutPlan(
                source_name=dep.name,
                producer_name=producer.get_name(),
                consumer_name=consumer.get_name(),
                kind=topology.kind,
                consumer_core_count=consumer_core_count,
                producer_core_id_to_device_slice=producer_core_slices,
                consumer_work_slice_dims=consumer_work_slice_dims,
            )
            planned.append(plan)

    if planned:
        logger.debug("found %d LX relayout edge candidate(s)", len(planned))
    return planned


def record_lx_relayout_plan(graph: GraphLowering, plan: LXRelayoutPlan) -> None:
    """Stamp a relayout plan after the source buffer is placed in LX."""

    ops = {op.get_name(): op for op in graph.operations}
    consumer = ops.get(plan.consumer_name)
    producer = ops.get(plan.producer_name)
    if consumer is None or producer is None:
        return
    _record_plan(consumer, plan)
