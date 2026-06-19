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

"""Experimental cross-core LX-to-LX movement planner.

The regular LX planner already handles same-core scratchpad persistence.  This
module only records edges where producer and consumer slice the same buffer with
different per-core ownership, which requires explicit on-chip movement.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation

from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.compute_ops import num_bytes
from torch_spyre._inductor.logging_utils import get_inductor_logger
from torch_spyre._inductor.pass_utils import PerCoreView, _per_core_view_on_buf

logger = get_inductor_logger("onchip_move")

ONCHIP_MOVE_ATTR = "_spyre_onchip_move_plan"
ONCHIP_MOVE_OP_INFO_KEY = "onchip_move"


@dataclasses.dataclass(frozen=True)
class OnChipMoveCell:
    cell_index: int
    source_core: int
    dest_core: int
    dim_starts: dict[str, int]
    dim_sizes: dict[str, int]
    bytes: int
    source_offset_bytes: int
    dest_offset_bytes: int


@dataclasses.dataclass(frozen=True)
class OnChipMovePlan:
    source_name: str
    producer_name: str
    consumer_name: str
    producer_op: str
    consumer_op: str
    status: str
    fallback_reason: str | None
    realization_status: str
    carrier: str
    device_sizes: list[int]
    device_stride_map: list[int]
    element_bytes: int
    producer_core_count: int
    consumer_core_count: int
    producer_region_bytes: int
    consumer_region_bytes: int
    producer_view: dict[str, Any]
    consumer_view: dict[str, Any]
    cells: list[OnChipMoveCell]

    @property
    def bytes_moved(self) -> int:
        return sum(cell.bytes for cell in self.cells)


def _op_name(op: Operation) -> str:
    try:
        return str(op.get_operation_name())
    except Exception:  # noqa: BLE001
        return type(op).__name__


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


def _device_layout_and_element_bytes(buf: Any) -> tuple[list[int], list[int], int]:
    layout = getattr(buf, "layout", None)
    dev_layout = getattr(layout, "device_layout", None)
    if dev_layout is None:
        raise ValueError("buffer-has-no-device-layout")
    device_sizes = [int(size) for size in dev_layout.device_size]
    device_stride_map = [int(stride) for stride in dev_layout.stride_map]
    element_bytes = num_bytes(dev_layout.device_dtype)
    return device_sizes, device_stride_map, element_bytes


def _view_to_json(view: PerCoreView) -> dict[str, Any]:
    return {
        "work_slice_dims": [
            {"device_dim": int(dim), "split": int(split)}
            for dim, split in view.work_slice_dims
        ],
        "core_to_slot": [
            {"device_dim": int(dim), "slot_expr": str(expr)}
            for dim, expr in view.core_to_slot
        ],
    }


def _normalize_view_splits(view: PerCoreView) -> dict[int, int]:
    return {int(dim): int(split) for dim, split in view.work_slice_dims}


def _owner_lookup(
    view: PerCoreView,
    core_count: int,
) -> tuple[dict[tuple[int, ...], int], str | None]:
    dims = tuple(int(dim) for dim, _split in view.work_slice_dims)
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    core_id = sympy.Symbol("core_id")
    owners: dict[tuple[int, ...], int] = {}
    for core in range(core_count):
        key = tuple(
            int(sympy.sympify(expr_by_dim.get(dim, 0)).subs(core_id, core))
            for dim in dims
        )
        if key in owners:
            return {}, "duplicate-owner"
        owners[key] = core
    return owners, None


def _owner_key(
    common_index: dict[int, int],
    owner_dims: tuple[int, ...],
    *,
    side_splits: dict[int, int],
    common_splits: dict[int, int],
) -> tuple[int, ...]:
    return tuple(
        int(common_index.get(dim, 0))
        * int(side_splits.get(dim, 1))
        // int(common_splits.get(dim, 1))
        for dim in owner_dims
    )


def _side_slice_geometry(
    *,
    view: PerCoreView,
    owner_key: tuple[int, ...],
    device_sizes: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return per-device-dim start/size for one side's owning core slice."""

    splits = _normalize_view_splits(view)
    owner_dims = tuple(int(dim) for dim, _split in view.work_slice_dims)
    slot_by_dim = dict(zip(owner_dims, owner_key))
    starts: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for dim, dim_size in enumerate(device_sizes):
        split = int(splits.get(dim, 1))
        chunk = int(dim_size) // split
        slot = int(slot_by_dim.get(dim, 0))
        starts[dim] = slot * chunk
        sizes[dim] = chunk
    return starts, sizes


def _local_offset_bytes(
    *,
    cell_starts: dict[str, int],
    cell_sizes: dict[str, int],
    slice_starts: dict[int, int],
    slice_sizes: dict[int, int],
    element_bytes: int,
) -> tuple[int, str | None]:
    """Return the packed per-core byte offset for a cell.

    DataOp DSC layoutDimOrder is fastest-to-slowest, so d0_ has stride 1,
    d1_ has stride size(d0_), and so on.
    """

    stride = 1
    offset_elements = 0
    for dim in range(len(slice_sizes)):
        start = int(cell_starts[f"d{dim}_"])
        size = int(cell_sizes[f"d{dim}_"])
        slice_start = int(slice_starts[dim])
        slice_size = int(slice_sizes[dim])
        delta = start - slice_start
        if delta < 0 or delta + size > slice_size:
            return 0, "cell-outside-side-slice"
        offset_elements += delta * stride
        stride *= slice_size
    return offset_elements * int(element_bytes), None


def _view_region_bytes(
    view: PerCoreView,
    *,
    device_sizes: list[int],
    element_bytes: int,
) -> int:
    splits = _normalize_view_splits(view)
    elements = 1
    for dim, dim_size in enumerate(device_sizes):
        elements *= int(dim_size) // int(splits.get(dim, 1))
    return elements * int(element_bytes)


def build_onchip_move_cells(
    *,
    producer_view: PerCoreView,
    consumer_view: PerCoreView,
    device_sizes: list[int],
    element_bytes: int,
    producer_core_count: int,
    consumer_core_count: int,
    max_cells: int | None = None,
) -> tuple[list[OnChipMoveCell], str | None]:
    """Return common-refinement movement cells, or a skip reason."""

    producer_splits = _normalize_view_splits(producer_view)
    consumer_splits = _normalize_view_splits(consumer_view)
    moved_dims = tuple(sorted(set(producer_splits) | set(consumer_splits)))
    common_splits = {
        dim: math.lcm(producer_splits.get(dim, 1), consumer_splits.get(dim, 1))
        for dim in moved_dims
    }
    cell_count = math.prod(common_splits.values()) if common_splits else 1
    if max_cells is not None and cell_count > max_cells:
        return [], "too-many-common-refinement-cells"

    for dim, split in common_splits.items():
        if dim < 0 or dim >= len(device_sizes):
            return [], "view-dim-outside-device-layout"
        if int(device_sizes[dim]) % int(split) != 0:
            return [], "device-dim-not-divisible-by-common-split"

    producer_owners, reason = _owner_lookup(producer_view, producer_core_count)
    if reason is not None:
        return [], f"producer-{reason}"
    consumer_owners, reason = _owner_lookup(consumer_view, consumer_core_count)
    if reason is not None:
        return [], f"consumer-{reason}"

    producer_owner_dims = tuple(dim for dim, _split in producer_view.work_slice_dims)
    consumer_owner_dims = tuple(dim for dim, _split in consumer_view.work_slice_dims)
    cells: list[OnChipMoveCell] = []

    ranges = [range(common_splits[dim]) for dim in moved_dims]
    iterator = math.prod(len(r) for r in ranges)
    if iterator == 0:
        return [], "empty-common-refinement"

    for cell_index, indices in enumerate(itertools.product(*ranges)):
        common_index = dict(zip(moved_dims, indices))
        producer_key = _owner_key(
            common_index,
            producer_owner_dims,
            side_splits=producer_splits,
            common_splits=common_splits,
        )
        consumer_key = _owner_key(
            common_index,
            consumer_owner_dims,
            side_splits=consumer_splits,
            common_splits=common_splits,
        )
        if producer_key not in producer_owners:
            return [], "producer-owner-map-incomplete"
        if consumer_key not in consumer_owners:
            return [], "consumer-owner-map-incomplete"

        starts: dict[str, int] = {}
        sizes: dict[str, int] = {}
        for dim, dim_size in enumerate(device_sizes):
            split = common_splits.get(dim, 1)
            chunk = int(dim_size) // int(split)
            starts[f"d{dim}_"] = int(common_index.get(dim, 0)) * chunk
            sizes[f"d{dim}_"] = chunk
        cell_bytes = math.prod(sizes.values()) * int(element_bytes)
        producer_starts, producer_sizes = _side_slice_geometry(
            view=producer_view,
            owner_key=producer_key,
            device_sizes=device_sizes,
        )
        consumer_starts, consumer_sizes = _side_slice_geometry(
            view=consumer_view,
            owner_key=consumer_key,
            device_sizes=device_sizes,
        )
        source_offset, reason = _local_offset_bytes(
            cell_starts=starts,
            cell_sizes=sizes,
            slice_starts=producer_starts,
            slice_sizes=producer_sizes,
            element_bytes=element_bytes,
        )
        if reason is not None:
            return [], f"producer-{reason}"
        dest_offset, reason = _local_offset_bytes(
            cell_starts=starts,
            cell_sizes=sizes,
            slice_starts=consumer_starts,
            slice_sizes=consumer_sizes,
            element_bytes=element_bytes,
        )
        if reason is not None:
            return [], f"consumer-{reason}"
        cells.append(
            OnChipMoveCell(
                cell_index=cell_index,
                source_core=producer_owners[producer_key],
                dest_core=consumer_owners[consumer_key],
                dim_starts=starts,
                dim_sizes=sizes,
                bytes=cell_bytes,
                source_offset_bytes=source_offset,
                dest_offset_bytes=dest_offset,
            )
        )

    return cells, None


def _plan_json(plan: OnChipMovePlan) -> dict[str, Any]:
    return {
        "source_name": plan.source_name,
        "producer": plan.producer_name,
        "consumer": plan.consumer_name,
        "producer_op": plan.producer_op,
        "consumer_op": plan.consumer_op,
        "status": plan.status,
        "fallback_reason": plan.fallback_reason,
        "realization_status": plan.realization_status,
        "carrier": plan.carrier,
        "device_sizes": plan.device_sizes,
        "device_stride_map": plan.device_stride_map,
        "element_bytes": plan.element_bytes,
        "producer_core_count": plan.producer_core_count,
        "consumer_core_count": plan.consumer_core_count,
        "producer_region_bytes": plan.producer_region_bytes,
        "consumer_region_bytes": plan.consumer_region_bytes,
        "producer_view": plan.producer_view,
        "consumer_view": plan.consumer_view,
        "cell_count": len(plan.cells),
        "bytes_moved": plan.bytes_moved,
        "cells": [dataclasses.asdict(cell) for cell in plan.cells],
    }


def _skip_plan(
    *,
    source_name: str,
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    reason: str,
    producer_view: PerCoreView | None = None,
    consumer_view: PerCoreView | None = None,
) -> OnChipMovePlan:
    return OnChipMovePlan(
        source_name=source_name,
        producer_name=producer.get_name(),
        consumer_name=consumer.get_name(),
        producer_op=_op_name(producer),
        consumer_op=_op_name(consumer),
        status="skipped",
        fallback_reason=reason,
        realization_status="not-realized-skipped",
        carrier=config.onchip_move_carrier,
        device_sizes=[],
        device_stride_map=[],
        element_bytes=0,
        producer_core_count=_op_num_cores(producer),
        consumer_core_count=_op_num_cores(consumer),
        producer_region_bytes=0,
        consumer_region_bytes=0,
        producer_view=_view_to_json(producer_view or PerCoreView((), ())),
        consumer_view=_view_to_json(consumer_view or PerCoreView((), ())),
        cells=[],
    )


def plan_onchip_move_edge(
    graph: GraphLowering,
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    read_dep: MemoryDep,
    *,
    cache: dict | None = None,
) -> OnChipMovePlan:
    write_dep = _single_write_dep(producer, read_dep.name)
    if write_dep is None:
        return _skip_plan(
            source_name=read_dep.name,
            producer=producer,
            consumer=consumer,
            reason="producer-write-dep-not-unique",
        )

    producer_view, producer_partial = _per_core_view_on_buf(
        producer,
        write_dep,
        read_dep.name,
        cache=cache,
    )
    consumer_view, _consumer_partial = _per_core_view_on_buf(
        consumer,
        read_dep,
        read_dep.name,
        cache=cache,
    )
    if producer_partial:
        return _skip_plan(
            source_name=read_dep.name,
            producer=producer,
            consumer=consumer,
            reason="producer-k-split-partial-output",
            producer_view=producer_view,
            consumer_view=consumer_view,
        )
    if producer_view == consumer_view:
        return _skip_plan(
            source_name=read_dep.name,
            producer=producer,
            consumer=consumer,
            reason="same-per-core-view-owned-by-lx-planner",
            producer_view=producer_view,
            consumer_view=consumer_view,
        )

    try:
        buf = graph.get_buffer(read_dep.name)
        device_sizes, device_stride_map, element_bytes = _device_layout_and_element_bytes(
            buf
        )
        cells, reason = build_onchip_move_cells(
            producer_view=producer_view,
            consumer_view=consumer_view,
            device_sizes=device_sizes,
            element_bytes=element_bytes,
            producer_core_count=_op_num_cores(producer),
            consumer_core_count=_op_num_cores(consumer),
            max_cells=config.onchip_move_max_cells,
        )
    except Exception as exc:  # noqa: BLE001
        return _skip_plan(
            source_name=read_dep.name,
            producer=producer,
            consumer=consumer,
            reason=type(exc).__name__,
            producer_view=producer_view,
            consumer_view=consumer_view,
        )
    if reason is not None:
        return _skip_plan(
            source_name=read_dep.name,
            producer=producer,
            consumer=consumer,
            reason=reason,
            producer_view=producer_view,
            consumer_view=consumer_view,
        )

    return OnChipMovePlan(
        source_name=read_dep.name,
        producer_name=producer.get_name(),
        consumer_name=consumer.get_name(),
        producer_op=_op_name(producer),
        consumer_op=_op_name(consumer),
        status="planned",
        fallback_reason=None,
        realization_status=(
            "planned-mixed-carrier-enabled"
            if config.onchip_move_realize
            else "planned-not-realized"
        ),
        carrier=config.onchip_move_carrier,
        device_sizes=device_sizes,
        device_stride_map=device_stride_map,
        element_bytes=element_bytes,
        producer_core_count=_op_num_cores(producer),
        consumer_core_count=_op_num_cores(consumer),
        producer_region_bytes=_view_region_bytes(
            producer_view,
            device_sizes=device_sizes,
            element_bytes=element_bytes,
        ),
        consumer_region_bytes=_view_region_bytes(
            consumer_view,
            device_sizes=device_sizes,
            element_bytes=element_bytes,
        ),
        producer_view=_view_to_json(producer_view),
        consumer_view=_view_to_json(consumer_view),
        cells=cells,
    )


def _append_jsonl(path: str, plans: list[OnChipMovePlan]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(_plan_json(plan), sort_keys=True) + "\n")


def _write_debug_dir(path: str, plans: list[OnChipMovePlan]) -> None:
    if not path:
        return
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    payload = [_plan_json(plan) for plan in plans]
    (output / "onchip_move_plans.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _attach_plan_to_consumer(consumer: ComputedBuffer, plan: OnChipMovePlan) -> None:
    plan_payload = _plan_json(plan)
    existing_move_info = getattr(consumer, ONCHIP_MOVE_ATTR, None)
    move_info = existing_move_info if isinstance(existing_move_info, dict) else {}
    move_info[plan.source_name] = plan_payload
    setattr(consumer, ONCHIP_MOVE_ATTR, move_info)

    data = getattr(consumer, "data", None)
    op_info = getattr(data, "op_info", None)
    if isinstance(op_info, dict):
        op_move_info = op_info.setdefault(ONCHIP_MOVE_OP_INFO_KEY, {})
        if not isinstance(op_move_info, dict):
            op_move_info = {}
            op_info[ONCHIP_MOVE_OP_INFO_KEY] = op_move_info
        op_move_info[plan.source_name] = plan_payload


def plan_onchip_moves(graph: GraphLowering) -> None:
    if not config.onchip_move_planner:
        return

    name_to_op = {
        op.get_name(): op
        for op in graph.operations
        if isinstance(op, ComputedBuffer)
    }
    cache: dict = {}
    plans: list[OnChipMovePlan] = []
    for consumer in graph.operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        for dep in consumer.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            producer = name_to_op.get(dep.name)
            if producer is None:
                continue
            plan = plan_onchip_move_edge(
                graph,
                producer,
                consumer,
                dep,
                cache=cache,
            )
            plans.append(plan)
            if plan.status == "planned":
                _attach_plan_to_consumer(consumer, plan)

    planned = [plan for plan in plans if plan.status == "planned"]
    if plans:
        logger.info(
            "onchip_move summary edges=%d planned=%d bytes=%d realize=%s",
            len(plans),
            len(planned),
            sum(plan.bytes_moved for plan in planned),
            config.onchip_move_realize,
        )
    _append_jsonl(config.onchip_move_jsonl, plans)
    _write_debug_dir(config.onchip_move_debug_dir, plans)
