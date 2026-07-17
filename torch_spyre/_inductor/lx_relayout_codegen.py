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

"""Materialize allocator-approved LX relayout plans as SHUFFLE OpSpecs."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Sequence

from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_OP,
    LAYOUT_LABELS,
    MATMUL_DIM_LABELS,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.lx_relayout import LX_RELAYOUT_ATTR, LXRelayoutPlan
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _current_node_lx_relayout_inputs(current_node) -> dict[str, LXRelayoutPlan]:
    plans: dict[str, LXRelayoutPlan] = {}
    pending = [current_node]
    seen: set[int] = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        node = getattr(item, "node", None)
        node_plans = getattr(node, LX_RELAYOUT_ATTR, None)
        if isinstance(node_plans, dict):
            for source_name, plan in node_plans.items():
                if not isinstance(plan, LXRelayoutPlan):
                    continue
                previous = plans.get(source_name)
                if previous is not None and previous != plan:
                    raise Unsupported(
                        f"conflicting LX relayout plans for {source_name}"
                    )
                plans[source_name] = plan
        get_nodes = getattr(item, "get_nodes", None)
        if callable(get_nodes):
            pending.extend(child for child in get_nodes() if child is not item)
    return plans


def _materialize_explicit_lx_shuffle(
    source_arg: TensorArg,
    consumer_spec: OpSpec,
    plan: LXRelayoutPlan,
) -> tuple[OpSpec, TensorArg]:
    """Build one standard bounded S1 -> SHUFFLE -> S2 operation."""
    destination_address = plan.destination_lx_address
    destination_name = plan.destination_name
    producer_axis_map = plan.source_core_id_to_axis_slice
    consumer_axis_map = plan.destination_core_id_to_axis_slice
    participant_count = len(consumer_axis_map)
    if (
        "lx" not in source_arg.allocation
        or not isinstance(destination_address, int)
        or not isinstance(producer_axis_map, dict)
        or participant_count <= 0
    ):
        raise Unsupported(f"invalid allocated LX relayout for {plan.source_name}")

    if consumer_spec.op not in (BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP):
        raise Unsupported("LX relayout is only supported for matmul consumers")
    consumer_symbols = list(consumer_spec.iteration_space)
    consumer_labels = MATMUL_DIM_LABELS[-len(consumer_symbols) :]
    symbol_to_label = dict(zip(consumer_symbols, consumer_labels))
    geometry = plan.shuffle_geometry
    if len(consumer_symbols) != geometry.consumer_rank:
        raise Unsupported(
            f"LX relayout iteration rank changed before codegen for {plan.source_name}"
        )
    shuffle_symbols = tuple(consumer_symbols[axis] for axis in geometry.iteration_axes)
    shuffle_iteration_space = {
        symbol: (consumer_spec.iteration_space[symbol][0], destination_split)
        for symbol, destination_split in zip(
            shuffle_symbols, geometry.destination_splits
        )
    }
    consumer_axis_splits = tuple(
        int(split) for _, split in consumer_spec.iteration_space.values()
    )
    if consumer_axis_splits != geometry.consumer_splits:
        raise Unsupported(
            f"LX relayout work division changed before codegen for {plan.source_name}"
        )
    retained_splits = tuple(
        consumer_axis_splits[axis] for axis in geometry.iteration_axes
    )
    if retained_splits != geometry.destination_splits:
        raise Unsupported(
            f"LX relayout work split changed before codegen for {plan.source_name}"
        )
    if math.prod(consumer_axis_splits) != participant_count:
        raise Unsupported(
            f"LX relayout participant count changed before codegen for "
            f"{plan.source_name}"
        )

    expected_axes = set(plan.source_axis_splits)
    if expected_axes != set(plan.destination_axis_splits) or any(
        int(axis) not in geometry.iteration_axes for axis in expected_axes
    ):
        raise Unsupported(
            f"LX relayout distribution axes changed for {plan.source_name}"
        )
    symbol_name_by_axis: dict[str, str] = {}
    for axis in expected_axes:
        axis_index = int(axis)
        if axis_index < 0 or axis_index >= len(consumer_symbols):
            raise Unsupported(
                f"LX relayout distribution axis is out of range for {plan.source_name}"
            )
        symbol_name = str(consumer_symbols[axis_index])
        if symbol_name in symbol_name_by_axis.values():
            raise Unsupported(
                f"LX relayout distribution axis is ambiguous for {plan.source_name}"
            )
        symbol_name_by_axis[axis] = symbol_name

    def distribution_by_axis_name(
        core_map: dict[str, dict[str, int]], axis_splits: dict[str, int]
    ) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
        if set(axis_splits) != expected_axes:
            raise Unsupported(
                f"LX relayout distribution axes changed for {plan.source_name}"
            )
        named_map: dict[str, dict[str, int]] = {}
        for core, per_axis in core_map.items():
            if set(per_axis) != expected_axes:
                raise Unsupported(
                    f"LX relayout core ownership changed for {plan.source_name}"
                )
            named_map[str(core)] = {
                symbol_name_by_axis[axis]: int(slot) for axis, slot in per_axis.items()
            }
        named_splits = {
            symbol_name_by_axis[axis]: int(split) for axis, split in axis_splits.items()
        }
        return named_map, named_splits

    producer_map, producer_splits = distribution_by_axis_name(
        producer_axis_map, plan.source_axis_splits
    )
    consumer_map, consumer_splits = distribution_by_axis_name(
        consumer_axis_map, plan.destination_axis_splits
    )

    source = replace(
        source_arg,
        is_input=True,
        name=plan.source_name,
        allocation=dict(source_arg.allocation),
        allocation_core_id_to_axis_slice=producer_map,
        allocation_axis_splits=producer_splits,
    )
    destination_input = replace(
        source_arg,
        is_input=True,
        name=destination_name,
        allocation={"lx": destination_address},
        allocation_core_id_to_axis_slice=None,
        allocation_axis_splits=None,
    )
    destination_output = replace(
        destination_input,
        is_input=False,
        allocation_core_id_to_axis_slice=consumer_map,
        allocation_axis_splits=consumer_splits,
    )
    shuffle = OpSpec(
        op="shuffle",
        is_reduction=False,
        iteration_space=shuffle_iteration_space,
        args=[source, destination_output],
        op_info={},
        symbolic_dim_bounds=dict(consumer_spec.symbolic_dim_bounds),
        num_cores_override=participant_count,
        dim_labels_override=[
            symbol_to_label[symbol] for symbol in shuffle_iteration_space
        ],
        layout_labels_override=["KERNEL", *LAYOUT_LABELS],
    )
    return shuffle, destination_input


def materialize_lx_relayout_inputs(
    current_node,
    args: list[TensorArg],
    tensors: Sequence[Any],
    consumer_spec: OpSpec,
) -> list[OpSpec]:
    """Materialize every planned consumer input before its compute row."""

    plans = _current_node_lx_relayout_inputs(current_node)
    prefix_specs: list[OpSpec] = []
    materialized: dict[str, TensorArg] = {}
    for arg_index, tensor in enumerate(tensors):
        plan = plans.get(tensor.name)
        if not isinstance(plan, LXRelayoutPlan):
            continue
        destination_arg = materialized.get(plan.source_name)
        if destination_arg is not None:
            args[arg_index] = destination_arg
            continue
        shuffle_spec, destination_arg = _materialize_explicit_lx_shuffle(
            args[arg_index],
            consumer_spec,
            plan,
        )
        prefix_specs.append(shuffle_spec)
        materialized[plan.source_name] = destination_arg
        args[arg_index] = destination_arg
    return prefix_specs
