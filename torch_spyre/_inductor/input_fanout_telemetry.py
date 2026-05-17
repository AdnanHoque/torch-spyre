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

"""Default-off telemetry for graph input, weight, and constant fanout."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Operation
from torch._inductor.virtualized import V

from . import config
from .logging_utils import get_inductor_logger
from .restickify_ring import (
    _bytes_moved_or_zero,
    _graph_buffer_by_name,
    _op_kind,
    _stride_map_from_buffer,
    build_name_to_op_map,
    is_restickify_op,
    source_kind_from_buffer,
)

logger = get_inductor_logger("input_fanout_telemetry")


@dataclasses.dataclass(frozen=True)
class InputFanoutEstimate:
    source_name: str
    source_kind: str
    consumer_count: int
    consumers: list[str]
    consumer_kinds: dict[str, int]
    restickify_consumers: list[str]
    restickify_bytes_moved: int
    approximate_consumer_bytes: int
    source_stride_map: list[int] | None
    target_stride_maps: list[list[int]]


def _graph_input_names() -> list[str] | None:
    try:
        return list(V.graph.graph_input_names)
    except Exception:  # noqa: BLE001
        return None


def _source_kind(source_name: str) -> tuple[str, list[int] | None]:
    source_buffer = _graph_buffer_by_name(source_name)
    return (
        source_kind_from_buffer(source_name, source_buffer, _graph_input_names()),
        _stride_map_from_buffer(source_buffer),
    )


def _estimate_to_json(estimate: InputFanoutEstimate) -> dict[str, Any]:
    return {
        "source_name": estimate.source_name,
        "source_kind": estimate.source_kind,
        "consumer_count": estimate.consumer_count,
        "consumers": estimate.consumers,
        "consumer_kinds": estimate.consumer_kinds,
        "restickify_consumers": estimate.restickify_consumers,
        "restickify_bytes_moved": estimate.restickify_bytes_moved,
        "approximate_consumer_bytes": estimate.approximate_consumer_bytes,
        "source_stride_map": estimate.source_stride_map,
        "target_stride_maps": estimate.target_stride_maps,
    }


def _append_jsonl(path: str, estimates: list[InputFanoutEstimate]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for estimate in estimates:
            handle.write(json.dumps(_estimate_to_json(estimate), sort_keys=True) + "\n")


def input_fanout_telemetry(operations: list[Operation]) -> None:
    """Log graph-input/weight/constant fanout attribution after work distribution."""
    if not config.input_fanout_telemetry:
        return

    name_to_op = build_name_to_op_map(operations)
    consumers_by_source: dict[str, list[ComputedBuffer]] = defaultdict(list)
    for consumer in operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        for dep in consumer.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            if dep.name in name_to_op:
                continue
            consumers_by_source[dep.name].append(consumer)

    estimates: list[InputFanoutEstimate] = []
    for source_name, consumers in consumers_by_source.items():
        source_kind, source_stride_map = _source_kind(source_name)
        if source_kind not in {
            "graph_input_or_weight",
            "constant_or_extern",
            "mutation_target",
            "unknown",
        }:
            continue

        consumer_names = [consumer.get_name() for consumer in consumers]
        restickify_consumers = [
            consumer.get_name() for consumer in consumers if is_restickify_op(consumer)
        ]
        target_stride_maps = []
        for consumer in consumers:
            stride_map = _stride_map_from_buffer(consumer)
            if stride_map is not None and stride_map not in target_stride_maps:
                target_stride_maps.append(stride_map)

        estimates.append(
            InputFanoutEstimate(
                source_name=source_name,
                source_kind=source_kind,
                consumer_count=len(consumers),
                consumers=consumer_names,
                consumer_kinds=dict(Counter(_op_kind(consumer) for consumer in consumers)),
                restickify_consumers=restickify_consumers,
                restickify_bytes_moved=sum(
                    _bytes_moved_or_zero(consumer)
                    for consumer in consumers
                    if is_restickify_op(consumer)
                ),
                approximate_consumer_bytes=sum(
                    _bytes_moved_or_zero(consumer) for consumer in consumers
                ),
                source_stride_map=source_stride_map,
                target_stride_maps=target_stride_maps,
            )
        )

    estimates.sort(
        key=lambda estimate: (
            estimate.restickify_bytes_moved,
            estimate.consumer_count,
            estimate.approximate_consumer_bytes,
        ),
        reverse=True,
    )

    total_restickify_bytes = sum(
        estimate.restickify_bytes_moved for estimate in estimates
    )
    for estimate in estimates[:20]:
        logger.info(
            "input_fanout source=%s kind=%s consumers=%d restickify_bytes=%d "
            "consumer_kinds=%s target_stride_maps=%s",
            estimate.source_name,
            estimate.source_kind,
            estimate.consumer_count,
            estimate.restickify_bytes_moved,
            estimate.consumer_kinds,
            estimate.target_stride_maps,
        )
    if estimates:
        logger.info(
            "input_fanout summary sources=%d total_restickify_bytes=%d",
            len(estimates),
            total_restickify_bytes,
        )

    _append_jsonl(config.input_fanout_telemetry_jsonl, estimates)

