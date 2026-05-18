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

"""Default-off telemetry for producer-consumer core ownership continuity."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Operation

from . import config
from .logging_utils import get_inductor_logger
from .restickify_ring import (
    _bytes_moved_or_zero,
    _element_size_bytes,
    _mapping_for_op,
    _op_kind,
    build_name_to_op_map,
    build_symbol_correspondence,
    decode_op_splits,
    estimate_byte_hops_from_mappings,
    extract_strides,
    op_iteration_sizes,
    split_dims_only,
)

logger = get_inductor_logger("core_continuity_telemetry")

CORE_CONTINUITY_ALIGNMENT_ATTR = "_spyre_core_continuity_alignment"


@dataclasses.dataclass(frozen=True)
class CoreContinuityEstimate:
    source_name: str
    producer_name: str
    consumer_name: str
    producer_kind: str
    consumer_kind: str
    bytes_moved: int
    byte_hops: int
    avg_hops: float
    max_hops: int
    producer_splits: dict[str, int]
    consumer_splits: dict[str, int]
    symbol_map: dict[str, str]
    context: dict[str, Any] = dataclasses.field(default_factory=dict)
    skip_reason: str | None = None
    continuity_aligned: bool = False
    continuity_assertion: str = "not-run"
    continuity_skip_reason: str | None = None
    baseline_byte_hops: int | None = None
    aligned_byte_hops: int | None = None


def edge_symbol_map(
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    read_dep: MemoryDep,
) -> tuple[dict[str, str], str | None]:
    """Map consumer iteration symbols to producer symbols for one buffer edge.

    The first telemetry version is intentionally strict. It only estimates exact
    byte-hops when the read edge describes the same logical tensor region on
    both sides. Matmul/reduction edges with extra consumer-only dimensions are
    reported with a skip reason and can be generalized after we inspect data.
    """
    producer_writes = [
        dep
        for dep in producer.get_read_writes().writes
        if isinstance(dep, MemoryDep)
    ]
    if len(producer_writes) != 1:
        return {}, "producer-write-unsupported"

    producer_write = producer_writes[0]
    producer_strides = extract_strides(producer_write.index, producer_write.var_names)
    consumer_strides = extract_strides(read_dep.index, read_dep.var_names)
    if not producer_strides or not consumer_strides:
        return {}, "empty-stride-map"

    symbol_map, reason = build_symbol_correspondence(
        producer_strides,
        consumer_strides,
    )
    if reason is not None:
        return {}, reason

    producer_sizes = op_iteration_sizes(producer)
    consumer_sizes = op_iteration_sizes(consumer)
    mapped_producer_symbols = set(symbol_map.values())

    missing_consumer = [
        sym
        for sym, size in consumer_sizes.items()
        if size > 1 and sym not in symbol_map
    ]
    missing_producer = [
        sym
        for sym, size in producer_sizes.items()
        if size > 1 and sym not in mapped_producer_symbols
    ]
    if missing_consumer or missing_producer:
        return {}, "incomplete-symbol-map"

    for consumer_sym, producer_sym in symbol_map.items():
        if consumer_sizes[consumer_sym] != producer_sizes[producer_sym]:
            return {}, "mismatched-symbol-size"
    return symbol_map, None


_edge_symbol_map = edge_symbol_map


def _alignment_payload(
    consumer: ComputedBuffer,
    source_name: str,
) -> dict[str, Any]:
    raw = getattr(consumer, CORE_CONTINUITY_ALIGNMENT_ATTR, None)
    if not isinstance(raw, dict):
        return {}
    payload = raw.get(source_name)
    return payload if isinstance(payload, dict) else {}


def estimate_core_continuity_edge(
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    read_dep: MemoryDep,
    ring_size: int,
    k_fast_ops: list[Operation] | None = None,
) -> CoreContinuityEstimate:
    """Estimate logical byte-hops for one in-graph producer-consumer edge."""
    source_name = read_dep.name
    producer_name = producer.get_name()
    consumer_name = consumer.get_name()
    producer_splits = decode_op_splits(producer)
    consumer_splits = decode_op_splits(consumer)
    bytes_moved = _bytes_moved_or_zero(producer)
    context = _telemetry_context()
    alignment_payload = _alignment_payload(consumer, source_name)

    symbol_map, reason = edge_symbol_map(producer, consumer, read_dep)
    if reason is not None:
        return CoreContinuityEstimate(
            source_name=source_name,
            producer_name=producer_name,
            consumer_name=consumer_name,
            producer_kind=_op_kind(producer),
            consumer_kind=_op_kind(consumer),
            bytes_moved=bytes_moved,
            byte_hops=0,
            avg_hops=0.0,
            max_hops=0,
            producer_splits=split_dims_only(producer_splits),
            consumer_splits=split_dims_only(consumer_splits),
            symbol_map={},
            context=context,
            skip_reason=reason,
            **alignment_payload,
        )

    try:
        producer_sizes = op_iteration_sizes(producer)
        consumer_sizes = op_iteration_sizes(consumer)
        elem_size = _element_size_bytes(producer)
        producer_mapping = _mapping_for_op(
            producer,
            producer_sizes,
            producer_splits,
            k_fast_ops,
        )
        consumer_mapping = _mapping_for_op(
            consumer,
            consumer_sizes,
            consumer_splits,
            k_fast_ops,
        )
        bytes_moved, byte_hops, max_hops = estimate_byte_hops_from_mappings(
            producer_sizes,
            consumer_sizes,
            producer_splits,
            consumer_splits,
            producer_mapping,
            consumer_mapping,
            symbol_map,
            elem_size,
            ring_size,
        )
    except Exception as exc:  # noqa: BLE001
        return CoreContinuityEstimate(
            source_name=source_name,
            producer_name=producer_name,
            consumer_name=consumer_name,
            producer_kind=_op_kind(producer),
            consumer_kind=_op_kind(consumer),
            bytes_moved=bytes_moved,
            byte_hops=0,
            avg_hops=0.0,
            max_hops=0,
            producer_splits=split_dims_only(producer_splits),
            consumer_splits=split_dims_only(consumer_splits),
            symbol_map=symbol_map,
            context=context,
            skip_reason=type(exc).__name__,
            **alignment_payload,
        )

    return CoreContinuityEstimate(
        source_name=source_name,
        producer_name=producer_name,
        consumer_name=consumer_name,
        producer_kind=_op_kind(producer),
        consumer_kind=_op_kind(consumer),
        bytes_moved=bytes_moved,
        byte_hops=byte_hops,
        avg_hops=byte_hops / bytes_moved if bytes_moved else 0.0,
        max_hops=max_hops,
        producer_splits=split_dims_only(producer_splits),
        consumer_splits=split_dims_only(consumer_splits),
        symbol_map=symbol_map,
        context=context,
        skip_reason=None,
        **alignment_payload,
    )


def _telemetry_context() -> dict[str, Any]:
    raw = os.environ.get("SPYRE_TELEMETRY_CONTEXT")
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"label": raw}
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def _estimate_to_json(estimate: CoreContinuityEstimate) -> dict[str, Any]:
    return {
        "context": estimate.context,
        "case": estimate.context.get("case"),
        "size": estimate.context.get("size"),
        "scenario": estimate.context.get("scenario"),
        "source_name": estimate.source_name,
        "producer": estimate.producer_name,
        "consumer": estimate.consumer_name,
        "producer_kind": estimate.producer_kind,
        "consumer_kind": estimate.consumer_kind,
        "bytes_moved": estimate.bytes_moved,
        "byte_hops": estimate.byte_hops,
        "avg_hops": estimate.avg_hops,
        "max_hops": estimate.max_hops,
        "producer_splits": estimate.producer_splits,
        "consumer_splits": estimate.consumer_splits,
        "symbol_map": estimate.symbol_map,
        "skip_reason": estimate.skip_reason,
        "continuity_aligned": estimate.continuity_aligned,
        "continuity_assertion": estimate.continuity_assertion,
        "continuity_skip_reason": estimate.continuity_skip_reason,
        "baseline_byte_hops": estimate.baseline_byte_hops,
        "aligned_byte_hops": estimate.aligned_byte_hops,
    }


def _append_jsonl(path: str, estimates: list[CoreContinuityEstimate]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for estimate in estimates:
            handle.write(json.dumps(_estimate_to_json(estimate), sort_keys=True) + "\n")


def core_continuity_telemetry(
    operations: list[Operation],
    k_fast_ops: list[Operation] | None = None,
) -> None:
    """Log producer-consumer byte-hop estimates after work distribution."""
    if not config.core_continuity_telemetry:
        return

    name_to_op = build_name_to_op_map(operations)
    estimates: list[CoreContinuityEstimate] = []
    for consumer in operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        for dep in consumer.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            producer = name_to_op.get(dep.name)
            if producer is None:
                continue
            estimates.append(
                estimate_core_continuity_edge(
                    producer,
                    consumer,
                    dep,
                    ring_size=config.sencores,
                    k_fast_ops=k_fast_ops,
                )
            )

    estimates.sort(key=lambda estimate: estimate.byte_hops, reverse=True)
    total_byte_hops = sum(estimate.byte_hops for estimate in estimates)
    total_bytes = sum(estimate.bytes_moved for estimate in estimates)
    skipped = sum(1 for estimate in estimates if estimate.skip_reason is not None)

    for estimate in estimates[:20]:
        if estimate.skip_reason is not None:
            logger.info(
                "core_continuity producer=%s consumer=%s source=%s "
                "skip_reason=%s producer_splits=%s consumer_splits=%s",
                estimate.producer_name,
                estimate.consumer_name,
                estimate.source_name,
                estimate.skip_reason,
                estimate.producer_splits,
                estimate.consumer_splits,
            )
            continue
        logger.info(
            "core_continuity producer=%s consumer=%s source=%s byte_hops=%d "
            "bytes=%d avg_hops=%.2f max_hops=%d producer_splits=%s "
            "consumer_splits=%s symbol_map=%s",
            estimate.producer_name,
            estimate.consumer_name,
            estimate.source_name,
            estimate.byte_hops,
            estimate.bytes_moved,
            estimate.avg_hops,
            estimate.max_hops,
            estimate.producer_splits,
            estimate.consumer_splits,
            estimate.symbol_map,
        )

    if estimates:
        logger.info(
            "core_continuity summary total_edges=%d skipped=%d "
            "total_bytes=%d total_byte_hops=%d avg_hops=%.2f",
            len(estimates),
            skipped,
            total_bytes,
            total_byte_hops,
            total_byte_hops / total_bytes if total_bytes else 0.0,
        )

    _append_jsonl(config.core_continuity_telemetry_jsonl, estimates)
