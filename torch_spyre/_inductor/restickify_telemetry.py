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

"""Default-off byte-hop telemetry for restickify ops."""

from __future__ import annotations

import json
from pathlib import Path

from torch._inductor.ir import ComputedBuffer, Operation

from . import config
from .logging_utils import get_inductor_logger
from .restickify_ring import (
    RestickifyRingEstimate,
    build_consumers_of,
    build_name_to_op_map,
    estimate_restickify_ring_cost,
    is_restickify_op,
)

logger = get_inductor_logger("restickify_telemetry")


def _estimate_to_json(estimate: RestickifyRingEstimate) -> dict:
    return {
        "restickify": estimate.restickify_name,
        "producer": estimate.producer_name,
        "source_name": estimate.source_name,
        "source_kind": estimate.source_kind,
        "consumer": estimate.consumer_name,
        "consumer_kind": estimate.consumer_kind,
        "consumers": estimate.consumer_names,
        "bytes_moved": estimate.bytes_moved,
        "byte_hops": estimate.byte_hops,
        "avg_hops": estimate.avg_hops,
        "max_hops": estimate.max_hops,
        "producer_splits": estimate.producer_splits,
        "restickify_splits": estimate.restickify_splits,
        "symbol_map": estimate.symbol_map,
        "target_stride_map": estimate.target_stride_map,
        "source_stride_map": estimate.source_stride_map,
        "skip_reason": estimate.skip_reason,
    }


def _append_jsonl(path: str, estimates: list[RestickifyRingEstimate]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for estimate in estimates:
            handle.write(json.dumps(_estimate_to_json(estimate), sort_keys=True) + "\n")


def restickify_ring_telemetry(
    operations: list[Operation],
    k_fast_ops: list[Operation] | None = None,
) -> None:
    """Log restickify byte-hop estimates after work distribution."""
    if not config.restickify_ring_telemetry:
        return

    name_to_op = build_name_to_op_map(operations)
    consumers_of = build_consumers_of(operations)
    estimates: list[RestickifyRingEstimate] = []

    for op in operations:
        if not isinstance(op, ComputedBuffer) or not is_restickify_op(op):
            continue
        estimates.append(
            estimate_restickify_ring_cost(
                op,
                name_to_op,
                consumers_of,
                ring_size=config.sencores,
                k_fast_ops=k_fast_ops,
            )
        )

    estimates.sort(key=lambda estimate: estimate.byte_hops, reverse=True)
    total_byte_hops = sum(estimate.byte_hops for estimate in estimates)
    total_bytes = sum(estimate.bytes_moved for estimate in estimates)
    skipped = sum(1 for estimate in estimates if estimate.skip_reason is not None)

    for estimate in estimates:
        if estimate.skip_reason is not None:
            logger.info(
                "restickify_ring restickify=%s source=%s source_kind=%s "
                "producer=%s consumer=%s consumer_kind=%s skip_reason=%s "
                "producer_splits=%s restickify_splits=%s consumers=%s",
                estimate.restickify_name,
                estimate.source_name,
                estimate.source_kind,
                estimate.producer_name,
                estimate.consumer_name,
                estimate.consumer_kind,
                estimate.skip_reason,
                estimate.producer_splits,
                estimate.restickify_splits,
                estimate.consumer_names,
            )
            continue
        logger.info(
            "restickify_ring restickify=%s source=%s source_kind=%s "
            "producer=%s consumer=%s consumer_kind=%s byte_hops=%d "
            "bytes=%d avg_hops=%.2f max_hops=%d producer_splits=%s "
            "restickify_splits=%s symbol_map=%s consumers=%s",
            estimate.restickify_name,
            estimate.source_name,
            estimate.source_kind,
            estimate.producer_name,
            estimate.consumer_name,
            estimate.consumer_kind,
            estimate.byte_hops,
            estimate.bytes_moved,
            estimate.avg_hops,
            estimate.max_hops,
            estimate.producer_splits,
            estimate.restickify_splits,
            estimate.symbol_map,
            estimate.consumer_names,
        )

    if estimates:
        logger.info(
            "restickify_ring summary total_restickifies=%d skipped=%d "
            "total_bytes=%d total_byte_hops=%d avg_hops=%.2f",
            len(estimates),
            skipped,
            total_bytes,
            total_byte_hops,
            total_byte_hops / total_bytes if total_bytes else 0.0,
        )

    _append_jsonl(config.restickify_ring_telemetry_jsonl, estimates)
