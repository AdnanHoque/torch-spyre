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

"""Tier 1 same-stick on-chip handoff planner.

This pass is deliberately a planner, not a lowering replacement. It identifies
in-graph producer -> consumer edges where both sides already use a compatible
stick layout but the committed work divisions imply modeled RIU byte-hops. A
future Deeptools Foundation contract can realize those plans as mixed
data-op/DL-op SuperDSCs. Until then every plan explicitly fail-closes to the
stock path.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Operation

from . import config
from .core_continuity_telemetry import edge_symbol_map
from .logging_utils import get_inductor_logger
from .restickify_ring import (
    _bytes_moved_or_zero,
    _element_size_bytes,
    _mapping_for_op,
    _op_kind,
    build_name_to_op_map,
    decode_op_splits,
    estimate_byte_hops_from_mappings,
    is_restickify_op,
    op_iteration_sizes,
    split_dims_only,
)

logger = get_inductor_logger("on_chip_handoff")

ON_CHIP_HANDOFF_ATTR = "_spyre_on_chip_handoff_plan"


@dataclasses.dataclass(frozen=True)
class OnChipHandoffEstimate:
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
    status: str
    skip_reason: str | None
    realization_status: str
    foundation_contract_available: bool
    same_bundle_required: bool = True
    source_kind: str = "in_graph_computed"
    transport_kind: str = "same-stick-lx-to-lx"
    context: dict[str, Any] = dataclasses.field(default_factory=dict)


def realization_status(foundation_contract_available: bool) -> str:
    """Return the realization state for a Tier 1 plan."""

    if foundation_contract_available:
        return "planned-foundation-contract-present-codegen-not-enabled"
    return "blocked-missing-foundation-contract"


def _telemetry_context() -> dict[str, Any]:
    raw = os.environ.get("SPYRE_TELEMETRY_CONTEXT")
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"label": raw}
    return decoded if isinstance(decoded, dict) else {"value": decoded}


def _estimate_to_json(estimate: OnChipHandoffEstimate) -> dict[str, Any]:
    return {
        "context": estimate.context,
        "case": estimate.context.get("case"),
        "size": estimate.context.get("size"),
        "scenario": estimate.context.get("scenario"),
        "source_name": estimate.source_name,
        "source_kind": estimate.source_kind,
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
        "status": estimate.status,
        "skip_reason": estimate.skip_reason,
        "transport_kind": estimate.transport_kind,
        "same_bundle_required": estimate.same_bundle_required,
        "foundation_contract_available": estimate.foundation_contract_available,
        "realization_status": estimate.realization_status,
        "requirements": {
            "mixed_dataop_dlop_superdsc": True,
            "producer_lx_lifetime_through_consumer": True,
            "dataop_output_to_consumer_labeled_ds_binding": True,
            "stock_hbm_fallback": True,
        },
    }


def _append_jsonl(path: str, estimates: list[OnChipHandoffEstimate]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for estimate in estimates:
            handle.write(json.dumps(_estimate_to_json(estimate), sort_keys=True) + "\n")


def _skip_estimate(
    *,
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    read_dep: MemoryDep,
    reason: str,
    symbol_map: dict[str, str] | None = None,
) -> OnChipHandoffEstimate:
    return OnChipHandoffEstimate(
        source_name=read_dep.name,
        producer_name=producer.get_name(),
        consumer_name=consumer.get_name(),
        producer_kind=_op_kind(producer),
        consumer_kind=_op_kind(consumer),
        bytes_moved=_bytes_moved_or_zero(producer),
        byte_hops=0,
        avg_hops=0.0,
        max_hops=0,
        producer_splits=split_dims_only(decode_op_splits(producer)),
        consumer_splits=split_dims_only(decode_op_splits(consumer)),
        symbol_map=symbol_map or {},
        status="skipped",
        skip_reason=reason,
        foundation_contract_available=config.on_chip_handoff_foundation_contract,
        realization_status="not-realized-skipped",
        context=_telemetry_context(),
    )


def plan_on_chip_handoff_edge(
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    read_dep: MemoryDep,
    *,
    ring_size: int,
    k_fast_ops: list[Operation] | None = None,
) -> OnChipHandoffEstimate:
    """Plan one same-stick producer -> consumer LX handoff candidate."""

    if is_restickify_op(producer) or is_restickify_op(consumer):
        return _skip_estimate(
            producer=producer,
            consumer=consumer,
            read_dep=read_dep,
            reason="stick-changing-restickify-edge-is-tier2",
        )

    symbol_map, reason = edge_symbol_map(producer, consumer, read_dep)
    if reason is not None:
        return _skip_estimate(
            producer=producer,
            consumer=consumer,
            read_dep=read_dep,
            reason=reason,
        )

    try:
        producer_sizes = op_iteration_sizes(producer)
        consumer_sizes = op_iteration_sizes(consumer)
        producer_splits = decode_op_splits(producer)
        consumer_splits = decode_op_splits(consumer)
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
        return _skip_estimate(
            producer=producer,
            consumer=consumer,
            read_dep=read_dep,
            reason=type(exc).__name__,
            symbol_map=symbol_map,
        )

    if byte_hops == 0:
        return _skip_estimate(
            producer=producer,
            consumer=consumer,
            read_dep=read_dep,
            reason="already-core-local",
            symbol_map=symbol_map,
        )

    return OnChipHandoffEstimate(
        source_name=read_dep.name,
        producer_name=producer.get_name(),
        consumer_name=consumer.get_name(),
        producer_kind=_op_kind(producer),
        consumer_kind=_op_kind(consumer),
        bytes_moved=bytes_moved,
        byte_hops=byte_hops,
        avg_hops=byte_hops / bytes_moved if bytes_moved else 0.0,
        max_hops=max_hops,
        producer_splits=split_dims_only(producer_splits),
        consumer_splits=split_dims_only(consumer_splits),
        symbol_map=symbol_map,
        status="planned",
        skip_reason=None,
        foundation_contract_available=config.on_chip_handoff_foundation_contract,
        realization_status=realization_status(
            config.on_chip_handoff_foundation_contract
        ),
        context=_telemetry_context(),
    )


def plan_on_chip_handoffs(
    operations: list[Operation],
    k_fast_ops: list[Operation] | None = None,
) -> None:
    """Find same-stick in-graph handoff candidates after work distribution."""

    if not config.on_chip_handoff_planning:
        return

    name_to_op = build_name_to_op_map(operations)
    estimates: list[OnChipHandoffEstimate] = []

    for consumer in operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        consumer_payloads: dict[str, dict[str, Any]] = {}
        for dep in consumer.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            producer = name_to_op.get(dep.name)
            if producer is None:
                continue
            estimate = plan_on_chip_handoff_edge(
                producer,
                consumer,
                dep,
                ring_size=config.sencores,
                k_fast_ops=k_fast_ops,
            )
            estimates.append(estimate)
            if estimate.status == "planned":
                consumer_payloads[estimate.source_name] = _estimate_to_json(estimate)

        if consumer_payloads:
            setattr(consumer, ON_CHIP_HANDOFF_ATTR, consumer_payloads)

    estimates.sort(key=lambda estimate: estimate.byte_hops, reverse=True)
    planned = [estimate for estimate in estimates if estimate.status == "planned"]
    skipped = len(estimates) - len(planned)
    total_byte_hops = sum(estimate.byte_hops for estimate in planned)
    total_bytes = sum(estimate.bytes_moved for estimate in planned)

    for estimate in estimates[:20]:
        if estimate.status == "planned":
            logger.info(
                "on_chip_handoff planned producer=%s consumer=%s source=%s "
                "bytes=%d byte_hops=%d avg_hops=%.2f max_hops=%d "
                "producer_splits=%s consumer_splits=%s realization=%s",
                estimate.producer_name,
                estimate.consumer_name,
                estimate.source_name,
                estimate.bytes_moved,
                estimate.byte_hops,
                estimate.avg_hops,
                estimate.max_hops,
                estimate.producer_splits,
                estimate.consumer_splits,
                estimate.realization_status,
            )
        else:
            logger.info(
                "on_chip_handoff skipped producer=%s consumer=%s source=%s "
                "reason=%s producer_splits=%s consumer_splits=%s",
                estimate.producer_name,
                estimate.consumer_name,
                estimate.source_name,
                estimate.skip_reason,
                estimate.producer_splits,
                estimate.consumer_splits,
            )

    if estimates:
        logger.info(
            "on_chip_handoff summary total_edges=%d planned=%d skipped=%d "
            "planned_bytes=%d planned_byte_hops=%d realization=%s",
            len(estimates),
            len(planned),
            skipped,
            total_bytes,
            total_byte_hops,
            realization_status(config.on_chip_handoff_foundation_contract),
        )

    _append_jsonl(config.on_chip_handoff_plan_jsonl, estimates)
