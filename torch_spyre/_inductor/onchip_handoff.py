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

"""Tier 1: same-layout cross-core on-chip handoff planner.

A *same-layout cross-core handoff* is a producer -> consumer activation edge
where the producer output and the consumer input share the SAME stick layout
(so no restickify is inserted -- ``stick_compatible`` is True), BUT the two ops
are split across cores differently. Under such an edge the activation would
round-trip through HBM at the bundle boundary even though the data could stay
resident in LX and be shuffled core-to-core on the ring.

This module DETECTS those edges and PLANS the on-chip cross-core transfer that
would keep them in LX. It is a planner ONLY: it records plans and emits
telemetry but does NOT modify the graph, layouts, splits, or values. Realizing
the transfer needs a deeptools Foundation contract that is out of scope here,
so every plan FAILS CLOSED (``realizable`` is always ``False``).

The pure transfer cost math lives in :mod:`.restickify_cost` and the
IR-coupled glue (op maps, split decoding, symbol correspondence) lives in
:mod:`.restickify_ring`; both are reused here rather than reimplemented.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Operation

from . import config
from .logging_utils import get_inductor_logger
from .pass_utils import device_coordinates, stick_compatible
from .restickify_cost import (
    build_transfer_plan,
    materialize_default_core_mapping,
)
from .restickify_ring import (
    build_consumers_of,
    build_name_to_op_map,
    build_symbol_correspondence,
    decode_op_splits,
    extract_strides,
    is_restickify_op,
    op_iteration_sizes,
)

logger = get_inductor_logger("onchip_handoff")

# Realizing the on-chip transfer requires a deeptools-side data-movement
# program (the Foundation contract). Until that exists every plan fails closed.
FAIL_CLOSED_REASON = "needs-deeptools-foundation-contract"


@dataclasses.dataclass(frozen=True)
class OnChipHandoffPlan:
    """A planned (but unrealized) same-layout cross-core LX handoff.

    Records one producer -> consumer activation edge whose stick layout is
    shared but whose per-core ownership diverges, together with the cross-core
    transfer plan that would keep the activation resident in LX.
    """

    producer_name: str
    consumer_name: str
    shared_stick_dim: str | None
    producer_splits: dict[str, int]
    consumer_splits: dict[str, int]
    symbol_map: dict[str, str]
    transfers: int
    local_elements: int
    remote_elements: int
    total_byte_hops: int
    max_hops: int
    bytes_moved: int
    realizable: bool = False
    fail_closed_reason: str = FAIL_CLOSED_REASON


def _producer_write_dep(producer: ComputedBuffer) -> MemoryDep | None:
    """Return the producer's single output MemoryDep, or None if not unique."""
    writes = [
        dep
        for dep in producer.get_read_writes().writes
        if isinstance(dep, MemoryDep)
    ]
    if len(writes) != 1:
        return None
    return writes[0]


def _shared_stick_dim(
    producer_coords: list[Any],
    consumer_coords: list[Any],
) -> str | None:
    """Return the single shared stick symbol name, or None when absent."""
    stick_syms = set()
    for coords in (producer_coords, consumer_coords):
        stick_syms |= set(coords[-1].free_symbols)
    if len(stick_syms) != 1:
        return None
    return str(next(iter(stick_syms)))


def _consumer_to_producer_symbol_map(
    producer: ComputedBuffer,
    producer_write: MemoryDep,
    consumer_read: MemoryDep,
) -> tuple[dict[str, str], str | None]:
    """Map consumer iteration symbols to producer symbols via buffer strides.

    Mirrors the restickify_ring approach: the activation is the producer's
    output buffer, so a consumer read symbol corresponds to the producer write
    symbol that indexes the same buffer stride. The returned map is keyed by
    consumer symbol (the build_transfer_plan ``symbol_map`` convention maps the
    consumer/restickify side to the producer side).
    """
    producer_strides = extract_strides(
        producer_write.index, producer_write.var_names
    )
    consumer_strides = extract_strides(consumer_read.index, consumer_read.var_names)
    return build_symbol_correspondence(producer_strides, consumer_strides)


def _ownership_identical(
    producer_splits: Mapping[str, int],
    consumer_splits: Mapping[str, int],
    symbol_map: Mapping[str, str],
) -> bool:
    """True when producer and consumer split every shared dim the same way.

    Same core count AND a matching split factor on every mapped dim means each
    core already owns the same slice on both sides -- the edge is local and
    there is nothing to plan.
    """
    if math.prod(producer_splits.values()) != math.prod(consumer_splits.values()):
        return False
    for consumer_sym, producer_sym in symbol_map.items():
        if consumer_splits.get(consumer_sym, 1) != producer_splits.get(
            producer_sym, 1
        ):
            return False
    # Any unmapped split on either side also counts as divergent ownership.
    mapped_producer = set(symbol_map.values())
    for producer_sym, split in producer_splits.items():
        if split > 1 and producer_sym not in mapped_producer:
            return False
    for consumer_sym, split in consumer_splits.items():
        if split > 1 and consumer_sym not in symbol_map:
            return False
    return True


def _plan_edge(
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    consumer_read: MemoryDep,
    ring_size: int,
) -> OnChipHandoffPlan | None:
    """Plan one producer -> consumer edge, or None if it should be skipped."""
    producer_write = _producer_write_dep(producer)
    if producer_write is None:
        return None

    producer_coords = device_coordinates(producer.get_layout(), producer_write)
    consumer_coords = device_coordinates(consumer.get_layout(), consumer_read)

    # Tier 1 only owns SAME-stick edges. A restickify-needed edge (different
    # stick layout) is Tier 2's territory -- skip it.
    if not stick_compatible([producer_coords, consumer_coords]):
        return None

    symbol_map, reason = _consumer_to_producer_symbol_map(
        producer, producer_write, consumer_read
    )
    if reason is not None:
        return None

    producer_splits = decode_op_splits(producer)
    consumer_splits = decode_op_splits(consumer)

    # Identical ownership => already local, nothing to hand off.
    if _ownership_identical(producer_splits, consumer_splits, symbol_map):
        return None

    producer_sizes = op_iteration_sizes(producer)
    consumer_sizes = op_iteration_sizes(consumer)
    producer_mapping = materialize_default_core_mapping(
        list(producer_sizes.keys()),
        producer_splits,
        math.prod(producer_splits.values()),
    )
    consumer_mapping = materialize_default_core_mapping(
        list(consumer_sizes.keys()),
        consumer_splits,
        math.prod(consumer_splits.values()),
    )

    transfers, summary = build_transfer_plan(
        producer_sizes,
        consumer_sizes,
        producer_splits,
        consumer_splits,
        producer_mapping,
        consumer_mapping,
        symbol_map,
        ring_size,
    )

    # No remote movement => the divergent splits still landed every tile on
    # its owning core. Nothing to plan.
    if summary["remote_elements"] <= 0:
        return None

    elem_size = _element_size_bytes(producer)
    return OnChipHandoffPlan(
        producer_name=producer.get_name(),
        consumer_name=consumer.get_name(),
        shared_stick_dim=_shared_stick_dim(producer_coords, consumer_coords),
        producer_splits={s: v for s, v in producer_splits.items() if v > 1},
        consumer_splits={s: v for s, v in consumer_splits.items() if v > 1},
        symbol_map=dict(symbol_map),
        transfers=summary["total_transfers"],
        local_elements=summary["local_elements"],
        remote_elements=summary["remote_elements"],
        total_byte_hops=summary["total_byte_hops"] * elem_size,
        max_hops=summary["max_hops"],
        bytes_moved=summary["remote_elements"] * elem_size,
    )


def _element_size_bytes(op: ComputedBuffer) -> int:
    dtype = op.get_layout().dtype
    itemsize = getattr(dtype, "itemsize", None)
    return int(itemsize) if itemsize is not None else 2


def plan_onchip_handoffs(
    operations: list[Operation],
    sencores: int,
) -> list[OnChipHandoffPlan]:
    """Detect same-layout cross-core handoff edges and plan their LX transfers.

    Gated on ``config.onchip_handoff_planner``. Walks every in-graph
    producer -> consumer activation edge; for each edge whose stick layout is
    shared (no restickify) but whose per-core ownership diverges, records an
    :class:`OnChipHandoffPlan`. Per-edge work is wrapped so one bad edge cannot
    crash compilation. This is a pure observer: it never mutates the graph.
    """
    if not config.onchip_handoff_planner:
        return []

    name_to_op = build_name_to_op_map(operations)
    consumers_of = build_consumers_of(operations)

    plans: list[OnChipHandoffPlan] = []
    for consumer in operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        for read_dep in consumer.get_read_writes().reads:
            if not isinstance(read_dep, MemoryDep):
                continue
            producer = name_to_op.get(read_dep.name)
            if producer is None:
                # Graph input / weight / extern -- no in-graph producer split.
                continue
            try:
                plan = _plan_edge(producer, consumer, read_dep, sencores)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "onchip_handoff skipping edge %s -> %s: %s: %s",
                    producer.get_name(),
                    consumer.get_name(),
                    type(exc).__name__,
                    exc,
                )
                continue
            if plan is not None:
                plans.append(plan)

    # consumers_of is built so future stages can fan a producer's plan across
    # all of its consumers; the per-edge walk above already covers them.
    del consumers_of
    return plans


def _plan_to_json(plan: OnChipHandoffPlan) -> dict[str, Any]:
    return {
        "producer": plan.producer_name,
        "consumer": plan.consumer_name,
        "shared_stick_dim": plan.shared_stick_dim,
        "producer_splits": plan.producer_splits,
        "consumer_splits": plan.consumer_splits,
        "symbol_map": plan.symbol_map,
        "transfers": plan.transfers,
        "local_elements": plan.local_elements,
        "remote_elements": plan.remote_elements,
        "bytes_moved": plan.bytes_moved,
        "byte_hops": plan.total_byte_hops,
        "max_hops": plan.max_hops,
        "realizable": plan.realizable,
        "fail_closed_reason": plan.fail_closed_reason,
    }


def emit_onchip_handoff_telemetry(plans: list[OnChipHandoffPlan]) -> None:
    """Append one JSON line per plan to the configured telemetry path."""
    path = config.onchip_handoff_telemetry_jsonl
    if not path or not plans:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(_plan_to_json(plan), sort_keys=True) + "\n")


def run_onchip_handoff_planner(
    operations: list[Operation],
    sencores: int,
) -> list[OnChipHandoffPlan]:
    """Top-level entrypoint: plan same-layout cross-core handoffs + telemetry.

    Pure observer -- it plans and reports but performs NO graph mutation. Safe
    to call from the pre-scheduling pass sequence after work division.
    """
    plans = plan_onchip_handoffs(operations, sencores)
    if plans:
        total_bytes = sum(plan.bytes_moved for plan in plans)
        total_byte_hops = sum(plan.total_byte_hops for plan in plans)
        logger.info(
            "onchip_handoff summary edges=%d remote_bytes=%d byte_hops=%d "
            "(fail_closed=%s)",
            len(plans),
            total_bytes,
            total_byte_hops,
            FAIL_CLOSED_REASON,
        )
    emit_onchip_handoff_telemetry(plans)
    return plans


# is_restickify_op is imported for callers that want to exclude restickify
# buffers from their own edge walks; surfaced here as part of the public API.
__all__ = [
    "FAIL_CLOSED_REASON",
    "OnChipHandoffPlan",
    "emit_onchip_handoff_telemetry",
    "is_restickify_op",
    "plan_onchip_handoffs",
    "run_onchip_handoff_planner",
]
