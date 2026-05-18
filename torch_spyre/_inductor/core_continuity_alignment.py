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

"""Default-off producer-consumer core-division continuity prototype."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

from sympy import Symbol
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise

from . import config
from .core_continuity_telemetry import (
    CORE_CONTINUITY_ALIGNMENT_ATTR,
    edge_symbol_map,
)
from .logging_utils import get_inductor_logger
from .restickify_ring import (
    CORE_MAPPING_OVERRIDE_ATTR,
    _element_size_bytes,
    _mapping_for_op,
    build_name_to_op_map,
    decode_op_splits,
    estimate_byte_hops_from_mappings,
    is_restickify_op,
    op_iteration_sizes,
    producer_aligned_dim_order,
    split_dims_only,
)

logger = get_inductor_logger("core_continuity_alignment")


@dataclasses.dataclass(frozen=True)
class CoreContinuityAlignmentCandidate:
    source_name: str
    producer_name: str
    consumer_name: str
    override: dict[str, dict[str, int]] | None
    payload: dict[str, Any]
    reason: str | None = None


def split_factors_match_after_symbol_map(
    producer_splits: Mapping[str, int],
    consumer_splits: Mapping[str, int],
    symbol_map: Mapping[str, str],
) -> tuple[bool, str | None]:
    """Return true when producer and consumer use identical mapped splits."""
    producer_core_count = math.prod(producer_splits.values())
    consumer_core_count = math.prod(consumer_splits.values())
    if producer_core_count != consumer_core_count:
        return False, "different-core-count"

    reverse_symbol_map = {
        producer_sym: consumer_sym
        for consumer_sym, producer_sym in symbol_map.items()
    }
    for producer_sym, producer_split in producer_splits.items():
        consumer_sym = reverse_symbol_map.get(producer_sym)
        consumer_split = consumer_splits.get(consumer_sym, 1)
        if producer_split != consumer_split:
            return False, "different-split-factors"

    for consumer_sym, consumer_split in consumer_splits.items():
        producer_sym = symbol_map.get(consumer_sym)
        producer_split = producer_splits.get(producer_sym, 1)
        if consumer_split != producer_split:
            return False, "different-split-factors"

    return True, None


def producer_mapping_override_for_consumer(
    producer_mapping: Mapping[str, Mapping[str, int]],
    consumer_sizes: Mapping[str, int],
    symbol_map: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    """Convert a producer core mapping into consumer iteration symbols."""
    override: dict[str, dict[str, int]] = {}
    for core_id, producer_slices in producer_mapping.items():
        per_dim: dict[str, int] = {}
        for consumer_sym in consumer_sizes:
            producer_sym = symbol_map.get(consumer_sym)
            per_dim[consumer_sym] = (
                int(producer_slices.get(producer_sym, 0))
                if producer_sym is not None
                else 0
            )
        override[str(core_id)] = per_dim
    return override


def _payload(
    *,
    continuity_aligned: bool,
    continuity_assertion: str,
    continuity_skip_reason: str | None,
    baseline_byte_hops: int | None,
    aligned_byte_hops: int | None,
) -> dict[str, Any]:
    return {
        "continuity_aligned": continuity_aligned,
        "continuity_assertion": continuity_assertion,
        "continuity_skip_reason": continuity_skip_reason,
        "baseline_byte_hops": baseline_byte_hops,
        "aligned_byte_hops": aligned_byte_hops,
    }


def _eligible_consumer(op: ComputedBuffer) -> bool:
    return isinstance(op.data, Pointwise) and not is_restickify_op(op)


def core_continuity_dim_order(
    op: ComputedBuffer,
    output_dims: Sequence[Symbol],
    name_to_op: Mapping[str, ComputedBuffer] | None,
) -> tuple[list[Symbol] | None, str | None]:
    """Return a producer-aligned pointwise output-dim order when unambiguous."""
    if not config.align_core_division_continuity:
        return None, "disabled"
    if name_to_op is None:
        return None, "missing-name-map"
    if not _eligible_consumer(op):
        return None, "unsupported-consumer"

    candidates: list[list[Symbol]] = []
    for dep in op.get_read_writes().reads:
        if not isinstance(dep, MemoryDep):
            continue
        producer = name_to_op.get(dep.name)
        if producer is None:
            continue
        symbol_map, reason = edge_symbol_map(producer, op, dep)
        if reason is not None:
            continue
        producer_splits = decode_op_splits(producer)
        if len(split_dims_only(producer_splits)) != 1:
            continue
        prioritized, reason = producer_aligned_dim_order(
            output_dims,
            producer_splits,
            symbol_map,
        )
        if prioritized is not None:
            candidates.append(prioritized)

    if not candidates:
        return None, "no-compatible-producer-edge"

    preferred = candidates[0][0]
    if any(candidate[0] != preferred for candidate in candidates[1:]):
        return None, "conflicting-preferred-dims"
    return candidates[0], None


def build_core_continuity_mapping_candidate(
    producer: ComputedBuffer,
    consumer: ComputedBuffer,
    read_dep: MemoryDep,
    ring_size: int,
    k_fast_ops: list[Operation] | None = None,
) -> CoreContinuityAlignmentCandidate:
    """Build and certify a zero-hop producer-aligned override for one edge."""
    source_name = read_dep.name
    producer_name = producer.get_name()
    consumer_name = consumer.get_name()

    symbol_map, reason = edge_symbol_map(producer, consumer, read_dep)
    if reason is not None:
        return CoreContinuityAlignmentCandidate(
            source_name,
            producer_name,
            consumer_name,
            None,
            _payload(
                continuity_aligned=False,
                continuity_assertion="skipped",
                continuity_skip_reason=reason,
                baseline_byte_hops=None,
                aligned_byte_hops=None,
            ),
            reason,
        )

    producer_splits = decode_op_splits(producer)
    consumer_splits = decode_op_splits(consumer)
    if len(split_dims_only(producer_splits)) != 1:
        return CoreContinuityAlignmentCandidate(
            source_name,
            producer_name,
            consumer_name,
            None,
            _payload(
                continuity_aligned=False,
                continuity_assertion="skipped",
                continuity_skip_reason="producer-not-single-split",
                baseline_byte_hops=None,
                aligned_byte_hops=None,
            ),
            "producer-not-single-split",
        )
    if len(split_dims_only(consumer_splits)) != 1:
        return CoreContinuityAlignmentCandidate(
            source_name,
            producer_name,
            consumer_name,
            None,
            _payload(
                continuity_aligned=False,
                continuity_assertion="skipped",
                continuity_skip_reason="consumer-not-single-split",
                baseline_byte_hops=None,
                aligned_byte_hops=None,
            ),
            "consumer-not-single-split",
        )
    matched, reason = split_factors_match_after_symbol_map(
        producer_splits,
        consumer_splits,
        symbol_map,
    )
    if not matched:
        return CoreContinuityAlignmentCandidate(
            source_name,
            producer_name,
            consumer_name,
            None,
            _payload(
                continuity_aligned=False,
                continuity_assertion="skipped",
                continuity_skip_reason=reason,
                baseline_byte_hops=None,
                aligned_byte_hops=None,
            ),
            reason,
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
        _, baseline_byte_hops, _ = estimate_byte_hops_from_mappings(
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
        override = producer_mapping_override_for_consumer(
            producer_mapping,
            consumer_sizes,
            symbol_map,
        )
        _, aligned_byte_hops, _ = estimate_byte_hops_from_mappings(
            producer_sizes,
            consumer_sizes,
            producer_splits,
            consumer_splits,
            producer_mapping,
            override,
            symbol_map,
            elem_size,
            ring_size,
        )
    except Exception as exc:  # noqa: BLE001
        reason = type(exc).__name__
        return CoreContinuityAlignmentCandidate(
            source_name,
            producer_name,
            consumer_name,
            None,
            _payload(
                continuity_aligned=False,
                continuity_assertion="skipped",
                continuity_skip_reason=reason,
                baseline_byte_hops=None,
                aligned_byte_hops=None,
            ),
            reason,
        )

    if baseline_byte_hops == 0:
        return CoreContinuityAlignmentCandidate(
            source_name,
            producer_name,
            consumer_name,
            None,
            _payload(
                continuity_aligned=False,
                continuity_assertion="skipped",
                continuity_skip_reason="already-local",
                baseline_byte_hops=baseline_byte_hops,
                aligned_byte_hops=aligned_byte_hops,
            ),
            "already-local",
        )
    if aligned_byte_hops != 0:
        return CoreContinuityAlignmentCandidate(
            source_name,
            producer_name,
            consumer_name,
            None,
            _payload(
                continuity_aligned=False,
                continuity_assertion="failed",
                continuity_skip_reason="nonzero-aligned-byte-hops",
                baseline_byte_hops=baseline_byte_hops,
                aligned_byte_hops=aligned_byte_hops,
            ),
            "nonzero-aligned-byte-hops",
        )

    return CoreContinuityAlignmentCandidate(
        source_name,
        producer_name,
        consumer_name,
        override,
        _payload(
            continuity_aligned=True,
            continuity_assertion="passed",
            continuity_skip_reason=None,
            baseline_byte_hops=baseline_byte_hops,
            aligned_byte_hops=aligned_byte_hops,
        ),
        None,
    )


def align_core_continuity_mappings(
    operations: list[Operation],
    k_fast_ops: list[Operation] | None = None,
) -> None:
    """Attach certified producer-aligned core mappings to pointwise consumers."""
    if not config.align_core_division_continuity:
        return

    name_to_op = build_name_to_op_map(operations)
    for consumer in operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        if not _eligible_consumer(consumer):
            continue
        if getattr(consumer, CORE_MAPPING_OVERRIDE_ATTR, None) is not None:
            continue

        candidates: list[CoreContinuityAlignmentCandidate] = []
        for dep in consumer.get_read_writes().reads:
            if not isinstance(dep, MemoryDep):
                continue
            producer = name_to_op.get(dep.name)
            if producer is None:
                continue
            candidates.append(
                build_core_continuity_mapping_candidate(
                    producer,
                    consumer,
                    dep,
                    ring_size=config.sencores,
                    k_fast_ops=k_fast_ops,
                )
            )

        if not candidates:
            continue

        payloads = {candidate.source_name: candidate.payload for candidate in candidates}
        attachable = [
            candidate for candidate in candidates if candidate.override is not None
        ]
        if not attachable:
            setattr(consumer, CORE_CONTINUITY_ALIGNMENT_ATTR, payloads)
            continue

        override = attachable[0].override
        assert override is not None
        if any(candidate.override != override for candidate in attachable[1:]):
            for candidate in attachable:
                payloads[candidate.source_name] = _payload(
                    continuity_aligned=False,
                    continuity_assertion="skipped",
                    continuity_skip_reason="conflicting-core-mapping-overrides",
                    baseline_byte_hops=candidate.payload.get("baseline_byte_hops"),
                    aligned_byte_hops=candidate.payload.get("aligned_byte_hops"),
                )
            setattr(consumer, CORE_CONTINUITY_ALIGNMENT_ATTR, payloads)
            logger.info(
                "skip core continuity mapping alignment for %s: "
                "conflicting core mapping overrides",
                consumer.get_name(),
            )
            continue

        setattr(consumer, CORE_MAPPING_OVERRIDE_ATTR, override)
        setattr(consumer, CORE_CONTINUITY_ALIGNMENT_ATTR, payloads)
        logger.info(
            "attached core continuity mapping override for %s (%d cores, %d edges)",
            consumer.get_name(),
            len(override),
            len(attachable),
        )
