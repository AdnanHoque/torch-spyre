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

"""Phase 0 telemetry for ring-aware restickify (RFC draft).

Read-only diagnostic pass. For each restickify op, logs producer/consumer
identity and op_it_space_splits so we can see whether the two ends of the
restickify agree on a core mapping. Off by default; enable with
SPYRE_RESTICKIFY_TELEMETRY=1.

This is the MVP for Phase 0 of the Ring-Aware Restickify RFC. Hop-cost
math comes later once we can confirm the plumbing works end-to-end.
"""

from __future__ import annotations

import torch
from torch._inductor.ir import ComputedBuffer, Operation

from . import config
from .logging_utils import get_inductor_logger
from .pass_utils import apply_splits_from_index_coeff, iteration_space_from_op

logger = get_inductor_logger("restickify_telemetry")


def _is_restickify(op: ComputedBuffer) -> bool:
    """True if op was inserted by the restickify pass.

    insert_restickify sets op.origins to the synthetic FX node it created
    with target torch.ops.spyre.restickify.default. That's the cleanest
    structural signal pre-codegen.
    """
    origins = getattr(op, "origins", None)
    if not origins:
        return False
    for o in origins:
        if not isinstance(o, torch.fx.Node):
            continue
        target = o.target
        # FX targets for spyre.restickify show up as the OpOverload itself.
        if target is torch.ops.spyre.restickify.default:
            return True
    return False


def _decode_splits(op: ComputedBuffer) -> dict[str, int] | None:
    """Decode op.op_it_space_splits to per-symbol splits.

    Returns None if the op has no splits assigned (e.g. graph inputs, or
    work_distribution_pass declined to split because cores_used <= 1).
    Returns {} for ops with op_it_space_splits set but all values == 1.
    """
    encoded = getattr(op, "op_it_space_splits", None)
    if encoded is None:
        return None
    try:
        rw = op.get_read_writes()
        write_index = next(iter(rw.writes)).index
        read_index = next((d.index for d in rw.reads), write_index)
        it_space = iteration_space_from_op(op)
        per_dim = apply_splits_from_index_coeff(
            encoded, write_index, read_index, it_space
        )
        # Drop unsplit dims for readability; sort for deterministic output.
        return {
            str(s): v
            for s, v in sorted(per_dim.items(), key=lambda kv: str(kv[0]))
            if v > 1
        }
    except Exception as e:  # noqa: BLE001
        return {"<decode-error>": type(e).__name__}  # type: ignore[dict-item]


def _alignment_status(producer_splits: dict | None, self_splits: dict | None) -> str:
    """Classify producer/consumer mapping alignment.

    - 'no-producer': producer is a graph input (no op_it_space_splits)
    - 'aligned':     same dim symbols split the same way (zero-hop restickify)
    - 'mismatched':  different dim symbols → orthogonal mapping, worst case
    - 'partial':     some dims match, some don't
    """
    if producer_splits is None:
        return "no-producer"
    if not producer_splits and not self_splits:
        return "aligned"
    if producer_splits == self_splits:
        return "aligned"
    common = set(producer_splits or {}) & set(self_splits or {})
    if not common:
        return "mismatched"
    return "partial"


def restickify_telemetry(operations: list[Operation]) -> None:
    """Diagnostic pass: log each restickify with producer/consumer mappings.

    Runs after work_distribution (and k_fast_override), so op_it_space_splits
    is final on every ComputedBuffer.
    """
    if not config.restickify_telemetry:
        return

    # Map buffer name -> producing ComputedBuffer for fast producer lookup.
    name_to_op: dict[str, ComputedBuffer] = {}
    for op in operations:
        if isinstance(op, ComputedBuffer):
            name_to_op[op.get_name()] = op

    # Build consumer index: for each buffer name, which ops read it.
    consumers_of: dict[str, list[ComputedBuffer]] = {}
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        rw = op.get_read_writes()
        for dep in rw.reads:
            consumers_of.setdefault(dep.name, []).append(op)

    restick_count = 0
    for op in operations:
        if not isinstance(op, ComputedBuffer):
            continue
        if not _is_restickify(op):
            continue
        restick_count += 1

        my_name = op.get_name()
        rw = op.get_read_writes()
        producer_names = [d.name for d in rw.reads]
        producers = [name_to_op[n] for n in producer_names if n in name_to_op]
        consumers = consumers_of.get(my_name, [])

        self_splits = _decode_splits(op)
        if producers:
            producer = producers[0]
            producer_name = producer.get_name()
            producer_splits = _decode_splits(producer)
        else:
            # Producer is a graph input (placeholder) or an unmapped op.
            producer_name = producer_names[0] if producer_names else "<no-input>"
            producer_splits = None
        status = _alignment_status(producer_splits, self_splits)

        logger.info(
            "restickify=%s status=%s producer=%s producer_splits=%s "
            "consumers=%s self_splits=%s",
            my_name,
            status,
            producer_name,
            producer_splits if producer_splits is not None else "<none>",
            [c.get_name() for c in consumers],
            self_splits if self_splits is not None else "<none>",
        )

    if restick_count > 0:
        logger.info("total restickify ops: %d", restick_count)
