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

"""Plan LX relayouts from producer and consumer per-core views.

The regular LX planner handles same-core scratchpad persistence. This module
finds edges where a producer and consumer use different per-core views of the
same tensor. Accepted edges are materialized as an explicit, frontend-allocated
S1 -> SHUFFLE -> S2 sequence.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from enum import StrEnum

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.graph import GraphLowering
from torch._inductor.ir import ComputedBuffer, Operation, Pointwise, Reduction

from torch_spyre._inductor import config
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _is_matmul_op,
    _per_core_view_on_buf,
    op_read_writes,
)
from torch_spyre._inductor.scratchpad.utils import _op_num_cores, _op_short_name

LX_RELAYOUT_ATTR = "_spyre_lx_relayout_inputs"
LX_RELAYOUT_DESTINATION_PREFIX = "__spyre_lx_relayout_destination__"


class LXCollectiveKind(StrEnum):
    """Logical communication classes lowered through the physical SHUFFLE op."""

    ALL_TO_ALL = "all_to_all"
    ALL_GATHER = "all_gather"
    BROADCAST = "broadcast"


@dataclasses.dataclass(frozen=True)
class _LXRelayoutClassification:
    collective_kind: LXCollectiveKind
    destination_size_ratio: int
    destination_size_divisor: int = 1


def make_lx_relayout_destination_name(
    source_name: str, consumer_name: str = ""
) -> str:
    suffix = f":{consumer_name}" if consumer_name else ""
    return f"{LX_RELAYOUT_DESTINATION_PREFIX}:{source_name}{suffix}"


def is_lx_relayout_destination(name: str) -> bool:
    return name.startswith(f"{LX_RELAYOUT_DESTINATION_PREFIX}:")


@dataclasses.dataclass(frozen=True)
class LXRelayoutPlan:
    """Names and exact per-core geometry for one LX shuffle."""

    source_name: str
    consumer_name: str
    source_core_id_to_device_slice: dict[str, dict[str, int]]
    destination_core_id_to_device_slice: dict[str, dict[str, int]]
    source_device_dim_splits: dict[str, int]
    destination_device_dim_splits: dict[str, int]
    collective_kind: LXCollectiveKind
    destination_size_ratio: int
    destination_size_divisor: int = 1
    destination_lx_address: int | None = None
    private_destination: bool = False

    @property
    def destination_name(self) -> str:
        return make_lx_relayout_destination_name(
            self.source_name,
            self.consumer_name if self.private_destination else "",
        )


def _intervals_overlap(
    lhs_slot: int, lhs_split: int, rhs_slot: int, rhs_split: int
) -> bool:
    return lhs_slot * rhs_split < (rhs_slot + 1) * lhs_split and (
        rhs_slot * lhs_split < (lhs_slot + 1) * rhs_split
    )


def _classify_lx_relayout(
    producer_map: dict[str, dict[str, int]],
    producer_splits: dict[str, int],
    consumer_map: dict[str, dict[str, int]],
    consumer_splits: dict[str, int],
    *,
    allow_grouped_collectives: bool = False,
) -> _LXRelayoutClassification | None:
    """Classify exact, uniform geometry supported by the LX relayout path."""

    producer_cores = set(producer_map)
    consumer_cores = set(consumer_map)
    if not producer_cores or not consumer_cores:
        return None

    def slice_count(core_map: dict[str, dict[str, int]]) -> int:
        return len(
            {
                tuple(sorted((dim, int(slot)) for dim, slot in per_core.items()))
                for per_core in core_map.values()
            }
        )

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

    fanout_values = set(fanout.values())
    fanin_values = set(fanin.values())
    uniform = len(fanout_values) == 1 and len(fanin_values) == 1
    max_fanout = max(fanout_values, default=0)
    max_fanin = max(fanin_values, default=0)
    covers_all_cores = (
        min(fanout_values, default=0) > 0 and min(fanin_values, default=0) > 0
    )
    producer_is_partitioned = slice_count(producer_map) == len(
        producer_map
    ) and math.prod(producer_splits.values()) == len(producer_map)
    consumer_is_partitioned = slice_count(consumer_map) == len(
        consumer_map
    ) and math.prod(consumer_splits.values()) == len(consumer_map)
    if (
        transfer_count == 0
        or not covers_all_cores
        or not uniform
        or (
            not allow_grouped_collectives
            and not producer_cores.issubset(consumer_cores)
        )
    ):
        return None

    same_participants = producer_cores == consumer_cores
    if same_participants and producer_is_partitioned and consumer_is_partitioned:
        return _LXRelayoutClassification(LXCollectiveKind.ALL_TO_ALL, 1)

    # A broadcast has one physical source owner and replicates that complete
    # source slice to every participating consumer core. Keep this source map
    # sparse: padding it with repeated owners would falsely claim that the
    # value was already resident on every core.
    producer_slice_count = slice_count(producer_map)
    consumer_slice_count = slice_count(consumer_map)
    if (
        len(producer_map) == 1
        and producer_is_partitioned
        and producer_slice_count == 1
        and not consumer_is_partitioned
        and consumer_slice_count == 1
        and max_fanout == len(consumer_map)
        and max_fanin == 1
    ):
        return _LXRelayoutClassification(LXCollectiveKind.BROADCAST, 1)

    # P10/P11-style grouped multicast: every source slice has one owner and
    # every consumer slice is a repeated copy of exactly one source slice.
    # Keep this behind the exact-graph oracle until it has broader coverage.
    if (
        allow_grouped_collectives
        and producer_is_partitioned
        and not consumer_is_partitioned
        and producer_slice_count == consumer_slice_count
        and max_fanout > 1
        and max_fanin == 1
    ):
        return _LXRelayoutClassification(LXCollectiveKind.BROADCAST, 1)

    # P12-style sparse repartition: fewer, larger source pieces are split and
    # reassembled into a denser destination partition.  This is neither a
    # conventional gather nor a same-participant all-to-all, but it uses the
    # same physical SHUFFLE lowering.  Record the exact rational S2:S1 size so
    # the allocator can represent destinations smaller than their sources.
    if (
        allow_grouped_collectives
        and not same_participants
        and producer_is_partitioned
        and consumer_is_partitioned
        and max_fanout > 1
        and max_fanin > 1
    ):
        common = math.gcd(len(producer_map), len(consumer_map))
        return _LXRelayoutClassification(
            LXCollectiveKind.ALL_TO_ALL,
            len(producer_map) // common,
            len(consumer_map) // common,
        )

    # P05-style reducing gather: a 32-core partition is coalesced into a
    # smaller sparse set of complete destination pieces.  The shuffle still
    # executes on the union of source and destination participants.
    if (
        allow_grouped_collectives
        and not same_participants
        and producer_is_partitioned
        and consumer_is_partitioned
        and max_fanout == 1
        and max_fanin > 1
    ):
        return _LXRelayoutClassification(LXCollectiveKind.ALL_GATHER, max_fanin)

    # P13-style subset all-gather: every producer owns one hidden shard while
    # a smaller set of LM-head cores each consumes the complete vector.  The
    # shuffle executes on the union of the 32 source cores and the destination
    # subset, just like the sparse reducing gather above.
    if (
        allow_grouped_collectives
        and not same_participants
        and producer_is_partitioned
        and not consumer_is_partitioned
        and consumer_slice_count == 1
        and max_fanout == len(consumer_map)
        and max_fanin == len(producer_map)
    ):
        return _LXRelayoutClassification(LXCollectiveKind.ALL_GATHER, max_fanin)

    # SenDNN P07/P09 gathers multiple independently owned source fragments into
    # a larger destination slice and replicates that slice over a consumer
    # cohort. The destination has fewer unique slices but more participants.
    if (
        producer_is_partitioned
        and not consumer_is_partitioned
        and len(producer_map) < len(consumer_map)
        and max_fanout > 1
        and max_fanin > 1
    ):
        return _LXRelayoutClassification(LXCollectiveKind.ALL_GATHER, max_fanin)

    if (
        same_participants
        and producer_is_partitioned
        and not consumer_is_partitioned
        and max_fanout > 1
        and max_fanin > 1
    ):
        return _LXRelayoutClassification(LXCollectiveKind.ALL_GATHER, max_fanin)
    return None


def _destination_size_ratio(
    producer_map: dict[str, dict[str, int]],
    producer_splits: dict[str, int],
    consumer_map: dict[str, dict[str, int]],
    consumer_splits: dict[str, int],
) -> int | None:
    """Return S2:S1 per-core size ratio for supported relayout geometry."""

    classification = _classify_lx_relayout(
        producer_map,
        producer_splits,
        consumer_map,
        consumer_splits,
    )
    return None if classification is None else classification.destination_size_ratio


def _single_write_dep(op: ComputedBuffer, buf_name: str) -> MemoryDep | None:
    matches = [
        dep
        for dep in op_read_writes(op).writes
        if isinstance(dep, MemoryDep) and dep.name == buf_name
    ]
    return matches[0] if len(matches) == 1 else None


def _operations_by_name(graph: GraphLowering) -> dict[str, Operation]:
    """Index the current graph, including operations inserted by graph editing."""

    return {op.get_name(): op for op in graph.operations}


def _restickify_reads_computed_input(
    operations: dict[str, Operation], op: Operation
) -> bool:
    if _op_short_name(op) != "restickify":
        return False
    return any(
        isinstance(operations.get(dep.name), ComputedBuffer)
        for dep in op_read_writes(op).reads
        if isinstance(dep, MemoryDep)
    )


def _matmul_operand_source_good_for_lx_relayout(
    operations: dict[str, Operation], op: Operation
) -> bool:
    """Exclude graph-input and weight restickifies from activation relayout."""

    return _op_short_name(op) != "restickify" or _restickify_reads_computed_input(
        operations, op
    )


def _core_id_to_device_slice(
    view: PerCoreView,
    core_count: int,
) -> dict[str, dict[str, int]] | None:
    """Return ownership as ``core -> device-dim -> slice-index``."""

    core_id = sympy.Symbol("core_id")
    expr_by_dim = {int(dim): expr for dim, expr in view.core_to_slot}
    split_dims = {int(dim): int(split) for dim, split in view.work_slice_dims}
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


def _work_slice_dims(view: PerCoreView) -> dict[str, int]:
    return {str(int(dim)): int(split) for dim, split in view.work_slice_dims}


def _buffer_nbytes(graph: GraphLowering, buffer: ComputedBuffer) -> int | None:
    """Return a conservative concrete byte size for relayout cost gating."""

    try:
        numel = graph.sizevars.size_hint(math.prod(buffer.get_size()))
        return int(numel) * int(buffer.get_dtype().itemsize)
    except (AttributeError, TypeError, ValueError):
        # A configured limit fails closed for symbolic or otherwise unknown
        # candidates; the unlimited default never consults this helper.
        return None


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


def clear_lx_relayout_metadata(graph: GraphLowering) -> None:
    for op in graph.operations:
        if hasattr(op, LX_RELAYOUT_ATTR):
            delattr(op, LX_RELAYOUT_ATTR)


def collect_lx_relayout_plans(
    graph: GraphLowering, cache: dict | None = None
) -> list[LXRelayoutPlan]:
    """Plan bounded LX relayouts from producer and consumer coordinates.

    Only records movement for single-writer, single-reader intermediates whose
    producer output is final (not K-split partials) and whose producer and
    consumer PerCoreViews differ.  Same-view edges remain owned by the existing
    LX planner.
    """

    if not config.lx_planner_relayout:
        return []

    operations = _operations_by_name(graph)
    disabled_sources = {
        value.strip()
        for value in config.lx_relayout_disabled_sources.split(",")
        if value.strip()
    }
    disabled_edges = {
        tuple(part.strip() for part in value.split("->", 1))
        for value in config.lx_relayout_disabled_edges.split(",")
        if "->" in value
        and all(part.strip() for part in value.split("->", 1))
    }
    read_counts: dict[str, int] = {}
    for op in graph.operations:
        for dep in op_read_writes(op).reads:
            if isinstance(dep, MemoryDep):
                read_counts[dep.name] = read_counts.get(dep.name, 0) + 1
    plans: list[LXRelayoutPlan] = []

    for consumer in graph.operations:
        if not isinstance(consumer, ComputedBuffer):
            continue
        is_matmul_consumer = _is_matmul_op(consumer)
        oracle_attention_permutation_consumer = (
            config.relayout_oracle_prefill_attention_permutation
            and consumer.get_name() == "buf41"
        )
        oracle_mlp_norm_reduction = (
            config.relayout_oracle_prefill_mlp_normalization
            and consumer.get_name() == "buf48"
            and isinstance(consumer.data, Reduction)
            and [str(size) for size in consumer.get_size()] == ["1", "512", "1"]
        )
        if (
            not is_matmul_consumer
            and not isinstance(consumer.data, Pointwise)
            and not oracle_mlp_norm_reduction
            and not oracle_attention_permutation_consumer
        ):
            continue
        reads = (
            dep for dep in op_read_writes(consumer).reads if isinstance(dep, MemoryDep)
        )
        for read_index, dep in enumerate(reads):
            if dep.name in disabled_sources:
                continue
            if oracle_attention_permutation_consumer and dep.name != "buf40":
                continue
            if (dep.name, consumer.get_name()) in disabled_edges:
                continue
            read_count = read_counts.get(dep.name, 0)
            oracle_shared_mlp_input = (
                config.relayout_oracle_prefill_mlp_inputs
                and dep.name == "buf52"
                and read_count == 2
            )
            oracle_shared_attention_output = (
                oracle_attention_permutation_consumer
                and dep.name == "buf40"
                and read_count > 1
            )
            producer = operations.get(dep.name)
            oracle_shared_rope_input = (
                config.relayout_oracle_prefill_rope_input
                and isinstance(producer, ComputedBuffer)
                and getattr(producer, "_spyre_oracle_p07_source", False)
                and read_count == 2
            )
            oracle_shared_qkv_input = (
                config.relayout_oracle_prefill_qkv_inputs
                and dep.name == "buf10"
                and read_count == 3
            )
            if (
                read_count != 1
                and not oracle_shared_mlp_input
                and not oracle_shared_attention_output
                and not oracle_shared_rope_input
                and not oracle_shared_qkv_input
            ):
                continue
            if not isinstance(producer, ComputedBuffer) or producer is consumer:
                continue

            write_dep = _single_write_dep(producer, dep.name)
            if write_dep is None:
                continue

            producer_view, producer_has_partial, producer_representable = (
                _per_core_view_on_buf(producer, write_dep, dep.name, cache)
            )
            if producer_has_partial or not producer_representable:
                continue

            consumer_view, consumer_has_partial, consumer_representable = (
                _per_core_view_on_buf(consumer, dep, dep.name, cache)
            )
            if consumer_has_partial or not consumer_representable:
                continue

            producer_core_count = _op_num_cores(producer)
            consumer_core_count = _op_num_cores(consumer)
            if (
                producer_view == consumer_view
                and producer_core_count == consumer_core_count
            ):
                continue
            producer_core_slices = _core_id_to_device_slice(
                producer_view, producer_core_count
            )
            if producer_core_slices is None:
                continue

            oracle_residual_add_edge = (
                config.relayout_oracle_prefill_residual_add
                and dep.name
                == config.relayout_oracle_prefill_residual_add_source
                and consumer.get_name()
                == config.relayout_oracle_prefill_residual_add_consumer
                and [str(size) for size in producer.get_size()]
                in (["512", "4096"], ["1", "512", "4096"])
                and [str(size) for size in consumer.get_size()]
                in (["512", "4096"], ["1", "512", "4096"])
            )
            oracle_granite_p14_edge = (
                config.relayout_oracle_granite_p14
                and dep.name == "buf5"
                and consumer.get_name() == "buf6"
                and [str(size) for size in producer.get_size()]
                == ["1", "512", "4096"]
                and [str(size) for size in consumer.get_size()]
                in (["1", "1", "4096"], ["1", "4096"], ["4096"])
            )
            oracle_granite_head_gather_edge = (
                config.work_div_oracle_granite_last_token_head
                and (
                    (
                        dep.name == "buf0"
                        and consumer.get_name() == "buf1"
                    )
                    or (
                        config.relayout_oracle_granite_p14
                        and dep.name == "buf6"
                        and consumer.get_name() == "buf7"
                    )
                )
                and [str(size) for size in producer.get_size()]
                == ["1", "1", "4096"]
                and [str(size) for size in consumer.get_size()]
                in (["1", "1", "49280"], ["1", "1", "50176"])
            )
            if oracle_residual_add_edge:
                # The source executes 16 logical token slices as adjacent
                # physical replica pairs.  SenDNN names the even member of
                # each pair as the authoritative P12 source owner.
                if (
                    producer_core_count != 16
                    or producer_view.work_slice_dims != ((0, 16),)
                    or set(producer_core_slices)
                    != {str(core) for core in range(16)}
                ):
                    continue
                producer_core_slices = {
                    str(2 * int(core)): per_core
                    for core, per_core in producer_core_slices.items()
                }

            oracle_rope_input_edge = (
                oracle_shared_rope_input
                and consumer.get_name() in ("buf12", "buf16")
                and [str(size) for size in producer.get_size()]
                == ["1", "512", "2", "2", "64"]
            )
            if oracle_rope_input_edge:
                # The source clone computes 16 logical token slices as adjacent
                # replica pairs. SenDNN names each pair's even core as the
                # authoritative P07 owner.
                if (
                    producer_core_count != 16
                    or len(producer_view.work_slice_dims) != 1
                    or producer_view.work_slice_dims[0][1] != 16
                    or set(producer_core_slices)
                    != {str(core) for core in range(16)}
                ):
                    continue
                producer_core_slices = {
                    str(2 * int(core)): per_core
                    for core, per_core in producer_core_slices.items()
                }

            oracle_qkv_input_edge = (
                config.relayout_oracle_prefill_qkv_inputs
                and dep.name == "buf10"
                and consumer.get_name() in ("buf11", "buf15", "buf29")
                and [str(size) for size in producer.get_size()]
                == ["1", "512", "4096"]
            )
            if oracle_qkv_input_edge:
                # The normalized activation executes 16 logical token slices
                # as adjacent replica pairs. SenDNN names the even member of
                # every pair as the authoritative P09 source owner.
                if (
                    producer_core_count == 16
                    and producer_view.work_slice_dims == ((0, 16),)
                    and set(producer_core_slices)
                    == {str(core) for core in range(16)}
                ):
                    producer_core_slices = {
                        str(2 * int(core)): per_core
                        for core, per_core in producer_core_slices.items()
                    }
                elif not (
                    producer_core_count == 32
                    and producer_view.work_slice_dims == ((0, 32),)
                    and set(producer_core_slices)
                    == {str(core) for core in range(32)}
                ):
                    continue

            consumer_work_slice_dims = _work_slice_dims(consumer_view)
            consumer_core_slices = _core_id_to_device_slice(
                consumer_view, consumer_core_count
            )
            if consumer_core_slices is None:
                continue

            producer_work_slice_dims = _work_slice_dims(producer_view)
            if oracle_granite_p14_edge:
                expected_source_map = {
                    str(core): {"0": core // 4, "1": core % 4}
                    for core in range(32)
                }
                expected_destination_map_full_rank = {
                    str(core): {"0": 0, "1": core} for core in range(32)
                }
                expected_destination_map_rank1 = {
                    str(core): {"0": core} for core in range(32)
                }
                expected_destination_map_hidden_only = {
                    str(core): {"1": core} for core in range(32)
                }
                destination_is_full_rank = (
                    consumer_work_slice_dims == {"0": 1, "1": 32}
                    and consumer_core_slices == expected_destination_map_full_rank
                )
                destination_is_rank1 = (
                    consumer_work_slice_dims == {"0": 32}
                    and consumer_core_slices == expected_destination_map_rank1
                )
                destination_is_hidden_only = (
                    consumer_work_slice_dims == {"1": 32}
                    and consumer_core_slices
                    == expected_destination_map_hidden_only
                )
                if (
                    producer_core_count != 32
                    or consumer_core_count != 32
                    or producer_work_slice_dims != {"0": 8, "1": 4}
                    or producer_core_slices != expected_source_map
                    or not (
                        destination_is_full_rank
                        or destination_is_rank1
                        or destination_is_hidden_only
                    )
                ):
                    continue
                if destination_is_rank1 or destination_is_hidden_only:
                    # The slice has removed the token axis from buf6's IR
                    # shape. Reintroduce its singleton coordinate so the
                    # source's [token, hidden] ownership and destination's
                    # [hidden] ownership share one relayout coordinate space.
                    consumer_core_slices = expected_destination_map_full_rank
                    consumer_work_slice_dims = {"0": 1, "1": 32}
                # Only the final 64-token cohort owns token 511. Normalize
                # that selected cohort to one transfer-axis slot. Final
                # pointwise emission retains the same token-major core order.
                producer_core_slices = {
                    core: {"0": 0, "1": per_core["1"]}
                    for core, per_core in producer_core_slices.items()
                    if per_core["0"] == 7
                }
                producer_work_slice_dims = {"0": 1, "1": 4}
            if oracle_rope_input_edge:
                # ``PerCoreView`` numbers the clone's non-trivial iteration
                # axes, so token is axis 0 there.  The consumer TensorArg keeps
                # Granite's leading singleton stick quotient and therefore
                # exposes the same token coordinate as physical device axis 1.
                # Bridge that exact view difference before emitting allocation
                # metadata; leaving this as axis 0 incorrectly partitions the
                # 64-wide stick and trips DeepTools fold validation.
                if (
                    producer_work_slice_dims != {"0": 16}
                    or consumer_work_slice_dims != {"0": 8}
                ):
                    continue
                producer_core_slices = {
                    core: {"1": per_core["0"]}
                    for core, per_core in producer_core_slices.items()
                }
                consumer_core_slices = {
                    core: {"1": per_core["0"]}
                    for core, per_core in consumer_core_slices.items()
                }
                producer_work_slice_dims = {"1": 16}
                consumer_work_slice_dims = {"1": 8}

            relayout_dims = set(producer_work_slice_dims) | set(
                consumer_work_slice_dims
            )
            producer_core_slices, producer_work_slice_dims = _dense_view(
                producer_core_slices, producer_work_slice_dims, relayout_dims
            )
            consumer_core_slices, consumer_work_slice_dims = _dense_view(
                consumer_core_slices, consumer_work_slice_dims, relayout_dims
            )
            classification = (
                _LXRelayoutClassification(
                    LXCollectiveKind.ALL_TO_ALL,
                    destination_size_ratio=1,
                    destination_size_divisor=512,
                )
                if oracle_granite_p14_edge
                else _classify_lx_relayout(
                    producer_core_slices,
                    producer_work_slice_dims,
                    consumer_core_slices,
                    consumer_work_slice_dims,
                    allow_grouped_collectives=(
                        (
                            config.relayout_oracle_prefill_mlp_normalization
                            and (
                                (
                                    dep.name == "buf47"
                                    and consumer.get_name() == "buf48"
                                )
                                or (
                                    dep.name == "buf50"
                                    and consumer.get_name() == "buf51"
                                )
                            )
                        )
                        or oracle_granite_head_gather_edge
                        or oracle_residual_add_edge
                        or oracle_rope_input_edge
                        or oracle_qkv_input_edge
                    ),
                )
            )
            if classification is None:
                continue
            enabled_collectives = {
                value.strip()
                for value in config.lx_relayout_collectives.split(",")
                if value.strip()
            }
            if classification.collective_kind.value not in enabled_collectives:
                continue
            if (
                classification.collective_kind is LXCollectiveKind.ALL_TO_ALL
                and config.lx_relayout_all_to_all_max_bytes >= 0
            ):
                producer_nbytes = _buffer_nbytes(graph, producer)
                if (
                    producer_nbytes is None
                    or producer_nbytes > config.lx_relayout_all_to_all_max_bytes
                ):
                    continue

            if is_matmul_consumer and (
                not _matmul_operand_source_good_for_lx_relayout(operations, producer)
                or read_index not in (0, 1)
            ):
                continue

            plans.append(
                LXRelayoutPlan(
                    source_name=dep.name,
                    consumer_name=consumer.get_name(),
                    source_core_id_to_device_slice=producer_core_slices,
                    destination_core_id_to_device_slice=consumer_core_slices,
                    source_device_dim_splits=producer_work_slice_dims,
                    destination_device_dim_splits=consumer_work_slice_dims,
                    collective_kind=classification.collective_kind,
                    destination_size_ratio=classification.destination_size_ratio,
                    destination_size_divisor=classification.destination_size_divisor,
                )
            )

    dump_path = os.environ.get("SPYRE_LX_RELAYOUT_DUMP_PLANS")
    if dump_path:
        records = []
        for plan in plans:
            producer = operations[plan.source_name]
            consumer = operations[plan.consumer_name]
            records.append(
                {
                    **dataclasses.asdict(plan),
                    "source_shape": [str(size) for size in producer.get_size()],
                    "source_dtype": str(producer.get_dtype()),
                    "producer_kind": _op_short_name(producer),
                    "consumer_kind": _op_short_name(consumer),
                    "consumer_reads": [
                        dep.name
                        for dep in op_read_writes(consumer).reads
                        if isinstance(dep, MemoryDep)
                    ],
                }
            )
        with open(dump_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(records, sort_keys=True) + "\n")

    return plans


def record_lx_relayout_plan(graph: GraphLowering, plan: LXRelayoutPlan) -> None:
    """Stamp a relayout plan after the source buffer is placed in LX."""

    consumer = _operations_by_name(graph)[plan.consumer_name]
    plans = getattr(consumer, LX_RELAYOUT_ATTR, {})
    plans[plan.source_name] = plan
    setattr(consumer, LX_RELAYOUT_ATTR, plans)
