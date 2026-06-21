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

"""Coordinate-remap SDSC carrier helpers for experimental on-chip movement."""

from __future__ import annotations

import copy
import json
from typing import Any

from torch_spyre._inductor import config
from torch_spyre._inductor.onchip_move import (
    ONCHIP_MOVE_OP_INFO_KEY,
    _dataop_movement_ranges,
    _expand_dataop_movement_ranges,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg

_LX_SIZE_BYTES = 2 * 1024 * 1024


def patch_onchip_move_mixed_schedules(
    compiled: list[tuple[Any, list[int], list[dict], list[Any]]],
    specs: list[Any],
) -> list[dict[str, Any]]:
    """Attach a data-op carrier to adjacent producer/consumer SDSCs.

    This v1 realization is intentionally narrow: unrolled non-symbolic OpSpec
    lists only.  The planner still records all candidates; unsupported bundle
    shapes fail closed and keep the original SDSCs.
    """

    rows: list[dict[str, Any]] = []
    if not config.onchip_move_realize:
        return rows
    if config.onchip_move_carrier != "coordinate_remap":
        return [{"status": "skipped", "reason": "unsupported-carrier"}]
    if any(isinstance(spec, LoopSpec) for spec in specs):
        return [{"status": "skipped", "reason": "loop-specs-not-supported"}]
    if any(not isinstance(spec, OpSpec) for spec in specs):
        return [{"status": "skipped", "reason": "non-opspec-entry-not-supported"}]
    if len(compiled) != len(specs):
        return [{"status": "skipped", "reason": "compiled-spec-count-mismatch"}]

    rewritten_consumers: set[int] = set()
    reusable_lx_sources: dict[tuple[str, str, str], int] = {}
    for producer_index in range(max(len(specs) - 1, 0)):
        consumer_index = producer_index + 1
        if consumer_index in rewritten_consumers:
            continue
        producer = specs[producer_index]
        consumer = specs[consumer_index]
        if not isinstance(producer, OpSpec) or not isinstance(consumer, OpSpec):
            continue
        match = _adjacent_move_match(producer, consumer)
        if match is None:
            continue
        plan, producer_output_idx, consumer_input_idx = match
        producer_entry = compiled[producer_index]
        consumer_entry = compiled[consumer_index]
        if producer_entry[0] is None or consumer_entry[0] is None:
            rows.append(
                _row(
                    producer_index,
                    "skipped",
                    "producer-or-consumer-already-rewritten",
                    plan,
                )
            )
            continue
        if producer_entry[1] or consumer_entry[1]:
            rows.append(
                _row(
                    producer_index,
                    "skipped",
                    "symbolic-or-local-addresses-not-supported",
                    plan,
                )
            )
            continue

        try:
            patched_producer, mixed_consumer = (
                build_coordinate_remap_onchip_move_sdsc(
                    producer_index,
                    consumer_index,
                    producer_entry[0],
                    consumer_entry[0],
                    producer.args[producer_output_idx],
                    consumer.args[consumer_input_idx],
                    producer_output_idx,
                    consumer_input_idx,
                    plan,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(_row(producer_index, "skipped", type(exc).__name__, plan))
            continue

        compiled[producer_index] = (patched_producer, [], [], [])
        compiled[consumer_index] = (mixed_consumer, [], [], [])
        rewritten_consumers.add(consumer_index)
        reusable_lx_sources[_reuse_key(plan)] = int(
            config.onchip_move_consumer_lx_base
        )
        rows.append(_row(producer_index, "patched", None, plan))

    for consumer_index, consumer in enumerate(specs):
        if consumer_index in rewritten_consumers or not isinstance(consumer, OpSpec):
            continue
        consumer_entry = compiled[consumer_index]
        if consumer_entry[0] is None:
            continue
        if consumer_entry[1]:
            continue
        move_info = (consumer.op_info or {}).get(ONCHIP_MOVE_OP_INFO_KEY)
        if not isinstance(move_info, dict):
            continue
        for source_name, plan in move_info.items():
            if not isinstance(plan, dict):
                continue
            reuse_base = reusable_lx_sources.get(_reuse_key(plan))
            try:
                consumer_input_idx = _consumer_input_arg_idx(consumer, source_name)
                if reuse_base is not None:
                    patched_consumer = copy.deepcopy(consumer_entry[0])
                    consumer_root = next(iter(patched_consumer.values()))
                    _patch_lx_endpoint(
                        consumer_root,
                        dsc_index=0,
                        lds_idx=_matching_input_lds_idx(
                            consumer_root, consumer_input_idx
                        ),
                        base=reuse_base,
                    )
                    compiled[consumer_index] = (patched_consumer, [], [], [])
                    rows.append(_row(consumer_index, "patched-reuse", None, plan))
                    break

                if config.onchip_move_carrier != "coordinate_remap":
                    continue
                producer_match = _producer_match_for_plan(
                    specs,
                    compiled,
                    consumer_index=consumer_index,
                    source_name=source_name,
                    plan=plan,
                )
                if producer_match is None:
                    continue
                producer_index, producer, producer_output_idx = producer_match
                producer_entry = compiled[producer_index]
                if producer_entry[0] is None:
                    continue
                if producer_entry[1]:
                    rows.append(
                        _row(
                            consumer_index,
                            "skipped",
                            "producer-symbolic-or-local-addresses-not-supported",
                            plan,
                        )
                    )
                    continue

                patched_producer, mixed_consumer = (
                    build_coordinate_remap_onchip_move_sdsc(
                        producer_index,
                        consumer_index,
                        producer_entry[0],
                        consumer_entry[0],
                        producer.args[producer_output_idx],
                        consumer.args[consumer_input_idx],
                        producer_output_idx,
                        consumer_input_idx,
                        plan,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(_row(consumer_index, "skipped", type(exc).__name__, plan))
                continue
            compiled[producer_index] = (patched_producer, [], [], [])
            compiled[consumer_index] = (mixed_consumer, [], [], [])
            rewritten_consumers.add(consumer_index)
            reusable_lx_sources[_reuse_key(plan)] = int(
                config.onchip_move_consumer_lx_base
            )
            rows.append(_row(consumer_index, "patched-nonadjacent", None, plan))
            break
    return rows


def build_coordinate_remap_onchip_move_sdsc(
    producer_index: int,
    consumer_index: int,
    producer_payload: dict[str, Any],
    consumer_payload: dict[str, Any],
    producer_output: TensorArg,
    consumer_input: TensorArg,
    producer_output_idx: int,
    consumer_input_idx: int,
    plan: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    producer_name, producer_root_src = next(iter(producer_payload.items()))
    producer_root = copy.deepcopy(producer_root_src)
    consumer_root = copy.deepcopy(next(iter(consumer_payload.values())))
    producer_base = int(config.onchip_move_producer_lx_base)
    consumer_base = int(config.onchip_move_consumer_lx_base)
    producer_region_bytes = _region_bytes(plan, "producer_region_bytes", "source")
    consumer_region_bytes = _region_bytes(plan, "consumer_region_bytes", "dest")
    _validate_lx_regions(
        producer_base=producer_base,
        consumer_base=consumer_base,
        producer_region_bytes=producer_region_bytes,
        consumer_region_bytes=consumer_region_bytes,
    )

    dataop_name = f"{producer_index}_OnChipMoveLXCoordinateRemapOp"
    datadscs, dataop_core_sets = _coordinate_remap_datadsc_chunks(plan, dataop_name)

    _patch_lx_endpoint(
        producer_root,
        dsc_index=0,
        lds_idx=producer_output_idx,
        base=producer_base,
    )
    _patch_lx_endpoint(
        consumer_root,
        dsc_index=0,
        lds_idx=_matching_input_lds_idx(consumer_root, consumer_input_idx),
        base=consumer_base,
    )

    root = copy.deepcopy(consumer_root)
    root["dscs_"] = copy.deepcopy(consumer_root.get("dscs_", []) or [])
    root["datadscs_"] = datadscs
    root["numCoresUsed_"] = max(
        int(consumer_root.get("numCoresUsed_", 1) or 1),
        max((max(core_set) for core_set in dataop_core_sets if core_set), default=0)
        + 1,
    )
    _ensure_root_core_maps(root, root["numCoresUsed_"])
    root["coreIdToDscSchedule"] = _consumer_chunked_mixed_schedule(
        dataop_core_sets=dataop_core_sets,
        consumer_cores=_dsc_core_ids(root, 0),
        num_cores=root["numCoresUsed_"],
    )
    root["opFuncsUsed_"] = sorted(
        _dldsc_op_names(root)
        | {"LXCoordinateRemapOp"}
        | set(root.get("opFuncsUsed_", []) or [])
    )
    root["onchipMove_"] = {
        "source_name": plan.get("source_name"),
        "producer": plan.get("producer"),
        "consumer": plan.get("consumer"),
        "carrier": "coordinate_remap",
        "cell_count": int(plan.get("cell_count", 0) or 0),
        "bytes_moved": int(plan.get("bytes_moved", 0) or 0),
        "producer_lx_base": producer_base,
        "consumer_lx_base": consumer_base,
        "producer_region_bytes": producer_region_bytes,
        "consumer_region_bytes": consumer_region_bytes,
        "dataop_chunks": len(datadscs),
        "fallback": "stock-hbm-path-when-disabled",
    }
    return {producer_name: producer_root}, {
        f"{consumer_index}_OnChipMoveCoordinateRemap": root
    }


def _ensure_root_core_maps(root: dict[str, Any], num_cores: int) -> None:
    core_to_dsc = {
        str(core): int(dsc)
        for core, dsc in (root.get("coreIdToDsc_") or {}).items()
    }
    default_dsc = next(iter(core_to_dsc.values()), 0)
    for core in range(num_cores):
        core_to_dsc.setdefault(str(core), default_dsc)
    root["coreIdToDsc_"] = core_to_dsc

    wk_slices = {
        str(core): dict(value)
        for core, value in (root.get("coreIdToWkSlice_") or {}).items()
    }
    default_slice = next(iter(wk_slices.values()), None)
    if default_slice is None:
        default_slice = {
            dim: 0 for dim in (root.get("numWkSlicesPerDim_") or {}).keys()
        }
    for core in range(num_cores):
        wk_slices.setdefault(str(core), dict(default_slice))
    root["coreIdToWkSlice_"] = wk_slices


def _adjacent_move_match(
    producer: OpSpec,
    consumer: OpSpec,
) -> tuple[dict[str, Any], int, int] | None:
    move_info = (consumer.op_info or {}).get(ONCHIP_MOVE_OP_INFO_KEY)
    if not isinstance(move_info, dict):
        return None
    producer_outputs = [
        (idx, arg) for idx, arg in enumerate(producer.args) if not arg.is_input
    ]
    consumer_inputs = [
        (idx, arg) for idx, arg in enumerate(consumer.args) if arg.is_input
    ]
    for source_name, plan in move_info.items():
        if not isinstance(plan, dict):
            continue
        for producer_idx, producer_arg in producer_outputs:
            if producer_arg.name != source_name:
                continue
            for consumer_idx, consumer_arg in consumer_inputs:
                if consumer_arg.name == source_name:
                    return plan, producer_idx, consumer_idx
    return None


def _consumer_input_arg_idx(consumer: OpSpec, source_name: str) -> int:
    for idx, arg in enumerate(consumer.args):
        if arg.is_input and arg.name == source_name:
            return idx
    raise ValueError(f"consumer-input-not-found:{source_name}")


def _producer_match_for_plan(
    specs: list[Any],
    compiled: list[tuple[Any, list[int], list[dict], list[Any]]],
    *,
    consumer_index: int,
    source_name: str,
    plan: dict[str, Any],
) -> tuple[int, OpSpec, int] | None:
    producer_name = str(plan.get("producer", ""))
    for producer_index in range(consumer_index - 1, -1, -1):
        producer = specs[producer_index]
        if not isinstance(producer, OpSpec):
            continue
        if compiled[producer_index][0] is None:
            continue
        if producer_name and _op_spec_output_name(producer) != producer_name:
            continue
        for arg_index, arg in enumerate(producer.args):
            if not arg.is_input and arg.name == source_name:
                return producer_index, producer, arg_index
    return None


def _op_spec_output_name(spec: OpSpec) -> str:
    for arg in spec.args:
        if not arg.is_input and arg.name:
            return str(arg.name)
    return ""


def _reuse_key(plan: dict[str, Any]) -> tuple[str, str, str]:
    dataop = (
        ((plan.get("coordinate_remap") or {}).get("deeptools_dataop") or {})
        if isinstance(plan.get("coordinate_remap"), dict)
        else {}
    )
    return (
        str(plan.get("source_name")),
        str(plan.get("producer")),
        json.dumps(
            {
                "movement_subview": plan.get("movement_subview"),
                "movement_ranges": dataop.get("movementRanges"),
                "bytes_moved": plan.get("bytes_moved"),
                "consumer_region_bytes": plan.get("consumer_region_bytes"),
            },
            sort_keys=True,
        ),
    )


def _coordinate_remap_dataop_movements(dataop: dict[str, Any]) -> list[dict[str, Any]]:
    movements = list(dataop.get("movements", []) or [])
    if movements:
        return movements
    return _expand_dataop_movement_ranges(list(dataop.get("movementRanges", []) or []))


def _coordinate_remap_dataop(plan: dict[str, Any]) -> dict[str, Any]:
    metadata = plan.get("coordinate_remap")
    if not isinstance(metadata, dict):
        raise ValueError("coordinate-remap-metadata-missing")
    dataop = metadata.get("deeptools_dataop")
    if not isinstance(dataop, dict):
        raise ValueError("coordinate-remap-dataop-missing")
    if (dataop.get("op") or {}).get("name") != "LXCoordinateRemapOp":
        raise ValueError("coordinate-remap-dataop-op-mismatch")
    movements = _coordinate_remap_dataop_movements(dataop)
    if not movements:
        raise ValueError("coordinate-remap-dataop-has-no-movements")
    core_ids = sorted(
        {int(movement["source"]["core"]) for movement in movements}
        | {int(movement["destination"]["core"]) for movement in movements}
    )
    result = copy.deepcopy(dataop)
    result["movements"] = movements
    result["coreIdsUsed_"] = core_ids
    return result


def _coordinate_remap_datadsc_chunks(
    plan: dict[str, Any],
    dataop_name: str,
) -> tuple[list[dict[str, Any]], list[set[int]]]:
    dataop = _coordinate_remap_dataop(plan)
    movements = list(dataop.get("movements", []) or [])
    chunk_size = max(1, int(config.onchip_move_coordinate_remap_chunk_cells))

    datadscs: list[dict[str, Any]] = []
    core_sets: list[set[int]] = []
    static_fields = {
        key: copy.deepcopy(value)
        for key, value in dataop.items()
        if key not in {"movements", "movementRanges", "coreIdsUsed_"}
    }

    def append_chunk(chunk_movements_src: list[dict[str, Any]]) -> None:
        if not chunk_movements_src:
            return
        chunk_movements = copy.deepcopy(chunk_movements_src)
        chunk_core_ids = sorted(
            {int(movement["source"]["core"]) for movement in chunk_movements}
            | {int(movement["destination"]["core"]) for movement in chunk_movements}
        )
        chunk = copy.deepcopy(static_fields)
        if config.onchip_move_range_encoding:
            movement_ranges = _dataop_movement_ranges(chunk_movements)
            chunk["movementRanges"] = movement_ranges
            lowering = chunk.setdefault("lowering", {})
            lowering["rangeEncoded"] = True
            lowering["movementRanges"] = len(movement_ranges)
            lowering["expandedMovements"] = len(chunk_movements)
        else:
            chunk["movements"] = chunk_movements
            lowering = chunk.setdefault("lowering", {})
            lowering["rangeEncoded"] = False
        chunk["coreIdsUsed_"] = chunk_core_ids
        datadscs.append({f"{dataop_name}_{len(datadscs)}": chunk})
        core_sets.append(set(chunk_core_ids))

    cross_movements = [
        movement
        for movement in movements
        if int(movement["source"]["core"]) != int(movement["destination"]["core"])
    ]
    local_movements = [
        movement
        for movement in movements
        if int(movement["source"]["core"]) == int(movement["destination"]["core"])
    ]

    if not local_movements and len(cross_movements) <= chunk_size:
        compact_dataop = copy.deepcopy(dataop)
        if config.onchip_move_range_encoding:
            movement_ranges = _dataop_movement_ranges(cross_movements)
            compact_dataop.pop("movements", None)
            compact_dataop["movementRanges"] = movement_ranges
            lowering = compact_dataop.setdefault("lowering", {})
            lowering["rangeEncoded"] = True
            lowering["movementRanges"] = len(movement_ranges)
            lowering["expandedMovements"] = len(cross_movements)
        return [{dataop_name: compact_dataop}], [set(dataop["coreIdsUsed_"])]

    for start in range(0, len(cross_movements), chunk_size):
        append_chunk(cross_movements[start : start + chunk_size])

    for first_legs, second_legs in _relay_local_coordinate_remap_chunks(
        local_movements,
        producer_base=int(config.onchip_move_producer_lx_base),
        producer_region_bytes=_region_bytes(plan, "producer_region_bytes", "source"),
        consumer_base=int(config.onchip_move_consumer_lx_base),
        consumer_region_bytes=_region_bytes(plan, "consumer_region_bytes", "dest"),
        chunk_size=chunk_size,
    ):
        append_chunk(first_legs)
        append_chunk(second_legs)
    return datadscs, core_sets


def _relay_local_coordinate_remap_chunks(
    movements: list[dict[str, Any]],
    *,
    producer_base: int,
    producer_region_bytes: int,
    consumer_base: int,
    consumer_region_bytes: int,
    chunk_size: int,
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    if not movements:
        return []

    relay_base, relay_capacity = _local_relay_scratch_window(
        producer_base=producer_base,
        producer_region_bytes=producer_region_bytes,
        consumer_base=consumer_base,
        consumer_region_bytes=consumer_region_bytes,
    )
    if relay_capacity <= 0:
        raise ValueError("coordinate-remap-local-relay-lx-capacity")

    next_move_index = (
        max((int(movement["moveIndex"]) for movement in movements), default=-1) + 1
    )
    batches: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for chunk in _local_relay_batches(
        sorted(
            movements,
            key=lambda movement: (
                int(movement["source"]["core"]),
                int(movement["source"]["lxAddress"]),
                int(movement["destination"]["lxAddress"]),
                int(movement["moveIndex"]),
            ),
        ),
        chunk_size=chunk_size,
        relay_capacity=relay_capacity,
    ):
        first_legs: list[dict[str, Any]] = []
        second_legs: list[dict[str, Any]] = []
        relay_offset = 0
        for movement in chunk:
            source_core = int(movement["source"]["core"])
            sencores = int(config.sencores)
            if sencores <= 1:
                raise ValueError(
                    "local relay coordinate remap requires at least two SEN cores"
                )
            if source_core < 0 or source_core >= sencores:
                raise ValueError(
                    f"local relay source core {source_core} is outside SENCORES={sencores}"
                )
            relay_core = (source_core + 1) % sencores
            byte_count = int(movement["bytes"])
            relay_lx_address = relay_base + relay_offset
            relay_offset += byte_count

            first = copy.deepcopy(movement)
            first["moveIndex"] = next_move_index
            next_move_index += 1
            first["destination"]["core"] = relay_core
            first["destination"]["lxAddress"] = relay_lx_address
            first["destination"]["localByteRange"] = {
                "start": relay_lx_address - relay_base,
                "end": relay_lx_address - relay_base + byte_count,
            }
            first["destination"]["lxByteRange"] = {
                "start": relay_lx_address,
                "end": relay_lx_address + byte_count,
            }
            first["relay"] = {
                "kind": "local_first_leg",
                "originalMoveIndex": int(movement["moveIndex"]),
                "relayCore": relay_core,
            }
            first_legs.append(first)

            second = copy.deepcopy(movement)
            second["moveIndex"] = next_move_index
            next_move_index += 1
            second["source"]["core"] = relay_core
            second["source"]["lxAddress"] = relay_lx_address
            second["source"]["localByteRange"] = first["destination"][
                "localByteRange"
            ]
            second["source"]["lxByteRange"] = first["destination"]["lxByteRange"]
            second["relay"] = {
                "kind": "local_second_leg",
                "originalMoveIndex": int(movement["moveIndex"]),
                "relayCore": relay_core,
            }
            second_legs.append(second)

        batches.append((first_legs, second_legs))
    return batches


def _local_relay_batches(
    movements: list[dict[str, Any]],
    *,
    chunk_size: int,
    relay_capacity: int,
) -> list[list[dict[str, Any]]]:
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0
    for movement in movements:
        byte_count = int(movement["bytes"])
        if byte_count > relay_capacity:
            raise ValueError("coordinate-remap-local-relay-lx-capacity")
        if current and (
            len(current) >= chunk_size or current_bytes + byte_count > relay_capacity
        ):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(movement)
        current_bytes += byte_count
    if current:
        batches.append(current)
    return batches


def _local_relay_scratch_window(
    *,
    producer_base: int,
    producer_region_bytes: int,
    consumer_base: int,
    consumer_region_bytes: int,
) -> tuple[int, int]:
    protected: list[tuple[int, int]] = []
    if producer_region_bytes > 0:
        protected.append((producer_base, producer_base + producer_region_bytes))
    if consumer_region_bytes > 0:
        protected.append((consumer_base, consumer_base + consumer_region_bytes))
    protected = [
        (max(0, start), min(_LX_SIZE_BYTES, end))
        for start, end in protected
        if start < _LX_SIZE_BYTES and end > 0 and start < end
    ]
    protected.sort()

    best_start = 0
    best_end = 0
    cursor = 0
    for start, end in protected:
        aligned_cursor = ((cursor + 127) // 128) * 128
        aligned_start = (start // 128) * 128
        if aligned_start - aligned_cursor > best_end - best_start:
            best_start, best_end = aligned_cursor, aligned_start
        cursor = max(cursor, end)
    aligned_cursor = ((cursor + 127) // 128) * 128
    if _LX_SIZE_BYTES - aligned_cursor > best_end - best_start:
        best_start, best_end = aligned_cursor, _LX_SIZE_BYTES
    return best_start, max(0, best_end - best_start)


def _first_compute_dsc(root: dict[str, Any]) -> dict[str, Any] | None:
    for dsc_group in root.get("dscs_", []) or []:
        if isinstance(dsc_group, dict) and dsc_group:
            dsc = next(iter(dsc_group.values()))
            if isinstance(dsc, dict):
                return dsc
    return None


def _keep_missing_size_one_dims(
    output_layout: dict[str, Any] | None,
    input_layout: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if output_layout is None or input_layout is None:
        return output_layout

    output_dim_order = [str(dim) for dim in output_layout["layout_dim_order"]]
    output_dim_set = set(output_dim_order)
    missing_size_one_dims = [
        str(dim)
        for dim in input_layout["layout_dim_order"]
        if str(dim) not in output_dim_set
    ]
    if not missing_size_one_dims:
        return output_layout

    result = dict(output_layout)
    result["layout_dim_order"] = [*output_dim_order, *missing_size_one_dims]
    result["layout_sizes"] = dict(output_layout["layout_sizes"])
    for dim in missing_size_one_dims:
        if int(input_layout["layout_sizes"][dim]) != 1:
            raise ValueError("output-layout-drops-non-size-one-dim")
        result["layout_sizes"][dim] = 1
    result["stick_dim_order"] = list(output_layout["stick_dim_order"])
    result["stick_sizes"] = dict(output_layout["stick_sizes"])
    return result


def _region_bytes(plan: dict[str, Any], explicit_key: str, offset_side: str) -> int:
    explicit = int(plan.get(explicit_key, 0) or 0)
    if explicit > 0:
        return explicit
    offset_key = f"{offset_side}_offset_bytes"
    cells = list(plan.get("cells", []) or [])
    if not cells:
        return 0
    return max(
        int(cell.get(offset_key, 0) or 0) + int(cell.get("bytes", 0) or 0)
        for cell in cells
    )


def _validate_lx_regions(
    *,
    producer_base: int,
    consumer_base: int,
    producer_region_bytes: int,
    consumer_region_bytes: int,
) -> None:
    if producer_base < 0 or consumer_base < 0:
        raise ValueError("lx-base-negative")
    producer_end = producer_base + producer_region_bytes
    consumer_end = consumer_base + consumer_region_bytes
    if producer_end > _LX_SIZE_BYTES:
        raise ValueError("producer-lx-region-exceeds-capacity")
    if consumer_end > _LX_SIZE_BYTES:
        raise ValueError("consumer-lx-region-exceeds-capacity")
    if producer_region_bytes and consumer_region_bytes:
        if max(producer_base, consumer_base) < min(producer_end, consumer_end):
            raise ValueError("producer-consumer-lx-regions-overlap")


def _matching_input_lds_idx(root: dict[str, Any], consumer_input_idx: int) -> int:
    dsc = next(iter((root.get("dscs_", []) or [])[0].values()))
    input_labels = dsc.get("computeOp_", [{}])[0].get("inputLabeledDs", [])
    input_indices = {_lds_idx_from_label(label) for label in input_labels}
    if consumer_input_idx in input_indices:
        return consumer_input_idx
    raise ValueError(f"consumer-input-not-found:{consumer_input_idx}")


def _lds_idx_from_label(label: str) -> int:
    if "-idx" not in label:
        return -1
    return int(label.rsplit("-idx", 1)[1])


def _patch_lx_endpoint(
    root: dict[str, Any],
    *,
    dsc_index: int,
    lds_idx: int,
    base: int,
) -> None:
    dsc = next(iter(root["dscs_"][dsc_index].values()))
    num_cores = int(dsc.get("numCoresUsed_", root.get("numCoresUsed_", 1)) or 1)
    core_ids = [int(core) for core in dsc.get("coreIdsUsed_", range(num_cores))]
    core_factor = int(root.get("coreFoldProp_", {}).get("factor_", 32) or 32)
    for node in dsc.get("scheduleTree_", []):
        if int(node.get("ldsIdx_", -1)) != int(lds_idx):
            continue
        node["component_"] = "lx"
        node["name_"] = str(node.get("name_", "")).replace("_hbm", "_lx")
        node["startAddressCoreCorelet_"] = _folded_start_address(
            core_ids, int(base), core_factor=core_factor
        )
        for dim_gap in (node.get("backGapCore_") or {}).values():
            if "-1" not in dim_gap:
                continue
            uniform_gap = dim_gap["-1"]
            for core in core_ids:
                dim_gap[str(core)] = uniform_gap
    for lds in dsc.get("labeledDs_", []):
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            lds["memOrg_"] = {"lx": {"isPresent": 1}}


def _folded_start_address(
    core_ids: list[int],
    base: int,
    *,
    core_factor: int = 32,
) -> dict[str, Any]:
    return {
        "dim_prop_func": [
            {"Map": {}},
            {"Const": {}},
            {"Const": {}},
        ],
        "dim_prop_attr": [
            {"factor_": int(core_factor), "label_": "core"},
            {"factor_": 1, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": {f"[{int(core)}, 0, 0]": int(base) for core in core_ids},
    }


def _dsc_core_ids(root: dict[str, Any], dsc_index: int) -> set[int]:
    dsc = next(iter(root["dscs_"][dsc_index].values()))
    return {int(core) for core in dsc.get("coreIdsUsed_", [])}


def _consumer_chunked_mixed_schedule(
    *,
    dataop_core_sets: list[set[int]],
    consumer_cores: set[int],
    num_cores: int,
) -> dict[str, list[list[int]]]:
    schedule: dict[str, list[list[int]]] = {}
    for core in range(num_cores):
        steps: list[list[int]] = []
        for dataop_index, dataop_cores in enumerate(dataop_core_sets):
            if core in dataop_cores:
                steps.append([dataop_index, -1, 0, 0])
        if core in consumer_cores:
            steps.append([-1, 0, 0, 0])
        schedule[str(core)] = _with_dependencies(steps)
    return schedule


def _with_dependencies(steps: list[list[int]]) -> list[list[int]]:
    return [
        [step[0], step[1], 1 if idx > 0 else 0, 1 if idx < len(steps) - 1 else 0]
        for idx, step in enumerate(steps)
    ]


def _dldsc_op_names(root: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for dsc in root.get("dscs_", []) or []:
        names.update(str(name) for name in dsc.keys())
    return names


def _row(
    index: int,
    status: str,
    reason: str | None,
    plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "index": index,
        "status": status,
        "reason": reason,
        "source_name": plan.get("source_name"),
        "producer": plan.get("producer"),
        "consumer": plan.get("consumer"),
        "cell_count": plan.get("cell_count"),
        "carrier": config.onchip_move_carrier,
    }


def _collect_opspecs(specs: list[Any], result: list[OpSpec]) -> None:
    for spec in specs:
        if isinstance(spec, LoopSpec):
            _collect_opspecs(spec.body, result)
        elif isinstance(spec, OpSpec):
            result.append(spec)
