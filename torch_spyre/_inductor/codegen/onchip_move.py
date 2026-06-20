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

"""Mixed-SDSC carrier helpers for experimental on-chip movement."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from torch_spyre._inductor import config
from torch_spyre._inductor.constants import BATCH_MATMUL_FP8_OP, BATCH_MATMUL_OP
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
    if config.onchip_move_carrier not in {"mixed", "coordinate_remap"}:
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
        if plan.get("status") != "planned":
            continue
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
            if config.onchip_move_carrier == "coordinate_remap":
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
            else:
                patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
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
            if not isinstance(plan, dict) or plan.get("status") != "planned":
                continue
            reuse_base = reusable_lx_sources.get(_reuse_key(plan))
            if reuse_base is None:
                continue
            try:
                consumer_input_idx = _consumer_input_arg_idx(consumer, source_name)
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
            except Exception as exc:  # noqa: BLE001
                rows.append(_row(consumer_index, "skipped", type(exc).__name__, plan))
                continue
            compiled[consumer_index] = (patched_consumer, [], [], [])
            rows.append(_row(consumer_index, "patched-reuse", None, plan))
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


def _reuse_key(plan: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(plan.get("source_name")),
        str(plan.get("producer")),
        json.dumps(plan.get("consumer_view", {}), sort_keys=True),
    )


def build_mixed_onchip_move_sdsc(
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
    dataop_name = f"{producer_index}_OnChipMoveSTCDPOpLx"

    _patch_lx_endpoint(
        producer_root,
        dsc_index=0,
        lds_idx=producer_output_idx,
        base=producer_base,
    )
    consumer_input_lds_idx = _matching_input_lds_idx(consumer_root, consumer_input_idx)
    _patch_lx_endpoint(
        consumer_root,
        dsc_index=0,
        lds_idx=consumer_input_lds_idx,
        base=consumer_base,
    )
    producer_layout = _logical_dataop_layout(
        producer_root, producer_output, plan, producer_output_idx
    )
    consumer_layout = _logical_dataop_layout(
        consumer_root, consumer_input, plan, consumer_input_lds_idx
    )
    if consumer_layout is None:
        consumer_layout = _dsc_logical_layout(consumer_root, consumer_input_lds_idx)
    datadsc = build_stcdp_datadsc(
        dataop_name,
        plan,
        data_format=producer_output.device_dtype.name,
        word_length=_word_length(producer_output),
        producer_base=producer_base,
        consumer_base=consumer_base,
        logical_layout=producer_layout,
        output_logical_layout=consumer_layout,
    )

    root = copy.deepcopy(consumer_root)
    root["dscs_"] = copy.deepcopy(consumer_root.get("dscs_", []) or [])
    root["datadscs_"] = [{dataop_name: datadsc}]
    root["numCoresUsed_"] = max(
        int(consumer_root.get("numCoresUsed_", 1) or 1),
        max(datadsc.get("coreIdsUsed_", [0])) + 1,
    )
    root["coreIdToDscSchedule"] = _consumer_mixed_schedule(
        dataop_cores=set(datadsc["coreIdsUsed_"]),
        consumer_cores=_dsc_core_ids(root, 0),
        num_cores=root["numCoresUsed_"],
    )
    root["opFuncsUsed_"] = sorted(
        _dldsc_op_names(root) | {"STCDPOpLx"} | set(root.get("opFuncsUsed_", []) or [])
    )
    root["onchipMove_"] = {
        "source_name": plan.get("source_name"),
        "producer": plan.get("producer"),
        "consumer": plan.get("consumer"),
        "carrier": "mixed",
        "cell_count": int(plan.get("cell_count", 0) or 0),
        "bytes_moved": int(plan.get("bytes_moved", 0) or 0),
        "producer_lx_base": producer_base,
        "consumer_lx_base": consumer_base,
        "producer_region_bytes": producer_region_bytes,
        "consumer_region_bytes": consumer_region_bytes,
        "fallback": "stock-hbm-path-when-disabled",
    }
    return {producer_name: producer_root}, {f"{consumer_index}_OnChipMoveMixedSTCDP": root}


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

    movements, local_relay_first, local_relay_second = (
        _expand_local_coordinate_remap_movements_via_relay(movements)
    )
    cross_movements = [
        movement
        for movement in movements
        if int(movement["source"]["core"]) != int(movement["destination"]["core"])
    ]
    local_by_core: dict[int, list[dict[str, Any]]] = {}
    for movement in movements:
        source_core = int(movement["source"]["core"])
        if source_core == int(movement["destination"]["core"]):
            local_by_core.setdefault(source_core, []).append(movement)

    if (
        not local_by_core
        and not local_relay_first
        and not local_relay_second
        and len(cross_movements) <= chunk_size
    ):
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

    for start in range(0, len(local_relay_first), chunk_size):
        append_chunk(local_relay_first[start : start + chunk_size])
    for start in range(0, len(local_relay_second), chunk_size):
        append_chunk(local_relay_second[start : start + chunk_size])

    for core_movements in local_by_core.values():
        core_movements.sort(
            key=lambda movement: (
                int(movement["source"]["lxAddress"]),
                int(movement["destination"]["lxAddress"]),
                int(movement["moveIndex"]),
            )
        )
    for _core, rows in sorted(local_by_core.items()):
        for start in range(0, len(rows), chunk_size):
            append_chunk(rows[start : start + chunk_size])
    return datadscs, core_sets


def _expand_local_coordinate_remap_movements_via_relay(
    movements: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cross_movements: list[dict[str, Any]] = []
    local_movements: list[dict[str, Any]] = []
    for movement in movements:
        if int(movement["source"]["core"]) == int(movement["destination"]["core"]):
            local_movements.append(movement)
        else:
            cross_movements.append(movement)
    if not local_movements:
        return movements, [], []

    used_end = 0
    for movement in movements:
        used_end = max(
            used_end,
            int(movement["source"]["lxByteRange"]["end"]),
            int(movement["destination"]["lxByteRange"]["end"]),
        )
    relay_base = ((used_end + 127) // 128) * 128
    relay_bytes = sum(int(movement["bytes"]) for movement in local_movements)
    if relay_base + relay_bytes > _LX_SIZE_BYTES:
        raise ValueError("coordinate-remap-local-relay-lx-capacity")

    first_legs: list[dict[str, Any]] = []
    second_legs: list[dict[str, Any]] = []
    next_move_index = (
        max((int(movement["moveIndex"]) for movement in movements), default=-1) + 1
    )
    relay_offset = 0
    for movement in local_movements:
        source_core = int(movement["source"]["core"])
        relay_core = (source_core + 1) % 32
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

    return cross_movements, first_legs, second_legs


def _first_compute_dsc(root: dict[str, Any]) -> dict[str, Any] | None:
    for dsc_group in root.get("dscs_", []) or []:
        if isinstance(dsc_group, dict) and dsc_group:
            dsc = next(iter(dsc_group.values()))
            if isinstance(dsc, dict):
                return dsc
    return None


def _logical_dataop_layout(
    producer_root: dict[str, Any],
    producer_output: TensorArg,
    plan: dict[str, Any],
    lds_idx: int | None = None,
) -> dict[str, Any] | None:
    layout = _dsc_logical_layout(producer_root, lds_idx)
    if layout is None:
        return None
    layout_dim_order = list(layout["layout_dim_order"])
    stick_dim_order = list(layout["stick_dim_order"])
    layout_sizes = dict(layout["layout_sizes"])

    device_sizes = [int(size) for size in plan.get("device_sizes", []) or []]
    stride_map = [
        int(stride)
        for stride in (
            plan.get("device_stride_map", []) or producer_output.stride_map or []
        )
    ]
    if len(device_sizes) < 2 or len(stride_map) != len(device_sizes):
        return None

    mapping = _device_to_logical_mapping(
        device_sizes=device_sizes,
        stride_map=stride_map,
        layout_dim_order=layout_dim_order,
        layout_sizes=layout_sizes,
        stick_dim=stick_dim_order[0],
    )
    if mapping is None:
        return None
    result = dict(layout)
    result["device_to_logical"] = mapping
    result["stick_elems"] = int(device_sizes[-1])
    return result


def _dsc_logical_layout(
    root: dict[str, Any],
    lds_idx: int | None = None,
) -> dict[str, Any] | None:
    dsc = _first_compute_dsc(root)
    if dsc is None:
        return None
    label: str | None = None
    if lds_idx is not None:
        for lds in dsc.get("labeledDs_", []) or []:
            if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
                label = str(lds.get("dsType_"))
                break
    if not label:
        label = "OUTPUT"
    layout_info = (dsc.get("primaryDsInfo_", {}) or {}).get(label)
    if not isinstance(layout_info, dict):
        layout_info = (dsc.get("primaryDsInfo_", {}) or {}).get("OUTPUT")
    if not isinstance(layout_info, dict):
        return None
    layout_dim_order = [str(dim) for dim in layout_info.get("layoutDimOrder_", [])]
    stick_dim_order = [str(dim) for dim in layout_info.get("stickDimOrder_", [])]
    if not layout_dim_order or len(stick_dim_order) != 1:
        return None
    raw_stick_sizes = list(layout_info.get("stickSize_", []) or [])
    if len(raw_stick_sizes) != len(stick_dim_order):
        return None
    n_info = dsc.get("N_", {}) or {}
    layout_sizes: dict[str, int] = {}
    for dim in layout_dim_order:
        value = n_info.get(f"{dim}_", n_info.get(dim))
        if value is None:
            return None
        layout_sizes[dim] = int(value)
    return {
        "layout_dim_order": layout_dim_order,
        "stick_dim_order": stick_dim_order,
        "layout_sizes": layout_sizes,
        "stick_sizes": {
            dim: int(size) for dim, size in zip(stick_dim_order, raw_stick_sizes)
        },
    }


def _logical_host_strides(
    layout_dim_order: list[str],
    layout_sizes: dict[str, int],
) -> dict[str, int]:
    stride = 1
    strides: dict[str, int] = {}
    for dim in reversed(layout_dim_order):
        strides[dim] = stride
        stride *= int(layout_sizes[dim])
    return strides


def _device_to_logical_mapping(
    *,
    device_sizes: list[int],
    stride_map: list[int],
    layout_dim_order: list[str],
    layout_sizes: dict[str, int],
    stick_dim: str,
) -> dict[int, dict[str, Any]] | None:
    host_strides = _logical_host_strides(layout_dim_order, layout_sizes)
    if stick_dim not in host_strides:
        return None
    stick_elems = int(device_sizes[-1])
    stick_host_stride = int(host_strides[stick_dim])
    mapping: dict[int, dict[str, Any]] = {}
    outer_stick_dim: int | None = None
    inner_stick_dim: int | None = None
    used_direct_dims: set[str] = set()

    for device_dim, stride in enumerate(stride_map):
        stride = int(stride)
        if device_dim == len(device_sizes) - 1:
            if stride != stick_host_stride:
                return None
            inner_stick_dim = device_dim
            mapping[device_dim] = {"dim": stick_dim, "kind": "inner"}
            continue

        if stride == stick_host_stride * stick_elems:
            if outer_stick_dim is not None:
                return None
            outer_stick_dim = device_dim
            mapping[device_dim] = {"dim": stick_dim, "kind": "outer"}
            continue

        if stride > 0:
            matches = [
                dim
                for dim in layout_dim_order
                if dim != stick_dim
                and dim not in used_direct_dims
                and int(host_strides[dim]) == stride
            ]
        else:
            matches = [
                dim
                for dim in layout_dim_order
                if dim != stick_dim
                and dim not in used_direct_dims
                and int(layout_sizes[dim]) == 1
                and int(device_sizes[device_dim]) == 1
            ]
        if len(matches) != 1:
            return None
        used_direct_dims.add(matches[0])
        mapping[device_dim] = {"dim": matches[0], "kind": "direct"}

    direct_dims = {entry["dim"] for entry in mapping.values() if entry["kind"] == "direct"}
    if outer_stick_dim is None or inner_stick_dim is None:
        return None
    if any(dim != stick_dim and dim not in direct_dims for dim in layout_dim_order):
        return None
    return mapping


def _logical_cells(
    plan: dict[str, Any],
    logical_layout: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if logical_layout is None:
        return list(plan.get("cells", []) or [])
    mapping: dict[int, dict[str, Any]] = logical_layout["device_to_logical"]
    layout_dim_order: list[str] = logical_layout["layout_dim_order"]
    stick_dim = logical_layout["stick_dim_order"][0]
    stick_elems = int(logical_layout["stick_elems"])
    element_bytes = int(plan.get("element_bytes", 0) or 0)

    direct_device_dim_by_logical = {
        str(entry["dim"]): device_dim
        for device_dim, entry in mapping.items()
        if entry["kind"] == "direct"
    }
    outer_stick_dim = next(
        device_dim
        for device_dim, entry in mapping.items()
        if entry["dim"] == stick_dim and entry["kind"] == "outer"
    )
    inner_stick_dim = next(
        device_dim
        for device_dim, entry in mapping.items()
        if entry["dim"] == stick_dim and entry["kind"] == "inner"
    )

    logical_cells: list[dict[str, Any]] = []
    for cell in plan.get("cells", []) or []:
        dim_starts: dict[str, int] = {}
        dim_sizes: dict[str, int] = {}
        for dim in layout_dim_order:
            if dim == stick_dim:
                outer_start = int(cell["dim_starts"][f"d{outer_stick_dim}_"])
                outer_size = int(cell["dim_sizes"][f"d{outer_stick_dim}_"])
                inner_start = int(cell["dim_starts"][f"d{inner_stick_dim}_"])
                inner_size = int(cell["dim_sizes"][f"d{inner_stick_dim}_"])
                dim_starts[dim] = outer_start * stick_elems + inner_start
                dim_sizes[dim] = (outer_size - 1) * stick_elems + inner_size
            else:
                device_dim = direct_device_dim_by_logical[dim]
                dim_starts[dim] = int(cell["dim_starts"][f"d{device_dim}_"])
                dim_sizes[dim] = int(cell["dim_sizes"][f"d{device_dim}_"])

        if element_bytes:
            logical_bytes = math.prod(dim_sizes.values()) * element_bytes
            if logical_bytes != int(cell.get("bytes", 0) or 0):
                raise ValueError("logical-cell-byte-size-mismatch")
        logical_cell = dict(cell)
        logical_cell["dim_starts"] = dim_starts
        logical_cell["dim_sizes"] = dim_sizes
        logical_cells.append(logical_cell)
    return logical_cells


def _project_logical_cells(
    cells: list[dict[str, Any]],
    output_layout: dict[str, Any],
) -> list[dict[str, Any]]:
    dim_order = [str(dim) for dim in output_layout["layout_dim_order"]]
    projected: list[dict[str, Any]] = []
    for cell in cells:
        starts: dict[str, int] = {}
        sizes: dict[str, int] = {}
        for dim in dim_order:
            if dim in cell["dim_starts"]:
                starts[dim] = int(cell["dim_starts"][dim])
                sizes[dim] = int(cell["dim_sizes"][dim])
            else:
                if int(output_layout["layout_sizes"][dim]) != 1:
                    raise ValueError("output-layout-dim-not-in-cell")
                starts[dim] = 0
                sizes[dim] = 1
        projected_cell = dict(cell)
        projected_cell["dim_starts"] = starts
        projected_cell["dim_sizes"] = sizes
        projected.append(projected_cell)
    return projected


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


def _coalesce_piece_cells(
    cells: list[dict[str, Any]],
    *,
    core_key: str,
    offset_key: str,
    element_bytes: int,
) -> list[dict[str, Any]]:
    by_core: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        by_core.setdefault(int(cell[core_key]), []).append(cell)
    if not by_core:
        return cells

    coalesced: list[dict[str, Any]] = []
    for new_index, (core, group) in enumerate(sorted(by_core.items())):
        dim_names = list(group[0]["dim_starts"].keys())
        if any(list(cell["dim_starts"].keys()) != dim_names for cell in group):
            return cells
        starts = {
            dim: min(int(cell["dim_starts"][dim]) for cell in group)
            for dim in dim_names
        }
        ends = {
            dim: max(
                int(cell["dim_starts"][dim]) + int(cell["dim_sizes"][dim])
                for cell in group
            )
            for dim in dim_names
        }
        sizes = {dim: ends[dim] - starts[dim] for dim in dim_names}
        bytes_moved = sum(int(cell.get("bytes", 0) or 0) for cell in group)
        if element_bytes:
            rect_bytes = math.prod(sizes.values()) * int(element_bytes)
            if rect_bytes != bytes_moved:
                return cells
            bytes_moved = rect_bytes

        first = group[0]
        merged = dict(first)
        merged["cell_index"] = new_index
        merged["source_core"] = int(first["source_core"])
        merged["dest_core"] = int(first["dest_core"])
        merged[core_key] = core
        merged[offset_key] = min(int(cell.get(offset_key, 0) or 0) for cell in group)
        merged["bytes"] = bytes_moved
        merged["dim_starts"] = starts
        merged["dim_sizes"] = sizes
        coalesced.append(merged)
    return coalesced


def _span_partial_stick_dim_for_output(
    cells: list[dict[str, Any]],
    output_layout: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if config.onchip_move_output_piece_mode == "dense_actual":
        return cells
    if output_layout is None:
        return cells
    stick_dim_order = list(output_layout.get("stick_dim_order", []))
    if len(stick_dim_order) != 1:
        return cells
    stick_dim = str(stick_dim_order[0])
    full_size = int(output_layout["layout_sizes"][stick_dim])
    spanned: list[dict[str, Any]] = []
    for cell in cells:
        starts = dict(cell["dim_starts"])
        sizes = dict(cell["dim_sizes"])
        if stick_dim not in starts or stick_dim not in sizes:
            spanned.append(cell)
            continue
        start = int(starts[stick_dim])
        size = int(sizes[stick_dim])
        if size <= 0 or start < 0 or start + size > full_size:
            raise ValueError("output-stick-piece-out-of-bounds")
        if size == full_size:
            spanned.append(cell)
            continue

        starts[stick_dim] = 0
        sizes[stick_dim] = full_size
        valid_gap = _valid_gap(sizes)
        if start == 0:
            valid_gap[stick_dim] = [[size, full_size - size]]
        else:
            valid_gap[stick_dim] = [[0, start], [size, full_size - start - size]]
        spanned_cell = dict(cell)
        spanned_cell["dim_starts"] = starts
        spanned_cell["dim_sizes"] = sizes
        spanned_cell["_valid_gap"] = valid_gap
        spanned.append(spanned_cell)
    return spanned


def build_stcdp_datadsc(
    name: str,
    plan: dict[str, Any],
    *,
    data_format: str,
    word_length: int,
    producer_base: int,
    consumer_base: int,
    logical_layout: dict[str, Any] | None = None,
    output_logical_layout: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device_sizes = [int(size) for size in plan.get("device_sizes", [])]
    if not device_sizes:
        raise ValueError("plan-has-no-device-sizes")
    input_cells = _logical_cells(plan, logical_layout) if logical_layout else list(
        plan.get("cells", []) or []
    )
    if not input_cells:
        raise ValueError("plan-has-no-cells")
    output_logical_layout = _keep_missing_size_one_dims(
        output_logical_layout,
        logical_layout,
    )
    output_cells = (
        _project_logical_cells(input_cells, output_logical_layout)
        if output_logical_layout
        else input_cells
    )
    output_cells = _span_partial_stick_dim_for_output(
        output_cells,
        output_logical_layout or logical_layout,
    )

    def layout_fields(layout: dict[str, Any] | None) -> tuple[
        list[str], list[str], dict[str, int], dict[str, int]
    ]:
        if layout:
            dim_names = list(layout["layout_dim_order"])
            stick_dim_order = list(layout.get("stick_dim_order", []))
            layout_sizes = {
                str(dim): int(size) for dim, size in layout["layout_sizes"].items()
            }
            stick_sizes = {
                str(dim): int(size)
                for dim, size in layout["stick_sizes"].items()
            }
            return dim_names, stick_dim_order, layout_sizes, stick_sizes
        dim_names = [f"d{idx}_" for idx in range(len(device_sizes))]
        stick_dim_order = [dim_names[-1]]
        layout_sizes = dict(zip(dim_names, device_sizes))
        stick_sizes = {stick_dim_order[0]: int(layout_sizes[stick_dim_order[0]])}
        return dim_names, stick_dim_order, layout_sizes, stick_sizes

    input_dim_names, input_stick_dim_order, input_layout_sizes, input_stick_sizes = (
        layout_fields(logical_layout)
    )
    output_dim_names, output_stick_dim_order, output_layout_sizes, output_stick_sizes = (
        layout_fields(output_logical_layout or logical_layout)
    )
    dim_names = list(dict.fromkeys(input_dim_names + output_dim_names))
    input_pieces = [
        _piece(
            cell,
            source=True,
            base=producer_base,
        )
        for cell in input_cells
    ]
    output_pieces = [
        _piece(
            cell,
            source=False,
            base=consumer_base,
        )
        for cell in output_cells
    ]
    core_ids = sorted(
        {
            int(cell["source_core"])
            for cell in input_cells
        }
        | {int(cell["dest_core"]) for cell in output_cells}
    )
    return {
        "coreIdsUsed_": core_ids,
        "dimPool_": dim_names,
        "outDimTodimRelation_": [],
        "primaryDs_": [
            {"name_": "dataIN", "dimNames": dim_names},
            {"name_": "dataOUT", "dimNames": dim_names},
        ],
        "labeledDs_": [
            _labeled_ds(
                "dataIN_L0",
                "dataIN",
                data_format,
                word_length,
                input_dim_names,
                input_stick_dim_order,
                input_layout_sizes,
                input_stick_sizes,
                input_pieces,
            ),
            _labeled_ds(
                "dataOUT_L0",
                "dataOUT",
                data_format,
                word_length,
                output_dim_names,
                output_stick_dim_order,
                output_layout_sizes,
                output_stick_sizes,
                output_pieces,
            ),
        ],
        "op": {"name": "STCDPOpLx"},
    }


def emit_swiglu_warpspec_audit(
    kernel_name: str,
    output_dir: str,
    specs: list[Any],
) -> None:
    if not config.swiglu_warpspec_audit:
        return
    flat: list[OpSpec] = []
    _collect_opspecs(specs, flat)
    ops = [spec.op for spec in flat]
    matmul_indices = [
        idx for idx, op in enumerate(ops) if op in {BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP}
    ]
    has_silu_op = any("silu" in op.lower() for op in ops)
    has_decomposed_silu = all(op in ops for op in ("neg", "exp", "add", "realdiv"))
    pointwise_ops = [
        op
        for op in ops
        if op not in {BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP, "ReStickifyOpHBM"}
    ]
    payload = {
        "kernel_name": kernel_name,
        "output_dir": output_dir,
        "ops": ops,
        "matmul_indices": matmul_indices,
        "has_standalone_silu_op": has_silu_op,
        "has_decomposed_silu_chain": has_decomposed_silu,
        "pointwise_ops": pointwise_ops,
        "warp_specialization_candidate": bool(
            matmul_indices and (has_silu_op or has_decomposed_silu) and pointwise_ops
        ),
        "status": "audit-only",
    }
    if config.swiglu_warpspec_audit_jsonl:
        path = Path(config.swiglu_warpspec_audit_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    if config.onchip_move_debug_dir:
        path = Path(config.onchip_move_debug_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "swiglu_warpspec_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _piece(cell: dict[str, Any], *, source: bool, base: int) -> dict[str, Any]:
    core = int(cell["source_core"] if source else cell["dest_core"])
    offset_key = "source_offset_bytes" if source else "dest_offset_bytes"
    start_addr = int(base) + int(cell.get(offset_key, 0) or 0)
    return {
        "key_": f"p{int(cell['cell_index']) + 1}",
        "dimToStartCordinate": dict(cell["dim_starts"]),
        "dimToSize_": dict(cell["dim_sizes"]),
        "validGap_": dict(cell.get("_valid_gap") or _valid_gap(cell["dim_sizes"])),
        "PlacementInfo": [
            {"type": "lx", "memId": [core], "startAddr": [start_addr]}
        ],
    }


def _labeled_ds(
    lds_name: str,
    pds_name: str,
    data_format: str,
    word_length: int,
    dim_names: list[str],
    stick_dim_order: list[str],
    layout_sizes: dict[str, int],
    stick_sizes: dict[str, int],
    pieces: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "ldsName_": lds_name,
        "pdsName_": pds_name,
        "wordLength": int(word_length),
        "dataformat": data_format,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": dim_names,
        "stickDimOrder_": stick_dim_order,
        "dimToLayoutSize_": layout_sizes,
        "dimToStickSize_": {
            stick_dim: int(stick_sizes[stick_dim]) for stick_dim in stick_dim_order
        },
        "validGap_": _valid_gap(layout_sizes),
        "totElements": -1,
        "PieceInfo": pieces,
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": _LX_SIZE_BYTES,
        "lxStartAddress_": {},
    }


def _valid_gap(sizes: dict[str, Any]) -> dict[str, list[list[int]]]:
    return {str(dim): [[int(size), 0]] for dim, size in sizes.items()}


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


def _word_length(arg: TensorArg) -> int:
    return int(128 // arg.device_dtype.elems_per_stick())


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


def _consumer_mixed_schedule(
    *,
    dataop_cores: set[int],
    consumer_cores: set[int],
    num_cores: int,
) -> dict[str, list[list[int]]]:
    return _consumer_chunked_mixed_schedule(
        dataop_core_sets=[dataop_cores],
        consumer_cores=consumer_cores,
        num_cores=num_cores,
    )


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
        "carrier": plan.get("carrier") or config.onchip_move_carrier,
    }


def _collect_opspecs(specs: list[Any], result: list[OpSpec]) -> None:
    for spec in specs:
        if isinstance(spec, LoopSpec):
            _collect_opspecs(spec.body, result)
        elif isinstance(spec, OpSpec):
            result.append(spec)
