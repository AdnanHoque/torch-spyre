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
from pathlib import Path
from typing import Any

from torch_spyre._inductor import config
from torch_spyre._inductor.constants import BATCH_MATMUL_FP8_OP, BATCH_MATMUL_OP
from torch_spyre._inductor.onchip_move import ONCHIP_MOVE_OP_INFO_KEY
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg

_LX_SIZE_BYTES = 2 * 1024 * 1024


def patch_onchip_move_mixed_schedules(
    compiled: list[tuple[Any, list[int], list[dict], list[Any]]],
    specs: list[Any],
) -> list[dict[str, Any]]:
    """Attach an STCDP data-op carrier to adjacent producer/consumer SDSCs.

    This v1 realization is intentionally narrow: unrolled non-symbolic OpSpec
    lists only.  The planner still records all candidates; unsupported bundle
    shapes fail closed and keep the original SDSCs.
    """

    rows: list[dict[str, Any]] = []
    if not config.onchip_move_realize:
        return rows
    if config.onchip_move_carrier != "mixed":
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
            patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
                producer_index,
                consumer_index,
                producer_entry[0],
                consumer_entry[0],
                producer.args[producer_output_idx],
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
                _patch_lx_endpoint(
                    patched_consumer,
                    dsc_index=0,
                    lds_idx=_matching_input_lds_idx(
                        patched_consumer, consumer_input_idx
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
    _patch_lx_endpoint(
        consumer_root,
        dsc_index=0,
        lds_idx=_matching_input_lds_idx(consumer_root, consumer_input_idx),
        base=consumer_base,
    )
    datadsc = build_stcdp_datadsc(
        dataop_name,
        plan,
        data_format=producer_output.device_dtype.name,
        word_length=_word_length(producer_output),
        producer_base=producer_base,
        consumer_base=consumer_base,
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


def build_stcdp_datadsc(
    name: str,
    plan: dict[str, Any],
    *,
    data_format: str,
    word_length: int,
    producer_base: int,
    consumer_base: int,
) -> dict[str, Any]:
    device_sizes = [int(size) for size in plan.get("device_sizes", [])]
    if not device_sizes:
        raise ValueError("plan-has-no-device-sizes")
    dim_names = [f"d{idx}_" for idx in range(len(device_sizes))]
    cells = list(plan.get("cells", []) or [])
    if not cells:
        raise ValueError("plan-has-no-cells")
    input_pieces = [
        _piece(
            cell,
            source=True,
            base=producer_base,
        )
        for cell in cells
    ]
    output_pieces = [
        _piece(
            cell,
            source=False,
            base=consumer_base,
        )
        for cell in cells
    ]
    core_ids = sorted(
        {
            int(cell["source_core"])
            for cell in cells
        }
        | {int(cell["dest_core"]) for cell in cells}
    )
    stick_dim = dim_names[-1]
    layout_sizes = dict(zip(dim_names, device_sizes))
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
                dim_names,
                stick_dim,
                layout_sizes,
                input_pieces,
            ),
            _labeled_ds(
                "dataOUT_L0",
                "dataOUT",
                data_format,
                word_length,
                dim_names,
                stick_dim,
                layout_sizes,
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
        "validGap_": _valid_gap(cell["dim_sizes"]),
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
    stick_dim: str,
    layout_sizes: dict[str, int],
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
        "stickDimOrder_": [stick_dim],
        "dimToLayoutSize_": layout_sizes,
        "dimToStickSize_": {stick_dim: int(layout_sizes[stick_dim])},
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
    for node in dsc.get("scheduleTree_", []):
        if int(node.get("ldsIdx_", -1)) != int(lds_idx):
            continue
        node["component_"] = "lx"
        node["name_"] = str(node.get("name_", "")).replace("_hbm", "_lx")
        node.setdefault("startAddressCoreCorelet_", {})["data_"] = {
            f"[{core}, 0, 0]": str(int(base)) for core in range(num_cores)
        }
    for lds in dsc.get("labeledDs_", []):
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            lds["memOrg_"] = {"lx": {"isPresent": 1}}


def _dsc_core_ids(root: dict[str, Any], dsc_index: int) -> set[int]:
    dsc = next(iter(root["dscs_"][dsc_index].values()))
    return {int(core) for core in dsc.get("coreIdsUsed_", [])}


def _consumer_mixed_schedule(
    *,
    dataop_cores: set[int],
    consumer_cores: set[int],
    num_cores: int,
) -> dict[str, list[list[int]]]:
    schedule: dict[str, list[list[int]]] = {}
    for core in range(num_cores):
        steps: list[list[int]] = []
        if core in dataop_cores:
            steps.append([0, -1, 0, 0])
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
        "carrier": "mixed",
    }


def _collect_opspecs(specs: list[Any], result: list[OpSpec]) -> None:
    for spec in specs:
        if isinstance(spec, LoopSpec):
            _collect_opspecs(spec.body, result)
        elif isinstance(spec, OpSpec):
            result.append(spec)
