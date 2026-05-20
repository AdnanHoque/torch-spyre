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

"""Prototype LX boundary patching for DDL restickify bridges.

The DDL bridge alone is not enough: if the following consumer SDSC still
describes the restickified input as HBM-backed, DXP regenerates an
``hbm -> lx`` load and the bridge result is not consumed.  This module patches
one adjacent producer -> bridge -> consumer triple as a single internal LX edge.
It is deliberately default-off and only operates on the already-generated JSON
payloads immediately before bundle files are written.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec

_INTERSLICE_TRANSPOSE_FP16_OP = "interslicetranspose_fp16"
_COMPACT_ALIAS_ENV = "SPYRE_RESTICKIFY_DDL_BRIDGE_COMPACT_ALIAS"
_PROPAGATE_CORE_MAPPING_ENV = "SPYRE_RESTICKIFY_DDL_BRIDGE_PROPAGATE_CORE_MAPPING"
_PATCH_PRODUCER_LAYOUT_ENV = (
    "SPYRE_RESTICKIFY_DDL_BRIDGE_PATCH_PRODUCER_OUTPUT_LAYOUT"
)
_VALUE_FLOW_ASSERT_ENV = "SPYRE_RESTICKIFY_DDL_BRIDGE_VALUE_FLOW_ASSERT"


def patch_restickify_ddl_bridge_boundaries(
    sdsc_payloads: list[dict[str, Any]],
    specs: list[OpSpec],
) -> list[dict[str, Any]]:
    """Patch eligible adjacent DDL bridge triples in-place.

    Returns one audit-style row per restickify SDSC considered.  The caller is
    expected to gate this behind a prototype config flag.
    """

    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        if spec.op != RESTICKIFY_OP:
            continue
        row = _patch_one_boundary(idx, sdsc_payloads, specs)
        rows.append(row)
    return rows


def _patch_one_boundary(
    idx: int,
    sdsc_payloads: list[dict[str, Any]],
    specs: list[OpSpec],
) -> dict[str, Any]:
    if idx == 0 or idx + 1 >= len(specs):
        return _row(idx, "skipped", "restickify-not-between-adjacent-sdscs")

    bridge_payload = sdsc_payloads[idx]
    if not _is_ddl_bridge_payload(bridge_payload):
        return _row(idx, "skipped", "restickify-was-not-ddl-bridge")

    producer_payload = sdsc_payloads[idx - 1]
    consumer_payload = sdsc_payloads[idx + 1]
    restickify_spec = specs[idx]
    producer_spec = specs[idx - 1]
    consumer_spec = specs[idx + 1]
    if len(restickify_spec.args) != 2:
        return _row(idx, "skipped", "unsupported-restickify-arity")

    producer_lds_idx = _arg_position_for_arg_index(
        producer_spec,
        int(restickify_spec.args[0].arg_index),
        want_input=False,
    )
    consumer_lds_idx = _arg_position_for_arg_index(
        consumer_spec,
        int(restickify_spec.args[-1].arg_index),
        want_input=True,
    )
    if producer_lds_idx is None:
        return _row(idx, "skipped", "producer-output-arg-not-adjacent")
    if consumer_lds_idx is None:
        return _row(idx, "skipped", "consumer-input-arg-not-adjacent")

    producer_root, producer_dsc = _single_payload_dsc(producer_payload)
    bridge_root, bridge_dsc = _single_payload_dsc(bridge_payload)
    consumer_root, consumer_dsc = _single_payload_dsc(consumer_payload)
    bridge_input_idx = _single_input_lds_idx(bridge_dsc)
    bridge_output_idx = _single_output_lds_idx(bridge_dsc)

    producer_start = _allocation_start_payload(producer_dsc, producer_lds_idx)
    bridge_input_start = _allocation_start_payload(bridge_dsc, bridge_input_idx)
    bridge_output_start = _allocation_start_payload(bridge_dsc, bridge_output_idx)
    consumer_start = _allocation_start_payload(consumer_dsc, consumer_lds_idx)
    compact_alias = os.environ.get(_COMPACT_ALIAS_ENV, "0") == "1"
    if compact_alias:
        producer_start = _compact_like(producer_start, base=0)
        bridge_input_start = _compact_like(bridge_input_start, base=0)
        output_base = _lds_lx_size(bridge_dsc, bridge_input_idx)
        bridge_output_start = _compact_like(bridge_output_start, base=output_base)
        consumer_start = _compact_like(consumer_start, base=output_base)
    else:
        bridge_input_start = producer_start
        consumer_start = bridge_output_start

    propagate_core_mapping = os.environ.get(_PROPAGATE_CORE_MAPPING_ENV, "0") == "1"
    if propagate_core_mapping:
        _copy_core_mapping(bridge_root, producer_root)
        _copy_core_mapping(bridge_root, consumer_root)

    producer_name = _lds_name(producer_dsc, producer_lds_idx)
    bridge_input_name = _lds_name(bridge_dsc, bridge_input_idx)
    bridge_output_name = _lds_name(bridge_dsc, bridge_output_idx)
    bridge_input_primary = _primary_for_lds(bridge_dsc, bridge_input_idx)
    bridge_output_primary = _primary_for_lds(bridge_dsc, bridge_output_idx)
    producer_output_ds_type = _ds_type_for_lds(producer_dsc, producer_lds_idx)
    producer_output_primary = _primary_for_lds(producer_dsc, producer_lds_idx)
    consumer_input_ds_type = _ds_type_for_lds(consumer_dsc, consumer_lds_idx)
    consumer_input_primary = _primary_for_lds(consumer_dsc, consumer_lds_idx)
    preserve_consumer_role = (
        consumer_input_primary is not None
        and bridge_output_primary is not None
        and consumer_input_primary == bridge_output_primary
    )
    patch_producer_layout = (
        os.environ.get(_PATCH_PRODUCER_LAYOUT_ENV, "0") == "1"
        and bridge_input_primary is not None
    )
    producer_patch_ds_type = (
        "KERNEL"
        if patch_producer_layout and _ds_type_is_shared(producer_dsc, producer_lds_idx)
        else None
    )

    _patch_payload_lx_only(
        producer_payload,
        lds_idx=producer_lds_idx,
        allocate_name=f"allocate-{producer_name}_lx",
        start_payload=producer_start,
        force_ds_type=producer_patch_ds_type,
        primary_override=bridge_input_primary if patch_producer_layout else None,
        layout_override=(
            list(bridge_input_primary.get("layoutDimOrder_", []))
            if patch_producer_layout
            else None
        ),
    )
    _patch_payload_lx_only(
        bridge_payload,
        lds_idx=bridge_input_idx,
        allocate_name=f"allocate_{bridge_input_name}_lx",
        start_payload=bridge_input_start,
    )
    _patch_payload_lx_only(
        bridge_payload,
        lds_idx=bridge_output_idx,
        allocate_name=f"allocate_{bridge_output_name}_lx",
        start_payload=bridge_output_start,
    )
    _patch_payload_lx_only(
        consumer_payload,
        lds_idx=consumer_lds_idx,
        allocate_name=f"allocate-{bridge_output_name}_lx",
        start_payload=consumer_start,
        force_ds_type=None if preserve_consumer_role else "INPUT",
        primary_override=None if preserve_consumer_role else bridge_output_primary,
    )

    _patch_bridge_transfer_start(
        bridge_dsc,
        transfer_name="transfer_lds0_src:no_component_dst:lx_lx_local",
        offset_side="dst",
        start_payload=bridge_input_start,
    )
    _patch_bridge_transfer_start(
        bridge_dsc,
        transfer_name="transfer_lds1_src:lx_dst:no_component_lx_local",
        offset_side="src",
        start_payload=bridge_output_start,
    )
    value_flow_contract = _check_value_flow_contract(
        producer_root=producer_root,
        producer_dsc=producer_dsc,
        producer_lds_idx=producer_lds_idx,
        bridge_root=bridge_root,
        bridge_dsc=bridge_dsc,
        bridge_input_lds_idx=bridge_input_idx,
        bridge_output_lds_idx=bridge_output_idx,
        consumer_root=consumer_root,
        consumer_dsc=consumer_dsc,
        consumer_lds_idx=consumer_lds_idx,
    )
    if (
        os.environ.get(_VALUE_FLOW_ASSERT_ENV, "0") == "1"
        and not value_flow_contract["ok"]
    ):
        raise ValueError(_format_value_flow_failure(idx, value_flow_contract))

    return {
        "sdsc_index": idx,
        "status": "patched",
        "reason": None,
        "producer_lds_idx": producer_lds_idx,
        "bridge_input_lds_idx": bridge_input_idx,
        "bridge_output_lds_idx": bridge_output_idx,
        "consumer_lds_idx": consumer_lds_idx,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "bridge_output_lx_unique_starts": _unique_start_values(bridge_output_start),
        "consumer_input_original_ds_type": consumer_input_ds_type,
        "consumer_input_role_preserved": preserve_consumer_role,
        "producer_output_original_ds_type": producer_output_ds_type,
        "producer_output_original_primary": producer_output_primary,
        "producer_output_layout_patched": patch_producer_layout,
        "producer_output_patched_ds_type": producer_patch_ds_type,
        "compact_alias": compact_alias,
        "propagate_core_mapping": propagate_core_mapping,
        "value_flow_contract_ok": value_flow_contract["ok"],
        "value_flow_contract": value_flow_contract,
    }


def _row(idx: int, status: str, reason: str) -> dict[str, Any]:
    return {"sdsc_index": idx, "status": status, "reason": reason}


def _arg_position_for_arg_index(
    spec: OpSpec,
    arg_index: int,
    *,
    want_input: bool,
) -> int | None:
    for position, arg in enumerate(spec.args):
        if bool(arg.is_input) == want_input and int(arg.arg_index) == arg_index:
            return position
    return None


def _is_ddl_bridge_payload(payload: dict[str, Any]) -> bool:
    if len(payload) != 1:
        return False
    name = next(iter(payload))
    if not name.endswith("_ddl_bridge"):
        return False
    _, dsc = _single_payload_dsc(payload)
    if not dsc.get("computeOp_"):
        return False
    opfunc = dsc["computeOp_"][0].get("opFuncName", "")
    opfunc = str(opfunc)
    return opfunc.startswith("ReStickifyOp") or opfunc == _INTERSLICE_TRANSPOSE_FP16_OP


def _single_payload_dsc(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC payload")
    root = next(iter(payload.values()))
    dscs = root.get("dscs_", [])
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DSC inside SDSC payload")
    return root, next(iter(dscs[0].values()))


def _single_input_lds_idx(dsc: dict[str, Any]) -> int:
    labels = dsc["computeOp_"][0].get("inputLabeledDs", [])
    if len(labels) != 1:
        raise ValueError("expected one restickify input labeled DS")
    return _parse_lds_idx(labels[0])


def _single_output_lds_idx(dsc: dict[str, Any]) -> int:
    labels = dsc["computeOp_"][0].get("outputLabeledDs", [])
    if len(labels) != 1:
        raise ValueError("expected one restickify output labeled DS")
    return _parse_lds_idx(labels[0])


def _parse_lds_idx(label: str) -> int:
    _, idx = str(label).rsplit("-idx", 1)
    return int(idx)


def _lds_name(dsc: dict[str, Any], lds_idx: int) -> str:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            return str(lds.get("dsName_", f"Tensor{lds_idx}"))
    raise ValueError(f"LDS index {lds_idx} not found")


def _allocation_start_payload(dsc: dict[str, Any], lds_idx: int) -> dict[str, Any]:
    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
        ):
            return copy.deepcopy(node["startAddressCoreCorelet_"])
    raise ValueError(f"allocation for LDS index {lds_idx} not found")


def _patch_payload_lx_only(
    payload: dict[str, Any],
    *,
    lds_idx: int,
    allocate_name: str,
    start_payload: dict[str, Any],
    force_ds_type: str | None = None,
    primary_override: dict[str, Any] | None = None,
    layout_override: list[str] | None = None,
) -> None:
    root, dsc = _single_payload_dsc(payload)
    corelet_factor = _corelet_factor(start_payload)
    root["coreletFoldProp_"] = {"factor_": corelet_factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = corelet_factor
    dsc["numCoreletsUsed_DSC2_"] = corelet_factor

    original_ds_type = None
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) != int(lds_idx):
            continue
        original_ds_type = lds.get("dsType_")
        lx_meta = dict((lds.get("memOrg_", {}) or {}).get("lx", {}))
        lx_meta["isPresent"] = 1
        lx_meta["allocateNode_"] = allocate_name
        lds["memOrg_"] = {"lx": lx_meta}
        lds["hbmStartAddress_"] = -1
        if force_ds_type is not None:
            lds["dsType_"] = force_ds_type

    if primary_override is not None and original_ds_type is not None:
        primary = dsc.setdefault("primaryDsInfo_", {})
        primary[force_ds_type or original_ds_type] = copy.deepcopy(primary_override)

    if force_ds_type is not None:
        primary = dsc.setdefault("primaryDsInfo_", {})
        if primary_override is not None:
            primary[force_ds_type] = copy.deepcopy(primary_override)
        elif force_ds_type not in primary:
            source = primary.get(original_ds_type)
            if source is None and primary:
                source = next(iter(primary.values()))
            if source is not None:
                primary[force_ds_type] = copy.deepcopy(source)

    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
        ):
            node["name_"] = allocate_name
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = copy.deepcopy(start_payload)
            if layout_override is not None:
                node["layoutDimOrder_"] = list(layout_override)
                node["maxDimSizes_"] = [-1 for _ in layout_override]


def _patch_bridge_transfer_start(
    dsc: dict[str, Any],
    *,
    transfer_name: str,
    offset_side: str,
    start_payload: dict[str, Any],
) -> None:
    for node in dsc.get("scheduleTree_", []) or []:
        if node.get("nodeType_") != "transfer" or node.get("name_") != transfer_name:
            continue
        if offset_side == "src":
            offset = node.get("srcLdsAndLoopOffsets_")
            if isinstance(offset, dict):
                offset["startAddr_"] = copy.deepcopy(start_payload)
        elif offset_side == "dst":
            for offset in node.get("dstLdsAndLoopOffsets_", []) or []:
                if isinstance(offset, dict):
                    offset["startAddr_"] = copy.deepcopy(start_payload)
        else:
            raise ValueError(f"unknown offset side {offset_side!r}")


def _corelet_factor(start_payload: dict[str, Any]) -> int:
    attrs = start_payload.get("dim_prop_attr", [])
    if len(attrs) > 1:
        return int(attrs[1].get("factor_", 1) or 1)
    return 1


def _unique_start_values(start_payload: dict[str, Any]) -> list[int]:
    return sorted({int(value) for value in start_payload.get("data_", {}).values()})


def _compact_like(start_payload: dict[str, Any], *, base: int) -> dict[str, Any]:
    out = copy.deepcopy(start_payload)
    out["data_"] = {key: str(base) for key in start_payload.get("data_", {})}
    return out


def _lds_lx_size(dsc: dict[str, Any], lds_idx: int) -> int:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            return int(lds.get("lxSize_", 0) or 0)
    return 0


def _primary_for_lds(dsc: dict[str, Any], lds_idx: int) -> dict[str, Any] | None:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            role = lds.get("dsType_")
            primary = dsc.get("primaryDsInfo_", {}).get(role)
            return copy.deepcopy(primary) if primary is not None else None
    return None


def _ds_type_for_lds(dsc: dict[str, Any], lds_idx: int) -> str | None:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            role = lds.get("dsType_")
            return str(role) if role is not None else None
    return None


def _ds_type_is_shared(dsc: dict[str, Any], lds_idx: int) -> bool:
    role = _ds_type_for_lds(dsc, lds_idx)
    if role is None:
        return False
    return (
        sum(1 for lds in dsc.get("labeledDs_", []) if lds.get("dsType_") == role)
        > 1
    )


def _check_value_flow_contract(
    *,
    producer_root: dict[str, Any],
    producer_dsc: dict[str, Any],
    producer_lds_idx: int,
    bridge_root: dict[str, Any],
    bridge_dsc: dict[str, Any],
    bridge_input_lds_idx: int,
    bridge_output_lds_idx: int,
    consumer_root: dict[str, Any],
    consumer_dsc: dict[str, Any],
    consumer_lds_idx: int,
) -> dict[str, Any]:
    """Check whether the LX bridge describes one coherent internal value.

    Matching LX addresses only proves that payloads alias the same scratchpad
    byte ranges. For a value-correct local restickify, the producer's physical
    output view must also match the bridge source view, and the bridge output
    view must match the consumer input view.
    """

    producer_output = _view_contract(producer_root, producer_dsc, producer_lds_idx)
    bridge_input = _view_contract(bridge_root, bridge_dsc, bridge_input_lds_idx)
    bridge_output = _view_contract(bridge_root, bridge_dsc, bridge_output_lds_idx)
    consumer_input = _view_contract(consumer_root, consumer_dsc, consumer_lds_idx)
    producer_edge = _compare_view_contracts(producer_output, bridge_input)
    consumer_edge = _compare_view_contracts(bridge_output, consumer_input)
    return {
        "ok": producer_edge["ok"] and consumer_edge["ok"],
        "producer_to_bridge_input": producer_edge,
        "bridge_output_to_consumer": consumer_edge,
        "producer_output": producer_output,
        "bridge_input": bridge_input,
        "bridge_output": bridge_output,
        "consumer_input": consumer_input,
    }


def _view_contract(
    root: dict[str, Any],
    dsc: dict[str, Any],
    lds_idx: int,
) -> dict[str, Any]:
    primary = _primary_for_lds(dsc, lds_idx) or {}
    return {
        "num_work_slices_per_dim": _normalize_int_dict(
            root.get("numWkSlicesPerDim_", {}) or {}
        ),
        "core_id_to_work_slice": _normalize_core_mapping(
            root.get("coreIdToWkSlice_", {}) or {}
        ),
        "primary": {
            "layoutDimOrder_": _normalize_str_list(
                primary.get("layoutDimOrder_", []) or []
            ),
            "stickDimOrder_": _normalize_str_list(
                primary.get("stickDimOrder_", []) or []
            ),
            "stickSize_": _normalize_int_list(primary.get("stickSize_", []) or []),
        },
    }


def _compare_view_contracts(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
) -> dict[str, Any]:
    mismatches = [
        field
        for field in ("num_work_slices_per_dim", "core_id_to_work_slice", "primary")
        if lhs.get(field) != rhs.get(field)
    ]
    return {"ok": not mismatches, "mismatches": mismatches}


def _format_value_flow_failure(
    idx: int,
    contract: dict[str, Any],
) -> str:
    producer_mismatches = contract["producer_to_bridge_input"]["mismatches"]
    consumer_mismatches = contract["bridge_output_to_consumer"]["mismatches"]
    return (
        f"restickify DDL bridge {idx} failed LX value-flow contract: "
        f"producer->bridge mismatches={producer_mismatches}, "
        f"bridge->consumer mismatches={consumer_mismatches}"
    )


def _normalize_int_dict(values: dict[Any, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(values.items(), key=str)}


def _normalize_core_mapping(values: dict[Any, Any]) -> dict[str, dict[str, int]]:
    return {
        str(core): _normalize_int_dict(wk_slice or {})
        for core, wk_slice in sorted(values.items(), key=lambda item: int(item[0]))
    }


def _normalize_str_list(values: list[Any]) -> list[str]:
    return [str(value) for value in values]


def _normalize_int_list(values: list[Any]) -> list[int]:
    return [int(value) for value in values]


def _copy_core_mapping(source_root: dict[str, Any], target_root: dict[str, Any]) -> None:
    target_root["numWkSlicesPerDim_"] = copy.deepcopy(
        source_root.get("numWkSlicesPerDim_", {}) or {}
    )
    target_root["coreIdToWkSlice_"] = copy.deepcopy(
        source_root.get("coreIdToWkSlice_", {}) or {}
    )
