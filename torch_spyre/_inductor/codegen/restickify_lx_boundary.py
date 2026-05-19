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
from typing import Any

from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec


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

    _, producer_dsc = _single_payload_dsc(producer_payload)
    _, bridge_dsc = _single_payload_dsc(bridge_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)
    bridge_input_idx = _single_input_lds_idx(bridge_dsc)
    bridge_output_idx = _single_output_lds_idx(bridge_dsc)

    producer_start = _allocation_start_payload(producer_dsc, producer_lds_idx)
    bridge_output_start = _allocation_start_payload(bridge_dsc, bridge_output_idx)

    producer_name = _lds_name(producer_dsc, producer_lds_idx)
    bridge_input_name = _lds_name(bridge_dsc, bridge_input_idx)
    bridge_output_name = _lds_name(bridge_dsc, bridge_output_idx)

    _patch_payload_lx_only(
        producer_payload,
        lds_idx=producer_lds_idx,
        allocate_name=f"allocate-{producer_name}_lx",
        start_payload=producer_start,
    )
    _patch_payload_lx_only(
        bridge_payload,
        lds_idx=bridge_input_idx,
        allocate_name=f"allocate_{bridge_input_name}_lx",
        start_payload=producer_start,
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
        start_payload=bridge_output_start,
        force_ds_type="INPUT",
    )

    _patch_bridge_transfer_start(
        bridge_dsc,
        transfer_name="transfer_lds0_src:no_component_dst:lx_lx_local",
        offset_side="dst",
        start_payload=producer_start,
    )
    _patch_bridge_transfer_start(
        bridge_dsc,
        transfer_name="transfer_lds1_src:lx_dst:no_component_lx_local",
        offset_side="src",
        start_payload=bridge_output_start,
    )

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
    return str(opfunc).startswith("ReStickifyOp")


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
) -> None:
    root, dsc = _single_payload_dsc(payload)
    corelet_factor = _corelet_factor(start_payload)
    root["coreletFoldProp_"] = {"factor_": corelet_factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = corelet_factor
    dsc["numCoreletsUsed_DSC2_"] = corelet_factor

    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) != int(lds_idx):
            continue
        lx_meta = dict((lds.get("memOrg_", {}) or {}).get("lx", {}))
        lx_meta["isPresent"] = 1
        lx_meta["allocateNode_"] = allocate_name
        lds["memOrg_"] = {"lx": lx_meta}
        lds["hbmStartAddress_"] = -1
        if force_ds_type is not None:
            lds["dsType_"] = force_ds_type

    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
        ):
            node["name_"] = allocate_name
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = copy.deepcopy(start_payload)


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
