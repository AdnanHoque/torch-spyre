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

"""PT-aware LX restickify boundary lowering prototype.

This module replaces a proven adjacent producer -> ReStickifyOpHBM -> consumer
triple before final bundle files are written. It is the normal-lowering version
of the same-artifact splice prototype: the producer output is made LX-resident,
the restickify SDSC is replaced by a two-step PT-aware LX data-op bridge, and
the consumer input is made LX-resident at the bridge output address.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from torch_spyre._inductor import config as _spyre_config
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec
from torch_spyre._inductor.restickify_ring import (
    CORE_MAPPING_OVERRIDE_OP_INFO_KEY,
    LOCALITY_CERTIFICATE_OP_INFO_KEY,
)

from .restickify_lx_dataop import generate_ptlx_restickify_bridge_sdsc

_PRODUCER_BASE_ENV = "SPYRE_RESTICKIFY_PTLX_BRIDGE_PRODUCER_BASE"
_CONSUMER_BASE_ENV = "SPYRE_RESTICKIFY_PTLX_BRIDGE_CONSUMER_BASE"
_DEFAULT_PRODUCER_BASE = 16 * 1024
_DEFAULT_CONSUMER_BASE = 8 * 1024


def patch_restickify_ptlx_bridge_boundaries(
    sdsc_payloads: list[dict[str, Any]],
    specs: list[OpSpec],
) -> list[dict[str, Any]]:
    """Patch eligible adjacent restickify triples in-place."""

    rows = []
    for idx, spec in enumerate(specs):
        if spec.op != RESTICKIFY_OP:
            continue
        row = _patch_one_boundary(idx, sdsc_payloads, specs)
        rows.append(row)
        _append_audit(row)
    return rows


def _patch_one_boundary(
    idx: int,
    sdsc_payloads: list[dict[str, Any]],
    specs: list[OpSpec],
) -> dict[str, Any]:
    if idx == 0 or idx + 1 >= len(specs):
        return _row(idx, "skipped", "restickify-not-between-adjacent-sdscs")

    restickify_spec = specs[idx]
    reason = _eligibility_skip_reason(restickify_spec)
    if reason is not None:
        return _row(idx, "skipped", reason)

    restickify_payload = sdsc_payloads[idx]
    if not _is_restickify_hbm_payload(restickify_payload):
        return _row(idx, "skipped", "restickify-payload-not-hbm-compute")

    producer_payload = sdsc_payloads[idx - 1]
    consumer_payload = sdsc_payloads[idx + 1]
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
    restickify_root, restickify_dsc = _single_payload_dsc(restickify_payload)
    _, consumer_dsc = _single_payload_dsc(consumer_payload)
    restickify_input_idx = _first_compute_input_index(restickify_dsc)
    restickify_output_idx = _first_compute_output_index(restickify_dsc)
    restickify_logical_direction = _infer_restickify_direction(
        restickify_dsc,
        input_lds_idx=restickify_input_idx,
        output_lds_idx=restickify_output_idx,
    )
    direction = _infer_endpoint_direction(
        producer_dsc,
        producer_lds_idx=producer_lds_idx,
        consumer_dsc=consumer_dsc,
        consumer_lds_idx=consumer_lds_idx,
        restickify_logical_direction=restickify_logical_direction,
    )
    if direction != "kernel-to-output":
        return _row(idx, "skipped", f"unsupported-direction:{direction}")

    size, num_cores = _infer_size_and_cores(restickify_root, restickify_dsc)
    producer_base = int(os.environ.get(_PRODUCER_BASE_ENV, _DEFAULT_PRODUCER_BASE))
    consumer_base = int(os.environ.get(_CONSUMER_BASE_ENV, _DEFAULT_CONSUMER_BASE))
    producer_start = _constant_lx_start_payload(
        num_cores=num_cores,
        base=producer_base,
    )
    consumer_start = _constant_lx_start_payload(
        num_cores=num_cores,
        base=consumer_base,
    )

    producer_patches = _patch_lx_allocation_by_index(
        producer_payload,
        lds_idx=producer_lds_idx,
        start_payload=producer_start,
    )
    consumer_name = _lds_name(consumer_dsc, consumer_lds_idx)
    _patch_consumer_input_lx_map(
        consumer_payload,
        input_name=consumer_name,
        lds_idx=consumer_lds_idx,
        start_payload=consumer_start,
    )
    _force_consumer_corelets(
        consumer_payload,
        factor=_corelet_factor(consumer_start),
    )

    bridge_payload = generate_ptlx_restickify_bridge_sdsc(
        f"{idx}_TwoStepReStickifyOpWithPTLxStcdp",
        size=size,
        num_cores=num_cores,
        mode="stage3b",
        direction=direction,
        input_start_address=producer_base,
        output_start_address=consumer_base,
        restickify_op_name="ReStickifyOpWithPTLx",
    )
    endpoint_patch = _patch_bridge_endpoint_pieces(
        bridge_payload,
        producer_starts={core: producer_base for core in range(num_cores)},
        consumer_starts={core: consumer_base for core in range(num_cores)},
    )
    sdsc_payloads[idx] = bridge_payload

    return {
        **_row(idx, "patched", None),
        "direction": direction,
        "restickify_logical_direction": restickify_logical_direction,
        "size": size,
        "num_cores": num_cores,
        "producer_lds_idx": producer_lds_idx,
        "consumer_lds_idx": consumer_lds_idx,
        "restickify_input_lds_idx": restickify_input_idx,
        "restickify_output_lds_idx": restickify_output_idx,
        "producer_lx_unique_starts": _unique_start_values(producer_start),
        "consumer_lx_unique_starts": _unique_start_values(consumer_start),
        "producer_allocation_patches": producer_patches,
        "bridge_endpoint_patch": endpoint_patch,
        "replacement_sdsc": next(iter(bridge_payload)),
    }


def _eligibility_skip_reason(spec: OpSpec) -> str | None:
    op_info = spec.op_info or {}
    if op_info.get("restickify_source_kind") != "in_graph_computed":
        return "source-not-in-graph-computed"
    if CORE_MAPPING_OVERRIDE_OP_INFO_KEY not in op_info:
        return "missing-core-mapping-override"
    certificate = op_info.get(LOCALITY_CERTIFICATE_OP_INFO_KEY)
    if not isinstance(certificate, dict):
        return "missing-locality-certificate"
    if not certificate.get("locality_certified"):
        return "locality-not-certified"
    if int(certificate.get("certified_byte_hops", -1)) != 0:
        return "nonzero-certified-byte-hops"
    return None


def _row(idx: int, status: str, reason: str | None) -> dict[str, Any]:
    return {"sdsc_index": idx, "status": status, "reason": reason}


def _append_audit(row: dict[str, Any]) -> None:
    path = _spyre_config.restickify_ptlx_bridge_audit_jsonl
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


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


def _is_restickify_hbm_payload(payload: dict[str, Any]) -> bool:
    try:
        _, dsc = _single_payload_dsc(payload)
    except (KeyError, ValueError, StopIteration, TypeError):
        return False
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return False
    return str(ops[0].get("opFuncName", "")) == "ReStickifyOpHBM"


def _single_payload_dsc(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC payload")
    root = next(iter(payload.values()))
    dscs = root.get("dscs_", []) or []
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DSC inside SDSC payload")
    return root, next(iter(dscs[0].values()))


def _first_compute_input_index(dsc: dict[str, Any]) -> int:
    indices = _compute_input_indices(dsc)
    if not indices:
        raise ValueError("DSC has no compute input LDS")
    return indices[0]


def _first_compute_output_index(dsc: dict[str, Any]) -> int:
    indices = _compute_output_indices(dsc)
    if not indices:
        raise ValueError("DSC has no compute output LDS")
    return indices[0]


def _compute_input_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_label_index(token) for token in ops[0].get("inputLabeledDs", []) or []]


def _compute_output_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [_lds_label_index(token) for token in ops[0].get("outputLabeledDs", []) or []]


def _lds_label_index(token: str) -> int:
    match = re.search(r"-idx(\d+)$", str(token))
    if not match:
        raise ValueError(f"could not parse LDS index from {token!r}")
    return int(match.group(1))


def _lds_name(dsc: dict[str, Any], lds_idx: int) -> str:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            return str(lds.get("dsName_", f"lds{lds_idx}"))
    raise ValueError(f"LDS index {lds_idx} not found")


def _primary_for_lds(dsc: dict[str, Any], lds_idx: int) -> dict[str, Any]:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            key = str(lds.get("dsType_"))
            return dsc.get("primaryDsInfo_", {}).get(key, {})
    raise ValueError(f"LDS index {lds_idx} not found")


def _normalize_dim(dim: Any) -> str:
    return str(dim).removesuffix("_")


def _primary_layout(primary: dict[str, Any]) -> list[str]:
    return [_normalize_dim(dim) for dim in primary.get("layoutDimOrder_", [])]


def _primary_stick(primary: dict[str, Any]) -> list[str]:
    return [_normalize_dim(dim) for dim in primary.get("stickDimOrder_", [])]


def _infer_endpoint_direction(
    producer_dsc: dict[str, Any],
    *,
    producer_lds_idx: int,
    consumer_dsc: dict[str, Any],
    consumer_lds_idx: int,
    restickify_logical_direction: str,
) -> str:
    source_primary = _primary_for_lds(producer_dsc, producer_lds_idx)
    destination_primary = _primary_for_lds(consumer_dsc, consumer_lds_idx)
    source_layout = _primary_layout(source_primary)
    destination_layout = _primary_layout(destination_primary)
    source_stick = _primary_stick(source_primary)
    destination_stick = _primary_stick(destination_primary)
    if (
        source_layout == ["mb", "out"]
        and destination_layout == ["out", "mb"]
        and source_stick == ["out"]
        and destination_stick == ["mb"]
    ):
        return "kernel-to-output"
    if (
        source_layout == ["out", "mb"]
        and destination_layout == ["mb", "out"]
        and source_stick == ["mb"]
        and destination_stick == ["out"]
    ):
        return "output-to-kernel"
    if (
        source_layout == ["mb", "out"]
        and destination_layout == ["mb", "out"]
        and source_stick == ["out"]
        and destination_stick == ["out"]
        and restickify_logical_direction == "output-to-kernel"
    ):
        return "kernel-to-output"
    return f"unknown:{source_layout}:{source_stick}->{destination_layout}:{destination_stick}"


def _infer_restickify_direction(
    dsc: dict[str, Any],
    *,
    input_lds_idx: int,
    output_lds_idx: int,
) -> str:
    source_primary = _primary_for_lds(dsc, input_lds_idx)
    destination_primary = _primary_for_lds(dsc, output_lds_idx)
    source_layout = _primary_layout(source_primary)
    destination_layout = _primary_layout(destination_primary)
    source_stick = _primary_stick(source_primary)
    destination_stick = _primary_stick(destination_primary)
    if (
        source_layout == ["mb", "out"]
        and destination_layout == ["out", "mb"]
        and source_stick == ["out"]
        and destination_stick == ["mb"]
    ):
        return "kernel-to-output"
    if (
        source_layout == ["out", "mb"]
        and destination_layout == ["mb", "out"]
        and source_stick == ["mb"]
        and destination_stick == ["out"]
    ):
        return "output-to-kernel"
    return f"unknown:{source_layout}:{source_stick}->{destination_layout}:{destination_stick}"


def _infer_size_and_cores(
    root: dict[str, Any],
    dsc: dict[str, Any],
) -> tuple[int, int]:
    num_cores = int(root.get("numCoresUsed_", 32) or 32)
    n = dsc.get("N_", {}) or {}
    mb = int(n.get("mb_", n.get("mb", -1)))
    out = int(n.get("out_", n.get("out", -1)))
    if mb <= 0 or out <= 0 or mb != out:
        raise ValueError(f"PT-LX bridge requires square mb/out sizes, got {n}")
    return mb, num_cores


def _constant_lx_start_payload(*, num_cores: int, base: int) -> dict[str, Any]:
    return {
        "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
        "dim_prop_attr": [
            {"factor_": num_cores, "label_": "core"},
            {"factor_": 1, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": {f"[{core}, 0, 0]": str(base) for core in range(num_cores)},
    }


def _corelet_factor(start_payload: dict[str, Any]) -> int:
    attrs = start_payload.get("dim_prop_attr", [])
    if len(attrs) > 1:
        return int(attrs[1].get("factor_", 1) or 1)
    return 1


def _unique_start_values(start_payload: dict[str, Any]) -> list[int]:
    return sorted({int(value) for value in start_payload.get("data_", {}).values()})


def _constant_lx_core_state_init(start_payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = start_payload.get("data_", {}) or {}
    core_values: dict[int, dict[int, int]] = {}
    for key, raw_value in data.items():
        try:
            core_str, corelet_str, _time_str = key.strip("[]").split(",")
            core = int(core_str.strip())
            corelet = int(corelet_str.strip())
        except ValueError:
            continue
        core_values.setdefault(core, {})[corelet] = int(raw_value)
    return [
        {
            "ebrInit_": -1,
            "gtr_": {
                "type": "multicast",
                "id": 18446744073709551615,
                "count": 0,
                "sharers": 0,
                "groupInfo_": {},
            },
            "condGtr_": [],
            "lbrInit_": [
                core_values[core][corelet]
                for corelet in sorted(core_values[core])
            ],
            "gapPerDim_": {},
            "lxSizeWithGaps_": 2_147_483_647,
            "lbrInitForwardGap_": 0,
        }
        for core in sorted(core_values)
    ]


def _patch_lx_allocation_by_index(
    payload: dict[str, Any],
    *,
    lds_idx: int,
    start_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    root, dsc = _single_payload_dsc(payload)
    corelet_factor = _corelet_factor(start_payload)
    root["coreletFoldProp_"] = {"factor_": corelet_factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = corelet_factor
    dsc["numCoreletsUsed_DSC2_"] = corelet_factor
    patches: list[dict[str, Any]] = []
    lds_name = None
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            lds_name = str(lds.get("dsName_", f"lds{lds_idx}"))
            lx_meta = dict(lds.get("memOrg_", {}).get("lx", {}))
            lx_meta.update(
                {
                    "isPresent": 1,
                    "isPadded": 0,
                    "isZeroPadded": 0,
                    "zpadGapFront": [0, 0],
                    "gapPerDim": {},
                    "dsOffset": 0,
                    "allocateNode_": f"allocate-{lds_name}_lx",
                }
            )
            lds["memOrg_"] = {"lx": lx_meta}
            lds["hbmStartAddress_"] = -1
            lds["hbmSize_"] = 0
            if int(lds.get("lxSize_", 0) or 0) <= 0:
                lds["lxSize_"] = 2_147_483_647
            if int(lds.get("lxBufferSize_", 0) or 0) <= 0:
                lds["lxBufferSize_"] = 2_147_483_647
            break
    if lds_name is None:
        raise ValueError(f"LDS index {lds_idx} not found")
    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
        ):
            before = node.get("component_")
            node["name_"] = f"allocate-{lds_name}_lx"
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload
            patches.append({"node": node["name_"], "before_component": before})
    return patches


def _patch_consumer_input_lx_map(
    payload: dict[str, Any],
    *,
    input_name: str,
    lds_idx: int,
    start_payload: dict[str, Any],
) -> None:
    _, dsc = _single_payload_dsc(payload)
    allocate_name = f"allocate-{input_name}_lx"
    for lds in dsc["labeledDs_"]:
        if int(lds["ldsIdx_"]) != int(lds_idx):
            continue
        lx_meta = dict((lds.get("memOrg_", {}) or {}).get("lx", {}))
        lx_meta.update(
            {
                "isPresent": 1,
                "allocateNode_": allocate_name,
            }
        )
        lds["memOrg_"] = {"lx": lx_meta}
        lds["hbmStartAddress_"] = -1
        lds["hbmSize_"] = 0
        if int(lds.get("lxSize_", 0) or 0) <= 0:
            lds["lxSize_"] = 2_147_483_647
        if int(lds.get("lxBufferSize_", 0) or 0) <= 0:
            lds["lxBufferSize_"] = 2_147_483_647
        lds["coreStateInit_"] = _constant_lx_core_state_init(start_payload)
    for node in dsc.get("scheduleTree_", []) or []:
        if (
            node.get("nodeType_") == "allocate"
            and int(node.get("ldsIdx_", -1)) == int(lds_idx)
        ):
            node["name_"] = allocate_name
            node["component_"] = "lx"
            node["startAddressCoreCorelet_"] = start_payload


def _force_consumer_corelets(payload: dict[str, Any], *, factor: int) -> None:
    root, dsc = _single_payload_dsc(payload)
    root["coreletFoldProp_"] = {"factor_": factor, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = factor
    dsc["numCoreletsUsed_DSC2_"] = factor


def _patch_bridge_endpoint_pieces(
    payload: dict[str, Any],
    *,
    producer_starts: dict[int, int],
    consumer_starts: dict[int, int],
) -> dict[str, Any]:
    root = next(iter(payload.values()))
    datadscs = root.get("datadscs_", []) or []
    if not datadscs:
        raise ValueError("expected at least one bridge datadsc")
    first = next(iter(datadscs[0].values()))
    last = next(iter(datadscs[-1].values()))
    return {
        "producer_pieces_patched": len(
            _patch_piece_starts(first["labeledDs_"][0], producer_starts)
        ),
        "consumer_pieces_patched": len(
            _patch_piece_starts(last["labeledDs_"][-1], consumer_starts)
        ),
        "num_dataops": len(datadscs),
    }


def _patch_piece_starts(
    lds: dict[str, Any],
    starts_by_core: dict[int, int],
) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for piece in lds.get("PieceInfo", []) or []:
        placements = piece.get("PlacementInfo", []) or []
        lx_placements = [
            placement for placement in placements if placement.get("type") == "lx"
        ]
        if not lx_placements:
            continue
        placement = lx_placements[0]
        mem_id = placement.get("memId") or []
        if not mem_id:
            continue
        core = int(mem_id[0])
        if core not in starts_by_core:
            continue
        before = list(placement.get("startAddr", []))
        placement["startAddr"] = [int(starts_by_core[core])]
        patched.append(
            {
                "piece": piece.get("key_"),
                "core": core,
                "before": before,
                "after": placement["startAddr"],
            }
        )
    return patched
