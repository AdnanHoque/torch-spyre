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

"""Default-off DDL-shaped LX-local restickify bridge.

This prototype reshapes a normal Torch-Spyre ``ReStickifyOpHBM`` compute SDSC
into the compact pre-DDC contract accepted by Deeptools' restickify DDL
template. It is intentionally conservative and exists only behind an explicit
config flag.
"""

from __future__ import annotations

import copy
from typing import Any

from sympy import Expr, Symbol

from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec

from .compute_ops import num_bytes
from .superdsc import SDSCSpec

_MAX_PROTOTYPE_LX_BYTES_PER_CORE = 512 * 1024


def restickify_ddl_bridge_skip_reason(
    op_spec: OpSpec,
    sdsc_spec: SDSCSpec,
) -> str | None:
    """Return why the DDL bridge should not handle this restickify."""
    if op_spec.op != RESTICKIFY_OP or sdsc_spec.opfunc != RESTICKIFY_OP:
        return "not-restickify"
    source_kind = op_spec.op_info.get("restickify_source_kind")
    if source_kind != "in_graph_computed":
        return (
            "source-not-in-graph-computed"
            if source_kind is not None
            else "source-kind-unknown"
        )
    if sdsc_spec.num_inputs != 1 or len(sdsc_spec.args) != 2:
        return "unsupported-restickify-arity"
    if sdsc_spec.constants:
        return "has-constants"
    if sdsc_spec.padding:
        return "has-padding"
    if sdsc_spec.coordinate_masking:
        return "has-coordinate-masking"

    split_dims = [
        dim for dim, split in sdsc_spec.work_slices.items() if _as_int_or_none(split) != 1
    ]
    if len(split_dims) != 1:
        return "expected-one-split-dim"
    split_dim = split_dims[0]
    if _as_int_or_none(sdsc_spec.work_slices[split_dim]) != sdsc_spec.num_cores:
        return "split-dim-does-not-cover-all-cores"

    # Stage 42 only allowed the direction where the DDL output was stickified on
    # the split/owner dimension. Stage 49 showed that the mirrored direction also
    # lowers through DDC/DCC/DXP once restickify.ddl uses the SFP/LX input-port
    # source spelling. Keep the remaining gates shape/source based and let the
    # default-off prototype exercise both restickify directions.

    per_core_bytes = _arg_bytes_per_core(sdsc_spec)
    if per_core_bytes is None:
        return "non-concrete-shape"
    if per_core_bytes > _MAX_PROTOTYPE_LX_BYTES_PER_CORE:
        return "lx-bytes-per-core-too-large"
    return None


def generate_restickify_ddl_bridge_sdsc(
    idx: int,
    sdsc_spec: SDSCSpec,
    compute_payload: dict[str, Any],
) -> dict[str, Any]:
    """Generate a DDL-template input SDSC from a normal restickify SDSC."""
    sdsc_name, root = _single_root(compute_payload)
    dsc_name, dsc = _single_dsc(root)
    op = copy.deepcopy(dsc["computeOp_"][0])

    input_idx = _parse_idx(op["inputLabeledDs"][0])
    output_idx = _parse_idx(op["outputLabeledDs"][0])
    lds_by_idx = {int(lds["ldsIdx_"]): lds for lds in dsc["labeledDs_"]}
    input_lds = lds_by_idx[input_idx]
    output_lds = lds_by_idx[output_idx]

    input_primary = copy.deepcopy(dsc["primaryDsInfo_"][input_lds["dsType_"]])
    output_primary = copy.deepcopy(dsc["primaryDsInfo_"][output_lds["dsType_"]])
    input_layout = list(input_primary["layoutDimOrder_"])
    output_layout = list(output_primary["layoutDimOrder_"])

    dims = _known_dims(root, dsc)
    reduced_dims = _positive_layout_dims(dims, [input_layout, output_layout])
    n_struct = _new_dim_struct("n", {**dims, **reduced_dims})
    neg_dims = {dim: -1 for dim in dims}

    num_cores = int(root.get("numCoresUsed_") or dsc.get("numCoresUsed_") or 1)
    input_name = input_lds["dsName_"]
    output_name = output_lds["dsName_"]
    input_alloc = f"allocate_{input_name}_lx"
    output_alloc = f"allocate_{output_name}_lx"
    input_transfer = "transfer_lds0_src:no_component_dst:lx_lx_local"
    output_transfer = "transfer_lds1_src:lx_dst:no_component_lx_local"

    alloc_templates = [
        node for node in dsc.get("scheduleTree_", []) if node.get("nodeType_") == "allocate"
    ]
    if not alloc_templates:
        raise ValueError("restickify DDL bridge needs allocate nodes")
    input_alloc_template = next(
        (node for node in alloc_templates if node.get("ldsIdx_") == input_idx),
        alloc_templates[0],
    )
    output_alloc_template = next(
        (node for node in alloc_templates if node.get("ldsIdx_") == output_idx),
        alloc_templates[-1],
    )

    loops = _loop_skeleton(list(reversed(input_layout)))
    input_lx_size = _arg_lx_size(sdsc_spec.args[0], sdsc_spec)
    output_lx_size = _arg_lx_size(sdsc_spec.args[-1], sdsc_spec)

    stage0 = _base_stage_param(
        dsc,
        dims,
        root.get("numWkSlicesPerDim_", {}) or {},
        name="core",
    )
    stage1 = copy.deepcopy(stage0)
    stage1["ss_"]["name_"] = "chunk"
    stage1["el_"]["name_"] = "chunk"

    out_root = copy.deepcopy(root)
    out_dsc = copy.deepcopy(dsc)
    out_sdsc_name = f"{idx}_{RESTICKIFY_OP}_ddl_bridge"
    out_dsc_name = f"{RESTICKIFY_OP}_ddl_bridge"
    out_root.update(
        {
            "coreFoldProp_": copy.deepcopy(
                root.get("coreFoldProp_") or {"factor_": num_cores, "label_": "core"}
            ),
            "coreletFoldProp_": copy.deepcopy(
                root.get("coreletFoldProp_") or {"factor_": 1, "label_": "corelet"}
            ),
            "numCoresUsed_": num_cores,
            "unpadN_": copy.deepcopy(n_struct),
            "N_": copy.deepcopy(n_struct),
            "opFuncsUsed_": [],
            "ldsShareInfo_": [],
            "prodConsList": {},
            "dimToSymbolMappingOpcodeCorrection_": {},
            "inputSymbolsAndTags_": {},
            "symbolDefinitions_": {},
        }
    )
    out_dsc.update(
        {
            "numCoresUsed_": num_cores,
            "numCoreletsUsed_": 1,
            "coreIdsUsed_": list(range(num_cores)),
            "unpadN_": copy.deepcopy(n_struct),
            "N_": copy.deepcopy(n_struct),
            "dscN_": _new_dim_struct("dscn", neg_dims),
            "ChipD_": _new_dim_struct("chipd", neg_dims),
            "ChipletD_": _new_dim_struct("chipletd", neg_dims),
            "CoreD_": _new_dim_struct("d", neg_dims),
            "CoreletD_": _new_dim_struct("coreletd", neg_dims),
            "B_": _new_dim_struct("b", neg_dims),
            "T_": _new_dim_struct("t", neg_dims),
            "Tel_": _new_dim_struct("tel", neg_dims),
            "P_": _new_dim_struct("p", neg_dims),
            "Pel_": _new_dim_struct("pel", neg_dims),
            "loopOrder_": [],
            "loopProperties_": {},
            "auxLoopOrder_": [],
            "coordinateMasking_": {},
            "maskingConstId_": -1,
            "dimToSymbolMapping_": {},
            "numCoreletsUsed_DSC2_": 1,
            "dataStageParam_": {"0": stage0, "1": stage1},
            "constantInfo_": {},
            "gtrIdsUsed_": [],
            "l0TetheredMode_": "none",
            "scheduleTreeHeadDenId_": 0,
            "primaryDsInfo_": {
                input_lds["dsType_"]: input_primary,
                output_lds["dsType_"]: output_primary,
            },
            "pdsRelation_": {},
            "labeledDs_": [
                _lx_labeled_ds(
                    input_lds,
                    idx=0,
                    name=input_name,
                    role=input_lds["dsType_"],
                    layout=input_layout,
                    lx_size=input_lx_size,
                    alloc_name=input_alloc,
                ),
                _lx_labeled_ds(
                    output_lds,
                    idx=1,
                    name=output_name,
                    role=output_lds["dsType_"],
                    layout=output_layout,
                    lx_size=output_lx_size,
                    alloc_name=output_alloc,
                ),
            ],
        }
    )
    out_dsc["scheduleTree_"] = [
        _alloc_node(
            input_alloc_template,
            name=input_alloc,
            lds_idx=0,
            layout=input_layout,
            user=input_transfer,
        ),
        _alloc_node(
            output_alloc_template,
            name=output_alloc,
            lds_idx=1,
            layout=output_layout,
            user=output_transfer,
        ),
        _transfer_node(input_transfer, src_lx_idx=None, dst_lx_idx=0),
        *loops,
        {
            "nodeType_": "block",
            "name_": "lx_below_schedule",
            "prev_": loops[-1]["name_"] if loops else "",
            "relevantComps_": {},
            "next_": [],
        },
        _transfer_node(output_transfer, src_lx_idx=1, dst_lx_idx=None),
    ]
    op.update(
        {
            "opFuncName": RESTICKIFY_OP,
            "inputLabeledDs": [f"{input_name}-idx0"],
            "interimLabeledDs": [],
            "outputLabeledDs": [f"{output_name}-idx1"],
            "indirectAccessIndexLabeledDs": [],
        }
    )
    out_dsc["computeOp_"] = [op]
    out_root["dscs_"] = [{out_dsc_name: out_dsc}]
    return {out_sdsc_name: out_root}


def _as_int_or_none(value: Any) -> int | None:
    try:
        if isinstance(value, Expr) and value.free_symbols:
            return None
        return int(value)
    except Exception:  # noqa: BLE001
        return None


def _arg_bytes_per_core(sdsc_spec: SDSCSpec) -> int | None:
    sizes: list[int] = []
    for arg in sdsc_spec.args:
        size = _arg_lx_size(arg, sdsc_spec)
        if size is None:
            return None
        sizes.append(size)
    return max(sizes) if sizes else None


def _arg_lx_size(arg: Any, sdsc_spec: SDSCSpec) -> int:
    layout_dims = sdsc_spec.layouts[arg.layout]["dim_order"]
    volume = 1
    for dim in layout_dims:
        dim_size = _as_int_or_none(sdsc_spec.iteration_space[dim])
        if dim_size is None:
            raise ValueError(f"non-concrete restickify dimension {dim}")
        split = _as_int_or_none(sdsc_spec.work_slices.get(dim, 1))
        if split is None:
            raise ValueError(f"non-concrete restickify split {dim}")
        volume *= max(1, (dim_size + split - 1) // split)
    return max(1024, volume * num_bytes(arg.data_format))


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC")
    return next(iter(payload.items()))


def _single_dsc(root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dscs = root.get("dscs_", [])
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DSC inside the SDSC")
    return next(iter(dscs[0].items()))


def _parse_idx(label: str) -> int:
    _, idx = label.rsplit("-idx", 1)
    return int(idx)


def _dim_names_from_struct(dim_struct: dict[str, Any]) -> list[str]:
    skip = {
        "name_",
        "symbolicDimInfo_",
        "maxSymbolicVolume_",
        "coreletSplit_",
        "rowSplit_",
        "peSfpSplit_",
        "paddingSizes_",
    }
    return [key[:-1] for key in dim_struct if key.endswith("_") and key not in skip]


def _new_dim_struct(name: str, dims: dict[str, int]) -> dict[str, Any]:
    struct: dict[str, Any] = {"name_": name}
    for dim, value in dims.items():
        struct[f"{dim}_"] = value
    struct.update(
        {
            "symbolicDimInfo_": {},
            "maxSymbolicVolume_": {},
            "coreletSplit_": {},
            "rowSplit_": {},
            "peSfpSplit_": {},
            "paddingSizes_": {},
        }
    )
    return struct


def _known_dims(root: dict[str, Any], dsc: dict[str, Any]) -> dict[str, int]:
    n_struct = root.get("N_") or dsc.get("N_") or {}
    dims = {dim: -1 for dim in _dim_names_from_struct(n_struct)}
    for pds in dsc.get("primaryDsInfo_", {}).values():
        for dim in pds.get("layoutDimOrder_", []):
            value = n_struct.get(f"{dim}_", -1)
            dims[dim] = int(value) if isinstance(value, int) and value > 0 else -1
    for dim in root.get("numWkSlicesPerDim_", {}) or {}:
        dims.setdefault(dim, int(n_struct.get(f"{dim}_", -1)))
    return dims


def _positive_layout_dims(
    dims: dict[str, int],
    layouts: list[list[str]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for layout in layouts:
        for dim in layout:
            value = dims.get(dim, -1)
            out[dim] = value if value > 0 else 1
    return out


def _base_stage_param(
    source_dsc: dict[str, Any],
    dims: dict[str, int],
    slices: dict[str, int],
    *,
    name: str,
) -> dict[str, Any]:
    if source_dsc.get("dataStageParam_"):
        first = copy.deepcopy(next(iter(source_dsc["dataStageParam_"].values())))
        for side in ("ss_", "el_"):
            first[side]["name_"] = name
        return first

    per_slice = {}
    for dim, value in dims.items():
        divisor = int(slices.get(dim, 1) or 1)
        per_slice[dim] = max(1, (value + divisor - 1) // divisor) if value > 0 else value
    return {"ss_": _new_dim_struct(name, per_slice), "el_": _new_dim_struct(name, per_slice)}


def _alloc_node(
    template: dict[str, Any],
    *,
    name: str,
    lds_idx: int,
    layout: list[str],
    user: str,
) -> dict[str, Any]:
    node = copy.deepcopy(template)
    node.update(
        {
            "nodeType_": "allocate",
            "name_": name,
            "prev_": "",
            "ldsIdx_": lds_idx,
            "constIdx_": -1,
            "tempStorageForCompute_": "",
            "component_": "lx",
            "padding_": {},
            "layoutDimOrder_": layout,
            "maxDimSizes_": [-1 for _ in layout],
            "numBuffers_": 1,
            "isStartAddrSymbolic_": 0,
            "backGapCore_": {},
            "indirectAllocType_": "no_indirection",
            "ignoreSymbolicVolumeLimits_": 0,
            "nonUnifiedAllocInHBM_": 0,
            "gapStickSpread_": {},
            "allocUsers_": {user: 1},
        }
    )
    return node


def _loop_skeleton(loop_dims: list[str]) -> list[dict[str, Any]]:
    loops: list[dict[str, Any]] = []
    for index, dim in enumerate(loop_dims):
        prev = "" if index == 0 else f"loop_ds0_ds1_{loop_dims[index - 1]}"
        next_name = (
            f"loop_ds0_ds1_{loop_dims[index + 1]}"
            if index + 1 < len(loop_dims)
            else "lx_below_schedule"
        )
        loops.append(
            {
                "nodeType_": "loop",
                "name_": f"loop_ds0_ds1_{dim}",
                "prev_": prev,
                "relevantComps_": {},
                "next_": [next_name],
                "numId_": 0,
                "denId_": 1,
                "parametricLoop_": 0,
                "parametricIterCount_": -1,
                "parametricLdsIdx_": -1,
                "dims_": [{"dim_": dim, "kind_": "unpadded"}],
                "loopCountSymbolIds_": {},
            }
        )
    return loops


def _transfer_node(
    name: str,
    *,
    src_lx_idx: int | None,
    dst_lx_idx: int | None,
) -> dict[str, Any]:
    src_storage = "lx" if src_lx_idx is not None else "no_component"
    dst_storage = "lx" if dst_lx_idx is not None else "no_component"
    return {
        "nodeType_": "transfer",
        "name_": name,
        "prev_": "",
        "relevantComps_": {},
        "src_": {"unit_": "no_component", "storage_": src_storage},
        "dstVias_": [{"loc_": {"unit_": "no_component", "storage_": dst_storage}, "via_": []}],
        "lastFusableParentLoopSrc_": "",
        "lastFusableParentLoopDst_": [],
        "srcLdsAndLoopOffsets_": _lds_loop_offsets(src_lx_idx),
        "dstLdsAndLoopOffsets_": [_lds_loop_offsets(dst_lx_idx)],
        "replicationFactor_": 1,
        "unitTimeTransfer_": [],
        "rotateNumElements_": 0,
        "coreIdToGTRInfo_": {},
        "transferSize_": {},
        "coreletViews_": {},
    }


def _lds_loop_offsets(lds_idx: int | None) -> dict[str, Any]:
    return {
        "myLdsIdx_": lds_idx if lds_idx is not None else -1,
        "startAddr_": "0",
        "isStartAddrSymbolic_": 0,
        "latchDataId_": -1,
        "constantId_": -1,
        "constEleOffsets_": {},
        "loopEleOffsets_": {},
        "bufferAddrOffset_": {},
        "bufferSwitchPosition_": "",
        "dataConnect_": "",
    }


def _lx_labeled_ds(
    template: dict[str, Any],
    *,
    idx: int,
    name: str,
    role: str,
    layout: list[str],
    lx_size: int,
    alloc_name: str,
) -> dict[str, Any]:
    lds = copy.deepcopy(template)
    lds.update(
        {
            "ldsIdx_": idx,
            "dsName_": name,
            "dsType_": role,
            "segment_": "stack",
            "isFirstUse_": 0,
            "isExternal_": 0,
            "scale_": [1 for _ in layout],
            "density_": [1 for _ in layout],
            "level": 0,
            "memOrg_": {
                "lx": {
                    "isPresent": 1,
                    "isPadded": 0,
                    "isZeroPadded": 0,
                    "zpadGapFront": [0, 0],
                    "gapPerDim": {},
                    "dsOffset": 0,
                    "allocateNode_": alloc_name,
                }
            },
            "dataTransfers_": [],
            "hbmStartAddress_": -1,
            "lxStartAddress_": -1,
            "hbmSize_": 2_147_483_647,
            "lxSize_": lx_size,
            "lxBufferSize_": 2_147_483_647,
            "totSlicesPerDim_": {},
            "coreStateInit_": [],
        }
    )
    return lds
