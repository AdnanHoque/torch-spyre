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
import os
from typing import Any

from sympy import Expr, Symbol

from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.constants import SEGMENT_OFFSETS
from torch_spyre._inductor.op_spec import OpSpec

from .compute_ops import num_bytes
from .superdsc import SDSCSpec

_MAX_PROTOTYPE_LX_BYTES_PER_CORE = 512 * 1024
_BRIDGE_OPFUNC_ENV = "SPYRE_RESTICKIFY_DDL_BRIDGE_OPFUNC"
_INTERSLICE_TRANSPOSE_FP16_OP = "interslicetranspose_fp16"
_SUPPORTED_BRIDGE_OPFUNCS = {
    RESTICKIFY_OP,
    "ReStickifyOpLx",
    "ReStickifyOpWithPTLx",
    "ReStickifyOpWithPTHBM",
    _INTERSLICE_TRANSPOSE_FP16_OP,
}
_BRIDGE_SOURCE_ADDRESS_ENV = "SPYRE_RESTICKIFY_DDL_BRIDGE_SOURCE_ADDRESS"
_BRIDGE_SOURCE_ADDRESS_DEFAULT = "runtime-segment"
_BRIDGE_SOURCE_ADDRESS_COMPACT_LXLU = "compact-lxlu"
_SUPPORTED_BRIDGE_SOURCE_ADDRESS = {
    _BRIDGE_SOURCE_ADDRESS_DEFAULT,
    _BRIDGE_SOURCE_ADDRESS_COMPACT_LXLU,
}
_ALLOW_MULTI_SPLIT_ENV = "SPYRE_RESTICKIFY_DDL_BRIDGE_ALLOW_MULTI_SPLIT"
_BRIDGE_LOOP_ORDER_ENV = "SPYRE_RESTICKIFY_DDL_BRIDGE_LOOP_ORDER"
_BRIDGE_LOOP_ORDER_REVERSED_INPUT = "reversed-input"
_SUPPORTED_BRIDGE_LOOP_ORDERS = {
    "input",
    "output",
    _BRIDGE_LOOP_ORDER_REVERSED_INPUT,
    "reversed-output",
}
_BRIDGE_INTERSLICE_GLOBAL_LAYOUT_ENV = (
    "SPYRE_RESTICKIFY_DDL_BRIDGE_INTERSLICE_GLOBAL_LAYOUT"
)
_BRIDGE_INTERSLICE_REFERENCE_CONTRACT_ENV = (
    "SPYRE_RESTICKIFY_DDL_BRIDGE_INTERSLICE_REFERENCE_CONTRACT"
)
_BRIDGE_INTERSLICE_PRESERVE_CORE_MAPPING_ENV = (
    "SPYRE_RESTICKIFY_DDL_BRIDGE_INTERSLICE_PRESERVE_CORE_MAPPING"
)
_BRIDGE_INTERSLICE_GLOBAL_LAYOUT_AS_IS = "as-is"
_SUPPORTED_BRIDGE_INTERSLICE_GLOBAL_LAYOUTS = {
    _BRIDGE_INTERSLICE_GLOBAL_LAYOUT_AS_IS,
    "input",
    "output",
}


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
    if any(
        _runtime_segment_for_start_address(arg.start_address) is None
        for arg in sdsc_spec.args
    ):
        return "unsupported-runtime-segment"

    split_dims = [
        dim for dim, split in sdsc_spec.work_slices.items() if _as_int_or_none(split) != 1
    ]
    allow_multi_split = os.environ.get(_ALLOW_MULTI_SPLIT_ENV, "0") == "1"
    if len(split_dims) != 1 and not allow_multi_split:
        return "expected-one-split-dim"
    split_product = 1
    for split_dim in split_dims:
        split_factor = _as_int_or_none(sdsc_spec.work_slices[split_dim])
        if split_factor is None:
            return "non-concrete-split-dim"
        split_product *= split_factor
    if split_product != sdsc_spec.num_cores:
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
    bridge_opfunc = _bridge_opfunc_name()
    reference_interslice = (
        bridge_opfunc == _INTERSLICE_TRANSPOSE_FP16_OP
        and os.environ.get(_BRIDGE_INTERSLICE_REFERENCE_CONTRACT_ENV, "0") == "1"
    )
    if bridge_opfunc == _INTERSLICE_TRANSPOSE_FP16_OP:
        _apply_interslice_global_layout_probe(input_primary, output_primary)
    input_layout = list(input_primary["layoutDimOrder_"])
    output_layout = list(output_primary["layoutDimOrder_"])
    input_role = input_lds["dsType_"]
    output_role = output_lds["dsType_"]

    if reference_interslice:
        (
            input_role,
            output_role,
            input_primary,
            output_primary,
            input_layout,
            output_layout,
        ) = _reference_interslice_contract(input_primary, output_primary)

    dims = _known_dims(root, dsc)
    reduced_dims = _positive_layout_dims(dims, [input_layout, output_layout])
    if reference_interslice:
        _ensure_reference_interslice_dims(dims, reduced_dims)
    n_struct = _new_dim_struct("n", {**dims, **reduced_dims})
    neg_dims = {dim: -1 for dim in dims}

    num_cores = int(root.get("numCoresUsed_") or dsc.get("numCoresUsed_") or 1)
    # The value-correct interslice prototype uses PT internally, but its DDL
    # contract is a single-corelet SDSC.  Forcing a two-corelet fold here makes
    # DXP's caller/cardinality import disagree with the JSON even for the
    # 512-size bridge that should otherwise compile.
    num_corelets = 1
    input_name = input_lds["dsName_"]
    output_name = output_lds["dsName_"]
    input_alloc = f"allocate_{input_name}_lx"
    output_alloc = f"allocate_{output_name}_lx"
    input_transfer = "transfer_lds0_src:no_component_dst:lx_lx_local"
    output_transfer = "transfer_lds1_src:lx_dst:no_component_lx_local"
    if reference_interslice:
        input_transfer = f"prefill_{input_transfer}"
        output_transfer = f"prefill_{output_transfer}"
    source_address_mode = _bridge_source_address_mode()

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

    loops = _loop_skeleton(_bridge_loop_dims(input_layout, output_layout))
    input_lx_size = _arg_lx_size(sdsc_spec.args[0], sdsc_spec)
    output_lx_size = _arg_lx_size(sdsc_spec.args[-1], sdsc_spec)
    input_segment = _required_runtime_segment(sdsc_spec.args[0].start_address)
    output_segment = _required_runtime_segment(sdsc_spec.args[-1].start_address)

    stage0 = _base_stage_param(
        dsc,
        dims,
        root.get("numWkSlicesPerDim_", {}) or {},
        name="core",
    )
    stage1 = copy.deepcopy(stage0)
    stage1["ss_"]["name_"] = "chunk"
    stage1["el_"]["name_"] = "chunk"
    num_wk_slices = copy.deepcopy(root.get("numWkSlicesPerDim_", {}) or {})
    if reference_interslice:
        stage0, stage1 = _reference_interslice_stage_params(
            n_struct,
            num_wk_slices,
            input_primary,
        )

    out_root = copy.deepcopy(root)
    out_dsc = copy.deepcopy(dsc)
    out_sdsc_name = f"{idx}_{bridge_opfunc}_ddl_bridge"
    out_dsc_name = f"{bridge_opfunc}_ddl_bridge"
    out_root.update(
        {
            "coreFoldProp_": copy.deepcopy(
                root.get("coreFoldProp_") or {"factor_": num_cores, "label_": "core"}
            ),
            "coreletFoldProp_": copy.deepcopy(
                {"factor_": num_corelets, "label_": "corelet"}
                if reference_interslice
                else root.get("coreletFoldProp_")
                or {"factor_": 1, "label_": "corelet"}
            ),
            "numCoresUsed_": num_cores,
            "numWkSlicesPerDim_": num_wk_slices,
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
    if reference_interslice:
        _ensure_reference_interslice_wk_slice(out_root)
    out_dsc.update(
        {
            "numCoresUsed_": num_cores,
            "numCoreletsUsed_": num_corelets,
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
            "numCoreletsUsed_DSC2_": num_corelets,
            "dataStageParam_": {"0": stage0, "1": stage1},
            "constantInfo_": {},
            "gtrIdsUsed_": [],
            "l0TetheredMode_": "none",
            "scheduleTreeHeadDenId_": 0,
            "primaryDsInfo_": {
                input_role: input_primary,
                output_role: output_primary,
            },
            "pdsRelation_": {},
            "labeledDs_": [
                _lx_labeled_ds(
                    input_lds,
                    idx=0,
                    name=input_name,
                    role=input_role,
                    layout=input_layout,
                    segment=input_segment,
                    lx_size=input_lx_size,
                    alloc_name=input_alloc,
                ),
                _lx_labeled_ds(
                    output_lds,
                    idx=1,
                    name=output_name,
                    role=output_role,
                    layout=output_layout,
                    segment=output_segment,
                    lx_size=output_lx_size,
                    alloc_name=output_alloc,
                ),
            ],
        }
    )
    input_alloc_node = _alloc_node(
        input_alloc_template,
        name=input_alloc,
        lds_idx=0,
        layout=input_layout,
        user=input_transfer,
    )
    input_transfer_node = _transfer_node(input_transfer, src_lx_idx=None, dst_lx_idx=0)
    if source_address_mode == _BRIDGE_SOURCE_ADDRESS_COMPACT_LXLU:
        _set_compact_start_address(input_alloc_node, num_cores, num_corelets)
        dst_offset = input_transfer_node["dstLdsAndLoopOffsets_"][0]
        _set_compact_start_address(
            dst_offset,
            num_cores,
            num_corelets,
            field="startAddr_",
        )
        dst_offset["dataConnect_"] = "lxlu_input"

    output_alloc_node = _alloc_node(
        output_alloc_template,
        name=output_alloc,
        lds_idx=1,
        layout=output_layout,
        user=output_transfer,
    )
    if reference_interslice and source_address_mode == _BRIDGE_SOURCE_ADDRESS_COMPACT_LXLU:
        _set_compact_start_address(
            output_alloc_node,
            num_cores,
            num_corelets,
            base=input_lx_size,
        )

    output_transfer_node = _transfer_node(output_transfer, src_lx_idx=1, dst_lx_idx=None)
    if reference_interslice:
        output_transfer_node["srcLdsAndLoopOffsets_"]["dataConnect_"] = "lxsu_input"

    out_dsc["scheduleTree_"] = [
        input_alloc_node,
        output_alloc_node,
        input_transfer_node,
        *loops,
        _block_node(loops[-1]["name_"] if loops else ""),
        output_transfer_node,
    ]
    if reference_interslice:
        out_dsc["scheduleTree_"] = _reference_interslice_schedule_tree(
            input_alloc_node,
            input_transfer_node,
            output_alloc_node,
            output_transfer_node,
            loop_dims=input_layout,
        )
    op.update(
        {
            "opFuncName": bridge_opfunc,
            "inputLabeledDs": [f"{input_name}-idx0"],
            "interimLabeledDs": [],
            "outputLabeledDs": [f"{output_name}-idx1"],
            "indirectAccessIndexLabeledDs": [],
        }
    )
    if reference_interslice:
        op["exUnit"] = "pt"
        op["location"] = "Inner"
        op["isAtMainLoop"] = 1
        op["isAtTop"] = 1
        op["level"] = 0
        op.pop("attributes_", None)
    out_dsc["computeOp_"] = [op]
    out_root["dscs_"] = [{out_dsc_name: out_dsc}]
    return {out_sdsc_name: out_root}


def _bridge_opfunc_name() -> str:
    opfunc = os.environ.get(_BRIDGE_OPFUNC_ENV, RESTICKIFY_OP)
    if opfunc not in _SUPPORTED_BRIDGE_OPFUNCS:
        raise ValueError(
            f"{_BRIDGE_OPFUNC_ENV}={opfunc!r} is unsupported; "
            f"choose one of {sorted(_SUPPORTED_BRIDGE_OPFUNCS)}"
        )
    return opfunc


def _bridge_source_address_mode() -> str:
    mode = os.environ.get(_BRIDGE_SOURCE_ADDRESS_ENV, _BRIDGE_SOURCE_ADDRESS_DEFAULT)
    if mode not in _SUPPORTED_BRIDGE_SOURCE_ADDRESS:
        raise ValueError(
            f"{_BRIDGE_SOURCE_ADDRESS_ENV}={mode!r} is unsupported; "
            f"choose one of {sorted(_SUPPORTED_BRIDGE_SOURCE_ADDRESS)}"
        )
    return mode


def _apply_interslice_global_layout_probe(
    input_primary: dict[str, Any],
    output_primary: dict[str, Any],
) -> None:
    mode = os.environ.get(
        _BRIDGE_INTERSLICE_GLOBAL_LAYOUT_ENV,
        _BRIDGE_INTERSLICE_GLOBAL_LAYOUT_AS_IS,
    )
    if mode not in _SUPPORTED_BRIDGE_INTERSLICE_GLOBAL_LAYOUTS:
        raise ValueError(
            f"{_BRIDGE_INTERSLICE_GLOBAL_LAYOUT_ENV}={mode!r} is unsupported; "
            f"choose one of {sorted(_SUPPORTED_BRIDGE_INTERSLICE_GLOBAL_LAYOUTS)}"
        )
    if mode == "input":
        output_primary["layoutDimOrder_"] = list(input_primary["layoutDimOrder_"])
    elif mode == "output":
        input_primary["layoutDimOrder_"] = list(output_primary["layoutDimOrder_"])


def _reference_interslice_contract(
    input_primary: dict[str, Any],
    output_primary: dict[str, Any],
) -> tuple[str, str, dict[str, Any], dict[str, Any], list[str], list[str]]:
    """Return the reference-shaped interslice contract accepted by DDC.

    Deeptools' checked-in interslice SDSC models the output stick as the input
    stick dimension carried through the slice layout plus the new output stick
    dimension.  Preserve that shape for this prototype instead of inheriting
    Torch-Spyre's OUTPUT/KERNEL role names directly.
    """

    output_layout = list(output_primary["layoutDimOrder_"])
    if not output_layout:
        output_layout = list(input_primary["layoutDimOrder_"])
    canonical_layout = list(output_layout)

    input_stick = list(input_primary.get("stickDimOrder_", []) or [])
    if not input_stick:
        input_stick = canonical_layout[:1]
    output_stick = list(output_primary.get("stickDimOrder_", []) or [])
    if not output_stick:
        output_stick = canonical_layout[-1:]

    ref_input = {
        "layoutDimOrder_": canonical_layout,
        "stickDimOrder_": input_stick,
        "stickSize_": [64 for _ in input_stick],
        "stickRepl_": [1 for _ in input_stick],
    }
    ref_output = {
        "layoutDimOrder_": canonical_layout,
        "stickDimOrder_": output_stick,
        "stickSize_": [64 for _ in output_stick],
        "stickRepl_": [1 for _ in output_stick],
    }
    return "INPUT", "OUTPUT", ref_input, ref_output, canonical_layout, canonical_layout


def _ensure_reference_interslice_dims(
    dims: dict[str, int],
    reduced_dims: dict[str, int],
) -> None:
    # The Deeptools reference SDSC for this template carries the standard DSC
    # dimension vocabulary even when most dimensions are dropped.  DDC's
    # fold/cardinality import path expects those names to exist.
    for dim in (
        "in",
        "i",
        "j",
        "ki",
        "kj",
        "x",
        "x1",
        "r",
        "c",
        "ij",
        "rc",
        "kij",
        "sij",
        "zij",
        "si",
        "sj",
        "zi",
        "zj",
    ):
        dims.setdefault(dim, -1)
    for dim in ("i", "j", "ij"):
        reduced_dims.setdefault(dim, 1)


def _ensure_reference_interslice_wk_slice(root: dict[str, Any]) -> None:
    if os.environ.get(_BRIDGE_INTERSLICE_PRESERVE_CORE_MAPPING_ENV, "0") == "1":
        for wk_slice in (root.get("coreIdToWkSlice_") or {}).values():
            if isinstance(wk_slice, dict):
                if "y" in (root.get("numWkSlicesPerDim_") or {}):
                    wk_slice.setdefault("y", 0)
        return
    slices = root.get("numWkSlicesPerDim_") or {}
    num_cores = int(root.get("numCoresUsed_") or 1)
    out_slices = int(slices.get("out", 0) or 0)
    mb_slices = int(slices.get("mb", 0) or 0)
    if out_slices > 0 and mb_slices > 0 and out_slices * mb_slices == num_cores:
        root["coreIdToWkSlice_"] = {
            str(core): {
                "out": core % out_slices,
                "mb": core // out_slices,
            }
            for core in range(num_cores)
        }
        return
    for wk_slice in (root.get("coreIdToWkSlice_") or {}).values():
        if isinstance(wk_slice, dict):
            if "y" in slices:
                wk_slice.setdefault("y", 0)


def _reference_interslice_stage_params(
    n_struct: dict[str, Any],
    slices: dict[str, int],
    input_primary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_stick = list(input_primary.get("stickDimOrder_", []) or [])
    split_dim = input_stick[0] if input_stick else "mb"
    split_peer = "mb" if split_dim != "mb" else "out"
    core_sizes: dict[str, int] = {}
    for dim in input_primary["layoutDimOrder_"]:
        value = int(n_struct.get(f"{dim}_", 1) or 1)
        divisor = int(slices.get(dim, 1) or 1)
        core_sizes[dim] = max(1, (value + divisor - 1) // divisor)
    if split_peer in core_sizes:
        corelet_dim = split_peer
    else:
        corelet_dim = next(
            (dim for dim in input_primary["layoutDimOrder_"] if dim != split_dim),
            split_dim,
        )

    def stage(name: str, sizes: dict[str, int], split_values: list[int]) -> dict[str, Any]:
        payload = _new_dim_struct(name, sizes)
        payload["coreletSplit_"] = {corelet_dim: split_values}
        return {"ss_": copy.deepcopy(payload), "el_": copy.deepcopy(payload)}

    corelet_extent = max(2, core_sizes.get(corelet_dim, 1))
    first = corelet_extent // 2
    second = corelet_extent - first
    core_split = [first, second]
    chunk_sizes = dict(core_sizes)
    chunk_sizes[corelet_dim] = min(corelet_extent, 32)
    chunk_split = [
        max(1, chunk_sizes[corelet_dim] // 2),
        max(1, chunk_sizes[corelet_dim] - chunk_sizes[corelet_dim] // 2),
    ]
    return stage("core", core_sizes, core_split), stage("chunk", chunk_sizes, chunk_split)


def _bridge_loop_dims(
    input_layout: list[str],
    output_layout: list[str],
) -> list[str]:
    mode = os.environ.get(
        _BRIDGE_LOOP_ORDER_ENV,
        _BRIDGE_LOOP_ORDER_REVERSED_INPUT,
    )
    if mode not in _SUPPORTED_BRIDGE_LOOP_ORDERS:
        raise ValueError(
            f"{_BRIDGE_LOOP_ORDER_ENV}={mode!r} is unsupported; "
            f"choose one of {sorted(_SUPPORTED_BRIDGE_LOOP_ORDERS)}"
        )
    if mode == "input":
        return list(input_layout)
    if mode == "output":
        return list(output_layout)
    if mode == "reversed-output":
        return list(reversed(output_layout))
    return list(reversed(input_layout))


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


_RUNTIME_SEGMENTS_BY_OFFSET = {
    SEGMENT_OFFSETS[0]: "output",
    SEGMENT_OFFSETS[1]: "input",
    SEGMENT_OFFSETS[2]: "model",
    SEGMENT_OFFSETS[3]: "stack",
    SEGMENT_OFFSETS[4]: "heap",
    SEGMENT_OFFSETS[5]: "reserve1",
    SEGMENT_OFFSETS[6]: "reserve2",
}


def _runtime_segment_for_start_address(start_address: Any) -> str | None:
    if isinstance(start_address, Expr) and start_address.free_symbols:
        return None
    try:
        return _RUNTIME_SEGMENTS_BY_OFFSET.get(int(start_address))
    except Exception:  # noqa: BLE001
        return None


def _required_runtime_segment(start_address: Any) -> str:
    segment = _runtime_segment_for_start_address(start_address)
    if segment is None:
        raise ValueError(f"unsupported DDL bridge runtime segment {start_address!r}")
    return segment


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
    # The source compute allocation may already have post-planning coordinates.
    # They describe the original HBM restickify contract and can disagree with
    # the DDL bridge's corelet/layout contract; let DDC recreate them.
    node.pop("coordinates_", None)
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


def _compact_start_map(
    num_cores: int,
    num_corelets: int,
    *,
    base: int = 0,
) -> dict[str, str]:
    return {
        f"[{core}, {corelet}, 0]": str(base)
        for core in range(num_cores)
        for corelet in range(num_corelets)
    }


def _compact_start_payload(
    num_cores: int,
    num_corelets: int = 1,
    *,
    base: int = 0,
) -> dict[str, Any]:
    return {
        "dim_prop_func": [
            {"Map": {}},
            {"Map": {}} if num_corelets > 1 else {"Const": {}},
            {"Const": {}},
        ],
        "dim_prop_attr": [
            {"factor_": num_cores, "label_": "core"},
            {"factor_": num_corelets, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": _compact_start_map(num_cores, num_corelets, base=base),
    }


def _set_compact_start_address(
    node_or_offset: dict[str, Any],
    num_cores: int,
    num_corelets: int = 1,
    *,
    base: int = 0,
    field: str = "startAddressCoreCorelet_",
) -> None:
    node_or_offset[field] = _compact_start_payload(
        num_cores,
        num_corelets,
        base=base,
    )


def _loop_skeleton(
    loop_dims: list[str],
    *,
    prefix: str = "",
    terminal_block: str = "lx_below_schedule",
) -> list[dict[str, Any]]:
    loops: list[dict[str, Any]] = []
    for index, dim in enumerate(loop_dims):
        prev = "" if index == 0 else f"{prefix}loop_ds0_ds1_{loop_dims[index - 1]}"
        next_name = (
            f"{prefix}loop_ds0_ds1_{loop_dims[index + 1]}"
            if index + 1 < len(loop_dims)
            else terminal_block
        )
        loops.append(
            {
                "nodeType_": "loop",
                "name_": f"{prefix}loop_ds0_ds1_{dim}",
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


def _block_node(prev: str, *, name: str = "lx_below_schedule") -> dict[str, Any]:
    return {
        "nodeType_": "block",
        "name_": name,
        "prev_": prev,
        "relevantComps_": {},
        "next_": [],
    }


def _sync_node(
    name: str,
    *,
    prev: str,
    unit: str,
    is_receive: int,
    other: str,
) -> dict[str, Any]:
    return {
        "nodeType_": "sync",
        "name_": name,
        "prev_": prev,
        "relevantComps_": {},
        "units_": [unit],
        "isReceive_": is_receive,
        "isSoft_": 0,
        "implicitSyncRefTransfer_": "",
        "otherEndOfTheSignals_": [other],
    }


def _reference_interslice_schedule_tree(
    input_alloc_node: dict[str, Any],
    input_transfer_node: dict[str, Any],
    output_alloc_node: dict[str, Any],
    output_transfer_node: dict[str, Any],
    *,
    loop_dims: list[str],
) -> list[dict[str, Any]]:
    block_name = "lx_below_schedule"
    loops = _loop_skeleton(
        list(reversed(loop_dims)),
        prefix="prefill_",
        terminal_block=block_name,
    )
    body_names = [
        input_alloc_node["name_"],
        input_transfer_node["name_"],
        output_alloc_node["name_"],
        "sync_send_l3lu_to_lxlu",
        "sync_receive_lxlu_from_l3lu",
        "sync_send_lxlu_to_l3lu",
        "sync_receive_l3lu_from_lxlu",
        block_name,
        "sync_send_lxsu_to_l3su",
        "sync_receive_l3su_from_lxsu",
        "sync_send_l3su_to_lxsu",
        "sync_receive_lxsu_from_l3su",
        output_transfer_node["name_"],
    ]
    loops[-1]["next_"] = body_names
    for node in (input_alloc_node, input_transfer_node, output_alloc_node, output_transfer_node):
        node["prev_"] = loops[-1]["name_"]
    return [
        *loops,
        input_alloc_node,
        input_transfer_node,
        output_alloc_node,
        _sync_node(
            "sync_send_l3lu_to_lxlu",
            prev=loops[-1]["name_"],
            unit="l3lu",
            is_receive=0,
            other="sync_receive_lxlu_from_l3lu",
        ),
        _sync_node(
            "sync_receive_lxlu_from_l3lu",
            prev=loops[-1]["name_"],
            unit="lxlu",
            is_receive=1,
            other="sync_send_l3lu_to_lxlu",
        ),
        _sync_node(
            "sync_send_lxlu_to_l3lu",
            prev=loops[-1]["name_"],
            unit="lxlu",
            is_receive=0,
            other="sync_receive_l3lu_from_lxlu",
        ),
        _sync_node(
            "sync_receive_l3lu_from_lxlu",
            prev=loops[-1]["name_"],
            unit="l3lu",
            is_receive=1,
            other="sync_send_lxlu_to_l3lu",
        ),
        _block_node(loops[-1]["name_"], name=block_name),
        _sync_node(
            "sync_send_lxsu_to_l3su",
            prev=loops[-1]["name_"],
            unit="lxsu",
            is_receive=0,
            other="sync_receive_l3su_from_lxsu",
        ),
        _sync_node(
            "sync_receive_l3su_from_lxsu",
            prev=loops[-1]["name_"],
            unit="l3su",
            is_receive=1,
            other="sync_send_lxsu_to_l3su",
        ),
        _sync_node(
            "sync_send_l3su_to_lxsu",
            prev=loops[-1]["name_"],
            unit="l3su",
            is_receive=0,
            other="sync_receive_lxsu_from_l3su",
        ),
        _sync_node(
            "sync_receive_lxsu_from_l3su",
            prev=loops[-1]["name_"],
            unit="lxsu",
            is_receive=1,
            other="sync_send_l3su_to_lxsu",
        ),
        output_transfer_node,
    ]


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
    segment: str,
    lx_size: int,
    alloc_name: str,
) -> dict[str, Any]:
    lds = copy.deepcopy(template)
    lds.update(
        {
            "ldsIdx_": idx,
            "dsName_": name,
            "dsType_": role,
            "segment_": segment,
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
