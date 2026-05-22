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

"""Standalone restickify data-op SDSC prototype.

Torch-Spyre currently emits restickification as a normal compute-op SDSC named
``ReStickifyOpHBM``. Deeptools also has a separate data-op path under
``datadscs_`` for movement operations such as ``STCDPOpLx`` and
``ReStickifyOpLx``. This module intentionally stays out of production lowering:
it converts an already-built ``SDSCSpec`` into a standalone data-op SuperDsc so
we can test the Deeptools contract before considering graph integration.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from sympy import Expr, Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.op_spec import OpSpec

from .compute_ops import num_bytes
from .superdsc import SDSCArgs, SDSCSpec, parse_op_spec

SUPPORTED_RESTICKIFY_DATA_OPS = frozenset(
    {"STCDPOpLx", "ReStickifyOpLx", "ReStickifyOpWithPTLx", "ReStickifyOpHBM"}
)

_LX_SIZE_BYTES = 2 * 1024 * 1024
_DEEPTOOLS_DATAOP_DIM_LABELS = frozenset(
    {"mb", "out", "in", "x", "y", "i", "j", "ki", "kj"}
)


def generate_restickify_dataop_sdsc(
    idx: int,
    op_spec: OpSpec,
    op_name: str = "STCDPOpLx",
) -> dict[str, Any]:
    """Generate a standalone data-op SuperDsc for an existing OpSpec.

    This wrapper mirrors normal SDSC lowering by first using ``parse_op_spec``.
    Most prototype tests call ``generate_restickify_dataop_sdsc_from_spec``
    directly with a hand-built ``SDSCSpec`` because it is easier to make small
    synthetic movement cases without constructing a full Inductor graph.
    """

    return generate_restickify_dataop_sdsc_from_spec(
        idx,
        parse_op_spec(op_spec),
        op_name=op_name,
    )


def generate_restickify_dataop_sdsc_from_spec(
    idx: int,
    sdsc_spec: SDSCSpec,
    op_name: str = "STCDPOpLx",
    input_work_slices: Mapping[Any, Any] | None = None,
    input_core_to_work_slice: Mapping[str, Mapping[str, int]] | None = None,
    output_work_slices: Mapping[Any, Any] | None = None,
    output_core_to_work_slice: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Generate a standalone SuperDsc with one ``datadscs_`` entry."""

    if op_name not in SUPPORTED_RESTICKIFY_DATA_OPS:
        raise ValueError(
            f"unsupported restickify data op {op_name!r}; "
            f"expected one of {sorted(SUPPORTED_RESTICKIFY_DATA_OPS)}"
        )
    if len(sdsc_spec.args) < 2:
        raise ValueError("restickify data-op SDSC needs at least one input and output")

    dataop_name = f"{idx}_{op_name}_dataop"
    input_arg = sdsc_spec.args[0]
    output_arg = sdsc_spec.args[-1]
    input_lds = _labeled_ds(
        input_arg,
        sdsc_spec,
        lds_name="dataIN_L0",
        pds_name="dataIN",
        start_address=int(input_arg.start_address),
        include_hbm=op_name == "ReStickifyOpHBM",
        work_slices=input_work_slices,
        core_to_work_slice=input_core_to_work_slice,
    )
    output_lds = _labeled_ds(
        output_arg,
        sdsc_spec,
        lds_name="dataOUT_L0",
        pds_name="dataOUT",
        start_address=int(output_arg.start_address),
        include_hbm=op_name == "ReStickifyOpHBM",
        work_slices=output_work_slices,
        core_to_work_slice=output_core_to_work_slice,
    )

    dim_pool = _dim_pool(sdsc_spec)
    dim_sizes = _dim_size_map(sdsc_spec)
    dataop = {
        "coreIdsUsed_": [core for core in range(sdsc_spec.num_cores)],
        "dimPool_": dim_pool,
        "outDimTodimRelation_": [],
        "primaryDs_": [
            {"name_": "dataIN", "dimNames": dim_pool},
            {"name_": "dataOUT", "dimNames": dim_pool},
        ],
        "labeledDs_": [input_lds, output_lds],
        "op": _op_payload(op_name),
    }

    return {
        dataop_name: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": sdsc_spec.num_cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": sdsc_spec.num_cores,
            "unpadN_": {"name_": "unpadn", **{dim: -1 for dim in dim_pool}},
            "N_": {"name_": "n", **dim_sizes},
            "coreIdToDsc_": {},
            "numWkSlicesPerDim_": {},
            "coreIdToWkSlice_": {},
            "opFuncsUsed_": [],
            "ldsShareInfo_": [],
            "prodConsList": {},
            "coreIdToDscSchedule": {},
            "pcfg_": {},
            "target_": "senulator",
            "dscs_": [],
            "datadscs_": [{dataop_name: dataop}],
            "dimToSymbolMappingOpcodeCorrection_": {},
            "inputSymbolsAndTags_": {},
            "symbolDefinitions_": {},
        }
    }


def _op_payload(op_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": op_name}
    if op_name == "ReStickifyOpWithPTLx":
        payload.update(
            {
                "numClToUse": 1,
                "defaultClId": 0,
                "workSplitDim": "null_ptr",
                "cl0ToLxOffsetLU": 0,
                "cl0ToLxOffsetSU": 0,
                "useARF": 1,
                "doInPlace": 0,
            }
        )
    return payload


def combine_dataop_sdscs(
    name: str,
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine standalone single-dataop SuperDsc payloads into one SuperDsc.

    Deeptools represents data movement operations under ``datadscs_``. This
    helper is intentionally small and prototype-oriented: it keeps the first
    payload's global shape metadata and concatenates each payload's data-op
    entry so we can test a composed movement sequence without adding it to
    normal TorchInductor lowering.
    """

    if not payloads:
        raise ValueError("at least one data-op payload is required")

    roots: list[dict[str, Any]] = []
    datadscs: list[dict[str, Any]] = []
    for payload in payloads:
        if len(payload) != 1:
            raise ValueError("each data-op payload must contain exactly one root SDSC")
        root = copy.deepcopy(next(iter(payload.values())))
        roots.append(root)
        datadscs.extend(copy.deepcopy(root.get("datadscs_", [])))

    combined = copy.deepcopy(roots[0])
    combined["numCoresUsed_"] = max(_as_int(root["numCoresUsed_"]) for root in roots)
    combined["dscs_"] = []
    combined["datadscs_"] = datadscs
    if len(datadscs) > 1:
        combined["coreIdToDscSchedule"] = _sequential_dataop_schedule(
            combined["numCoresUsed_"], len(datadscs)
        )
    return {name: combined}


def generate_streaming_ptlx_tile_bridge_sdsc(
    name: str,
    streaming_artifact: Mapping[str, Any],
    *,
    tile_index: int = 0,
) -> dict[str, Any]:
    """Lower one streaming PT-LX descriptor tile into a SuperDSC-shaped object.

    This helper is intentionally static/codegen-only. It proves the descriptor
    can be represented as Deeptools-style data-op JSON with explicit LX
    fragments, but it does not yet claim the generated shape is accepted by DDC
    or safe to run on hardware.
    """

    descriptor = _single_streaming_descriptor(streaming_artifact)
    if descriptor.get("kind") != "streaming_ptlx_restickify_descriptor":
        raise ValueError("expected a streaming_ptlx_restickify_descriptor")
    tiles = descriptor.get("tiles") or []
    if tile_index < 0 or tile_index >= len(tiles):
        raise ValueError(f"tile_index {tile_index} is outside materialized tiles")

    tile = tiles[tile_index]
    buffers = descriptor.get("lx_buffers") or {}
    size = _as_int(descriptor["size"])
    tile_size = _as_int(descriptor["tile_size"])
    source_count = _as_int(descriptor.get("source_core_count", 1))
    dest_count = _as_int(descriptor.get("dest_core_count", 1))
    core_ids = _streaming_tile_core_ids(tile, source_count, dest_count)

    gather_stage, restickify_stage, scatter_stage = tile["stages"]
    bridge_core = _as_int(tile["bridge_core"])
    source_fragments = list(gather_stage.get("fragments") or [])
    dest_fragments = list(scatter_stage.get("fragments") or [])

    gather_output_fragments = [
        _coalesced_tile_fragment(source_fragments, core=bridge_core)
    ]
    scatter_input_fragments = [
        _coalesced_tile_fragment(dest_fragments, core=bridge_core)
    ]
    gather_core_ids = _fragment_core_ids(source_fragments, bridge_core)
    restickify_core_ids = [bridge_core]
    scatter_core_ids = _fragment_core_ids(dest_fragments, bridge_core)

    stage_core_ids = [gather_core_ids, restickify_core_ids, scatter_core_ids]
    datadscs = [
        {
            f"0_STCDPOpLx_gather_tile{tile_index}": _tile_dataop(
                "STCDPOpLx",
                core_ids=gather_core_ids,
                input_layout=("mb_", "out_"),
                input_stick=("out_",),
                output_layout=("mb_", "out_"),
                output_stick=("out_",),
                input_base=_as_int(gather_stage["input_base"]),
                output_base=_as_int(gather_stage["output_base"]),
                input_fragments=source_fragments,
                output_fragments=gather_output_fragments,
            )
        },
        {
            f"1_ReStickifyOpWithPTLx_tile{tile_index}": _tile_dataop(
                "ReStickifyOpWithPTLx",
                core_ids=restickify_core_ids,
                input_layout=("mb_", "out_"),
                input_stick=("out_",),
                output_layout=("out_", "mb_"),
                output_stick=("mb_",),
                input_base=_as_int(restickify_stage["input_base"]),
                output_base=_as_int(restickify_stage["output_base"]),
                input_fragments=gather_output_fragments,
                output_fragments=scatter_input_fragments,
            )
        },
        {
            f"2_STCDPOpLx_scatter_tile{tile_index}": _tile_dataop(
                "STCDPOpLx",
                core_ids=scatter_core_ids,
                input_layout=("out_", "mb_"),
                input_stick=("mb_",),
                output_layout=("out_", "mb_"),
                output_stick=("mb_",),
                input_base=_as_int(scatter_stage["input_base"]),
                output_base=_as_int(scatter_stage["output_base"]),
                input_fragments=scatter_input_fragments,
                output_fragments=dest_fragments,
            )
        },
    ]
    num_cores = max(core_ids) + 1
    return {
        name: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": num_cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": num_cores,
            "unpadN_": {"name_": "unpadn", "mb_": -1, "out_": -1},
            "N_": {"name_": "n", "mb_": size, "out_": size},
            "coreIdToDsc_": {},
            "numWkSlicesPerDim_": {},
            "coreIdToWkSlice_": {},
            "opFuncsUsed_": [
                "STCDPOpLx",
                "ReStickifyOpWithPTLx",
                "STCDPOpLx",
            ],
            "ldsShareInfo_": [],
            "prodConsList": {},
            "coreIdToDscSchedule": _sparse_dataop_schedule(
                num_cores,
                stage_core_ids,
            ),
            "pcfg_": {},
            "target_": "senulator",
            "dscs_": [],
            "datadscs_": datadscs,
            "dimToSymbolMappingOpcodeCorrection_": {},
            "inputSymbolsAndTags_": {},
            "symbolDefinitions_": {},
            "streamingPTLXTile_": {
                "tile_index": int(tile_index),
                "tile_row": _as_int(tile["tile_row"]),
                "tile_col": _as_int(tile["tile_col"]),
                "tile_size": tile_size,
                "bridge_core": bridge_core,
                "status": "static-codegen-only",
                "fallback": "ReStickifyOpHBM",
                "workspace_base": _as_int(buffers.get("tile_workspace_base", 0)),
            },
        }
    }


def generate_ptlx_restickify_bridge_sdsc(
    name: str,
    *,
    size: int,
    num_cores: int,
    mode: str = "stage3b",
    direction: str = "kernel-to-output",
    input_start_address: int = 0,
    output_start_address: int = 1536 * 1024,
    restickify_op_name: str = "ReStickifyOpWithPTLx",
    input_work_slices: Mapping[Any, Any] | None = None,
    input_core_to_work_slice: Mapping[str, Mapping[str, int]] | None = None,
    intermediate_work_slices: Mapping[Any, Any] | None = None,
    intermediate_core_to_work_slice: Mapping[str, Mapping[str, int]] | None = None,
    intermediate_start_address: int | None = None,
    output_work_slices: Mapping[Any, Any] | None = None,
    output_core_to_work_slice: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Generate the PT-aware two-step LX restickify bridge.

    This is the compiler-side form of the standalone probe used during the LX
    restickify study. It emits a combined data-op SDSC with:

        ReStickifyOpWithPTLx -> STCDPOpLx

    The generated bridge is intentionally narrow: it covers the proven
    ``kernel-to-output`` materialization shape used by the high-signal
    producer/restickify/consumer bundle. Callers may patch endpoint PieceInfo
    starts after generation if they want to preserve scheduler-selected LX
    addresses from neighboring SDSCs.
    """

    if restickify_op_name not in {"ReStickifyOpLx", "ReStickifyOpWithPTLx"}:
        raise ValueError(
            f"unsupported PT-LX restickify op {restickify_op_name!r}"
        )
    if direction != "kernel-to-output":
        raise ValueError(
            "PT-LX restickify bridge currently supports only "
            f"direction='kernel-to-output', got {direction!r}"
        )
    if mode not in {"baseline", "stage3b"}:
        raise ValueError(f"unsupported PT-LX bridge mode {mode!r}")

    d0 = Symbol("mb_")
    d1 = Symbol("out_")
    dims = [d0, d1]
    default_input_splits = {d0: 1, d1: num_cores}
    default_input_mapping = _explicit_core_mapping(dims, d1, num_cores)

    # ReStickifyOpWithPTLx handles the stick/layout conversion locally. Keep
    # the intermediate split off the input stick dimension, then use STCDPOpLx
    # to remap ownership to the Stage 3B final split.
    default_intermediate_splits = {d0: num_cores, d1: 1}
    default_intermediate_mapping = _explicit_core_mapping(dims, d0, num_cores)
    final_split_dim = d0 if mode == "baseline" else d1
    default_output_splits = {d0: 1, d1: 1}
    default_output_splits[final_split_dim] = num_cores
    default_output_mapping = _explicit_core_mapping(dims, final_split_dim, num_cores)

    input_splits = input_work_slices or default_input_splits
    input_mapping = input_core_to_work_slice or default_input_mapping
    intermediate_splits = intermediate_work_slices or default_intermediate_splits
    intermediate_mapping = intermediate_core_to_work_slice or default_intermediate_mapping
    final_splits = output_work_slices or default_output_splits
    final_mapping = output_core_to_work_slice or default_output_mapping

    intermediate_start = (
        1024 * 1024
        if intermediate_start_address is None
        else int(intermediate_start_address)
    )
    restickify_spec = _synthetic_ptlx_bridge_spec(
        size,
        num_cores,
        output_split_dim=d0,
        output_stick_dim=d0,
        input_start_address=input_start_address,
        output_start_address=intermediate_start,
    )
    restickify_payload = generate_restickify_dataop_sdsc_from_spec(
        0,
        restickify_spec,
        op_name=restickify_op_name,
        input_work_slices=input_splits,
        input_core_to_work_slice=input_mapping,
        output_work_slices=intermediate_splits,
        output_core_to_work_slice=intermediate_mapping,
    )

    restickified_strides = {d0: 1, d1: size}
    stcdp_spec = _synthetic_ptlx_bridge_spec(
        size,
        num_cores,
        output_split_dim=final_split_dim,
        output_stick_dim=d0,
        input_stick_dim=d0,
        input_start_address=intermediate_start,
        output_start_address=output_start_address,
        input_layout_order=[d1, d0],
        output_layout_order=[d1, d0],
        input_strides=restickified_strides,
        output_strides=restickified_strides,
    )
    stcdp_payload = generate_restickify_dataop_sdsc_from_spec(
        1,
        stcdp_spec,
        op_name="STCDPOpLx",
        input_work_slices=intermediate_splits,
        input_core_to_work_slice=intermediate_mapping,
        output_work_slices=final_splits,
        output_core_to_work_slice=final_mapping,
    )

    return combine_dataop_sdscs(
        name,
        [restickify_payload, stcdp_payload],
    )


def _explicit_core_mapping(
    dims: Sequence[Symbol],
    split_dim: Symbol,
    num_cores: int,
) -> dict[str, dict[str, int]]:
    return {
        str(core): {str(dim): core if dim == split_dim else 0 for dim in dims}
        for core in range(num_cores)
    }


def _synthetic_ptlx_bridge_spec(
    size: int,
    num_cores: int,
    output_split_dim: Symbol,
    output_stick_dim: Symbol,
    *,
    input_stick_dim: Symbol | None = None,
    input_start_address: int = 0,
    output_start_address: int = 1024 * 1024,
    input_layout_order: list[Symbol] | None = None,
    output_layout_order: list[Symbol] | None = None,
    input_strides: dict[Symbol, int] | None = None,
    output_strides: dict[Symbol, int] | None = None,
) -> SDSCSpec:
    d0 = Symbol("mb_")
    d1 = Symbol("out_")
    dims = [d0, d1]
    input_stick_dim = input_stick_dim or d1
    input_layout_order = input_layout_order or [d0, d1]
    output_layout_order = output_layout_order or [d1, d0]
    input_strides = input_strides or {d0: size, d1: 1}
    output_strides = output_strides or {d0: 1, d1: size}
    data_format = DataFormats.SEN169_FP16
    work_slices = {d0: 1, d1: 1}
    work_slices[output_split_dim] = num_cores
    return SDSCSpec(
        opfunc="ReStickifyOpHBM",
        execution_unit="sfp",
        data_format=data_format,
        num_inputs=1,
        iteration_space={d0: size, d1: size},
        num_cores=num_cores,
        work_slices=work_slices,
        core_id_to_work_slice={},
        core_id_to_work_slice_override=_explicit_core_mapping(
            dims,
            output_split_dim,
            num_cores,
        ),
        padding={},
        layouts={
            "INPUT": {
                "dim_order": input_layout_order,
                "stick_dim_order": input_stick_dim,
                "stick_size": 64,
            },
            "OUTPUT": {
                "dim_order": output_layout_order,
                "stick_dim_order": output_stick_dim,
                "stick_size": 64,
            },
        },
        args=[
            SDSCArgs(
                layout="INPUT",
                data_format=data_format,
                scales={d0: 1, d1: 1},
                strides=input_strides,
                offsets={},
                max_dim_sizes={d0: -1, d1: -1},
                allocation={"lx": 0},
                start_address=input_start_address,
                backGap={},
            ),
            SDSCArgs(
                layout="OUTPUT",
                data_format=data_format,
                scales={d0: 1, d1: 1},
                strides=output_strides,
                offsets={},
                max_dim_sizes={d0: -1, d1: -1},
                allocation={"lx": 0},
                start_address=output_start_address,
                backGap={},
            ),
        ],
        constants={},
        coordinate_masking={},
    )


def _sequential_dataop_schedule(
    num_cores: int,
    num_dataops: int,
) -> dict[str, list[list[int]]]:
    return {
        str(core_id): [
            [
                dataop_idx,
                -1,
                1 if dataop_idx > 0 else 0,
                1 if dataop_idx < num_dataops - 1 else 0,
            ]
            for dataop_idx in range(num_dataops)
        ]
        for core_id in range(num_cores)
    }


def _sparse_dataop_schedule(
    num_cores: int,
    stage_core_ids: Sequence[Sequence[int]],
) -> dict[str, list[list[int]]]:
    schedule: dict[str, list[list[int]]] = {}
    for core_id in range(num_cores):
        local_stages = [
            stage_idx
            for stage_idx, core_ids in enumerate(stage_core_ids)
            if core_id in {int(core) for core in core_ids}
        ]
        schedule[str(core_id)] = [
            [
                stage_idx,
                -1,
                1 if local_idx > 0 else 0,
                1 if local_idx < len(local_stages) - 1 else 0,
            ]
            for local_idx, stage_idx in enumerate(local_stages)
        ]
    return schedule


def _labeled_ds(
    arg: SDSCArgs,
    sdsc_spec: SDSCSpec,
    *,
    lds_name: str,
    pds_name: str,
    start_address: int,
    include_hbm: bool,
    work_slices: Mapping[Any, Any] | None,
    core_to_work_slice: Mapping[str, Mapping[str, int]] | None,
) -> dict[str, Any]:
    layout_info = sdsc_spec.layouts[arg.layout]
    layout_dims = _layout_dim_names(layout_info)
    layout_sizes = _layout_sizes(layout_dims, sdsc_spec)
    stick_sizes = _stick_size_map(layout_info)

    return {
        "ldsName_": lds_name,
        "pdsName_": pds_name,
        "wordLength": num_bytes(arg.data_format),
        "dataformat": arg.data_format.name,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": layout_dims,
        "stickDimOrder_": list(stick_sizes.keys()),
        "dimToLayoutSize_": layout_sizes,
        "dimToStickSize_": stick_sizes,
        "validGap_": _valid_gap(layout_sizes),
        "totElements": -1,
        "PieceInfo": _piece_info(
            arg,
            sdsc_spec,
            layout_dims=layout_dims,
            start_address=start_address,
            include_hbm=include_hbm,
            work_slices=work_slices,
            core_to_work_slice=core_to_work_slice,
        ),
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": _LX_SIZE_BYTES,
        "lxStartAddress_": {},
    }


def _piece_info(
    arg: SDSCArgs,
    sdsc_spec: SDSCSpec,
    *,
    layout_dims: Sequence[str],
    start_address: int,
    include_hbm: bool,
    work_slices: Mapping[Any, Any] | None,
    core_to_work_slice: Mapping[str, Mapping[str, int]] | None,
) -> list[dict[str, Any]]:
    core_to_slice = (
        _normalize_core_to_work_slice(core_to_work_slice)
        if core_to_work_slice is not None
        else _core_to_work_slice(sdsc_spec)
    )
    size_by_name = _dim_size_map(sdsc_spec)
    split_by_name = _normalize_work_slices(work_slices or sdsc_spec.work_slices)

    pieces: list[dict[str, Any]] = []
    for core_id in range(sdsc_spec.num_cores):
        wk_slice = core_to_slice[str(core_id)]
        starts: dict[str, int] = {}
        sizes: dict[str, int] = {}
        for dim in layout_dims:
            dim_size = size_by_name[dim]
            split = split_by_name.get(dim, 1)
            if dim_size % split != 0:
                raise ValueError(
                    f"dimension {dim} size {dim_size} is not divisible by split {split}"
                )
            chunk = dim_size // split
            starts[dim] = int(wk_slice.get(dim, 0)) * chunk
            sizes[dim] = chunk if split > 1 else dim_size

        placement = [{"type": "lx", "memId": [core_id], "startAddr": [start_address]}]
        if include_hbm:
            placement.append(
                {
                    "type": "hbm",
                    "memId": [-1],
                    "startAddr": [start_address + _hbm_piece_offset(starts, arg)],
                }
            )
        pieces.append(
            {
                "key_": f"p{core_id + 1}",
                "dimToStartCordinate": starts,
                "dimToSize_": sizes,
                "validGap_": _valid_gap(sizes),
                "PlacementInfo": placement,
            }
        )
    return pieces


def _hbm_piece_offset(starts: Mapping[str, int], arg: SDSCArgs) -> int:
    offset = 0
    for dim, stride in arg.strides.items():
        offset += starts.get(_dataop_dim_name(str(dim)), 0) * _as_int(stride)
    return offset * num_bytes(arg.data_format)


def _core_to_work_slice(sdsc_spec: SDSCSpec) -> dict[str, dict[str, int]]:
    if sdsc_spec.core_id_to_work_slice_override is not None:
        return {
            str(core_id): {
                _dataop_dim_name(str(dim)): int(value)
                for dim, value in per_dim.items()
            }
            for core_id, per_dim in sdsc_spec.core_id_to_work_slice_override.items()
        }

    result: dict[str, dict[str, int]] = {}
    core_id_sym = Symbol("core_id")
    for core_id in range(sdsc_spec.num_cores):
        result[str(core_id)] = {
            _dataop_dim_name(str(dim)): int(expr.subs({core_id_sym: core_id}))
            if isinstance(expr, Expr)
            else int(expr)
            for dim, expr in sdsc_spec.core_id_to_work_slice.items()
        }
    return result


def _normalize_core_to_work_slice(
    mapping: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        str(core_id): {
            _dataop_dim_name(str(dim)): int(value)
            for dim, value in per_dim.items()
        }
        for core_id, per_dim in mapping.items()
    }


def _normalize_work_slices(work_slices: Mapping[Any, Any]) -> dict[str, int]:
    return {
        _dataop_dim_name(str(dim)): _as_int(split)
        for dim, split in work_slices.items()
    }


def _dim_pool(sdsc_spec: SDSCSpec) -> list[str]:
    dims: set[str] = set()
    for layout_info in sdsc_spec.layouts.values():
        dims.update(_layout_dim_names(layout_info))
        dims.update(_stick_dim_names(layout_info))
    return sorted(_dataop_dim_name(dim) for dim in dims)


def _dim_size_map(sdsc_spec: SDSCSpec) -> dict[str, int]:
    return {
        _dataop_dim_name(str(dim)): _as_int(size)
        for dim, size in sdsc_spec.iteration_space.items()
    }


def _layout_sizes(layout_dims: Sequence[str], sdsc_spec: SDSCSpec) -> dict[str, int]:
    sizes = _dim_size_map(sdsc_spec)
    return {dim: sizes[dim] for dim in layout_dims}


def _valid_gap(sizes: Mapping[str, int]) -> dict[str, list[list[int]]]:
    return {dim: [[size, 0]] for dim, size in sizes.items()}


def _layout_dim_names(layout_info: Mapping[str, Any]) -> list[str]:
    return [_dataop_dim_name(str(dim)) for dim in layout_info["dim_order"]]


def _stick_dim_names(layout_info: Mapping[str, Any]) -> list[str]:
    raw = layout_info.get("stick_dim_order")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [_dataop_dim_name(str(dim)) for dim in raw]
    return [_dataop_dim_name(str(raw))]


def _dataop_dim_name(dim: str) -> str:
    if dim.endswith("_"):
        return dim
    if dim in _DEEPTOOLS_DATAOP_DIM_LABELS:
        return f"{dim}_"
    return dim


def _stick_size_map(layout_info: Mapping[str, Any]) -> dict[str, int]:
    stick_dims = _stick_dim_names(layout_info)
    if not stick_dims:
        return {}
    raw_size = layout_info.get("stick_size", [])
    if isinstance(raw_size, (list, tuple)):
        sizes = [_as_int(size) for size in raw_size]
    else:
        sizes = [_as_int(raw_size)]
    if len(sizes) == 1 and len(stick_dims) > 1:
        sizes = sizes * len(stick_dims)
    if len(sizes) != len(stick_dims):
        raise ValueError(
            f"stick size count {len(sizes)} does not match stick dims {stick_dims}"
        )
    return dict(zip(stick_dims, sizes))


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except TypeError as exc:
        raise ValueError(
            f"data-op prototype requires concrete integer value, got {value}"
        ) from exc


def _single_streaming_descriptor(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if len(artifact) != 1:
        raise ValueError("streaming artifact must contain exactly one descriptor")
    return next(iter(artifact.values()))


def _streaming_tile_core_ids(
    tile: Mapping[str, Any],
    source_count: int,
    dest_count: int,
) -> list[int]:
    max_core = max(
        [int(source_count) - 1, int(dest_count) - 1]
        + [_as_int(core) for core in tile.get("source_cores", []) or []]
        + [_as_int(core) for core in tile.get("dest_cores", []) or []]
        + [_as_int(tile.get("bridge_core", 0))]
    )
    return list(range(max_core + 1))


def _fragment_core_ids(
    fragments: Sequence[Mapping[str, Any]],
    bridge_core: int,
) -> list[int]:
    return sorted(
        {_as_int(fragment["core"]) for fragment in fragments} | {int(bridge_core)}
    )


def _tile_dataop(
    op_name: str,
    *,
    core_ids: Sequence[int],
    input_layout: Sequence[str],
    input_stick: Sequence[str],
    output_layout: Sequence[str],
    output_stick: Sequence[str],
    input_base: int,
    output_base: int,
    input_fragments: Sequence[Mapping[str, Any]],
    output_fragments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "coreIdsUsed_": [int(core) for core in core_ids],
        "dimPool_": ["mb_", "out_"],
        "outDimTodimRelation_": [],
        "primaryDs_": [
            {"name_": "dataIN", "dimNames": ["mb_", "out_"]},
            {"name_": "dataOUT", "dimNames": ["mb_", "out_"]},
        ],
        "labeledDs_": [
            _tile_labeled_ds(
                "dataIN_L0",
                "dataIN",
                layout=input_layout,
                stick=input_stick,
                base=input_base,
                fragments=input_fragments,
            ),
            _tile_labeled_ds(
                "dataOUT_L0",
                "dataOUT",
                layout=output_layout,
                stick=output_stick,
                base=output_base,
                fragments=output_fragments,
            ),
        ],
        "op": _op_payload(op_name),
    }


def _tile_labeled_ds(
    lds_name: str,
    pds_name: str,
    *,
    layout: Sequence[str],
    stick: Sequence[str],
    base: int,
    fragments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    layout_dims = [str(dim) for dim in layout]
    stick_dims = [str(dim) for dim in stick]
    layout_sizes = _tile_layout_size(fragments)
    return {
        "ldsName_": lds_name,
        "pdsName_": pds_name,
        "wordLength": num_bytes(DataFormats.SEN169_FP16),
        "dataformat": DataFormats.SEN169_FP16.name,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": layout_dims,
        "stickDimOrder_": stick_dims,
        "dimToLayoutSize_": layout_sizes,
        "dimToStickSize_": {dim: 64 for dim in stick_dims},
        "validGap_": _valid_gap(layout_sizes),
        "totElements": -1,
        "PieceInfo": [
            _tile_piece_info(fragment, base=base, key=f"p{idx + 1}")
            for idx, fragment in enumerate(fragments)
        ],
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": _LX_SIZE_BYTES,
        "lxStartAddress_": {},
    }


def _tile_layout_size(fragments: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    mb_end = max((_as_int(fragment["row_end"]) for fragment in fragments), default=0)
    out_end = max((_as_int(fragment["col_end"]) for fragment in fragments), default=0)
    return {"mb_": mb_end, "out_": out_end}


def _tile_piece_info(
    fragment: Mapping[str, Any],
    *,
    base: int,
    key: str,
) -> dict[str, Any]:
    sizes = {
        "mb_": _as_int(fragment["row_end"]) - _as_int(fragment["row_start"]),
        "out_": _as_int(fragment["col_end"]) - _as_int(fragment["col_start"]),
    }
    return {
        "key_": key,
        "dimToStartCordinate": {
            "mb_": _as_int(fragment["row_start"]),
            "out_": _as_int(fragment["col_start"]),
        },
        "dimToSize_": sizes,
        "validGap_": _valid_gap(sizes),
        "PlacementInfo": [
            {
                "type": "lx",
                "memId": [_as_int(fragment["core"])],
                "startAddr": [int(base)],
            }
        ],
    }


def _coalesced_tile_fragment(
    fragments: Sequence[Mapping[str, Any]],
    *,
    core: int,
) -> dict[str, int]:
    if not fragments:
        raise ValueError("cannot coalesce an empty fragment list")
    row_start = min(_as_int(fragment["row_start"]) for fragment in fragments)
    row_end = max(_as_int(fragment["row_end"]) for fragment in fragments)
    col_start = min(_as_int(fragment["col_start"]) for fragment in fragments)
    col_end = max(_as_int(fragment["col_end"]) for fragment in fragments)
    return {
        "core": int(core),
        "row_start": row_start,
        "row_end": row_end,
        "col_start": col_start,
        "col_end": col_end,
        "bytes": (row_end - row_start) * (col_end - col_start) * 2,
        "hops": 0,
    }
