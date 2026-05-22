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
import os
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
_COMPACT_TILE_WORKSPACE_ENV = "SPYRE_RESTICKIFY_PTLX_COMPACT_TILE_WORKSPACE"
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

    internal_fragment = (
        _compact_tile_fragment
        if _compact_tile_workspace_enabled()
        else _coalesced_tile_fragment
    )
    gather_output_fragments = [internal_fragment(source_fragments, core=bridge_core)]
    scatter_input_fragments = [internal_fragment(dest_fragments, core=bridge_core)]
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


def generate_streaming_ptlx_native_tile_bridge_sdsc(
    name: str,
    streaming_artifact: Mapping[str, Any],
    *,
    tile_index: int = 0,
) -> dict[str, Any]:
    """Lower one tile using the native 4D PT-LX local transform contract.

    This is the production-shaped successor to
    ``generate_streaming_ptlx_tile_bridge_sdsc``.  The older helper represents
    the gather, restickify, and scatter phases as 2D ``mb_/out_`` data ops; that
    is enough to prove endpoint plumbing, but it is not a semantic certificate
    for the restickify transform.  This helper keeps the same three phases but
    expresses the tile workspace in Deeptools' native
    ``j_, i_, out_, mb_`` PT-LX shape:

        STCDPOpLx             gather source fragments into tile-local coords
        ReStickifyOpWithPTLx  switch stick dimension out_ -> j_
        STCDPOpLx             scatter tile-local coords to destination fragments

    The helper remains default-off and codegen-only until the surrounding
    producer/consumer LX endpoint contract is value-checked on hardware.
    """

    descriptor = _single_streaming_descriptor(streaming_artifact)
    if descriptor.get("kind") != "streaming_ptlx_restickify_descriptor":
        raise ValueError("expected a streaming_ptlx_restickify_descriptor")
    tiles = descriptor.get("tiles") or []
    if tile_index < 0 or tile_index >= len(tiles):
        raise ValueError(f"tile_index {tile_index} is outside materialized tiles")

    tile = tiles[tile_index]
    tile_size = _as_int(descriptor["tile_size"])
    tile_row_start = _as_int(tile["tile_row"]) * tile_size
    tile_col_start = _as_int(tile["tile_col"]) * tile_size
    gather_stage, restickify_stage, scatter_stage = tile["stages"]
    bridge_core = _as_int(tile["bridge_core"])
    source_fragments = list(gather_stage.get("fragments") or [])
    dest_fragments = list(scatter_stage.get("fragments") or [])
    tile_rows, tile_cols = _native_tile_shape(source_fragments, dest_fragments)
    if tile_rows % 64 != 0 or tile_cols % 64 != 0:
        raise ValueError(
            "native PT-LX tile bridge currently requires 64-aligned tile fragments"
        )

    gather_input_fragments = [
        _native_fragment(fragment, tile_row_start, tile_col_start)
        for fragment in source_fragments
    ]
    gather_output_fragments = [
        _native_whole_tile_fragment(
            core=bridge_core,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )
    ]
    scatter_input_fragments = [
        _native_whole_tile_fragment(
            core=bridge_core,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )
    ]
    scatter_output_fragments = [
        _native_fragment(fragment, tile_row_start, tile_col_start)
        for fragment in dest_fragments
    ]

    gather_core_ids = _fragment_core_ids(source_fragments, bridge_core)
    restickify_core_ids = [bridge_core]
    scatter_core_ids = _fragment_core_ids(dest_fragments, bridge_core)
    stage_core_ids = [gather_core_ids, restickify_core_ids, scatter_core_ids]
    num_cores = max(
        _streaming_tile_core_ids(
            tile,
            _as_int(descriptor.get("source_core_count", 1)),
            _as_int(descriptor.get("dest_core_count", 1)),
        )
    ) + 1
    datadscs = [
        {
            f"0_STCDPOpLx_native_gather_tile{tile_index}": _native_tile_dataop(
                "STCDPOpLx",
                core_ids=gather_core_ids,
                input_stick=("out_",),
                output_stick=("out_",),
                input_base=_as_int(gather_stage["input_base"]),
                output_base=_as_int(gather_stage["output_base"]),
                input_fragments=gather_input_fragments,
                output_fragments=gather_output_fragments,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
            )
        },
        {
            f"1_ReStickifyOpWithPTLx_native_tile{tile_index}": _native_tile_dataop(
                "ReStickifyOpWithPTLx",
                core_ids=restickify_core_ids,
                input_stick=("out_",),
                output_stick=("j_",),
                input_base=_as_int(restickify_stage["input_base"]),
                output_base=_as_int(restickify_stage["output_base"]),
                input_fragments=gather_output_fragments,
                output_fragments=scatter_input_fragments,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
            )
        },
        {
            f"2_STCDPOpLx_native_scatter_tile{tile_index}": _native_tile_dataop(
                "STCDPOpLx",
                core_ids=scatter_core_ids,
                input_stick=("j_",),
                output_stick=("j_",),
                input_base=_as_int(scatter_stage["input_base"]),
                output_base=_as_int(scatter_stage["output_base"]),
                input_fragments=scatter_input_fragments,
                output_fragments=scatter_output_fragments,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
            )
        },
    ]
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
            "unpadN_": {"name_": "unpadn", **_native_tile_unpad()},
            "N_": {"name_": "n", **_native_tile_sizes(tile_rows, tile_cols)},
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
            "streamingPTLXNativeTile_": {
                "tile_index": int(tile_index),
                "tile_row": _as_int(tile["tile_row"]),
                "tile_col": _as_int(tile["tile_col"]),
                "tile_rows": tile_rows,
                "tile_cols": tile_cols,
                "bridge_core": bridge_core,
                "status": "static-codegen-only",
                "semantic_transform_certified": False,
                "fallback": "ReStickifyOpHBM",
            },
        }
    }


def generate_streaming_ptlx_direct_tile_bridge_sdsc(
    name: str,
    streaming_artifact: Mapping[str, Any],
    *,
    tile_index: int = 0,
) -> dict[str, Any]:
    """Lower one tile as a direct 2D PT-LX restickify data op.

    This diagnostic path avoids using ``STCDPOpLx`` as the final coordinate
    remapper. The source fragments are the producer-owned ``mb_/out_`` tile
    pieces, and the output fragments are the consumer-owned ``out_/mb_`` tile
    pieces. If producer fragments are smaller than one PT tile, a same-layout
    gather coalesces them before ``ReStickifyOpWithPTLx`` writes directly to
    the consumer endpoint.
    """

    descriptor = _single_streaming_descriptor(streaming_artifact)
    if descriptor.get("kind") != "streaming_ptlx_restickify_descriptor":
        raise ValueError("expected a streaming_ptlx_restickify_descriptor")
    tiles = descriptor.get("tiles") or []
    if tile_index < 0 or tile_index >= len(tiles):
        raise ValueError(f"tile_index {tile_index} is outside materialized tiles")

    tile = tiles[tile_index]
    gather_stage, _restickify_stage, scatter_stage = tile["stages"]
    source_fragments = list(gather_stage.get("fragments") or [])
    dest_fragments = list(scatter_stage.get("fragments") or [])
    if not source_fragments or not dest_fragments:
        raise ValueError("direct PT-LX tile bridge needs source and dest fragments")

    bridge_core = _as_int(tile["bridge_core"])
    needs_gather = any(
        _as_int(fragment["row_end"]) - _as_int(fragment["row_start"]) < 64
        or _as_int(fragment["col_end"]) - _as_int(fragment["col_start"]) < 64
        for fragment in source_fragments
    )
    restickify_input_fragments = source_fragments
    restickify_input_base = _as_int(gather_stage["input_base"])
    datadscs = []
    stage_core_ids = []
    if needs_gather:
        gathered_fragment = _coalesced_tile_fragment(
            source_fragments,
            core=bridge_core,
        )
        gather_core_ids = _fragment_core_ids(source_fragments, bridge_core)
        datadscs.append(
            {
                f"0_STCDPOpLx_gather_direct_tile{tile_index}": _tile_dataop(
                    "STCDPOpLx",
                    core_ids=gather_core_ids,
                    input_layout=("mb_", "out_"),
                    input_stick=("out_",),
                    output_layout=("mb_", "out_"),
                    output_stick=("out_",),
                    input_base=_as_int(gather_stage["input_base"]),
                    output_base=_as_int(gather_stage["output_base"]),
                    input_fragments=source_fragments,
                    output_fragments=[gathered_fragment],
                )
            }
        )
        stage_core_ids.append(gather_core_ids)
        restickify_input_fragments = [gathered_fragment]
        restickify_input_base = _as_int(gather_stage["output_base"])

    restickify_core_ids = _fragment_core_ids(
        [*restickify_input_fragments, *dest_fragments],
        bridge_core,
    )
    restickify_idx = len(datadscs)
    datadscs.append(
        {
            f"{restickify_idx}_ReStickifyOpWithPTLx_direct_tile{tile_index}": _tile_dataop(
                "ReStickifyOpWithPTLx",
                core_ids=restickify_core_ids,
                input_layout=("mb_", "out_"),
                input_stick=("out_",),
                output_layout=("out_", "mb_"),
                output_stick=("mb_",),
                input_base=restickify_input_base,
                output_base=_as_int(scatter_stage["output_base"]),
                input_fragments=restickify_input_fragments,
                output_fragments=dest_fragments,
            )
        }
    )
    stage_core_ids.append(restickify_core_ids)

    num_cores = max(
        _streaming_tile_core_ids(
            tile,
            _as_int(descriptor.get("source_core_count", 1)),
            _as_int(descriptor.get("dest_core_count", 1)),
        )
    ) + 1
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
            "N_": {
                "name_": "n",
                "mb_": _as_int(descriptor["size"]),
                "out_": _as_int(descriptor["size"]),
            },
            "coreIdToDsc_": {},
            "numWkSlicesPerDim_": {},
            "coreIdToWkSlice_": {},
            "opFuncsUsed_": [
                next(iter(datadsc.values()))["op"]["name"] for datadsc in datadscs
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
            "streamingPTLXDirectTile_": {
                "tile_index": int(tile_index),
                "tile_row": _as_int(tile["tile_row"]),
                "tile_col": _as_int(tile["tile_col"]),
                "source_fragment_count": len(source_fragments),
                "dest_fragment_count": len(dest_fragments),
                "gather_stage": needs_gather,
                "status": "static-codegen-only",
                "semantic_transform_certified": False,
                "fallback": "ReStickifyOpHBM",
            },
        }
    }


def generate_ptlx_local_tile_restickify_sdsc(
    name: str,
    *,
    core_id: int,
    input_base: int,
    output_base: int,
    tile_rows: int = 64,
    tile_cols: int = 64,
    i_size: int = 1,
    mb_size: int = 1,
) -> dict[str, Any]:
    """Generate a native PT-LX local-tile restickify data-op payload.

    Deeptools' ``ReStickifyOpWithPTLx`` examples model the PT transform using
    four dimensions (``j_, i_, out_, mb_``) and switch the stick dimension from
    ``out_`` to ``j_``.  This helper captures that local transform contract for
    one bridge core.  It is intentionally only the middle phase of the
    production streaming bridge; same-stick gather/scatter must surround it.
    """

    if tile_rows <= 0 or tile_cols <= 0:
        raise ValueError("tile rows and columns must be positive")
    if tile_rows % 64 != 0 or tile_cols % 64 != 0:
        raise ValueError("PT-LX local tile dimensions must be 64-aligned")
    if i_size <= 0 or mb_size <= 0:
        raise ValueError("i_size and mb_size must be positive")

    dims = ["j_", "i_", "out_", "mb_"]
    sizes = {
        "j_": int(tile_rows),
        "i_": int(i_size),
        "out_": int(tile_cols),
        "mb_": int(mb_size),
    }
    dataop_name = f"0_ReStickifyOpWithPTLx_local_tile_core{int(core_id)}"
    dataop = {
        "coreIdsUsed_": [int(core_id)],
        "dimPool_": dims,
        "outDimTodimRelation_": [],
        "primaryDs_": [
            {"name_": "dataIN", "dimNames": dims},
            {"name_": "dataOUT", "dimNames": dims},
        ],
        "labeledDs_": [
            _local_ptlx_labeled_ds(
                "dataIN_L0",
                "dataIN",
                stick=("out_",),
                sizes=sizes,
                core_id=int(core_id),
                base=int(input_base),
            ),
            _local_ptlx_labeled_ds(
                "dataOUT_L0",
                "dataOUT",
                stick=("j_",),
                sizes=sizes,
                core_id=int(core_id),
                base=int(output_base),
            ),
        ],
        "op": _op_payload("ReStickifyOpWithPTLx"),
    }

    return {
        name: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": 1, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": 1,
            "unpadN_": {"name_": "unpadn", **{dim: -1 for dim in dims}},
            "N_": {"name_": "n", **sizes},
            "coreIdToDsc_": {},
            "numWkSlicesPerDim_": {},
            "coreIdToWkSlice_": {},
            "opFuncsUsed_": ["ReStickifyOpWithPTLx"],
            "ldsShareInfo_": [],
            "prodConsList": {},
            "coreIdToDscSchedule": {
                str(int(core_id)): [[0, -1, 0, 0]],
            },
            "pcfg_": {},
            "target_": "senulator",
            "dscs_": [],
            "datadscs_": [{dataop_name: dataop}],
            "dimToSymbolMappingOpcodeCorrection_": {},
            "inputSymbolsAndTags_": {},
            "symbolDefinitions_": {},
            "streamingPTLXLocalTile_": {
                "semantic_transform_certified": True,
                "tile_rows": int(tile_rows),
                "tile_cols": int(tile_cols),
                "core_id": int(core_id),
                "input_stick_dim": "out_",
                "output_stick_dim": "j_",
            },
        }
    }


def generate_streaming_ptlx_full_bridge_sdsc(
    name: str,
    streaming_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine every materialized streaming tile into one static payload."""

    descriptor = _single_streaming_descriptor(streaming_artifact)
    tiles = descriptor.get("tiles") or []
    if not tiles:
        raise ValueError("streaming descriptor has no materialized tiles")

    striped_payload = _generate_streaming_ptlx_row_stripe_bridge_sdsc(
        name,
        descriptor,
    )
    if striped_payload is not None:
        return striped_payload

    roots: list[dict[str, Any]] = []
    all_datadscs: list[dict[str, Any]] = []
    combined_schedule: dict[str, list[list[int]]] = {}
    offset = 0
    for tile_index in range(len(tiles)):
        tile_payload = generate_streaming_ptlx_tile_bridge_sdsc(
            f"{name}_tile{tile_index}",
            streaming_artifact,
            tile_index=tile_index,
        )
        tile_root = copy.deepcopy(next(iter(tile_payload.values())))
        roots.append(tile_root)
        tile_datadscs = tile_root.get("datadscs_", []) or []
        all_datadscs.extend(tile_datadscs)
        for core_id, steps in (tile_root.get("coreIdToDscSchedule") or {}).items():
            core_steps = combined_schedule.setdefault(str(core_id), [])
            for step in steps:
                adjusted = list(step)
                adjusted[0] = int(adjusted[0]) + offset
                core_steps.append(adjusted)
        offset += len(tile_datadscs)

    combined = copy.deepcopy(roots[0])
    combined["datadscs_"] = all_datadscs
    combined["numCoresUsed_"] = max(_as_int(root["numCoresUsed_"]) for root in roots)
    for core_id in range(combined["numCoresUsed_"]):
        combined_schedule.setdefault(str(core_id), [])
    combined["coreIdToDscSchedule"] = _with_local_schedule_dependencies(
        combined_schedule
    )
    combined["opFuncsUsed_"] = [
        next(iter(datadsc.values()))["op"]["name"] for datadsc in all_datadscs
    ]
    combined["streamingPTLXTile_"] = {}
    combined["streamingPTLXFull_"] = {
        "status": "static-codegen-only",
        "tile_count": len(tiles),
        "datadsc_count": len(all_datadscs),
        "fallback": "ReStickifyOpHBM",
    }
    return {name: combined}


def generate_streaming_ptlx_native_full_bridge_sdsc(
    name: str,
    streaming_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine every materialized native PT-LX tile bridge into one payload."""

    descriptor = _single_streaming_descriptor(streaming_artifact)
    tiles = descriptor.get("tiles") or []
    if not tiles:
        raise ValueError("streaming descriptor has no materialized tiles")

    roots: list[dict[str, Any]] = []
    all_datadscs: list[dict[str, Any]] = []
    combined_schedule: dict[str, list[list[int]]] = {}
    offset = 0
    for tile_index in range(len(tiles)):
        tile_payload = generate_streaming_ptlx_native_tile_bridge_sdsc(
            f"{name}_native_tile{tile_index}",
            streaming_artifact,
            tile_index=tile_index,
        )
        tile_root = copy.deepcopy(next(iter(tile_payload.values())))
        roots.append(tile_root)
        tile_datadscs = tile_root.get("datadscs_", []) or []
        all_datadscs.extend(tile_datadscs)
        for core_id, steps in (tile_root.get("coreIdToDscSchedule") or {}).items():
            core_steps = combined_schedule.setdefault(str(core_id), [])
            for step in steps:
                adjusted = list(step)
                adjusted[0] = int(adjusted[0]) + offset
                core_steps.append(adjusted)
        offset += len(tile_datadscs)

    combined = copy.deepcopy(roots[0])
    combined["datadscs_"] = all_datadscs
    combined["numCoresUsed_"] = max(_as_int(root["numCoresUsed_"]) for root in roots)
    for core_id in range(combined["numCoresUsed_"]):
        combined_schedule.setdefault(str(core_id), [])
    combined["coreIdToDscSchedule"] = _with_local_schedule_dependencies(
        combined_schedule
    )
    combined["opFuncsUsed_"] = [
        next(iter(datadsc.values()))["op"]["name"] for datadsc in all_datadscs
    ]
    combined["streamingPTLXNativeTile_"] = {}
    combined["streamingPTLXFull_"] = {
        "status": "static-codegen-only",
        "coalescing": "native-64x64-tiles",
        "tile_count": len(tiles),
        "logical_tile_count": _as_int(descriptor.get("total_tiles", len(tiles))),
        "datadsc_count": len(all_datadscs),
        "native_local_transform_contract": True,
        "semantic_transform_certified": False,
        "fallback": "ReStickifyOpHBM",
    }
    return {name: combined}


def generate_streaming_ptlx_direct_full_bridge_sdsc(
    name: str,
    streaming_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine every direct 2D PT-LX restickify tile into one payload."""

    descriptor = _single_streaming_descriptor(streaming_artifact)
    tiles = descriptor.get("tiles") or []
    if not tiles:
        raise ValueError("streaming descriptor has no materialized tiles")

    roots: list[dict[str, Any]] = []
    all_datadscs: list[dict[str, Any]] = []
    combined_schedule: dict[str, list[list[int]]] = {}
    offset = 0
    for tile_index in range(len(tiles)):
        tile_payload = generate_streaming_ptlx_direct_tile_bridge_sdsc(
            f"{name}_direct_tile{tile_index}",
            streaming_artifact,
            tile_index=tile_index,
        )
        tile_root = copy.deepcopy(next(iter(tile_payload.values())))
        roots.append(tile_root)
        tile_datadscs = tile_root.get("datadscs_", []) or []
        all_datadscs.extend(tile_datadscs)
        for core_id, steps in (tile_root.get("coreIdToDscSchedule") or {}).items():
            core_steps = combined_schedule.setdefault(str(core_id), [])
            for step in steps:
                adjusted = list(step)
                adjusted[0] = int(adjusted[0]) + offset
                core_steps.append(adjusted)
        offset += len(tile_datadscs)

    combined = copy.deepcopy(roots[0])
    combined["datadscs_"] = all_datadscs
    combined["numCoresUsed_"] = max(_as_int(root["numCoresUsed_"]) for root in roots)
    for core_id in range(combined["numCoresUsed_"]):
        combined_schedule.setdefault(str(core_id), [])
    combined["coreIdToDscSchedule"] = _with_local_schedule_dependencies(
        combined_schedule
    )
    combined["opFuncsUsed_"] = [
        next(iter(datadsc.values()))["op"]["name"] for datadsc in all_datadscs
    ]
    combined["streamingPTLXDirectTile_"] = {}
    combined["streamingPTLXFull_"] = {
        "status": "static-codegen-only",
        "coalescing": "direct-64x64-tiles",
        "tile_count": len(tiles),
        "logical_tile_count": _as_int(descriptor.get("total_tiles", len(tiles))),
        "datadsc_count": len(all_datadscs),
        "direct_restickify_contract": True,
        "semantic_transform_certified": False,
        "fallback": "ReStickifyOpHBM",
    }
    return {name: combined}


def _generate_streaming_ptlx_row_stripe_bridge_sdsc(
    name: str,
    descriptor: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Coalesce simple one-owner tiles into row-stripe bridge data ops.

    The 2048 high-signal shape decomposes into 1024 64x64 logical tiles, but
    every tile has exactly one producer owner and one consumer owner. Emitting a
    gather/restickify/scatter triplet per tile is correct but too instruction
    heavy for DCC. For this simple case, gather a whole destination row stripe
    into the bridge core and have ``ReStickifyOpWithPTLx`` write directly to the
    consumer LX endpoint. More fragmented shapes keep the conservative per-tile
    lowering above.
    """

    tiles = list(descriptor.get("tiles") or [])
    if len(tiles) != _as_int(descriptor.get("total_tiles", len(tiles))):
        return None
    if not all(_simple_one_owner_tile(tile) for tile in tiles):
        return None

    groups = _row_stripe_groups(tiles)
    if groups is None:
        return None

    skeleton = generate_streaming_ptlx_tile_bridge_sdsc(
        f"{name}_row_stripe_skeleton",
        {name: dict(descriptor)},
        tile_index=0,
    )
    combined = copy.deepcopy(next(iter(skeleton.values())))
    datadscs: list[dict[str, Any]] = []
    stage_core_ids: list[list[int]] = []

    for stripe_idx, group in enumerate(groups):
        first = group[0]
        gather_stage, _restickify_stage, scatter_stage = first["stages"]
        bridge_core = _as_int(first["bridge_core"])
        source_fragments = [
            fragment
            for tile in group
            for fragment in tile["stages"][0].get("fragments", []) or []
        ]
        dest_fragments = [
            fragment
            for tile in group
            for fragment in tile["stages"][2].get("fragments", []) or []
        ]
        internal_fragment = (
            _compact_tile_fragment
            if _compact_tile_workspace_enabled()
            else _coalesced_tile_fragment
        )
        gathered_fragment = internal_fragment(
            source_fragments,
            core=bridge_core,
        )
        output_fragment = _coalesced_tile_fragment(
            dest_fragments,
            core=bridge_core,
        )
        gather_core_ids = _fragment_core_ids(source_fragments, bridge_core)
        restickify_core_ids = [bridge_core]
        gather_idx = len(datadscs)
        datadscs.append(
            {
                f"{gather_idx}_STCDPOpLx_gather_row_stripe{stripe_idx}": _tile_dataop(
                    "STCDPOpLx",
                    core_ids=gather_core_ids,
                    input_layout=("mb_", "out_"),
                    input_stick=("out_",),
                    output_layout=("mb_", "out_"),
                    output_stick=("out_",),
                    input_base=_as_int(gather_stage["input_base"]),
                    output_base=_as_int(gather_stage["output_base"]),
                    input_fragments=source_fragments,
                    output_fragments=[gathered_fragment],
                )
            }
        )
        stage_core_ids.append(gather_core_ids)
        restickify_idx = len(datadscs)
        restickify_name = (
            f"{restickify_idx}_ReStickifyOpWithPTLx_"
            f"row_stripe{stripe_idx}_direct_output"
        )
        datadscs.append(
            {
                restickify_name: _tile_dataop(
                    "ReStickifyOpWithPTLx",
                    core_ids=restickify_core_ids,
                    input_layout=("mb_", "out_"),
                    input_stick=("out_",),
                    output_layout=("out_", "mb_"),
                    output_stick=("mb_",),
                    input_base=_as_int(gather_stage["output_base"]),
                    output_base=_as_int(scatter_stage["output_base"]),
                    input_fragments=[gathered_fragment],
                    output_fragments=[output_fragment],
                )
            }
        )
        stage_core_ids.append(restickify_core_ids)

    num_cores = max(
        _as_int(descriptor.get("source_core_count", 1)),
        _as_int(descriptor.get("dest_core_count", 1)),
        max((max(core_ids) for core_ids in stage_core_ids if core_ids), default=-1) + 1,
    )
    combined["datadscs_"] = datadscs
    combined["numCoresUsed_"] = num_cores
    combined["coreIdToDscSchedule"] = _sparse_dataop_schedule(
        num_cores,
        stage_core_ids,
    )
    combined["opFuncsUsed_"] = [
        next(iter(datadsc.values()))["op"]["name"] for datadsc in datadscs
    ]
    combined["streamingPTLXTile_"] = {}
    combined["streamingPTLXFull_"] = {
        "status": "static-codegen-only",
        "coalescing": "row-stripe-direct-output",
        "tile_count": len(tiles),
        "logical_tile_count": len(tiles),
        "stripe_count": len(groups),
        "datadsc_count": len(datadscs),
        "fallback": "ReStickifyOpHBM",
    }
    return {name: combined}


def _simple_one_owner_tile(tile: Mapping[str, Any]) -> bool:
    if _as_int(tile.get("fan_in", 0)) != 1 or _as_int(tile.get("fan_out", 0)) != 1:
        return False
    stages = tile.get("stages") or []
    if len(stages) != 3:
        return False
    source_fragments = stages[0].get("fragments", []) or []
    dest_fragments = stages[2].get("fragments", []) or []
    if len(source_fragments) != 1 or len(dest_fragments) != 1:
        return False
    return _as_int(dest_fragments[0]["core"]) == _as_int(tile.get("bridge_core", -1))


def _row_stripe_groups(
    tiles: Sequence[Mapping[str, Any]],
) -> list[list[Mapping[str, Any]]] | None:
    by_row_and_core: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for tile in tiles:
        key = (_as_int(tile["tile_row"]), _as_int(tile["bridge_core"]))
        by_row_and_core.setdefault(key, []).append(tile)

    groups: list[list[Mapping[str, Any]]] = []
    for key in sorted(by_row_and_core):
        group = sorted(by_row_and_core[key], key=lambda tile: _as_int(tile["tile_col"]))
        if not _is_contiguous_row_stripe(group):
            return None
        groups.append(group)
    return groups


def _is_contiguous_row_stripe(group: Sequence[Mapping[str, Any]]) -> bool:
    if not group:
        return False
    dest_fragments = [
        tile["stages"][2]["fragments"][0]
        for tile in group
    ]
    row_start = _as_int(dest_fragments[0]["row_start"])
    row_end = _as_int(dest_fragments[0]["row_end"])
    if any(
        _as_int(fragment["row_start"]) != row_start
        or _as_int(fragment["row_end"]) != row_end
        for fragment in dest_fragments
    ):
        return False
    expected_col = _as_int(dest_fragments[0]["col_start"])
    for fragment in dest_fragments:
        if _as_int(fragment["col_start"]) != expected_col:
            return False
        expected_col = _as_int(fragment["col_end"])
    return True


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
    if direction not in {"kernel-to-output", "output-to-kernel"}:
        raise ValueError(
            "PT-LX restickify bridge currently supports only "
            "direction='kernel-to-output' or 'output-to-kernel', "
            f"got {direction!r}"
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
    if direction == "output-to-kernel":
        input_splits = _swap_mb_out_splits(input_splits)
        input_mapping = _swap_mb_out_mapping(input_mapping)
        # Keep the PT-LX restickify output split in the proven forward shape
        # and let STCDPOpLx remap ownership into the consumer split.
        intermediate_splits = default_intermediate_splits
        intermediate_mapping = default_intermediate_mapping
        final_splits = _swap_mb_out_splits(final_splits)
        final_mapping = _swap_mb_out_mapping(final_mapping)

    intermediate_start = (
        1024 * 1024
        if intermediate_start_address is None
        else int(intermediate_start_address)
    )
    # ``ReStickifyOpWithPTLx`` is only proven on hardware for the forward
    # kernel-to-output shape.  For an output-to-kernel logical edge, keep that
    # Deeptools contract intact and swap the bridge dimension names instead:
    # synthetic mb_ represents the source/output dimension and synthetic out_
    # represents the destination/reduction dimension.
    restickify_input_layout = [d0, d1]
    restickify_input_stick = d1
    restickify_input_strides = {d0: size, d1: 1}
    restickify_output_layout = [d1, d0]
    restickify_output_stick = d0
    restickified_strides = {d0: 1, d1: size}

    restickify_spec = _synthetic_ptlx_bridge_spec(
        size,
        num_cores,
        output_split_dim=d0,
        output_stick_dim=restickify_output_stick,
        input_stick_dim=restickify_input_stick,
        input_start_address=input_start_address,
        output_start_address=intermediate_start,
        input_layout_order=restickify_input_layout,
        output_layout_order=restickify_output_layout,
        input_strides=restickify_input_strides,
        output_strides=restickified_strides,
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

    stcdp_spec = _synthetic_ptlx_bridge_spec(
        size,
        num_cores,
        output_split_dim=final_split_dim,
        output_stick_dim=restickify_output_stick,
        input_stick_dim=restickify_output_stick,
        input_start_address=intermediate_start,
        output_start_address=output_start_address,
        input_layout_order=restickify_output_layout,
        output_layout_order=restickify_output_layout,
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


def _swap_mb_out_splits(splits: Mapping[Any, Any]) -> dict[str, int]:
    normalized = _normalize_work_slices(splits)
    return {
        dim: int(normalized.get(_swap_mb_out_dim(dim), value))
        for dim, value in normalized.items()
    }


def _swap_mb_out_mapping(
    mapping: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    normalized = _normalize_core_to_work_slice(mapping)
    return {
        str(core): {
            dim: int(per_dim.get(_swap_mb_out_dim(dim), value))
            for dim, value in per_dim.items()
        }
        for core, per_dim in normalized.items()
    }


def _swap_mb_out_dim(dim: str) -> str:
    if dim == "mb_":
        return "out_"
    if dim == "out_":
        return "mb_"
    return dim


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


def _with_local_schedule_dependencies(
    schedule: Mapping[str, Sequence[Sequence[int]]],
) -> dict[str, list[list[int]]]:
    """Recompute local before/after dependency bits after concatenating tiles."""

    result: dict[str, list[list[int]]] = {}
    for core_id, steps in schedule.items():
        normalized_steps = [list(step) for step in steps]
        result[str(core_id)] = [
            [
                int(step[0]),
                int(step[1]),
                1 if idx > 0 else 0,
                1 if idx < len(normalized_steps) - 1 else 0,
            ]
            for idx, step in enumerate(normalized_steps)
        ]
    return result


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


def _local_ptlx_labeled_ds(
    lds_name: str,
    pds_name: str,
    *,
    stick: Sequence[str],
    sizes: Mapping[str, int],
    core_id: int,
    base: int,
) -> dict[str, Any]:
    layout_dims = ["j_", "i_", "out_", "mb_"]
    normalized_sizes = {dim: int(sizes[dim]) for dim in layout_dims}
    return {
        "ldsName_": lds_name,
        "pdsName_": pds_name,
        "wordLength": num_bytes(DataFormats.SEN169_FP16),
        "dataformat": DataFormats.SEN169_FP16.name,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": layout_dims,
        "stickDimOrder_": list(stick),
        "dimToLayoutSize_": normalized_sizes,
        "dimToStickSize_": {dim: 64 for dim in stick},
        "validGap_": _valid_gap(normalized_sizes),
        "totElements": -1,
        "PieceInfo": [
            {
                "key_": "p1",
                "dimToStartCordinate": {dim: 0 for dim in layout_dims},
                "dimToSize_": normalized_sizes,
                "validGap_": _valid_gap(normalized_sizes),
                "PlacementInfo": [
                    {
                        "type": "lx",
                        "memId": [int(core_id)],
                        "startAddr": [int(base)],
                    }
                ],
            }
        ],
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": _LX_SIZE_BYTES,
        "lxStartAddress_": {},
    }


def _native_tile_dataop(
    op_name: str,
    *,
    core_ids: Sequence[int],
    input_stick: Sequence[str],
    output_stick: Sequence[str],
    input_base: int,
    output_base: int,
    input_fragments: Sequence[Mapping[str, Any]],
    output_fragments: Sequence[Mapping[str, Any]],
    tile_rows: int,
    tile_cols: int,
) -> dict[str, Any]:
    return {
        "coreIdsUsed_": [int(core) for core in core_ids],
        "dimPool_": ["j_", "i_", "out_", "mb_"],
        "outDimTodimRelation_": [],
        "primaryDs_": [
            {"name_": "dataIN", "dimNames": ["out_", "mb_", "i_", "j_"]},
            {"name_": "dataOUT", "dimNames": ["out_", "mb_", "i_", "j_"]},
        ],
        "labeledDs_": [
            _native_tile_labeled_ds(
                "dataIN_L0",
                "dataIN",
                stick=input_stick,
                base=input_base,
                fragments=input_fragments,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
            ),
            _native_tile_labeled_ds(
                "dataOUT_L0",
                "dataOUT",
                stick=output_stick,
                base=output_base,
                fragments=output_fragments,
                tile_rows=tile_rows,
                tile_cols=tile_cols,
            ),
        ],
        "op": _op_payload(op_name),
    }


def _native_tile_labeled_ds(
    lds_name: str,
    pds_name: str,
    *,
    stick: Sequence[str],
    base: int,
    fragments: Sequence[Mapping[str, Any]],
    tile_rows: int,
    tile_cols: int,
) -> dict[str, Any]:
    layout_sizes = _native_tile_sizes(tile_rows, tile_cols)
    stick_dims = [str(dim) for dim in stick]
    return {
        "ldsName_": lds_name,
        "pdsName_": pds_name,
        "wordLength": num_bytes(DataFormats.SEN169_FP16),
        "dataformat": DataFormats.SEN169_FP16.name,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": ["j_", "i_", "out_", "mb_"],
        "stickDimOrder_": stick_dims,
        "dimToLayoutSize_": layout_sizes,
        "dimToStickSize_": {dim: 64 for dim in stick_dims},
        "validGap_": _valid_gap(layout_sizes),
        "totElements": -1,
        "PieceInfo": [
            _native_tile_piece_info(fragment, base=base, key=f"p{idx + 1}")
            for idx, fragment in enumerate(fragments)
        ],
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": _LX_SIZE_BYTES,
        "lxStartAddress_": {},
    }


def _native_tile_piece_info(
    fragment: Mapping[str, Any],
    *,
    base: int,
    key: str,
) -> dict[str, Any]:
    sizes = {
        "j_": _as_int(fragment["j_end"]) - _as_int(fragment["j_start"]),
        "i_": 1,
        "out_": _as_int(fragment["out_end"]) - _as_int(fragment["out_start"]),
        "mb_": 1,
    }
    return {
        "key_": key,
        "dimToStartCordinate": {
            "j_": _as_int(fragment["j_start"]),
            "i_": 0,
            "out_": _as_int(fragment["out_start"]),
            "mb_": 0,
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


def _native_fragment(
    fragment: Mapping[str, Any],
    tile_row_start: int,
    tile_col_start: int,
) -> dict[str, int]:
    return {
        "core": _as_int(fragment["core"]),
        "j_start": _as_int(fragment["row_start"]) - int(tile_row_start),
        "j_end": _as_int(fragment["row_end"]) - int(tile_row_start),
        "out_start": _as_int(fragment["col_start"]) - int(tile_col_start),
        "out_end": _as_int(fragment["col_end"]) - int(tile_col_start),
    }


def _native_whole_tile_fragment(
    *,
    core: int,
    tile_rows: int,
    tile_cols: int,
) -> dict[str, int]:
    return {
        "core": int(core),
        "j_start": 0,
        "j_end": int(tile_rows),
        "out_start": 0,
        "out_end": int(tile_cols),
    }


def _native_tile_shape(
    source_fragments: Sequence[Mapping[str, Any]],
    dest_fragments: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    fragments = list(source_fragments) + list(dest_fragments)
    if not fragments:
        raise ValueError("native PT-LX tile bridge needs at least one fragment")
    row_start = min(_as_int(fragment["row_start"]) for fragment in fragments)
    row_end = max(_as_int(fragment["row_end"]) for fragment in fragments)
    col_start = min(_as_int(fragment["col_start"]) for fragment in fragments)
    col_end = max(_as_int(fragment["col_end"]) for fragment in fragments)
    return row_end - row_start, col_end - col_start


def _native_tile_sizes(tile_rows: int, tile_cols: int) -> dict[str, int]:
    return {
        "j_": int(tile_rows),
        "i_": 1,
        "out_": int(tile_cols),
        "mb_": 1,
    }


def _native_tile_unpad() -> dict[str, int]:
    return {dim: -1 for dim in ["j_", "i_", "out_", "mb_"]}


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


def _compact_tile_workspace_enabled() -> bool:
    return os.environ.get(_COMPACT_TILE_WORKSPACE_ENV, "0") == "1"


def _compact_tile_fragment(
    fragments: Sequence[Mapping[str, Any]],
    *,
    core: int,
) -> dict[str, int]:
    if not fragments:
        raise ValueError("cannot compact an empty fragment list")
    row_start = min(_as_int(fragment["row_start"]) for fragment in fragments)
    row_end = max(_as_int(fragment["row_end"]) for fragment in fragments)
    col_start = min(_as_int(fragment["col_start"]) for fragment in fragments)
    col_end = max(_as_int(fragment["col_end"]) for fragment in fragments)
    return {
        "core": int(core),
        "row_start": 0,
        "row_end": row_end - row_start,
        "col_start": 0,
        "col_end": col_end - col_start,
        "bytes": (row_end - row_start) * (col_end - col_start) * 2,
        "hops": 0,
    }
