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

from torch_spyre._inductor.op_spec import OpSpec

from .compute_ops import num_bytes
from .superdsc import SDSCArgs, SDSCSpec, parse_op_spec

SUPPORTED_RESTICKIFY_DATA_OPS = frozenset(
    {"STCDPOpLx", "ReStickifyOpLx", "ReStickifyOpHBM"}
)

_LX_SIZE_BYTES = 2 * 1024 * 1024


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
        "op": {"name": op_name},
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
    return {name: combined}


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
        offset += starts.get(str(dim), 0) * _as_int(stride)
    return offset * num_bytes(arg.data_format)


def _core_to_work_slice(sdsc_spec: SDSCSpec) -> dict[str, dict[str, int]]:
    if sdsc_spec.core_id_to_work_slice_override is not None:
        return {
            str(core_id): {str(dim): int(value) for dim, value in per_dim.items()}
            for core_id, per_dim in sdsc_spec.core_id_to_work_slice_override.items()
        }

    result: dict[str, dict[str, int]] = {}
    core_id_sym = Symbol("core_id")
    for core_id in range(sdsc_spec.num_cores):
        result[str(core_id)] = {
            str(dim): int(expr.subs({core_id_sym: core_id}))
            if isinstance(expr, Expr)
            else int(expr)
            for dim, expr in sdsc_spec.core_id_to_work_slice.items()
        }
    return result


def _normalize_core_to_work_slice(
    mapping: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    return {
        str(core_id): {str(dim): int(value) for dim, value in per_dim.items()}
        for core_id, per_dim in mapping.items()
    }


def _normalize_work_slices(work_slices: Mapping[Any, Any]) -> dict[str, int]:
    return {str(dim): _as_int(split) for dim, split in work_slices.items()}


def _dim_pool(sdsc_spec: SDSCSpec) -> list[str]:
    dims: set[str] = set()
    for layout_info in sdsc_spec.layouts.values():
        dims.update(_layout_dim_names(layout_info))
        dims.update(_stick_dim_names(layout_info))
    return sorted(dims)


def _dim_size_map(sdsc_spec: SDSCSpec) -> dict[str, int]:
    return {
        str(dim): _as_int(size) for dim, size in sdsc_spec.iteration_space.items()
    }


def _layout_sizes(layout_dims: Sequence[str], sdsc_spec: SDSCSpec) -> dict[str, int]:
    sizes = _dim_size_map(sdsc_spec)
    return {dim: sizes[dim] for dim in layout_dims}


def _valid_gap(sizes: Mapping[str, int]) -> dict[str, list[list[int]]]:
    return {dim: [[size, 0]] for dim, size in sizes.items()}


def _layout_dim_names(layout_info: Mapping[str, Any]) -> list[str]:
    return [str(dim) for dim in layout_info["dim_order"]]


def _stick_dim_names(layout_info: Mapping[str, Any]) -> list[str]:
    raw = layout_info.get("stick_dim_order")
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(dim) for dim in raw]
    return [str(raw)]


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
