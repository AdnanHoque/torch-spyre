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

"""Causal mask data-op descriptors for the score-bias bring-up path.

This module is intentionally independent of torch/torch_spyre imports so probes
and no-torch tests can use it even when the backend aborts during compilation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _iteration_size(contract: Mapping[str, Any], dim: str | None) -> int | None:
    if not dim:
        return None
    sizes = contract.get("iteration_sizes", {})
    if not isinstance(sizes, Mapping):
        return None
    for key in (f"{dim}_", dim):
        size = _maybe_int(sizes.get(key))
        if size is not None:
            return size
    return None


def _dim_index(layout_dim_order: list[str], dim: str | None) -> int | None:
    if not dim:
        return None
    try:
        return layout_dim_order.index(dim)
    except ValueError:
        return None


def _dci_output_shape(
    layout_dim_order: list[str],
    stick_dim_order: list[str],
    stick_size: list[int],
    mask_layout_sizes: Mapping[str, int],
) -> list[int] | None:
    if len(stick_dim_order) != 1 or len(stick_size) != 1:
        return None
    stick_dim = stick_dim_order[0]
    stick = stick_size[0]
    if stick <= 0:
        return None
    shape = [stick]
    for dim in layout_dim_order:
        size = mask_layout_sizes.get(dim)
        if size is None:
            return None
        if dim == stick_dim:
            if size % stick != 0:
                return None
            shape.append(size // stick)
        else:
            shape.append(size)
    return shape


def _causal_dci_stride_info(
    layout_dim_order: list[str],
    query_dim: str | None,
    key_dim: str | None,
    query_len: int | None,
    key_len: int | None,
    stick_size: list[int],
) -> dict[str, int | None]:
    if (
        query_dim is None
        or key_dim is None
        or query_len is None
        or key_len is None
        or len(stick_size) != 1
    ):
        return {"causalDimLength_": query_len}
    query_pos = _dim_index(layout_dim_order, query_dim)
    key_pos = _dim_index(layout_dim_order, key_dim)
    if query_pos is None or key_pos is None:
        return {"causalDimLength_": query_len}
    if query_pos > key_pos:
        continuous = key_len
        stride = 1
    else:
        continuous = stick_size[0]
        stride = stick_size[0] * query_len
    return {
        "causalDimLength_": query_len,
        "continuousMaskElems_": continuous,
        "strideAfterContinuous_": stride,
    }


def build_causal_idx_to_mask_candidate(
    contract: Mapping[str, Any], *, key_start: int | str | None
) -> dict[str, Any]:
    """Describe the IdxToMask plus where3 implementation for a score-bias SDSC.

    The returned object is a JSON-serializable contract descriptor. It records
    whether the current generated score layout is eligible for the known
    DeepTools causal IdxToMask path and, when eligible, the exact metadata that
    Torch-Spyre still needs to emit.
    """

    output_layout = contract.get("output_layout", {})
    if not isinstance(output_layout, Mapping):
        output_layout = {}
    layout_dim_order = list(output_layout.get("layout_dim_order") or [])
    stick_dim_order = list(output_layout.get("stick_dim_order") or [])
    stick_size = list(output_layout.get("stick_size") or [])

    query_dim = contract.get("inferred_query_dim")
    key_dim = contract.get("inferred_key_dim")
    query_len = _iteration_size(contract, query_dim)
    key_len = _iteration_size(contract, key_dim)
    key_start_value = _maybe_int(key_start)
    key_dim_index = _dim_index(layout_dim_order, key_dim)
    work_slices = contract.get("work_slices", {})
    if not isinstance(work_slices, Mapping):
        work_slices = {}
    key_dim_slices = _maybe_int(work_slices.get(key_dim)) if key_dim else None
    score_layout_sizes = {
        dim: _iteration_size(contract, dim) for dim in layout_dim_order
    }
    mask_layout_sizes = {
        dim: (
            score_layout_sizes[dim]
            if dim in (query_dim, key_dim)
            else 1
        )
        for dim in layout_dim_order
        if score_layout_sizes.get(dim) is not None
    }
    broadcast_dims = [
        dim
        for dim in layout_dim_order
        if dim not in (query_dim, key_dim)
        and (score_layout_sizes.get(dim) or 1) != 1
    ]
    dci_output_shape = _dci_output_shape(
        layout_dim_order,
        stick_dim_order,
        stick_size,
        mask_layout_sizes,
    )
    dci_stride_info = _causal_dci_stride_info(
        layout_dim_order,
        query_dim,
        key_dim,
        query_len,
        key_len,
        stick_size,
    )

    rejection_reasons: list[str] = []
    if contract.get("opfunc") != "causal_score_bias_like":
        rejection_reasons.append("opfunc is not causal_score_bias_like")
    if contract.get("input_count") != 1 or contract.get("output_count") != 1:
        rejection_reasons.append("expected one layout-anchor input and one output")
    if "keyStart" not in list(contract.get("constants") or []):
        rejection_reasons.append("missing keyStart constant")
    if not contract.get("supported_score_layout", False):
        rejection_reasons.append("unsupported score output layout")
    if query_dim != "x":
        rejection_reasons.append("query dimension is not x")
    if key_dim != "out":
        rejection_reasons.append("key dimension is not out")
    if stick_dim_order != [key_dim]:
        rejection_reasons.append("key dimension is not the sole stick dimension")
    if key_dim_index is None:
        rejection_reasons.append("key dimension is absent from layoutDimOrder_")
    if query_len is None:
        rejection_reasons.append("query length is absent from N_")
    if key_len is None:
        rejection_reasons.append("key length is absent from N_")
    if dci_output_shape is None:
        rejection_reasons.append("cannot derive IdxToMask DCI output shape")
    if key_start_value is None:
        rejection_reasons.append("key_start is not an integer")
    elif key_start_value < 0:
        rejection_reasons.append("key_start must be non-negative")
    if key_dim_slices not in (None, 1):
        rejection_reasons.append("key stick dimension must remain unsplit")

    valid_offset = -key_start_value if key_start_value is not None else None
    return {
        "strategy": "idx_to_mask_plus_where3",
        "feasible": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "runtime_emission": {
            "torch_spyre_descriptor_only": True,
            "datadsc_json_accepts_idx_to_mask": False,
            "requires_deeptools_dataop_parser_extension": True,
            "blocking_reason": (
                "DeepTools DataOpDsc does not currently accept "
                "op.name=IdxToMask from imported SuperDSC datadscs_ JSON"
            ),
        },
        "layout": {
            "query_dim": query_dim,
            "key_dim": key_dim,
            "layout_dim_order": layout_dim_order,
            "stick_dim_order": stick_dim_order,
            "query_length": query_len,
            "key_length": key_len,
            "score_layout_sizes": score_layout_sizes,
            "mask_layout_sizes": mask_layout_sizes,
            "broadcast_dims": broadcast_dims,
        },
        "idx_to_mask": {
            "isIdxToMaskSdc": True,
            "idxToMaskDim": key_dim,
            "idxToMaskDimIdx": key_dim_index,
            "idxToMaskValidElementOffset": valid_offset,
            "causalMask": True,
            "invertedMask": False,
            "reversedMask": False,
            "input": {
                "kind": "length_one_query_length_vector",
                "shape": [1],
                "value": query_len,
                "dtype": "IEEE_INT64",
            },
            "output": {
                "dtype": "SEN169_FP16",
                "valid_value": 1.0,
                "invalid_value": 0.0,
                "layout_sizes": mask_layout_sizes,
            },
        },
        "dci": {
            "dcOpName_": "IDX_TO_MASK",
            "dataformat_src_": "IEEE_INT64",
            "dataformat_dst_": "SEN169_FP16",
            "input_shape_": [1],
            "output_shape_": dci_output_shape,
            "imi_": {
                "idxToMaskValidElementOffset_": valid_offset,
                "maskInnerRepeat_": 1,
                "invertMask_": False,
                "reverseMask_": False,
                "isCausalMask_": True,
                **dci_stride_info,
            },
        },
        "dataop_json_extension": {
            "op": {
                "name": "IdxToMask",
                "idxToMaskDimIdx": key_dim_index,
                "idxToMaskValidElementOffset": valid_offset,
                "invertedMask": 0,
                "reversedMask": 0,
                "causalMask": 1,
            }
        },
        "where3": {
            "opFuncName": "where3",
            "predicate": "idx_to_mask.output",
            "true_value": 0.0,
            "false_value": "-inf",
            "output": "causal_score_bias_like.output",
            "broadcast_predicate_dims": broadcast_dims,
        },
        "required_codegen": [
            "emit IdxToMask data-convert DCI/NodeProperty metadata",
            "allocate an internal causal-plane mask tensor with non-causal dims set to 1",
            "compose where3(mask, 0, -inf) into the bias output with predicate broadcast",
        ],
    }
