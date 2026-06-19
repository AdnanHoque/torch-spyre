#!/usr/bin/env python3
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

"""Emit a minimal dense-actual STCDP bundle for the SwiGLU DXP blocker.

This is a diagnostic artifact generator.  It intentionally mirrors
``test_mixed_carrier_diagnoses_dense_actual_output_stride_blocker`` so the
backend-facing SDSC can be compiled or inspected independently of a full AIU
SwiGLU run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.onchip_move import (
    build_mixed_onchip_move_sdsc,
    diagnose_stcdp_output_layout_contiguity,
)
from torch_spyre._inductor.op_spec import TensorArg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spyre_config.onchip_move_output_piece_mode = "dense_actual"
    _patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
        0,
        1,
        _producer_payload(),
        _consumer_payload(),
        _producer_output_arg(),
        _consumer_input_arg(),
        2,
        0,
        _plan(),
    )

    dataop = mixed_consumer["1_OnChipMoveMixedSTCDP"]["datadscs_"][0][
        "0_OnChipMoveSTCDPOpLx"
    ]
    mismatches = diagnose_stcdp_output_layout_contiguity(dataop)

    _write_json(output_dir / "sdsc_0.json", mixed_consumer)
    _write_json(output_dir / "layout_mismatches.json", mismatches)
    (output_dir / "bundle.mlir").write_text(
        "\n".join(
            [
                "module {",
                "\tfunc.func @sdsc_bundle() {",
                '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}',
                "\t\treturn",
                "\t}",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "status": (
            "blocked-output-layout-requires-strided-placement"
            if mismatches
            else "no-layout-contiguity-mismatch"
        ),
        "sdsc": "sdsc_0.json",
        "bundle": "bundle.mlir",
        "mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _plan() -> dict[str, Any]:
    return {
        "source_name": "buf0",
        "producer": "buf0",
        "consumer": "buf1",
        "device_sizes": [512, 8, 1, 64],
        "device_stride_map": [512, 64, -1, 1],
        "producer_region_bytes": 16 * 1024,
        "consumer_region_bytes": 16 * 1024,
        "bytes_moved": 2048,
        "cell_count": 1,
        "cells": [
            {
                "cell_index": 0,
                "source_core": 0,
                "dest_core": 0,
                "source_offset_bytes": 0,
                "dest_offset_bytes": 0,
                "bytes": 2048,
                "dim_starts": {"d0_": 0, "d1_": 1, "d2_": 0, "d3_": 0},
                "dim_sizes": {"d0_": 16, "d1_": 1, "d2_": 1, "d3_": 64},
            }
        ],
    }


def _producer_output_arg() -> TensorArg:
    return TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 8, 1, 64],
        device_coordinates=[],
        allocation=None,
        stride_map=[512, 64, -1, 64],
        name="buf0",
    )


def _consumer_input_arg() -> TensorArg:
    return dataclasses.replace(_producer_output_arg(), is_input=True, arg_index=0)


def _producer_payload() -> dict[str, Any]:
    return {
        "0_batchmatmul": _layout_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=list(range(32)),
            include_x=True,
        )
    }


def _consumer_payload() -> dict[str, Any]:
    return {
        "1_neg": _layout_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=list(range(32)),
            input_layout_dim_order=["out", "mb"],
            output_layout_dim_order=["out", "mb"],
        )
    }


def _layout_sdsc_payload(
    name: str,
    *,
    output_lds_idx: int,
    input_lds_indices: list[int],
    core_ids: list[int],
    include_x: bool = False,
    input_layout_dim_order: list[str] | None = None,
    output_layout_dim_order: list[str] | None = None,
) -> dict[str, Any]:
    payload = _minimal_sdsc_payload(
        name,
        output_lds_idx=output_lds_idx,
        input_lds_indices=input_lds_indices,
        core_ids=core_ids,
    )
    dsc = next(iter(payload["dscs_"][0].values()))
    default_layout_dim_order = ["mb", "out", "x"] if include_x else ["mb", "out"]
    input_layout_dim_order = input_layout_dim_order or default_layout_dim_order
    output_layout_dim_order = output_layout_dim_order or default_layout_dim_order
    all_layout_dims = list(
        dict.fromkeys([*input_layout_dim_order, *output_layout_dim_order])
    )
    n_info = {"name_": "n"}
    for dim in all_layout_dims:
        n_info[f"{dim}_"] = 1 if dim == "x" else 512
    dsc["N_"] = n_info
    dsc["primaryDsInfo_"] = {
        "INPUT": {
            "layoutDimOrder_": input_layout_dim_order,
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": output_layout_dim_order,
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    return payload


def _minimal_sdsc_payload(
    name: str,
    *,
    output_lds_idx: int,
    input_lds_indices: list[int],
    core_ids: list[int],
) -> dict[str, Any]:
    max_lds_idx = max([output_lds_idx, *input_lds_indices])
    return {
        "numCoresUsed_": len(core_ids),
        "opFuncsUsed_": [name],
        "dscs_": [
            {
                name: {
                    "numCoresUsed_": len(core_ids),
                    "coreIdsUsed_": core_ids,
                    "scheduleTree_": [
                        {
                            "name_": f"allocate_lds{idx}_hbm",
                            "ldsIdx_": idx,
                            "component_": "hbm",
                        }
                        for idx in range(max_lds_idx + 1)
                    ],
                    "labeledDs_": [
                        {
                            "ldsIdx_": idx,
                            "dsName_": f"Tensor{idx}",
                            "dsType_": (
                                "OUTPUT"
                                if idx == output_lds_idx
                                else "INPUT"
                                if idx in input_lds_indices
                                else "LOCAL"
                            ),
                            "scale_": [1, 1],
                            "wordLength": 2,
                            "dataFormat_": DataFormats.SEN169_FP16.name,
                            "memOrg_": {"hbm": {"isPresent": 1}},
                        }
                        for idx in range(max_lds_idx + 1)
                    ],
                    "computeOp_": [
                        {
                            "inputLabeledDs": [
                                f"Tensor{idx}-idx{idx}" for idx in input_lds_indices
                            ],
                            "outputLabeledDs": [
                                f"Tensor{output_lds_idx}-idx{output_lds_idx}"
                            ],
                        }
                    ],
                }
            }
        ],
    }


if __name__ == "__main__":
    main()
