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

"""Emit a minimal mixed-SDSC LXCoordinateRemapOp bundle.

This is a diagnostic artifact generator.  It exercises the same torch-spyre
codegen path used by ``SPYRE_ONCHIP_MOVE_CARRIER=coordinate_remap`` so the
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
    _folded_start_address,
    build_coordinate_remap_onchip_move_sdsc,
)
from torch_spyre._inductor.onchip_move import (
    OnChipMoveCell,
    OnChipMovePlan,
    build_coordinate_remap_metadata,
)
from torch_spyre._inductor.op_spec import TensorArg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    spyre_config.onchip_move_carrier = "coordinate_remap"
    spyre_config.onchip_move_producer_lx_base = 0x1000
    spyre_config.onchip_move_consumer_lx_base = 0x9000

    patched_producer, mixed_consumer = build_coordinate_remap_onchip_move_sdsc(
        0,
        1,
        _producer_payload(),
        _consumer_payload(),
        _producer_output_arg(),
        _consumer_input_arg(),
        1,
        0,
        _plan_payload(),
    )

    _write_json(output_dir / "sdsc_0.json", patched_producer)
    _write_json(output_dir / "sdsc_1.json", mixed_consumer)
    (output_dir / "bundle.mlir").write_text(_bundle_mlir(), encoding="utf-8")

    root = mixed_consumer["1_OnChipMoveCoordinateRemap"]
    dataop = next(iter(root["datadscs_"][0].values()))
    summary = {
        "status": "coordinate-remap-bundle-emitted",
        "producer_sdsc": "sdsc_0.json",
        "consumer_sdsc": "sdsc_1.json",
        "bundle": "bundle.mlir",
        "op": dataop["op"]["name"],
        "movement_count": len(dataop["movements"]),
        "core_ids": dataop["coreIdsUsed_"],
        "coverage": dataop["coverage"],
        "python_byte_simulation_value_correct": _simulate_dataop(dataop),
        "expected_stock_deeptools_status": "unsupported-lx-coordinate-remap-op",
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bundle_mlir() -> str:
    return "\n".join(
        [
            "module {",
            "\tfunc.func @sdsc_bundle() {",
            '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_0.json"}',
            '\t\tsdscbundle.sdsc_execute () {sdsc_filename="sdsc_1.json"}',
            "\t\treturn",
            "\t}",
            "}",
            "",
        ]
    )


def _plan_payload() -> dict[str, Any]:
    cells = [
        OnChipMoveCell(
            cell_index=0,
            source_core=0,
            dest_core=0,
            dim_starts={"d0_": 0},
            dim_sizes={"d0_": 4},
            bytes=8,
            source_offset_bytes=0,
            dest_offset_bytes=0,
        ),
        OnChipMoveCell(
            cell_index=1,
            source_core=1,
            dest_core=0,
            dim_starts={"d0_": 4},
            dim_sizes={"d0_": 4},
            bytes=8,
            source_offset_bytes=0,
            dest_offset_bytes=8,
        ),
        OnChipMoveCell(
            cell_index=2,
            source_core=2,
            dest_core=1,
            dim_starts={"d0_": 8},
            dim_sizes={"d0_": 4},
            bytes=8,
            source_offset_bytes=0,
            dest_offset_bytes=0,
        ),
        OnChipMoveCell(
            cell_index=3,
            source_core=3,
            dest_core=1,
            dim_starts={"d0_": 12},
            dim_sizes={"d0_": 4},
            bytes=8,
            source_offset_bytes=0,
            dest_offset_bytes=8,
        ),
    ]
    plan = OnChipMovePlan(
        source_name="buf0",
        producer_name="buf0",
        consumer_name="buf1",
        producer_op="producer_op",
        consumer_op="consumer_op",
        status="planned",
        fallback_reason=None,
        realization_status="planned-coordinate-remap-needs-deeptools",
        carrier="coordinate_remap",
        device_sizes=[16],
        device_stride_map=[1],
        element_bytes=2,
        producer_core_count=4,
        consumer_core_count=2,
        producer_region_bytes=8,
        consumer_region_bytes=16,
        producer_view={},
        consumer_view={},
        cells=cells,
    )
    payload = dataclasses.asdict(plan)
    payload["producer"] = plan.producer_name
    payload["consumer"] = plan.consumer_name
    payload["cell_count"] = len(cells)
    payload["bytes_moved"] = plan.bytes_moved
    payload["cells"] = [dataclasses.asdict(cell) for cell in cells]
    payload["coordinate_remap"] = build_coordinate_remap_metadata(plan)
    return payload


def _producer_output_arg() -> TensorArg:
    return TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[16],
        device_coordinates=[],
        allocation=None,
        stride_map=[1],
        name="buf0",
    )


def _consumer_input_arg() -> TensorArg:
    return dataclasses.replace(_producer_output_arg(), is_input=True, arg_index=0)


def _producer_payload() -> dict[str, Any]:
    return {
        "0_neg": _minimal_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=[0, 1, 2, 3],
        )
    }


def _consumer_payload() -> dict[str, Any]:
    return {
        "1_neg": _minimal_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=[0, 1],
        )
    }


def _minimal_sdsc_payload(
    name: str,
    *,
    output_lds_idx: int,
    input_lds_indices: list[int],
    core_ids: list[int],
) -> dict[str, Any]:
    max_lds_idx = max([output_lds_idx, *input_lds_indices])
    primary_ds_info = {
        role: {
            "layoutDimOrder_": ["out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        }
        for role in ("INPUT", "OUTPUT", "INTERNAL")
    }
    return {
        "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
        "sdscFolds_": {
            "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
            "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
            "data_": {"[0]": 0},
        },
        "coreFoldProp_": {"factor_": 32, "label_": "core"},
        "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
        "numCoresUsed_": len(core_ids),
        "coreIdToDsc_": {str(core): 0 for core in core_ids},
        "numWkSlicesPerDim_": {"out": len(core_ids)},
        "coreIdToWkSlice_": {
            str(core): {"out": idx} for idx, core in enumerate(core_ids)
        },
        "opFuncsUsed_": [name],
        "dscs_": [
            {
                name: {
                    "numCoresUsed_": len(core_ids),
                    "coreIdsUsed_": core_ids,
                    "N_": {
                        "name_": "n",
                        "out_": 64,
                    },
                    "coordinateMasking_": {},
                    "maskingConstId_": -1,
                    "dataStageParam_": {
                        "0": {
                            "ss_": {"name_": "core", "out_": 64},
                            "el_": {"name_": "core", "out_": 64},
                        }
                    },
                    "primaryDsInfo_": primary_ds_info,
                    "scheduleTree_": [
                        {
                            "nodeType_": "allocate",
                            "name_": f"allocate-Tensor{idx}_hbm",
                            "prev_": "",
                            "ldsIdx_": idx,
                            "component_": "hbm",
                            "layoutDimOrder_": ["out"],
                            "maxDimSizes_": [-1],
                            "startAddressCoreCorelet_": _folded_start_address(
                                core_ids, idx * 1024, core_factor=32
                            ),
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
                                else "INTERNAL"
                            ),
                            "scale_": [1],
                            "wordLength": 2,
                            "dataFormat_": DataFormats.SEN169_FP16.name,
                            "memOrg_": {
                                "hbm": {"isPresent": 1},
                                "lx": {"isPresent": 0},
                            },
                        }
                        for idx in range(max_lds_idx + 1)
                    ],
                    "computeOp_": [
                        {
                            "exUnit": "sfp",
                            "opFuncName": name,
                            "attributes_": {
                                "dataFormat_": DataFormats.SEN169_FP16.name,
                                "fidelity_": "regular",
                            },
                            "location": "Inner",
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


def _simulate_dataop(dataop: dict[str, Any]) -> bool:
    source_by_core: dict[int, bytearray] = {}
    destination_by_core: dict[int, bytearray] = {}
    for move in dataop["movements"]:
        source_range = move["source"]["localByteRange"]
        dest_range = move["destination"]["localByteRange"]
        source = source_by_core.setdefault(
            move["source"]["core"], bytearray(source_range["end"])
        )
        if len(source) < source_range["end"]:
            source.extend(b"\x00" * (source_range["end"] - len(source)))
        destination = destination_by_core.setdefault(
            move["destination"]["core"], bytearray(dest_range["end"])
        )
        if len(destination) < dest_range["end"]:
            destination.extend(b"\x00" * (dest_range["end"] - len(destination)))

    for core, source in source_by_core.items():
        for offset in range(len(source)):
            source[offset] = (core * 17 + offset) % 251

    for move in dataop["movements"]:
        source_range = move["source"]["localByteRange"]
        dest_range = move["destination"]["localByteRange"]
        source = source_by_core[move["source"]["core"]]
        destination = destination_by_core[move["destination"]["core"]]
        destination[dest_range["start"] : dest_range["end"]] = source[
            source_range["start"] : source_range["end"]
        ]

    for move in dataop["movements"]:
        source_range = move["source"]["localByteRange"]
        dest_range = move["destination"]["localByteRange"]
        source = source_by_core[move["source"]["core"]]
        destination = destination_by_core[move["destination"]["core"]]
        if destination[dest_range["start"] : dest_range["end"]] != source[
            source_range["start"] : source_range["end"]
        ]:
            return False
    return True


if __name__ == "__main__":
    main()
