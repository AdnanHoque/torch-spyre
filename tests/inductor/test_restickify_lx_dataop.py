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

import pytest
from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.restickify_lx_dataop import (
    combine_dataop_sdscs,
    generate_ptlx_restickify_bridge_sdsc,
    generate_restickify_dataop_sdsc_from_spec,
)
from torch_spyre._inductor.codegen.restickify_ptlx_boundary import (
    _combine_ptlx_bridge_with_consumer,
)
from torch_spyre._inductor.codegen.superdsc import SDSCArgs, SDSCSpec


def _core_mapping(dims, split_dim, num_cores):
    return {
        str(core): {
            str(dim): core if dim == split_dim else 0
            for dim in dims
        }
        for core in range(num_cores)
    }


def _spec(size=128, num_cores=2, output_split_dim=None):
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    output_split_dim = output_split_dim or d0
    data_format = DataFormats.SEN169_FP16
    work_slices = {d0: 1, d1: 1}
    work_slices[output_split_dim] = num_cores
    args = [
        SDSCArgs(
            layout="INPUT",
            data_format=data_format,
            scales={d0: 1, d1: 1},
            strides={d0: size, d1: 1},
            offsets={},
            max_dim_sizes={d0: -1, d1: -1},
            allocation={"lx": 0},
            start_address=0,
            backGap={},
        ),
        SDSCArgs(
            layout="OUTPUT",
            data_format=data_format,
            scales={d0: 1, d1: 1},
            strides={d0: 1, d1: size},
            offsets={},
            max_dim_sizes={d0: -1, d1: -1},
            allocation={"lx": 0},
            start_address=1024,
            backGap={},
        ),
    ]
    return SDSCSpec(
        opfunc="ReStickifyOpHBM",
        execution_unit="sfp",
        data_format=data_format,
        num_inputs=1,
        iteration_space={d0: size, d1: size},
        num_cores=num_cores,
        work_slices=work_slices,
        core_id_to_work_slice={},
        core_id_to_work_slice_override=_core_mapping(
            [d0, d1], output_split_dim, num_cores
        ),
        padding={},
        layouts={
            "INPUT": {
                "dim_order": [d0, d1],
                "stick_dim_order": d1,
                "stick_size": 64,
            },
            "OUTPUT": {
                "dim_order": [d1, d0],
                "stick_dim_order": d0,
                "stick_size": 64,
            },
        },
        args=args,
        constants={},
        coordinate_masking={},
    )


def _dataop(payload):
    root = payload[next(iter(payload))]
    return next(iter(root["datadscs_"][0].values()))


def test_generate_stcdp_lx_dataop_sdsc_shape():
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    spec = _spec(output_split_dim=d0)

    payload = generate_restickify_dataop_sdsc_from_spec(
        0,
        spec,
        op_name="STCDPOpLx",
        input_work_slices={d0: 1, d1: 2},
        input_core_to_work_slice=_core_mapping([d0, d1], d1, 2),
        output_work_slices={d0: 2, d1: 1},
        output_core_to_work_slice=_core_mapping([d0, d1], d0, 2),
    )

    root = payload["0_STCDPOpLx_dataop"]
    dataop = _dataop(payload)
    assert root["dscs_"] == []
    assert len(root["datadscs_"]) == 1
    assert dataop["op"]["name"] == "STCDPOpLx"
    assert dataop["coreIdsUsed_"] == [0, 1]
    assert [lds["ldsName_"] for lds in dataop["labeledDs_"]] == [
        "dataIN_L0",
        "dataOUT_L0",
    ]

    input_piece = dataop["labeledDs_"][0]["PieceInfo"][1]
    output_piece = dataop["labeledDs_"][1]["PieceInfo"][1]
    assert input_piece["dimToStartCordinate"] == {"d0": 0, "d1": 64}
    assert output_piece["dimToStartCordinate"] == {"d1": 0, "d0": 64}
    assert input_piece["PlacementInfo"] == [
        {"type": "lx", "memId": [1], "startAddr": [0]}
    ]


def test_stage3b_like_dataop_keeps_same_logical_owner_dimension():
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    spec = _spec(output_split_dim=d1)

    payload = generate_restickify_dataop_sdsc_from_spec(
        0,
        spec,
        op_name="ReStickifyOpLx",
        input_work_slices={d0: 1, d1: 2},
        input_core_to_work_slice=_core_mapping([d0, d1], d1, 2),
        output_work_slices={d0: 1, d1: 2},
        output_core_to_work_slice=_core_mapping([d0, d1], d1, 2),
    )

    dataop = _dataop(payload)
    input_piece = dataop["labeledDs_"][0]["PieceInfo"][1]
    output_piece = dataop["labeledDs_"][1]["PieceInfo"][1]
    assert input_piece["dimToStartCordinate"]["d1"] == 64
    assert output_piece["dimToStartCordinate"]["d1"] == 64
    assert dataop["op"]["name"] == "ReStickifyOpLx"


def test_restickify_hbm_dataop_includes_hbm_placements():
    payload = generate_restickify_dataop_sdsc_from_spec(
        0,
        _spec(),
        op_name="ReStickifyOpHBM",
    )

    dataop = _dataop(payload)
    placements = dataop["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"]
    assert placements[0]["type"] == "lx"
    assert placements[1]["type"] == "hbm"
    assert placements[1]["memId"] == [-1]


def test_rejects_unknown_dataop_name():
    with pytest.raises(ValueError, match="unsupported restickify data op"):
        generate_restickify_dataop_sdsc_from_spec(0, _spec(), op_name="identity")


def test_combine_dataop_sdscs_keeps_multiple_dataops():
    first = generate_restickify_dataop_sdsc_from_spec(
        0,
        _spec(),
        op_name="ReStickifyOpLx",
    )
    second = generate_restickify_dataop_sdsc_from_spec(
        1,
        _spec(),
        op_name="STCDPOpLx",
    )

    combined = combine_dataop_sdscs("0_two_step", [first, second])

    root = combined["0_two_step"]
    assert root["dscs_"] == []
    assert len(root["datadscs_"]) == 2
    assert root["coreIdToDscSchedule"] == {
        "0": [[0, -1, 0, 1], [1, -1, 1, 0]],
        "1": [[0, -1, 0, 1], [1, -1, 1, 0]],
    }
    ops = [
        next(iter(datadsc.values()))["op"]["name"]
        for datadsc in root["datadscs_"]
    ]
    assert ops == ["ReStickifyOpLx", "STCDPOpLx"]


def test_mixed_ptlx_bridge_with_consumer_schedule_shape():
    bridge = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=128,
        num_cores=2,
        mode="stage3b",
        direction="kernel-to-output",
        restickify_op_name="ReStickifyOpWithPTLx",
    )
    consumer = {
        "2_add": {
            "numCoresUsed_": 2,
            "opFuncsUsed_": ["add"],
            "dscs_": [{"add": {"computeOp_": [{"opFuncName": "add"}]}}],
            "datadscs_": [],
            "coreIdToDscSchedule": {},
        }
    }

    mixed = _combine_ptlx_bridge_with_consumer("1_mixed", bridge, consumer)

    root = mixed["1_mixed"]
    assert len(root["dscs_"]) == 1
    assert len(root["datadscs_"]) == 2
    assert root["opFuncsUsed_"] == [
        "ReStickifyOpWithPTLx",
        "STCDPOpLx",
        "add",
    ]
    assert root["coreIdToDscSchedule"] == {
        "0": [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]],
        "1": [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]],
    }
