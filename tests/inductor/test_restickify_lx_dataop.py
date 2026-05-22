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
from torch_spyre._inductor import config
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.codegen.restickify_lx_dataop import (
    combine_dataop_sdscs,
    generate_ptlx_restickify_bridge_sdsc,
    generate_restickify_dataop_sdsc_from_spec,
    generate_streaming_ptlx_full_bridge_sdsc,
    generate_streaming_ptlx_tile_bridge_sdsc,
)
from torch_spyre._inductor.codegen.restickify_ptlx_streaming import (
    default_core_mapping,
    generate_streaming_ptlx_artifact,
    plan_streaming_ptlx_tiles,
)
from torch_spyre._inductor.codegen.restickify_ptlx_boundary import (
    _constant_lx_start_payload,
    _combine_ptlx_bridge_with_consumer,
    _mixed_value_flow_contract,
    _endpoint_core_starts,
    _materialize_bridge_lx_endpoints,
    _patch_bridge_endpoint_pieces,
    _patch_consumer_input_lx_map,
    _patch_lx_allocation_by_index,
    _streaming_value_flow_contract,
    patch_implicit_restickify_ptlx_aliases,
    patch_restickify_ptlx_cross_bundle_handoffs,
    patch_restickify_ptlx_mixed_schedules,
    plan_restickify_ptlx_mixed_schedules,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.restickify_ring import (
    CORE_MAPPING_OVERRIDE_OP_INFO_KEY,
    LOCALITY_CERTIFICATE_OP_INFO_KEY,
    PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY,
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


def _dataop_at(payload, idx):
    root = payload[next(iter(payload))]
    return next(iter(root["datadscs_"][idx].values()))


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


def test_streaming_ptlx_tile_bridge_sdsc_materializes_three_lx_dataops():
    source = {"mb": 32, "out": 1}
    dest = {"mb": 4, "out": 8}
    summary = plan_streaming_ptlx_tiles(
        size=512,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
        sample_limit=1,
    )
    artifact = generate_streaming_ptlx_artifact("streaming", summary, max_tiles=1)

    payload = generate_streaming_ptlx_tile_bridge_sdsc("tile_bridge", artifact)
    root = payload["tile_bridge"]

    assert root["streamingPTLXTile_"]["status"] == "static-codegen-only"
    assert root["dscs_"] == []
    assert [
        next(iter(dataop.values()))["op"]["name"] for dataop in root["datadscs_"]
    ] == [
        "STCDPOpLx",
        "ReStickifyOpWithPTLx",
        "STCDPOpLx",
    ]
    assert root["opFuncsUsed_"] == [
        "STCDPOpLx",
        "ReStickifyOpWithPTLx",
        "STCDPOpLx",
    ]
    gather = next(iter(root["datadscs_"][0].values()))
    restickify = next(iter(root["datadscs_"][1].values()))
    scatter = next(iter(root["datadscs_"][2].values()))
    assert gather["coreIdsUsed_"] == [0, 1, 2, 3]
    assert restickify["coreIdsUsed_"] == [0]
    assert scatter["coreIdsUsed_"] == [0]

    gather_input_placements = [
        piece["PlacementInfo"][0] for piece in gather["labeledDs_"][0]["PieceInfo"]
    ]
    gather_output_placements = [
        piece["PlacementInfo"][0] for piece in gather["labeledDs_"][1]["PieceInfo"]
    ]
    assert [placement["type"] for placement in gather_input_placements] == [
        "lx",
        "lx",
        "lx",
        "lx",
    ]
    assert [placement["memId"][0] for placement in gather_input_placements] == [
        0,
        1,
        2,
        3,
    ]
    assert len(gather_output_placements) == 1
    assert {placement["memId"][0] for placement in gather_output_placements} == {0}
    assert restickify["labeledDs_"][0]["stickDimOrder_"] == ["out_"]
    assert restickify["labeledDs_"][1]["stickDimOrder_"] == ["mb_"]
    assert scatter["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"][0]["type"] == "lx"
    assert all(
        placement["type"] != "hbm"
        for dataop in root["datadscs_"]
        for ds in next(iter(dataop.values()))["labeledDs_"]
        for piece in ds["PieceInfo"]
        for placement in piece["PlacementInfo"]
    )
    assert len(root["coreIdToDscSchedule"]) == 32
    assert root["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 0],
    ]
    assert root["coreIdToDscSchedule"]["1"] == [[0, -1, 0, 0]]
    assert root["coreIdToDscSchedule"]["4"] == []


def test_streaming_ptlx_full_bridge_sdsc_combines_materialized_tiles():
    source = {"mb": 32, "out": 1}
    dest = {"mb": 4, "out": 8}
    summary = plan_streaming_ptlx_tiles(
        size=512,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
        sample_limit=2,
    )
    artifact = generate_streaming_ptlx_artifact("streaming", summary, max_tiles=2)

    payload = generate_streaming_ptlx_full_bridge_sdsc("full_bridge", artifact)
    root = payload["full_bridge"]

    assert root["streamingPTLXFull_"]["tile_count"] == 2
    assert root["streamingPTLXFull_"]["datadsc_count"] == 6
    assert len(root["datadscs_"]) == 6
    assert root["coreIdToDscSchedule"]["0"][:4] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 0],
        [3, -1, 0, 0],
    ]
    assert root["coreIdToDscSchedule"]["4"] == [
        [3, -1, 0, 1],
        [4, -1, 1, 1],
        [5, -1, 1, 0],
    ]
    assert root["coreIdToDscSchedule"]["1"] == [
        [0, -1, 0, 0],
        [3, -1, 0, 0],
    ]


def test_streaming_ptlx_full_bridge_coalesces_single_owner_row_stripes():
    source = {"mb": 1, "out": 32}
    dest = {"mb": 32, "out": 1}
    summary = plan_streaming_ptlx_tiles(
        size=2048,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
        sample_limit=1024,
        sample_all_tiles=True,
    )
    artifact = generate_streaming_ptlx_artifact(
        "streaming",
        summary,
        producer_base=0,
        consumer_base=512 * 1024,
        max_tiles=summary.total_tiles,
    )

    payload = generate_streaming_ptlx_full_bridge_sdsc("full_bridge", artifact)
    root = payload["full_bridge"]
    first_gather = next(iter(root["datadscs_"][0].values()))
    first_restickify = next(iter(root["datadscs_"][1].values()))
    contract = _streaming_value_flow_contract(
        bridge_payload=payload,
        producer_base=0,
        consumer_base=512 * 1024,
        expected_tiles=summary.total_tiles,
    )

    assert root["streamingPTLXFull_"]["coalescing"] == "row-stripe-direct-output"
    assert root["streamingPTLXFull_"]["tile_count"] == 1024
    assert root["streamingPTLXFull_"]["stripe_count"] == 32
    assert root["streamingPTLXFull_"]["datadsc_count"] == 64
    assert len(root["datadscs_"]) == 64
    assert first_gather["op"]["name"] == "STCDPOpLx"
    assert len(first_gather["labeledDs_"][0]["PieceInfo"]) == 32
    assert first_restickify["op"]["name"] == "ReStickifyOpWithPTLx"
    assert first_restickify["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [512 * 1024]}
    ]
    assert root["coreIdToDscSchedule"]["0"][:3] == [
        [0, -1, 0, 1],
        [1, -1, 1, 1],
        [2, -1, 1, 1],
    ]
    assert contract["valid"] is True
    assert contract["gather_count"] == 32
    assert contract["scatter_count"] == 0
    assert contract["direct_consumer_write_count"] == 32


def test_streaming_ptlx_full_bridge_combines_with_consumer_schedule():
    source = {"mb": 32, "out": 1}
    dest = {"mb": 4, "out": 8}
    summary = plan_streaming_ptlx_tiles(
        size=512,
        source_work_slices=source,
        source_core_mapping=default_core_mapping(source),
        dest_work_slices=dest,
        dest_core_mapping=default_core_mapping(dest),
        sample_limit=2,
    )
    artifact = generate_streaming_ptlx_artifact("streaming", summary, max_tiles=2)
    bridge = generate_streaming_ptlx_full_bridge_sdsc("full_bridge", artifact)
    consumer = {
        "2_add": {
            "numCoresUsed_": 32,
            "opFuncsUsed_": ["add"],
            "dscs_": [{"add": {"computeOp_": [{"opFuncName": "add"}]}}],
            "datadscs_": [],
            "coreIdToDscSchedule": {},
        }
    }

    mixed = _combine_ptlx_bridge_with_consumer("streaming_mixed", bridge, consumer)
    root = mixed["streaming_mixed"]

    assert len(root["datadscs_"]) == 6
    assert len(root["dscs_"]) == 1
    assert root["streamingPTLXFull_"]["tile_count"] == 2
    assert root["coreIdToDscSchedule"]["0"][-1] == [-1, 0, 1, 0]
    assert root["coreIdToDscSchedule"]["1"] == [
        [0, -1, 0, 0],
        [3, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    assert root["coreIdToDscSchedule"]["8"] == [[-1, 0, 0, 0]]


def test_streaming_ptlx_patch_replaces_small_shape_hbm_restickify_boundary():
    producer_payload = _minimal_layout_payload(
        "0_add",
        opfunc="add",
        size=512,
        work_slices={"mb": 32, "out": 1},
        core_mapping=default_core_mapping({"mb": 32, "out": 1}),
        lds=[
            _layout_lds(0, "producer_in", "dataIN", ["mb", "out"], ["out"]),
            _layout_lds(1, "producer_out", "dataOUT", ["mb", "out"], ["out"]),
        ],
        input_indices=[0],
        output_indices=[1],
    )
    restickify_payload = _minimal_layout_payload(
        "1_restickify",
        opfunc="ReStickifyOpHBM",
        size=512,
        work_slices={"mb": 4, "out": 8},
        core_mapping=default_core_mapping({"mb": 4, "out": 8}),
        lds=[
            _layout_lds(0, "producer_out", "dataIN", ["mb", "out"], ["out"]),
            _layout_lds(1, "restickify_out", "dataOUT", ["out", "mb"], ["mb"]),
        ],
        input_indices=[0],
        output_indices=[1],
    )
    consumer_payload = _minimal_layout_payload(
        "2_add",
        opfunc="add",
        size=512,
        work_slices={"mb": 4, "out": 8},
        core_mapping=default_core_mapping({"mb": 4, "out": 8}),
        lds=[
            _layout_lds(0, "restickify_out", "dataIN", ["out", "mb"], ["mb"]),
            _layout_lds(1, "consumer_out", "dataOUT", ["out", "mb"], ["mb"]),
        ],
        input_indices=[0],
        output_indices=[1],
    )
    specs = [
        _minimal_op_spec(
            "add",
            [_arg(True, 0), _arg(False, 4, allocation={"lx": 0})],
        ),
        _minimal_op_spec(
            RESTICKIFY_OP,
            [_arg(True, 4), _arg(False, 5)],
            op_info=_ptlx_restickify_info(
                _endpoint_allocation(0, 256 * 1024, size=64 * 1024)
            ),
        ),
        _minimal_op_spec(
            "add",
            [_arg(True, 5, allocation={"lx": 256 * 1024}), _arg(False, 6)],
        ),
    ]
    payloads = [producer_payload, restickify_payload, consumer_payload]
    plans = plan_restickify_ptlx_mixed_schedules(specs)

    with config.patch(restickify_ptlx_streaming_e2e=True):
        rows = patch_restickify_ptlx_mixed_schedules(payloads, specs, plans=plans)

    patched = rows[0]
    assert patched["status"] == "patched"
    assert patched["kind"] == "ptlx-streaming-mixed-schedule"
    assert patched["trigger_reason"].startswith("ptlx-piece-smaller-than-stick")
    assert patched["streaming_summary"]["total_tiles"] == 64
    assert patched["value_flow_contract"]["valid"] is True
    assert payloads[2] is None
    root = next(iter(payloads[1].values()))
    assert root["streamingPTLXFull_"]["tile_count"] == 64
    assert len(root["datadscs_"]) == 64 * 3
    assert all(
        next(iter(datadsc.values()))["op"]["name"] != "ReStickifyOpHBM"
        for datadsc in root["datadscs_"]
    )
    assert root["coreIdToDscSchedule"]["0"][-1] == [-1, 0, 1, 0]


def test_implicit_alias_streaming_patch_materializes_consumer_input_bridge():
    c0 = Symbol("c0")
    c1 = Symbol("c1")
    source_coords = [c0, c1]
    dest_coords = [c1, c0]
    producer_payload = _minimal_layout_payload(
        "0_add",
        opfunc="add",
        size=512,
        work_slices={"mb": 32, "out": 1},
        core_mapping=default_core_mapping({"mb": 32, "out": 1}),
        lds=[
            _layout_lds(0, "producer_in", "dataIN", ["mb", "out"], ["out"]),
            _layout_lds(1, "producer_out", "dataOUT", ["mb", "out"], ["out"]),
        ],
        input_indices=[0],
        output_indices=[1],
    )
    consumer_payload = _minimal_layout_payload(
        "1_add",
        opfunc="add",
        size=512,
        work_slices={"mb": 1, "out": 32},
        core_mapping=default_core_mapping({"mb": 1, "out": 32}),
        lds=[
            _layout_lds(0, "source_alias", "INPUT0", ["mb", "out"], ["out"]),
            _layout_lds(1, "view_alias", "INPUT1", ["out", "mb"], ["mb"]),
            _layout_lds(2, "consumer_out", "OUTPUT", ["out", "mb"], ["mb"]),
        ],
        input_indices=[0, 1],
        output_indices=[2],
    )
    _patch_consumer_input_lx_map(
        consumer_payload,
        input_name="source_alias",
        lds_idx=0,
        start_payload=_constant_lx_start_payload(num_cores=32, base=0),
    )
    _patch_consumer_input_lx_map(
        consumer_payload,
        input_name="view_alias",
        lds_idx=1,
        start_payload=_constant_lx_start_payload(num_cores=32, base=0),
    )
    specs = [
        _minimal_op_spec(
            "add",
            [
                _arg(True, 0),
                _arg(False, -1, allocation={"lx": 0}, device_coordinates=source_coords),
            ],
        ),
        _minimal_op_spec(
            "add",
            [
                _arg(True, -1, allocation={"lx": 0}, device_coordinates=source_coords),
                _arg(True, -1, allocation={"lx": 0}, device_coordinates=dest_coords),
                _arg(False, 2, device_coordinates=dest_coords),
            ],
        ),
    ]
    payloads = [producer_payload, consumer_payload]

    with config.patch(
        restickify_use_specific_insert=True,
        restickify_ptlx_mixed_schedule_e2e=True,
        restickify_ptlx_streaming_e2e=True,
        restickify_ptlx_value_flow_assert=True,
    ):
        rows = patch_implicit_restickify_ptlx_aliases(payloads, specs)

    assert len(rows) == 1
    patched = rows[0]
    assert patched["status"] == "patched"
    assert patched["kind"] == "ptlx-implicit-alias-producer-streaming"
    assert patched["value_flow_contract"]["valid"] is True
    assert patched["streaming_summary"]["tile_size"] == 64
    assert patched["streaming_summary"]["total_tiles"] == 64
    assert patched["split_bridge_sdsc"] is False
    assert patched["producer_mixed_bridge_sdsc"] is True
    assert patched["plan"]["consumer_input_position"] == 0
    assert patched["consumer_lx_unique_starts"] != [0]

    root = next(iter(payloads[0].values()))
    assert len(payloads) == 2
    assert "ImplicitAliasProducerStreamingReStickifyOpWithPTLx" in next(
        iter(payloads[0])
    )
    assert root["streamingPTLXFull_"]["tile_count"] == 64
    assert root["dscs_"]
    assert root["datadscs_"]
    assert {"add", "ReStickifyOpWithPTLx", "STCDPOpLx"} <= set(
        root["opFuncsUsed_"]
    )
    assert root["coreIdToDscSchedule"]["0"][0] == [-1, 0, 0, 0]
    assert all(
        next(iter(datadsc.values()))["op"]["name"] != "ReStickifyOpHBM"
        for datadsc in root["datadscs_"]
    )
    assert next(iter(payloads[1])) == "1_add"


def test_streaming_ptlx_cross_bundle_patch_rewrites_handoff_pair():
    producer_payload = _minimal_layout_payload(
        "0_add",
        opfunc="add",
        size=512,
        work_slices={"mb": 1, "out": 32},
        core_mapping=default_core_mapping({"mb": 1, "out": 32}),
        lds=[
            _layout_lds(0, "producer_in", "dataIN", ["mb", "out"], ["out"]),
            _layout_lds(1, "producer_out", "dataOUT", ["mb", "out"], ["out"]),
        ],
        input_indices=[0],
        output_indices=[1],
    )
    restickify_payload = _minimal_layout_payload(
        "1_restickify",
        opfunc="ReStickifyOpHBM",
        size=512,
        work_slices={"mb": 8, "out": 4},
        core_mapping=default_core_mapping({"mb": 8, "out": 4}),
        lds=[
            _layout_lds(0, "producer_out", "dataIN", ["mb", "out"], ["out"]),
            _layout_lds(1, "restickify_out", "dataOUT", ["out", "mb"], ["mb"]),
        ],
        input_indices=[0],
        output_indices=[1],
    )
    consumer_payload = _minimal_layout_payload(
        "0_batchmatmul",
        opfunc="batchmatmul",
        size=512,
        work_slices={"mb": 32, "out": 1},
        core_mapping=default_core_mapping({"mb": 32, "out": 1}),
        lds=[
            _layout_lds(0, "restickify_out", "dataIN", ["out", "mb"], ["mb"]),
            _layout_lds(1, "weight", "KERNEL", ["out", "mb"], ["mb"]),
            _layout_lds(2, "consumer_out", "dataOUT", ["out", "mb"], ["mb"]),
        ],
        input_indices=[0, 1],
        output_indices=[2],
    )
    endpoint_allocation = _endpoint_allocation(0, 256 * 1024, size=64 * 1024)
    c0 = Symbol("c0")
    c1 = Symbol("c1")
    c2 = Symbol("c2")
    restickify_device_size = [8, 512, 64]
    restickify_output_coords = [c1, c0, c1]
    left_specs = [
        _minimal_op_spec(
            "add",
            [_arg(True, 0), _arg(False, 5, allocation={"lx": 0})],
        ),
        _minimal_op_spec(
            RESTICKIFY_OP,
            [
                _arg(True, 5),
                _arg(
                    False,
                    6,
                    device_size=restickify_device_size,
                    device_coordinates=restickify_output_coords,
                ),
            ],
            op_info=_ptlx_restickify_info(endpoint_allocation),
        ),
    ]
    right_specs = [
        _minimal_op_spec(
            "batchmatmul",
            [
                _arg(
                    True,
                    0,
                    allocation={"lx": 256 * 1024},
                    device_size=restickify_device_size,
                    device_coordinates=[c2, c0, c2],
                ),
                _arg(
                    True,
                    1,
                    device_size=restickify_device_size,
                    device_coordinates=[c1, c2, c1],
                ),
                _arg(False, 2),
            ],
        )
    ]
    records = [
        {
            "kernel_name": "producer_bundle",
            "specs": left_specs,
            "sdscs_json": [producer_payload, restickify_payload],
        },
        {
            "kernel_name": "consumer_bundle",
            "specs": right_specs,
            "sdscs_json": [consumer_payload],
        },
    ]

    with config.patch(
        restickify_ptlx_cross_bundle_e2e=True,
        restickify_ptlx_streaming_e2e=True,
    ):
        rows = patch_restickify_ptlx_cross_bundle_handoffs(records)

    assert len(rows) == 1
    patched = rows[0]
    assert patched["status"] == "patched"
    assert patched["kind"] == "ptlx-streaming-cross-bundle-handoff"
    assert patched["value_flow_contract"]["valid"] is True
    assert patched["streaming_summary"]["total_tiles"] == 64
    assert records[0]["sdscs_json"][1] is None
    bridge_root = next(iter(records[0]["sdscs_json"][0].values()))
    assert "CrossBundleProducerStreamingReStickifyOpWithPTLx" in next(
        iter(records[0]["sdscs_json"][0])
    )
    assert bridge_root["streamingPTLXFull_"]["tile_count"] == 64
    assert len(bridge_root["datadscs_"]) == 64 * 3
    assert bridge_root["dscs_"]
    assert bridge_root["coreIdToDscSchedule"]["0"][0] == [-1, 0, 0, 0]
    consumer_root = next(iter(records[1]["sdscs_json"][0].values()))
    consumer_dsc = next(iter(consumer_root["dscs_"][0].values()))
    allocate_nodes = [
        node
        for node in consumer_dsc["scheduleTree_"]
        if node["nodeType_"] == "allocate"
    ]
    assert allocate_nodes[0]["component_"] == "lx"
    assert allocate_nodes[1]["component_"] == "hbm"


def test_ptlx_bridge_accepts_stock_mixed_restickify_split():
    bridge = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=512,
        num_cores=32,
        mode="stage3b",
        direction="kernel-to-output",
        restickify_op_name="ReStickifyOpWithPTLx",
        input_work_slices={"mb": 32, "out": 1},
        input_core_to_work_slice={
            str(core): {"mb": core, "out": 0} for core in range(32)
        },
        intermediate_work_slices={"mb": 4, "out": 8},
        intermediate_core_to_work_slice={
            str(core): {"mb": core % 4, "out": core // 4}
            for core in range(32)
        },
        output_work_slices={"mb": 32, "out": 1},
        output_core_to_work_slice={
            str(core): {"mb": core, "out": 0} for core in range(32)
        },
    )

    root = bridge["ptlx_bridge"]
    first = next(iter(root["datadscs_"][0].values()))
    output_piece = first["labeledDs_"][1]["PieceInfo"][0]

    assert output_piece["dimToSize_"] == {"out_": 64, "mb_": 128}
    assert output_piece["dimToStartCordinate"] == {"out_": 0, "mb_": 0}


def test_ptlx_bridge_accepts_output_to_kernel_direction():
    bridge = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=2048,
        num_cores=32,
        mode="baseline",
        direction="output-to-kernel",
        restickify_op_name="ReStickifyOpWithPTLx",
        input_work_slices={"mb": 1, "out": 32},
        input_core_to_work_slice={
            str(core): {"mb": 0, "out": core} for core in range(32)
        },
        intermediate_work_slices={"mb": 32, "out": 1},
        intermediate_core_to_work_slice={
            str(core): {"mb": core, "out": 0} for core in range(32)
        },
        output_work_slices={"mb": 32, "out": 1},
        output_core_to_work_slice={
            str(core): {"mb": core, "out": 0} for core in range(32)
        },
    )

    first = _dataop_at(bridge, 0)
    second = _dataop_at(bridge, 1)

    assert first["labeledDs_"][0]["layoutDimOrder_"] == ["mb_", "out_"]
    assert first["labeledDs_"][0]["stickDimOrder_"] == ["out_"]
    assert first["labeledDs_"][1]["layoutDimOrder_"] == ["out_", "mb_"]
    assert first["labeledDs_"][1]["stickDimOrder_"] == ["mb_"]
    intermediate_piece = first["labeledDs_"][1]["PieceInfo"][0]
    assert intermediate_piece["dimToSize_"] == {"out_": 2048, "mb_": 64}
    assert second["labeledDs_"][0]["layoutDimOrder_"] == ["out_", "mb_"]
    assert second["labeledDs_"][1]["layoutDimOrder_"] == ["out_", "mb_"]
    # The bridge uses a synthetic dimension alias: out_ corresponds to the
    # consumer row/reduction dimension, so the final split is still core-local.
    output_piece = second["labeledDs_"][1]["PieceInfo"][0]
    assert output_piece["dimToStartCordinate"] == {"out_": 0, "mb_": 0}
    assert output_piece["dimToSize_"] == {"out_": 64, "mb_": 2048}


def test_ptlx_bridge_uses_planned_intermediate_lx_start():
    bridge = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=128,
        num_cores=2,
        mode="stage3b",
        direction="kernel-to-output",
        input_start_address=0,
        intermediate_start_address=128 * 1024,
        output_start_address=256 * 1024,
        restickify_op_name="ReStickifyOpWithPTLx",
    )

    first = _dataop_at(bridge, 0)
    second = _dataop_at(bridge, 1)
    first_output = first["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"]
    second_input = second["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"]

    assert first_output == [{"type": "lx", "memId": [0], "startAddr": [128 * 1024]}]
    assert second_input == [{"type": "lx", "memId": [0], "startAddr": [128 * 1024]}]


def test_plan_mixed_ptlx_schedule_from_opspecs():
    specs = [
        _minimal_op_spec(
            "add",
            [_arg(True, 0), _arg(False, 4, allocation={"lx": 64 * 1024})],
        ),
        _minimal_op_spec(
            RESTICKIFY_OP,
            [_arg(True, 4), _arg(False, 5)],
            op_info=_certified_restickify_info(
                _endpoint_allocation(64 * 1024, 96 * 1024)
            ),
        ),
        _minimal_op_spec(
            "add",
            [_arg(True, 5, allocation={"lx": 96 * 1024}), _arg(False, 6)],
        ),
    ]

    plans = plan_restickify_ptlx_mixed_schedules(specs)

    plan = plans[1]
    assert plan.sdsc_index == 1
    assert plan.producer_index == 0
    assert plan.consumer_index == 2
    assert plan.producer_lds_idx == 1
    assert plan.consumer_lds_idx == 0
    assert plan.producer_arg_index == 4
    assert plan.consumer_arg_index == 5
    assert plan.producer_endpoint.role == "producer_output"
    assert plan.producer_endpoint.is_input is False
    assert plan.producer_endpoint.sdsc_index == 0
    assert plan.producer_endpoint.base == 64 * 1024
    assert plan.producer_endpoint.base_source == "op-spec-allocation"
    assert plan.consumer_endpoint.role == "consumer_input"
    assert plan.consumer_endpoint.is_input is True
    assert plan.consumer_endpoint.sdsc_index == 2
    assert plan.consumer_endpoint.base == 96 * 1024
    assert plan.consumer_endpoint.base_source == "op-spec-allocation"


def test_plan_mixed_ptlx_schedule_uses_op_spec_lx_allocations():
    specs = [
        _minimal_op_spec(
            "add",
            [_arg(True, 0), _arg(False, 4, allocation={"lx": 64 * 1024})],
        ),
        _minimal_op_spec(
            RESTICKIFY_OP,
            [_arg(True, 4), _arg(False, 5)],
            op_info=_certified_restickify_info(
                _endpoint_allocation(64 * 1024, 96 * 1024)
            ),
        ),
        _minimal_op_spec(
            "add",
            [_arg(True, 5, allocation={"lx": 96 * 1024}), _arg(False, 6)],
        ),
    ]

    plan = plan_restickify_ptlx_mixed_schedules(specs)[1]

    assert plan.producer_endpoint.base == 64 * 1024
    assert plan.producer_endpoint.base_source == "op-spec-allocation"
    assert plan.consumer_endpoint.base == 96 * 1024
    assert plan.consumer_endpoint.base_source == "op-spec-allocation"


def test_endpoint_core_starts_come_from_endpoint_plan():
    endpoint = plan_restickify_ptlx_mixed_schedules(
        [
            _minimal_op_spec(
                "add",
                [_arg(True, 0), _arg(False, 4, allocation={"lx": 16 * 1024})],
            ),
            _minimal_op_spec(
                RESTICKIFY_OP,
                [_arg(True, 4), _arg(False, 5)],
                op_info=_certified_restickify_info(
                    _endpoint_allocation(16 * 1024, 8 * 1024)
                ),
            ),
            _minimal_op_spec(
                "add",
                [_arg(True, 5, allocation={"lx": 8 * 1024}), _arg(False, 6)],
            ),
        ]
    )[1].producer_endpoint

    assert _endpoint_core_starts(endpoint, num_cores=3) == {
        0: 16 * 1024,
        1: 16 * 1024,
        2: 16 * 1024,
    }


def test_plan_mixed_ptlx_schedule_skips_non_in_graph_restickify():
    specs = [
        _minimal_op_spec("add", [_arg(True, 0), _arg(False, 4)]),
        _minimal_op_spec(RESTICKIFY_OP, [_arg(True, 4), _arg(False, 5)]),
        _minimal_op_spec("add", [_arg(True, 5), _arg(False, 6)]),
    ]

    plans = plan_restickify_ptlx_mixed_schedules(specs)

    assert plans == {}


def test_plan_mixed_ptlx_schedule_allows_uncertified_in_graph_restickify():
    specs = [
        _minimal_op_spec(
            "add",
            [_arg(True, 0), _arg(False, 4, allocation={"lx": 64 * 1024})],
        ),
        _minimal_op_spec(
            RESTICKIFY_OP,
            [_arg(True, 4), _arg(False, 5)],
            op_info=_ptlx_restickify_info(
                _endpoint_allocation(64 * 1024, 96 * 1024)
            ),
        ),
        _minimal_op_spec(
            "add",
            [_arg(True, 5, allocation={"lx": 96 * 1024}), _arg(False, 6)],
        ),
    ]

    plan = plan_restickify_ptlx_mixed_schedules(specs)[1]

    assert plan.producer_endpoint.base == 64 * 1024
    assert plan.consumer_endpoint.base == 96 * 1024


def test_plan_mixed_ptlx_schedule_skips_without_allocator_endpoints():
    specs = [
        _minimal_op_spec("add", [_arg(True, 0), _arg(False, 4)]),
        _minimal_op_spec(
            RESTICKIFY_OP,
            [_arg(True, 4), _arg(False, 5)],
            op_info=_ptlx_restickify_info(),
        ),
        _minimal_op_spec("add", [_arg(True, 5), _arg(False, 6)]),
    ]

    plans = plan_restickify_ptlx_mixed_schedules(specs)

    assert plans == {}


def test_plan_mixed_ptlx_schedule_skips_invalid_endpoint_overlap():
    specs = [
        _minimal_op_spec(
            "add",
            [_arg(True, 0), _arg(False, 4, allocation={"lx": 64 * 1024})],
        ),
        _minimal_op_spec(
            RESTICKIFY_OP,
            [_arg(True, 4), _arg(False, 5)],
            op_info=_certified_restickify_info(
                _endpoint_allocation(64 * 1024, 96 * 1024, valid=False)
            ),
        ),
        _minimal_op_spec(
            "add",
            [_arg(True, 5, allocation={"lx": 96 * 1024}), _arg(False, 6)],
        ),
    ]

    plans = plan_restickify_ptlx_mixed_schedules(specs)

    assert plans == {}


def test_mixed_ptlx_value_flow_contract_matches_bridge_endpoints():
    producer = _minimal_compute_payload("0_add", "producer_out", lds_idx=1)
    consumer = _minimal_compute_payload("2_add", "consumer_in", lds_idx=0)
    producer_start = _constant_lx_start_payload(num_cores=2, base=16 * 1024)
    consumer_start = _constant_lx_start_payload(num_cores=2, base=8 * 1024)
    _patch_lx_allocation_by_index(
        producer,
        lds_idx=1,
        start_payload=producer_start,
    )
    _patch_consumer_input_lx_map(
        consumer,
        input_name="consumer_in",
        lds_idx=0,
        start_payload=consumer_start,
    )
    bridge = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=128,
        num_cores=2,
        mode="stage3b",
        direction="kernel-to-output",
        restickify_op_name="ReStickifyOpWithPTLx",
        input_start_address=16 * 1024,
        output_start_address=8 * 1024,
    )
    _patch_bridge_endpoint_pieces(
        bridge,
        producer_starts={0: 16 * 1024, 1: 16 * 1024},
        consumer_starts={0: 8 * 1024, 1: 8 * 1024},
    )

    contract = _mixed_value_flow_contract(
        producer_payload=producer,
        bridge_payload=bridge,
        consumer_payload=consumer,
        producer_lds_idx=1,
        consumer_lds_idx=0,
    )

    assert contract["valid"] is True
    assert contract["producer_to_bridge_input_match"] is True
    assert contract["bridge_output_to_consumer_match"] is True
    assert contract["producer_unique_starts"] == [16 * 1024]
    assert contract["consumer_unique_starts"] == [8 * 1024]


def test_materialize_bridge_lx_endpoints_uses_planned_endpoints():
    plan = plan_restickify_ptlx_mixed_schedules(
        [
            _minimal_op_spec(
                "add",
                [_arg(True, 0), _arg(False, 4, allocation={"lx": 16 * 1024})],
            ),
            _minimal_op_spec(
                RESTICKIFY_OP,
                [_arg(True, 4), _arg(False, 5)],
                op_info=_certified_restickify_info(
                    _endpoint_allocation(16 * 1024, 8 * 1024)
                ),
            ),
            _minimal_op_spec(
                "add",
                [_arg(True, 5, allocation={"lx": 8 * 1024}), _arg(False, 6)],
            ),
        ]
    )[1]
    bridge = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=128,
        num_cores=2,
        mode="stage3b",
        direction="kernel-to-output",
        restickify_op_name="ReStickifyOpWithPTLx",
        input_start_address=plan.producer_endpoint.base,
        output_start_address=plan.consumer_endpoint.base,
    )

    patch = _materialize_bridge_lx_endpoints(bridge, plan=plan, num_cores=2)

    assert patch["producer_pieces_patched"] == 2
    assert patch["consumer_pieces_patched"] == 2
    root = next(iter(bridge.values()))
    first = next(iter(root["datadscs_"][0].values()))
    last = next(iter(root["datadscs_"][-1].values()))
    assert {
        piece["PlacementInfo"][0]["startAddr"][0]
        for piece in first["labeledDs_"][0]["PieceInfo"]
    } == {16 * 1024}
    assert {
        piece["PlacementInfo"][0]["startAddr"][0]
        for piece in last["labeledDs_"][-1]["PieceInfo"]
    } == {8 * 1024}


def test_mixed_ptlx_value_flow_contract_catches_bridge_mismatch():
    producer = _minimal_compute_payload("0_add", "producer_out", lds_idx=1)
    consumer = _minimal_compute_payload("2_add", "consumer_in", lds_idx=0)
    producer_start = _constant_lx_start_payload(num_cores=2, base=16 * 1024)
    consumer_start = _constant_lx_start_payload(num_cores=2, base=8 * 1024)
    _patch_lx_allocation_by_index(
        producer,
        lds_idx=1,
        start_payload=producer_start,
    )
    _patch_consumer_input_lx_map(
        consumer,
        input_name="consumer_in",
        lds_idx=0,
        start_payload=consumer_start,
    )
    bridge = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=128,
        num_cores=2,
        mode="stage3b",
        direction="kernel-to-output",
        restickify_op_name="ReStickifyOpWithPTLx",
        input_start_address=16 * 1024,
        output_start_address=8 * 1024,
    )
    _patch_bridge_endpoint_pieces(
        bridge,
        producer_starts={0: 1234, 1: 16 * 1024},
        consumer_starts={0: 8 * 1024, 1: 8 * 1024},
    )

    contract = _mixed_value_flow_contract(
        producer_payload=producer,
        bridge_payload=bridge,
        consumer_payload=consumer,
        producer_lds_idx=1,
        consumer_lds_idx=0,
    )

    assert contract["valid"] is False
    assert contract["producer_to_bridge_input_match"] is False
    assert contract["bridge_output_to_consumer_match"] is True


def _minimal_compute_payload(name: str, lds_name: str, *, lds_idx: int):
    return {
        name: {
            "numCoresUsed_": 2,
            "opFuncsUsed_": ["add"],
            "dscs_": [
                {
                    "add": {
                        "numCoreletsUsed_": 1,
                        "numCoreletsUsed_DSC2_": 1,
                        "labeledDs_": [
                            {
                                "ldsIdx_": lds_idx,
                                "dsName_": lds_name,
                                "memOrg_": {"hbm": {"isPresent": 1}},
                                "hbmStartAddress_": 0,
                                "hbmSize_": 256,
                                "lxSize_": 0,
                                "lxBufferSize_": 0,
                            }
                        ],
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "ldsIdx_": lds_idx,
                                "name_": f"allocate-{lds_name}_hbm",
                                "component_": "hbm",
                            }
                        ],
                        "computeOp_": [{"opFuncName": "add"}],
                    }
                }
            ],
            "datadscs_": [],
            "coreIdToDscSchedule": {},
        }
    }


def _layout_lds(
    lds_idx: int,
    name: str,
    ds_type: str,
    layout: list[str],
    stick: list[str],
) -> dict:
    return {
        "ldsIdx_": lds_idx,
        "dsName_": name,
        "dsType_": ds_type,
        "layoutDimOrder_": layout,
        "stickDimOrder_": stick,
        "memOrg_": {"hbm": {"isPresent": 1}},
        "hbmStartAddress_": 0,
        "hbmSize_": 512 * 512 * 2,
        "lxSize_": 0,
        "lxBufferSize_": 0,
    }


def _minimal_layout_payload(
    name: str,
    *,
    opfunc: str,
    size: int,
    work_slices: dict[str, int],
    core_mapping: dict[str, dict[str, int]],
    lds: list[dict],
    input_indices: list[int],
    output_indices: list[int],
) -> dict:
    dsc = {
        "numCoreletsUsed_": 1,
        "numCoreletsUsed_DSC2_": 1,
        "N_": {"name_": "n", "mb_": size, "out_": size},
        "labeledDs_": lds,
        "primaryDsInfo_": {
            lds_item["dsType_"]: {
                "layoutDimOrder_": lds_item["layoutDimOrder_"],
                "stickDimOrder_": lds_item["stickDimOrder_"],
            }
            for lds_item in lds
        },
        "scheduleTree_": [
            {
                "nodeType_": "allocate",
                "ldsIdx_": lds_item["ldsIdx_"],
                "name_": f"allocate-{lds_item['dsName_']}_hbm",
                "component_": "hbm",
            }
            for lds_item in lds
        ],
        "computeOp_": [
            {
                "opFuncName": opfunc,
                "inputLabeledDs": [
                    f"dataIN_L{idx}-idx{idx}" for idx in input_indices
                ],
                "outputLabeledDs": [
                    f"dataOUT_L{idx}-idx{idx}" for idx in output_indices
                ],
            }
        ],
    }
    return {
        name: {
            "numCoresUsed_": 32,
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numWkSlicesPerDim_": work_slices,
            "coreIdToWkSlice_": core_mapping,
            "opFuncsUsed_": [opfunc],
            "dscs_": [{opfunc: dsc}],
            "datadscs_": [],
            "coreIdToDscSchedule": {},
        }
    }


def _arg(
    is_input: bool,
    arg_index: int,
    *,
    allocation: dict | None = None,
    device_size: list | None = None,
    device_coordinates: list | None = None,
) -> TensorArg:
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=device_size or [],
        device_coordinates=device_coordinates or [],
        allocation=allocation,
    )


def _minimal_op_spec(
    op: str,
    args: list[TensorArg],
    *,
    op_info: dict | None = None,
) -> OpSpec:
    return OpSpec(
        op=op,
        is_reduction=False,
        iteration_space={},
        args=args,
        op_info=op_info or {},
    )


def _endpoint_allocation(
    producer_base: int,
    consumer_base: int,
    *,
    valid: bool = True,
    size: int = 128,
) -> dict:
    return {
        "kind": "ptlx_endpoint_allocation",
        "producer_buffer": "producer",
        "consumer_buffer": "consumer",
        "producer": {
            "buffer": "producer",
            "start": producer_base,
            "size": size,
            "end": producer_base + size,
        },
        "consumer": {
            "buffer": "consumer",
            "start": consumer_base,
            "size": size,
            "end": consumer_base + size,
        },
        "overlap_check": {
            "valid": valid,
            "overlaps": [] if valid else [{"endpoint": "producer"}],
        },
    }


def _certified_restickify_info(endpoint_allocation: dict | None = None) -> dict:
    info = _ptlx_restickify_info(endpoint_allocation)
    info.update(
        {
            CORE_MAPPING_OVERRIDE_OP_INFO_KEY: {"0": {}},
            LOCALITY_CERTIFICATE_OP_INFO_KEY: {
                "locality_certified": True,
                "certified_byte_hops": 0,
            },
        }
    )
    return info


def _ptlx_restickify_info(endpoint_allocation: dict | None = None) -> dict:
    info = {
        "restickify_source_kind": "in_graph_computed",
    }
    if endpoint_allocation is not None:
        info[PTLX_ENDPOINT_ALLOCATION_OP_INFO_KEY] = endpoint_allocation
    return info
