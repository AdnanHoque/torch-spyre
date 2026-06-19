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

import dataclasses

import sympy

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config as spyre_config
from torch_spyre._inductor.codegen.onchip_move import (
    _validate_lx_regions,
    build_mixed_onchip_move_sdsc,
    build_stcdp_datadsc,
    diagnose_stcdp_output_layout_contiguity,
    patch_onchip_move_mixed_schedules,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.onchip_move import (
    ONCHIP_MOVE_ATTR,
    ONCHIP_MOVE_OP_INFO_KEY,
    OnChipMovePlan,
    _attach_plan_to_consumer,
    build_onchip_move_cells,
)
from torch_spyre._inductor.pass_utils import PerCoreView


def test_onchip_move_cells_cover_2d_matmul_to_pure_m_reshard():
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 4), (1, 8)),
        core_to_slot=((0, sympy.Mod(core_id, 4)), (1, sympy.floor(core_id / 4))),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 32),),
        core_to_slot=((0, sympy.Mod(core_id, 32)),),
    )

    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[512, 12800],
        element_bytes=2,
        producer_core_count=32,
        consumer_core_count=32,
    )

    assert reason is None
    assert len(cells) == 256
    assert sum(cell.bytes for cell in cells) == 512 * 12800 * 2
    assert cells[0].source_core == 0
    assert cells[0].dest_core == 0
    assert cells[0].dim_starts == {"d0_": 0, "d1_": 0}
    assert cells[0].dim_sizes == {"d0_": 16, "d1_": 1600}
    assert cells[0].source_offset_bytes == 0
    assert cells[0].dest_offset_bytes == 0
    assert cells[1].source_core == 4
    assert cells[1].dest_core == 0
    assert cells[1].source_offset_bytes == 0
    assert cells[1].dest_offset_bytes == 16 * 1600 * 2
    assert cells[8].source_core == 0
    assert cells[8].dest_core == 1
    assert cells[8].source_offset_bytes == 16 * 2
    assert cells[8].dest_offset_bytes == 0
    assert cells[64].source_core == 1
    assert cells[64].dest_core == 8
    assert cells[64].source_offset_bytes == 0
    assert cells[64].dest_offset_bytes == 0


def test_onchip_move_cells_reject_ambiguous_fanout_owner():
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(work_slice_dims=(), core_to_slot=())
    consumer_view = PerCoreView(
        work_slice_dims=((0, 2),),
        core_to_slot=((0, sympy.Mod(core_id, 2)),),
    )

    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[128, 64],
        element_bytes=2,
        producer_core_count=2,
        consumer_core_count=2,
    )

    assert cells == []
    assert reason == "producer-duplicate-owner"


def test_stcdp_datadsc_uses_plan_cells_as_lx_piece_info():
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 2),),
        core_to_slot=((0, sympy.Mod(core_id, 2)),),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((1, 2),),
        core_to_slot=((1, sympy.Mod(core_id, 2)),),
    )
    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[128, 64],
        element_bytes=2,
        producer_core_count=2,
        consumer_core_count=2,
    )
    assert reason is None
    plan = {
        "device_sizes": [128, 64],
        "cells": [dataclasses.asdict(cell) for cell in cells],
    }

    dataop = build_stcdp_datadsc(
        "0_OnChipMoveSTCDPOpLx",
        plan,
        data_format="SEN169_FP16",
        word_length=2,
        producer_base=16 * 1024,
        consumer_base=8 * 1024,
    )

    assert dataop["op"]["name"] == "STCDPOpLx"
    assert dataop["coreIdsUsed_"] == [0, 1]
    assert dataop["labeledDs_"][0]["PieceInfo"][0]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [16 * 1024]}
    ]
    assert dataop["labeledDs_"][1]["PieceInfo"][0]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [8 * 1024]}
    ]
    assert dataop["labeledDs_"][0]["PieceInfo"][1]["PlacementInfo"] == [
        {"type": "lx", "memId": [0], "startAddr": [16 * 1024 + 4096]}
    ]
    assert dataop["labeledDs_"][1]["PieceInfo"][1]["PlacementInfo"] == [
        {"type": "lx", "memId": [1], "startAddr": [8 * 1024]}
    ]
    assert len(dataop["labeledDs_"][0]["PieceInfo"]) == 4


def test_mixed_carrier_keeps_producer_as_standalone_sdsc():
    plan = {
        "source_name": "buf0",
        "producer": "buf0",
        "consumer": "buf1",
        "device_sizes": [32, 64],
        "producer_region_bytes": 4096,
        "consumer_region_bytes": 4096,
        "bytes_moved": 4096,
        "cell_count": 1,
        "cells": [
            {
                "cell_index": 0,
                "source_core": 0,
                "dest_core": 0,
                "source_offset_bytes": 0,
                "dest_offset_bytes": 0,
                "bytes": 4096,
                "dim_starts": {"d0_": 0, "d1_": 0},
                "dim_sizes": {"d0_": 32, "d1_": 64},
            }
        ],
    }
    producer_payload = {
        "1_batchmatmul": _minimal_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=[0],
        )
    }
    consumer_payload = {
        "2_neg": _minimal_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=[0],
        )
    }
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[32, 64],
        device_coordinates=[],
        allocation=None,
        name="buf0",
    )

    patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
        1,
        2,
        producer_payload,
        consumer_payload,
        output_arg,
        dataclasses.replace(output_arg, is_input=True, arg_index=0),
        2,
        0,
        plan,
    )

    producer_root = patched_producer["1_batchmatmul"]
    producer_dsc = next(iter(producer_root["dscs_"][0].values()))
    assert producer_dsc["labeledDs_"][2]["memOrg_"] == {"lx": {"isPresent": 1}}
    assert len(producer_root["dscs_"]) == 1

    mixed_root = mixed_consumer["2_OnChipMoveMixedSTCDP"]
    assert len(mixed_root["dscs_"]) == 1
    assert len(mixed_root["datadscs_"]) == 1
    assert mixed_root["coreIdToDscSchedule"]["0"] == [
        [0, -1, 0, 1],
        [-1, 0, 1, 0],
    ]
    consumer_dsc = next(iter(mixed_root["dscs_"][0].values()))
    assert consumer_dsc["labeledDs_"][0]["memOrg_"] == {"lx": {"isPresent": 1}}


def test_mixed_carrier_emits_logical_dataop_layout_for_stickified_cells():
    plan = {
        "source_name": "buf0",
        "producer": "buf0",
        "consumer": "buf1",
        "device_sizes": [8, 512, 64],
        "device_stride_map": [64, 512, 1],
        "producer_region_bytes": 16 * 1024,
        "consumer_region_bytes": 16 * 1024,
        "bytes_moved": 3 * 2048,
        "cell_count": 3,
        "cells": [
            {
                "cell_index": 0,
                "source_core": 0,
                "dest_core": 0,
                "source_offset_bytes": 0,
                "dest_offset_bytes": 0,
                "bytes": 2048,
                "dim_starts": {"d0_": 0, "d1_": 0, "d2_": 0},
                "dim_sizes": {"d0_": 1, "d1_": 16, "d2_": 64},
            },
            {
                "cell_index": 1,
                "source_core": 0,
                "dest_core": 1,
                "source_offset_bytes": 32,
                "dest_offset_bytes": 0,
                "bytes": 2048,
                "dim_starts": {"d0_": 0, "d1_": 16, "d2_": 0},
                "dim_sizes": {"d0_": 1, "d1_": 16, "d2_": 64},
            },
            {
                "cell_index": 2,
                "source_core": 4,
                "dest_core": 0,
                "source_offset_bytes": 0,
                "dest_offset_bytes": 2,
                "bytes": 2048,
                "dim_starts": {"d0_": 1, "d1_": 0, "d2_": 0},
                "dim_sizes": {"d0_": 1, "d1_": 16, "d2_": 64},
            },
        ],
    }
    producer_payload = {
        "0_add": _layout_sdsc_payload(
            "add",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=list(range(32)),
        )
    }
    consumer_payload = {
        "1_neg": _layout_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=list(range(32)),
        )
    }
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 512, 64],
        device_coordinates=[],
        allocation=None,
        stride_map=[64, 512, 64],
        name="buf0",
    )

    _patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
        0,
        1,
        producer_payload,
        consumer_payload,
        output_arg,
        dataclasses.replace(output_arg, is_input=True, arg_index=0),
        1,
        0,
        plan,
    )

    dataop = mixed_consumer["1_OnChipMoveMixedSTCDP"]["datadscs_"][0][
        "0_OnChipMoveSTCDPOpLx"
    ]
    input_lds = dataop["labeledDs_"][0]
    assert input_lds["layoutDimOrder_"] == ["mb", "out"]
    assert input_lds["stickDimOrder_"] == ["out"]
    assert input_lds["dimToLayoutSize_"] == {"mb": 512, "out": 512}
    assert input_lds["dimToStickSize_"] == {"out": 64}
    pieces = input_lds["PieceInfo"]
    assert len(pieces) == 3
    assert pieces[0]["dimToStartCordinate"] == {"mb": 0, "out": 0}
    assert pieces[0]["dimToSize_"] == {"mb": 16, "out": 64}
    assert pieces[1]["dimToStartCordinate"] == {"mb": 16, "out": 0}
    assert pieces[1]["dimToSize_"] == {"mb": 16, "out": 64}
    assert pieces[2]["dimToStartCordinate"] == {"mb": 0, "out": 64}
    assert pieces[2]["dimToSize_"] == {"mb": 16, "out": 64}
    output_lds = dataop["labeledDs_"][1]
    output_pieces = output_lds["PieceInfo"]
    assert len(output_pieces) == 3
    assert output_pieces[0]["dimToStartCordinate"] == {"mb": 0, "out": 0}
    assert output_pieces[0]["dimToSize_"] == {"mb": 16, "out": 512}
    assert output_pieces[0]["validGap_"]["out"] == [[64, 448]]
    assert output_pieces[1]["dimToStartCordinate"] == {"mb": 16, "out": 0}
    assert output_pieces[1]["dimToSize_"] == {"mb": 16, "out": 512}
    assert output_pieces[1]["validGap_"]["out"] == [[64, 448]]
    assert output_pieces[2]["dimToStartCordinate"] == {"mb": 0, "out": 0}
    assert output_pieces[2]["dimToSize_"] == {"mb": 16, "out": 512}
    assert output_pieces[2]["validGap_"]["out"] == [[0, 64], [64, 384]]


def test_mixed_carrier_maps_collapsed_size_one_logical_dims():
    plan = {
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
    producer_payload = {
        "0_batchmatmul": _layout_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=list(range(32)),
            include_x=True,
        )
    }
    consumer_payload = {
        "1_neg": _layout_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=list(range(32)),
            input_layout_dim_order=["out", "mb"],
            output_layout_dim_order=["out", "mb"],
        )
    }
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 8, 1, 64],
        device_coordinates=[],
        allocation=None,
        stride_map=[512, 64, -1, 64],
        name="buf0",
    )

    _patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
        0,
        1,
        producer_payload,
        consumer_payload,
        output_arg,
        dataclasses.replace(output_arg, is_input=True, arg_index=0),
        2,
        0,
        plan,
    )

    dataop = mixed_consumer["1_OnChipMoveMixedSTCDP"]["datadscs_"][0][
        "0_OnChipMoveSTCDPOpLx"
    ]
    input_lds = dataop["labeledDs_"][0]
    assert input_lds["layoutDimOrder_"] == ["mb", "out", "x"]
    assert input_lds["stickDimOrder_"] == ["out"]
    assert input_lds["dimToLayoutSize_"] == {"mb": 512, "out": 512, "x": 1}
    assert input_lds["dimToStickSize_"] == {"out": 64}
    assert len(input_lds["PieceInfo"]) == 1
    piece = input_lds["PieceInfo"][0]
    assert piece["dimToStartCordinate"] == {"mb": 0, "out": 64, "x": 0}
    assert piece["dimToSize_"] == {"mb": 16, "out": 64, "x": 1}
    output_lds = dataop["labeledDs_"][1]
    assert output_lds["layoutDimOrder_"] == ["out", "mb", "x"]
    assert output_lds["stickDimOrder_"] == ["out"]
    assert output_lds["dimToLayoutSize_"] == {"out": 512, "mb": 512, "x": 1}
    assert output_lds["dimToStickSize_"] == {"out": 64}
    assert len(output_lds["PieceInfo"]) == 1
    output_piece = output_lds["PieceInfo"][0]
    assert output_piece["dimToStartCordinate"] == {"out": 0, "mb": 0, "x": 0}
    assert output_piece["dimToSize_"] == {"out": 512, "mb": 16, "x": 1}
    assert output_piece["validGap_"]["out"] == [[0, 64], [64, 384]]


def test_mixed_carrier_dense_actual_output_piece_mode(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_output_piece_mode", "dense_actual")
    plan = {
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
    producer_payload = {
        "0_batchmatmul": _layout_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=list(range(32)),
            include_x=True,
        )
    }
    consumer_payload = {
        "1_neg": _layout_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=list(range(32)),
            input_layout_dim_order=["out", "mb"],
            output_layout_dim_order=["out", "mb"],
        )
    }
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 8, 1, 64],
        device_coordinates=[],
        allocation=None,
        stride_map=[512, 64, -1, 64],
        name="buf0",
    )

    _patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
        0,
        1,
        producer_payload,
        consumer_payload,
        output_arg,
        dataclasses.replace(output_arg, is_input=True, arg_index=0),
        2,
        0,
        plan,
    )

    dataop = mixed_consumer["1_OnChipMoveMixedSTCDP"]["datadscs_"][0][
        "0_OnChipMoveSTCDPOpLx"
    ]
    output_piece = dataop["labeledDs_"][1]["PieceInfo"][0]
    assert output_piece["dimToStartCordinate"] == {
        "out": 64,
        "mb": 0,
        "x": 0,
    }
    assert output_piece["dimToSize_"] == {"out": 64, "mb": 16, "x": 1}
    assert output_piece["validGap_"]["out"] == [[64, 0]]


def test_mixed_carrier_diagnoses_dense_actual_output_stride_blocker(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_output_piece_mode", "dense_actual")
    plan = {
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
    producer_payload = {
        "0_batchmatmul": _layout_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=list(range(32)),
            include_x=True,
        )
    }
    consumer_payload = {
        "1_neg": _layout_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=list(range(32)),
            input_layout_dim_order=["out", "mb"],
            output_layout_dim_order=["out", "mb"],
        )
    }
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 8, 1, 64],
        device_coordinates=[],
        allocation=None,
        stride_map=[512, 64, -1, 64],
        name="buf0",
    )

    _patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
        0,
        1,
        producer_payload,
        consumer_payload,
        output_arg,
        dataclasses.replace(output_arg, is_input=True, arg_index=0),
        2,
        0,
        plan,
    )

    dataop = mixed_consumer["1_OnChipMoveMixedSTCDP"]["datadscs_"][0][
        "0_OnChipMoveSTCDPOpLx"
    ]
    mismatches = diagnose_stcdp_output_layout_contiguity(dataop)

    assert len(mismatches) == 1
    assert mismatches[0]["piece"] == "p1"
    assert mismatches[0]["reason"] == "output-layout-requires-strided-placement"
    assert mismatches[0]["layoutDimOrder_"] == ["out", "mb", "x"]
    assert mismatches[0]["first_coord"] == {"out": 64, "mb": 0, "x": 0}
    first_mismatch = mismatches[0]["first_mismatch"]
    assert first_mismatch["linear_index"] == 64
    assert first_mismatch["coord"] == {"out": 64, "mb": 1, "x": 0}
    assert first_mismatch["stcdp_contiguous_element_delta"] == 64
    assert first_mismatch["required_layout_element_delta"] == 512
    assert first_mismatch["stcdp_contiguous_byte_delta"] == 128
    assert first_mismatch["required_layout_byte_delta"] == 1024


def test_mixed_carrier_diagnoses_mlp_swiglu_dense_actual_output_stride_blocker(
    monkeypatch,
):
    monkeypatch.setattr(spyre_config, "onchip_move_output_piece_mode", "dense_actual")
    plan = {
        "source_name": "buf0",
        "producer": "buf0",
        "consumer": "buf6",
        "device_sizes": [64, 256, 64],
        "device_stride_map": [64, 4096, 1],
        "producer_region_bytes": 64 * 1024,
        "consumer_region_bytes": 64 * 1024,
        "bytes_moved": 8192,
        "cell_count": 1,
        "cells": [
            {
                "cell_index": 0,
                "source_core": 0,
                "dest_core": 0,
                "source_offset_bytes": 0,
                "dest_offset_bytes": 0,
                "bytes": 8192,
                "dim_starts": {"d0_": 0, "d1_": 0, "d2_": 0},
                "dim_sizes": {"d0_": 8, "d1_": 8, "d2_": 64},
            }
        ],
    }
    producer_payload = {
        "0_batchmatmul": _layout_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=list(range(32)),
            input_layout_dim_order=["mb", "out"],
            output_layout_dim_order=["mb", "out"],
        )
    }
    producer_dsc = next(iter(producer_payload["0_batchmatmul"]["dscs_"][0].values()))
    producer_dsc["N_"]["mb_"] = 256
    producer_dsc["N_"]["out_"] = 4096
    consumer_payload = {
        "2_neg": _layout_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=list(range(32)),
            input_layout_dim_order=["mb", "out"],
            output_layout_dim_order=["mb", "out"],
        )
    }
    consumer_dsc = next(iter(consumer_payload["2_neg"]["dscs_"][0].values()))
    consumer_dsc["N_"]["mb_"] = 256
    consumer_dsc["N_"]["out_"] = 4096
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[64, 256, 64],
        device_coordinates=[],
        allocation=None,
        stride_map=[64, 4096, 64],
        name="buf0",
    )

    _patched_producer, mixed_consumer = build_mixed_onchip_move_sdsc(
        0,
        2,
        producer_payload,
        consumer_payload,
        output_arg,
        dataclasses.replace(output_arg, is_input=True, arg_index=0),
        2,
        0,
        plan,
    )

    dataop = mixed_consumer["2_OnChipMoveMixedSTCDP"]["datadscs_"][0][
        "0_OnChipMoveSTCDPOpLx"
    ]
    mismatches = diagnose_stcdp_output_layout_contiguity(dataop)

    assert len(mismatches) == 1
    assert mismatches[0]["reason"] == "output-layout-requires-strided-placement"
    assert mismatches[0]["layoutDimOrder_"] == ["mb", "out"]
    assert mismatches[0]["dimToStartCordinate"] == {"mb": 0, "out": 0}
    assert mismatches[0]["dimToSize_"] == {"mb": 8, "out": 512}
    first_mismatch = mismatches[0]["first_mismatch"]
    assert first_mismatch["linear_index"] == 8
    assert first_mismatch["coord"] == {"mb": 0, "out": 1}
    assert first_mismatch["stcdp_contiguous_element_delta"] == 8
    assert first_mismatch["required_layout_element_delta"] == 256
    assert first_mismatch["stcdp_contiguous_byte_delta"] == 16
    assert first_mismatch["required_layout_byte_delta"] == 512


def test_mixed_carrier_reuses_lx_source_for_later_fanout_consumer(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_realize", True)
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "mixed")
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 1024 * 1024)

    plan = {
        "status": "planned",
        "source_name": "buf1",
        "producer": "buf1",
        "consumer": "buf2",
        "device_sizes": [32, 64],
        "producer_region_bytes": 4096,
        "consumer_region_bytes": 4096,
        "bytes_moved": 4096,
        "cell_count": 1,
        "consumer_view": {"work_slice_dims": [{"device_dim": 0, "split": 32}]},
        "cells": [
            {
                "cell_index": 0,
                "source_core": 0,
                "dest_core": 0,
                "source_offset_bytes": 0,
                "dest_offset_bytes": 0,
                "bytes": 4096,
                "dim_starts": {"d0_": 0, "d1_": 0},
                "dim_sizes": {"d0_": 32, "d1_": 64},
            }
        ],
    }
    later_plan = dict(plan, consumer="buf5")
    specs = [
        OpSpec(
            op="batchmatmul",
            is_reduction=False,
            iteration_space={},
            args=[
                _tensor_arg("x", True, 0),
                _tensor_arg("w", True, 1),
                _tensor_arg("buf1", False, 2),
            ],
            op_info={},
        ),
        OpSpec(
            op="neg",
            is_reduction=False,
            iteration_space={},
            args=[_tensor_arg("buf1", True, 0), _tensor_arg("buf2", False, 1)],
            op_info={ONCHIP_MOVE_OP_INFO_KEY: {"buf1": plan}},
        ),
        OpSpec(
            op="exp",
            is_reduction=False,
            iteration_space={},
            args=[_tensor_arg("buf2", True, 0), _tensor_arg("buf3", False, 1)],
            op_info={},
        ),
        OpSpec(
            op="realdiv",
            is_reduction=False,
            iteration_space={},
            args=[
                _tensor_arg("buf1", True, 0),
                _tensor_arg("buf4", True, 1),
                _tensor_arg("buf5", False, 2),
            ],
            op_info={ONCHIP_MOVE_OP_INFO_KEY: {"buf1": later_plan}},
        ),
    ]
    compiled = [
        (
            {
                "0_batchmatmul": _minimal_sdsc_payload(
                    "batchmatmul",
                    output_lds_idx=2,
                    input_lds_indices=[0, 1],
                    core_ids=[0],
                )
            },
            [],
            [],
            [],
        ),
        (
            {
                "1_neg": _minimal_sdsc_payload(
                    "neg",
                    output_lds_idx=1,
                    input_lds_indices=[0],
                    core_ids=[0],
                )
            },
            [],
            [],
            [],
        ),
        (
            {
                "2_exp": _minimal_sdsc_payload(
                    "exp",
                    output_lds_idx=1,
                    input_lds_indices=[0],
                    core_ids=[0],
                )
            },
            [],
            [],
            [],
        ),
        (
            {
                "3_realdiv": _minimal_sdsc_payload(
                    "realdiv",
                    output_lds_idx=2,
                    input_lds_indices=[0, 1],
                    core_ids=[0],
                )
            },
            [],
            [],
            [],
        ),
    ]

    rows = patch_onchip_move_mixed_schedules(compiled, specs)

    assert [row["status"] for row in rows] == ["patched", "patched-reuse"]
    realdiv_root = compiled[3][0]["3_realdiv"]
    realdiv_dsc = next(iter(realdiv_root["dscs_"][0].values()))
    assert realdiv_dsc["scheduleTree_"][0]["component_"] == "lx"
    assert realdiv_dsc["scheduleTree_"][0]["startAddressCoreCorelet_"]["data_"] == {
        "[0, 0, 0]": str(1024 * 1024)
    }
    assert realdiv_dsc["labeledDs_"][0]["memOrg_"] == {"lx": {"isPresent": 1}}


def test_lx_region_validation_rejects_overlap_and_overflow():
    _validate_lx_regions(
        producer_base=0,
        consumer_base=1024 * 1024,
        producer_region_bytes=512 * 1024,
        consumer_region_bytes=512 * 1024,
    )

    try:
        _validate_lx_regions(
            producer_base=16 * 1024,
            consumer_base=8 * 1024,
            producer_region_bytes=409600,
            consumer_region_bytes=409600,
        )
    except ValueError as exc:
        assert str(exc) == "producer-consumer-lx-regions-overlap"
    else:
        raise AssertionError("expected overlapping LX regions to fail")

    try:
        _validate_lx_regions(
            producer_base=0,
            consumer_base=1536 * 1024,
            producer_region_bytes=1024,
            consumer_region_bytes=1024 * 1024,
        )
    except ValueError as exc:
        assert str(exc) == "consumer-lx-region-exceeds-capacity"
    else:
        raise AssertionError("expected overflowing LX region to fail")


def test_attach_plan_to_consumer_tolerates_frozen_ir_data():
    @dataclasses.dataclass(frozen=True)
    class FrozenData:
        op_info: object = None

    class Consumer:
        data = FrozenData()

    plan = OnChipMovePlan(
        source_name="buf0",
        producer_name="buf0",
        consumer_name="buf1",
        producer_op="op0",
        consumer_op="op1",
        status="planned",
        fallback_reason=None,
        realization_status="planned-not-realized",
        carrier="mixed",
        device_sizes=[16],
        device_stride_map=[1],
        element_bytes=2,
        producer_core_count=1,
        consumer_core_count=1,
        producer_region_bytes=32,
        consumer_region_bytes=32,
        producer_view={},
        consumer_view={},
        cells=[],
    )

    consumer = Consumer()
    _attach_plan_to_consumer(consumer, plan)

    move_info = getattr(consumer, ONCHIP_MOVE_ATTR)
    assert move_info["buf0"]["producer"] == "buf0"
    assert consumer.data.op_info is None

    class MutableData:
        def __init__(self):
            self.op_info = {}

    class MutableConsumer:
        def __init__(self):
            self.data = MutableData()

    mutable_consumer = MutableConsumer()
    _attach_plan_to_consumer(mutable_consumer, plan)
    assert (
        mutable_consumer.data.op_info[ONCHIP_MOVE_OP_INFO_KEY]["buf0"]["consumer"]
        == "buf1"
    )


def _minimal_sdsc_payload(
    name: str,
    *,
    output_lds_idx: int,
    input_lds_indices: list[int],
    core_ids: list[int],
) -> dict:
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
                            "dsType_": (
                                "OUTPUT"
                                if idx == output_lds_idx
                                else "INPUT"
                                if idx in input_lds_indices
                                else "LOCAL"
                            ),
                            "memOrg_": {"hbm": {"isPresent": 1}},
                        }
                        for idx in range(max_lds_idx + 1)
                    ],
                    "computeOp_": [
                        {
                            "inputLabeledDs": [
                                f"tensor-idx{idx}" for idx in input_lds_indices
                            ],
                            "outputLabeledDs": [f"tensor-idx{output_lds_idx}"],
                        }
                    ],
                }
            }
        ],
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
) -> dict:
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


def _tensor_arg(name: str, is_input: bool, arg_index: int) -> TensorArg:
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[32, 64],
        device_coordinates=[],
        allocation=None,
        name=name,
    )
