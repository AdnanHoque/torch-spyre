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
    build_coordinate_remap_onchip_move_sdsc,
    build_mixed_onchip_move_sdsc,
    build_stcdp_datadsc,
    patch_onchip_move_mixed_schedules,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.onchip_move import (
    ONCHIP_MOVE_ATTR,
    ONCHIP_MOVE_OP_INFO_KEY,
    OnChipMoveCell,
    OnChipMovePlan,
    _coordinate_remap_v1_support_reason,
    _attach_plan_to_consumer,
    _plan_json,
    build_coordinate_remap_metadata,
    build_onchip_move_cells,
    validate_onchip_move_cell_coverage,
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


def test_coordinate_remap_v1_rejects_unlowerable_cells(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0x100000)
    aligned = OnChipMoveCell(
        cell_index=0,
        source_core=0,
        dest_core=0,
        dim_starts={"d0_": 0},
        dim_sizes={"d0_": 64},
        bytes=128,
        source_offset_bytes=0,
        dest_offset_bytes=0,
    )

    assert _coordinate_remap_v1_support_reason([aligned]) is None

    assert (
        _coordinate_remap_v1_support_reason(
            [dataclasses.replace(aligned, dest_offset_bytes=16)]
        )
        == "coordinate-remap-v1-requires-stick-aligned-destination-address"
    )

    overlapping = [
        dataclasses.replace(aligned, bytes=256),
        dataclasses.replace(
            aligned,
            cell_index=1,
            source_offset_bytes=128,
            dest_offset_bytes=128,
        ),
    ]
    assert (
        _coordinate_remap_v1_support_reason(overlapping)
        == "coordinate-remap-v1-requires-contiguous-destination-cells"
    )


def test_coordinate_remap_v1_swiglu_reshard_uses_physical_sticks(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0x100000)
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 8), (1, 4)),
        core_to_slot=(
            (0, sympy.Mod(sympy.floor(core_id / 4), 8)),
            (1, sympy.Mod(core_id, 4)),
        ),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((1, 32),),
        core_to_slot=((1, sympy.Mod(core_id, 32)),),
    )

    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[64, 256, 64],
        device_stride_map=[64, 4096, 1],
        element_bytes=2,
        producer_core_count=32,
        consumer_core_count=32,
        max_cells=65536,
        coordinate_remap_v1=True,
    )

    assert reason is None
    assert len(cells) == 64 * 256
    assert sum(cell.bytes for cell in cells) == 64 * 256 * 64 * 2
    assert all(cell.bytes == 128 for cell in cells)
    assert _coordinate_remap_v1_support_reason(cells) is None

    assert cells[0].source_core == 0
    assert cells[0].dest_core == 0
    assert cells[0].dim_starts == {"d0_": 0, "d1_": 0, "d2_": 0}
    assert cells[0].dim_sizes == {"d0_": 1, "d1_": 1, "d2_": 64}
    assert cells[0].source_offset_bytes == 0
    assert cells[0].dest_offset_bytes == 0

    assert cells[1].source_core == 0
    assert cells[1].dest_core == 0
    assert cells[1].dim_starts == {"d0_": 0, "d1_": 1, "d2_": 0}
    assert cells[1].source_offset_bytes == 128
    assert cells[1].dest_offset_bytes == 128

    assert cells[256].dim_starts == {"d0_": 1, "d1_": 0, "d2_": 0}
    assert cells[256].source_core == 0
    assert cells[256].dest_core == 0
    assert cells[256].source_offset_bytes == 8192
    assert cells[256].dest_offset_bytes == 1024

    assert cells[2048].dim_starts == {"d0_": 8, "d1_": 0, "d2_": 0}
    assert cells[2048].source_core == 4
    assert cells[2048].dest_core == 0
    assert cells[2048].source_offset_bytes == 0
    assert cells[2048].dest_offset_bytes == 8192


def test_coordinate_remap_v1_negmm_pure_m_consumer_uses_lx_stick_major_order(
    monkeypatch,
):
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 8), (1, 4)),
        core_to_slot=(
            (0, sympy.Mod(sympy.floor(core_id / 4), 8)),
            (1, sympy.Mod(core_id, 4)),
        ),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((1, 32),),
        core_to_slot=((1, sympy.Mod(core_id, 32)),),
    )

    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[8, 256, 64],
        device_stride_map=[64, 512, 1],
        element_bytes=2,
        producer_core_count=32,
        consumer_core_count=32,
        max_cells=65536,
        coordinate_remap_v1=True,
    )

    assert reason is None
    assert len(cells) == 8 * 256
    assert all(cell.bytes == 128 for cell in cells)
    assert _coordinate_remap_v1_support_reason(cells) is None

    # d1 is the M/row dimension.  Consecutive rows in one pure-M consumer tile
    # are adjacent sticks in LX.
    assert cells[1].dim_starts == {"d0_": 0, "d1_": 1, "d2_": 0}
    assert cells[1].source_offset_bytes == 128
    assert cells[1].dest_offset_bytes == 128

    # d0 is the outer output-stick dimension.  It is slowest in Deeptools' LX
    # local view, so advancing one output stick skips the 8-row consumer tile.
    assert cells[256].dim_starts == {"d0_": 1, "d1_": 0, "d2_": 0}
    assert cells[256].source_core == 4
    assert cells[256].dest_core == 0
    assert cells[256].source_offset_bytes == 0
    assert cells[256].dest_offset_bytes == 1024


def test_coordinate_remap_v1_bmm_swiglu_layout_canonicalizes_trailing_stick_stride(
    monkeypatch,
):
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 4), (1, 8)),
        core_to_slot=(
            (0, sympy.Mod(core_id, 4)),
            (1, sympy.Mod(sympy.floor(core_id / 4), 8)),
        ),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 32),),
        core_to_slot=((0, sympy.Mod(core_id, 32)),),
    )

    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[256, 8, 1, 64],
        device_stride_map=[512, 64, -1, 1],
        element_bytes=2,
        producer_core_count=32,
        consumer_core_count=32,
        max_cells=65536,
        coordinate_remap_v1=True,
    )

    assert reason is None
    assert len(cells) == 256 * 8
    assert sum(cell.bytes for cell in cells) == 256 * 8 * 64 * 2
    assert all(cell.bytes == 128 for cell in cells)
    assert _coordinate_remap_v1_support_reason(cells) is None

    assert cells[0].source_core == 0
    assert cells[0].dest_core == 0
    assert cells[0].dim_starts == {"d0_": 0, "d1_": 0, "d2_": 0, "d3_": 0}
    assert cells[0].source_offset_bytes == 0
    assert cells[0].dest_offset_bytes == 0

    # The BMM-shaped layout orders device dims as
    # [mb, out_stick, collapsed_x, stick_elem], so adjacent output sticks are
    # not adjacent in the pure-M consumer LX tile.
    assert cells[1].dim_starts == {"d0_": 0, "d1_": 1, "d2_": 0, "d3_": 0}
    assert cells[1].source_core == 4
    assert cells[1].dest_core == 0
    assert cells[1].source_offset_bytes == 0
    assert cells[1].dest_offset_bytes == 1024

    # Advancing one row within the same output stick is adjacent in LX.
    assert cells[8].dim_starts == {"d0_": 1, "d1_": 0, "d2_": 0, "d3_": 0}
    assert cells[8].source_core == 0
    assert cells[8].dest_core == 0
    assert cells[8].source_offset_bytes == 128
    assert cells[8].dest_offset_bytes == 128


def test_coordinate_remap_cells_cover_1d_without_overlap_or_gaps(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0x1000)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0x9000)
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 4),),
        core_to_slot=((0, sympy.Mod(core_id, 4)),),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 2),),
        core_to_slot=((0, sympy.Mod(core_id, 2)),),
    )

    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[16],
        element_bytes=2,
        producer_core_count=4,
        consumer_core_count=2,
    )

    assert reason is None
    assert validate_onchip_move_cell_coverage(cells, device_sizes=[16]) is None
    assert [cell.source_core for cell in cells] == [0, 1, 2, 3]
    assert [cell.dest_core for cell in cells] == [0, 0, 1, 1]

    plan = _remap_plan(cells, device_sizes=[16], element_bytes=2)
    metadata = build_coordinate_remap_metadata(plan)

    assert metadata["primitive"] == "lx_coordinate_remap_v0"
    assert metadata["coverage"]["status"] == "complete"
    assert [row["kind"] for row in metadata["dependency_order"]] == [
        "producer_lx_write_before_remap",
        "coordinate_remap",
        "consumer_lx_read_after_remap",
    ]
    move = metadata["movements"][1]
    assert move["source_core"] == 1
    assert move["destination_core"] == 0
    assert move["source_slice"] == {
        "starts": {"d0_": 4},
        "sizes": {"d0_": 4},
    }
    assert move["destination_slice"] == move["source_slice"]
    assert move["source_lx_address"] == 0x1000
    assert move["source_lx_byte_range"] == {"start": 0x1000, "end": 0x1008}
    assert move["destination_lx_address"] == 0x9008
    assert move["destination_lx_byte_range"] == {"start": 0x9008, "end": 0x9010}
    dataop = metadata["deeptools_dataop"]
    assert dataop["op"] == {"name": "LXCoordinateRemapOp"}
    assert dataop["schemaVersion"] == 0
    assert dataop["sourceName"] == "buf0"
    assert dataop["producerLxBase"] == 0x1000
    assert dataop["consumerLxBase"] == 0x9000
    assert dataop["coverage"] == {"device_sizes": [16], "status": "complete"}
    assert [row["kind"] for row in dataop["dependencyOrder"]] == [
        "producer_lx_write_before_remap",
        "coordinate_remap",
        "consumer_lx_read_after_remap",
    ]
    assert dataop["movements"][1] == {
        "moveIndex": 1,
        "bytes": 8,
        "source": {
            "core": 1,
            "logicalSlice": {"starts": {"d0_": 4}, "sizes": {"d0_": 4}},
            "lxAddress": 0x1000,
            "localByteRange": {"start": 0, "end": 8},
            "lxByteRange": {"start": 0x1000, "end": 0x1008},
        },
        "destination": {
            "core": 0,
            "logicalSlice": {"starts": {"d0_": 4}, "sizes": {"d0_": 4}},
            "lxAddress": 0x9008,
            "localByteRange": {"start": 8, "end": 16},
            "lxByteRange": {"start": 0x9008, "end": 0x9010},
        },
    }
    _assert_no_local_destination_overlap_or_gap(dataop)
    _assert_patterned_coordinate_remap_is_value_correct(dataop)


def test_coordinate_remap_cells_cover_2d_without_overlap_or_gaps(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0x2000)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0xA000)
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 2), (1, 2)),
        core_to_slot=((0, sympy.Mod(core_id, 2)), (1, sympy.floor(core_id / 2))),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 4),),
        core_to_slot=((0, sympy.Mod(core_id, 4)),),
    )

    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[8, 6],
        element_bytes=4,
        producer_core_count=4,
        consumer_core_count=4,
    )

    assert reason is None
    assert validate_onchip_move_cell_coverage(cells, device_sizes=[8, 6]) is None
    assert len(cells) == 8
    assert sum(cell.bytes for cell in cells) == 8 * 6 * 4

    plan = _remap_plan(cells, device_sizes=[8, 6], element_bytes=4)
    metadata = build_coordinate_remap_metadata(plan)

    assert metadata["coverage"] == {"device_sizes": [8, 6], "status": "complete"}
    assert metadata["producer_lx_base"] == 0x2000
    assert metadata["consumer_lx_base"] == 0xA000
    crossing_moves = [
        move
        for move in metadata["movements"]
        if move["source_core"] != move["destination_core"]
    ]
    assert crossing_moves
    assert crossing_moves[0]["source_slice"] == crossing_moves[0][
        "destination_slice"
    ]
    assert {
        "source_core",
        "source_slice",
        "source_lx_address",
        "source_lx_byte_range",
        "destination_core",
        "destination_slice",
        "destination_lx_address",
        "destination_lx_byte_range",
    }.issubset(crossing_moves[0])
    dataop = metadata["deeptools_dataop"]
    assert dataop["op"] == {"name": "LXCoordinateRemapOp"}
    assert dataop["producer"] == "buf0"
    assert dataop["consumer"] == "buf1"
    assert dataop["movements"][0]["source"]["logicalSlice"] == dataop[
        "movements"
    ][0]["destination"]["logicalSlice"]
    assert dataop["movements"][0]["source"]["lxByteRange"]["end"] - dataop[
        "movements"
    ][0]["source"]["lxByteRange"]["start"] == dataop["movements"][0]["bytes"]
    _assert_no_local_destination_overlap_or_gap(dataop)
    _assert_patterned_coordinate_remap_is_value_correct(dataop)


def test_coordinate_remap_plan_json_omits_raw_cells_by_default(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_debug_cells", False)
    cells = [
        OnChipMoveCell(
            cell_index=0,
            source_core=0,
            dest_core=1,
            dim_starts={"d0_": 0},
            dim_sizes={"d0_": 64},
            bytes=128,
            source_offset_bytes=0,
            dest_offset_bytes=0,
        )
    ]
    payload = _plan_json(_remap_plan(cells, device_sizes=[64], element_bytes=2))

    assert payload["cell_count"] == 1
    assert payload["bytes_moved"] == 128
    assert "cells" not in payload
    assert "movements" not in payload["coordinate_remap"]
    assert payload["coordinate_remap"]["deeptools_dataop"]["movements"]


def test_coordinate_remap_plan_json_can_include_debug_cells(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_debug_cells", True)
    cells = [
        OnChipMoveCell(
            cell_index=0,
            source_core=0,
            dest_core=1,
            dim_starts={"d0_": 0},
            dim_sizes={"d0_": 64},
            bytes=128,
            source_offset_bytes=0,
            dest_offset_bytes=0,
        )
    ]
    payload = _plan_json(_remap_plan(cells, device_sizes=[64], element_bytes=2))

    assert payload["cells"]
    assert payload["coordinate_remap"]["movements"]


def test_coordinate_remap_coalesces_contiguous_nonrectangular_metadata(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0x100000)
    cells = [
        OnChipMoveCell(
            cell_index=0,
            source_core=0,
            dest_core=1,
            dim_starts={"d0_": 0, "d1_": 0},
            dim_sizes={"d0_": 1, "d1_": 64},
            bytes=128,
            source_offset_bytes=0,
            dest_offset_bytes=0,
        ),
        OnChipMoveCell(
            cell_index=1,
            source_core=0,
            dest_core=1,
            dim_starts={"d0_": 1, "d1_": 1},
            dim_sizes={"d0_": 1, "d1_": 64},
            bytes=128,
            source_offset_bytes=128,
            dest_offset_bytes=128,
        ),
    ]

    metadata = build_coordinate_remap_metadata(
        _remap_plan(cells, device_sizes=[2, 65], element_bytes=1),
    )
    dataop = metadata["deeptools_dataop"]

    assert dataop["lowering"]["sourceMovements"] == 2
    assert dataop["lowering"]["coalescedMovements"] == 1
    assert dataop["movements"] == [
        {
            "moveIndex": 0,
            "bytes": 256,
            "source": {
                "core": 0,
                "logicalSlice": {"starts": {}, "sizes": {}, "coalesced": True},
                "lxAddress": 0,
                "localByteRange": {"start": 0, "end": 256},
                "lxByteRange": {"start": 0, "end": 256},
            },
            "destination": {
                "core": 1,
                "logicalSlice": {"starts": {}, "sizes": {}, "coalesced": True},
                "lxAddress": 0x100000,
                "localByteRange": {"start": 0, "end": 256},
                "lxByteRange": {"start": 0x100000, "end": 0x100100},
            },
        }
    ]


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


def test_coordinate_remap_carrier_emits_real_dataop_sdsc(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0x1000)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0x9000)

    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 4),),
        core_to_slot=((0, sympy.Mod(core_id, 4)),),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 2),),
        core_to_slot=((0, sympy.Mod(core_id, 2)),),
    )
    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[16],
        element_bytes=2,
        producer_core_count=4,
        consumer_core_count=2,
    )
    assert reason is None
    plan = _remap_plan_payload(cells, device_sizes=[16], element_bytes=2)
    producer_payload = {
        "1_batchmatmul": _minimal_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=[0, 1, 2, 3],
        )
    }
    consumer_payload = {
        "2_neg": _minimal_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=[0, 1],
        )
    }
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[16],
        device_coordinates=[],
        allocation=None,
        name="buf0",
    )

    patched_producer, mixed_consumer = build_coordinate_remap_onchip_move_sdsc(
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

    mixed_root = mixed_consumer["2_OnChipMoveCoordinateRemap"]
    assert len(mixed_root["dscs_"]) == 1
    assert len(mixed_root["datadscs_"]) == mixed_root["onchipMove_"][
        "dataop_chunks"
    ]
    dataops = [next(iter(row.values())) for row in mixed_root["datadscs_"]]
    dataop = dataops[0]
    assert dataop["op"] == {"name": "LXCoordinateRemapOp"}
    assert dataop["coreIdsUsed_"] == [0, 1, 2, 3]
    assert dataop["schemaVersion"] == 0
    assert dataop["producerLxBase"] == 0x1000
    assert dataop["consumerLxBase"] == 0x9000
    movements = [move for row in dataops for move in row["movements"]]
    assert len(movements) == 5
    assert any(
        move["source"]["core"] == 1 and move["destination"]["core"] == 0
        for move in movements
    )
    relay_moves = [move for move in movements if "relay" in move]
    assert [move["relay"]["kind"] for move in relay_moves] == [
        "local_first_leg",
        "local_second_leg",
    ]
    assert mixed_root["coreIdToDscSchedule"]["0"][-1][0:2] == [-1, 0]
    assert mixed_root["coreIdToDscSchedule"]["2"] == [[0, -1, 0, 0]]
    assert mixed_root["coreIdToDsc_"] == {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
    }
    assert set(mixed_root["coreIdToWkSlice_"]) == {"0", "1", "2", "3"}
    assert "LXCoordinateRemapOp" in mixed_root["opFuncsUsed_"]
    assert "STCDPOpLx" not in mixed_root["opFuncsUsed_"]
    assert mixed_root["onchipMove_"]["carrier"] == "coordinate_remap"
    consumer_dsc = next(iter(mixed_root["dscs_"][0].values()))
    assert consumer_dsc["labeledDs_"][0]["memOrg_"] == {"lx": {"isPresent": 1}}


def test_coordinate_remap_carrier_chunks_large_dataop(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    monkeypatch.setattr(spyre_config, "onchip_move_coordinate_remap_chunk_cells", 2)
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0x1000)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0x9000)

    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 4),),
        core_to_slot=((0, sympy.Mod(core_id, 4)),),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 2),),
        core_to_slot=((0, sympy.Mod(core_id, 2)),),
    )
    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[16],
        element_bytes=2,
        producer_core_count=4,
        consumer_core_count=2,
    )
    assert reason is None
    plan = _remap_plan_payload(cells, device_sizes=[16], element_bytes=2)
    producer_payload = {
        "1_batchmatmul": _minimal_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=[0, 1, 2, 3],
        )
    }
    consumer_payload = {
        "2_neg": _minimal_sdsc_payload(
            "neg",
            output_lds_idx=1,
            input_lds_indices=[0],
            core_ids=[0, 1],
        )
    }
    output_arg = TensorArg(
        is_input=False,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[16],
        device_coordinates=[],
        allocation=None,
        name="buf0",
    )

    _, mixed_consumer = build_coordinate_remap_onchip_move_sdsc(
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

    mixed_root = mixed_consumer["2_OnChipMoveCoordinateRemap"]
    assert len(mixed_root["datadscs_"]) == 4
    assert [
        list(datadsc)[0] for datadsc in mixed_root["datadscs_"]
    ] == [
        f"1_OnChipMoveLXCoordinateRemapOp_{idx}"
        for idx in range(len(mixed_root["datadscs_"]))
    ]
    assert mixed_root["onchipMove_"]["dataop_chunks"] == len(
        mixed_root["datadscs_"]
    )
    movement_counts = [
        len(next(iter(row.values()))["movements"])
        for row in mixed_root["datadscs_"]
    ]
    assert sum(movement_counts) == 5
    assert max(movement_counts) <= 2
    movements = [
        movement
        for row in mixed_root["datadscs_"]
        for movement in next(iter(row.values()))["movements"]
    ]
    assert sum(1 for movement in movements if "relay" in movement) == 2
    assert mixed_root["coreIdToDscSchedule"]["0"][-1][0:2] == [-1, 0]
    assert mixed_root["coreIdToDscSchedule"]["1"][-1][0:2] == [-1, 0]
    assert all(row[1] == -1 for row in mixed_root["coreIdToDscSchedule"]["2"])


def test_coordinate_remap_carrier_patch_rewrites_adjacent_specs(monkeypatch):
    monkeypatch.setattr(spyre_config, "onchip_move_realize", True)
    monkeypatch.setattr(spyre_config, "onchip_move_carrier", "coordinate_remap")
    monkeypatch.setattr(spyre_config, "onchip_move_producer_lx_base", 0x1000)
    monkeypatch.setattr(spyre_config, "onchip_move_consumer_lx_base", 0x9000)

    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 4),),
        core_to_slot=((0, sympy.Mod(core_id, 4)),),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 2),),
        core_to_slot=((0, sympy.Mod(core_id, 2)),),
    )
    cells, reason = build_onchip_move_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[16],
        element_bytes=2,
        producer_core_count=4,
        consumer_core_count=2,
    )
    assert reason is None
    plan = _remap_plan_payload(cells, device_sizes=[16], element_bytes=2)
    specs = [
        OpSpec(
            op="batchmatmul",
            is_reduction=False,
            iteration_space={},
            args=[
                _tensor_arg("x", True, 0),
                _tensor_arg("w", True, 1),
                _tensor_arg("buf0", False, 2),
            ],
            op_info={},
        ),
        OpSpec(
            op="neg",
            is_reduction=False,
            iteration_space={},
            args=[_tensor_arg("buf0", True, 0), _tensor_arg("buf1", False, 1)],
            op_info={ONCHIP_MOVE_OP_INFO_KEY: {"buf0": plan}},
        ),
    ]
    compiled = [
        (
            {
                "0_batchmatmul": _minimal_sdsc_payload(
                    "batchmatmul",
                    output_lds_idx=2,
                    input_lds_indices=[0, 1],
                    core_ids=[0, 1, 2, 3],
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
                    core_ids=[0, 1],
                )
            },
            [],
            [],
            [],
        ),
    ]

    rows = patch_onchip_move_mixed_schedules(compiled, specs)

    assert rows == [
        {
            "index": 0,
            "status": "patched",
            "reason": None,
            "source_name": "buf0",
            "producer": "buf0",
            "consumer": "buf1",
            "cell_count": 4,
            "carrier": "coordinate_remap",
        }
    ]
    mixed_root = compiled[1][0]["1_OnChipMoveCoordinateRemap"]
    dataop = next(iter(mixed_root["datadscs_"][0].values()))
    assert dataop["op"] == {"name": "LXCoordinateRemapOp"}
    assert mixed_root["coreIdToDscSchedule"]["0"][0][0:2] == [0, -1]
    assert mixed_root["coreIdToDscSchedule"]["0"][-1][0:2] == [-1, 0]


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
        "[0, 0, 0]": 1024 * 1024
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


def _remap_plan(
    cells,
    *,
    device_sizes: list[int],
    element_bytes: int,
) -> OnChipMovePlan:
    return OnChipMovePlan(
        source_name="buf0",
        producer_name="buf0",
        consumer_name="buf1",
        producer_op="producer_op",
        consumer_op="consumer_op",
        status="planned",
        fallback_reason=None,
        realization_status="planned-coordinate-remap-needs-deeptools",
        carrier="coordinate_remap",
        device_sizes=device_sizes,
        device_stride_map=list(range(len(device_sizes))),
        element_bytes=element_bytes,
        producer_core_count=32,
        consumer_core_count=32,
        producer_region_bytes=sum(cell.bytes for cell in cells),
        consumer_region_bytes=sum(cell.bytes for cell in cells),
        producer_view={},
        consumer_view={},
        cells=cells,
    )


def _remap_plan_payload(
    cells,
    *,
    device_sizes: list[int],
    element_bytes: int,
) -> dict:
    plan = _remap_plan(
        cells,
        device_sizes=device_sizes,
        element_bytes=element_bytes,
    )
    payload = dataclasses.asdict(plan)
    payload["producer"] = plan.producer_name
    payload["consumer"] = plan.consumer_name
    payload["cell_count"] = len(cells)
    payload["bytes_moved"] = sum(cell.bytes for cell in cells)
    payload["cells"] = [dataclasses.asdict(cell) for cell in cells]
    payload["coordinate_remap"] = build_coordinate_remap_metadata(plan)
    return payload


def _assert_no_local_destination_overlap_or_gap(dataop: dict) -> None:
    ranges_by_core: dict[int, list[tuple[int, int]]] = {}
    for move in dataop["movements"]:
        byte_range = move["destination"]["localByteRange"]
        assert byte_range["end"] - byte_range["start"] == move["bytes"]
        ranges_by_core.setdefault(move["destination"]["core"], []).append(
            (byte_range["start"], byte_range["end"])
        )

    for ranges in ranges_by_core.values():
        ranges.sort()
        assert ranges[0][0] == 0
        for left, right in zip(ranges, ranges[1:]):
            assert left[1] == right[0]


def _assert_patterned_coordinate_remap_is_value_correct(dataop: dict) -> None:
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
        expected = source[source_range["start"] : source_range["end"]]
        core = move["destination"]["core"]
        start = dest_range["start"]
        end = dest_range["end"]
        assert bytes(destination_by_core[core][start:end]) == expected


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
