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

from torch_spyre._inductor.codegen.onchip_move import build_stcdp_datadsc
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
    assert cells[1].source_core == 4
    assert cells[1].dest_core == 0
    assert cells[8].source_core == 0
    assert cells[8].dest_core == 1
    assert cells[64].source_core == 1
    assert cells[64].dest_core == 8


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
    assert len(dataop["labeledDs_"][0]["PieceInfo"]) == 4


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
        element_bytes=2,
        producer_core_count=1,
        consumer_core_count=1,
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
