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
from torch_spyre._inductor.codegen.lx_relayout import (
    _expand_dataop_movement_ranges,
    build_lx_relayout_sdsc,
)
from torch_spyre._inductor.lx_relayout import (
    LXRelayoutCell,
    LXRelayoutPlan,
    _lx_relayout_v1_support_reason,
    _planned_payload,
    build_lx_relayout_cells,
    validate_lx_relayout_cell_coverage,
)
from torch_spyre._inductor.op_spec import TensorArg
from torch_spyre._inductor.pass_utils import PerCoreView


def test_lx_relayout_cells_cover_matmul_to_pointwise_reshard():
    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 4), (1, 8)),
        core_to_slot=((0, sympy.Mod(core_id, 4)), (1, sympy.floor(core_id / 4))),
    )
    consumer_view = PerCoreView(
        work_slice_dims=((0, 32),),
        core_to_slot=((0, sympy.Mod(core_id, 32)),),
    )

    cells, reason = build_lx_relayout_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[512, 12800],
        element_bytes=2,
        producer_core_count=32,
        consumer_core_count=32,
    )

    assert reason is None
    assert validate_lx_relayout_cell_coverage(
        cells, device_sizes=[512, 12800]
    ) is None
    assert len(cells) == 256
    assert sum(cell.bytes for cell in cells) == 512 * 12800 * 2
    assert cells[1].source_core == 4
    assert cells[1].dest_core == 0
    assert cells[8].source_core == 0
    assert cells[8].dest_core == 1


def test_lx_relayout_v1_rejects_unaligned_and_overlapping_cells(monkeypatch):
    monkeypatch.setattr(spyre_config, "lx_relayout_producer_lx_base", 0)
    monkeypatch.setattr(spyre_config, "lx_relayout_consumer_lx_base", 0x100000)
    aligned = LXRelayoutCell(
        cell_index=0,
        source_core=0,
        dest_core=0,
        dim_starts={"d0_": 0},
        dim_sizes={"d0_": 64},
        bytes=128,
        source_offset_bytes=0,
        dest_offset_bytes=0,
    )

    assert _lx_relayout_v1_support_reason([aligned]) is None
    assert (
        _lx_relayout_v1_support_reason(
            [dataclasses.replace(aligned, dest_offset_bytes=16)]
        )
        == "lx-relayout-v1-requires-stick-aligned-destination-address"
    )
    assert (
        _lx_relayout_v1_support_reason(
            [
                dataclasses.replace(aligned, bytes=256),
                dataclasses.replace(
                    aligned,
                    cell_index=1,
                    source_offset_bytes=128,
                    dest_offset_bytes=128,
                ),
            ]
        )
        == "lx-relayout-v1-requires-contiguous-destination-cells"
    )


def test_stcdp_range_carrier_wraps_relayout_as_stcdp_lx(monkeypatch):
    monkeypatch.setattr(spyre_config, "lx_relayout_producer_lx_base", 0x1000)
    monkeypatch.setattr(spyre_config, "lx_relayout_consumer_lx_base", 0x9000)
    monkeypatch.setattr(spyre_config, "lx_relayout_chunk_cells", 32)

    core_id = sympy.Symbol("core_id")
    producer_view = PerCoreView(
        work_slice_dims=((0, 2),),
        core_to_slot=((0, sympy.Mod(core_id, 2)),),
    )
    consumer_view = PerCoreView(work_slice_dims=(), core_to_slot=())
    cells, reason = build_lx_relayout_cells(
        producer_view=producer_view,
        consumer_view=consumer_view,
        device_sizes=[2, 64],
        device_stride_map=[64, 1],
        element_bytes=2,
        producer_core_count=2,
        consumer_core_count=1,
        lx_relayout_v1=True,
    )
    assert reason is None

    plan = _lx_relayout_plan_payload(cells, device_sizes=[2, 64], element_bytes=2)
    producer_payload = {
        "1_batchmatmul": _minimal_sdsc_payload(
            "batchmatmul",
            output_lds_idx=2,
            input_lds_indices=[0, 1],
            core_ids=[0, 1],
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
        device_size=[2, 64],
        device_coordinates=[],
        allocation=None,
        name="buf0",
    )

    patched_producer, mixed_consumer = build_lx_relayout_sdsc(
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

    producer_dsc = next(iter(patched_producer["1_batchmatmul"]["dscs_"][0].values()))
    assert producer_dsc["labeledDs_"][2]["memOrg_"] == {"lx": {"isPresent": 1}}

    mixed_root = mixed_consumer["2_LXRelayoutSTCDPOpLx"]
    dataops = [next(iter(row.values())) for row in mixed_root["datadscs_"]]
    movements = [move for dataop in dataops for move in _dataop_movements(dataop)]
    consumer_dsc = next(iter(mixed_root["dscs_"][0].values()))

    assert all(dataop["op"]["name"] == "STCDPOpLx" for dataop in dataops)
    assert all("rangedLxRemap" in dataop["op"] for dataop in dataops)
    assert all(dataop["op"]["rangedLxRemap"]["schemaVersion"] == 0 for dataop in dataops)
    assert all(
        dataop["op"]["rangedLxRemap"]["producerLxBase"] == 0x1000
        for dataop in dataops
    )
    assert all(
        dataop["op"]["rangedLxRemap"]["consumerLxBase"] == 0x9000
        for dataop in dataops
    )
    assert all("movementRanges" in dataop["op"]["rangedLxRemap"] for dataop in dataops)
    assert all("movements" not in dataop["op"]["rangedLxRemap"] for dataop in dataops)
    assert {move["source"]["core"] for move in movements} == {0, 1}
    assert mixed_root["lxRelayout_"]["carrier"] == "stcdp_range"
    assert "STCDPOpLx" in mixed_root["opFuncsUsed_"]
    assert consumer_dsc["labeledDs_"][0]["memOrg_"] == {"lx": {"isPresent": 1}}


def _lx_relayout_plan_payload(
    cells: list[LXRelayoutCell],
    *,
    device_sizes: list[int],
    element_bytes: int,
) -> dict:
    plan = LXRelayoutPlan(
        source_name="buf0",
        producer_name="buf0",
        consumer_name="buf1",
        device_sizes=device_sizes,
        device_stride_map=list(range(len(device_sizes))),
        element_bytes=element_bytes,
        producer_core_count=2,
        consumer_core_count=1,
        producer_region_bytes=sum(cell.bytes for cell in cells),
        consumer_region_bytes=sum(cell.bytes for cell in cells),
        cells=cells,
    )
    return _planned_payload(plan)


def _dataop_movements(dataop: dict) -> list[dict]:
    if (dataop.get("op") or {}).get("name") == "STCDPOpLx":
        dataop = (dataop.get("op") or {}).get("rangedLxRemap") or {}
    return _expand_dataop_movement_ranges(dataop.get("movementRanges") or [])


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
