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

from sympy import Integer, Mod, Symbol, floor

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.lx_relayout import (
    LXRelayoutPlan,
    _core_id_to_device_slice,
    _record_plan,
    get_lx_relayout_classifications,
    get_lx_relayout_inputs,
    is_lx_relayout_reservation,
    make_lx_relayout_reservation_name,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import PerCoreView


def _fixed_tile_arg(
    *,
    is_input: bool,
    allocation: dict,
    lx_residency_core_id_to_wk_slice=None,
) -> TensorArg:
    mb = Symbol("x0")
    out = Symbol("x1")
    return TensorArg(
        is_input=is_input,
        arg_index=0 if is_input else 1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 200, 64],
        device_coordinates=[mb, floor(out / 64), Mod(out, 64)],
        allocation=allocation,
        lx_residency_core_id_to_wk_slice=lx_residency_core_id_to_wk_slice,
    )


def test_core_view_residency_payload_is_static_per_core():
    core_id = Symbol("core_id")
    view = PerCoreView(
        work_slice_dims=((0, 2), (1, 2)),
        core_to_slot=((0, Mod(core_id, 2)), (1, floor(core_id / 2))),
    )

    assert _core_id_to_device_slice(view, 4) == {
        "0": {"0": 0, "1": 0},
        "1": {"0": 1, "1": 0},
        "2": {"0": 0, "1": 1},
        "3": {"0": 1, "1": 1},
    }


def test_lx_relayout_reservation_names_are_identifiable():
    name = make_lx_relayout_reservation_name("consumer", "producer")

    assert is_lx_relayout_reservation(name)
    assert not is_lx_relayout_reservation("producer")


def test_lx_relayout_plan_records_scatter_kind():
    class DummyOp:
        pass

    consumer = DummyOp()
    _record_plan(
        consumer,
        LXRelayoutPlan(
            source_name="buf0",
            producer_name="producer",
            consumer_name="consumer",
            kind="scatter",
            producer_core_count=32,
            consumer_core_count=32,
            producer_core_id_to_device_slice={
                "0": {"0": 0},
                "1": {"0": 1},
            },
            producer_work_slice_dims={"0": 32},
            consumer_work_slice_dims={"0": 32},
        ),
    )

    assert get_lx_relayout_inputs(consumer)["buf0"]["kind"] == "scatter"


def test_unrealized_collective_is_classified_but_not_realized():
    class DummyOp:
        pass

    consumer = DummyOp()
    _record_plan(
        consumer,
        LXRelayoutPlan(
            source_name="buf0",
            producer_name="producer",
            consumer_name="consumer",
            kind="matmul_operand_broadcast",
            producer_core_count=32,
            consumer_core_count=32,
            producer_core_id_to_device_slice={
                "0": {"2": 0},
                "1": {"2": 1},
            },
            producer_work_slice_dims={"2": 32},
            consumer_work_slice_dims={"0": 32},
            read_index=1,
            realized=False,
            communication_pattern="all_gather_replicate",
            unsupported_reason="needs loop-scoped collective lowering",
        ),
    )

    classified = get_lx_relayout_classifications(consumer)["buf0"]
    assert classified["kind"] == "matmul_operand_broadcast"
    assert classified["communication_pattern"] == "all_gather_replicate"
    assert not classified["realized"]
    assert get_lx_relayout_inputs(consumer) == {}


def test_lx_input_allocation_coordinates_describe_producer_residency():
    mb = Symbol("x0")
    out = Symbol("x1")
    producer_residency = {
        "0": {"0": 0, "1": 0},
        "1": {"0": 1, "1": 0},
        "2": {"0": 0, "1": 1},
        "3": {"0": 1, "1": 1},
    }
    op_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={mb: (Integer(512), 4), out: (Integer(12800), 1)},
        args=[
            _fixed_tile_arg(
                is_input=True,
                allocation={"lx": 0},
                lx_residency_core_id_to_wk_slice=producer_residency,
            ),
            _fixed_tile_arg(is_input=False, allocation={"hbm": 0x1000}),
        ],
        op_info={},
    )

    sdsc, _symbols, _affine_strides, _symbol_kinds = compile_op_spec(0, op_spec, [])

    root = next(iter(sdsc.values()))
    compute_dsc = next(iter(root["dscs_"][0].values()))
    input_alloc = compute_dsc["scheduleTree_"][0]

    assert "dataOpdscs_" not in root
    assert input_alloc["component_"] == "lx"
    assert input_alloc["coordinates_"]["coreIdToWkSlice_"] == {
        "0": {"mb": 0, "out": 0},
        "1": {"mb": 1, "out": 0},
        "2": {"mb": 0, "out": 1},
        "3": {"mb": 1, "out": 1},
    }


def test_regular_lx_input_keeps_empty_allocation_coordinates():
    mb = Symbol("x0")
    out = Symbol("x1")
    op_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={mb: (Integer(512), 4), out: (Integer(12800), 1)},
        args=[
            _fixed_tile_arg(is_input=True, allocation={"lx": 0}),
            _fixed_tile_arg(is_input=False, allocation={"hbm": 0x1000}),
        ],
        op_info={},
    )

    sdsc, _symbols, _affine_strides, _symbol_kinds = compile_op_spec(0, op_spec, [])

    root = next(iter(sdsc.values()))
    compute_dsc = next(iter(root["dscs_"][0].values()))
    input_alloc = compute_dsc["scheduleTree_"][0]

    assert input_alloc["coordinates_"]["coreIdToWkSlice_"] == {}


def test_lx_relayout_classification_metadata_is_emitted_top_level():
    mb = Symbol("x0")
    out = Symbol("x1")
    classification = {
        "buf0": {
            "kind": "matmul_operand_broadcast",
            "communication_pattern": "all_gather_replicate",
            "realized": False,
        }
    }
    op_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={mb: (Integer(512), 4), out: (Integer(12800), 1)},
        args=[
            _fixed_tile_arg(is_input=True, allocation={"lx": 0}),
            _fixed_tile_arg(is_input=False, allocation={"hbm": 0x1000}),
        ],
        op_info={"lx_relayout_classifications": classification},
    )

    sdsc, _symbols, _affine_strides, _symbol_kinds = compile_op_spec(0, op_spec, [])

    root = next(iter(sdsc.values()))
    assert root["lxRelayoutClassifications_"] == classification
