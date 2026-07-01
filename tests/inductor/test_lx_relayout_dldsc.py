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
from torch_spyre._inductor.constants import RESTICKIFY_LX_OP
from torch_spyre._inductor.lx_relayout import (
    COMM_CLASS_ALL_GATHER,
    COMM_CLASS_ALL_REDUCE,
    COMM_CLASS_BROADCAST,
    COMM_CLASS_GATHER,
    COMM_CLASS_MULTICAST,
    COMM_CLASS_REDUCE,
    COMM_CLASS_SCATTER,
    LAYOUT_ALLGATHER_RESTICKIFY,
    LXRelayoutPlan,
    _classify_communication_class,
    _core_id_to_device_slice,
    _static_buffer_nbytes,
    drop_lx_relayout_reservations,
    _record_plan,
    get_lx_relayout_classifications,
    get_lx_relayout_inputs,
    is_lx_relayout_reservation,
    lx_relayout_needs_resident_reservation,
    make_lx_relayout_reservation_name,
    parse_lx_relayout_reservation_name,
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


def _lx_restickify_arg(*, is_input: bool, allocation: dict) -> TensorArg:
    mb = Symbol("x0")
    out = Symbol("x1")
    if is_input:
        coordinates = [mb, floor(out / 64), Mod(out, 64)]
        size = [512, 200, 64]
    else:
        coordinates = [floor(mb / 64), out, Mod(mb, 64)]
        size = [8, 12800, 64]
    return TensorArg(
        is_input=is_input,
        arg_index=0 if is_input else 1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=size,
        device_coordinates=coordinates,
        allocation=allocation,
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
    assert parse_lx_relayout_reservation_name(name) == ("consumer", "producer")
    assert parse_lx_relayout_reservation_name("producer") is None


def test_static_buffer_nbytes_uses_buffer_size_and_dtype_itemsize():
    class DummyDType:
        itemsize = 2

    class DummyBuffer:
        def get_size(self):
            return [2, Integer(3), 4]

        def get_dtype(self):
            return DummyDType()

    class DummyGraph:
        name_to_buffer = {"buf0": DummyBuffer()}

    assert _static_buffer_nbytes(DummyGraph(), "buf0") == 48
    assert _static_buffer_nbytes(DummyGraph(), "missing") is None


def test_coordinate_classifier_marks_one_to_one_as_scatter():
    assert (
        _classify_communication_class(
            {"0": {"0": 0}, "1": {"0": 1}},
            {"0": 2},
            {"0": {"0": 1}, "1": {"0": 0}},
            {"0": 2},
        )
        == COMM_CLASS_SCATTER
    )


def test_coordinate_classifier_marks_one_to_many_as_broadcast_or_multicast():
    assert (
        _classify_communication_class(
            {"0": {}},
            {},
            {"0": {"0": 0}, "1": {"0": 1}},
            {"0": 2},
        )
        == COMM_CLASS_BROADCAST
    )
    assert (
        _classify_communication_class(
            {"0": {"0": 0}, "1": {"0": 1}},
            {"0": 2},
            {
                "0": {"0": 0},
                "1": {"0": 1},
                "2": {"0": 2},
                "3": {"0": 3},
            },
            {"0": 4},
        )
        == COMM_CLASS_MULTICAST
    )


def test_coordinate_classifier_marks_fan_in_as_gather():
    assert (
        _classify_communication_class(
            {
                "0": {"1": 0},
                "1": {"1": 1},
                "2": {"1": 2},
                "3": {"1": 3},
            },
            {"1": 4},
            {"0": {"1": 0}, "1": {"1": 1}},
            {"1": 2},
        )
        == COMM_CLASS_GATHER
    )


def test_coordinate_classifier_marks_full_replication_as_all_gather():
    assert (
        _classify_communication_class(
            {
                "0": {"1": 0},
                "1": {"1": 1},
                "2": {"1": 2},
                "3": {"1": 3},
            },
            {"1": 4},
            {"0": {"0": 0}, "1": {"0": 1}},
            {"0": 2},
        )
        == COMM_CLASS_ALL_GATHER
    )


def test_coordinate_classifier_marks_reduction_collectives():
    assert (
        _classify_communication_class(
            {
                "0": {"0": 0},
                "1": {"0": 1},
                "2": {"0": 2},
                "3": {"0": 3},
            },
            {"0": 4},
            {"0": {"0": 0}, "1": {"0": 1}},
            {"0": 2},
            is_reduction=True,
        )
        == COMM_CLASS_REDUCE
    )
    assert (
        _classify_communication_class(
            {
                "0": {"1": 0},
                "1": {"1": 1},
                "2": {"1": 2},
                "3": {"1": 3},
            },
            {"1": 4},
            {"0": {"0": 0}, "1": {"0": 1}},
            {"0": 2},
            is_reduction=True,
        )
        == COMM_CLASS_ALL_REDUCE
    )


def test_coordinate_classifier_marks_attention_substick_operand_as_gather():
    producer = {str(core): {"1": core} for core in range(32)}
    consumer = {str(core): {"0": core % 16, "1": core // 16} for core in range(32)}

    assert (
        _classify_communication_class(
            producer,
            {"1": 32},
            consumer,
            {"0": 16, "1": 2},
        )
        == COMM_CLASS_GATHER
    )


def test_failed_reservation_disables_only_that_realized_plan():
    class DummyOp:
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    class DummyGraph:
        operations = [DummyOp("consumer")]

    consumer = DummyGraph.operations[0]
    _record_plan(
        consumer,
        LXRelayoutPlan(
            source_name="buf0",
            producer_name="producer",
            consumer_name="consumer",
            kind="matmul_operand_broadcast",
            producer_core_count=32,
            consumer_core_count=32,
            producer_core_id_to_device_slice={"0": {"0": 0}},
            producer_work_slice_dims={"0": 32},
            consumer_work_slice_dims={"0": 32},
            realized=True,
            communication_pattern="all_gather_replicate",
        ),
    )

    removed = drop_lx_relayout_reservations(
        DummyGraph(),
        [make_lx_relayout_reservation_name("consumer", "buf0")],
    )

    assert removed == 1
    assert get_lx_relayout_inputs(consumer) == {}
    classification = get_lx_relayout_classifications(consumer)["buf0"]
    assert not classification["realized"]
    assert "did not fit" in classification["unsupported_reason"]


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


def test_realized_collective_is_classified_and_recorded_as_input():
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
            realized=True,
            communication_pattern="all_gather_replicate",
            realization_strategy="loop_scoped_input_fetch",
        ),
    )

    classified = get_lx_relayout_classifications(consumer)["buf0"]
    realized = get_lx_relayout_inputs(consumer)["buf0"]
    assert classified["kind"] == "matmul_operand_broadcast"
    assert classified["communication_pattern"] == "all_gather_replicate"
    assert classified["realized"]
    assert classified["realization_strategy"] == "loop_scoped_input_fetch"
    assert realized == classified


def test_collective_relayout_does_not_need_resident_reservation():
    assert not lx_relayout_needs_resident_reservation(
        {"kind": "matmul_operand_broadcast"}
    )
    assert lx_relayout_needs_resident_reservation({"kind": "scatter"})


def test_loop_scoped_layout_restickify_does_not_need_resident_reservation():
    assert not lx_relayout_needs_resident_reservation(
        {
            "kind": "layout_restickify_activation",
            "communication_pattern": "layout_transform_then_operand_broadcast",
        }
    )
    assert lx_relayout_needs_resident_reservation(
        {
            "kind": "layout_restickify_activation",
            "communication_pattern": "layout_transform",
        }
    )


def test_layout_allgather_restickify_contract_is_recorded():
    class DummyOp:
        pass

    consumer = DummyOp()
    _record_plan(
        consumer,
        LXRelayoutPlan(
            source_name="buf0",
            producer_name="restickify_buf",
            consumer_name="consumer",
            kind=LAYOUT_ALLGATHER_RESTICKIFY,
            producer_core_count=32,
            consumer_core_count=32,
            producer_core_id_to_device_slice={
                "0": {"mb": 0, "x": 0, "out": 0},
                "1": {"mb": 1, "x": 0, "out": 0},
            },
            producer_work_slice_dims={"mb": 4, "x": 8, "out": 1},
            consumer_work_slice_dims={"x": 4, "mb": 8, "out": 1, "in": 1},
            read_index=1,
            realized=False,
            communication_class=COMM_CLASS_ALL_GATHER,
            communication_pattern=LAYOUT_ALLGATHER_RESTICKIFY,
            realization_strategy=(
                "staged_lx_restickify_then_loop_scoped_input_fetch"
            ),
            requires_staged_realization=True,
            producer_layout={
                "layoutDimOrder_": ["out", "x", "mb"],
                "stickDimOrder_": ["out"],
            },
            restickify_kernel_layout={
                "layoutDimOrder_": ["x", "out", "mb"],
                "stickDimOrder_": ["x"],
            },
            consumer_kernel_layout={
                "layoutDimOrder_": ["out", "in", "x"],
                "stickDimOrder_": ["out"],
            },
            dimension_rename={
                "restickify.x": "batchmatmul.out",
                "restickify.out": "batchmatmul.in",
                "restickify.mb": "batchmatmul.x",
            },
            unsupported_reason="backend lowering is not implemented",
        ),
    )

    classified = get_lx_relayout_classifications(consumer)["buf0"]
    assert classified["kind"] == LAYOUT_ALLGATHER_RESTICKIFY
    assert classified["communication_class"] == COMM_CLASS_ALL_GATHER
    assert classified["communication_pattern"] == LAYOUT_ALLGATHER_RESTICKIFY
    assert classified["producer_layout"] == {
        "layoutDimOrder_": ["out", "x", "mb"],
        "stickDimOrder_": ["out"],
    }
    assert classified["restickify_kernel_layout"] == {
        "layoutDimOrder_": ["x", "out", "mb"],
        "stickDimOrder_": ["x"],
    }
    assert classified["consumer_kernel_layout"] == {
        "layoutDimOrder_": ["out", "in", "x"],
        "stickDimOrder_": ["out"],
    }
    assert classified["dimension_rename"] == {
        "restickify.x": "batchmatmul.out",
        "restickify.out": "batchmatmul.in",
        "restickify.mb": "batchmatmul.x",
    }
    assert classified["requires_staged_realization"]
    assert not classified["realized"]
    assert get_lx_relayout_inputs(consumer) == {}


def test_computed_layout_restickify_is_classified_but_not_realized():
    class DummyOp:
        pass

    consumer = DummyOp()
    _record_plan(
        consumer,
        LXRelayoutPlan(
            source_name="buf0",
            producer_name="restickify_buf",
            consumer_name="consumer",
            kind="layout_restickify_activation",
            producer_core_count=32,
            consumer_core_count=32,
            producer_core_id_to_device_slice={
                "0": {"0": 0},
                "1": {"0": 1},
            },
            producer_work_slice_dims={"0": 32},
            consumer_work_slice_dims={"0": 4, "1": 8},
            read_index=1,
            realized=False,
            communication_pattern="layout_transform_then_operand_broadcast",
            realization_strategy=("staged_lx_restickify_then_loop_scoped_input_fetch"),
            requires_staged_realization=True,
            unsupported_reason=(
                "computed activation restickify needs staged LX layout transform "
                "before loop-scoped matmul operand lowering"
            ),
        ),
    )

    classified = get_lx_relayout_classifications(consumer)["buf0"]
    assert classified["kind"] == "layout_restickify_activation"
    assert classified["communication_pattern"] == (
        "layout_transform_then_operand_broadcast"
    )
    assert classified["realization_strategy"] == (
        "staged_lx_restickify_then_loop_scoped_input_fetch"
    )
    assert classified["requires_staged_realization"]
    assert not classified["realized"]
    assert get_lx_relayout_inputs(consumer) == {}


def test_layout_restickify_transform_can_be_realized():
    class DummyOp:
        pass

    consumer = DummyOp()
    _record_plan(
        consumer,
        LXRelayoutPlan(
            source_name="buf0",
            producer_name="restickify_buf",
            consumer_name="consumer",
            kind="layout_restickify_activation",
            producer_core_count=32,
            consumer_core_count=32,
            producer_core_id_to_device_slice={
                "0": {"0": 0},
                "1": {"0": 1},
            },
            producer_work_slice_dims={"0": 32},
            consumer_work_slice_dims={"0": 32},
            read_index=0,
            realized=True,
            communication_pattern="layout_transform",
        ),
    )

    classified = get_lx_relayout_classifications(consumer)["buf0"]
    realized = get_lx_relayout_inputs(consumer)["buf0"]
    assert classified["kind"] == "layout_restickify_activation"
    assert classified["communication_pattern"] == "layout_transform"
    assert classified["realized"]
    assert realized == classified
    assert lx_relayout_needs_resident_reservation(realized)


def test_restickify_lx_op_emits_lx_to_lx_sdsc_contract():
    mb = Symbol("x0")
    out = Symbol("x1")
    op_spec = OpSpec(
        op=RESTICKIFY_LX_OP,
        is_reduction=False,
        iteration_space={mb: (Integer(512), 32), out: (Integer(12800), 1)},
        args=[
            _lx_restickify_arg(is_input=True, allocation={"lx": 0}),
            _lx_restickify_arg(is_input=False, allocation={"lx": 0x1000}),
        ],
        op_info={},
    )

    sdsc, _symbols, _affine_strides, _symbol_kinds = compile_op_spec(0, op_spec, [])

    root = sdsc[f"0_{RESTICKIFY_LX_OP}"]
    compute_dsc = root["dscs_"][0][RESTICKIFY_LX_OP]
    input_alloc, output_alloc = compute_dsc["scheduleTree_"]

    assert root["numCoresUsed_"] == 32
    assert input_alloc["component_"] == "lx"
    assert output_alloc["component_"] == "lx"
    assert compute_dsc["computeOp_"][0]["opFuncName"] == RESTICKIFY_LX_OP
    assert compute_dsc["labeledDs_"][0]["memOrg_"] == {"lx": {"isPresent": 1}}
    assert compute_dsc["labeledDs_"][1]["memOrg_"] == {"lx": {"isPresent": 1}}


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
            "realization_strategy": "",
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
