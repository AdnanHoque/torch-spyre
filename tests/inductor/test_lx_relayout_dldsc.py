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

import json

from sympy import Integer, Mod, Symbol, floor

from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import ComputedBuffer

from torch_spyre._inductor import config
from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.layout_allgather_restickify import (
    COMM_CLASS_ALL_GATHER,
    MATMUL_OPERAND_ALLGATHER_REPLICATE,
    MATMUL_OPERAND_BROADCAST,
    make_matmul_operand_allgather_contract,
)
from torch_spyre._inductor.lx_relayout import (
    LXRelayoutPlan,
    LXRelayoutTopology,
    _classify_coordinate_topology,
    _core_id_to_device_slice,
    _dense_core_id_to_device_slice,
    _dense_work_slice_dims,
    _prefer_matmul_operand_contract,
    _record_plan,
    get_lx_relayout_inputs,
    is_lx_relayout_reservation,
    make_lx_relayout_reservation_name,
    plan_lx_relayouts,
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


def test_coordinate_topology_classifies_one_to_one_scatter():
    topology = _classify_coordinate_topology(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {"1": {"0": 0}, "0": {"0": 1}},
        {"0": 2},
    )

    assert topology.communication_class == "scatter"
    assert topology.communication_pattern == "one_to_one"
    assert topology.max_fanout == 1
    assert topology.max_fanin == 1
    assert topology.transfer_count == 2


def test_dense_coordinate_payload_expands_unsplit_dims():
    dims = {"0", "1"}

    assert _dense_work_slice_dims({"1": 4}, dims) == {"0": 1, "1": 4}
    assert _dense_core_id_to_device_slice(
        {"7": {"1": 2}, "15": {}}, dims
    ) == {"7": {"0": 0, "1": 2}, "15": {"0": 0, "1": 0}}


def test_coordinate_topology_classifies_broadcast():
    topology = _classify_coordinate_topology(
        {"0": {"0": 0}},
        {"0": 1},
        {"0": {"0": 0}, "1": {"0": 1}, "2": {"0": 2}, "3": {"0": 3}},
        {"0": 4},
    )

    assert topology.communication_class == "broadcast"
    assert topology.communication_pattern == "one_to_many"
    assert topology.max_fanout == 4
    assert topology.max_fanin == 1
    assert topology.transfer_count == 4


def test_coordinate_topology_classifies_multicast():
    topology = _classify_coordinate_topology(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {
            "0": {"0": 0, "1": 0},
            "1": {"0": 0, "1": 1},
            "2": {"0": 1, "1": 0},
            "3": {"0": 1, "1": 1},
        },
        {"0": 2, "1": 2},
    )

    assert topology.communication_class == "multicast"
    assert topology.communication_pattern == "one_to_many"
    assert topology.max_fanout == 2
    assert topology.max_fanin == 1
    assert topology.transfer_count == 4


def test_coordinate_topology_classifies_gather():
    topology = _classify_coordinate_topology(
        {"0": {"0": 0}, "1": {"0": 1}, "2": {"0": 2}, "3": {"0": 3}},
        {"0": 4},
        {"0": {"0": 0}},
        {"0": 1},
    )

    assert topology.communication_class == "gather"
    assert topology.communication_pattern == "many_to_one"
    assert topology.max_fanout == 1
    assert topology.max_fanin == 4
    assert topology.transfer_count == 4


def test_coordinate_topology_classifies_all_gather():
    topology = _classify_coordinate_topology(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {"0": {"1": 0}, "1": {"1": 1}},
        {"1": 2},
    )

    assert topology.communication_class == "all_gather"
    assert topology.communication_pattern == "many_to_many"
    assert topology.max_fanout == 2
    assert topology.max_fanin == 2
    assert topology.transfer_count == 4


def test_matmul_operand_contract_is_preferred_for_rhs_all_gather(monkeypatch):
    monkeypatch.setattr(config, "lx_planner_relayout_collectives", True)
    monkeypatch.setattr(config, "lx_planner_relayout_matmul_operand_contract", True)
    topology = LXRelayoutTopology(
        COMM_CLASS_ALL_GATHER,
        "many_to_many",
        max_fanout=32,
        max_fanin=32,
        transfer_count=1024,
    )

    assert _prefer_matmul_operand_contract(1, topology)
    multicast_topology = LXRelayoutTopology(
        "multicast",
        "one_to_many",
        max_fanout=4,
        max_fanin=1,
        transfer_count=32,
    )
    assert _prefer_matmul_operand_contract(1, multicast_topology)
    assert not _prefer_matmul_operand_contract(0, topology)
    assert not _prefer_matmul_operand_contract(None, topology)


def test_matmul_operand_contract_marks_tensor1_all_gather_not_scatter():
    contract = make_matmul_operand_allgather_contract(
        producer_op="transpose",
        consumer_op="batchmatmul",
        read_index=1,
        producer_work_slice_dims={"2": 32},
        consumer_tensor_work_slice_dims={},
        consumer_compute_work_slice_dims={"mb": 32},
        communication_class=COMM_CLASS_ALL_GATHER,
    )

    assert contract["kind"] == MATMUL_OPERAND_BROADCAST
    assert contract["classification"] == MATMUL_OPERAND_BROADCAST
    assert contract["communication_class"] == COMM_CLASS_ALL_GATHER
    assert contract["communication_pattern"] == MATMUL_OPERAND_ALLGATHER_REPLICATE
    assert (
        contract["materialization_pattern"]
        == "all_gather_replicate_with_layout_conversion"
    )
    assert contract["requires_layout_conversion"]
    assert contract["requires_staged_realization"]
    assert contract["operand_role"] == "rhs"
    assert contract["operand_kernel_layout"] == {
        "layoutDimOrder_": ["out", "in", "x"],
        "stickDimOrder_": ["out"],
    }
    assert contract["layout_transform"] == {
        "kind": "activation_lx_to_matmul_kernel_operand",
        "source": "producer_lx_residency",
        "target": "consumer_matmul_kernel_operand",
        "source_coordinates": "producer_tensor_distribution",
        "target_coordinates": "consumer_compute_distribution",
        "carrier_hint": "lx_all_gather_then_local_restickify",
    }
    assert contract["staged_destination"] == {
        "component_": "KERNEL",
        "operand_read_index": 1,
        "scope": "matmul_transfer_loop",
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


def test_lx_relayout_plan_records_matmul_operand_all_gather_contract():
    class DummyOp:
        pass

    consumer = DummyOp()
    contract = make_matmul_operand_allgather_contract(
        producer_op="transpose",
        consumer_op="batchmatmul",
        read_index=1,
        producer_work_slice_dims={"2": 32},
        consumer_tensor_work_slice_dims={},
        consumer_compute_work_slice_dims={"mb": 32},
        communication_class=COMM_CLASS_ALL_GATHER,
    )
    _record_plan(
        consumer,
        LXRelayoutPlan(
            source_name="buf21",
            producer_name="buf21",
            consumer_name="buf22",
            kind=MATMUL_OPERAND_BROADCAST,
            producer_core_count=32,
            consumer_core_count=32,
            producer_core_id_to_device_slice={
                str(core): {"2": core} for core in range(32)
            },
            producer_work_slice_dims={"2": 32},
            consumer_work_slice_dims={},
            consumer_core_id_to_device_slice={str(core): {} for core in range(32)},
            read_index=1,
            realized=False,
            communication_class=COMM_CLASS_ALL_GATHER,
            communication_pattern=MATMUL_OPERAND_ALLGATHER_REPLICATE,
            max_fanout=32,
            max_fanin=32,
            transfer_count=1024,
            requires_staged_realization=True,
            layout_contract=contract,
            unsupported_reason="metadata only",
        ),
    )

    plan = get_lx_relayout_inputs(consumer)["buf21"]
    assert plan["kind"] == MATMUL_OPERAND_BROADCAST
    assert plan["communication_class"] == COMM_CLASS_ALL_GATHER
    assert plan["communication_pattern"] == MATMUL_OPERAND_ALLGATHER_REPLICATE
    assert plan["transfer_count"] == 1024
    assert plan["consumer_tensor_work_slice_dims"] == {}
    assert plan["consumer_compute_work_slice_dims"] == {"mb": 32}
    assert plan["materialization_pattern"] == (
        "all_gather_replicate_with_layout_conversion"
    )
    assert plan["requires_layout_conversion"]
    assert plan["layout_transform"]["kind"] == "activation_lx_to_matmul_kernel_operand"
    assert plan["staging_scope"] == "matmul_transfer_loop"


def test_bundle_enriches_matmul_operand_contract_with_source_target_layouts(
    tmp_path,
):
    mb = Symbol("c0")
    out = Symbol("c1")
    in_dim = Symbol("c2")
    producer_residency = {
        "0": {"0": 0},
        "1": {"0": 1},
        "2": {"0": 2},
        "3": {"0": 3},
    }
    contract = make_matmul_operand_allgather_contract(
        producer_op="mul",
        consumer_op="batchmatmul",
        read_index=1,
        producer_work_slice_dims={"0": 4},
        consumer_tensor_work_slice_dims={"0": 1},
        consumer_compute_work_slice_dims={"0": 4},
        communication_class=COMM_CLASS_ALL_GATHER,
    )
    contract.update(
        {
            "source_name": "buf0",
            "producer_name": "buf0",
            "consumer_name": "buf1",
            "producer_core_id_to_device_slice": producer_residency,
            "consumer_core_id_to_device_slice": {
                str(core): {"0": 0} for core in range(4)
            },
        }
    )

    producer = OpSpec(
        op="mul",
        is_reduction=False,
        iteration_space={mb: (Integer(64), 1), out: (Integer(512), 4)},
        op_info={},
        args=[
            TensorArg(
                is_input=True,
                arg_index=0,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[8, 64, 64],
                device_coordinates=[floor(out / 64), mb, Mod(out, 64)],
                allocation={"hbm": 0},
                name="arg0",
            ),
            TensorArg(
                is_input=True,
                arg_index=1,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[8, 64, 64],
                device_coordinates=[floor(out / 64), mb, Mod(out, 64)],
                allocation={"hbm": 0x1000},
                name="arg1",
            ),
            TensorArg(
                is_input=False,
                arg_index=-1,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[8, 64, 64],
                device_coordinates=[floor(out / 64), mb, Mod(out, 64)],
                allocation={"lx": 0},
                name="buf0",
            ),
        ],
    )
    consumer = OpSpec(
        op="batchmatmul",
        is_reduction=True,
        iteration_space={
            mb: (Integer(64), 4),
            out: (Integer(512), 1),
            in_dim: (Integer(64), 1),
        },
        op_info={"lx_relayout_classifications": [contract]},
        args=[
            TensorArg(
                is_input=True,
                arg_index=2,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[1, 64, 64],
                device_coordinates=[floor(in_dim / 64), mb, Mod(in_dim, 64)],
                allocation={"hbm": 0x2000},
                name="arg2",
            ),
            TensorArg(
                is_input=True,
                arg_index=-1,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[8, 64, 64],
                device_coordinates=[floor(out / 64), in_dim, Mod(out, 64)],
                allocation={"lx": 0},
                name="buf0",
                lx_residency_core_id_to_wk_slice=producer_residency,
            ),
            TensorArg(
                is_input=False,
                arg_index=3,
                device_dtype=DataFormats.SEN169_FP16,
                device_size=[8, 64, 64],
                device_coordinates=[floor(out / 64), mb, Mod(out, 64)],
                allocation={"hbm": 0x3000},
                name="buf1",
            ),
        ],
    )

    generate_bundle("lx_relayout_contract", str(tmp_path), [producer, consumer])

    sdsc_1 = json.loads((tmp_path / "sdsc_1.json").read_text())
    root = next(iter(sdsc_1.values()))
    classification = root["lxRelayoutClassifications_"][0]

    assert classification["source_lx_tensor"]["buffer_name"] == "buf0"
    assert classification["source_lx_tensor"]["dsName_"] == "Tensor2"
    assert classification["source_lx_tensor"]["dsType_"] == "OUTPUT"
    assert classification["source_lx_tensor"]["layoutDimOrder_"] == ["mb", "out"]
    assert classification["source_lx_tensor"]["stickDimOrder_"] == ["out"]
    assert classification["source_lx_tensor"]["dataFormat_"] == "SEN169_FP16"
    assert classification["source_lx_tensor"]["wordLength"] == 2
    assert classification["source_lx_tensor"]["startAddressCoreCorelet_"]
    assert classification["source_lx_tensor"]["coordinateInfo_"]
    assert classification["source_lx_tensor"]["coreIdToWkSlice_"] == {
        "0": {"mb": 0, "out": 0},
        "1": {"mb": 0, "out": 1},
        "2": {"mb": 0, "out": 2},
        "3": {"mb": 0, "out": 3},
    }
    assert classification["target_kernel_tensor"]["dsName_"] == "Tensor1"
    assert classification["target_kernel_tensor"]["dsType_"] == "KERNEL"
    assert classification["target_kernel_tensor"]["layoutDimOrder_"] == [
        "in",
        "out",
    ]
    assert classification["target_kernel_tensor"]["stickDimOrder_"] == ["out"]
    assert classification["target_kernel_tensor"]["dataFormat_"] == "SEN169_FP16"
    assert classification["target_kernel_tensor"]["wordLength"] == 2
    assert classification["target_kernel_tensor"]["startAddressCoreCorelet_"]
    assert classification["target_kernel_tensor"]["coordinateInfo_"]
    assert (
        classification["layout_transform"]["conversion_stage"]
        == "local_restickify_to_kernel"
    )


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


def test_matmul_operand_classification_metadata_is_emitted_top_level():
    mb = Symbol("x0")
    out = Symbol("x1")
    classification = [
        {
            "kind": MATMUL_OPERAND_BROADCAST,
            "communication_class": COMM_CLASS_ALL_GATHER,
            "communication_pattern": MATMUL_OPERAND_ALLGATHER_REPLICATE,
            "requires_staged_realization": True,
        }
    ]
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


def _compile_lx_relayout_contract(classification, producer_residency):
    mb = Symbol("x0")
    out = Symbol("x1")
    op_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={mb: (Integer(512), 1), out: (Integer(12800), 1)},
        args=[
            _fixed_tile_arg(
                is_input=True,
                allocation={"lx": 0},
                lx_residency_core_id_to_wk_slice=producer_residency,
            ),
            _fixed_tile_arg(is_input=False, allocation={"hbm": 0x1000}),
        ],
        op_info={"lx_relayout_classifications": [classification]},
    )

    sdsc, _symbols, _affine_strides, _symbol_kinds = compile_op_spec(0, op_spec, [])
    root = next(iter(sdsc.values()))
    compute_dsc = next(iter(root["dscs_"][0].values()))
    input_alloc = compute_dsc["scheduleTree_"][0]
    return root, input_alloc


def test_generic_gather_classification_and_producer_residency_emit_dldsc_contract():
    producer_residency = {
        "0": {"0": 0},
        "1": {"0": 1},
        "2": {"0": 2},
        "3": {"0": 3},
    }
    classification = {
        "kind": "gather",
        "source_name": "buf0",
        "producer_name": "producer",
        "consumer_name": "consumer",
        "producer_core_count": 4,
        "consumer_core_count": 1,
        "producer_core_id_to_device_slice": producer_residency,
        "producer_work_slice_dims": {"0": 4},
        "consumer_work_slice_dims": {"0": 1},
        "consumer_core_id_to_device_slice": {"0": {"0": 0}},
        "communication_class": "gather",
        "communication_pattern": "many_to_one",
        "max_fanout": 1,
        "max_fanin": 4,
        "transfer_count": 4,
    }

    root, input_alloc = _compile_lx_relayout_contract(
        classification, producer_residency
    )

    assert root["lxRelayoutClassifications_"] == [classification]
    assert input_alloc["component_"] == "lx"
    assert input_alloc["coordinates_"]["coreIdToWkSlice_"] == {
        "0": {"mb": 0, "out": 0},
        "1": {"mb": 1, "out": 0},
        "2": {"mb": 2, "out": 0},
        "3": {"mb": 3, "out": 0},
    }


def test_generic_broadcast_classification_and_producer_residency_emit_dldsc_contract():
    producer_residency = {"0": {"0": 0}}
    classification = {
        "kind": "broadcast",
        "source_name": "buf0",
        "producer_name": "producer",
        "consumer_name": "consumer",
        "producer_core_count": 1,
        "consumer_core_count": 4,
        "producer_core_id_to_device_slice": producer_residency,
        "producer_work_slice_dims": {"0": 1},
        "consumer_work_slice_dims": {"0": 4},
        "consumer_core_id_to_device_slice": {
            "0": {"0": 0},
            "1": {"0": 1},
            "2": {"0": 2},
            "3": {"0": 3},
        },
        "communication_class": "broadcast",
        "communication_pattern": "one_to_many",
        "max_fanout": 4,
        "max_fanin": 1,
        "transfer_count": 4,
    }

    root, input_alloc = _compile_lx_relayout_contract(
        classification, producer_residency
    )

    assert root["lxRelayoutClassifications_"] == [classification]
    assert input_alloc["coordinates_"]["coreIdToWkSlice_"] == {
        "0": {"mb": 0, "out": 0},
    }


def test_generic_multicast_classification_and_producer_residency_emit_dldsc_contract():
    producer_residency = {"0": {"0": 0}, "1": {"0": 1}}
    classification = {
        "kind": "multicast",
        "source_name": "buf0",
        "producer_name": "producer",
        "consumer_name": "consumer",
        "producer_core_count": 2,
        "consumer_core_count": 4,
        "producer_core_id_to_device_slice": producer_residency,
        "producer_work_slice_dims": {"0": 2},
        "consumer_work_slice_dims": {"0": 2, "1": 2},
        "consumer_core_id_to_device_slice": {
            "0": {"0": 0, "1": 0},
            "1": {"0": 0, "1": 1},
            "2": {"0": 1, "1": 0},
            "3": {"0": 1, "1": 1},
        },
        "communication_class": "multicast",
        "communication_pattern": "one_to_many",
        "max_fanout": 2,
        "max_fanin": 1,
        "transfer_count": 4,
    }

    root, input_alloc = _compile_lx_relayout_contract(
        classification, producer_residency
    )

    assert root["lxRelayoutClassifications_"] == [classification]
    assert input_alloc["coordinates_"]["coreIdToWkSlice_"] == {
        "0": {"mb": 0, "out": 0},
        "1": {"mb": 1, "out": 0},
    }


def test_generic_all_gather_classification_and_producer_residency_emit_dldsc_contract():
    producer_residency = {"0": {"0": 0}, "1": {"0": 1}}
    classification = {
        "kind": COMM_CLASS_ALL_GATHER,
        "source_name": "buf0",
        "producer_name": "producer",
        "consumer_name": "consumer",
        "producer_core_count": 2,
        "consumer_core_count": 2,
        "producer_core_id_to_device_slice": producer_residency,
        "producer_work_slice_dims": {"0": 2},
        "consumer_work_slice_dims": {"1": 2},
        "consumer_core_id_to_device_slice": {
            "0": {"1": 0},
            "1": {"1": 1},
        },
        "communication_class": COMM_CLASS_ALL_GATHER,
        "communication_pattern": "many_to_many",
        "max_fanout": 2,
        "max_fanin": 2,
        "transfer_count": 4,
    }

    root, input_alloc = _compile_lx_relayout_contract(
        classification, producer_residency
    )

    assert root["lxRelayoutClassifications_"] == [classification]
    assert input_alloc["coordinates_"]["coreIdToWkSlice_"] == {
        "0": {"mb": 0, "out": 0},
        "1": {"mb": 1, "out": 0},
    }


def test_partial_reduction_outputs_are_not_copy_relayout_candidates(monkeypatch):
    dep = MemoryDep("buf0", Symbol("i"), (Symbol("i"),), (Integer(4),))

    class FakeReadWrites:
        def __init__(self, *, reads=(), writes=()):
            self.reads = list(reads)
            self.writes = list(writes)

    producer = ComputedBuffer.__new__(ComputedBuffer)
    producer.name = "buf0"
    producer.data = object()
    producer.get_name = lambda: "buf0"
    producer.get_read_writes = lambda: FakeReadWrites(writes=(dep,))

    consumer = ComputedBuffer.__new__(ComputedBuffer)
    consumer.name = "consumer"
    consumer.data = object()
    consumer.get_name = lambda: "consumer"
    consumer.get_read_writes = lambda: FakeReadWrites(reads=(dep,))

    class FakeGraph:
        operations = [producer, consumer]

    def fake_per_core_view(op, _dep, _buf_name, _cache):
        assert op is producer
        view = PerCoreView(work_slice_dims=((0, 4),), core_to_slot=())
        return view, True

    monkeypatch.setattr(config, "lx_planner_relayout", True)
    monkeypatch.setattr(
        "torch_spyre._inductor.lx_relayout._per_core_view_on_buf",
        fake_per_core_view,
    )

    assert plan_lx_relayouts(FakeGraph()) == []

def _make_test_scratchpad_allocator():
    from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator

    class _TestScratchpadAllocator(ScratchpadAllocator):
        def plan_allocation(self, graph):
            return None

    return _TestScratchpadAllocator()


class _FakeDep:
    def __init__(self, name):
        self.name = name


def test_computed_source_clone_is_lx_relayout_eligible(monkeypatch):
    class FakeReadWrites:
        reads = [_FakeDep("producer")]

    class FakeComputedBuffer(ComputedBuffer):
        pass

    class FakeGraph:
        name_to_buffer = {"producer": FakeComputedBuffer.__new__(FakeComputedBuffer)}

    class FakeTarget:
        __name__ = "clone"

    op = FakeComputedBuffer.__new__(FakeComputedBuffer)
    op.origin_node = type("Origin", (), {"target": FakeTarget()})()
    op.get_read_writes = lambda: FakeReadWrites()

    monkeypatch.setattr(config, "lx_planner_relayout_collectives", True)
    allocator = _make_test_scratchpad_allocator()
    assert allocator._clone_output_good_for_lx_relayout(FakeGraph(), op)


def test_graph_input_clone_is_not_lx_relayout_eligible(monkeypatch):
    class FakeReadWrites:
        reads = [_FakeDep("arg0")]

    class FakeGraph:
        name_to_buffer = {}

    class FakeTarget:
        __name__ = "clone"

    op = ComputedBuffer.__new__(ComputedBuffer)
    op.origin_node = type("Origin", (), {"target": FakeTarget()})()
    op.get_read_writes = lambda: FakeReadWrites()

    monkeypatch.setattr(config, "lx_planner_relayout_collectives", True)
    allocator = _make_test_scratchpad_allocator()
    assert not allocator._clone_output_good_for_lx_relayout(FakeGraph(), op)
