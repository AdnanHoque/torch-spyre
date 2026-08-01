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

from dataclasses import replace
from types import SimpleNamespace

import pytest
from sympy import Integer, Mod, Symbol, floor

import torch_spyre._inductor.scratchpad.allocator as allocator_module

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.compute_ops import gen_coord_info_value
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.lx_relayout import (
    LX_RELAYOUT_ATTR,
    LXRelayoutPlan,
    _destination_size_ratio,
    _same_core_placement,
    make_lx_relayout_destination_name,
)
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.scratchpad.allocator import (
    ScratchpadAllocator,
    _as_core_division_buffers,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.plan_solver import (
    BufferType,
    GreedyLayoutSolver,
    LifetimeBoundBuffer,
)
from torch_spyre._inductor.spyre_kernel import (
    _bind_lx_matmul_operand_broadcast,
    _materialize_explicit_lx_shuffle,
    simplify_op_spec,
)


class _DummyOp:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class _DummyGraph:
    def __init__(self, *names, graph_input_names=(), graph_output_names=()):
        self.operations = [_DummyOp(name) for name in names]
        self.name_to_buffer = {op.name: op for op in self.operations}
        self.graph_input_names = list(graph_input_names)
        self._graph_output_names = list(graph_output_names)

    def try_get_buffer(self, name):
        return self.name_to_buffer.get(name)

    def get_output_names(self):
        return self._graph_output_names


def _all_gather_plan() -> LXRelayoutPlan:
    producer_map = {str(core): {"0": core % 4, "1": core // 4} for core in range(32)}
    consumer_map = {str(core): {"0": core % 4, "1": 0} for core in range(32)}
    return LXRelayoutPlan(
        source_name="buf_k",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 4, "1": 1},
        destination_size_ratio=8,
    )


def _shuffle_plan() -> LXRelayoutPlan:
    producer_map = {str(core): {"0": core} for core in range(4)}
    consumer_map = {str(core): {"0": 3 - core} for core in range(4)}
    return LXRelayoutPlan(
        source_name="buf_x",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4},
        destination_device_dim_splits={"0": 4},
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )


def _assert_transfer_layout(allocation, destination_name):
    by_name = {buffer.name: buffer for buffer in allocation}
    assert all(buffer.address is not None for buffer in allocation)
    source = by_name["buf_k"]
    destination = by_name[destination_name]
    scores = by_name["scores"]
    assert not (
        source.address < destination.address + destination.size
        and destination.address < source.address + source.size
    )
    assert (
        source.address < scores.address + scores.size
        and scores.address < source.address + source.size
    )


def test_coordinate_geometry_accepts_bounded_shuffle_geometry_only():
    shuffle = _destination_size_ratio(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {"0": {"0": 1}, "1": {"0": 0}},
        {"0": 2},
    )
    all_gather = _destination_size_ratio(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {"0": {"0": 0}, "1": {"0": 0}},
        {"0": 1},
    )
    broadcast = _destination_size_ratio(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {
            "0": {"0": 0},
            "1": {"0": 0},
            "2": {"0": 1},
            "3": {"0": 1},
        },
        {"0": 2},
    )
    unsupported = _destination_size_ratio(
        {
            "0": {"0": 0, "1": 0},
            "1": {"0": 0, "1": 1},
            "2": {"0": 1, "1": 0},
            "3": {"0": 1, "1": 1},
        },
        {"0": 2, "1": 2},
        {str(core): {"0": core, "1": 0} for core in range(4)},
        {"0": 4, "1": 1},
    )

    assert shuffle == 1
    assert all_gather == 2
    assert broadcast == 1
    assert unsupported is None


def test_equal_logical_view_on_different_core_counts_requires_relayout():
    view = PerCoreView(work_slice_dims=((0, 8),), core_to_slot=())

    assert _same_core_placement(view, 8, view, 8)
    assert not _same_core_placement(view, 8, view, 32)


def test_shuffle_corelet_fold_supports_nonstick_dimension():
    coord = gen_coord_info_value(
        size=128,
        nsplits=1,
        elems_per_stick=64,
        is_stick_dim=False,
        split_shuffle_corelets=True,
    )
    folds = coord["folds"]
    assert folds["dim_prop_attr"][1] == {
        "factor_": 2,
        "label_": "corelet_fold",
    }
    assert folds["dim_prop_func"][1]["Affine"]["alpha_"] == 64


def test_expanding_geometry_is_allocated_atomically_or_falls_back():
    graph = _DummyGraph("producer", "independent_1", "independent_2", "consumer")
    # GraphEditor may insert operations without updating GraphLowering's buffer map.
    graph.name_to_buffer.pop("consumer")
    plan = _all_gather_plan()
    source = LifetimeBoundBuffer("buf_k", size=128 * 1024, uses=[0, 3])
    consumer_output = LifetimeBoundBuffer(
        "scores", size=512 * 1024, uses=[3, 4], in_place_parents=["buf_k"]
    )
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    buffers = [source, consumer_output]
    allocator._append_lx_relayout_destinations(graph, buffers)

    destination = next(
        buffer
        for buffer in buffers
        if buffer.name == make_lx_relayout_destination_name("buf_k")
    )
    assert destination.size == 1024 * 1024
    assert source.uses == [0, 3]
    assert destination.uses == [3, 4]
    assert consumer_output.uses == [4, 5]
    assert consumer_output.in_place_parents == []
    converted = _as_core_division_buffers(
        buffers, _DummyGraph(), allocator._lx_relayout_plans_by_source
    )
    layout_planning = allocator._layout_planner_for_buffers(converted)
    assert isinstance(layout_planning, FirstFitLayoutSolver)
    allocation = allocator._plan_layout_with_atomic_relayouts(
        layout_planning, converted
    )
    _assert_transfer_layout(allocation, destination.name)
    allocator._record_successful_lx_relayouts(graph, allocation)
    recorded = getattr(graph.operations[3], LX_RELAYOUT_ATTR)["buf_k"]
    assert recorded.destination_lx_address == next(
        buffer.address for buffer in allocation if buffer.name == destination.name
    )

    for buffer in converted:
        buffer.address = None
    fallback_allocator = ScratchpadAllocator(GreedyLayoutSolver(1024 * 1024))
    fallback_allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    allocation = fallback_allocator._plan_layout_with_atomic_relayouts(
        FirstFitLayoutSolver(1024 * 1024), converted
    )
    by_name = {buffer.name: buffer for buffer in allocation}
    assert by_name["buf_k"].address is None
    assert by_name[destination.name].address is None
    assert by_name["scores"].address is not None
    fallback_allocator._record_successful_lx_relayouts(graph, allocation)
    assert not hasattr(graph.operations[3], LX_RELAYOUT_ATTR)


def test_resident_matmul_broadcast_aliases_destination_to_source():
    graph = _DummyGraph("producer", "consumer")
    producer_map = {
        str(core): {"0": cohort} for cohort, core in enumerate((0, 8, 16, 24))
    }
    consumer_map = {str(core): {"0": core // 8} for core in range(32)}
    plan = LXRelayoutPlan(
        source_name="buf_a",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4},
        destination_device_dim_splits={"0": 4},
        destination_size_ratio=1,
        destination_aliases_source=True,
    )
    source = LifetimeBoundBuffer("buf_a", size=1024 * 1024, uses=[0, 1])
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_a": plan}
    buffers = [source]

    allocator._append_lx_relayout_destinations(graph, buffers)

    assert [buffer.name for buffer in buffers] == ["buf_a"]
    assert source.uses == [0, 1, 2]
    allocation = allocator._plan_layout_with_atomic_relayouts(
        GreedyLayoutSolver(1536 * 1024), buffers
    )
    assert allocation[0].address is not None
    allocator._record_successful_lx_relayouts(graph, allocation)
    recorded = getattr(graph.operations[1], LX_RELAYOUT_ATTR)["buf_a"]
    assert recorded.destination_lx_address == allocation[0].address


def test_core_division_conversion_preserves_graph_boundaries():
    graph = _DummyGraph(graph_input_names=("input",), graph_output_names=("output",))
    buffers = [
        LifetimeBoundBuffer("input", size=128, uses=[0, 1], first_use_is_read=True),
        LifetimeBoundBuffer("intermediate", size=128, uses=[1, 2]),
        LifetimeBoundBuffer("output", size=128, uses=[2, 3]),
    ]

    converted = _as_core_division_buffers(buffers, graph)

    boundaries = {buffer.name: buffer.boundary for buffer in converted}
    assert boundaries == {
        "input": BufferType.Input,
        "intermediate": BufferType.Intermediate,
        "output": BufferType.Output,
    }


def test_relayout_keeps_producer_inputs_live_through_shuffle(monkeypatch):
    graph = _DummyGraph("producer_input", "buf_k", "consumer")
    plan = _all_gather_plan()
    producer_input = LifetimeBoundBuffer("producer_input", size=128, uses=[0, 1])
    source = LifetimeBoundBuffer("buf_k", size=128, uses=[1, 2])
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    monkeypatch.setattr(
        allocator_module,
        "op_read_writes",
        lambda _: SimpleNamespace(reads=[SimpleNamespace(name="producer_input")]),
    )

    allocator._append_lx_relayout_destinations(graph, [producer_input, source])

    assert producer_input.uses == [0, 1, 2]


def test_all_gather_emits_standard_shuffle_fold_geometry():
    h = Symbol("h")
    lq = Symbol("lq")
    lk = Symbol("lk")
    d = Symbol("d")
    plan = _all_gather_plan()
    plan = replace(plan, destination_lx_address=0x44000)
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 4096, 2, 64],
        device_coordinates=[h, lk, floor(d / 64), Mod(d, 64)],
        allocation={"lx": 0x24000},
        name="buf_k",
    )
    consumer_spec = OpSpec(
        op=BATCH_MATMUL_OP,
        is_reduction=True,
        iteration_space={
            h: (Integer(4), 4),
            lq: (Integer(512), 8),
            lk: (Integer(4096), 1),
            d: (Integer(128), 1),
        },
        args=[source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(
        source_arg,
        consumer_spec,
        plan,
    )
    assert result is not None
    shuffle_spec, consumer_arg = result
    assert consumer_arg.allocation == {"lx": 0x44000}
    assert shuffle_spec.num_cores_override == 32
    assert (
        shuffle_spec.args[0].allocation_core_id_to_device_slice
        == plan.source_core_id_to_device_slice
    )
    assert (
        shuffle_spec.args[1].allocation_core_id_to_device_slice
        == plan.destination_core_id_to_device_slice
    )

    simplify_op_spec(shuffle_spec)
    sdsc, *_ = compile_op_spec(0, shuffle_spec, [])
    root = next(iter(sdsc.values()))
    assert len(root["coreIdToWkSlice_"]) == 32
    shuffle_dsc = next(iter(root["dscs_"][0].values()))
    allocations = [
        row for row in shuffle_dsc["scheduleTree_"] if row["nodeType_"] == "allocate"
    ]

    input_map = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    output_map = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert input_map["0"] != input_map["4"]
    assert output_map["0"] == output_map["4"]

    input_out = allocations[0]["coordinates_"]["coordInfo"]["out"]["folds"]
    output_out = allocations[1]["coordinates_"]["coordInfo"]["out"]["folds"]
    assert input_out["dim_prop_attr"][0]["factor_"] == 8
    assert input_out["dim_prop_func"][0]["Affine"]["alpha_"] == 512
    assert output_out["dim_prop_attr"][0]["factor_"] == 1
    assert output_out["dim_prop_func"][0]["Affine"]["alpha_"] == 4096

    for allocation in allocations:
        out_folds = allocation["coordinates_"]["coordInfo"]["out"]["folds"]
        assert out_folds["dim_prop_attr"][1] == {
            "factor_": 1,
            "label_": "corelet_fold",
        }
        assert out_folds["dim_prop_func"][1]["Affine"]["alpha_"] == 0

        stick_folds = allocation["coordinates_"]["coordInfo"]["in"]["folds"]
        assert stick_folds["dim_prop_attr"][1] == {
            "factor_": 2,
            "label_": "corelet_fold",
        }
        assert stick_folds["dim_prop_func"][1]["Affine"]["alpha_"] == 64

    shuffle_spec.args[0].allocation = {"hbm": 0}
    with pytest.raises(ValueError, match="requires LX storage"):
        compile_op_spec(0, shuffle_spec, [])


def test_matmul_broadcast_preserves_sparse_source_ownership():
    x = Symbol("x")
    producer_map = {
        str(core): {"0": cohort} for cohort, core in enumerate((0, 8, 16, 24))
    }
    consumer_map = {str(core): {"0": core // 8} for core in range(32)}
    plan = LXRelayoutPlan(
        source_name="buf_a",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4},
        destination_device_dim_splits={"0": 4},
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 64],
        device_coordinates=[floor(x / 64), Mod(x, 64)],
        allocation={"lx": 0x24000},
        name="buf_a",
    )
    consumer = OpSpec(
        op=BATCH_MATMUL_OP,
        is_reduction=True,
        iteration_space={x: (Integer(256), 4)},
        args=[source_arg],
        op_info={},
    )

    result = _bind_lx_matmul_operand_broadcast(source_arg, consumer, plan, 0)

    assert result is not None
    assert result.allocation == {"lx": 0x24000}
    assert result.allocation_core_id_to_device_slice == producer_map
    assert result.allocation_device_dim_splits == {"0": 4}
    classifications = consumer.op_info["lx_relayout_classifications"]
    assert classifications == [
        {
            "source_name": "buf_a",
            "producer_name": "buf_a",
            "consumer_name": "consumer",
            "kind": "matmul_operand_broadcast",
            "producer_core_count": 4,
            "consumer_core_count": 32,
            "producer_core_id_to_device_slice": producer_map,
            "producer_work_slice_dims": {"0": 4},
            "consumer_work_slice_dims": {"0": 4},
            "consumer_tensor_work_slice_dims": {"0": 4},
            "consumer_core_id_to_device_slice": consumer_map,
            "read_index": 0,
            "operand_index": 0,
            "consumer_operand_ds_type": "INPUT",
            "realized": False,
            "communication_class": "all_gather",
            "communication_pattern": "all_gather_replicate",
            "transfer_count": 32,
            "requires_staged_realization": True,
            "requires_layout_conversion": False,
            "estimated_tensor_bytes": 512,
        }
    ]


def test_matmul_broadcast_rejects_expanding_all_gather():
    x = Symbol("x")
    plan = LXRelayoutPlan(
        source_name="buf_a",
        consumer_name="consumer",
        source_core_id_to_device_slice={"0": {"0": 0}, "1": {"0": 1}},
        destination_core_id_to_device_slice={"0": {}, "1": {}},
        source_device_dim_splits={"0": 2},
        destination_device_dim_splits={},
        destination_size_ratio=2,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 64],
        device_coordinates=[floor(x / 64), Mod(x, 64)],
        allocation={"lx": 0x24000},
        name="buf_a",
    )
    consumer = OpSpec(
        op=BATCH_MATMUL_OP,
        is_reduction=True,
        iteration_space={x: (Integer(128), 2)},
        args=[source_arg],
        op_info={},
    )

    assert _bind_lx_matmul_operand_broadcast(source_arg, consumer, plan, 0) is None
    assert "lx_relayout_classifications" not in consumer.op_info


def test_one_to_one_geometry_emits_standard_shuffle():
    x = Symbol("x")
    plan = _shuffle_plan()
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 64],
        device_coordinates=[floor(x / 64), Mod(x, 64)],
        allocation={"lx": 0x24000},
        name="buf_x",
    )
    consumer_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={x: (Integer(256), 4)},
        args=[source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(
        source_arg,
        consumer_spec,
        plan,
    )
    assert result is not None
    shuffle_spec, _ = result

    simplify_op_spec(shuffle_spec)
    sdsc, *_ = compile_op_spec(0, shuffle_spec, [])
    root = next(iter(sdsc.values()))
    shuffle_dsc = next(iter(root["dscs_"][0].values()))
    allocations = [
        row for row in shuffle_dsc["scheduleTree_"] if row["nodeType_"] == "allocate"
    ]
    assert allocations[0]["coordinates_"]["coreIdToWkSlice_"]["0"] == {"out": 0}
    assert allocations[1]["coordinates_"]["coreIdToWkSlice_"]["0"] == {"out": 3}
