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

from sympy import Integer, Mod, Symbol, floor

import torch_spyre._inductor.lx_relayout as lx_relayout_module
import torch_spyre._inductor.scratchpad.allocator as allocator_module

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.lx_relayout import (
    LX_RELAYOUT_ATTR,
    LXCollectiveKind,
    LXRelayoutPlan,
    _classify_lx_relayout,
    _destination_size_ratio,
    collect_lx_relayout_plans,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import PerCoreView, copy_fx_custom_meta
from torch_spyre._inductor.propagate_hints import get_gather_dim
from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.spyre_kernel import (
    _materialize_explicit_lx_shuffle,
    _materialize_lx_relayout_inputs,
    simplify_op_spec,
)


class _DummyOp:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class _DummyGraph:
    def __init__(self, *names):
        self.operations = [_DummyOp(name) for name in names]
        self.name_to_buffer = {op.name: op for op in self.operations}

    def try_get_buffer(self, name):
        return self.name_to_buffer.get(name)


def _all_gather_plan() -> LXRelayoutPlan:
    producer_map = {str(core): {"0": core // 8, "1": core % 8} for core in range(32)}
    consumer_map = {str(core): {"0": core // 8, "1": 0} for core in range(32)}
    return LXRelayoutPlan(
        source_name="buf_k",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 4, "1": 1},
        collective_kind=LXCollectiveKind.ALL_GATHER,
        destination_size_ratio=8,
    )


def _granite_mlp_all_to_all_plan() -> LXRelayoutPlan:
    producer_map = {str(core): {"0": core // 8, "1": core % 8} for core in range(32)}
    consumer_map = {str(core): {"0": core, "1": 0} for core in range(32)}
    return LXRelayoutPlan(
        source_name="buf_mlp",
        consumer_name="pointwise",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 32, "1": 1},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )


def _broadcast_plan() -> LXRelayoutPlan:
    producer_map = {"0": {}}
    consumer_map = {str(core): {} for core in range(32)}
    return LXRelayoutPlan(
        source_name="broadcast_source",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={},
        destination_device_dim_splits={},
        collective_kind=LXCollectiveKind.BROADCAST,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )


def _overlap(lhs, rhs):
    return lhs.address < rhs.address + rhs.size and rhs.address < lhs.address + lhs.size


def _compile_shuffle(shuffle_spec):
    simplify_op_spec(shuffle_spec)
    sdsc, *_ = compile_op_spec(0, shuffle_spec, [])
    root = next(iter(sdsc.values()))
    shuffle_dsc = next(iter(root["dscs_"][0].values()))
    allocations = [
        row for row in shuffle_dsc["scheduleTree_"] if row["nodeType_"] == "allocate"
    ]
    return root, allocations


def test_gather_dim_hint_survives_metadata_copy():
    src = SimpleNamespace(meta={"custom": {"_hint_1": {"gather_dim": "Lq"}}})
    dst = SimpleNamespace(meta={"custom": {"_hint_2": {"work_div": {"H": 4}}}})
    copy_fx_custom_meta(src, dst)

    lq = Symbol("lq")
    op = SimpleNamespace(origins=[dst], work_div_loop_info={lq: ["Lq"]})
    assert get_gather_dim(op) == lq
    assert set(dst.meta["custom"]) == {"_hint_1", "_hint_2"}


def test_all_to_all_geometry_and_emission():
    producer_map = {"0": {"0": 0}, "1": {"0": 1}}
    consumer_map = {"0": {"0": 1}, "1": {"0": 0}}
    all_to_all = _classify_lx_relayout(
        producer_map,
        {"0": 2},
        consumer_map,
        {"0": 2},
    )
    all_gather = _classify_lx_relayout(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {"0": {"0": 0}, "1": {"0": 0}},
        {"0": 1},
    )

    assert all_to_all is not None
    assert all_to_all.collective_kind is LXCollectiveKind.ALL_TO_ALL
    assert all_to_all.destination_size_ratio == 1
    assert all_gather is not None
    assert all_gather.collective_kind is LXCollectiveKind.ALL_GATHER
    assert all_gather.destination_size_ratio == 2

    x = Symbol("x")
    plan = LXRelayoutPlan(
        source_name="buf_x",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 2},
        destination_device_dim_splits={"0": 2},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 64],
        device_coordinates=[floor(x / 64), Mod(x, 64)],
        allocation={"lx": 0x24000},
        name="buf_x",
    )
    consumer_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={x: (Integer(128), 2)},
        args=[source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)
    assert result is not None
    _, allocations = _compile_shuffle(result[0])
    assert allocations[0]["coordinates_"]["coreIdToWkSlice_"]["0"] == {"out": 0}
    assert allocations[1]["coordinates_"]["coreIdToWkSlice_"]["0"] == {"out": 1}

    node = SimpleNamespace(**{LX_RELAYOUT_ATTR: {"buf_x": plan}})
    current_node = SimpleNamespace(get_nodes=lambda: [SimpleNamespace(node=node)])
    args = [source_arg, source_arg]
    prefix = _materialize_lx_relayout_inputs(
        current_node,
        args,
        [(0, source_arg), (1, source_arg)],
        consumer_spec,
    )
    assert len(prefix) == 1
    assert args[0].name == args[1].name == plan.destination_name


def test_broadcast_geometry_requires_one_physical_source_owner():
    producer_map = {"0": {}}
    consumer_map = {str(core): {} for core in range(32)}

    classification = _classify_lx_relayout(
        producer_map,
        {},
        consumer_map,
        {},
    )

    assert classification is not None
    assert classification.collective_kind is LXCollectiveKind.BROADCAST
    assert classification.destination_size_ratio == 1

    # Repeated producer owners would falsely claim the value already exists on
    # every core. They are neither a partition nor a valid broadcast source.
    padded_producer_map = {str(core): {} for core in range(32)}
    assert (
        _classify_lx_relayout(
            padded_producer_map,
            {},
            consumer_map,
            {},
        )
        is None
    )

    # Multiple independently replicated source shards are multicast cohorts,
    # which are not yet supported by this planner.
    multicast_producer_map = {"0": {"0": 0}, "1": {"0": 1}}
    multicast_consumer_map = {
        str(core): {"0": 0 if core < 16 else 1} for core in range(32)
    }
    assert (
        _classify_lx_relayout(
            multicast_producer_map,
            {"0": 2},
            multicast_consumer_map,
            {"0": 2},
        )
        is None
    )


def test_planner_keeps_equal_view_edge_when_core_counts_require_broadcast(monkeypatch):
    class _FakePointwise:
        pass

    class _FakeBuffer:
        def __init__(self, name):
            self.name = name
            self.data = _FakePointwise()

        def get_name(self):
            return self.name

    class _FakeDep:
        def __init__(self, name):
            self.name = name

    producer = _FakeBuffer("broadcast_source")
    consumer = _FakeBuffer("consumer")
    source_write = _FakeDep(producer.name)
    source_read = _FakeDep(producer.name)
    consumer_write = _FakeDep(consumer.name)
    graph = SimpleNamespace(operations=[producer, consumer])
    whole_buffer_view = PerCoreView(work_slice_dims=(), core_to_slot=())

    def read_writes(op):
        if op is producer:
            return SimpleNamespace(reads=[], writes=[source_write])
        return SimpleNamespace(reads=[source_read], writes=[consumer_write])

    monkeypatch.setattr(lx_relayout_module.config, "lx_planner_relayout", True)
    monkeypatch.setattr(lx_relayout_module, "ComputedBuffer", _FakeBuffer)
    monkeypatch.setattr(lx_relayout_module, "Pointwise", _FakePointwise)
    monkeypatch.setattr(lx_relayout_module, "MemoryDep", _FakeDep)
    monkeypatch.setattr(lx_relayout_module, "op_read_writes", read_writes)
    monkeypatch.setattr(lx_relayout_module, "_is_matmul_op", lambda _: False)
    monkeypatch.setattr(
        lx_relayout_module,
        "_per_core_view_on_buf",
        lambda *_: (whole_buffer_view, False, True),
    )
    monkeypatch.setattr(
        lx_relayout_module,
        "_op_num_cores",
        lambda op: 1 if op is producer else 32,
    )

    plans = collect_lx_relayout_plans(graph)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.collective_kind is LXCollectiveKind.BROADCAST
    assert plan.source_core_id_to_device_slice == {"0": {}}
    assert len(plan.destination_core_id_to_device_slice) == 32
    assert all(
        not per_core for per_core in plan.destination_core_id_to_device_slice.values()
    )


def test_broadcast_emits_one_to_all_standard_shuffle():
    m = Symbol("m")
    n = Symbol("n")
    plan = _broadcast_plan()
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 64],
        device_coordinates=[floor(n / 64), Mod(n, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={m: (Integer(32), 32), n: (Integer(128), 1)},
        args=[source_arg, source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, consumer_arg = result
    assert shuffle_spec.op == "shuffle"
    assert shuffle_spec.num_cores_override == 32
    assert consumer_arg.allocation == {"lx": plan.destination_lx_address}
    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32

    producer_geometry = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    consumer_geometry = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert producer_geometry == {"0": {"out": 0}}
    assert len(consumer_geometry) == 32
    assert all(per_core == {"out": 0} for per_core in consumer_geometry.values())


def test_dense_common_refinement_geometries():
    producer_8x4 = {str(core): {"0": core // 4, "1": core % 4} for core in range(32)}
    partition_4x8 = {str(core): {"0": core // 8, "1": core % 8} for core in range(32)}
    partition_32x1 = {str(core): {"0": core, "1": 0} for core in range(32)}

    assert (
        _destination_size_ratio(
            producer_8x4,
            {"0": 8, "1": 4},
            partition_4x8,
            {"0": 4, "1": 8},
        )
        == 1
    )
    assert (
        _destination_size_ratio(
            partition_4x8,
            {"0": 4, "1": 8},
            partition_32x1,
            {"0": 32, "1": 1},
        )
        == 1
    )

    incomplete_partition = {
        str(core): {"0": core // 8, "1": core % 8} for core in range(32)
    }
    assert (
        _destination_size_ratio(
            incomplete_partition,
            {"0": 8, "1": 8},
            partition_32x1,
            {"0": 32, "1": 1},
        )
        is None
    )


def test_granite_mlp_common_refinement_emits_standard_shuffle():
    m = Symbol("m")
    n = Symbol("n")
    plan = _granite_mlp_all_to_all_plan()
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 200, 64],
        device_coordinates=[m, floor(n / 64), Mod(n, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="silu",
        is_reduction=False,
        iteration_space={m: (Integer(512), 32), n: (Integer(12800), 1)},
        args=[source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, consumer_arg = result
    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32
    assert consumer_arg.name == plan.destination_name
    assert consumer_arg.allocation == {"lx": plan.destination_lx_address}

    producer_geometry = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    consumer_geometry = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert producer_geometry["0"] == {"mb": 0, "out": 0}
    assert producer_geometry["31"] == {"mb": 3, "out": 7}
    assert consumer_geometry["0"] == {"mb": 0, "out": 0}
    assert consumer_geometry["31"] == {"mb": 31, "out": 0}


def test_expanding_geometry_is_allocated_atomically_or_falls_back():
    graph = _DummyGraph("producer", "independent_1", "independent_2", "consumer")
    # GraphEditor may insert operations without updating GraphLowering's buffer map.
    graph.name_to_buffer.pop("consumer")
    plan = _all_gather_plan()
    source = LifetimeBoundBuffer("buf_k", size=128 * 1024, uses=[0, 3])
    consumer_output = LifetimeBoundBuffer(
        "scores", size=512 * 1024, uses=[3, 4], in_place_parents=["buf_k"]
    )
    barred = LifetimeBoundBuffer(
        "barred", size=128, uses=[0, 4], residency_reason="op not allowed"
    )
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    buffers = [barred, source, consumer_output]
    allocator._append_lx_relayout_destinations(graph, buffers)

    destination = {buffer.name: buffer for buffer in buffers}[plan.destination_name]
    assert destination.size == 1024 * 1024
    assert source.uses == [0, 3]
    assert destination.uses == [3, 4]
    assert consumer_output.uses == [4, 5]
    assert consumer_output.in_place_parents == []
    layout_planning = allocator._layout_planner_for_buffers(buffers)
    assert isinstance(layout_planning, FirstFitLayoutSolver)
    allocation = allocator._plan_layout_with_atomic_relayouts(layout_planning, buffers)
    by_name = {buffer.name: buffer for buffer in allocation}
    assert by_name["barred"].address is None
    assert by_name["barred"].residency_reason == "op not allowed"
    assert all(
        buffer.address is not None for buffer in allocation if buffer.name != "barred"
    )
    assert not _overlap(by_name["buf_k"], by_name[plan.destination_name])
    assert _overlap(by_name["buf_k"], by_name["scores"])
    allocator._record_successful_lx_relayouts(graph, allocation)
    recorded = getattr(graph.operations[3], LX_RELAYOUT_ATTR)["buf_k"]
    assert recorded.destination_lx_address == by_name[plan.destination_name].address

    for buffer in buffers:
        buffer.address = None
    fallback_allocator = ScratchpadAllocator(GreedyLayoutSolver(1024 * 1024))
    fallback_allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    allocation = fallback_allocator._plan_layout_with_atomic_relayouts(
        FirstFitLayoutSolver(1024 * 1024), buffers
    )
    by_name = {buffer.name: buffer for buffer in allocation}
    assert by_name["buf_k"].address is None
    assert by_name[plan.destination_name].address is None
    assert by_name["scores"].address is not None
    fallback_allocator._record_successful_lx_relayouts(graph, allocation)
    assert not hasattr(graph.operations[3], LX_RELAYOUT_ATTR)


def test_broadcast_destination_is_full_sized_and_disjoint():
    graph = _DummyGraph("producer", "consumer")
    plan = _broadcast_plan()
    source = LifetimeBoundBuffer(plan.source_name, size=128 * 1024, uses=[0, 1])
    consumer_output = LifetimeBoundBuffer(
        "consumer_output",
        size=128 * 1024,
        uses=[1, 2],
        in_place_parents=[plan.source_name],
    )
    allocator = ScratchpadAllocator(GreedyLayoutSolver(384 * 1024))
    allocator._lx_relayout_plans_by_source = {plan.source_name: plan}
    buffers = [source, consumer_output]

    allocator._append_lx_relayout_destinations(graph, buffers)

    destination = {buffer.name: buffer for buffer in buffers}[plan.destination_name]
    assert destination.size == source.size
    assert source.uses == [0, 1]
    assert destination.uses == [1, 2]
    assert consumer_output.uses == [2, 3]
    assert consumer_output.in_place_parents == []

    layout_planning = allocator._layout_planner_for_buffers(buffers)
    allocation = allocator._plan_layout_with_atomic_relayouts(layout_planning, buffers)
    by_name = {buffer.name: buffer for buffer in allocation}
    assert all(buffer.address is not None for buffer in allocation)
    assert not _overlap(by_name[plan.source_name], by_name[plan.destination_name])

    allocator._record_successful_lx_relayouts(graph, allocation)
    recorded = getattr(graph.operations[1], LX_RELAYOUT_ATTR)[plan.source_name]
    assert recorded.collective_kind is LXCollectiveKind.BROADCAST
    assert recorded.destination_lx_address == destination.address


def test_relayout_keeps_producer_inputs_live_through_shuffle(monkeypatch):
    graph = _DummyGraph("producer_input", "buf_k", "consumer")
    plan = _all_gather_plan()
    producer_input = LifetimeBoundBuffer("producer_input", size=128, uses=[0, 1])
    source = LifetimeBoundBuffer("buf_k", size=128, uses=[1, 2])
    producer_child = LifetimeBoundBuffer(
        "producer_child", size=128, uses=[1, 2], in_place_parents=["producer_input"]
    )
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    monkeypatch.setattr(
        allocator_module,
        "op_read_writes",
        lambda _: SimpleNamespace(reads=[SimpleNamespace(name="producer_input")]),
    )

    allocator._append_lx_relayout_destinations(
        graph, [producer_input, source, producer_child]
    )

    assert producer_input.uses == [0, 1, 2]
    assert producer_child.in_place_parents == []


def test_partial_relayout_fallback_keeps_complete_pairs():
    complete_plan = replace(
        _all_gather_plan(), source_name="complete", consumer_name="consumer_a"
    )
    incomplete_plan = replace(
        _all_gather_plan(), source_name="incomplete", consumer_name="consumer_b"
    )
    complete_source = LifetimeBoundBuffer("complete", size=128, uses=[0, 1])
    complete_destination = LifetimeBoundBuffer(
        complete_plan.destination_name, size=128, uses=[1, 2]
    )
    incomplete_source = LifetimeBoundBuffer("incomplete", size=128, uses=[2, 3])
    incomplete_destination = LifetimeBoundBuffer(
        incomplete_plan.destination_name, size=128, uses=[3, 4]
    )
    dependent = LifetimeBoundBuffer(
        "dependent",
        size=128,
        uses=[3, 4],
        in_place_parents=["incomplete"],
    )
    buffers = [
        complete_source,
        complete_destination,
        incomplete_source,
        incomplete_destination,
        dependent,
    ]

    class _PartialLayout:
        spill_reasons = {}

        def plan_layout(self, planned_buffers, log_lx_usage=False):
            del log_lx_usage
            addresses = {
                "complete": 0,
                complete_plan.destination_name: 256,
                "incomplete": 512,
                incomplete_plan.destination_name: None,
                "dependent": 640,
            }
            for buffer in planned_buffers:
                buffer.address = addresses[buffer.name]
            return list(planned_buffers)

    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {
        "complete": complete_plan,
        "incomplete": incomplete_plan,
    }
    allocation = allocator._plan_layout_with_atomic_relayouts(_PartialLayout(), buffers)
    by_name = {buffer.name: buffer for buffer in allocation}

    assert by_name["complete"].address == 0
    assert by_name[complete_plan.destination_name].address == 256
    assert by_name["incomplete"].address is None
    assert by_name[incomplete_plan.destination_name].address is None
    assert by_name["dependent"].address == 640
    assert by_name["dependent"].in_place_parents == []
    assert allocator._lx_relayout_plans_by_source == {"complete": complete_plan}


def test_planned_restickify_source_is_eligible_for_lx_reuse(monkeypatch):
    class _MutationLayout:
        pass

    class _ComputedBuffer:
        def __init__(self, name, layout=None):
            self.name = name
            self.layout = SimpleNamespace() if layout is None else layout

        def get_name(self):
            return self.name

    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": _all_gather_plan()}
    source = _ComputedBuffer("buf_k")
    unrelated = _ComputedBuffer("unrelated")
    mutated_source = _ComputedBuffer("buf_k", _MutationLayout())
    monkeypatch.setattr(allocator_module, "ComputedBuffer", _ComputedBuffer)
    monkeypatch.setattr(
        allocator_module,
        "MutationLayoutSHOULDREMOVE",
        _MutationLayout,
    )
    monkeypatch.setattr(
        allocator_module.config,
        "allow_all_ops_in_lx_planning",
        False,
    )
    monkeypatch.setattr(allocator, "_get_op_name", lambda _: "restickify")

    assert allocator._op_output_good_for_lx_reuse(source)
    assert not allocator._op_output_good_for_lx_reuse(unrelated)
    assert not allocator._op_output_good_for_lx_reuse(mutated_source)


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
        gather_dim=lq,
    )

    result = _materialize_explicit_lx_shuffle(
        source_arg,
        consumer_spec,
        plan,
    )
    assert result is not None
    shuffle_spec, consumer_arg = result
    assert shuffle_spec.gather_dim is None
    assert shuffle_spec.replicas_contiguous
    assert consumer_arg.allocation == {"lx": 0x44000}
    assert shuffle_spec.num_cores_override == 32
    root, allocations = _compile_shuffle(shuffle_spec)

    head_dim = next(
        dim for dim, splits in root["numWkSlicesPerDim_"].items() if splits == 4
    )
    assert [root["coreIdToWkSlice_"][str(core)][head_dim] for core in range(32)] == [
        head for head in range(4) for _ in range(8)
    ]

    input_map = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    output_map = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert len(input_map) == len(output_map) == 32
    assert input_map["0"] != input_map["4"]
    assert output_map["0"] == output_map["4"]

    input_out = allocations[0]["coordinates_"]["coordInfo"]["out"]["folds"]
    output_out = allocations[1]["coordinates_"]["coordInfo"]["out"]["folds"]
    assert (
        input_out["dim_prop_attr"][0]["factor_"],
        input_out["dim_prop_func"][0]["Affine"]["alpha_"],
    ) == (8, 512)
    assert (
        output_out["dim_prop_attr"][0]["factor_"],
        output_out["dim_prop_func"][0]["Affine"]["alpha_"],
    ) == (1, 4096)

    stick_folds = allocations[0]["coordinates_"]["coordInfo"]["in"]["folds"]
    assert (
        stick_folds["dim_prop_attr"][1]["factor_"],
        stick_folds["dim_prop_func"][1]["Affine"]["alpha_"],
    ) == (1, 0)
