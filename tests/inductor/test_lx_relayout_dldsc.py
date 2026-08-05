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
from torch._inductor.dependencies import MemoryDep
from torch._inductor.virtualized import V

import torch_spyre._inductor.lx_relayout as lx_relayout_module
import torch_spyre._inductor.scratchpad.allocator as allocator_module
import torch_spyre._inductor.scheduler as scheduler_module
import torch_spyre._inductor.spyre_kernel as spyre_kernel_module

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import compile_op_spec
from torch_spyre._inductor.lx_relayout import (
    LX_RELAYOUT_ATTR,
    LX_RELAYOUT_SHUFFLE_ATTR,
    LXRelayoutPlan,
    _is_equal_footprint_geometry,
    per_core_view_matches_lx_relayout_side,
    rebind_lx_relayout_iteration_geometry,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.spyre_kernel import (
    SpyreKernel,
    TensorAccess,
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


def _dense_plan(
    source_name: str = "buf_mlp", consumer_name: str = "pointwise"
) -> LXRelayoutPlan:
    m, n = Symbol("m"), Symbol("n")
    return LXRelayoutPlan(
        source_name=source_name,
        consumer_name=consumer_name,
        source_core_id_to_device_slice={
            str(core): {"0": core // 8, "1": core % 8} for core in range(32)
        },
        destination_core_id_to_device_slice={
            str(core): {"0": core, "1": 0} for core in range(32)
        },
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 32, "1": 1},
        shuffle_iteration_symbols=(m, n),
        device_dim_to_iteration_symbol={"0": m, "1": n},
        source_lx_address=0x24000,
        destination_lx_address=0x44000,
    )


def _compile_shuffle(shuffle_spec):
    simplify_op_spec(shuffle_spec)
    sdsc, *_ = compile_op_spec(0, shuffle_spec, [])
    root = next(iter(sdsc.values()))
    shuffle_dsc = next(iter(root["dscs_"][0].values()))
    allocations = [
        row for row in shuffle_dsc["scheduleTree_"] if row["nodeType_"] == "allocate"
    ]
    return root, allocations


def test_relayout_emits_owner_maps():
    plan = _dense_plan()
    assert _is_equal_footprint_geometry(
        plan.source_core_id_to_device_slice,
        plan.source_device_dim_splits,
        plan.destination_core_id_to_device_slice,
        plan.destination_device_dim_splits,
    )
    assert not _is_equal_footprint_geometry(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {"0": {"0": 0}, "1": {"0": 0}},
        {"0": 1},
    )

    m, n = plan.shuffle_iteration_symbols
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 200, 64],
        device_coordinates=[m, floor(n / 64), Mod(n, 64)],
        allocation={"lx": plan.source_lx_address},
        name=plan.source_name,
    )
    destination_arg = replace(
        source_arg,
        is_input=False,
        name=plan.destination_name,
        allocation={"lx": plan.destination_lx_address},
        allocation_core_id_to_device_slice=(plan.destination_core_id_to_device_slice),
        allocation_device_dim_splits=dict(plan.destination_device_dim_splits),
    )
    source_arg = replace(
        source_arg,
        allocation_core_id_to_device_slice=plan.source_core_id_to_device_slice,
        allocation_device_dim_splits=dict(plan.source_device_dim_splits),
    )
    shuffle_spec = OpSpec(
        op="shuffle",
        is_reduction=False,
        iteration_space={m: (Integer(512), 32), n: (Integer(12800), 1)},
        args=[source_arg, destination_arg],
        op_info={"lx_relayout_consumer_is_matmul": False},
    )

    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32
    assert destination_arg.allocation == {"lx": plan.destination_lx_address}
    producer_map = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    consumer_map = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert (producer_map["0"], producer_map["31"]) == (
        {"mb": 0, "out": 0},
        {"mb": 3, "out": 7},
    )
    assert (consumer_map["0"], consumer_map["31"]) == (
        {"mb": 0, "out": 0},
        {"mb": 31, "out": 0},
    )


def test_relayout_owner_maps_follow_aligned_device_dimensions():
    """Split/synthetic dimensions cannot leave allocation maps on old axes."""

    m, n = Symbol("m"), Symbol("n")
    plan = LXRelayoutPlan(
        source_name="buf_split",
        consumer_name="pointwise",
        source_core_id_to_device_slice={str(core): {"1": core} for core in range(8)},
        destination_core_id_to_device_slice={
            str(core): {"1": 7 - core} for core in range(8)
        },
        source_device_dim_splits={"1": 8},
        destination_device_dim_splits={"1": 8},
        shuffle_iteration_symbols=(n, m),
        device_dim_to_iteration_symbol={"1": n},
        source_lx_address=0x24000,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[32, 8, 64],
        device_coordinates=[Mod(n, 32), floor(n / 32), Mod(m, 64)],
        allocation={"lx": plan.source_lx_address},
        name=plan.source_name,
    )
    destination_arg = replace(
        source_arg,
        is_input=False,
        name=plan.destination_name,
        allocation={"lx": plan.destination_lx_address},
        allocation_core_id_to_device_slice=(plan.destination_core_id_to_device_slice),
        allocation_device_dim_splits=dict(plan.destination_device_dim_splits),
    )
    source_arg = replace(
        source_arg,
        allocation_core_id_to_device_slice=plan.source_core_id_to_device_slice,
        allocation_device_dim_splits=dict(plan.source_device_dim_splits),
    )
    shuffle_spec = OpSpec(
        op="shuffle",
        is_reduction=False,
        iteration_space={n: (Integer(256), 8), m: (Integer(64), 1)},
        args=[source_arg, destination_arg],
        op_info={"lx_relayout_consumer_is_matmul": False},
    )

    root, allocations = _compile_shuffle(shuffle_spec)

    assert list(shuffle_spec.iteration_space) == [n, Symbol("z0"), m]
    assert shuffle_spec.args[0].allocation_device_dim_splits == {"2": 8}
    assert shuffle_spec.args[1].allocation_device_dim_splits == {"2": 8}
    assert root["numCoresUsed_"] == 8
    producer_map = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    consumer_map = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert (producer_map["0"], producer_map["7"]) == (
        {"mb": 0, "x": 0, "out": 0},
        {"mb": 0, "x": 7, "out": 0},
    )
    assert (consumer_map["0"], consumer_map["7"]) == (
        {"mb": 0, "x": 7, "out": 0},
        {"mb": 0, "x": 0, "out": 0},
    )


def test_planner_requires_materializable_geometry_for_every_read(monkeypatch):
    class _Pointwise:
        pass

    class _ComputedBuffer:
        def __init__(self, name):
            self.name = name
            self.data = _Pointwise()
            self.layout = SimpleNamespace(device_layout=object())

        def get_name(self):
            return self.name

    x, y, core_id = Symbol("x"), Symbol("y"), Symbol("core_id")
    producer = _ComputedBuffer("shared")
    consumer_a = _ComputedBuffer("consumer_a")
    consumer_b = _ComputedBuffer("consumer_b")
    write = MemoryDep("shared", 64 * x + y, (x, y), (2, 64))
    read_a = MemoryDep("shared", 64 * x + y, (x, y), (2, 64))
    read_b = MemoryDep("shared", 64 * x + y, (x, y), (2, 64))
    read_writes = {
        "shared": SimpleNamespace(reads=[], writes=[write]),
        "consumer_a": SimpleNamespace(reads=[read_a], writes=[]),
        "consumer_b": SimpleNamespace(reads=[read_b], writes=[]),
    }
    producer_view = PerCoreView(((1, 2),), ((1, core_id),))
    consumer_view = PerCoreView(((1, 2),), ((1, 1 - core_id),))

    monkeypatch.setattr(lx_relayout_module.config, "lx_planner_relayout", True)
    monkeypatch.setattr(lx_relayout_module, "ComputedBuffer", _ComputedBuffer)
    monkeypatch.setattr(lx_relayout_module, "Pointwise", _Pointwise)
    monkeypatch.setattr(
        lx_relayout_module, "op_read_writes", lambda op: read_writes[op.get_name()]
    )
    monkeypatch.setattr(lx_relayout_module, "_is_matmul_op", lambda _: False)
    monkeypatch.setattr(lx_relayout_module, "_op_num_cores", lambda _: 2)
    monkeypatch.setattr(
        lx_relayout_module,
        "_per_core_view_on_buf",
        lambda op, *_: (
            producer_view if op is producer else consumer_view,
            False,
            True,
        ),
    )
    monkeypatch.setattr(
        lx_relayout_module,
        "iteration_space_from_op",
        lambda _: {x: Integer(2), y: Integer(64)},
    )
    valid_coordinates = [x, Mod(y, 64)]
    monkeypatch.setattr(
        lx_relayout_module, "try_device_coordinates", lambda *_: valid_coordinates
    )

    graph = SimpleNamespace(operations=[producer, consumer_a, consumer_b])
    assert {
        plan.edge_key for plan in lx_relayout_module.collect_lx_relayout_plans(graph)
    } == {
        ("shared", "consumer_a"),
        ("shared", "consumer_b"),
    }

    indirect_read = MemoryDep("indices", Symbol("indirect0"), (x, y), (2, 64))
    read_writes["consumer_a"].reads.append(indirect_read)
    assert lx_relayout_module.collect_lx_relayout_plans(graph) == []
    read_writes["consumer_a"].reads.remove(indirect_read)

    monkeypatch.setattr(
        lx_relayout_module,
        "try_device_coordinates",
        lambda _layout, dep, _sizes: None if dep is read_b else valid_coordinates,
    )
    assert lx_relayout_module.collect_lx_relayout_plans(graph) == []

    monkeypatch.setattr(
        lx_relayout_module,
        "try_device_coordinates",
        lambda _layout, dep, _sizes: (
            [Mod(y, 64)] if dep is read_b else valid_coordinates
        ),
    )
    assert lx_relayout_module.collect_lx_relayout_plans(graph) == []

    direct_read_a = MemoryDep("shared", 64 * x - y, (x, y), (2, 64))
    read_writes["consumer_a"].reads.append(direct_read_a)
    monkeypatch.setattr(
        lx_relayout_module,
        "_per_core_view_on_buf",
        lambda op, dep, *_: (
            producer_view if op is producer or dep is direct_read_a else consumer_view,
            False,
            True,
        ),
    )
    monkeypatch.setattr(
        lx_relayout_module, "try_device_coordinates", lambda *_: valid_coordinates
    )
    assert lx_relayout_module.collect_lx_relayout_plans(graph) == []


def test_data_op_lowers_real_relayout_node(monkeypatch):
    class _FixedLayout:
        allocation = {"lx": 0x1000}
        device_layout = SimpleNamespace(device_size=(1,))
        size = (1,)

    layout = _FixedLayout()
    source = TensorAccess("source", Integer(0), layout)
    input_arg = TensorArg(
        True,
        -1,
        DataFormats.SEN169_FP16,
        [1],
        [Integer(0)],
        {"lx": 0x1000},
        name="source",
    )
    output_arg = replace(
        input_arg,
        is_input=False,
        name="output",
        allocation={"lx": 0x2000},
    )
    plan = replace(
        _dense_plan("source", "consumer"),
        source_lx_address=0x1000,
        destination_lx_address=0x2000,
        materialized_destination_name="output",
    )
    shuffle_node = SimpleNamespace(get_name=lambda: "output")
    setattr(shuffle_node, LX_RELAYOUT_SHUFFLE_ATTR, plan)

    monkeypatch.setattr(spyre_kernel_module, "FixedTiledLayout", _FixedLayout)
    kernel = object.__new__(SpyreKernel)
    kernel.args = SimpleNamespace(output=lambda _: None)
    kernel.op_specs = []
    kernel.indirect_vars = {}
    kernel.current_node = SimpleNamespace(
        get_nodes=lambda: [SimpleNamespace(node=shuffle_node)]
    )
    kernel.create_tensor_arg = lambda is_input, *_args, **_kwargs: (
        input_arg if is_input else output_arg
    )
    kernel.create_op_spec = lambda op, _is_reduction, args, op_info, *_args: (
        SimpleNamespace(op=op, args=args, op_info=op_info)
    )

    graph = SimpleNamespace(
        scheduler=SimpleNamespace(mutation_real_name={}),
        removed_buffers=set(),
        get_buffer=lambda _: SimpleNamespace(get_layout=lambda: layout),
        sizevars=SimpleNamespace(precomputed_replacements={}),
    )
    with V.set_graph_handler(graph):
        SpyreKernel.store(kernel, "output", Integer(0), source)

    assert len(kernel.op_specs) == 1
    shuffle_spec = kernel.op_specs[0]
    assert shuffle_spec.op == "shuffle"
    assert shuffle_spec.args[0].allocation_core_id_to_device_slice == (
        plan.source_core_id_to_device_slice
    )
    assert shuffle_spec.args[1].allocation_core_id_to_device_slice == (
        plan.destination_core_id_to_device_slice
    )


def test_scheduler_checks_owner_map_and_rebinds_final_symbols(monkeypatch):
    plan = _dense_plan()
    m, n = plan.shuffle_iteration_symbols
    core_id = Symbol("core_id")
    op = SimpleNamespace(op_it_space_splits=({m: 4, n: 8}, {}))
    producer_view = PerCoreView(
        ((0, 4), (1, 8)),
        ((0, floor(core_id / 8)), (1, Mod(core_id, 8))),
    )
    assert per_core_view_matches_lx_relayout_side(
        op, producer_view, plan, destination=False
    )
    assert not per_core_view_matches_lx_relayout_side(
        op, producer_view, plan, destination=True
    )

    final_m, final_n = Symbol("c0"), Symbol("c1")
    monkeypatch.setattr(
        lx_relayout_module,
        "try_device_coordinates",
        lambda *_: [final_m, floor(final_n / 64), Mod(final_n, 64)],
    )
    rebound = rebind_lx_relayout_iteration_geometry(
        plan, object(), object(), (final_m, final_n)
    )
    assert rebound is not None
    assert rebound.shuffle_iteration_symbols == (final_m, final_n)
    assert rebound.device_dim_to_iteration_symbol == {"0": final_m, "1": final_n}


def test_scheduler_demotes_source_and_drops_stale_plan(monkeypatch):
    class _Dep:
        def __init__(self, name):
            self.name = name

    class _Op:
        def __init__(self, name, allocation=None):
            self.name = name
            self.layout = SimpleNamespace(allocation=allocation or {})

        def get_name(self):
            return self.name

    class _SchedulerNode:
        def __init__(self, op, reads=(), writes=()):
            self.node = op
            self.read_writes = SimpleNamespace(reads=set(reads), writes=set(writes))

        def get_nodes(self):
            return [self]

        def get_name(self):
            return self.node.get_name()

    plan = replace(_dense_plan(), materialized_destination_name="relayout_destination")
    producer = _Op(plan.source_name, {"lx": plan.source_lx_address})
    shuffle = _Op(
        plan.materialized_destination_name, {"lx": plan.destination_lx_address}
    )
    consumer = _Op(plan.consumer_name)
    setattr(shuffle, LX_RELAYOUT_SHUFFLE_ATTR, plan)
    source_dep = _Dep(plan.source_name)
    destination_dep = _Dep(plan.materialized_destination_name)
    nodes = [
        _SchedulerNode(producer, writes=(source_dep,)),
        _SchedulerNode(shuffle, reads=(source_dep,), writes=(destination_dep,)),
        _SchedulerNode(consumer, reads=(destination_dep,)),
    ]

    monkeypatch.setattr(scheduler_module, "SchedulerNode", _SchedulerNode)
    monkeypatch.setattr(scheduler_module, "MemoryDep", _Dep)
    monkeypatch.setattr(
        scheduler_module,
        "V",
        SimpleNamespace(
            graph=SimpleNamespace(
                try_get_buffer=lambda name: {
                    plan.source_name: producer,
                    plan.materialized_destination_name: shuffle,
                }.get(name)
            )
        ),
    )
    monkeypatch.setattr(scheduler_module._spyre_config, "lx_planning", True)
    monkeypatch.setattr(
        scheduler_module,
        "per_core_view_scheduled",
        lambda *_: (PerCoreView((), ()), False, True),
    )
    monkeypatch.setattr(
        scheduler_module,
        "per_core_view_matches_lx_relayout_side",
        lambda *_args, **_kwargs: False,
    )

    scheduler_module.demote_incoherent_lx_buffers(nodes)
    assert "lx" not in producer.layout.allocation
    assert "lx" not in shuffle.layout.allocation
    assert not hasattr(shuffle, LX_RELAYOUT_SHUFFLE_ATTR)


def test_atomic_allocation_records_complete_pair_or_falls_back():
    graph = _DummyGraph("producer", "consumer")
    graph.name_to_buffer.pop("consumer")
    plan = _dense_plan("buf_k", "consumer")
    source = LifetimeBoundBuffer("buf_k", size=128 * 1024, uses=[0, 1])
    allocator = ScratchpadAllocator(GreedyLayoutSolver(256 * 1024))
    allocator._lx_relayout_plans_by_edge = {plan.edge_key: plan}
    buffers = [source]
    allocator._append_lx_relayout_destinations(graph, buffers)

    destination = next(
        buffer for buffer in buffers if buffer.name == plan.destination_name
    )
    layout_planner = allocator._layout_planner_for_buffers(buffers)
    assert isinstance(layout_planner, FirstFitLayoutSolver)
    allocation = allocator._plan_layout_with_atomic_relayouts(layout_planner, buffers)
    by_name = {buffer.name: buffer for buffer in allocation}
    assert by_name["buf_k"].address is not None
    assert destination.address is not None
    assert by_name["buf_k"].address != destination.address
    allocator._record_successful_lx_relayouts(graph, allocation)
    recorded = getattr(graph.operations[1], LX_RELAYOUT_ATTR)["buf_k"]
    assert (recorded.source_lx_address, recorded.destination_lx_address) == (
        by_name["buf_k"].address,
        destination.address,
    )

    for buffer in buffers:
        buffer.address = None
    fallback = ScratchpadAllocator(GreedyLayoutSolver(128 * 1024))
    fallback._lx_relayout_plans_by_edge = {plan.edge_key: plan}
    allocation = fallback._plan_layout_with_atomic_relayouts(
        FirstFitLayoutSolver(128 * 1024), buffers
    )
    assert all(buffer.address is None for buffer in allocation)
    fallback._record_successful_lx_relayouts(graph, allocation)
    assert not hasattr(graph.operations[1], LX_RELAYOUT_ATTR)


def test_relayout_lifetimes_cover_inputs_and_multiple_consumers(monkeypatch):
    graph = _DummyGraph(
        "producer_input", "buf_k", "consumer_a", "independent", "consumer_b"
    )
    plan_a = _dense_plan("buf_k", "consumer_a")
    plan_b = _dense_plan("buf_k", "consumer_b")
    producer_input = LifetimeBoundBuffer("producer_input", size=128, uses=[0, 1])
    source = LifetimeBoundBuffer("buf_k", size=128, uses=[1, 2, 4])
    input_child = LifetimeBoundBuffer(
        "input_child", size=128, uses=[1, 2], in_place_parents=["producer_input"]
    )
    source_child = LifetimeBoundBuffer(
        "source_child", size=128, uses=[2, 3], in_place_parents=["buf_k"]
    )
    buffers = [producer_input, source, input_child, source_child]
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_edge = {
        plan_a.edge_key: plan_a,
        plan_b.edge_key: plan_b,
    }
    monkeypatch.setattr(
        allocator_module,
        "op_read_writes",
        lambda _: SimpleNamespace(reads=[SimpleNamespace(name="producer_input")]),
    )

    allocator._append_lx_relayout_destinations(graph, buffers)
    by_name = {buffer.name: buffer for buffer in buffers}
    assert source.uses == [1, 2, 5]
    assert by_name[plan_a.destination_name].uses == [2, 3]
    assert by_name[plan_b.destination_name].uses == [5, 6]
    assert producer_input.uses == [0, 1, 2, 5]
    assert input_child.in_place_parents == []
    assert source_child.in_place_parents == []


def test_partial_allocation_fallback_is_atomic_per_source():
    complete = _dense_plan("complete", "consumer_a")
    incomplete_a = _dense_plan("incomplete", "consumer_b")
    incomplete_b = _dense_plan("incomplete", "consumer_c")
    buffers = [
        LifetimeBoundBuffer("complete", size=128, uses=[0, 1]),
        LifetimeBoundBuffer(complete.destination_name, size=128, uses=[1, 2]),
        LifetimeBoundBuffer("incomplete", size=128, uses=[2, 3, 5]),
        LifetimeBoundBuffer(incomplete_a.destination_name, size=128, uses=[3, 4]),
        LifetimeBoundBuffer(incomplete_b.destination_name, size=128, uses=[5, 6]),
        LifetimeBoundBuffer(
            "dependent", size=128, uses=[5, 6], in_place_parents=["incomplete"]
        ),
    ]
    addresses = {
        "complete": 0,
        complete.destination_name: 256,
        "incomplete": 512,
        incomplete_a.destination_name: 768,
        incomplete_b.destination_name: None,
        "dependent": 896,
    }

    class _PartialLayout:
        spill_reasons = {}

        def plan_layout(self, planned_buffers, log_lx_usage=False):
            del log_lx_usage
            for buffer in planned_buffers:
                buffer.address = addresses[buffer.name]
            return list(planned_buffers)

    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_edge = {
        plan.edge_key: plan for plan in (complete, incomplete_a, incomplete_b)
    }
    allocation = allocator._plan_layout_with_atomic_relayouts(_PartialLayout(), buffers)
    by_name = {buffer.name: buffer for buffer in allocation}

    assert (
        by_name["complete"].address,
        by_name[complete.destination_name].address,
    ) == (
        0,
        256,
    )
    assert all(
        by_name[name].address is None
        for name in (
            "incomplete",
            incomplete_a.destination_name,
            incomplete_b.destination_name,
        )
    )
    assert by_name["dependent"].in_place_parents == []
    assert allocator._lx_relayout_plans_by_edge == {complete.edge_key: complete}


def test_planned_restickify_source_is_eligible_for_lx_reuse(monkeypatch):
    class _MutationLayout:
        pass

    class _FixedTiledLayout:
        pass

    class _ComputedBuffer:
        def __init__(self, name, layout=None):
            self.name = name
            self.layout = _FixedTiledLayout() if layout is None else layout

        def get_name(self):
            return self.name

    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    plan = _dense_plan("buf_k", "consumer")
    allocator._lx_relayout_plans_by_edge = {plan.edge_key: plan}
    monkeypatch.setattr(allocator_module, "ComputedBuffer", _ComputedBuffer)
    monkeypatch.setattr(allocator_module, "MutationLayoutSHOULDREMOVE", _MutationLayout)
    monkeypatch.setattr(allocator_module, "FixedTiledLayout", _FixedTiledLayout)
    monkeypatch.setattr(allocator_module.config, "allow_all_ops_in_lx_planning", False)
    monkeypatch.setattr(allocator, "_get_op_name", lambda _: "restickify")

    assert allocator._op_output_good_for_lx_reuse(_ComputedBuffer("buf_k"))
    assert not allocator._op_output_good_for_lx_reuse(_ComputedBuffer("unrelated"))
    assert not allocator._op_output_good_for_lx_reuse(
        _ComputedBuffer("buf_k", _MutationLayout())
    )
