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
import re
from types import SimpleNamespace

import pytest
import torch
from sympy import Integer, Mod, Symbol, floor
from torch._inductor.utils import run_and_get_code

import torch_spyre._inductor.scheduler as scheduler_module
import torch_spyre._inductor.wsr.propagate_named_dims as named_dims

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config, spyre_hint
from torch_spyre._inductor.codegen.superdsc import compile_op_spec, parse_op_spec
from torch_spyre._inductor.constants import IDENTITY_OP
from torch_spyre._inductor.lx_relayout import LXRelayoutPlan
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import PerCoreView
from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator
from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.spyre_kernel import (
    _work_division_from_view,
    simplify_op_spec,
)


_CORE_ID = Symbol("core_id")
_SOURCE_VIEW = PerCoreView(
    ((0, 4), (1, 8)),
    ((0, floor(_CORE_ID / 8)), (1, Mod(_CORE_ID, 8))),
)
_DESTINATION_VIEW = PerCoreView(((0, 32),), ((0, _CORE_ID),))


def _dense_plan(source="buf_mlp", consumer="pointwise") -> LXRelayoutPlan:
    return LXRelayoutPlan(
        source,
        consumer,
        _SOURCE_VIEW,
        _DESTINATION_VIEW,
        num_cores=32,
        source_lx_address=0x24000,
        destination_lx_address=0x44000,
    )


class _Dep:
    def __init__(self, name):
        self.name = name


class _Node:
    def __init__(self, name, reads=(), writes=(), allocation=None, lx_view=None):
        self.name = name
        self.node = SimpleNamespace(
            layout=SimpleNamespace(allocation=allocation or {}, lx_view=lx_view)
        )
        self.read_writes = SimpleNamespace(reads=set(reads), writes=set(writes))

    def get_nodes(self):
        return [self]

    def get_name(self):
        return self.name


def _compile_spec(op_spec, *, normalize=True):
    if normalize:
        simplify_op_spec(op_spec)
    sdsc, *_ = compile_op_spec(0, op_spec, [])
    root = next(iter(sdsc.values()))
    dsc = next(iter(root["dscs_"][0].values()))
    allocations = [
        row for row in dsc["scheduleTree_"] if row["nodeType_"] == "allocate"
    ]
    return root, allocations


def _copy_spec(plan, source_arg, iteration_space):
    source_work_division = _work_division_from_view(
        plan.source_view, source_arg.device_coordinates, iteration_space
    )
    destination_work_division = _work_division_from_view(
        plan.destination_view, source_arg.device_coordinates, iteration_space
    )
    return OpSpec(
        op=IDENTITY_OP,
        is_reduction=False,
        iteration_space=iteration_space,
        args=[
            replace(
                source_arg,
                work_division=source_work_division,
            ),
            replace(
                source_arg,
                is_input=False,
                name=plan.destination_name,
                allocation={"lx": plan.destination_lx_address},
                work_division=destination_work_division,
                lx_consumer_is_matmul=plan.consumer_is_matmul,
            ),
        ],
        op_info={},
    )


def test_relayout_owner_maps_follow_aligned_device_dimensions():
    """Split/synthetic dimensions cannot leave allocation maps on old axes."""

    m, n = Symbol("m"), Symbol("n")
    core_id = Symbol("core_id")
    plan = LXRelayoutPlan(
        source_name="buf_split",
        consumer_name="pointwise",
        source_view=PerCoreView(
            ((1, 8),),
            ((1, core_id),),
        ),
        destination_view=PerCoreView(
            ((1, 8),),
            ((1, 7 - core_id),),
        ),
        num_cores=8,
        consumer_is_matmul=True,
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
    copy_spec = _copy_spec(
        plan,
        source_arg,
        {n: (Integer(256), 8), m: (Integer(64), 1)},
    )

    root, allocations = _compile_spec(copy_spec)

    assert set(root["dscs_"][0]) == {"shuffle"}
    assert list(copy_spec.iteration_space) == [n, Symbol("z0"), m]
    assert copy_spec.args[0].work_division.work_slices == {Symbol("z0"): 8}
    assert copy_spec.args[1].work_division.work_slices == {Symbol("z0"): 8}
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

    ordinary_spec = OpSpec(
        op=IDENTITY_OP,
        is_reduction=False,
        iteration_space=dict(copy_spec.iteration_space),
        args=[
            replace(copy_spec.args[0], work_division=None),
            replace(copy_spec.args[1], work_division=None),
        ],
        op_info={},
    )
    ordinary_root, ordinary_allocations = _compile_spec(ordinary_spec, normalize=False)
    ordinary_sdsc, _ = parse_op_spec(ordinary_spec)
    assert all(arg.work_division is not None for arg in ordinary_sdsc.args)
    assert set(ordinary_root["dscs_"][0]) == {IDENTITY_OP}
    assert all(
        allocation["coordinates_"]["coreIdToWkSlice_"] == {}
        for allocation in ordinary_allocations
    )


@pytest.mark.parametrize("mismatched_destination", [False, True])
def test_scheduler_demotes_mismatched_relayout_side(
    monkeypatch, mismatched_destination
):
    plan = _dense_plan()
    destination_name = "relayout_destination"
    source_dep = _Dep(plan.source_name)
    destination_dep = _Dep(destination_name)
    producer = _Node(
        plan.source_name,
        writes=(source_dep,),
        allocation={"lx": plan.source_lx_address},
        lx_view=plan.source_view,
    )
    copy = _Node(
        destination_name,
        reads=(source_dep,),
        writes=(destination_dep,),
        allocation={"lx": plan.destination_lx_address},
        lx_view=plan.destination_view,
    )
    nodes = [producer, copy, _Node(plan.consumer_name, reads=(destination_dep,))]

    monkeypatch.setattr(scheduler_module, "SchedulerNode", _Node)
    monkeypatch.setattr(scheduler_module, "MemoryDep", _Dep)
    graph = SimpleNamespace(
        try_get_buffer=lambda name: {
            plan.source_name: producer.node,
            destination_name: copy.node,
        }.get(name),
        _spyre_lx_relayout_copies={
            plan.edge_key: (destination_name, plan),
        },
    )
    monkeypatch.setattr(scheduler_module, "V", SimpleNamespace(graph=graph))
    monkeypatch.setattr(scheduler_module._spyre_config, "lx_planning", True)

    def scheduled_view(node, _dep, name):
        mismatch = (mismatched_destination and node is nodes[-1]) or (
            not mismatched_destination and node is nodes[0]
        )
        expected = (
            plan.source_view if name == plan.source_name else plan.destination_view
        )
        return (PerCoreView((), ()) if mismatch else expected, False, True)

    monkeypatch.setattr(scheduler_module, "per_core_view_scheduled", scheduled_view)

    scheduler_module.demote_incoherent_lx_buffers(nodes)
    for node in (producer, copy):
        assert "lx" not in node.node.layout.allocation
        assert node.node.layout.lx_view is None
    assert graph._spyre_lx_relayout_copies == {}


def test_scheduler_does_not_treat_an_unregistered_unary_as_a_relayout(monkeypatch):
    source_dep = _Dep("ordinary_source")
    destination_dep = _Dep("ordinary_unary")
    producer = _Node(
        "ordinary_source",
        writes=(source_dep,),
        allocation={"lx": 0},
        lx_view=_SOURCE_VIEW,
    )
    unary = _Node(
        "ordinary_unary",
        reads=(source_dep,),
        writes=(destination_dep,),
        allocation={"lx": 256},
        lx_view=_DESTINATION_VIEW,
    )
    consumer = _Node("ordinary_consumer", reads=(destination_dep,))
    nodes = [producer, unary, consumer]
    buffers = {
        producer.name: producer.node,
        unary.name: unary.node,
    }
    graph = SimpleNamespace(
        try_get_buffer=buffers.get,
        _spyre_lx_relayout_copies={},
    )

    monkeypatch.setattr(scheduler_module, "SchedulerNode", _Node)
    monkeypatch.setattr(scheduler_module, "MemoryDep", _Dep)
    monkeypatch.setattr(scheduler_module, "V", SimpleNamespace(graph=graph))
    monkeypatch.setattr(scheduler_module._spyre_config, "lx_planning", True)

    def scheduled_view(node, _dep, name):
        if node is unary and name == producer.name:
            return _DESTINATION_VIEW, False, True
        return buffers[name].layout.lx_view, False, True

    monkeypatch.setattr(scheduler_module, "per_core_view_scheduled", scheduled_view)

    scheduler_module.demote_incoherent_lx_buffers(nodes)
    assert "lx" not in producer.node.layout.allocation
    assert "lx" in unary.node.layout.allocation


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

    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_edge = {
        plan.edge_key: plan for plan in (incomplete_a, incomplete_b)
    }
    graph_buffers = {
        "incomplete": SimpleNamespace(
            layout=SimpleNamespace(
                allocation={"lx": 512},
                lx_view=_SOURCE_VIEW,
                lx_consumer_is_matmul=False,
            )
        ),
        **{
            plan.destination_name: SimpleNamespace(
                layout=SimpleNamespace(
                    allocation={"lx": 768},
                    lx_view=_DESTINATION_VIEW,
                    lx_consumer_is_matmul=False,
                )
            )
            for plan in (incomplete_a, incomplete_b)
        },
    }
    graph = SimpleNamespace(
        try_get_buffer=graph_buffers.get,
        _spyre_lx_relayout_copies={
            plan.edge_key: (plan.destination_name, plan)
            for plan in (incomplete_a, incomplete_b)
        },
    )
    fallback = allocator._plan_layout_with_atomic_relayouts(
        _PartialLayout(),
        [
            by_name[name]
            for name in (
                "incomplete",
                *[p.destination_name for p in (incomplete_a, incomplete_b)],
            )
        ],
        graph=graph,
        stock_buffers=lambda: [
            LifetimeBoundBuffer("incomplete", size=128, uses=[2, 3, 5])
        ],
    )
    assert [(buffer.name, buffer.address) for buffer in fallback] == [("incomplete", 0)]
    assert graph._spyre_lx_relayout_copies == {}
    assert all(
        "lx" not in buffer.layout.allocation for buffer in graph_buffers.values()
    )
    assert all(buffer.layout.lx_view is None for buffer in graph_buffers.values())


def test_multiple_consumer_views_get_private_numeric_copies():
    torch.manual_seed(0)
    x = torch.randn(128, 256, dtype=torch.float16)

    named_dims.declare_tensor_dim("M", 128)
    named_dims.declare_tensor_dim("N", 256)

    def fn(x):
        with spyre_hint(work_div={"M": 4, "N": 2}):
            hidden = torch.neg(x)
        with spyre_hint(work_div={"M": 2, "N": 4}):
            consumer_a = torch.relu(hidden)
        with spyre_hint(work_div={"M": 8, "N": 1}):
            consumer_b = torch.abs(hidden)
        return consumer_a, consumer_b

    device_x = named_dims.name_tensor_dims(x.to("spyre"), ["M", "N"])
    with config.patch(
        {
            "sencores": 8,
            "lx_planning": True,
            "allow_all_ops_in_lx_planning": True,
        }
    ):
        actual, source_codes = run_and_get_code(
            torch.compile(
                fn,
                dynamic=False,
                options={"epilogue_fusion": False},
            ),
            device_x,
        )

    for result, expected in zip(actual, fn(x)):
        torch.testing.assert_close(result.cpu(), expected, atol=0.1, rtol=0.1)
    generated = "\n".join(source_codes)
    assert generated.count("op='identity'") == 2
    assert generated.count("work_division=TensorWorkDivision") == 4
    owner_maps = re.findall(
        r"TensorWorkDivision\(work_slices=\{([^}]*)\}, "
        r"core_id_to_work_slice=\{([^}]*)\}\)",
        generated,
    )
    assert len(owner_maps) == 4
    assert owner_maps[0] == owner_maps[2]
    assert {owner_maps[1], owner_maps[3]} == {
        (
            "sympify('c1'): 4, sympify('c0'): 2",
            "sympify('c1'): sympify('Mod(floor(Mod(floor(core_id/2), 4)), 4)'), "
            "sympify('c0'): sympify('Mod(floor(Mod(core_id, 2)), 2)')",
        ),
        (
            "sympify('c0'): 8",
            "sympify('c0'): sympify('Mod(floor(Mod(core_id, 8)), 8)')",
        ),
    }
    assert "op='shuffle'" not in generated
