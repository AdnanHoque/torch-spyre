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

import pytest
import torch

from torch_spyre._inductor.expert_execution import (
    ExpertGraphPlan,
    ExpertPlanningError,
    ExpertStrategy,
    materialize_expert_execution_graph,
    plan_expert_execution_graph,
    prepare_expert_execution_graph,
)
from torch_spyre._inductor.expert_execution.custom_op import (
    dense_expert_persistent_ffn,
    moe_ffn,
)


def _graph_module(*, experts=2):
    graph = torch.fx.Graph()
    specs = (
        ("x", (64, 64), None),
        ("gate", (experts, 64, 64), (64, experts * 64, 1)),
        ("up", (experts, 64, 64), (64, experts * 64, 1)),
        ("down", (experts, 64, 64), (64, experts * 64, 1)),
        ("routing", (64, experts, 1), (experts * 64, 64, 1)),
    )
    args = []
    for name, shape, stride in specs:
        node = graph.placeholder(name)
        value = torch.empty(shape, device="meta", dtype=torch.float16)
        if stride is not None:
            value = torch.as_strided(value, shape, stride)
        node.meta["val"] = value
        args.append(node)
    result = graph.call_function(moe_ffn._opoverload, args=(*args, 1, "silu"))
    result.meta["val"] = torch.empty((64, 64), device="meta", dtype=torch.float16)
    graph.output(result)
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def _call_target(graph_module):
    return next(
        node.target for node in graph_module.graph.nodes if node.op == "call_function"
    )


def test_planning_is_pure_and_selects_persistent_schedule():
    graph_module = _graph_module()
    before_code = graph_module.code
    before_meta = {node.name: dict(node.meta) for node in graph_module.graph.nodes}

    plan = plan_expert_execution_graph(
        graph_module,
        persistent_available=True,
        available_lx_bytes=1_000_000,
    )

    assert graph_module.code == before_code
    for node in graph_module.graph.nodes:
        assert node.meta.keys() == before_meta[node.name].keys()
        for key, value in node.meta.items():
            assert value is before_meta[node.name][key]
    selected = plan.nodes[0]
    assert selected.strategy is ExpertStrategy.PERSISTENT_DENSE
    assert selected.expert_count == 2
    assert selected.schedule.binding_kind == "sequential_affine"
    assert selected.schedule.weight_layout == "logical_expert_major_k_major_backing"
    assert selected.schedule.routing_layout == "logical_token_major_full_sticks"
    assert selected.schedule.preheader == ("stage_x", "fill_accumulator")
    assert selected.schedule.drain == ("drain_output",)
    assert selected.selection_reason == "persistent envelope satisfied"


def test_conservative_feasibility_selects_ordinary_dense():
    plan = plan_expert_execution_graph(
        _graph_module(), persistent_available=True, available_lx_bytes=1
    )

    assert plan.nodes[0].strategy is ExpertStrategy.ORDINARY_DENSE
    assert plan.nodes[0].schedule is None
    assert "LX capacity" in plan.nodes[0].selection_reason


def test_row_divisibility_is_checked_before_materialization():
    plan = plan_expert_execution_graph(
        _graph_module(),
        persistent_available=True,
        available_lx_bytes=1_000_000,
        row_divisor=128,
    )
    assert plan.nodes[0].strategy is ExpertStrategy.ORDINARY_DENSE
    assert "not divisible" in plan.nodes[0].selection_reason


def test_materialization_rewrites_only_an_isolated_clone():
    source = _graph_module()
    plan = plan_expert_execution_graph(
        source, persistent_available=True, available_lx_bytes=1_000_000
    )

    candidate = materialize_expert_execution_graph(source, plan)

    assert candidate is not source
    assert _call_target(source) == moe_ffn._opoverload
    assert _call_target(candidate) == dense_expert_persistent_ffn._opoverload
    call = next(node for node in candidate.graph.nodes if node.op == "call_function")
    assert call.args[-1] == call.name


def test_contiguous_expert_weights_do_not_select_persistent():
    graph_module = _graph_module()
    for name in ("gate", "up", "down"):
        node = next(node for node in graph_module.graph.nodes if node.name == name)
        node.meta["val"] = torch.empty(
            tuple(node.meta["val"].shape), device="meta", dtype=torch.float16
        )

    plan = plan_expert_execution_graph(
        graph_module,
        persistent_available=True,
        available_lx_bytes=1_000_000,
    )

    assert plan.nodes[0].strategy is ExpertStrategy.ORDINARY_DENSE


def test_contiguous_token_major_routing_does_not_select_persistent():
    graph_module = _graph_module()
    routing = next(node for node in graph_module.graph.nodes if node.name == "routing")
    routing.meta["val"] = torch.empty(
        tuple(routing.meta["val"].shape), device="meta", dtype=torch.float16
    )

    plan = plan_expert_execution_graph(
        graph_module,
        persistent_available=True,
        available_lx_bytes=1_000_000,
    )

    assert plan.nodes[0].strategy is ExpertStrategy.ORDINARY_DENSE
    assert "full-stick backing" in plan.nodes[0].selection_reason


def test_ordinary_dense_materialization_is_a_no_op():
    source = _graph_module()
    plan = plan_expert_execution_graph(
        source, persistent_available=False, available_lx_bytes=1_000_000
    )

    candidate = materialize_expert_execution_graph(source, plan)

    assert candidate is source
    assert _call_target(source) == moe_ffn._opoverload


def test_materialization_rejects_a_changed_source_graph():
    source = _graph_module()
    plan = plan_expert_execution_graph(
        source, persistent_available=True, available_lx_bytes=1_000_000
    )
    call = next(node for node in source.graph.nodes if node.op == "call_function")
    call.args = (*call.args[:-1], "gelu_tanh")
    source.recompile()

    with pytest.raises(ExpertPlanningError, match="changed after expert planning"):
        materialize_expert_execution_graph(source, plan)


def test_failed_materialization_does_not_change_the_source():
    source = _graph_module()
    plan = plan_expert_execution_graph(
        source, persistent_available=True, available_lx_bytes=1_000_000
    )
    invalid_node = dataclasses.replace(plan.nodes[0], schedule=None)
    invalid_plan = ExpertGraphPlan(plan.source_structure, (invalid_node,))

    with pytest.raises(ExpertPlanningError, match="not a valid persistent"):
        materialize_expert_execution_graph(source, invalid_plan)

    assert _call_target(source) == moe_ffn._opoverload


def test_prepare_uses_configured_lx_budget(monkeypatch):
    source = _graph_module(experts=128)
    monkeypatch.setattr(
        "torch_spyre._inductor.config.enable_dense_expert_persistent", True
    )
    monkeypatch.setattr("torch_spyre._inductor.config.sencores", 32)
    monkeypatch.setattr("torch_spyre._inductor.config.lx_planning", True)
    monkeypatch.setattr(
        "torch_spyre._inductor.scratchpad.allocator._lx_planning_size",
        lambda: 1_000_000,
    )

    candidate, plan = prepare_expert_execution_graph(source)

    assert len(plan.nodes) == 1
    assert _call_target(source) == moe_ffn._opoverload
    assert _call_target(candidate) == dense_expert_persistent_ffn._opoverload


def test_default_off_e128_preserves_ordinary_dense(monkeypatch):
    source = _graph_module(experts=128)
    monkeypatch.setattr(
        "torch_spyre._inductor.config.enable_dense_expert_persistent", False
    )
    monkeypatch.setattr("torch_spyre._inductor.config.sencores", 32)

    candidate, plan = prepare_expert_execution_graph(source)

    assert plan.nodes[0].strategy is ExpertStrategy.ORDINARY_DENSE
    assert candidate is source
    assert _call_target(candidate) == moe_ffn._opoverload


def test_enabled_e128_decline_fails_visibly(monkeypatch):
    source = _graph_module(experts=128)
    monkeypatch.setattr(
        "torch_spyre._inductor.config.enable_dense_expert_persistent", True
    )
    monkeypatch.setattr("torch_spyre._inductor.config.sencores", 32)
    monkeypatch.setattr("torch_spyre._inductor.config.lx_planning", True)
    monkeypatch.setattr(
        "torch_spyre._inductor.scratchpad.allocator._lx_planning_size",
        lambda: 1,
    )

    with pytest.raises(
        ExpertPlanningError, match="persistent expert planning declined"
    ):
        prepare_expert_execution_graph(source)


def test_persistent_strategy_is_not_user_enabled():
    from torch_spyre._inductor import config

    assert config.enable_dense_expert_persistent is False
