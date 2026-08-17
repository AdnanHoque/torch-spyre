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

"""Pure selection and transactional FX materialization for expert execution."""

from __future__ import annotations

import copy
import dataclasses
from enum import StrEnum
from typing import Any

import torch

from .custom_op import dense_expert_persistent_ffn, moe_ffn


class ExpertStrategy(StrEnum):
    """Compiler strategies currently available for the semantic MoE operation."""

    PERSISTENT_DENSE = "persistent_dense"
    ORDINARY_DENSE = "ordinary_dense"


@dataclasses.dataclass(frozen=True)
class PersistentExpertSchedule:
    """The schedule contract materialized by the persistent strategy."""

    preheader: tuple[str, ...] = ("stage_x", "fill_accumulator")
    loop_body: tuple[str, ...] = (
        "gate",
        "up",
        "activation",
        "gated_product",
        "down",
        "route_weight",
        "accumulator_combine",
    )
    drain: tuple[str, ...] = ("drain_output",)
    streamed_operands: tuple[str, ...] = (
        "gate_weight",
        "up_weight",
        "down_weight",
        "routing_weight",
    )
    binding_kind: str = "sequential_affine"
    weight_layout: str = "logical_expert_major_k_major_backing"
    routing_layout: str = "logical_token_major"


@dataclasses.dataclass(frozen=True)
class PlannedExpertNode:
    """The selected strategy for one semantic MoE FX node."""

    node_name: str
    strategy: ExpertStrategy
    expert_count: int
    minimum_lx_bytes: int
    schedule: PersistentExpertSchedule | None


@dataclasses.dataclass(frozen=True)
class ExpertGraphPlan:
    """Immutable result of planning every semantic MoE node in an FX graph."""

    source_structure: tuple[tuple[Any, ...], ...]
    nodes: tuple[PlannedExpertNode, ...]


class ExpertPlanningError(RuntimeError):
    """The selected strategy could not be materialized without mutation."""


def graph_structure(graph: torch.fx.Graph) -> tuple[tuple[Any, ...], ...]:
    """Return a stable, metadata-independent graph representation."""

    def encode(value):
        if isinstance(value, torch.fx.Node):
            return ("node", value.name)
        if isinstance(value, tuple):
            return tuple(encode(item) for item in value)
        if isinstance(value, list):
            return ("list", *(encode(item) for item in value))
        if isinstance(value, dict):
            return tuple((key, encode(value[key])) for key in sorted(value))
        return value

    return tuple(
        (node.name, node.op, str(node.target), encode(node.args), encode(node.kwargs))
        for node in graph.nodes
    )


def _tensor_argument(node: torch.fx.Node, index: int, name: str) -> torch.fx.Node:
    value = node.args[index]
    if not isinstance(value, torch.fx.Node):
        raise ValueError(f"{name} must be a tensor FX node")
    return value


def _tensor_metadata(node: torch.fx.Node, name: str):
    """Return the tensor metadata produced by Dynamo or post-grad FX."""

    value = node.meta.get("val")
    if value is None:
        value = node.meta.get("example_value")
    if value is None:
        raise ValueError(f"{name} has no tensor metadata")
    return value


def _shape(node: torch.fx.Node, name: str) -> tuple[int, ...]:
    shape = getattr(_tensor_metadata(node, name), "shape", None)
    if shape is None:
        raise ValueError(f"{name} has no tensor shape metadata")
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} requires static dimensions") from exc


def _element_size(node: torch.fx.Node, name: str) -> int:
    value = _tensor_metadata(node, name)
    if not hasattr(value, "element_size"):
        raise ValueError(f"{name} has no tensor dtype metadata")
    return int(value.element_size())


def _stride(node: torch.fx.Node, name: str) -> tuple[int, ...]:
    value = _tensor_metadata(node, name)
    if not hasattr(value, "stride"):
        raise ValueError(f"{name} has no tensor stride metadata")
    try:
        return tuple(int(dim) for dim in value.stride())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} requires static strides") from exc


def _has_persistent_weight_layout(
    node: torch.fx.Node,
    name: str,
    *,
    experts: int,
    reduction: int,
    output: int,
) -> bool:
    """Check the logical ``[E,K,N]`` view over packed ``[K,E,N]`` storage."""

    return _shape(node, name) == (experts, reduction, output) and _stride(
        node, name
    ) == (output, experts * output, 1)


def _has_persistent_routing_layout(
    node: torch.fx.Node,
    name: str,
    *,
    tokens: int,
    experts: int,
) -> bool:
    """Check a logical ``[T,E,1]`` route tensor the loop can stream directly."""

    if _shape(node, name) != (tokens, experts, 1):
        return False
    return _stride(node, name) in {
        (experts, 1, 1),  # contiguous token-major storage
        (1, tokens, 1),  # logical token-major view over expert-major storage
    }


def _plan_node(
    node: torch.fx.Node,
    *,
    persistent_available: bool,
    available_lx_bytes: int,
) -> PlannedExpertNode:
    if len(node.args) != 7:
        raise ValueError("spyre::moe_ffn expects exactly seven arguments")
    x = _tensor_argument(node, 0, "x")
    gate = _tensor_argument(node, 1, "gate_weight")
    up = _tensor_argument(node, 2, "up_weight")
    down = _tensor_argument(node, 3, "down_weight")
    routing = _tensor_argument(node, 4, "routing_weight")
    top_k, activation = node.args[5:7]

    if type(top_k) is not int or not isinstance(activation, str):
        raise ValueError("top_k and activation must be static")
    x_shape = _shape(x, "x")
    gate_shape = _shape(gate, "gate_weight")
    if len(x_shape) != 2 or len(gate_shape) != 3:
        raise ValueError("moe_ffn requires x[T,H] and gate_weight[E,H,F]")
    tokens, hidden = x_shape
    experts, gate_hidden, intermediate = gate_shape
    if gate_hidden != hidden:
        raise ValueError("gate_weight hidden dimension differs from x")
    if _shape(up, "up_weight") != (experts, hidden, intermediate):
        raise ValueError("up_weight does not match gate_weight")
    if _shape(down, "down_weight") != (experts, intermediate, hidden):
        raise ValueError("down_weight does not match gate_weight")
    if _shape(routing, "routing_weight") != (tokens, experts, 1):
        raise ValueError("routing_weight must have shape [T,E,1]")
    if not 0 < top_k <= experts:
        raise ValueError("top_k must be in [1, E]")
    if activation not in {"gelu_tanh", "silu"}:
        raise ValueError(f"unsupported activation {activation!r}")

    # Conservative feasibility check. Placement remains authoritative.
    minimum_lx_bytes = _element_size(x, "x") * (
        2 * tokens * hidden + 2 * tokens * intermediate
    )
    packed_weights = (
        _has_persistent_weight_layout(
            gate,
            "gate_weight",
            experts=experts,
            reduction=hidden,
            output=intermediate,
        )
        and _has_persistent_weight_layout(
            up,
            "up_weight",
            experts=experts,
            reduction=hidden,
            output=intermediate,
        )
        and _has_persistent_weight_layout(
            down,
            "down_weight",
            experts=experts,
            reduction=intermediate,
            output=hidden,
        )
    )
    packed_routing = _has_persistent_routing_layout(
        routing,
        "routing_weight",
        tokens=tokens,
        experts=experts,
    )
    use_persistent = (
        persistent_available
        and packed_weights
        and packed_routing
        and minimum_lx_bytes <= available_lx_bytes
    )
    return PlannedExpertNode(
        node_name=node.name,
        strategy=(
            ExpertStrategy.PERSISTENT_DENSE
            if use_persistent
            else ExpertStrategy.ORDINARY_DENSE
        ),
        expert_count=experts,
        minimum_lx_bytes=minimum_lx_bytes,
        schedule=PersistentExpertSchedule() if use_persistent else None,
    )


def plan_expert_execution_graph(
    graph_module: torch.fx.GraphModule,
    *,
    persistent_available: bool,
    available_lx_bytes: int,
) -> ExpertGraphPlan:
    """Select strategies without modifying graph structure or metadata."""
    before = graph_structure(graph_module.graph)
    plans = tuple(
        _plan_node(
            node,
            persistent_available=persistent_available,
            available_lx_bytes=available_lx_bytes,
        )
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == moe_ffn._opoverload
    )
    if graph_structure(graph_module.graph) != before:
        raise RuntimeError("expert planning mutated the source FX graph")
    return ExpertGraphPlan(before, plans)


def _find_node(graph: torch.fx.Graph, name: str) -> torch.fx.Node:
    matches = [node for node in graph.nodes if node.name == name]
    if len(matches) != 1:
        raise ExpertPlanningError(
            f"expected one FX node named {name}, got {len(matches)}"
        )
    return matches[0]


def materialize_expert_execution_graph(
    graph_module: torch.fx.GraphModule,
    plan: ExpertGraphPlan,
) -> torch.fx.GraphModule:
    """Rewrite a clone and publish it only after all checks succeed."""
    if graph_structure(graph_module.graph) != plan.source_structure:
        raise ExpertPlanningError("FX graph changed after expert planning")
    persistent_nodes = tuple(
        node for node in plan.nodes if node.strategy == ExpertStrategy.PERSISTENT_DENSE
    )
    if not persistent_nodes:
        return graph_module

    candidate = copy.deepcopy(graph_module)
    for node_plan in persistent_nodes:
        node = _find_node(candidate.graph, node_plan.node_name)
        if node.target != moe_ffn._opoverload or node_plan.schedule is None:
            raise ExpertPlanningError(
                f"{node_plan.node_name} is not a valid persistent MoE node"
            )
        node.target = dense_expert_persistent_ffn._opoverload
        # The stable FX node name is the compiler-owned region identity.  It is
        # serialized as a static internal-op argument so the selected region
        # survives AOT decomposition without relying on user hint IDs.
        node.args = (*node.args, node_plan.node_name)
    candidate.graph.lint()
    candidate.recompile()

    for node_plan in persistent_nodes:
        if (
            _find_node(candidate.graph, node_plan.node_name).target
            != dense_expert_persistent_ffn._opoverload
        ):
            raise ExpertPlanningError(f"failed to materialize {node_plan.node_name}")
    if graph_structure(graph_module.graph) != plan.source_structure:
        raise RuntimeError("expert materialization mutated the source FX graph")
    return candidate


def prepare_expert_execution_graph(
    graph_module: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, ExpertGraphPlan]:
    """Plan once, then transactionally materialize compiler-owned strategies."""
    from .. import config
    from ..scratchpad.allocator import _lx_planning_size

    plan = plan_expert_execution_graph(
        graph_module,
        persistent_available=config.enable_dense_expert_persistent,
        available_lx_bytes=config.sencores * _lx_planning_size(),
    )
    return materialize_expert_execution_graph(graph_module, plan), plan
