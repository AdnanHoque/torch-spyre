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

"""Semantic definition of the routed-expert FFN operation."""

from __future__ import annotations

import torch


SUPPORTED_ACTIVATIONS = ("gelu_tanh", "silu")


def validate_moe_ffn_inputs(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    top_k: int,
    activation: str,
) -> tuple[int, int, int, int]:
    """Validate the logical, strategy-neutral routed-FFN tensor contract."""
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"unsupported MoE activation {activation!r}; "
            f"expected one of {SUPPORTED_ACTIVATIONS}"
        )
    if x.ndim != 2:
        raise ValueError(f"x must have shape [T,H], got {tuple(x.shape)}")
    if gate_weight.ndim != 3 or up_weight.ndim != 3 or down_weight.ndim != 3:
        raise ValueError("expert weights must be rank-3 logical weight banks")
    if routing_weight.ndim != 3 or routing_weight.shape[-1] != 1:
        raise ValueError(
            "routing_weight must have shape [T,E,1]; the explicit singleton "
            "preserves post-down scalar broadcast semantics"
        )

    tokens, hidden = x.shape
    experts, gate_hidden, intermediate = gate_weight.shape
    if top_k <= 0 or top_k > experts:
        raise ValueError(f"top_k must be in [1,{experts}], got {top_k}")
    expected_up = (experts, hidden, intermediate)
    expected_down = (experts, intermediate, hidden)
    expected_routing = (tokens, experts, 1)
    if gate_hidden != hidden:
        raise ValueError(
            f"gate_weight has hidden size {gate_hidden}, expected {hidden}"
        )
    if tuple(up_weight.shape) != expected_up:
        raise ValueError(
            f"up_weight must have shape {expected_up}, got {tuple(up_weight.shape)}"
        )
    if tuple(down_weight.shape) != expected_down:
        raise ValueError(
            "down_weight must have shape "
            f"{expected_down}, got {tuple(down_weight.shape)}"
        )
    if tuple(routing_weight.shape) != expected_routing:
        raise ValueError(
            "routing_weight must have shape "
            f"{expected_routing}, got {tuple(routing_weight.shape)}"
        )
    if any(
        tensor.device != x.device
        for tensor in (gate_weight, up_weight, down_weight, routing_weight)
    ):
        raise ValueError("all MoE tensors must be on the same device")
    if any(
        tensor.dtype != x.dtype
        for tensor in (gate_weight, up_weight, down_weight, routing_weight)
    ):
        raise ValueError("all MoE tensors must have the same dtype")
    return tokens, experts, hidden, intermediate


def moe_ffn_reference(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    top_k: int,
    activation: str,
) -> torch.Tensor:
    """Evaluate the logical all-expert routed FFN without strategy assumptions.

    The reference intentionally expresses one expert at a time. It defines
    semantics for eager execution and correctness tests; compiler strategies
    are free to use a persistent device loop, partitioned dense execution, or
    another implementation that preserves this result.
    """
    _tokens, experts, _hidden, _intermediate = validate_moe_ffn_inputs(
        x,
        gate_weight,
        up_weight,
        down_weight,
        routing_weight,
        top_k,
        activation,
    )
    result = torch.zeros_like(x)
    for expert in range(experts):
        gate = torch.mm(x, gate_weight[expert])
        up = torch.mm(x, up_weight[expert])
        if activation == "gelu_tanh":
            gate = torch.nn.functional.gelu(gate, approximate="tanh")
        else:
            gate = torch.nn.functional.silu(gate)
        down = torch.mm(gate * up, down_weight[expert])
        result = result + down * routing_weight[:, expert, :]
    return result
