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

"""Strategy-neutral routed-expert FFN custom operation."""

import torch

from .semantics import moe_ffn_reference, validate_moe_ffn_inputs


@torch.library.custom_op(
    "spyre::expert_shared_lhs_mm", mutates_args=(), device_types=("cpu", "spyre")
)
def expert_shared_lhs_mm(x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    """Evaluate the internal shared-LHS expert projection in eager mode."""
    _validate_shared_lhs_mm(x, expert_weights)
    return torch.stack(
        tuple(
            torch.mm(x, expert_weights[expert]) for expert in range(len(expert_weights))
        )
    )


@expert_shared_lhs_mm.register_fake
def _(x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    _validate_shared_lhs_mm(x, expert_weights)
    return x.new_empty((expert_weights.shape[0], x.shape[0], expert_weights.shape[2]))


@torch.library.custom_op(
    "spyre::expert_shared_lhs_mm_prepacked",
    mutates_args=(),
    device_types=("cpu", "spyre"),
)
def expert_shared_lhs_mm_prepacked(
    x: torch.Tensor, expert_weights: torch.Tensor
) -> torch.Tensor:
    """Internal shared-LHS projection with physical weights ``[K,E,N]``."""

    _validate_shared_lhs_mm_prepacked(x, expert_weights)
    return torch.stack(
        tuple(
            torch.mm(x, expert_weights[:, expert, :])
            for expert in range(expert_weights.shape[1])
        )
    )


@expert_shared_lhs_mm_prepacked.register_fake
def _(x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    _validate_shared_lhs_mm_prepacked(x, expert_weights)
    return x.new_empty((expert_weights.shape[1], x.shape[0], expert_weights.shape[2]))


def _validate_shared_lhs_mm_prepacked(
    x: torch.Tensor, expert_weights: torch.Tensor
) -> None:
    if x.ndim != 2 or expert_weights.ndim != 3:
        raise ValueError("expected x[T,K] and expert_weights[K,E,N]")
    if x.shape[1] != expert_weights.shape[0]:
        raise ValueError("x and expert_weights have different reduction dimensions")
    if x.device != expert_weights.device or x.dtype != expert_weights.dtype:
        raise ValueError("x and expert_weights must share device and dtype")


@torch.library.custom_op(
    "spyre::expert_mm_prepacked", mutates_args=(), device_types=("cpu", "spyre")
)
def expert_mm_prepacked(x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    """Internal expert projection with physical weights ``[K,E,N]``."""

    _validate_expert_mm_prepacked(x, expert_weights)
    return torch.stack(
        tuple(
            torch.mm(x[expert], expert_weights[:, expert, :])
            for expert in range(expert_weights.shape[1])
        )
    )


@expert_mm_prepacked.register_fake
def _(x: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    _validate_expert_mm_prepacked(x, expert_weights)
    return x.new_empty((x.shape[0], x.shape[1], expert_weights.shape[2]))


def _validate_expert_mm_prepacked(
    x: torch.Tensor, expert_weights: torch.Tensor
) -> None:
    if x.ndim != 3 or expert_weights.ndim != 3:
        raise ValueError("expected x[E,T,K] and expert_weights[K,E,N]")
    if x.shape[0] != expert_weights.shape[1] or x.shape[2] != expert_weights.shape[0]:
        raise ValueError("expert or reduction dimension mismatch")
    if x.device != expert_weights.device or x.dtype != expert_weights.dtype:
        raise ValueError("x and expert_weights must share device and dtype")


@torch.library.custom_op(
    "spyre::expert_route_prepacked", mutates_args=(), device_types=("cpu", "spyre")
)
def expert_route_prepacked(routing_weight: torch.Tensor) -> torch.Tensor:
    """Internal identity over an expert-major ``[E,T,1]`` routing tensor."""

    _validate_expert_route_prepacked(routing_weight)
    return routing_weight.clone()


@expert_route_prepacked.register_fake
def _(routing_weight: torch.Tensor) -> torch.Tensor:
    _validate_expert_route_prepacked(routing_weight)
    return routing_weight.new_empty(routing_weight.shape)


def _validate_expert_route_prepacked(routing_weight: torch.Tensor) -> None:
    if routing_weight.ndim != 3 or routing_weight.shape[2] != 1:
        raise ValueError("expected routing_weight[E,T,1]")


def _validate_shared_lhs_mm(x: torch.Tensor, expert_weights: torch.Tensor) -> None:
    if x.ndim != 2:
        raise ValueError(f"x must have shape [T,K], got {tuple(x.shape)}")
    if expert_weights.ndim != 3:
        raise ValueError(
            f"expert_weights must have shape [E,K,N], got {tuple(expert_weights.shape)}"
        )
    if x.shape[1] != expert_weights.shape[1]:
        raise ValueError("x and expert_weights have different reduction dimensions")
    if x.device != expert_weights.device or x.dtype != expert_weights.dtype:
        raise ValueError("x and expert_weights must share device and dtype")


def _internal_moe_ffn(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    top_k: int,
    activation: str,
) -> torch.Tensor:
    return moe_ffn_reference(
        x,
        gate_weight,
        up_weight,
        down_weight,
        routing_weight,
        top_k,
        activation,
    )


def _fake_internal_moe_ffn(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    top_k: int,
    activation: str,
    region_id: str,
) -> torch.Tensor:
    del region_id
    validate_moe_ffn_inputs(
        x,
        gate_weight,
        up_weight,
        down_weight,
        routing_weight,
        top_k,
        activation,
    )
    return x.new_empty(x.shape)


@torch.library.custom_op(
    "spyre::dense_expert_persistent_ffn",
    mutates_args=(),
    device_types=("cpu", "spyre"),
    schema=(
        "(Tensor x, Tensor gate_weight, Tensor up_weight, Tensor down_weight, "
        "Tensor routing_weight, int top_k, str activation, str region_id) -> Tensor"
    ),
)
def dense_expert_persistent_ffn(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    top_k: int,
    activation: str,
    region_id: str,
) -> torch.Tensor:
    """Compiler-internal target for the persistent strategy."""
    del region_id
    return _internal_moe_ffn(
        x,
        gate_weight,
        up_weight,
        down_weight,
        routing_weight,
        top_k,
        activation,
    )


dense_expert_persistent_ffn.register_fake(_fake_internal_moe_ffn)


@torch.library.custom_op(
    "spyre::moe_ffn",
    mutates_args=(),
    device_types=("cpu", "spyre"),
    schema=(
        "(Tensor x, Tensor gate_weight, Tensor up_weight, Tensor down_weight, "
        "Tensor routing_weight, int top_k, str activation) -> Tensor"
    ),
)
def moe_ffn(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    top_k: int,
    activation: str,
) -> torch.Tensor:
    """Evaluate the routed-expert value path in eager mode.

    Router-logit generation and any shared-expert FFN are intentionally outside
    this operation. ``routing_weight`` is the already-normalized post-down
    expert weight with logical shape ``[T, E, 1]``.
    """
    return moe_ffn_reference(
        x,
        gate_weight,
        up_weight,
        down_weight,
        routing_weight,
        top_k,
        activation,
    )


@moe_ffn.register_fake
def _(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    routing_weight: torch.Tensor,
    top_k: int,
    activation: str,
) -> torch.Tensor:
    validate_moe_ffn_inputs(
        x,
        gate_weight,
        up_weight,
        down_weight,
        routing_weight,
        top_k,
        activation,
    )
    return x.new_empty(x.shape)
