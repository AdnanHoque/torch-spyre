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

import pytest
import torch

from torch_spyre._inductor.expert_execution.custom_op import (
    expert_shared_lhs_mm,
    moe_ffn,
)
from torch_spyre._inductor.expert_execution.semantics import moe_ffn_reference
from torch_spyre._inductor.decompositions import (
    decompose_dense_expert_persistent_ffn,
    get_spyre_decomp_table,
)


def _inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(17)
    tokens, experts, hidden, intermediate = 4, 3, 8, 6
    x = torch.randn(tokens, hidden, dtype=dtype, generator=generator)
    gate = torch.randn(experts, hidden, intermediate, dtype=dtype, generator=generator)
    up = torch.randn(experts, hidden, intermediate, dtype=dtype, generator=generator)
    down = torch.randn(experts, intermediate, hidden, dtype=dtype, generator=generator)
    routing = torch.randn(tokens, experts, 1, dtype=dtype, generator=generator)
    return x, gate, up, down, routing


@pytest.mark.parametrize("activation", ["gelu_tanh", "silu"])
def test_moe_ffn_cpu_matches_direct_reference(activation):
    inputs = _inputs()

    actual = moe_ffn(*inputs, 2, activation)
    expected = moe_ffn_reference(*inputs, 2, activation)

    torch.testing.assert_close(actual, expected)


def test_zero_routing_weight_removes_an_expert_contribution():
    x, gate, up, down, routing = _inputs()
    routing[:, 1, :] = 0

    actual = moe_ffn(x, gate, up, down, routing, 2, "silu")
    expected = moe_ffn(
        x,
        gate[[0, 2]],
        up[[0, 2]],
        down[[0, 2]],
        routing[:, [0, 2], :],
        2,
        "silu",
    )

    torch.testing.assert_close(actual, expected)


def test_moe_ffn_requires_explicit_singleton_routing_axis():
    x, gate, up, down, routing = _inputs()

    with pytest.raises(ValueError, match=r"\[T,E,1\]"):
        moe_ffn(x, gate, up, down, routing.squeeze(-1), 2, "silu")


def test_moe_ffn_rejects_mismatched_logical_weight_schema():
    x, gate, up, down, routing = _inputs()

    with pytest.raises(ValueError, match="up_weight"):
        moe_ffn(x, gate, up[:, :-1], down, routing, 2, "silu")


def test_moe_ffn_rejects_invalid_top_k():
    inputs = _inputs()

    with pytest.raises(ValueError, match="top_k"):
        moe_ffn(*inputs, 4, "silu")


def test_internal_shared_lhs_projection_matches_expert_loop():
    x, gate, _up, _down, _routing = _inputs()

    actual = expert_shared_lhs_mm(x, gate)
    expected = torch.stack(tuple(torch.mm(x, weight) for weight in gate))

    assert actual.shape == (gate.shape[0], x.shape[0], gate.shape[2])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("activation", ["gelu_tanh", "silu"])
def test_persistent_decomposition_matches_semantic_reference(activation):
    inputs = _inputs()

    actual = decompose_dense_expert_persistent_ffn(
        *inputs, 2, activation, "test_region"
    )
    expected = moe_ffn_reference(*inputs, 2, activation)

    torch.testing.assert_close(actual, expected)
    assert (
        torch.ops.spyre.dense_expert_persistent_ffn.default in get_spyre_decomp_table()
    )
