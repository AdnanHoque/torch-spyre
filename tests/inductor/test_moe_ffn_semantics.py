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

from contextlib import contextmanager

import pytest
import torch

from torch_spyre._inductor import decompositions
from torch_spyre._inductor.expert_execution.custom_op import (
    expert_mm_prepacked,
    expert_route_prepacked,
    expert_shared_lhs_mm,
    expert_shared_lhs_mm_prepacked,
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


def test_internal_prepacked_projections_match_logical_expert_loop():
    x, gate, _up, down, _routing = _inputs()
    gate_packed = gate.permute(1, 0, 2)
    down_packed = down.permute(1, 0, 2)

    projected = expert_shared_lhs_mm_prepacked(x, gate_packed)
    expected_projected = torch.stack(tuple(torch.mm(x, weight) for weight in gate))
    restored = expert_mm_prepacked(projected, down_packed)
    expected_restored = torch.stack(
        tuple(torch.mm(expected_projected[e], down[e]) for e in range(gate.shape[0]))
    )

    torch.testing.assert_close(projected, expected_projected)
    torch.testing.assert_close(restored, expected_restored)


def test_internal_prepacked_route_is_an_identity():
    routing = _inputs()[-1].permute(1, 0, 2).contiguous()

    torch.testing.assert_close(expert_route_prepacked(routing), routing)


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


def test_persistent_decomposition_marks_the_complete_loop_body(monkeypatch):
    regions = []

    @contextmanager
    def record_region(scope_id, **metadata):
        regions.append((scope_id, metadata))
        yield

    monkeypatch.setattr(decompositions, "compiler_hint", record_region)
    decompose_dense_expert_persistent_ffn(*_inputs(), 2, "gelu_tanh", "region-7")

    assert len(regions) == 8
    assert {scope_id for scope_id, _ in regions} == {"region-7"}
    assert all(
        metadata["expert_execution"] == "persistent_dense_expert"
        for _, metadata in regions
    )
    assert [metadata.get("streamed_operand_role") for _, metadata in regions] == [
        "gate_weight",
        "up_weight",
        None,
        None,
        "down_weight",
        "routing_weight",
        None,
        None,
    ]
