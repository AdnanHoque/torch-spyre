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

import torch_spyre._inductor.customops  # noqa: F401


def test_shared_lhs_fake_contract_does_not_expand_the_activation():
    x = torch.empty((5, 7), device="meta", dtype=torch.float16)
    weights = torch.empty((3, 7, 11), device="meta", dtype=torch.float16)

    out = torch.ops.spyre.activation_stationary_shared_lhs_mm.default(x, weights)

    assert tuple(out.shape) == (3, 5, 11)
    assert tuple(x.shape) == (5, 7)


def test_prepacked_fake_contracts_preserve_expert_axis():
    x = torch.empty((5, 7), device="meta", dtype=torch.float16)
    hidden = torch.empty((3, 5, 7), device="meta", dtype=torch.float16)
    weights = torch.empty((7, 3, 11), device="meta", dtype=torch.float16)

    gate = torch.ops.spyre.activation_stationary_shared_lhs_mm_prepacked.default(
        x, weights
    )
    down = torch.ops.spyre.activation_stationary_expert_mm_prepacked.default(
        hidden, weights
    )

    assert tuple(gate.shape) == (3, 5, 11)
    assert tuple(down.shape) == (3, 5, 11)


@pytest.mark.parametrize(
    "x_shape,weight_shape",
    [
        ((5, 7, 1), (3, 7, 11)),
        ((5, 7), (3, 8, 11)),
    ],
)
def test_shared_lhs_fake_contract_rejects_invalid_shapes(x_shape, weight_shape):
    x = torch.empty(x_shape, device="meta", dtype=torch.float16)
    weights = torch.empty(weight_shape, device="meta", dtype=torch.float16)

    with pytest.raises(RuntimeError):
        torch.ops.spyre.activation_stationary_shared_lhs_mm.default(x, weights)
