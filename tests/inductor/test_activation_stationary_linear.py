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

from torch_spyre._inductor import config
from torch_spyre._inductor.decompositions import spyre_linear
from torch_spyre._inductor.errors import Unsupported


@pytest.mark.parametrize(
    "shape",
    [
        (1, 128),
        (8, 128),
        (1, 1, 128),
        (2, 4, 128),
        (64, 128),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_activation_stationary_linear_matches_reference(shape, with_bias):
    torch.manual_seed(20260730)
    activation = torch.randn(shape, dtype=torch.float16) * 0.125
    weight = torch.randn((256, shape[-1]), dtype=torch.float16) * 0.125
    bias = (
        torch.randn((weight.shape[0],), dtype=torch.float16) * 0.125
        if with_bias
        else None
    )
    expected = torch.nn.functional.linear(activation, weight, bias)

    with config.patch({"matmul_dataflow": "activation_stationary"}):
        actual = spyre_linear(activation, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2.5e-1)
    assert actual.shape == expected.shape


def test_activation_stationary_linear_falls_back_above_physical_m64():
    torch.manual_seed(20260730)
    activation = torch.randn((65, 128), dtype=torch.float16) * 0.125
    weight = torch.randn((256, 128), dtype=torch.float16) * 0.125
    expected = torch.nn.functional.linear(activation, weight)

    with config.patch({"matmul_dataflow": "activation_stationary"}):
        actual = spyre_linear(activation, weight)

    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=2.5e-1)


def test_activation_stationary_shape_allowlist_can_fall_back(monkeypatch):
    activation = torch.randn((1, 128), dtype=torch.float16)
    weight = torch.randn((256, 128), dtype=torch.float16)

    def unexpected_pad(*args, **kwargs):
        raise AssertionError("non-selected shape entered activation stationary")

    monkeypatch.setattr(torch.nn.functional, "pad", unexpected_pad)
    with config.patch(
        {
            "matmul_dataflow": "activation_stationary",
            "activation_stationary_shapes": "128x64",
        }
    ):
        actual = spyre_linear(activation, weight)

    assert actual.shape == (1, 256)


def test_invalid_activation_stationary_shape_allowlist_is_rejected():
    activation = torch.randn((1, 128), dtype=torch.float16)
    weight = torch.randn((256, 128), dtype=torch.float16)

    with (
        config.patch(
            {
                "matmul_dataflow": "activation_stationary",
                "activation_stationary_shapes": "not-a-shape",
            }
        ),
        pytest.raises(Unsupported, match="comma-separated KxN list"),
    ):
        spyre_linear(activation, weight)


def test_weight_stationary_remains_default():
    assert config.matmul_dataflow == "weight_stationary"


def test_invalid_matmul_dataflow_is_rejected():
    activation = torch.randn((1, 128), dtype=torch.float16)
    weight = torch.randn((256, 128), dtype=torch.float16)

    with (
        config.patch({"matmul_dataflow": "invalid"}),
        pytest.raises(Unsupported, match="unsupported matmul dataflow"),
    ):
        spyre_linear(activation, weight)
