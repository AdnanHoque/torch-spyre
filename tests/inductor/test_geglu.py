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

"""Regression for #5458, with fixed weights and a common input prefix."""

import math

import pytest
import torch
import torch.nn.functional as F


def fixed_inputs(tokens):
    # Preserve the issue's *failing* 2048-token tensors exactly. Draw the extra
    # 64 tokens separately so changing the extent cannot change either weight.
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 2048, 768, generator=generator).half()
    wi = (0.02 * torch.randn(2304, 768, generator=generator)).half()
    wo = (0.02 * torch.randn(768, 1152, generator=generator)).half()
    tail = torch.randn(1, 64, 768, generator=torch.Generator().manual_seed(5678)).half()
    assert tokens in (1984, 2048, 2112)
    return torch.cat((x, tail), dim=1)[:, :tokens].contiguous(), wi, wo


def tanh_gelu(x):
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    )


def geglu(x, wi, wo, *, decomposed=False, approximate="tanh"):
    inp, gate = F.linear(x, wi).chunk(2, dim=-1)
    activated = tanh_gelu(inp) if decomposed else F.gelu(inp, approximate=approximate)
    return F.linear(activated * gate, wo)


def reference(x, wi, wo, *, approximate="tanh"):
    return geglu(x.float(), wi.float(), wo.float(), approximate=approximate).half()


@pytest.mark.parametrize("tokens", [1984, 2048, 2112])
@pytest.mark.parametrize(
    ("decomposed", "approximate"),
    [(False, "none"), (False, "tanh"), (True, "tanh")],
    ids=["native-none", "native-tanh", "decomposed-tanh"],
)
def test_fused_geglu(tokens, decomposed, approximate):
    # Keep the final-output-only graph: returning intermediates changes fusion
    # and allocation, so it could mask a defect. Match the GELU approximation.
    import torch_spyre  # noqa: F401

    # Each parameter case must compile independently of earlier guard sets.
    torch._dynamo.reset()
    x, wi, wo = fixed_inputs(tokens)
    # Native Spyre currently uses tanh for both modes. The none case guards
    # the public reproducer, not full erf semantics (the mode gap here is tiny).
    expected = reference(x, wi, wo, approximate=approximate)

    def fn(x, wi, wo):
        return geglu(x, wi, wo, decomposed=decomposed, approximate=approximate)

    with torch._inductor.config.patch(force_disable_caches=True):
        actual = torch.compile(fn, fullgraph=True, dynamic=False)(
            x.to("spyre"), wi.to("spyre"), wo.to("spyre")
        ).cpu()
    assert torch.isfinite(actual).all()
    difference = (actual.float() - expected.float()).abs()
    bad = difference > 0.05
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0,
        atol=0.05,
        msg=lambda message: (
            f"tokens={tokens}, decomposed={decomposed}, approximate={approximate}; "
            f"max_error={difference.max().item()}, "
            f"mismatches={bad.sum().item()}/{bad.numel()}, "
            f"affected_tokens={bad.any(dim=-1).sum().item()}, "
            f"affected_columns={bad.any(dim=1).sum().item()}\n{message}"
        ),
    )


def test_gelu_pinned_layout_reorder():
    """Keep #5458 covered even if Inductor changes its preferred GEGLU layout."""
    from sympy import Integer, Mod, floor, symbols

    from op_specs.runner import run_op_specs
    from torch_spyre._C import DataFormats, ElementArrangement, SpyreTensorLayout
    from torch_spyre._inductor import config
    from torch_spyre._inductor.op_spec import OpSpec, TensorArg

    c0, c1, core_id = symbols("c0 c1 core_id")
    dtype = DataFormats.SEN169_FP16
    source = torch.randn(
        2048, 2304, generator=torch.Generator().manual_seed(5458)
    ).half()
    expected = F.gelu(source[:, :1152].float(), approximate="tanh").half()
    actual = torch.zeros_like(expected)
    layouts = [
        SpyreTensorLayout(
            device_size=size,
            stride_map=stride,
            device_dtype=dtype,
            element_arrangement=ElementArrangement.STANDARD,
        )
        for size, stride in [
            ([2048, 36, 64], [2304, 64, 1]),
            ([18, 2048, 64], [64, 1152, 1]),
        ]
    ]
    op = OpSpec(
        op="gelufwd",
        is_reduction=False,
        iteration_space={c0: (Integer(2048), 32), c1: (Integer(1152), 1)},
        op_info={},
        core_id_to_work_slice={c0: Mod(core_id, 32), c1: Integer(0)},
        args=[
            TensorArg(
                is_input=index == 0,
                arg_index=index,
                device_dtype=dtype,
                device_size=size,
                device_coordinates=coordinates,
                allocation={"hbm": index},
            )
            for index, size, coordinates in [
                (0, [2048, 36, 64], [c0, floor(c1 / 64), Mod(c1, 64)]),
                (1, [18, 2048, 64], [floor(c1 / 64), c0, Mod(c1, 64)]),
            ]
        ],
    )
    with config.patch(bundle_symbolic_args=True):
        run_op_specs("gelu_pinned_layout_5458", [op], [source, actual], layouts=layouts)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.05, rtol=0)
