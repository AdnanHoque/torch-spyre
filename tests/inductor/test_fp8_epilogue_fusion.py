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

"""Unit coverage for the private DD2 FP8 BMM epilogue experiment."""

import os
from unittest.mock import patch

import sympy

from torch_spyre._C import DataFormats, ElementArrangement
from torch_spyre._inductor.codegen.bundle import (
    _fuse_fp8_bmm_first_scale_epilogue,
)
from torch_spyre._inductor.op_spec import LoopSpec, OpSpec, TensorArg


def _tensor(
    *,
    is_input: bool,
    allocation: dict[str, int],
    dtype=DataFormats.SEN169_FP16,
    arrangement=ElementArrangement.STANDARD,
) -> TensorArg:
    return TensorArg(
        is_input=is_input,
        arg_index=-1,
        device_dtype=dtype,
        device_size=[1, 64],
        device_coordinates=[sympy.Integer(0), sympy.Symbol("n")],
        allocation=allocation,
        element_arrangement=arrangement,
    )


def _op(op: str, args: list[TensorArg], *, is_reduction: bool = False) -> OpSpec:
    return OpSpec(
        op=op,
        is_reduction=is_reduction,
        iteration_space={sympy.Symbol("n"): (sympy.Integer(64), 1)},
        args=args,
        op_info={},
    )


def _candidate_specs(
    allocation_base: int = 0,
) -> tuple[list[OpSpec], TensorArg, TensorArg]:
    activation = _tensor(
        is_input=True,
        allocation={"hbm_pool": allocation_base},
        dtype=DataFormats.SEN143_FP8,
        arrangement=ElementArrangement.QFP8MB,
    )
    weight = _tensor(
        is_input=True,
        allocation={"hbm": 0},
        dtype=DataFormats.SEN143_FP8,
        arrangement=ElementArrangement.QFP8WT,
    )
    raw_output = _tensor(
        is_input=False, allocation={"hbm_pool": allocation_base + 4096}
    )
    scale = _tensor(is_input=False, allocation={"hbm_pool": allocation_base})
    zero = _tensor(is_input=True, allocation={"hbm_pool": allocation_base + 8192})
    scaled_output = _tensor(
        is_input=False, allocation={"hbm_pool": allocation_base + 12288}
    )

    scale_producer = _op(
        "fp32todl16",
        [
            _tensor(is_input=True, allocation={"hbm": 1}),
            scale,
        ],
    )
    bmm = _op(
        "batchmatmulfp8mb",
        [activation, weight, raw_output],
        is_reduction=True,
    )
    first_scale = _op(
        "batchnormfwd",
        [
            # Input/output role is intentionally different; storage identity
            # is what links the two OpSpecs.
            _tensor(
                is_input=True,
                allocation={"hbm_pool": allocation_base + 4096},
            ),
            _tensor(is_input=True, allocation={"hbm_pool": allocation_base}),
            zero,
            scaled_output,
        ],
    )
    return [bmm, scale_producer, first_scale], raw_output, scale


def test_private_fp8_epilogue_fuses_and_repairs_extended_scale_lifetime():
    specs, raw_output, scale = _candidate_specs()

    with patch.dict(
        os.environ,
        {"TORCH_SPYRE_FP8_FUSE_FIRST_SCALE_EPILOGUE": "1"},
    ):
        transformed = _fuse_fp8_bmm_first_scale_epilogue(specs)

    assert [spec.op for spec in transformed] == [
        "fp32todl16",
        "batchmatmulfp8mb",
    ]
    scale_producer, fused = transformed
    assert fused.op_info["fp8_batchnorm_epilogue"] is True
    assert len(fused.args) == 5
    assert fused.args[2].allocation == raw_output.allocation
    assert scale_producer.args[-1].allocation == raw_output.allocation
    assert scale.allocation == {"hbm_pool": 0}


def test_private_fp8_epilogue_is_opt_in_and_rejects_loop_specs():
    specs, _, _ = _candidate_specs()

    with patch.dict(
        os.environ,
        {"TORCH_SPYRE_FP8_FUSE_FIRST_SCALE_EPILOGUE": "0"},
    ):
        assert _fuse_fp8_bmm_first_scale_epilogue(specs) is specs

    loop_specs = [LoopSpec(count=sympy.Integer(1), body=specs)]
    with patch.dict(
        os.environ,
        {"TORCH_SPYRE_FP8_FUSE_FIRST_SCALE_EPILOGUE": "1"},
    ):
        assert _fuse_fp8_bmm_first_scale_epilogue(loop_specs) is loop_specs


def test_private_fp8_epilogue_fuses_every_eligible_pair_in_bundle():
    first, _, _ = _candidate_specs(0)
    second, _, _ = _candidate_specs(65536)

    with patch.dict(
        os.environ,
        {"TORCH_SPYRE_FP8_FUSE_FIRST_SCALE_EPILOGUE": "1"},
    ):
        transformed = _fuse_fp8_bmm_first_scale_epilogue(first + second)

    assert [spec.op for spec in transformed] == [
        "fp32todl16",
        "batchmatmulfp8mb",
        "fp32todl16",
        "batchmatmulfp8mb",
    ]
    fused = [spec for spec in transformed if spec.op == "batchmatmulfp8mb"]
    assert len(fused) == 2
    assert all(spec.op_info["fp8_batchnorm_epilogue"] for spec in fused)
    assert all(len(spec.args) == 5 for spec in fused)
