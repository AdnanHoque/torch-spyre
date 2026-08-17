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

import math

import pytest
import sympy

import torch_spyre._inductor.codegen.superdsc as superdsc_module
import torch_spyre._inductor.pass_utils as pass_utils_module
import torch_spyre._inductor.spyre_kernel as spyre_kernel_module
from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import compile_op_spec, parse_op_spec
from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_OP,
    CORE_MAPPING_CONTIGUOUS_DIM_INFO_KEY,
    IDENTITY_OP,
)
from torch_spyre._inductor.core_mapping import core_to_slice_mapping
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _coordinates(splits, num_cores, **kwargs):
    dims = sympy.symbols(f"dim_0:{len(splits)}")
    mapping = core_to_slice_mapping(dims, splits, num_cores, **kwargs)
    core_id = sympy.Symbol("core_id")
    return [
        tuple(int(mapping[dim].subs(core_id, core)) for dim in dims)
        for core in range(num_cores)
    ]


def test_default_mapping_preserves_existing_core_order():
    one_grid = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2)]
    assert _coordinates((2, 3), 12) == one_grid * 2


@pytest.mark.parametrize("contiguous_dim", [0, 1, 2])
def test_selected_dim_varies_first(contiguous_dim):
    splits = (2, 3, 4)
    coordinates = _coordinates(
        splits,
        math.prod(splits),
        contiguous_dim=contiguous_dim,
    )
    assert [
        coordinate[contiguous_dim]
        for coordinate in coordinates[: splits[contiguous_dim]]
    ] == list(range(splits[contiguous_dim]))
    assert all(
        coordinate[dim] == 0
        for coordinate in coordinates[: splits[contiguous_dim]]
        for dim in range(len(splits))
        if dim != contiguous_dim
    )


def _bmm_op_spec(op: str) -> OpSpec:
    mb, out, reduction = sympy.symbols("mb out reduction")
    args = [
        TensorArg(
            True,
            0,
            DataFormats.SEN169_FP16,
            [512, 64, 1, 64],
            [
                mb,
                sympy.floor(reduction / 64),
                sympy.Integer(0),
                sympy.Mod(reduction, 64),
            ],
            {"hbm": 0},
        ),
        TensorArg(
            True,
            1,
            DataFormats.SEN169_FP16,
            [200, 4096, 64],
            [sympy.floor(out / 64), reduction, sympy.Mod(out, 64)],
            {"hbm": 0x400000000},
        ),
        TensorArg(
            False,
            2,
            DataFormats.SEN169_FP16,
            [512, 200, 1, 64],
            [
                mb,
                sympy.floor(out / 64),
                sympy.Integer(0),
                sympy.Mod(out, 64),
            ],
            {"hbm": 0x800000000},
        ),
    ]
    return OpSpec(
        op,
        True,
        {mb: (512, 2), out: (12800, 4), reduction: (4096, 4)},
        args,
        {},
    )


@pytest.mark.parametrize("op", [BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP])
@pytest.mark.parametrize("reduction_contiguous", [False, True])
def test_planner_and_sdsc_use_the_same_mapping(monkeypatch, op, reduction_contiguous):
    class FakeReduction:
        def __init__(self, reduction_type):
            self.reduction_type = reduction_type

    class FakeComputedBuffer:
        def __init__(self, reduction_type):
            self.data = FakeReduction(reduction_type)

    monkeypatch.setattr(pass_utils_module, "Reduction", FakeReduction)
    monkeypatch.setattr(pass_utils_module, "ComputedBuffer", FakeComputedBuffer)
    monkeypatch.setattr(
        pass_utils_module.config,
        "core_id_k_fast_emission",
        reduction_contiguous,
    )
    monkeypatch.setattr(
        superdsc_module._spyre_config,
        "core_id_k_fast_emission",
        reduction_contiguous,
    )

    op_spec = _bmm_op_spec(op)
    dims = tuple(op_spec.iteration_space)
    splits = dict(zip(dims, (2, 4, 4)))
    monkeypatch.setattr(
        pass_utils_module, "apply_splits_from_index_coeff", lambda *_: splits
    )
    prep = pass_utils_module._ViewPrep(
        iter_space=op_spec.iteration_space,
        write_index=dims[0],
        read_index=dims[-1],
        dep_coeff={dims[0]: 1, dims[1]: 2, dims[2]: 0},
        device_size=[2, 4],
        stride_map=[1, 2],
        elems_per_stick=64,
        device_stride_to_dim={1: 0, 2: 1},
        stick_host_stride=None,
        num_stick_dim=None,
        num_stick=0,
        num_stick_stride=0,
        is_matmul=pass_utils_module._is_matmul_op(FakeComputedBuffer(op)),
    )
    planner_view, _, representable = pass_utils_module._per_core_view_from_prep(
        prep, ({1: 2, 2: 4}, {3: 4})
    )

    sdsc_spec, renamed = parse_op_spec(op_spec)
    sdsc_output_mapping = {
        device_dim: sdsc_spec.core_id_to_work_slice[renamed[dim]]
        for device_dim, dim in enumerate(dims[:2])
    }
    assert representable
    assert dict(planner_view.core_to_slot) == sdsc_output_mapping


def _identity_op_spec(contiguous_dim=1) -> tuple[OpSpec, sympy.Symbol, sympy.Symbol]:
    m, k = sympy.symbols("m k")
    coordinates = [sympy.floor(k / 64), m, sympy.Mod(k, 64)]
    args = [
        TensorArg(
            True,
            0,
            DataFormats.SEN169_FP16,
            [44, 512, 64],
            coordinates,
            {"hbm": 0},
        ),
        TensorArg(
            False,
            1,
            DataFormats.SEN169_FP16,
            [44, 512, 64],
            coordinates,
            {"lx": 0},
        ),
    ]
    op_info = (
        {}
        if contiguous_dim is None
        else {CORE_MAPPING_CONTIGUOUS_DIM_INFO_KEY: contiguous_dim}
    )
    return (
        OpSpec(
            IDENTITY_OP,
            False,
            {m: (512, 16), k: (2816, 2)},
            args,
            op_info,
        ),
        m,
        k,
    )


def _compiled_root(op_spec: OpSpec) -> dict:
    payload, *_ = compile_op_spec(0, op_spec, [])
    return next(iter(payload.values()))


def test_identity_override_matches_planner_and_sdsc_k_fast(monkeypatch):
    op_spec, m, k = _identity_op_spec()
    monkeypatch.setattr(
        pass_utils_module,
        "apply_splits_from_index_coeff",
        lambda *_: {m: 16, k: 2},
    )
    prep = pass_utils_module._ViewPrep(
        iter_space={m: 512, k: 2816},
        write_index=2816 * m + k,
        read_index=2816 * m + k,
        dep_coeff={m: 2816, k: 1},
        device_size=[44, 512, 64],
        stride_map=[64, 2816, 1],
        elems_per_stick=64,
        device_stride_to_dim={64: 0, 2816: 1, 1: 2},
        stick_host_stride=1,
        num_stick_dim=0,
        num_stick=44,
        num_stick_stride=64,
        is_matmul=False,
        core_mapping_contiguous_dim=1,
    )
    planner_view, partial, representable = pass_utils_module._per_core_view_from_prep(
        prep, ({2816: 16, 1: 2}, {})
    )

    sdsc_spec, renamed = parse_op_spec(op_spec)
    sdsc_mapping = {
        0: sdsc_spec.core_id_to_work_slice[renamed[k]],
        1: sdsc_spec.core_id_to_work_slice[renamed[m]],
    }

    assert representable and not partial
    assert dict(planner_view.work_slice_dims) == {0: 2, 1: 16}
    assert dict(planner_view.core_to_slot) == sdsc_mapping
    core_id = sympy.Symbol("core_id")
    assert [
        (
            int(sdsc_mapping[1].subs(core_id, core)),
            int(sdsc_mapping[0].subs(core_id, core)),
        )
        for core in range(4)
    ] == [(0, 0), (0, 1), (1, 0), (1, 1)]

    root = _compiled_root(op_spec)
    assert root["numWkSlicesPerDim_"] == {"mb": 16, "out": 2}
    assert root["coreIdToWkSlice_"] == {
        str(core): {"mb": core // 2, "out": core % 2} for core in range(32)
    }


def test_unmarked_identity_keeps_default_m_fast_mapping():
    op_spec, _, _ = _identity_op_spec(None)
    root = _compiled_root(op_spec)

    assert root["numWkSlicesPerDim_"] == {"mb": 16, "out": 2}
    assert root["coreIdToWkSlice_"] == {
        str(core): {"mb": core % 16, "out": core // 16} for core in range(32)
    }


def test_real_c32_identity_marker_survives_normalization():
    op_spec, _, _ = _identity_op_spec()
    spyre_kernel_module.simplify_op_spec(op_spec)

    contiguous_dim = op_spec.op_info[CORE_MAPPING_CONTIGUOUS_DIM_INFO_KEY]
    selected = tuple(op_spec.iteration_space)[contiguous_dim]
    assert selected in op_spec.args[-1].device_coordinates[-1].free_symbols
    root = _compiled_root(op_spec)
    assert root["numWkSlicesPerDim_"] == {"mb": 16, "out": 2}
    assert root["coreIdToWkSlice_"] == {
        str(core): {"mb": core // 2, "out": core % 2} for core in range(32)
    }


@pytest.mark.parametrize("invalid", [-1, 2, True])
def test_identity_override_rejects_invalid_dimension(invalid):
    op_spec, _, _ = _identity_op_spec(invalid)
    with pytest.raises(ValueError, match="invalid identity core-mapping"):
        parse_op_spec(op_spec)


def test_identity_override_rejects_unsplit_selected_dimension():
    op_spec, m, _ = _identity_op_spec(0)
    op_spec.iteration_space[m] = (512, 1)
    with pytest.raises(ValueError, match="invalid identity core-mapping"):
        parse_op_spec(op_spec)


def test_identity_override_tracks_symbol_reordering(monkeypatch):
    op_spec, m, k = _identity_op_spec()
    new_k, new_m = sympy.symbols("new_k new_m")

    def fake_align(_it_space, tensors, _indirect_sizes):
        symbol_map = {m: new_m, k: new_k}
        new_tensors = [
            {
                "size": tensor["size"],
                "coordinates": [
                    coordinate.xreplace(symbol_map)
                    for coordinate in tensor["coordinates"]
                ],
            }
            for tensor in tensors
        ]
        return (
            {new_k: (2816, 2), new_m: (512, 16)},
            new_tensors,
            {m: ((new_m, 512),), k: ((new_k, 2816),)},
        )

    monkeypatch.setattr(spyre_kernel_module, "align_tensors", fake_align)
    spyre_kernel_module.simplify_op_spec(op_spec)

    assert tuple(op_spec.iteration_space) == (new_k, new_m)
    assert op_spec.op_info[CORE_MAPPING_CONTIGUOUS_DIM_INFO_KEY] == 0
    root = _compiled_root(op_spec)
    assert root["numWkSlicesPerDim_"] == {"mb": 2, "out": 16}
    assert root["coreIdToWkSlice_"] == {
        str(core): {"mb": core % 2, "out": core // 2} for core in range(32)
    }


def test_identity_override_rejects_symbol_decomposition(monkeypatch):
    op_spec, m, k = _identity_op_spec()
    k_outer, k_inner = sympy.symbols("k_outer k_inner")

    monkeypatch.setattr(
        spyre_kernel_module,
        "align_tensors",
        lambda _it_space, tensors, _indirect_sizes: (
            {m: (512, 16), k_outer: (44, 1), k_inner: (64, 2)},
            tensors,
            {m: ((m, 512),), k: ((k_outer, 44), (k_inner, 64))},
        ),
    )
    with pytest.raises(
        spyre_kernel_module.Unsupported,
        match="must survive normalization one-to-one",
    ):
        spyre_kernel_module.simplify_op_spec(op_spec)
