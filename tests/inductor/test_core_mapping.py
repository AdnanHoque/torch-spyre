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
from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import parse_op_spec
from torch_spyre._inductor.constants import (
    BATCH_MATMUL_FP8_OP,
    BATCH_MATMUL_OP,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.core_mapping import (
    CoreMappingPolicy,
    ResolvedCoreMapping,
    materialize_core_mapping,
    resolve_core_mapping,
    select_core_mapping_policy,
    validate_core_mapping,
)


def _materialized_coordinates(
    mapping: ResolvedCoreMapping,
    splits: tuple[int, ...],
    num_cores: int,
) -> list[tuple[int, ...]]:
    axes = sympy.symbols(f"axis_0:{len(splits)}")
    expressions = materialize_core_mapping(mapping, axes, splits, num_cores)
    core_id = sympy.Symbol("core_id")
    return [
        tuple(int(expressions[str(axis)].subs(core_id, physical_id)) for axis in axes)
        for physical_id in range(num_cores)
    ]


@pytest.mark.parametrize("splits", [(1,), (2, 1, 3), (2, 3, 4)])
def test_natural_mapping_matches_legacy_linearization(splits):
    logical_cores = math.prod(splits)
    num_cores = 2 * logical_cores
    mapping = resolve_core_mapping(splits, num_cores, policy=CoreMappingPolicy.NATURAL)

    coordinates = _materialized_coordinates(mapping, splits, num_cores)
    for physical_id, actual in enumerate(coordinates):
        expected = tuple(
            (physical_id // math.prod(splits[:axis])) % split
            for axis, split in enumerate(splits)
        )
        assert actual == expected


@pytest.mark.parametrize(
    ("policy", "expected_order"),
    [
        (CoreMappingPolicy.NATURAL, (0, 1, 2)),
        (CoreMappingPolicy.REDUCTION_FAST, (2, 0, 1)),
        (CoreMappingPolicy.ROW_MAJOR, (2, 1, 0)),
    ],
)
def test_policy_selects_exact_axis_order(policy, expected_order):
    mapping = resolve_core_mapping((2, 3, 4), 24, policy=policy, reduction_axis=2)
    assert mapping.levels == tuple(("axis", axis) for axis in expected_order)


def test_row_major_does_not_implicitly_change_replica_placement():
    mapping = resolve_core_mapping((2, 3, 4), 48, policy=CoreMappingPolicy.ROW_MAJOR)
    assert mapping.levels == (
        ("axis", 2),
        ("axis", 1),
        ("axis", 0),
        ("replica", 2),
    )
    coordinates = _materialized_coordinates(mapping, (2, 3, 4), 48)
    assert coordinates[:24] == coordinates[24:]


def test_reduction_fast_makes_partial_sum_cohorts_adjacent():
    splits = (2, 3, 4)
    mapping = resolve_core_mapping(
        splits,
        math.prod(splits),
        policy=CoreMappingPolicy.REDUCTION_FAST,
        reduction_axis=2,
    )
    coordinates = _materialized_coordinates(mapping, splits, math.prod(splits))

    for m in range(splits[0]):
        for n in range(splits[1]):
            cohort = [
                core_id
                for core_id, coordinate in enumerate(coordinates)
                if coordinate[:2] == (m, n)
            ]
            assert cohort == list(range(cohort[0], cohort[0] + splits[2]))
            assert [coordinates[core_id][2] for core_id in cohort] == list(
                range(splits[2])
            )


@pytest.mark.parametrize(
    ("splits", "enable", "expected"),
    [
        ((2, 3, 4), True, CoreMappingPolicy.REDUCTION_FAST),
        ((2, 3, 1), True, CoreMappingPolicy.NATURAL),
        ((2, 3, 4), False, CoreMappingPolicy.NATURAL),
        ((3, 4), True, CoreMappingPolicy.NATURAL),
    ],
)
def test_automatic_policy_gate(splits, enable, expected):
    assert (
        select_core_mapping_policy(
            splits,
            reduction_axis=(None if not enable or len(splits) < 3 else len(splits) - 1),
            enable_reduction_fast=enable,
        )
        == expected
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
@pytest.mark.parametrize("k_fast", [False, True])
def test_planner_and_sdsc_consumers_use_the_same_mapping(monkeypatch, op, k_fast):
    class FakeReduction:
        def __init__(self, reduction_type):
            self.reduction_type = reduction_type

    class FakeComputedBuffer:
        def __init__(self, reduction_type):
            self.data = FakeReduction(reduction_type)

    monkeypatch.setattr(pass_utils_module, "Reduction", FakeReduction)
    monkeypatch.setattr(pass_utils_module, "ComputedBuffer", FakeComputedBuffer)
    monkeypatch.setattr(pass_utils_module.config, "core_id_k_fast_emission", k_fast)
    monkeypatch.setattr(
        superdsc_module._spyre_config, "core_id_k_fast_emission", k_fast
    )

    op_spec = _bmm_op_spec(op)
    symbols = tuple(op_spec.iteration_space)
    splits = dict(zip(symbols, (2, 4, 4)))
    monkeypatch.setattr(
        pass_utils_module, "apply_splits_from_index_coeff", lambda *_: splits
    )
    prep = pass_utils_module._ViewPrep(
        iter_space=op_spec.iteration_space,
        write_index=symbols[0],
        read_index=symbols[-1],
        dep_coeff={symbols[0]: 1, symbols[1]: 2, symbols[2]: 0},
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
        device_dim: sdsc_spec.core_id_to_work_slice[str(renamed[symbol])]
        for device_dim, symbol in enumerate(symbols[:2])
    }
    assert representable
    assert dict(planner_view.core_to_slot) == sdsc_output_mapping


@pytest.mark.parametrize("splits", [(0,), (-1,), (1.5,), (True,)])
def test_rejects_invalid_axis_splits(splits):
    with pytest.raises(ValueError, match="axis splits"):
        resolve_core_mapping(splits, 1)


@pytest.mark.parametrize("reduction_axis", [None, True, 1.5, -1, 3])
def test_rejects_invalid_reduction_axis(reduction_axis):
    with pytest.raises(ValueError, match="reduction_axis"):
        resolve_core_mapping(
            (2, 2, 2),
            8,
            policy=CoreMappingPolicy.REDUCTION_FAST,
            reduction_axis=reduction_axis,
        )
    with pytest.raises(ValueError, match="reduction_axis"):
        select_core_mapping_policy(
            (2, 2, 2),
            reduction_axis=reduction_axis,
            enable_reduction_fast=True,
        )


def test_rejects_invalid_mapping_contracts():
    with pytest.raises(ValueError, match="positive multiple"):
        resolve_core_mapping((2, 2), 5)
    with pytest.raises(ValueError, match="every axis ordinal exactly once"):
        validate_core_mapping(
            ResolvedCoreMapping((("axis", 0), ("axis", 0))), (2, 2), 4
        )
    with pytest.raises(ValueError, match="level product"):
        validate_core_mapping(ResolvedCoreMapping((("axis", 0),)), (2,), 4)
    with pytest.raises(ValueError, match="invalid axis level"):
        ResolvedCoreMapping((("axis", True),))
    with pytest.raises(ValueError, match="invalid replica level"):
        ResolvedCoreMapping((("replica", True),))
