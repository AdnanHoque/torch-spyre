# Copyright 2025 The Torch-Spyre Authors.
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

"""Unit tests for the reduction cost-model planner's cost function.

These tests cover the pure-Python scoring function in isolation: no
device, no torch.compile, no Spyre runtime. They lock down the shape
of the cost surface so the planner's choices stay predictable while
the placeholder throughput constant is calibrated.
"""

import math

from torch_spyre._inductor.work_division import (
    _COST_COHORT_LIMIT,
    _COST_DTYPE_BYTES,
    _COST_HBM_BW_GBS,
    _COST_PSUM_PER_ELEM_US,
    _COST_REDUCE_ELEM_PER_US_CORE,
    _SIMPLE_REDUCE_TYPES,
    _reduction_split_cost,
)
from torch_spyre._inductor.constants import BATCH_MATMUL_OP, TOPK_OPS


def _split_compute_us(elems_in: int, cores: int) -> float:
    return (elems_in / cores) / _COST_REDUCE_ELEM_PER_US_CORE


def test_compute_us_scales_inverse_with_cores() -> None:
    """compute_us must scale exactly as 1 / cores for a fixed elems_in."""
    elems_in = 1 << 20
    elems_out = 1  # keep HBM constant; only compute differs across runs

    # Two pure d-splits with the same cohort (so cohort_pen and hbm are equal).
    cost_1 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[1], r_splits=[], max_cores=32
    )
    cost_2 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[], max_cores=32
    )
    cost_4 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[4], r_splits=[], max_cores=32
    )

    # Strip the shared (hbm + psum) terms; what remains is compute_us.
    bytes_total = (elems_in + elems_out) * _COST_DTYPE_BYTES
    hbm_base = bytes_total / (_COST_HBM_BW_GBS * 1000)
    # cohort_pen = max(1, cohort / _COST_COHORT_LIMIT); cohort=1,2,4 all under
    # the limit, so cohort_pen == 1 for each.
    assert math.isclose(cost_1 - hbm_base, _split_compute_us(elems_in, 1))
    assert math.isclose(cost_2 - hbm_base, _split_compute_us(elems_in, 2))
    assert math.isclose(cost_4 - hbm_base, _split_compute_us(elems_in, 4))

    # Sanity: doubling cores halves the compute term.
    compute_1 = cost_1 - hbm_base
    compute_2 = cost_2 - hbm_base
    compute_4 = cost_4 - hbm_base
    assert math.isclose(compute_1 / 2, compute_2)
    assert math.isclose(compute_1 / 4, compute_4)


def test_psum_fires_when_r_prod_gt_1() -> None:
    """A reduction-split should add (r_prod - 1) * elems_out * PSUM cost."""
    elems_in = 1 << 16
    elems_out = 1024
    # Compare r_split=4 vs r_split=1, all else equal (same total cores).
    # d_splits=[1] in both keeps cohort and HBM identical, isolating PSUM.
    base = _reduction_split_cost(
        elems_in, elems_out, d_splits=[1], r_splits=[1], max_cores=32
    )
    with_psum = _reduction_split_cost(
        elems_in, elems_out, d_splits=[1], r_splits=[4], max_cores=32
    )
    # Difference = psum_us(4) + (compute_us(4) - compute_us(1)).
    delta_psum = (4 - 1) * elems_out * _COST_PSUM_PER_ELEM_US
    delta_compute = _split_compute_us(elems_in, 4) - _split_compute_us(elems_in, 1)
    assert math.isclose(with_psum - base, delta_psum + delta_compute)
    # And the PSUM contribution itself is strictly positive.
    assert delta_psum > 0.0


def test_psum_zero_when_no_reduction_split() -> None:
    """r_prod == 1 (or empty r_splits) must contribute zero PSUM cost."""
    elems_in = 1 << 16
    elems_out = 1024

    # With r_splits empty, math.prod([]) == 1 and (1 - 1) == 0.
    cost_no_r = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[], max_cores=32
    )
    # With r_splits=[1], r_prod == 1 and (1 - 1) == 0, same compute.
    cost_r_one = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[1], max_cores=32
    )
    assert math.isclose(cost_no_r, cost_r_one)

    # Verify the cost equals compute + hbm only (no psum, no redistribution).
    bytes_total = (elems_in + elems_out) * _COST_DTYPE_BYTES
    expected = (
        _split_compute_us(elems_in, 2)
        + bytes_total / (_COST_HBM_BW_GBS * 1000)
    )
    assert math.isclose(cost_no_r, expected)


def test_cohort_penalty_applies_on_output_dims_only() -> None:
    """cohort = max(d_splits); r_splits must not raise the cohort penalty."""
    elems_in = 1 << 18
    elems_out = 4096
    cohort_big = _COST_COHORT_LIMIT * 2  # well above the limit

    # d_split = cohort_big: cohort_pen > 1.
    cost_d_big = _reduction_split_cost(
        elems_in, elems_out,
        d_splits=[cohort_big], r_splits=[],
        max_cores=cohort_big,
    )
    # r_split = cohort_big: cohort = max([]) = 1, cohort_pen = 1.
    cost_r_big = _reduction_split_cost(
        elems_in, elems_out,
        d_splits=[], r_splits=[cohort_big],
        max_cores=cohort_big,
    )

    bytes_total = (elems_in + elems_out) * _COST_DTYPE_BYTES
    hbm_unpenalised = bytes_total / (_COST_HBM_BW_GBS * 1000)
    hbm_penalised = hbm_unpenalised * (cohort_big / _COST_COHORT_LIMIT)

    # Strip the non-HBM terms from each and check the HBM component.
    compute_us_big_cores = _split_compute_us(elems_in, cohort_big)
    psum_r_big = (cohort_big - 1) * elems_out * _COST_PSUM_PER_ELEM_US

    assert math.isclose(cost_d_big - compute_us_big_cores, hbm_penalised)
    assert math.isclose(
        cost_r_big - compute_us_big_cores - psum_r_big, hbm_unpenalised
    )

    # Sanity: penalised HBM is strictly larger.
    assert hbm_penalised > hbm_unpenalised


def test_simple_reduce_types_disjoint_from_matmul_and_topk() -> None:
    """The sibling planner's reduction set must not overlap with matmul/topk."""
    assert BATCH_MATMUL_OP not in _SIMPLE_REDUCE_TYPES
    assert TOPK_OPS.isdisjoint(_SIMPLE_REDUCE_TYPES)
    # And the unimplemented-on-Spyre reductions stay excluded.
    for unsupported in ("welford_reduce", "welford_combine", "any", "prod",
                        "xor_sum", "argmax"):
        assert unsupported not in _SIMPLE_REDUCE_TYPES


def test_infeasible_returns_inf() -> None:
    """cores == 0 or cores > max_cores must yield inf so the planner skips it."""
    inf_zero = _reduction_split_cost(
        1024, 16, d_splits=[0], r_splits=[1], max_cores=32
    )
    inf_over = _reduction_split_cost(
        1024, 16, d_splits=[33], r_splits=[1], max_cores=32
    )
    assert inf_zero == float("inf")
    assert inf_over == float("inf")
