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
of the cost surface so the planner's choices stay predictable across
calibration cycles.
"""

import math

from torch_spyre._inductor.work_division import (
    _COST_DTYPE_BYTES,
    _COST_HBM_BW_GBS,
    _COST_PSUM_PER_ELEM_US,
    _COST_REDUCE_ELEM_PER_US_CORE,
    _SIMPLE_REDUCE_TYPES,
    _reduction_split_cost,
)
from torch_spyre._inductor.constants import BATCH_MATMUL_OP, TOPK_OPS


def _compute_us(elems_in: int, cores: int) -> float:
    return (elems_in / cores) / _COST_REDUCE_ELEM_PER_US_CORE


def _hbm_us(elems_in: int, elems_out: int) -> float:
    bytes_total = (elems_in + elems_out) * _COST_DTYPE_BYTES
    return bytes_total / (_COST_HBM_BW_GBS * 1000)


def test_cost_is_roofline_plus_psum() -> None:
    """cost == max(compute, hbm) + psum (+ redistribution).

    Roofline: a single kernel either spends its time on math or on HBM,
    whichever dominates. PSUM is sequenced after the local compute and
    so adds on top.
    """
    elems_in = 1 << 16
    elems_out = 1024
    cost = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[4], max_cores=32
    )
    expected = (
        max(_compute_us(elems_in, 8), _hbm_us(elems_in, elems_out))
        + (4 - 1) * elems_out * _COST_PSUM_PER_ELEM_US
    )
    assert math.isclose(cost, expected)


def test_compute_us_scales_inverse_with_cores_when_compute_bound() -> None:
    """Compute term scales as 1/cores; HBM term is invariant to core count.

    Use a small elems_in / large elems_out so the kernel is compute-bound
    across all the splits under test, exposing the compute scaling.
    """
    # Pick sizes so compute_us(cores=8) > hbm_us. With K = 1.2e4 elem/us/core
    # and HBM 204.8 GB/s, hbm_us scales as elems_in / 102400 (fp16) while
    # compute_us scales as elems_in / (cores * 1.2e4). At cores=8 the ratio
    # is 1.2e4 * 8 / 102400 = 0.94, so we keep elems_in large and elems_out
    # tiny to bias the kernel into the compute-bound regime.
    elems_in = 1 << 24
    elems_out = 1  # tiny output keeps HBM small and constant.
    hbm = _hbm_us(elems_in, elems_out)

    for cores in (1, 2, 4, 8):
        cost = _reduction_split_cost(
            elems_in, elems_out, d_splits=[cores], r_splits=[], max_cores=32
        )
        compute = _compute_us(elems_in, cores)
        # Guard the test's premise: we want the compute-bound regime.
        assert compute > hbm
        assert math.isclose(cost, compute)  # roofline picks compute; psum = 0.

    # Doubling cores must halve the cost in the compute-bound regime.
    c1 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[1], r_splits=[], max_cores=32
    )
    c2 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[], max_cores=32
    )
    c4 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[4], r_splits=[], max_cores=32
    )
    assert math.isclose(c1 / 2, c2)
    assert math.isclose(c1 / 4, c4)


def test_hbm_us_invariant_to_cores_when_hbm_bound() -> None:
    """In the HBM-bound regime, the cost floor does not drop with cores.

    Roofline says when hbm > compute, adding cores stops helping the
    compute term and the cost is pinned at hbm_us. We construct that
    regime by sizing elems_in so compute_us(low_cores) < hbm_us.
    """
    # HBM 204.8 GB/s = 102.4e3 elem/us (fp16, 2 B/elem). Compute K = 1.2e4
    # elem/us/core. The crossover sits at cores ~= 102.4/12 ~= 8.5, so we
    # compare 16 vs 32 cores: both are above the crossover and stay pinned
    # at the HBM floor.
    elems_in = 1 << 24
    elems_out = 1 << 12
    hbm = _hbm_us(elems_in, elems_out)
    # Premise: at 16 cores already compute < HBM.
    assert _compute_us(elems_in, 16) < hbm

    cost_16 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[16], r_splits=[], max_cores=32
    )
    cost_32 = _reduction_split_cost(
        elems_in, elems_out, d_splits=[32], r_splits=[], max_cores=32
    )
    # Both pinned at HBM floor; no cohort penalty distinguishes them.
    assert math.isclose(cost_16, hbm)
    assert math.isclose(cost_32, hbm)


def test_no_cohort_penalty_on_output_dim_splits() -> None:
    """Pure reductions have no broadcast; splitting an output dim across
    many cores must not raise the HBM cost.

    Regression: the prior cost model inherited a cohort_penalty from the
    matmul planner that multiplied hbm_us by max(d_splits)/_COST_COHORT_LIMIT
    above the limit. For reductions every core reads its own slice of the
    input and writes its own slice of the output, so there is no
    contention term to charge.

    Compare two splits at the same core count (no compute or HBM
    difference) where one would have triggered the old cohort penalty
    on the output dim and the other would not.
    """
    elems_in = 1 << 16
    elems_out = 1024
    # cohort_big (32) is well above the matmul planner's _COST_COHORT_LIMIT
    # of 8; cohort_small (4) is below it. Old model: cost_big_d > cost_big_r
    # because cost_big_d's HBM was multiplied by 32/8 = 4x while cost_big_r's
    # was not. New model: only the PSUM term differs.
    cohort_big = 32

    cost_d = _reduction_split_cost(
        elems_in, elems_out,
        d_splits=[cohort_big], r_splits=[],
        max_cores=cohort_big,
    )
    cost_r = _reduction_split_cost(
        elems_in, elems_out,
        d_splits=[], r_splits=[cohort_big],
        max_cores=cohort_big,
    )

    roofline = max(_compute_us(elems_in, cohort_big), _hbm_us(elems_in, elems_out))
    psum_r = (cohort_big - 1) * elems_out * _COST_PSUM_PER_ELEM_US

    # d-split: pure roofline, no PSUM, no cohort surcharge.
    assert math.isclose(cost_d, roofline)
    # r-split: same roofline plus the ring-reduce PSUM hops.
    assert math.isclose(cost_r, roofline + psum_r)
    # Therefore d-split is strictly cheaper, the opposite of the old
    # cohort-penalised ordering. This is the regression guard.
    assert cost_d < cost_r


def test_psum_grows_linearly_with_r_prod_minus_1() -> None:
    """psum_us == (r_prod - 1) * elems_out * _COST_PSUM_PER_ELEM_US.

    Hold d_splits constant so compute and hbm match across the runs and
    only the PSUM term varies.
    """
    elems_in = 1 << 16
    elems_out = 1024

    def psum_only(r: int) -> float:
        cost = _reduction_split_cost(
            elems_in, elems_out, d_splits=[1], r_splits=[r], max_cores=32
        )
        # Strip the roofline floor so what remains is the PSUM term.
        return cost - max(_compute_us(elems_in, r), _hbm_us(elems_in, elems_out))

    p1 = psum_only(1)
    p2 = psum_only(2)
    p4 = psum_only(4)
    p8 = psum_only(8)
    per_hop = elems_out * _COST_PSUM_PER_ELEM_US
    assert math.isclose(p1, 0.0)
    assert math.isclose(p2, 1 * per_hop)
    assert math.isclose(p4, 3 * per_hop)
    assert math.isclose(p8, 7 * per_hop)
    # Linear in (r_prod - 1): differences are constant per hop.
    assert math.isclose(p4 - p2, 2 * per_hop)
    assert math.isclose(p8 - p4, 4 * per_hop)


def test_psum_zero_when_no_reduction_split() -> None:
    """r_prod == 1 (or empty r_splits) must contribute zero PSUM cost."""
    elems_in = 1 << 16
    elems_out = 1024

    cost_no_r = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[], max_cores=32
    )
    cost_r_one = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[1], max_cores=32
    )
    assert math.isclose(cost_no_r, cost_r_one)

    expected = max(_compute_us(elems_in, 2), _hbm_us(elems_in, elems_out))
    assert math.isclose(cost_no_r, expected)


def test_redistribution_adds_on_top() -> None:
    """redistribution_us must add to the total without scaling any other term."""
    elems_in = 1 << 16
    elems_out = 1024
    base = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[], max_cores=32
    )
    with_redist = _reduction_split_cost(
        elems_in, elems_out, d_splits=[2], r_splits=[], max_cores=32,
        redistribution_us=12.5,
    )
    assert math.isclose(with_redist - base, 12.5)


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
