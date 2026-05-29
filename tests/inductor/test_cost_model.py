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

"""Unit tests for the work-division cost model.

Each cost function is pure arithmetic over a candidate core split, so these
tests exercise the functions directly. The matmul tests enumerate every
feasible (b, m, n, k) split the planner would and assert the argmin matches the
split measured to be fastest on hardware; the pointwise and reduction tests
check the individual cost terms (roofline, broadcast cohort, stick
fragmentation, PSUM ring) behave as the AIU model intends.
"""

import unittest

from sympy import divisors

from torch_spyre._inductor.work_division import (
    _COST_PEAK_ELEMENTS_US_CORE,
    _DTYPE_BYTES,
    _HBM_BW_GBS,
    _SIMPLE_REDUCE_TYPES,
    _matmul_split_cost,
    _pointwise_split_cost,
    _reduction_split_cost,
)

# fp16 stick = 64 elements; N and K only split on stick boundaries.
_ELEMS_PER_STICK = 64


def _best_split(B, M, K, N, max_cores=32):
    best = None
    best_cost = float("inf")
    for b in (int(d) for d in divisors(B)):
        for m in (int(d) for d in divisors(M)):
            for n in (int(d) for d in divisors(N // _ELEMS_PER_STICK)):
                for k in (int(d) for d in divisors(K // _ELEMS_PER_STICK)):
                    if b * m * n * k > max_cores:
                        continue
                    cost = _matmul_split_cost(B, M, K, N, b, m, n, k, max_cores)
                    if cost < best_cost:
                        best_cost = cost
                        best = (b, m, n, k)
    return best


class TestMatmulCostModel(unittest.TestCase):
    def test_decode_mlp_prefers_mn_cosplit(self):
        # [1, 512, 4096, 4096]: M is too short to fill all 32 cores on its own,
        # so the model co-splits M and N rather than over-splitting M.
        self.assertEqual(_best_split(1, 512, 4096, 4096), (1, 8, 4, 1))

    def test_moe_bmm_tiles_batch_and_cosplits(self):
        # [8, 128, 2048, 8192] MoE expert FFN: the b ** 1.4 penalty makes
        # core-splitting the batch lose to tiling it in time (b=1) while M and
        # N take the cores.
        self.assertEqual(_best_split(8, 128, 2048, 8192), (1, 4, 8, 1))

    def test_prefill_does_not_split_k(self):
        # Large M already saturates the PT array, so paying ring-reduction hops
        # for a K-split is never worth it.
        self.assertEqual(_best_split(1, 4096, 4096, 4096)[3], 1)

    def test_infeasible_split_is_inf(self):
        self.assertEqual(
            _matmul_split_cost(1, 512, 4096, 4096, 1, 8, 8, 1, max_cores=32),
            float("inf"),
        )


class TestPointwiseCostModel(unittest.TestCase):
    def test_cost_is_roofline_max_of_compute_and_hbm(self):
        # One partitioned input (fanout 1), single core, no stick/batch split:
        # cost is the larger of HBM transfer and per-core compute, nothing else.
        in_size = out_size = 1 << 16
        cost = _pointwise_split_cost(
            [in_size], [1], out_size, cores_used=1, stick_split=1, batch_split=1
        )
        hbm = (in_size + out_size) * _DTYPE_BYTES / (_HBM_BW_GBS * 1000)
        compute = out_size / _COST_PEAK_ELEMENTS_US_CORE
        self.assertEqual(cost, max(compute, hbm))

    def test_broadcast_input_pays_cohort_tax(self):
        # A broadcast input (fanout 4) is charged size * 4 bytes; a partitioned
        # input of the same size (fanout 1) is charged once. With a small
        # output the op is HBM-bound, so the broadcast variant costs strictly
        # more through the HBM term.
        in_size, out_size = 1 << 20, 1 << 10
        broadcast = _pointwise_split_cost(
            [in_size], [4], out_size, cores_used=32, stick_split=1, batch_split=1
        )
        partitioned = _pointwise_split_cost(
            [in_size], [1], out_size, cores_used=32, stick_split=1, batch_split=1
        )
        self.assertGreater(broadcast, partitioned)

    def test_stick_split_pays_fragmentation_penalty(self):
        # Same core count and bytes, but splitting the stick dim adds the
        # per-byte fragmentation tax on top of the (HBM-bound) roofline.
        size = 1 << 20
        no_stick = _pointwise_split_cost(
            [size], [1], size, cores_used=4, stick_split=1, batch_split=1
        )
        stick = _pointwise_split_cost(
            [size], [1], size, cores_used=4, stick_split=4, batch_split=1
        )
        self.assertGreater(stick, no_stick)


class TestReductionCostModel(unittest.TestCase):
    def test_no_cohort_term(self):
        # A pure reduction has no broadcast. Once the per-core slice is small
        # enough that the kernel is HBM-bound, the cost is exactly the HBM
        # transfer of the (split-independent) total bytes, with no r-split. A
        # cohort term would scale this up with the core count; here it must not.
        elems_in, elems_out = 1 << 22, 1 << 16
        hbm = (elems_in + elems_out) * _DTYPE_BYTES / (_HBM_BW_GBS * 1000)
        c16 = _reduction_split_cost(elems_in, elems_out, 16, 1, max_cores=32)
        c32 = _reduction_split_cost(elems_in, elems_out, 32, 1, max_cores=32)
        self.assertEqual(c16, hbm)
        self.assertEqual(c32, hbm)

    def test_psum_grows_with_reduction_split(self):
        # At a fixed core count the roofline is identical, so shifting cores
        # from the output dim onto the reduction dim only adds PSUM ring hops:
        # the (1, 4) split must cost more than the (4, 1) split.
        elems_in, elems_out = 1 << 22, 1 << 16
        no_r_split = _reduction_split_cost(elems_in, elems_out, 4, 1, max_cores=32)
        r_split = _reduction_split_cost(elems_in, elems_out, 1, 4, max_cores=32)
        self.assertGreater(r_split, no_r_split)

    def test_simple_reduce_types_excludes_batchmatmul(self):
        # Matmul is a Reduction op but is routed to the matmul planner, so the
        # reduction planner's gate must not claim it.
        self.assertNotIn("batchmatmul", _SIMPLE_REDUCE_TYPES)
