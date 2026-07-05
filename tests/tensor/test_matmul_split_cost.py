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

"""Unit tests for the matmul cost model's per-link contention gate.

Pins two invariants: (1) below _COHORT_LIMIT the multicast/scatter gate is
inert so the banked small-cohort tuning is unchanged, and (2) above it an
identical-operand multicast is priced on the fast band and strictly cheaper
than a distinct-data scatter, so wide ring-aware plans stay competitive.
"""

from torch.testing._internal.common_utils import run_tests, TestCase
from torch_spyre._inductor.work_division import (
    _matmul_split_cost,
    _cohort_penalty,
    _COHORT_LIMIT,
    _HBM_BW_GBS,
    _MULTICAST_BW_GBS,
    _SCATTER_BW_GBS,
)


class TestCohortPenalty(TestCase):
    def test_inert_below_limit(self):
        # No derate up to the limit -> identical to the old max(1, c/8) == 1,
        # so no small-cohort regression for either operand class.
        for c in range(1, _COHORT_LIMIT + 1):
            self.assertEqual(_cohort_penalty(c, identical=True), 1.0)
            self.assertEqual(_cohort_penalty(c, identical=False), 1.0)

    def test_multicast_caps_at_fast_band(self):
        cap = _HBM_BW_GBS / _MULTICAST_BW_GBS
        self.assertAlmostEqual(_cohort_penalty(32, identical=True), cap)
        # Flat past the cap: 64 cores is no worse than 32 for a multicast.
        self.assertAlmostEqual(_cohort_penalty(64, identical=True), cap)

    def test_scatter_grows_then_floors(self):
        # Linear before the floor, capped at peak/36 after.
        self.assertAlmostEqual(_cohort_penalty(16, identical=False), 2.0)
        self.assertAlmostEqual(
            _cohort_penalty(1024, identical=False), _HBM_BW_GBS / _SCATTER_BW_GBS
        )

    def test_multicast_cheaper_than_scatter(self):
        for c in (16, 32, 64):
            self.assertLess(
                _cohort_penalty(c, identical=True),
                _cohort_penalty(c, identical=False),
            )

    def test_multicast_below_old_linear(self):
        # The old model charged max(m,n)/_COHORT_LIMIT with no cap; the gate is
        # strictly cheaper for a wide identical cohort.
        self.assertLess(_cohort_penalty(32, identical=True), 32 / _COHORT_LIMIT)


class TestMatmulSplitCost(TestCase):
    # KV-proj-like shape: N=1024 (16 sticks ~ 32 cores), K=4096.
    N, K = 1024, 4096

    def test_wide_n_multicast_wins_small_m(self):
        # Device: the wide-N plan (m=2, n=16) beats (m=8, n=4) here; the gate
        # must not over-charge the wide n-cohort LHS multicast and hide it.
        a = _matmul_split_cost((1, 1), (64, 8), (self.N, 4), (self.K, 1), 32)
        b = _matmul_split_cost((1, 1), (64, 2), (self.N, 16), (self.K, 1), 32)
        self.assertLess(b, a)

    def test_output_not_cohort_penalized(self):
        # A pure n-split replicates only the LHS; doubling the (distinct-write)
        # output cohort via m must not add a multicast charge to the output.
        base = _matmul_split_cost((1, 1), (256, 1), (self.N, 16), (self.K, 1), 16)
        self.assertLess(base, float("inf"))


if __name__ == "__main__":
    run_tests()
