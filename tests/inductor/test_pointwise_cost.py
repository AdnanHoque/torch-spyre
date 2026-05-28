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

"""Unit tests for _pointwise_split_cost.

Pure-Python tests: no device, no torch.compile, no Spyre runtime. Only the
cost function arithmetic is exercised.
"""

import os

# Avoid pulling in the Spyre device runtime when running these tests in
# isolation; the cost function only needs the module-level constants and
# does not touch any device state.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import unittest

from torch_spyre._inductor.work_division import (
    _COST_COHORT_LIMIT,
    _COST_DTYPE_BYTES,
    _COST_HBM_BW_GBS,
    _pointwise_split_cost,
)


def _hbm_us(bytes_total: float, cohort_penalty: float = 1.0) -> float:
    return bytes_total / (_COST_HBM_BW_GBS * 1000) * cohort_penalty


class TestPointwiseSplitCost(unittest.TestCase):
    def test_cohort_penalty_at_knee_is_one(self):
        # Single input, no broadcast, cohort exactly at limit -> penalty 1.0.
        cost = _pointwise_split_cost(
            input_sizes=[1024],
            input_fanouts=[1],
            out_size=1024,
            max_cohort=_COST_COHORT_LIMIT,
        )
        bytes_total = (1024 + 1024) * _COST_DTYPE_BYTES
        self.assertAlmostEqual(cost, _hbm_us(bytes_total, 1.0))

    def test_cohort_penalty_below_knee_clamps_to_one(self):
        # cohort < limit -> max(1.0, cohort/limit) clamps to 1.0.
        cost_below = _pointwise_split_cost(
            input_sizes=[1024],
            input_fanouts=[1],
            out_size=1024,
            max_cohort=1,
        )
        cost_at = _pointwise_split_cost(
            input_sizes=[1024],
            input_fanouts=[1],
            out_size=1024,
            max_cohort=_COST_COHORT_LIMIT,
        )
        self.assertEqual(cost_below, cost_at)

    def test_cohort_penalty_doubles_above_knee(self):
        # cohort = 2 * limit -> penalty 2.0.
        cost_1x = _pointwise_split_cost(
            input_sizes=[1024],
            input_fanouts=[1],
            out_size=1024,
            max_cohort=_COST_COHORT_LIMIT,
        )
        cost_2x = _pointwise_split_cost(
            input_sizes=[1024],
            input_fanouts=[1],
            out_size=1024,
            max_cohort=2 * _COST_COHORT_LIMIT,
        )
        self.assertAlmostEqual(cost_2x, 2.0 * cost_1x)

    def test_fanout_multiplies_broadcast_input_bytes(self):
        # One input has fanout=4 (broadcast across a 4-way split).
        cost = _pointwise_split_cost(
            input_sizes=[100, 200],
            input_fanouts=[4, 1],
            out_size=800,
            max_cohort=1,
        )
        # bytes = (100*4 + 200*1 + 800) * dtype_bytes
        bytes_total = (100 * 4 + 200 * 1 + 800) * _COST_DTYPE_BYTES
        self.assertAlmostEqual(cost, _hbm_us(bytes_total, 1.0))

    def test_fanout_one_is_a_passthrough(self):
        # fanout=1 must not change the byte count.
        cost_f1 = _pointwise_split_cost(
            input_sizes=[100, 200],
            input_fanouts=[1, 1],
            out_size=800,
            max_cohort=1,
        )
        bytes_total = (100 + 200 + 800) * _COST_DTYPE_BYTES
        self.assertAlmostEqual(cost_f1, _hbm_us(bytes_total, 1.0))

    def test_out_size_always_contributes(self):
        # Zero inputs (degenerate but well-defined) -> only out_size counts.
        cost = _pointwise_split_cost(
            input_sizes=[],
            input_fanouts=[],
            out_size=512,
            max_cohort=1,
        )
        bytes_total = 512 * _COST_DTYPE_BYTES
        self.assertAlmostEqual(cost, _hbm_us(bytes_total, 1.0))

    def test_out_size_contribution_scales_linearly(self):
        cost_small = _pointwise_split_cost(
            input_sizes=[0],
            input_fanouts=[1],
            out_size=512,
            max_cohort=1,
        )
        cost_big = _pointwise_split_cost(
            input_sizes=[0],
            input_fanouts=[1],
            out_size=1024,
            max_cohort=1,
        )
        # Out doubles -> total bytes doubles -> cost doubles.
        self.assertAlmostEqual(cost_big, 2.0 * cost_small)

    def test_redistribution_us_is_additive(self):
        base = _pointwise_split_cost(
            input_sizes=[100],
            input_fanouts=[1],
            out_size=100,
            max_cohort=1,
            redistribution_us=0.0,
        )
        with_redist = _pointwise_split_cost(
            input_sizes=[100],
            input_fanouts=[1],
            out_size=100,
            max_cohort=1,
            redistribution_us=12.5,
        )
        self.assertAlmostEqual(with_redist, base + 12.5)


if __name__ == "__main__":
    unittest.main()
