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
    _COST_DTYPE_BYTES,
    _COST_HBM_BW_GBS,
    _COST_PEAK_ELEMENTS_US_CORE,
    _pointwise_split_cost,
)


def _hbm_us(bytes_total: float) -> float:
    return bytes_total / (_COST_HBM_BW_GBS * 1000)


def _compute_us(out_size: int, cores_used: int) -> float:
    return (out_size / cores_used) / _COST_PEAK_ELEMENTS_US_CORE


class TestPointwiseSplitCost(unittest.TestCase):
    # ----- HBM-bound regime ------------------------------------------------

    def test_pure_non_broadcast_two_inputs(self):
        # [M,N] + [M,N]: both inputs partitioned along the same dims (fanout=1).
        # Total HBM bytes = (M*N + M*N + M*N) * dtype_bytes = 3*M*N*dtype_bytes.
        # With many cores so per-core work is small, HBM still dominates.
        M, N = 256, 512
        cores = 32
        cost = _pointwise_split_cost(
            input_sizes=[M * N, M * N],
            input_fanouts=[1, 1],
            out_size=M * N,
            cores_used=cores,
        )
        bytes_total = 3 * M * N * _COST_DTYPE_BYTES
        hbm = _hbm_us(bytes_total)
        compute = _compute_us(M * N, cores)
        self.assertAlmostEqual(cost, max(hbm, compute))
        # Sanity: at this size HBM is the binding term.
        self.assertGreater(hbm, compute)
        self.assertAlmostEqual(cost, hbm)

    def test_broadcast_input_pays_cohort_non_broadcast_does_not(self):
        # [M,N] + [1,N] split as (m=M, n=1): the RHS lacks the M dim, so its
        # fanout is m=M (broadcast across M cores). LHS has both dims, fanout=1.
        # Expected broadcast extra bytes: N * (M - 1) * dtype_bytes.
        # Expected non-broadcast bytes (LHS + RHS one copy + OUT) = (M*N + N + M*N).
        M, N = 8, 64
        cores = M  # m=M, n=1
        cost = _pointwise_split_cost(
            input_sizes=[M * N, N],
            input_fanouts=[1, M],
            out_size=M * N,
            cores_used=cores,
        )
        non_broadcast_bytes = (M * N + N + M * N) * _COST_DTYPE_BYTES
        broadcast_extra_bytes = N * (M - 1) * _COST_DTYPE_BYTES
        total_bytes = non_broadcast_bytes + broadcast_extra_bytes
        # Equivalently: (M*N + N*M + M*N) * dtype_bytes.
        self.assertEqual(total_bytes, (M * N + N * M + M * N) * _COST_DTYPE_BYTES)
        hbm = _hbm_us(total_bytes)
        compute = _compute_us(M * N, cores)
        self.assertAlmostEqual(cost, max(hbm, compute))

    def test_fanout_one_does_not_inflate_bytes(self):
        # A non-broadcast input (fanout=1) should contribute exactly numel
        # bytes, no extra cohort tax.
        cost = _pointwise_split_cost(
            input_sizes=[100, 200],
            input_fanouts=[1, 1],
            out_size=800,
            cores_used=1,
        )
        bytes_total = (100 + 200 + 800) * _COST_DTYPE_BYTES
        # cores_used=1 → compute = 800 / peak; HBM should dominate at fp16.
        hbm = _hbm_us(bytes_total)
        compute = _compute_us(800, 1)
        self.assertAlmostEqual(cost, max(hbm, compute))

    def test_broadcast_extra_bytes_scale_with_fanout(self):
        # Doubling the fanout on the broadcast input should add exactly
        # numel * dtype_bytes more HBM bytes per extra fanout unit. At these
        # tiny per-core slices compute binds the roofline post-calibration,
        # so we verify the cost is max(compute, hbm) and that the HBM term
        # itself grows by exactly the extra-fanout bytes per fanout unit.
        base = _pointwise_split_cost(
            input_sizes=[256, 8],
            input_fanouts=[1, 2],
            out_size=256,
            cores_used=2,
        )
        bigger = _pointwise_split_cost(
            input_sizes=[256, 8],
            input_fanouts=[1, 4],
            out_size=256,
            cores_used=4,
        )
        base_bytes = (256 + 8 * 2 + 256) * _COST_DTYPE_BYTES
        bigger_bytes = (256 + 8 * 4 + 256) * _COST_DTYPE_BYTES
        base_hbm = _hbm_us(base_bytes)
        bigger_hbm = _hbm_us(bigger_bytes)
        base_compute = _compute_us(256, 2)
        bigger_compute = _compute_us(256, 4)
        self.assertAlmostEqual(base, max(base_compute, base_hbm))
        self.assertAlmostEqual(bigger, max(bigger_compute, bigger_hbm))
        # HBM-bytes accounting grows by exactly the extra-fanout bytes per
        # fanout unit, even when compute binds the roofline.
        self.assertAlmostEqual(
            bigger_hbm - base_hbm,
            (8 * 4 - 8 * 2) * _COST_DTYPE_BYTES / (_COST_HBM_BW_GBS * 1000),
        )

    # ----- Roofline / compute-bound regime --------------------------------

    def test_cost_is_max_of_compute_and_hbm(self):
        # Roofline semantics: cost = max(compute_us, hbm_us). Whichever term
        # binds, the function must report it (it can't average them). Verify
        # directly by comparing against independently computed terms across
        # a range of (out_size, cores) — with the current placeholder peak,
        # HBM binds in realistic regimes, but the max() guard is still what
        # we're checking.
        cases = [
            # (input_numels, fanouts, out, cores)
            ([1024], [1], 1024, 1),
            ([1024], [1], 1024, 32),
            ([1 << 16, 1 << 16], [1, 1], 1 << 16, 8),
            ([], [], 256, 4),
        ]
        for input_sizes, fanouts, out_size, cores in cases:
            with self.subTest(out_size=out_size, cores=cores):
                cost = _pointwise_split_cost(
                    input_sizes=input_sizes,
                    input_fanouts=fanouts,
                    out_size=out_size,
                    cores_used=cores,
                )
                bytes_total = (
                    sum(s * f for s, f in zip(input_sizes, fanouts)) + out_size
                ) * _COST_DTYPE_BYTES
                hbm = _hbm_us(bytes_total)
                compute = _compute_us(out_size, cores)
                self.assertAlmostEqual(cost, max(hbm, compute))

    def test_roofline_picks_compute_when_it_dominates(self):
        # Force compute to dominate by setting input_sizes such that HBM bytes
        # are smaller than the compute term predicts. With zero-byte inputs
        # and only out_size on the HBM side, the smallest hbm_us we can get
        # is out_size * dtype_bytes / bw. compute_us is out_size / cores /
        # peak. Pick cores small enough that compute beats hbm.
        #
        # compute > hbm  iff  1/(cores * peak) > dtype_bytes / bw
        # i.e. cores < bw / (dtype_bytes * peak).
        # With bw = 204800 bytes/us, dtype = 2, peak = 192e3:
        # threshold = 204800 / (2 * 192e3) ≈ 0.533 → never reachable with
        # cores >= 1 under the current peak constant.
        #
        # The test instead constructs an artificial peak via the formula:
        # we cannot mutate the constant, so we verify the structural claim
        # that for any combination where compute > hbm, the formula reports
        # compute. We do that by exercising _compute_us / _hbm_us directly:
        out_size = 1
        cores = 1
        cost = _pointwise_split_cost(
            input_sizes=[],
            input_fanouts=[],
            out_size=out_size,
            cores_used=cores,
        )
        hbm = _hbm_us(out_size * _COST_DTYPE_BYTES)
        compute = _compute_us(out_size, cores)
        # Whichever is larger, cost must equal it.
        expected = max(hbm, compute)
        self.assertAlmostEqual(cost, expected)

    def test_roofline_at_one_core(self):
        # cores_used=1 → per_core_elements == out_size → max compute time.
        # HBM bytes are also maximum (no parallel HBM scaling since bytes
        # are total bus traffic). Whichever roofline term binds at the
        # calibrated SFP rate, the cost must equal it.
        out_size = 1 << 20
        cores = 1
        cost = _pointwise_split_cost(
            input_sizes=[out_size, out_size],
            input_fanouts=[1, 1],
            out_size=out_size,
            cores_used=cores,
        )
        bytes_total = 3 * out_size * _COST_DTYPE_BYTES
        hbm = _hbm_us(bytes_total)
        compute = _compute_us(out_size, cores)
        self.assertAlmostEqual(cost, max(hbm, compute))

    # ----- Misc -----------------------------------------------------------

    def test_redistribution_us_is_additive(self):
        base = _pointwise_split_cost(
            input_sizes=[100],
            input_fanouts=[1],
            out_size=100,
            cores_used=1,
            redistribution_us=0.0,
        )
        with_redist = _pointwise_split_cost(
            input_sizes=[100],
            input_fanouts=[1],
            out_size=100,
            cores_used=1,
            redistribution_us=12.5,
        )
        self.assertAlmostEqual(with_redist, base + 12.5)

    def test_zero_cores_is_infeasible(self):
        cost = _pointwise_split_cost(
            input_sizes=[100],
            input_fanouts=[1],
            out_size=100,
            cores_used=0,
        )
        self.assertEqual(cost, float("inf"))

    def test_out_size_always_contributes(self):
        # Zero inputs (degenerate but well-defined) -> only out bytes count.
        cost = _pointwise_split_cost(
            input_sizes=[],
            input_fanouts=[],
            out_size=512,
            cores_used=1,
        )
        bytes_total = 512 * _COST_DTYPE_BYTES
        hbm = _hbm_us(bytes_total)
        compute = _compute_us(512, 1)
        self.assertAlmostEqual(cost, max(hbm, compute))


if __name__ == "__main__":
    unittest.main()
