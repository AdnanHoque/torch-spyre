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

"""Hardware-free unit tests for the k_fast planner override.

Exercises core_division._try_k_fast_split() directly with synthetic
iteration spaces and a stub output TensorDep. No torch.compile, no
Spyre device required.
"""

import unittest
from types import SimpleNamespace

import sympy
from sympy import Integer

from torch_spyre._inductor import config
from torch_spyre._inductor.core_division import _try_k_fast_split


def _stub_output_td(m_sym, n_sym, elems_per_stick=64):
    """Build a minimal output_td whose device_coords reference only M and N
    (matmul output is (M, N)). The reduction dim is whatever else is in the
    iteration space — typically K.

    elems_per_stick=64 is the fp16 default.
    """
    ns = SimpleNamespace()
    ns.device_coords = [m_sym, n_sym]  # output dims; K not in here
    ns.layout = SimpleNamespace(
        device_layout=SimpleNamespace(
            device_dtype=SimpleNamespace(
                elems_per_stick=lambda: elems_per_stick,
            )
        )
    )
    return ns


def _it_space(M, N, K):
    """Build an iteration_space dict with three matmul-style symbols
    in canonical [M, N, K] order."""
    m_sym = sympy.Symbol("m")
    n_sym = sympy.Symbol("n")
    k_sym = sympy.Symbol("k")
    return (
        {m_sym: Integer(M), n_sym: Integer(N), k_sym: Integer(K)},
        m_sym,
        n_sym,
        k_sym,
    )


def _pure_m_split(m_sym, n_sym, k_sym, ncores=32):
    return {m_sym: ncores, n_sym: 1, k_sym: 1}


class TestKFastPlannerOverride(unittest.TestCase):
    def setUp(self):
        self._saved_flag = config.core_id_k_fast_emission
        config.core_id_k_fast_emission = True

    def tearDown(self):
        config.core_id_k_fast_emission = self._saved_flag

    # ------- positive cases (heuristic should fire) --------------------

    def test_l3_70b_kv_proj_M128(self):
        """L3-70B kv_proj at M=128: N=1024 (16 sticks), K=8192 (128 sticks).
        Should propose (1, 16, 2)."""
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[m], 1)
        self.assertEqual(result[n], 16)
        self.assertEqual(result[k], 2)

    def test_dsv3_o_proj_M128(self):
        """DSv3 o_proj at M=128: N=7168 (112 sticks), K=16384 (256 sticks).
        n_sticks=112 ≥ 32 → pure-N IS valid. Heuristic should NOT fire even
        though we measured this combo as a big win — pure-N is the right
        target for a different planner change. Heuristic returns None here."""
        it_space, m, n, k = _it_space(128, 7168, 16384)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNone(result)

    def test_phi3_medium_kv_proj_M128(self):
        """Phi-3 medium kv_proj: N=1280 (20 sticks), K=5120 (80 sticks).
        n_sticks=20 doesn't divide 32 cleanly for n=16 (20%16!=0); should
        pick n=4, k=8."""
        it_space, m, n, k = _it_space(128, 1280, 5120)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[m], 1)
        self.assertEqual(result[n], 4)
        self.assertEqual(result[k], 8)

    def test_qa_proj_M128(self):
        """DSv3 q_a_proj: N=1536 (24 sticks), K=7168 (112 sticks).
        24 % 16 != 0 → fall through to n=8, k=4."""
        it_space, m, n, k = _it_space(128, 1536, 7168)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNotNone(result)
        self.assertEqual(result[n], 8)
        self.assertEqual(result[k], 4)

    # ------- negative cases (heuristic must skip) ----------------------

    def test_M_too_large(self):
        """M=2048 is outside the empirical band; planner is right."""
        it_space, m, n, k = _it_space(2048, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNone(result)

    def test_M_too_small(self):
        """M=8 is below the empirical band; per-core compute too small."""
        it_space, m, n, k = _it_space(8, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNone(result)

    def test_N_too_wide(self):
        """N=4096 (64 sticks) ≥ 32 → pure-N already valid; heuristic falls
        through to default planner."""
        it_space, m, n, k = _it_space(128, 4096, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNone(result)

    def test_K_too_narrow(self):
        """K=1024 (16 sticks) < 32 → PSUM cost dominates, no benefit."""
        it_space, m, n, k = _it_space(128, 1024, 1024)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNone(result)

    def test_planner_already_picked_k_split(self):
        """If the planner already split K, don't override — respect its choice."""
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        existing = {m: 4, n: 1, k: 8}  # planner picked (4, 1, 8)
        result = _try_k_fast_split(existing, it_space, out_td, None, max_cores=32)
        self.assertIsNone(result)

    def test_min_splits_constraint_on_k(self):
        """If hardware-driven min_splits constrains K, don't override."""
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, {k: 4}, max_cores=32
        )
        self.assertIsNone(result)

    def test_min_splits_constraint_on_m(self):
        """If hardware-driven min_splits constrains M, don't override."""
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, {m: 4}, max_cores=32
        )
        self.assertIsNone(result)

    def test_max_cores_not_32(self):
        """Empirical band was characterized at SENCORES=32; don't fire elsewhere."""
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k, ncores=16), it_space, out_td, None, max_cores=16
        )
        self.assertIsNone(result)

    def test_iteration_space_not_3d(self):
        """Pointwise / 4D bmm / etc. — heuristic is matmul-(mm)-only for now."""
        a, b, c, d = (sympy.Symbol(s) for s in "abcd")
        it_space = {a: Integer(2), b: Integer(128), c: Integer(1024), d: Integer(8192)}
        out_td = SimpleNamespace(
            device_coords=[a, b, c],
            layout=SimpleNamespace(
                device_layout=SimpleNamespace(
                    device_dtype=SimpleNamespace(elems_per_stick=lambda: 64)
                )
            ),
        )
        result = _try_k_fast_split(
            {a: 1, b: 32, c: 1, d: 1}, it_space, out_td, None, max_cores=32
        )
        self.assertIsNone(result)

    def test_flag_off_returns_none(self):
        """Disabling the feature flag short-circuits the override."""
        config.core_id_k_fast_emission = False
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNone(result)

    # ------- validity properties --------------------------------------

    def test_returned_split_product_equals_max_cores(self):
        """For any returned split, n × k must equal max_cores (SENCORES=32)."""
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertIsNotNone(result)
        product = result[m] * result[n] * result[k]
        self.assertEqual(product, 32)

    def test_largest_n_chosen_minimizes_k(self):
        """The heuristic should pick the largest valid n so chain length k
        is smallest. For N=1024 (16 sticks), we want n=16 (k=2), not
        n=8 (k=4) or n=4 (k=8)."""
        it_space, m, n, k = _it_space(128, 1024, 8192)
        out_td = _stub_output_td(m, n)
        result = _try_k_fast_split(
            _pure_m_split(m, n, k), it_space, out_td, None, max_cores=32
        )
        self.assertEqual(result[n], 16)
        self.assertEqual(result[k], 2)


if __name__ == "__main__":
    unittest.main()
