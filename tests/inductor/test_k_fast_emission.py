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

"""Hardware-free unit tests for the k_fast core_id emission permutation."""

import unittest

import sympy

from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.compute_ops import (
    _k_fast_core_id_permutation,
)


def _make_work_slices(m: int, n: int, k: int) -> dict:
    """Build a work_slices dict for matmul iteration order [M, N, K]."""
    return {
        sympy.Symbol("M"): m,
        sympy.Symbol("N"): n,
        sympy.Symbol("K"): k,
    }


def _is_permutation(p, num_cores):
    return sorted(p) == list(range(num_cores))


def _decode_identity(core_id: int, m: int, n: int, k: int) -> tuple:
    """Decode a logical core_id under the default identity emission."""
    m_slice = core_id % m
    n_slice = (core_id // m) % n
    k_slice = (core_id // (m * n)) % k
    return (m_slice, n_slice, k_slice)


class TestKFastEmission(unittest.TestCase):
    """Verify the algebraic properties claimed for k_fast."""

    def setUp(self):
        # Tests assume the feature is enabled.
        self._saved = config.core_id_k_fast_emission
        config.core_id_k_fast_emission = True

    def tearDown(self):
        config.core_id_k_fast_emission = self._saved

    # ---- algebraic properties -----------------------------------------

    def test_identity_when_k_is_one(self):
        """k=1 means no PSUM chain; perm must degenerate to identity."""
        for m, n in [(1, 32), (32, 1), (4, 8), (16, 2)]:
            ws = _make_work_slices(m, n, 1)
            perm = _k_fast_core_id_permutation(32, ws)
            self.assertEqual(
                perm,
                list(range(32)),
                msg=f"k=1 should give identity for split ({m}, {n}, 1)",
            )

    def test_disabled_returns_identity(self):
        """When the feature flag is off, return identity regardless of k."""
        config.core_id_k_fast_emission = False
        ws = _make_work_slices(1, 16, 2)
        perm = _k_fast_core_id_permutation(32, ws)
        self.assertEqual(perm, list(range(32)))

    def test_returns_valid_permutation(self):
        """Output must always be a valid permutation of [0, num_cores)."""
        for m, n, k in [
            (1, 32, 1),
            (1, 16, 2),
            (1, 8, 4),
            (1, 4, 8),
            (1, 1, 32),
            (2, 4, 4),
            (4, 1, 8),
            (8, 4, 1),
            (16, 2, 1),
        ]:
            ws = _make_work_slices(m, n, k)
            perm = _k_fast_core_id_permutation(32, ws)
            self.assertTrue(
                _is_permutation(perm, 32),
                msg=f"split ({m}, {n}, {k}) produced non-permutation: {perm}",
            )

    def test_k_collaborators_are_adjacent(self):
        """The defining property: cores sharing (m, n) but differing in
        k must end up at consecutive physical ring positions."""
        for m, n, k in [(1, 16, 2), (1, 8, 4), (1, 4, 8), (4, 1, 8), (2, 4, 4)]:
            ws = _make_work_slices(m, n, k)
            perm = _k_fast_core_id_permutation(32, ws)
            # For each (m, n) cell, find all physical cores assigned to it
            # and check that they form a contiguous block of length k.
            cells = {}
            for phys_c, logical in enumerate(perm):
                ms, ns, ks = _decode_identity(logical, m, n, k)
                cells.setdefault((ms, ns), []).append(phys_c)
            for (ms, ns), positions in cells.items():
                positions.sort()
                self.assertEqual(
                    len(positions),
                    k,
                    msg=f"split ({m}, {n}, {k}) cell ({ms}, {ns}) has "
                    f"{len(positions)} cores; expected {k}",
                )
                expected = list(range(positions[0], positions[0] + k))
                self.assertEqual(
                    positions,
                    expected,
                    msg=f"split ({m}, {n}, {k}) cell ({ms}, {ns}) "
                    f"K-cluster at {positions} is not contiguous "
                    f"(expected {expected})",
                )

    def test_k_pair_for_kv_proj_split_is_distance_one(self):
        """Specific check for the kv_proj (1, 16, 2) shape: K-pair for
        every (m=0, n_slice) cell must sit at adjacent physical cores."""
        ws = _make_work_slices(1, 16, 2)
        perm = _k_fast_core_id_permutation(32, ws)
        # Group physical cores by their assigned n_slice, k=anything.
        by_n = {}
        for phys_c, logical in enumerate(perm):
            _, ns, ks = _decode_identity(logical, 1, 16, 2)
            by_n.setdefault(ns, []).append((phys_c, ks))
        for ns, members in by_n.items():
            members.sort(key=lambda pair: pair[1])  # sort by k_slice
            self.assertEqual(len(members), 2)
            distance = abs(members[1][0] - members[0][0])
            self.assertEqual(
                distance,
                1,
                msg=f"K-pair for n_slice={ns} sits at {members}, "
                f"distance {distance} (expected 1)",
            )

    # ---- handles missing inputs ---------------------------------------

    def test_empty_work_slices_returns_identity(self):
        """No work_slices info -> safe fallback to identity."""
        perm = _k_fast_core_id_permutation(32, None)
        self.assertEqual(perm, list(range(32)))
        perm = _k_fast_core_id_permutation(32, {})
        self.assertEqual(perm, list(range(32)))


if __name__ == "__main__":
    unittest.main()
