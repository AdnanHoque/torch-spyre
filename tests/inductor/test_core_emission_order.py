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

"""Unit tests for the configurable core-ID-to-slice emission order in
`torch_spyre._inductor.codegen.superdsc._get_core_to_slice_mapping`.

Tests the pure emitter at the sympy-expression level — no Spyre runtime,
no compile. For each split, evaluates the emitted expressions at every
core_id ∈ [0, num_cores) and asserts the resulting (m, n, k) tuples
match the expected mapping for both default and reversed orderings.
"""

from __future__ import annotations

import pytest
from sympy import Symbol

import torch_spyre  # noqa: F401  -- ensure backend is registered

from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.superdsc import _get_core_to_slice_mapping


M_SYM = Symbol("M")
N_SYM = Symbol("N")
K_SYM = Symbol("K")
CORE_ID = Symbol("core_id")


@pytest.fixture(autouse=True)
def _reset_config():
    saved = config.core_emission_reverse
    yield
    config.core_emission_reverse = saved


def _materialize(it_space, dim_splits, num_cores):
    """Evaluate the symbolic emitter at every core_id and return a list of
    (m, n, k) tuples, in core_id order."""
    expr_map = _get_core_to_slice_mapping(it_space, dim_splits, num_cores)
    out = []
    for c in range(num_cores):
        coords = tuple(
            int(expr_map[str(d)].subs(CORE_ID, c))
            for d in [M_SYM, N_SYM, K_SYM]
        )
        out.append(coords)
    return out


# ---- default ordering: leftmost split>1 dim is fast-changing -----------

def test_default_pure_n_split_walks_n():
    """Pure (1, 32, 1): m=1 collapses to 0, so N becomes the fast dim."""
    config.core_emission_reverse = False
    it = {M_SYM: 1, N_SYM: 32, K_SYM: 1}
    splits = {M_SYM: 1, N_SYM: 32, K_SYM: 1}
    out = _materialize(it, splits, 32)
    assert out[0] == (0, 0, 0)
    assert out[1] == (0, 1, 0)
    assert out[31] == (0, 31, 0)


def test_default_pure_m_split_walks_m():
    """Pure (32, 1, 1): N=1 collapses to 0, M is fast."""
    config.core_emission_reverse = False
    it = {M_SYM: 32, N_SYM: 1, K_SYM: 1}
    splits = {M_SYM: 32, N_SYM: 1, K_SYM: 1}
    out = _materialize(it, splits, 32)
    assert out[0] == (0, 0, 0)
    assert out[1] == (1, 0, 0)
    assert out[31] == (31, 0, 0)


def test_default_mixed_split_m_fast():
    """Mixed (m=2, n=4, k=1): M is fast — adjacent cores walk M, share N."""
    config.core_emission_reverse = False
    it = {M_SYM: 2, N_SYM: 4, K_SYM: 1}
    splits = {M_SYM: 2, N_SYM: 4, K_SYM: 1}
    out = _materialize(it, splits, 8)
    expected = [
        (0, 0, 0), (1, 0, 0),  # core 0,1 share n=0
        (0, 1, 0), (1, 1, 0),  # core 2,3 share n=1
        (0, 2, 0), (1, 2, 0),
        (0, 3, 0), (1, 3, 0),
    ]
    assert out == expected


def test_default_l3_70b_mlp_down_shape():
    """(16, 2, 1) split as used for L3-70B MLP down. Adjacent cores 0..15
    share n=0, all needing the same huge B-column slice — the case the
    reverse flag is meant to address."""
    config.core_emission_reverse = False
    it = {M_SYM: 16, N_SYM: 2, K_SYM: 1}
    splits = {M_SYM: 16, N_SYM: 2, K_SYM: 1}
    out = _materialize(it, splits, 32)
    # cores 0..15 all have n=0 and walk m=0..15
    for c in range(16):
        assert out[c] == (c, 0, 0)
    # core 16 jumps to n=1, m=0
    assert out[16] == (0, 1, 0)
    # cores 16..31 walk m=0..15 with n=1
    for c in range(16, 32):
        assert out[c] == (c - 16, 1, 0)


# ---- reversed ordering: rightmost split>1 dim is fast-changing ---------

def test_reverse_pure_n_split_unchanged():
    """Pure (1, 32, 1) — only N has split>1 so reversal is a no-op."""
    config.core_emission_reverse = True
    it = {M_SYM: 1, N_SYM: 32, K_SYM: 1}
    splits = {M_SYM: 1, N_SYM: 32, K_SYM: 1}
    out = _materialize(it, splits, 32)
    assert out[0] == (0, 0, 0)
    assert out[1] == (0, 1, 0)
    assert out[31] == (0, 31, 0)


def test_reverse_pure_m_split_unchanged():
    """Pure (32, 1, 1) — only M has split>1 so reversal is a no-op."""
    config.core_emission_reverse = True
    it = {M_SYM: 32, N_SYM: 1, K_SYM: 1}
    splits = {M_SYM: 32, N_SYM: 1, K_SYM: 1}
    out = _materialize(it, splits, 32)
    assert out[0] == (0, 0, 0)
    assert out[1] == (1, 0, 0)


def test_reverse_mixed_split_n_fast():
    """Mixed (m=2, n=4, k=1) reversed: K=1 is no-op, N (rightmost split>1)
    becomes fast — adjacent cores walk N, share M."""
    config.core_emission_reverse = True
    it = {M_SYM: 2, N_SYM: 4, K_SYM: 1}
    splits = {M_SYM: 2, N_SYM: 4, K_SYM: 1}
    out = _materialize(it, splits, 8)
    expected = [
        (0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0),  # core 0..3 share m=0
        (1, 0, 0), (1, 1, 0), (1, 2, 0), (1, 3, 0),  # core 4..7 share m=1
    ]
    assert out == expected


def test_reverse_l3_70b_mlp_down_shape():
    """(16, 2, 1) reversed. Adjacent cores 0,1 now share m=0 (different n)
    so they share a tiny A-row slice instead of a huge B-column slice."""
    config.core_emission_reverse = True
    it = {M_SYM: 16, N_SYM: 2, K_SYM: 1}
    splits = {M_SYM: 16, N_SYM: 2, K_SYM: 1}
    out = _materialize(it, splits, 32)
    # cores 0..1 share m=0 and walk n=0,1
    assert out[0] == (0, 0, 0)
    assert out[1] == (0, 1, 0)
    # core 2 advances m
    assert out[2] == (1, 0, 0)
    assert out[3] == (1, 1, 0)
    # general pattern: out[c] = (m=c//2, n=c%2)
    for c in range(32):
        assert out[c] == (c // 2, c % 2, 0)


# ---- invariants -------------------------------------------------------

@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("split", [
    (1, 32, 1), (32, 1, 1), (2, 16, 1), (16, 2, 1),
    (4, 8, 1), (8, 4, 1), (1, 1, 32),
])
def test_emitter_is_a_permutation(reverse, split):
    """For any split, the emitter must produce a valid 1-to-1 mapping —
    every (m, n, k) tuple in the cartesian product appears exactly once
    across the num_cores cores. Otherwise we'd be either skipping or
    double-assigning slices."""
    m, n, k = split
    num_cores = m * n * k
    config.core_emission_reverse = reverse
    it = {M_SYM: m, N_SYM: n, K_SYM: k}
    splits = {M_SYM: m, N_SYM: n, K_SYM: k}
    out = _materialize(it, splits, num_cores)
    expected = {(mi, ni, ki) for mi in range(m) for ni in range(n) for ki in range(k)}
    assert set(out) == expected
    assert len(out) == len(expected)
