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

"""Unit tests for the Phase 2.0 element-priority work-division heuristic.

Tests the pure helper `_output_element_priority_should_fire` and the
sort-key swap inside `prioritize_dimensions`. Hardware-free; uses sympy
symbols and integer iteration sizes directly.

Phase 1.0 (tests/diag_split_gap_results.md) measured 15-40% wall-time gaps
between the planner's pure-M-split picks and empirical-best pure-N-split.
Root cause: ranking by stick-adjusted size deflates stick-dim N below
non-stick M. This heuristic switches to element-count ranking, which is
expected to flip those picks.
"""

from __future__ import annotations

import pytest
from sympy import Integer, Symbol

import torch_spyre  # noqa: F401  -- ensure backend is registered

from torch_spyre._inductor import config
from torch_spyre._inductor.core_division import (
    _output_element_priority_should_fire,
    prioritize_dimensions,
)


M_SYM = Symbol("M")
N_SYM = Symbol("N")
K_SYM = Symbol("K")


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset the config flag after each test so they don't leak."""
    saved = config.output_element_priority
    yield
    config.output_element_priority = saved


# ---- _output_element_priority_should_fire --------------------------------

def test_should_not_fire_when_disabled():
    config.output_element_priority = False
    assert not _output_element_priority_should_fire(
        [(M_SYM, Integer(128)), (N_SYM, Integer(64))],
    )


def test_fires_with_two_output_dims_when_enabled():
    config.output_element_priority = True
    assert _output_element_priority_should_fire(
        [(M_SYM, Integer(128)), (N_SYM, Integer(64))],
    )


def test_does_not_fire_with_one_output_dim():
    config.output_element_priority = True
    assert not _output_element_priority_should_fire(
        [(M_SYM, Integer(128))],
    )


def test_does_not_fire_with_zero_output_dims():
    config.output_element_priority = True
    assert not _output_element_priority_should_fire([])


# ---- prioritize_dimensions --------------------------------------------

class _StubOutput:
    """Stub for `output: TensorDep` — only `device_coords[:-1]` is used to
    derive coord_vars. The last entry is the stick dim and is excluded."""

    def __init__(self, *coord_exprs):
        self.device_coords = list(coord_exprs)


def _matmul_output():
    """Build a stub matmul output with shape [M, N] where N is the innermost
    stick dim. Device layout is [M, N_outer_chunks, N_stick] so device_coords
    has three entries; [:-1] gives [M, N_outer_chunks] from which coord_vars
    derives {M, N}."""
    return _StubOutput(M_SYM, N_SYM, N_SYM)


def test_default_ranking_uses_stick_adjusted_size():
    """L3-8B q_proj-like shape: M=128 elem, N=4096 elem (=64 sticks).
    Default behavior ranks M (128) > N (64) — picks M-priority."""
    config.output_element_priority = False
    output = _matmul_output()
    it_space = {
        M_SYM: Integer(128),
        N_SYM: Integer(64),
        K_SYM: Integer(64),
    }
    priorities = prioritize_dimensions(
        output, it_space, stick_vars={N_SYM: 64, K_SYM: 64},
    )
    assert priorities == [M_SYM, N_SYM, K_SYM]


def test_heuristic_swaps_to_element_count_ranking():
    """Same shape; with heuristic enabled, N (4096 elem) > M (128 elem)
    — N gets priority and the planner will pick N-split."""
    config.output_element_priority = True
    output = _matmul_output()
    it_space = {
        M_SYM: Integer(128),
        N_SYM: Integer(64),
        K_SYM: Integer(64),
    }
    priorities = prioritize_dimensions(
        output, it_space, stick_vars={N_SYM: 64, K_SYM: 64},
    )
    assert priorities == [N_SYM, M_SYM, K_SYM]


def test_heuristic_inactive_when_only_one_output_dim():
    """Reduction-style shape with single output dim. Heuristic should
    not fire — single output dim has no priority to flip."""
    config.output_element_priority = True
    output = _StubOutput(M_SYM)
    it_space = {M_SYM: Integer(128), K_SYM: Integer(64)}
    priorities = prioritize_dimensions(
        output, it_space, stick_vars={K_SYM: 64},
    )
    assert priorities == [M_SYM, K_SYM]


def test_heuristic_with_two_non_stick_output_dims():
    """When neither output is a stick var (e.g. small batched op), element
    count and stick-adjusted size are identical — order should match
    default."""
    config.output_element_priority = True
    output = _matmul_output()
    it_space = {
        M_SYM: Integer(128),
        N_SYM: Integer(64),
    }
    priorities_heuristic = prioritize_dimensions(
        output, it_space, stick_vars={},
    )
    config.output_element_priority = False
    priorities_default = prioritize_dimensions(
        output, it_space, stick_vars={},
    )
    assert priorities_heuristic == priorities_default


def test_l3_8b_mlp_gate_up_already_correct():
    """L3-8B MLP gate/up: M=128 elem, N=14336 elem (=224 sticks).
    Default: N (224 sticks) > M (128) — already prefers N. Heuristic
    should not change the ranking (still N first)."""
    config.output_element_priority = False
    output = _matmul_output()
    it_space = {
        M_SYM: Integer(128),
        N_SYM: Integer(224),
        K_SYM: Integer(64),
    }
    default = prioritize_dimensions(
        output, it_space, stick_vars={N_SYM: 64, K_SYM: 64},
    )
    config.output_element_priority = True
    heuristic = prioritize_dimensions(
        output, it_space, stick_vars={N_SYM: 64, K_SYM: 64},
    )
    assert default == [N_SYM, M_SYM, K_SYM]
    assert heuristic == [N_SYM, M_SYM, K_SYM]


def test_square_shape_tied_element_count():
    """L3-70B GQA TP=8 prefill: M=128, N=128 (=2 sticks). Element counts
    are tied; the sort is stable so the original it_space order
    (M, N) is preserved."""
    config.output_element_priority = True
    output = _matmul_output()
    it_space = {
        M_SYM: Integer(128),
        N_SYM: Integer(2),
        K_SYM: Integer(128),
    }
    priorities = prioritize_dimensions(
        output, it_space, stick_vars={N_SYM: 64, K_SYM: 64},
    )
    assert priorities == [M_SYM, N_SYM, K_SYM]


def test_exclude_reduction_drops_k():
    """When exclude_reduction=True, reduction dims should not appear in
    the priority list regardless of heuristic state."""
    config.output_element_priority = True
    output = _matmul_output()
    it_space = {
        M_SYM: Integer(128),
        N_SYM: Integer(64),
        K_SYM: Integer(64),
    }
    priorities = prioritize_dimensions(
        output, it_space,
        exclude_reduction=True,
        stick_vars={N_SYM: 64, K_SYM: 64},
    )
    assert K_SYM not in priorities
    assert priorities == [N_SYM, M_SYM]


def test_works_without_stick_vars():
    """Heuristic must not crash when stick_vars is None (e.g. callers that
    didn't update). Falls back to using the iter-space values directly."""
    config.output_element_priority = True
    output = _matmul_output()
    it_space = {
        M_SYM: Integer(128),
        N_SYM: Integer(64),
    }
    priorities = prioritize_dimensions(output, it_space)
    assert priorities == [M_SYM, N_SYM]
