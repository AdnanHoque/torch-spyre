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

"""Unit tests for the Phase 2 K-split work-division heuristic.

Tests the pure helper `_k_split_heuristic_should_fire` against the canonical
shapes measured in Phase 1 (see tests/splitk_phase1_findings.md). Hardware-
free; the helper takes already-concretized iteration-space tuples.
"""

from __future__ import annotations

import pytest
from sympy import Integer, Symbol

import torch_spyre  # noqa: F401  -- ensure backend is registered

from torch_spyre._inductor import config
from torch_spyre._inductor.core_division import _k_split_heuristic_should_fire


# ---- helpers -----------------------------------------------------------------

M_SYM = Symbol("M")
N_SYM = Symbol("N")
K_SYM = Symbol("K")


def _iter_space_pair(M_iter: int, N_iter: int, K_iter: int):
    """Return (output_dims, reduction_dims) tuples in the form the heuristic
    expects.

    Sizes are in **iteration-space units**: for stick dims they are stick
    counts (fp16 stick = 64 elements), for non-stick dims they are element
    counts. For a 2D matmul output:
      - M is non-stick → M_iter == M_elements
      - N is stick → N_iter == N_elements / 64
    Reduction K is stick on input A → K_iter == K_elements / 64.
    """
    output = [(M_SYM, Integer(M_iter)), (N_SYM, Integer(N_iter))]
    reduction = [(K_SYM, Integer(K_iter))]
    return output, reduction


def _fire(out, red, *, coord_vars=None, min_split_vars=None, num_cores=32):
    """Convenience wrapper supplying coord_vars / min_split_vars defaults that
    match a 2D matmul where M and N are output coord vars and nothing is
    span-pre-split."""
    if coord_vars is None:
        coord_vars = {M_SYM, N_SYM}
    if min_split_vars is None:
        min_split_vars = set()
    return _k_split_heuristic_should_fire(
        out, red, coord_vars, min_split_vars, num_cores
    )


@pytest.fixture
def heuristic_on(monkeypatch):
    """Enable the heuristic at default thresholds for the test."""
    monkeypatch.setattr(config, "k_split_heuristic", True)
    monkeypatch.setattr(config, "k_split_max_output_iter_units", 32768)
    monkeypatch.setattr(config, "k_split_min_k_iter_units", 64)


# ---- canonical Phase 1 shapes (measured) ------------------------------------

# Each case below corresponds to a Phase 1 bench measurement at SENCORES=32.
# Iter-space sizes assume fp16 (stick = 64 elems). Decode-shape rows where
# fixed-overhead dominates and forceK is a small (~4%) loss are still expected
# to fire — the heuristic is a perf+accuracy trade and accepts a marginal
# perf cost when accuracy improves meaningfully (Phase 1: 7-15% drift drop).

@pytest.mark.parametrize(
    "name, M_elems, N_elems, K_elems, expect_fire",
    [
        # forceK measured wins — heuristic should fire.
        ("L3-8B q_proj prefill",      128, 4096, 4096,  True),   # 1.09x
        ("L3-8B MLP-down prefill",    128, 4096, 14336, True),   # 1.23x
        ("L3-70B q_proj prefill",     128, 8192, 8192,  True),   # 1.26x
        ("M-scale 128 N=4096 K=8192", 128, 4096, 8192,  True),   # 1.16x

        # forceK measured losses — heuristic must NOT fire.
        ("M-scale 2048 large M*N",    2048, 4096, 8192, False),  # 0.56x

        # Decode-skinny: forceK is a slight perf loss (4%) but accuracy
        # improves 7-15%. Acceptable trade for an opt-in heuristic.
        ("Decode-skinny large K",     1, 4096, 16384,   True),   # 0.96x

        # Boundary: M*N just at the threshold should not fire.
        ("M*N at threshold (32K iter)", 256, 8192, 8192, False),
    ],
)
def test_canonical_phase1_shapes(
    heuristic_on, name, M_elems, N_elems, K_elems, expect_fire
):
    M_iter = M_elems  # M is non-stick
    N_iter = N_elems // 64  # N is stick (fp16)
    K_iter = K_elems // 64  # K is stick on A
    out, red = _iter_space_pair(M_iter, N_iter, K_iter)
    got = _fire(out, red)
    assert got is expect_fire, (
        f"{name}: expected fire={expect_fire} for "
        f"(M={M_elems}, N={N_elems}, K={K_elems}) "
        f"-> iter (M={M_iter}, N={N_iter}, K={K_iter}), got fire={got}"
    )


def test_span_pre_split_does_not_fire(heuristic_on):
    """L3-70B MLP-down (M=128, N=8192, K=28672): B = 28672·8192·2 = 470 MB
    exceeds the per-core 256 MB span limit. The planner pre-splits N, so N
    appears in `min_splits`. ForceK on this shape was 0.32× in Phase 1 — the
    heuristic must detect the span-pressure regime and decline.
    """
    out, red = _iter_space_pair(128, 8192 // 64, 28672 // 64)
    # Without span context, the heuristic would fire (M·N=16K is below
    # threshold and K=448 is divisible by 32). The span-pre-split signal
    # is what flips the decision.
    assert _fire(out, red) is True
    assert _fire(out, red, min_split_vars={N_SYM}) is False


# ---- gate conditions in isolation -------------------------------------------

def test_disabled_by_default():
    """Without the config flag, the heuristic returns False even for
    otherwise-perfect shapes. Phase 2 is opt-in initially."""
    out, red = _iter_space_pair(128, 128, 128)
    assert _fire(out, red) is False


def test_no_op_for_pointwise(heuristic_on):
    """Pointwise ops have no reduction dim; heuristic must short-circuit."""
    out, _ = _iter_space_pair(128, 128, 0)
    assert _fire(out, []) is False


def test_no_op_for_pure_reduction(heuristic_on):
    """No output dims (degenerate case). Defensive."""
    red = [(K_SYM, Integer(128))]
    assert _fire([], red, coord_vars=set()) is False


def test_small_K_does_not_fire(heuristic_on):
    """K below min threshold: parallelism gain too small."""
    out, red = _iter_space_pair(128, 128, 32)  # K_iter=32 < 64
    assert _fire(out, red) is False


def test_large_output_does_not_fire(heuristic_on):
    """M·N at or above threshold: cross-core reduction overhead dominates.
    M=2048, N=4096 elems → M_iter * N_iter = 2048 * 64 = 131072 > 32768."""
    out, red = _iter_space_pair(2048, 64, 128)
    assert _fire(out, red) is False


def test_threshold_tunable_via_config(monkeypatch):
    """The thresholds must be runtime-tunable via config so users can
    calibrate to non-default dtypes / shape regimes without a rebuild."""
    monkeypatch.setattr(config, "k_split_heuristic", True)
    monkeypatch.setattr(config, "k_split_max_output_iter_units", 1024)
    monkeypatch.setattr(config, "k_split_min_k_iter_units", 64)

    # M·N=16K — exceeds tightened threshold of 1K, must not fire.
    out, red = _iter_space_pair(128, 128, 128)
    assert _fire(out, red) is False

    # Loosen threshold; now should fire.
    monkeypatch.setattr(config, "k_split_max_output_iter_units", 32768)
    assert _fire(out, red) is True


def test_alignment_check_uses_num_cores(heuristic_on):
    """The alignment check must use num_cores, not a hardcoded 32 — so a
    user with SENCORES=16 still gets correct behavior."""
    # K=64 is divisible by 16 and 32 — fires at both. Not divisible by 24
    # (64 % 24 = 16) — declines at num_cores=24.
    out, red = _iter_space_pair(128, 128, 64)
    assert _fire(out, red, num_cores=16) is True
    assert _fire(out, red, num_cores=32) is True
    assert _fire(out, red, num_cores=24) is False


def test_min_split_only_blocks_when_output_var(heuristic_on):
    """min_splits containing a NON-output variable (i.e. K) should NOT
    block the heuristic — that means K was span-pre-split, which is fine.
    Only output-dim min-splits indicate the regime where the heuristic
    empirically loses."""
    out, red = _iter_space_pair(128, 128, 128)
    # K span-pre-split: not a blocker, heuristic still fires.
    assert _fire(out, red, min_split_vars={K_SYM}) is True
    # Some unknown sym in min_splits but it's not an output var: not a
    # blocker either.
    assert _fire(out, red, min_split_vars={Symbol("Z")}) is True
    # M is an output var: blocked.
    assert _fire(out, red, min_split_vars={M_SYM}) is False
