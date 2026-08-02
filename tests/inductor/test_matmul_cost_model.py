# Copyright 2026 The Torch-Spyre Authors.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import sympy

import torch_spyre._inductor.work_division as work_division


M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
GRANITE_TP1_SHAPES = {
    "kv": (4096, 1024),
    "qo": (4096, 4096),
    "gate_up": (4096, 12800),
    "down": (12800, 4096),
}
GRANITE_TP1_EXPECTED_SPLITS = {
    "kv": (
        (1, 8),
        (1, 8),
        (2, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
    ),
    "qo": (
        (1, 32),
        (1, 32),
        (2, 16),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (8, 4),
        (8, 4),
    ),
    "gate_up": (
        (1, 10),
        (1, 10),
        (2, 10),
        (4, 5),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
    ),
    "down": (
        (1, 8),
        (1, 8),
        (2, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (4, 8),
        (8, 4),
        (8, 4),
        (8, 4),
        (8, 4),
    ),
}
GRANITE_TP1_CASES = [
    pytest.param(
        projection,
        m,
        *GRANITE_TP1_SHAPES[projection],
        expected,
        id=f"{projection}-m{m}",
    )
    for projection in GRANITE_TP1_SHAPES
    for m, expected in zip(M_VALUES, GRANITE_TP1_EXPECTED_SPLITS[projection])
]


def _best_granite_fp8_split(
    m: int, k: int, n: int, *, enforce_weight_storage: bool = True
) -> tuple[int, int]:
    # QFP8MB packs two activation rows together for even M. M=1 takes the
    # QFP8CH path, so it has no two-row atomicity constraint.
    m_basis = m // 2 if m % 2 == 0 else m
    n_basis = n // 64
    candidates = []
    for m_split in map(int, work_division.divisors(m_basis)):
        for n_split in map(int, work_division.divisors(n_basis)):
            if m_split * n_split > 32:
                continue
            if (
                enforce_weight_storage
                and not work_division._physical_output_split_is_legal(n, n_split, 128)
            ):
                continue
            cost = work_division._matmul_split_cost(
                (1, 1),
                (m, m_split),
                (n, n_split),
                (k, 1),
                32,
                shared_weight=True,
                profile=work_division._FP8_MATMUL_COST_PROFILE,
            )
            candidates.append((cost, m_split, n_split))
    _, m_split, n_split = min(candidates)
    return m_split, n_split


def _run_fp8_work_division_override(
    m: int,
    k: int,
    n: int,
    *,
    m_split: int,
    n_split: int,
) -> dict[sympy.Symbol, int]:
    """Exercise the compiler planner with DD2 compound-FP8 tensor metadata."""
    m_dim, n_dim, k_dim = sympy.symbols("m n k", integer=True)

    def tensor_dep(device_coords, element_arrangement=None, physical_group=1):
        device_layout = SimpleNamespace(device_size=(physical_group,))
        if element_arrangement is not None:
            device_layout.element_arrangement = element_arrangement
        return SimpleNamespace(
            dep=SimpleNamespace(index=m_dim * n + n_dim),
            device_coords=device_coords,
            layout=SimpleNamespace(device_layout=device_layout),
        )

    lhs = tensor_dep(
        [m_dim, k_dim, k_dim],
        work_division.ElementArrangement.QFP8MB,
        physical_group=128,
    )
    rhs = tensor_dep(
        [k_dim, n_dim, n_dim],
        work_division.ElementArrangement.QFP8WT,
        physical_group=128,
    )
    output = tensor_dep([m_dim, n_dim, n_dim])

    reduction = MagicMock(spec=work_division.Reduction)
    reduction.reduction_type = work_division.BATCH_MATMUL_FP8MB_OP
    op = SimpleNamespace(data=reduction, get_name=lambda: "fp8_matmul")

    initial_splits = {m_dim: 4, n_dim: 8, k_dim: 1}
    with work_division.config.patch(
        {
            "fp8_lx_poc_m_split": m_split,
            "fp8_lx_poc_n_split": n_split,
        }
    ):
        return work_division._cost_model_matmul_planner(
            op,
            initial_splits,
            {
                m_dim: sympy.Integer(m // 2),
                n_dim: sympy.Integer(n // 64),
                k_dim: sympy.Integer(k // 128),
            },
            output,
            {m_dim: 2, n_dim: 64, k_dim: 128},
            dict(initial_splits),
            32,
            [lhs, rhs],
        )


@pytest.mark.parametrize(("projection", "m", "k", "n", "expected"), GRANITE_TP1_CASES)
def test_fp8_granite_cost_choice_is_physically_legal(projection, m, k, n, expected):
    del projection
    m_split, n_split = _best_granite_fp8_split(m, k, n)

    assert (m_split, n_split) == expected
    assert m_split * n_split <= 32
    assert (m // m_split) % (2 if m % 2 == 0 else 1) == 0
    assert (n // n_split) % 64 == 0
    assert work_division._physical_output_split_is_legal(n, n_split, 128)


def test_fp8_work_division_override_forces_legal_qo_grid():
    result = _run_fp8_work_division_override(
        512,
        4096,
        4096,
        m_split=8,
        n_split=4,
    )
    m_dim, n_dim, k_dim = sympy.symbols("m n k", integer=True)

    assert result == {m_dim: 8, n_dim: 4, k_dim: 1}


def test_fp8_work_division_override_rejects_split_through_weight_group():
    with pytest.raises(
        work_division.Unsupported,
        match="N split 8 cuts a QFP8WT physical group",
    ):
        _run_fp8_work_division_override(
            512,
            4096,
            12800,
            m_split=4,
            n_split=8,
        )


@pytest.mark.parametrize(("projection", "m", "k", "n", "expected"), GRANITE_TP1_CASES)
def test_fp8_granite_scale_mapping_preserves_matmul_output_split(
    projection, m, k, n, expected
):
    del projection, k
    m_split, n_split = expected
    m_dim, n_dim = sympy.symbols("m n", integer=True)
    producer_splits = {}
    if m_split > 1:
        producer_splits[sympy.Integer(n)] = m_split
    if n_split > 1:
        producer_splits[sympy.Integer(1)] = n_split

    expected_scale_splits = {}
    if m_split > 1:
        expected_scale_splits[m_dim] = m_split
    if n_split > 1:
        expected_scale_splits[n_dim] = n_split
    assert (
        work_division._map_producer_output_splits(
            producer_splits,
            m_dim * n + n_dim,
            {m_dim: sympy.Integer(m), n_dim: sympy.Integer(n)},
        )
        == expected_scale_splits
    )


@pytest.mark.parametrize(
    ("m", "expected"),
    [(1, (1, 10)), (2, (1, 10)), (4, (2, 10)), (8, (4, 5))],
)
def test_fp8_gate_up_supports_non_power_of_two_core_grids(m, expected):
    assert _best_granite_fp8_split(m, 4096, 12800) == expected


def test_fp8_gate_up_m512_rejects_unrepresentable_weight_split():
    # Output-stick alignment alone picks 4x8, but 100 packed QFP8WT storage
    # groups cannot be divided eight ways. Without this guard, later layout
    # rewriting clamps only the matmul to 4x4 and leaves both scale ops at 4x8.
    assert _best_granite_fp8_split(512, 4096, 12800, enforce_weight_storage=False) == (
        4,
        8,
    )
    assert not work_division._physical_output_split_is_legal(12800, 8, 128)
    assert _best_granite_fp8_split(512, 4096, 12800) == (8, 4)


def test_fp8_cost_profile_models_fma8_and_mixed_precision_bytes():
    fp16 = work_division._DL16_MATMUL_COST_PROFILE
    fp8 = work_division._FP8_MATMUL_COST_PROFILE

    assert fp8.peak_macs_us_core == 2 * fp16.peak_macs_us_core
    assert (fp8.lhs_bytes, fp8.rhs_bytes, fp8.output_bytes) == (1, 1, 2)


def test_fp8_scale_maps_both_producer_output_splits_by_tensor_coefficient():
    m, n = sympy.symbols("m n", integer=True)
    producer_splits = {sympy.Integer(4096): 4, sympy.Integer(1): 8}

    assert work_division._map_producer_output_splits(
        producer_splits,
        m * 4096 + n,
        {m: sympy.Integer(512), n: sympy.Integer(4096)},
    ) == {m: 4, n: 8}


def test_fp8_scale_rejects_partial_producer_split_mapping():
    m = sympy.symbols("m", integer=True)
    producer_splits = {sympy.Integer(4096): 4, sympy.Integer(1): 8}

    assert (
        work_division._map_producer_output_splits(
            producer_splits,
            m * 4096,
            {m: sympy.Integer(512)},
        )
        is None
    )


def test_qfp8wt_physical_output_group_rejects_gate_up_n8_split():
    # N=12800 has 200 FP16 output sticks but only 100 physical 128-element
    # QFP8WT storage groups.  N split 8 looks legal from the output alone and
    # is not legal for the stationary FP8 weight; N split 4 is legal for both.
    assert not work_division._physical_output_split_is_legal(12800, 8, 128)
    assert work_division._physical_output_split_is_legal(12800, 4, 128)


@pytest.mark.parametrize(("n", "split"), [(1024, 4), (4096, 8), (12800, 25)])
def test_qfp8wt_physical_output_group_accepts_granite_splits(n, split):
    assert work_division._physical_output_split_is_legal(n, split, 128)


def test_qfp8mb_packer_spends_cores_on_legal_m_row_pairs(monkeypatch):
    """A forbidden K split must not consume the greedy core budget first."""

    m_dim, k_dim = sympy.symbols("m k", integer=True)
    output = SimpleNamespace(
        dep=SimpleNamespace(index=m_dim * 4096 + k_dim),
        # QFP8MB's compound stick repeats K around the packed M:2 factor.
        device_coords=[k_dim, m_dim, k_dim],
        layout=SimpleNamespace(
            device_layout=SimpleNamespace(
                element_arrangement=work_division.ElementArrangement.QFP8MB,
                device_size=(8, 2, 8),
            )
        ),
    )
    op = SimpleNamespace(data=MagicMock(), get_name=lambda: "qfp8mb")
    committed = {}

    monkeypatch.setattr(
        work_division,
        "iteration_space_from_op",
        lambda _op: {m_dim: sympy.Integer(64), k_dim: sympy.Integer(4096)},
    )
    monkeypatch.setattr(
        work_division, "collect_tensor_deps", lambda _op, _args: ([], output)
    )
    monkeypatch.setattr(work_division, "_collect_symbol_metadata", lambda _it: {})
    monkeypatch.setattr(
        work_division,
        "adjust_it_space_for_sticks",
        lambda _it, _tds, _meta: (
            {m_dim: sympy.Integer(32), k_dim: sympy.Integer(64)},
            {m_dim: 2, k_dim: 64},
        ),
    )
    monkeypatch.setattr(
        work_division,
        "_get_fp8_compound_split_constraints",
        lambda _inputs, _output: {k_dim: 1},
    )
    monkeypatch.setattr(
        work_division, "coordinate_mask_blocked_vars", lambda *_args: set()
    )
    monkeypatch.setattr(work_division, "warn_if_per_core_overflow", lambda *_args: None)
    monkeypatch.setattr(
        work_division,
        "apply_splits",
        lambda _op, splits, _output: committed.update(splits),
    )

    with work_division.config.patch(
        {
            "ignore_work_division_hints": True,
            "fp8_pack_poc_m_split": 32,
        }
    ):
        work_division.work_distribution_pass(op, [], 32)

    assert committed == {m_dim: 32, k_dim: 1}


def test_qfp8mb_packing_grid_rejects_split_through_m_row_pair(monkeypatch):
    m_dim, k_dim = sympy.symbols("m k", integer=True)
    output = SimpleNamespace(
        device_coords=[k_dim, m_dim, k_dim],
        layout=SimpleNamespace(
            device_layout=SimpleNamespace(
                element_arrangement=work_division.ElementArrangement.QFP8MB
            )
        ),
    )
    op = SimpleNamespace(data=MagicMock(), get_name=lambda: "qfp8mb")

    monkeypatch.setattr(
        work_division,
        "iteration_space_from_op",
        lambda _op: {m_dim: sympy.Integer(64), k_dim: sympy.Integer(4096)},
    )
    monkeypatch.setattr(
        work_division, "collect_tensor_deps", lambda _op, _args: ([], output)
    )
    monkeypatch.setattr(work_division, "_collect_symbol_metadata", lambda _it: {})
    monkeypatch.setattr(
        work_division,
        "adjust_it_space_for_sticks",
        lambda _it, _tds, _meta: (
            {m_dim: sympy.Integer(32), k_dim: sympy.Integer(64)},
            {m_dim: 2, k_dim: 64},
        ),
    )
    monkeypatch.setattr(
        work_division,
        "_get_fp8_compound_split_constraints",
        lambda _inputs, _output: {k_dim: 1},
    )

    with work_division.config.patch(
        {
            "ignore_work_division_hints": True,
            "fp8_pack_poc_m_split": 3,
        }
    ):
        with pytest.raises(
            work_division.Unsupported,
            match="dividing 32 M-row pairs, got 3",
        ):
            work_division.work_distribution_pass(op, [], 32)


def test_fp8_packing_chain_oracle_targets_only_div_feeding_qfp8mb(monkeypatch):
    m_dim, k_dim = sympy.symbols("m k", integer=True)

    class FakeComputedBuffer:
        def __init__(self, target, name):
            self.data = SimpleNamespace(origins=(SimpleNamespace(target=target),))
            self._name = name

        def get_name(self):
            return self._name

    monkeypatch.setattr(work_division, "ComputedBuffer", FakeComputedBuffer)
    div = FakeComputedBuffer("aten.div.Tensor", "normalized")
    qfp8mb = FakeComputedBuffer("spyre.qfp8mb.default", "packed")
    graph = SimpleNamespace(operations=[div, qfp8mb])
    output = SimpleNamespace(device_coords=[m_dim, k_dim])
    committed = {}

    monkeypatch.setattr(
        work_division,
        "op_read_writes",
        lambda op: SimpleNamespace(
            reads=[SimpleNamespace(name="normalized")] if op is qfp8mb else []
        ),
    )
    monkeypatch.setattr(
        work_division,
        "iteration_space_from_op",
        lambda _op: {m_dim: sympy.Integer(64), k_dim: sympy.Integer(4096)},
    )
    monkeypatch.setattr(
        work_division, "collect_tensor_deps", lambda _op, _args: ([], output)
    )
    monkeypatch.setattr(work_division, "_collect_symbol_metadata", lambda _it: {})
    monkeypatch.setattr(
        work_division,
        "adjust_it_space_for_sticks",
        lambda _it, _tds, _meta: (
            {m_dim: sympy.Integer(64), k_dim: sympy.Integer(64)},
            {k_dim: 64},
        ),
    )
    monkeypatch.setattr(
        work_division,
        "apply_splits",
        lambda _op, splits, _output: committed.update(splits),
    )

    with work_division.config.patch({"fp8_pack_chain_poc_m_split": 1}):
        assert work_division._force_fp8_pack_normalization_work_division(
            graph, div, [], 32
        )

    assert committed == {m_dim: 1, k_dim: 1}
