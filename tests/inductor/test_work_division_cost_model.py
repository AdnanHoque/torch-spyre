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

import sys
import types

from sympy import Symbol


def _install_spyre_c_stub() -> None:
    if "torch_spyre._C" in sys.modules:
        return

    class _FakeDataFormat:
        def __init__(self, name: str):
            self.name = name

        def elems_per_stick(self) -> int:
            return 64

    class _FakeDataFormats:
        SEN169_FP16 = _FakeDataFormat("SEN169_FP16")
        IEEE_FP32 = _FakeDataFormat("IEEE_FP32")
        SEN143_FP8 = _FakeDataFormat("SEN143_FP8")

    class _FakeElementArrangement:
        STANDARD = "STANDARD"
        DL16_TO_FP32 = "DL16_TO_FP32"
        DL16_TO_FP8 = "DL16_TO_FP8"
        EXX2 = "EXX2"

    class _FakeSpyreTensorLayout:
        def __init__(
            self,
            device_size=None,
            stride_map=None,
            device_dtype=None,
            *_args,
            **_kwargs,
        ):
            self.device_size = device_size or []
            self.stride_map = stride_map or []
            self.device_dtype = device_dtype or _FakeDataFormats.SEN169_FP16

        def elems_per_stick(self) -> int:
            return self.device_dtype.elems_per_stick()

    stub = types.ModuleType("torch_spyre._C")
    stub.DataFormats = _FakeDataFormats
    stub.ElementArrangement = _FakeElementArrangement
    stub.SpyreTensorLayout = _FakeSpyreTensorLayout
    stub.get_elem_in_stick = lambda *_args, **_kwargs: 64
    stub.encode_constant = lambda *_args, **_kwargs: 0
    sys.modules["torch_spyre._C"] = stub


_install_spyre_c_stub()

from torch_spyre._inductor.work_division import (  # noqa: E402
    _matmul_split_cost,
    _pick_innermost_output_dim,
)


def test_shared_weight_dim_id_picks_innermost_row_dim():
    batch = Symbol("batch")
    m = Symbol("m")
    n = Symbol("n")

    output_index = batch * 512 * 12800 + m * 12800 + n

    assert _pick_innermost_output_dim([batch, m], output_index) == m


def test_shared_weight_cost_charges_weight_once():
    non_shared = _matmul_split_cost(
        (4, 1),
        (512, 8),
        (12800, 4),
        (4096, 1),
        32,
        shared_weight=False,
    )
    shared = _matmul_split_cost(
        (4, 1),
        (512, 8),
        (12800, 4),
        (4096, 1),
        32,
        shared_weight=True,
    )

    assert shared < non_shared


def _divisors(n: int):
    return [d for d in range(1, n + 1) if n % d == 0]


def _best_shared_weight_split(n_sticks: int, k_elems: int = 4096):
    best = None
    best_cost = float("inf")
    for m_split in (1, 2, 4, 8, 16, 32):
        for n_split in (1, 2, 4, 8, 16, 32):
            for k_split in (1, 2, 4, 8, 16, 32):
                if m_split * n_split * k_split > 32:
                    continue
                cost = _matmul_split_cost(
                    (1, 1),
                    (512, m_split),
                    (n_sticks * 64, n_split),
                    (k_elems, k_split),
                    32,
                    shared_weight=True,
                )
                if cost < best_cost:
                    best_cost = cost
                    best = (m_split, n_split, k_split)
    return best


def test_shared_weight_cost_model_keeps_pt_friendly_m_tile():
    assert _best_shared_weight_split(16) == (8, 4, 1)
    assert _best_shared_weight_split(64) == (4, 8, 1)
    assert _best_shared_weight_split(200) == (4, 8, 1)


def test_folded_long_k_projection_prefers_more_m_lanes():
    # Granite e2e MLP-down matmuls are folded to a no-batch 2D projection by
    # the time the planner runs. They still have an unbatched RHS loaded once,
    # and the long-K reduction shape is fastest with more M splitting.
    assert _best_shared_weight_split(64, k_elems=12800) == (8, 4, 1)


def _best_true_bmm_split(B: int, M: int, N: int, K: int):
    best = None
    best_cost = float("inf")
    for b_split in _divisors(B):
        for m_split in _divisors(M):
            for n_split in _divisors(N // 64):
                for k_split in _divisors(K // 64):
                    if b_split * m_split * n_split * k_split > 32:
                        continue
                    cost = _matmul_split_cost(
                        (B, b_split),
                        (M, m_split),
                        (N, n_split),
                        (K, k_split),
                        32,
                        shared_weight=False,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best = (b_split, m_split, n_split, k_split)
    return best


def test_true_bmm_attention_cost_model_uses_structural_parallelism():
    assert _best_true_bmm_split(512, 32, 512, 128) == (2, 2, 8, 1)
    assert _best_true_bmm_split(32, 512, 128, 512) == (1, 32, 1, 1)
    assert _best_true_bmm_split(32, 64, 128, 576) == (8, 4, 1, 1)
