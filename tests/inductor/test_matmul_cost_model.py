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

import pytest

import torch_spyre._inductor.work_division as work_division


def _best_qo_fp8_split(m: int) -> tuple[int, int]:
    candidates = []
    for m_split in (1, 2, 4, 8, 16, 32):
        if m % m_split != 0 or (m // m_split) % 2 != 0:
            continue
        n_split = 32 // m_split
        cost = work_division._matmul_split_cost(
            (1, 1),
            (m, m_split),
            (4096, n_split),
            (4096, 1),
            32,
            shared_weight=True,
            profile=work_division._FP8_MATMUL_COST_PROFILE,
        )
        candidates.append((cost, m_split, n_split))
    _, m_split, n_split = min(candidates)
    return m_split, n_split


@pytest.mark.parametrize(
    ("m", "expected"),
    [
        (2, (1, 32)),
        (4, (2, 16)),
        (8, (4, 8)),
        (16, (4, 8)),
        (32, (4, 8)),
        (64, (4, 8)),
        (128, (4, 8)),
        (256, (4, 8)),
        (512, (4, 8)),
        (1024, (8, 4)),
        (2048, (8, 4)),
    ],
)
def test_fp8_qo_cost_profile_tracks_dd2_oracle(m, expected):
    assert _best_qo_fp8_split(m) == expected


def test_fp8_cost_profile_models_fma8_and_mixed_precision_bytes():
    fp16 = work_division._DL16_MATMUL_COST_PROFILE
    fp8 = work_division._FP8_MATMUL_COST_PROFILE

    assert fp8.peak_macs_us_core == 2 * fp16.peak_macs_us_core
    assert (fp8.lhs_bytes, fp8.rhs_bytes, fp8.output_bytes) == (1, 1, 2)
