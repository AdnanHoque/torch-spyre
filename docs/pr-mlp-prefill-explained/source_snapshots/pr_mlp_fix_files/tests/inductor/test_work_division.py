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

import math

from sympy import divisors

from torch_spyre._inductor.work_division import _matmul_split_cost


def _best_matmul_split(M: int, N: int, K: int, max_cores: int = 32):
    best = None
    best_cost = float("inf")
    for m in (int(d) for d in divisors(M)):
        for n in (int(d) for d in divisors(N // 64)):
            for k in (int(d) for d in divisors(K // 64)):
                if m * n * k > max_cores:
                    continue
                cost = _matmul_split_cost(
                    (1, 1), (M, m), (N, n), (K, k), max_cores
                )
                if cost < best_cost:
                    best = (m, n, k)
                    best_cost = cost
    return best


def test_wide_prefill_matmul_prefers_output_parallelism_after_pt_is_filled():
    assert _best_matmul_split(M=512, N=12800, K=4096) == (4, 8, 1)


def test_small_m_matmul_still_penalizes_m_underfill():
    high_m_split = _matmul_split_cost((1, 1), (64, 16), (4096, 2), (4096, 1), 32)
    balanced_split = _matmul_split_cost((1, 1), (64, 4), (4096, 8), (4096, 1), 32)

    assert math.isfinite(high_m_split)
    assert balanced_split < high_m_split
