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

import sympy

from torch_spyre._inductor.split_multi_ops import _linearized_load_index


def test_linearized_load_index_same_rank():
    d0, d1 = sympy.symbols("d0 d1")

    index = _linearized_load_index((d0, d1), [4, 8], [8, 1])

    assert sympy.simplify(index - (8 * d0 + d1)) == 0


def test_linearized_load_index_unflattens_trailing_view_dims():
    d0, d1, d2 = sympy.symbols("d0 d1 d2")

    index = _linearized_load_index(
        (0, d0, d1, d2),
        [1, 512, 32, 2, 1, 64],
        [2097152, 4096, 128, 64, 64, 1],
    )

    expected = (
        4096 * d0
        + 128 * d1
        + 64 * sympy.Mod(sympy.floor(d2 / 64), 2)
        + sympy.Mod(d2, 64)
    )
    assert sympy.simplify(index - expected) == 0
