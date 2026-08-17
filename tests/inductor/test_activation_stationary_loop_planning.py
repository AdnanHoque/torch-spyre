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

import sympy
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_index_symbol

from torch_spyre._inductor.constants import COARSE_TILE_HOISTED_LOOP_GROUP_ATTR
from torch_spyre._inductor.loop_info import copy_op_metadata
from torch_spyre._inductor.scratchpad.utils import (
    hoisted_loop_lifetime_end_overrides,
)
from torch_spyre._inductor.wsr.coarse_tile import _host_tile_advances_for_dep


def test_expert_advance_is_measured_relative_to_the_window_base():
    expert = sympy_index_symbol("d0")
    row = sympy_index_symbol("d1")
    dep = MemoryDep(
        name="weight",
        index=17 + 4096 * expert + 64 * row,
        var_names=(expert, row),
        size=(2, 64),
    )

    advances = _host_tile_advances_for_dep(
        dep,
        [{}, {0: sympy.Integer(1)}],
    )

    assert advances == [[], [(0, sympy.Integer(4096))]]


def test_hoisted_copy_owner_survives_ir_reconstruction():
    source = SimpleNamespace()
    destination = SimpleNamespace()
    setattr(source, COARSE_TILE_HOISTED_LOOP_GROUP_ATTR, (3, 1))

    copy_op_metadata(source, destination)

    assert getattr(destination, COARSE_TILE_HOISTED_LOOP_GROUP_ATTR) == (3, 1)


def test_hoisted_copy_lifetime_ends_after_its_own_loop_group():
    preheader = SimpleNamespace(get_name=lambda: "x_copy")
    setattr(preheader, COARSE_TILE_HOISTED_LOOP_GROUP_ATTR, (2,))
    first_body = SimpleNamespace(loop_info=SimpleNamespace(loop_group_id=(2,)))
    nested_body = SimpleNamespace(loop_info=SimpleNamespace(loop_group_id=(2, 0)))
    later_group = SimpleNamespace(loop_info=SimpleNamespace(loop_group_id=(3,)))
    graph = SimpleNamespace(
        operations=[preheader, first_body, nested_body, later_group]
    )

    assert hoisted_loop_lifetime_end_overrides(graph) == {"x_copy": 3}
