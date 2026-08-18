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
import pytest
from torch._inductor.dependencies import MemoryDep
from torch._inductor.utils import sympy_index_symbol

from torch_spyre._inductor.loop_info import (
    CoarseTileInfo,
    LoopOperandBindingRequirement,
    copy_op_metadata,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import coeff_through_floor
from torch_spyre._inductor.spyre_kernel import SpyreKernel, TensorAccess
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


def test_typed_preheader_owner_survives_ir_reconstruction():
    source = type("Source", (), {})()
    source.loop_info = CoarseTileInfo(
        loop_group_id=(),
        loop_count=[],
        loop_tiled_dims=[],
        preheader_for_group=(3, 1),
    )
    destination = type("Destination", (), {})()

    copy_op_metadata(source, destination)

    assert destination.loop_info.preheader_for_group == (3, 1)


def _fully_squeezed_read_advance(binding):
    dep = MemoryDep(
        name="weight",
        index=sympy.Integer(0),
        var_names=(),
        size=(),
    )
    ir_node = SimpleNamespace(
        loop_info=CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[sympy.Integer(128)],
            loop_tiled_dims=[[0]],
            tiled_dims_per_read=[[[]]],
            squeezed_advance_per_read=[[[(sympy.Integer(64), sympy.Integer(1))]]],
            loop_operand_bindings=[binding],
        ),
        get_operation_name=lambda: "expert_weight_copy",
        get_read_writes=lambda: SimpleNamespace(reads=[dep], writes=[]),
    )
    kernel = SpyreKernel()
    kernel.current_node = SimpleNamespace(node=ir_node)
    tensor = TensorAccess(
        "weight",
        sympy.Integer(0),
        SimpleNamespace(
            device_layout=SimpleNamespace(
                device_size=[128, 64],
                stride_map=[64, 1],
            )
        ),
    )

    advance = kernel._general_tile_advance(tensor, True, "weight")
    level = kernel._tile_advance_symbols[0]
    return sympy.expand(advance).coeff(level)


def test_persistent_binding_and_ordinary_squeezed_advance_coexist():
    # The upstream squeezed-dim fallback remains the address source for an
    # ordinary copy. A persistent expert operand instead uses its pre-division
    # binding, which retains the weight-bank stride hidden by the unit E tile.
    assert _fully_squeezed_read_advance(None) == 64
    assert (
        _fully_squeezed_read_advance(
            LoopOperandBindingRequirement(
                kind="sequential_affine",
                host_advance_per_level=(sympy.Integer(4096),),
            )
        )
        == 4096
    )


def test_binding_conflict_with_surviving_dim_fails_closed():
    dim = sympy_index_symbol("d0")
    dep = MemoryDep(
        name="weight",
        index=dim,
        var_names=(dim,),
        size=(64,),
    )
    ir_node = SimpleNamespace(
        loop_info=CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[sympy.Integer(128)],
            loop_tiled_dims=[[0]],
            tiled_dims_per_read=[[[(0, sympy.Integer(1))]]],
            loop_operand_bindings=[
                LoopOperandBindingRequirement(
                    kind="sequential_affine",
                    host_advance_per_level=(sympy.Integer(4096),),
                )
            ],
        ),
        get_operation_name=lambda: "mixed_weight_copy",
        get_read_writes=lambda: SimpleNamespace(reads=[dep], writes=[]),
    )
    kernel = SpyreKernel()
    kernel.current_node = SimpleNamespace(node=ir_node)
    tensor = TensorAccess(
        "weight",
        sympy.Integer(0),
        SimpleNamespace(
            device_layout=SimpleNamespace(
                device_size=[64],
                stride_map=[1],
            )
        ),
    )

    with pytest.raises(Unsupported, match="conflicts with surviving tiled dimensions"):
        kernel._general_tile_advance(tensor, True, "weight")


def test_binding_owns_a_squeezed_expert_dim_without_aliasing_the_row_symbol():
    row = sympy_index_symbol("d0")
    dep = MemoryDep(
        name="routing_weight",
        index=row,
        var_names=(row,),
        size=(64,),
    )
    ir_node = SimpleNamespace(
        data=SimpleNamespace(ranges=[1, 64, 1]),
        loop_info=CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[sympy.Integer(128)],
            loop_tiled_dims=[[0]],
            tiled_dims_per_read=[[[(0, sympy.Integer(1))]]],
            loop_operand_bindings=[
                LoopOperandBindingRequirement(
                    kind="sequential_affine",
                    host_advance_per_level=(sympy.Integer(64),),
                )
            ],
        ),
        get_operation_name=lambda: "routing_weight_copy",
        get_read_writes=lambda: SimpleNamespace(reads=[dep], writes=[]),
    )
    kernel = SpyreKernel()
    kernel.current_node = SimpleNamespace(node=ir_node)
    tensor = TensorAccess(
        "routing_weight",
        sympy.Integer(0),
        SimpleNamespace(
            device_layout=SimpleNamespace(device_size=[64], stride_map=[1])
        ),
    )

    advance = kernel._general_tile_advance(tensor, True, "routing_weight")
    level = kernel._tile_advance_symbols[0]
    assert coeff_through_floor(advance, level) == 64
