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
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import FixedLayout
from torch._inductor.utils import sympy_index_symbol

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.loop_info import (
    CountedLoopPlan,
    CoarseTileInfo,
    LoopOperandBindingRequirement,
    LoopStoragePlan,
    copy_op_metadata,
)
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.pass_utils import coeff_through_floor
from torch_spyre._inductor.propagate_layouts import (
    _constrain_persistent_row_layouts,
)
from torch_spyre._inductor import ir as spyre_ir
from torch_spyre._inductor.ir import SpyreEmptyFallback
from torch_spyre._inductor.scratchpad.allocator import (
    _reject_required_loop_lx_relayouts,
    _safe_in_place_parents,
    _validate_required_loop_lx_allocation,
)
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.utils import (
    hoisted_loop_lifetime_end_overrides,
    required_loop_lx_storage_names,
)
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


def test_hoisted_copy_owner_survives_ir_reconstruction():
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
    return coeff_through_floor(advance, level)


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


def test_persistent_group_keeps_row_dimension_off_stick():
    row = sympy_index_symbol("d0")
    hidden = sympy_index_symbol("d1")
    output = FixedLayout(
        torch.device("privateuseone:0"),
        torch.float16,
        [32, 2816],
        [2816, 1],
    )
    output_dep = MemoryDep(
        name="accumulator",
        index=2816 * row + hidden,
        var_names=(row, hidden),
        size=(32, 2816),
    )
    hidden_stick = SpyreTensorLayout([32, 2816], [2816, 1], torch.float16, [0, 1])
    row_stick = SpyreTensorLayout([32, 2816], [2816, 1], torch.float16, [1, 0])
    op = SimpleNamespace(
        loop_info=CoarseTileInfo(
            loop_group_id=(0,),
            loop_count=[128],
            loop_tiled_dims=[[]],
            counted_loop_plan=CountedLoopPlan(
                kind="persistent_dense_expert", trip_count=128
            ),
            work_div_row_dim=0,
        ),
        get_name=lambda: "coarse_tile_fill_accumulator",
        data=SimpleNamespace(),
    )

    constrained = _constrain_persistent_row_layouts(
        op, output, output_dep, [row_stick, hidden_stick]
    )

    assert constrained == [hidden_stick]


def test_ordinary_accumulator_layout_candidates_are_unchanged():
    row = sympy_index_symbol("d0")
    hidden = sympy_index_symbol("d1")
    output = FixedLayout(
        torch.device("privateuseone:0"), torch.float16, [32, 64], [64, 1]
    )
    output_dep = MemoryDep(
        name="ordinary",
        index=64 * row + hidden,
        var_names=(row, hidden),
        size=(32, 64),
    )
    candidates = [
        SpyreTensorLayout([32, 64], [64, 1], torch.float16, [1, 0]),
        SpyreTensorLayout([32, 64], [64, 1], torch.float16, [0, 1]),
    ]
    op = SimpleNamespace(
        loop_info=None,
        get_name=lambda: "ordinary_accumulator",
        data=SimpleNamespace(),
    )

    assert (
        _constrain_persistent_row_layouts(op, output, output_dep, candidates)
        == candidates
    )


def test_hoisted_copy_lifetime_ends_after_its_own_loop_group():
    def op(name, group=(), preheader_for_group=None):
        value = type("Op", (), {"get_name": lambda self: name})()
        value.loop_info = CoarseTileInfo(
            loop_group_id=group,
            loop_count=[2] * len(group),
            loop_tiled_dims=[[] for _ in group],
            preheader_for_group=preheader_for_group,
        )
        if preheader_for_group is not None:
            value.loop_storage_plan = LoopStoragePlan(
                kind="loop_invariant",
                owner_group=preheader_for_group,
                execution_role="input_activation",
            )
        return value

    graph = type("Graph", (), {})()
    graph.operations = [
        op("x_copy", preheader_for_group=(2,)),
        op("first", (2,)),
        op("nested", (2, 0)),
        op("later", (3,)),
    ]

    assert hoisted_loop_lifetime_end_overrides(graph) == {"x_copy": 3}


def test_typed_loop_storage_is_required_in_lx():
    def op(name):
        return type("Op", (), {"get_name": lambda self: name})()

    x_copy = op("x_copy")
    x_copy.loop_info = CoarseTileInfo(
        loop_group_id=(),
        loop_count=[],
        loop_tiled_dims=[],
        preheader_for_group=(2,),
        counted_loop_plan=CountedLoopPlan(
            kind="persistent_dense_expert", trip_count=128
        ),
    )
    x_copy.loop_storage_plan = LoopStoragePlan(
        kind="loop_invariant",
        owner_group=(2,),
        execution_role="input_activation",
    )
    accumulator = op("accumulator")
    accumulator.loop_storage_plan = LoopStoragePlan(
        kind="loop_carried_accumulator",
        owner_group=(2,),
        execution_role="output_accumulator",
    )
    body = op("body")
    body.loop_info = CoarseTileInfo(
        loop_group_id=(2,),
        loop_count=[128],
        loop_tiled_dims=[[]],
        counted_loop_plan=CountedLoopPlan(
            kind="persistent_dense_expert",
            trip_count=128,
            body_memory_kind="lx",
        ),
    )
    graph = SimpleNamespace(operations=[x_copy, accumulator, body])

    assert required_loop_lx_storage_names(graph) == frozenset({"x_copy", "accumulator"})


def test_only_typed_loop_empty_skips_wrapper_allocation_for_lx(monkeypatch):
    class Layout:
        def __init__(self, allocation):
            self.allocation = allocation

    monkeypatch.setattr(spyre_ir, "FixedTiledLayout", Layout)

    ordinary = SimpleNamespace(
        get_layout=lambda: Layout({"lx": 0}),
        loop_storage_plan=None,
    )
    persistent = SimpleNamespace(
        get_layout=lambda: Layout({"lx": 0}),
        loop_storage_plan=LoopStoragePlan(
            kind="loop_carried_accumulator",
            owner_group=(2,),
            execution_role="output_accumulator",
        ),
    )

    assert SpyreEmptyFallback.should_allocate(ordinary)
    assert not SpyreEmptyFallback.should_allocate(persistent)


def test_required_loop_storage_rejects_relayout_and_spill():
    plan = SimpleNamespace(source_name="x_copy", destination_name="x_relayout")
    try:
        _reject_required_loop_lx_relayouts([plan], frozenset({"x_copy"}))
    except Unsupported:
        pass
    else:
        raise AssertionError("persistent loop storage relayout must fail closed")

    spilled = LifetimeBoundBuffer("x_copy", 128, [0, 1])
    try:
        _validate_required_loop_lx_allocation(frozenset({"x_copy"}), [spilled])
    except Unsupported:
        pass
    else:
        raise AssertionError("required persistent loop storage may not spill")

    spilled.address = 0
    _validate_required_loop_lx_allocation(frozenset({"x_copy"}), [spilled])


def test_loop_lifetime_disables_in_place_handoff():
    assert _safe_in_place_parents(["x_copy", "ordinary"], {"x_copy": 8}) == ["ordinary"]
