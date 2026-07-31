# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
from sympy import Integer, Mod, Symbol, floor
from torch._inductor.utils import IndentedBuffer

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.compute_ops import generate_sdsc
from torch_spyre._inductor.codegen.superdsc import (
    SDSCArgs,
    SDSCSpec,
    _get_core_to_slice_mapping,
    _select_core_to_slice_mapping,
)
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.lx_relayout import (
    LXRelayoutPlan,
    _core_id_to_device_slice,
    _destination_size_ratio,
    _same_core_placement,
)
from torch_spyre._inductor.op_spec import (
    OpSpec,
    TensorArg,
    WORK_DIV_INNER_FIRST,
)
from torch_spyre._inductor.propagate_hints import (
    get_physical_core_ids,
    get_physical_core_order,
)
from torch_spyre._inductor.pass_utils import (
    PerCoreView,
    _canonical_physical_core_ids,
)
from torch_spyre._inductor.spyre_kernel import (
    _codegen_op_spec_list,
    _materialize_explicit_lx_shuffle,
)


def _value(expr, core_id: int) -> int:
    return int(expr.subs(Symbol("core_id"), Integer(core_id)))


def _hinted_op(value=...):
    custom = {}
    if value is not ...:
        custom["_hint_7"] = {"physical_core_order": value}
    origin = SimpleNamespace(meta={"custom": custom})
    return SimpleNamespace(origins=[origin])


def _core_ids_hinted_op(value):
    custom = {"_hint_7": {"physical_core_ids": value}}
    origin = SimpleNamespace(meta={"custom": custom})
    return SimpleNamespace(origins=[origin])


def _consumer_and_plan(order):
    h = Symbol("h")
    lq = Symbol("lq")
    lk = Symbol("lk")
    d = Symbol("d")
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 4096, 2, 64],
        device_coordinates=[h, lk, floor(d / 64), Mod(d, 64)],
        allocation={"lx": 0x24000},
        name="buf_k",
    )
    consumer = OpSpec(
        op=BATCH_MATMUL_OP,
        is_reduction=True,
        iteration_space={
            h: (Integer(4), 4),
            lq: (Integer(512), 8),
            lk: (Integer(4096), 1),
            d: (Integer(128), 1),
        },
        args=[source_arg],
        op_info={},
        physical_core_order=order,
    )
    producer_map = {str(core): {"0": core % 4, "1": core // 4} for core in range(32)}
    consumer_map = {str(core): {"0": core % 4, "1": 0} for core in range(32)}
    plan = LXRelayoutPlan(
        source_name="buf_k",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 4, "1": 1},
        destination_size_ratio=8,
        destination_lx_address=0x44000,
    )
    return source_arg, consumer, plan


def test_absent_hint_is_production_default():
    assert get_physical_core_order(_hinted_op()) is None
    h, lane = Symbol("h"), Symbol("lane")
    iteration_space = {h: Integer(4), lane: Integer(8)}
    splits = {h: 4, lane: 8}
    assert _select_core_to_slice_mapping(
        False, iteration_space, splits, 32, None
    ) == _get_core_to_slice_mapping(iteration_space, splits, 32)


def test_hint_is_typed_and_unknown_value_fails_closed():
    assert get_physical_core_order(_hinted_op(WORK_DIV_INNER_FIRST)) == (
        WORK_DIV_INNER_FIRST
    )
    with pytest.raises(Unsupported, match="physical_core_order must be one of"):
        get_physical_core_order(_hinted_op("global_reverse_everything"))
    with pytest.raises(ValueError, match="physical_core_order must be one of"):
        _select_core_to_slice_mapping(
            False,
            {Symbol("h"): Integer(4)},
            {Symbol("h"): 4},
            4,
            "global_reverse_everything",  # type: ignore[arg-type]
        )


def test_sparse_physical_core_ids_are_typed_and_fail_closed():
    sparse = tuple(range(0, 32, 4))
    assert get_physical_core_ids(_core_ids_hinted_op(list(sparse))) == sparse
    with pytest.raises(Unsupported, match="invalid physical_core_ids"):
        get_physical_core_ids(_core_ids_hinted_op([0, 4, 4]))
    with pytest.raises(Unsupported, match="invalid physical_core_ids"):
        get_physical_core_ids(_core_ids_hinted_op([0, 32]))


def test_sparse_ids_map_logical_slices_to_physical_cores():
    core_id = Symbol("core_id")
    sparse = tuple(range(0, 32, 4))
    view = PerCoreView(
        work_slice_dims=((0, 8),),
        core_to_slot=((0, Mod(core_id, 8)),),
        physical_core_ids=sparse,
    )
    mapping = _core_id_to_device_slice(view, 8)
    assert mapping is not None
    assert list(map(int, mapping)) == list(sparse)
    assert [mapping[str(core)]["0"] for core in sparse] == list(range(8))
    assert _canonical_physical_core_ids(tuple(range(8)), 8) == ()
    assert _canonical_physical_core_ids(sparse, 8) == sparse


def test_uniform_one_to_many_geometry_is_a_bounded_broadcast():
    ratio = _destination_size_ratio(
        {"0": {"0": 0}, "4": {"0": 1}},
        {"0": 2},
        {
            "0": {"0": 0},
            "1": {"0": 0},
            "4": {"0": 1},
            "5": {"0": 1},
        },
        {"0": 2},
    )
    assert ratio == 1


def test_equal_logical_views_on_different_cores_are_not_same_placement():
    view = PerCoreView(work_slice_dims=((0, 8),), core_to_slot=())
    assert _same_core_placement(view, 8, view, 8)
    assert not _same_core_placement(view, 8, view, 32)


def test_sparse_ids_serialize_through_existing_sdsc_contract():
    h = Symbol("h")
    core_id = Symbol("core_id")
    sparse = (0, 8, 16, 24)
    tensor = SDSCArgs(
        layout="A",
        dim_order=[h],
        data_format=DataFormats.SEN169_FP16,
        scales={h: 1},
        strides={h: 64},
        offsets={h: 0},
        max_dim_sizes={h: -1},
        allocation={"hbm": 0x1000},
        start_address=0x1000,
        backGap={},
        arg_index=0,
    )
    spec = SDSCSpec(
        opfunc="mul",
        execution_unit="sfp",
        data_format=DataFormats.SEN169_FP16,
        num_inputs=0,
        iteration_space={h: 256},
        num_cores=4,
        work_slices={h: 4},
        core_id_to_work_slice={h: Mod(core_id, 4)},
        padding={},
        layouts={"A": {"dim_order": [h], "stick_dim_order": h, "stick_size": 64}},
        args=[tensor],
        constants={},
        coordinate_masking={},
        physical_core_ids=sparse,
    )
    generated, _, _, _ = generate_sdsc(0, spec, [], use_symbols=False)
    root = generated["0_mul"]
    body = root["dscs_"][0]["mul"]
    assert root["coreFoldProp_"]["factor_"] == 32
    assert root["numCoresUsed_"] == 4
    assert list(map(int, root["coreIdToDsc_"])) == list(sparse)
    assert list(map(int, root["coreIdToWkSlice_"])) == list(sparse)
    assert body["coreIdsUsed_"] == list(sparse)
    start_addresses = body["scheduleTree_"][0]["startAddressCoreCorelet_"]
    assert start_addresses["dim_prop_attr"][0]["factor_"] == 32
    assert len(start_addresses["data_"]) == 32

    dense, _, _, _ = generate_sdsc(
        0, replace(spec, physical_core_ids=None), [], use_symbols=False
    )
    dense_root = dense["0_mul"]
    dense_body = dense_root["dscs_"][0]["mul"]
    assert dense_root["coreFoldProp_"]["factor_"] == 4
    assert list(map(int, dense_root["coreIdToDsc_"])) == [0, 1, 2, 3]
    assert dense_body["coreIdsUsed_"] == [0, 1, 2, 3]
    dense_starts = dense_body["scheduleTree_"][0]["startAddressCoreCorelet_"]
    assert dense_starts["dim_prop_attr"][0]["factor_"] == 4
    assert len(dense_starts["data_"]) == 4


def test_inner_first_places_each_head_cohort_on_adjacent_cores():
    h, lane = Symbol("h"), Symbol("lane")
    mapping = _select_core_to_slice_mapping(
        False,
        {h: Integer(4), lane: Integer(8)},
        {h: 4, lane: 8},
        32,
        WORK_DIV_INNER_FIRST,
    )
    assert [_value(mapping["h"], core) for core in range(32)] == [
        head for head in range(4) for _ in range(8)
    ]
    assert [_value(mapping["lane"], core) for core in range(32)] == list(range(8)) * 4


def test_inner_first_makes_shuffle_replicas_fastest():
    h = Symbol("h")
    mapping = _select_core_to_slice_mapping(
        False,
        {h: Integer(4)},
        {h: 4},
        32,
        WORK_DIV_INNER_FIRST,
    )
    assert [_value(mapping["h"], core) for core in range(32)] == [
        head for head in range(4) for _ in range(8)
    ]


@pytest.mark.parametrize("order", [None, WORK_DIV_INNER_FIRST])
def test_matching_shuffle_inherits_only_consumer_contract(order):
    source_arg, consumer, plan = _consumer_and_plan(order)
    result = _materialize_explicit_lx_shuffle(source_arg, consumer, plan)
    assert result is not None
    shuffle, _ = result
    assert shuffle.physical_core_order == order


def test_generated_default_opspec_omits_contract_field():
    h = Symbol("h")
    default = OpSpec("mul", False, {h: (Integer(4), 4)}, [], {})
    targeted = OpSpec(
        "mul",
        False,
        {h: (Integer(4), 4)},
        [],
        {},
        physical_core_order=WORK_DIV_INNER_FIRST,
        physical_core_ids=(0, 4, 8, 12),
    )

    def render(spec):
        buf = IndentedBuffer()
        _codegen_op_spec_list([spec], buf, lambda value: f"sympify('{value}')")
        return buf.getvalue()

    assert "physical_core_order" not in render(default)
    assert "physical_core_ids" not in render(default)
    assert "physical_core_order='work_div_inner_first'" in render(targeted)
    assert "physical_core_ids=(0, 4, 8, 12)" in render(targeted)
