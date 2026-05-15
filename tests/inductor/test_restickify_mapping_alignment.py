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

from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.compute_ops import generate_sdsc
from torch_spyre._inductor.codegen.superdsc import (
    SDSCArgs,
    SDSCSpec,
    _core_mapping_override_for_sdsc,
    _get_core_to_slice_mapping,
)
from torch_spyre._inductor.restickify_ring import (
    CORE_MAPPING_OVERRIDE_OP_INFO_KEY,
    build_symbol_correspondence,
    estimate_byte_hops_from_mappings,
    materialize_default_core_mapping,
    materialize_k_fast_core_mapping,
    producer_aligned_dim_order,
    ring_distance,
)


def test_ring_distance_bidirectional():
    assert ring_distance(0, 31, 32) == 1
    assert ring_distance(0, 16, 32) == 16
    assert ring_distance(7, 3, 32) == ring_distance(3, 7, 32)
    assert ring_distance(0, 5, 8) == 3


def test_materialize_default_core_mapping_matches_superdsc():
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    dim_order = [d0, d1]
    dim_splits = {d0: 4, d1: 2}
    core_id = Symbol("core_id")

    expr_mapping = _get_core_to_slice_mapping(
        {d0: 128, d1: 64},
        dim_splits,
        num_cores=8,
    )
    expected = {
        str(core): {
            str(dim): int(expr.subs({core_id: core}))
            for dim, expr in expr_mapping.items()
        }
        for core in range(8)
    }

    assert materialize_default_core_mapping(dim_order, dim_splits, 8) == expected


def test_materialize_k_fast_core_mapping_moves_reduction_dim_innermost():
    default = materialize_default_core_mapping(
        ["m", "n", "k"],
        {"m": 2, "n": 2, "k": 2},
        8,
    )
    k_fast = materialize_k_fast_core_mapping(
        {"m": 4, "n": 4, "k": 4},
        {"m": 2, "n": 2, "k": 2},
        8,
    )

    assert default["1"] == {"m": 1, "n": 0, "k": 0}
    assert k_fast["1"] == {"k": 1, "m": 0, "n": 0}


def test_estimate_byte_hops_zero_for_aligned_and_positive_for_shifted():
    sizes = {"d0": 4}
    splits = {"d0": 4}
    aligned_mapping = materialize_default_core_mapping(["d0"], splits, 4)
    shifted_mapping = {
        "0": {"d0": 1},
        "1": {"d0": 2},
        "2": {"d0": 3},
        "3": {"d0": 0},
    }

    bytes_moved, byte_hops, max_hops = estimate_byte_hops_from_mappings(
        sizes,
        sizes,
        splits,
        splits,
        aligned_mapping,
        aligned_mapping,
        {"d0": "d0"},
        elem_size_bytes=2,
        ring_size=4,
    )

    assert bytes_moved == 8
    assert byte_hops == 0
    assert max_hops == 0

    bytes_moved, byte_hops, max_hops = estimate_byte_hops_from_mappings(
        sizes,
        sizes,
        splits,
        splits,
        aligned_mapping,
        shifted_mapping,
        {"d0": "d0"},
        elem_size_bytes=2,
        ring_size=4,
    )

    assert bytes_moved == 8
    assert byte_hops == 8
    assert max_hops == 1


def test_symbol_correspondence_skips_ambiguous_strides():
    mapping, reason = build_symbol_correspondence(
        {"p0": 128, "p1": 128},
        {"r0": 128, "r1": 1},
    )

    assert mapping == {}
    assert reason == "ambiguous-producer-stride"


def test_producer_aligned_dim_order_prioritizes_mapped_dominant_split():
    d0 = Symbol("d0")
    d1 = Symbol("d1")

    prioritized, reason = producer_aligned_dim_order(
        [d0, d1],
        {"p0": 1, "p1": 32},
        {"d0": "p0", "d1": "p1"},
    )

    assert prioritized == [d1, d0]
    assert reason is None


def test_producer_aligned_dim_order_skips_ambiguous_dominant_split():
    d0 = Symbol("d0")
    d1 = Symbol("d1")

    prioritized, reason = producer_aligned_dim_order(
        [d0, d1],
        {"p0": 4, "p1": 4},
        {"d0": "p0", "d1": "p1"},
    )

    assert prioritized is None
    assert reason == "ambiguous-producer-split"


def test_core_mapping_override_remaps_to_sdsc_symbols_and_fills_missing_dims():
    i0 = Symbol("i0")
    i1 = Symbol("i1")
    d0 = Symbol("d0")
    d1 = Symbol("d1")

    override = _core_mapping_override_for_sdsc(
        {
            CORE_MAPPING_OVERRIDE_OP_INFO_KEY: {
                "0": {"i0": 1},
                "1": {"i0": 0},
            }
        },
        {i0: d0, i1: d1},
        num_cores=2,
    )

    assert override == {
        "0": {"d0": 1, "d1": 0},
        "1": {"d0": 0, "d1": 0},
    }


def test_generate_sdsc_uses_core_mapping_override():
    d0 = Symbol("d0")
    core_id = Symbol("core_id")
    data_format = DataFormats.SEN169_FP16
    arg = SDSCArgs(
        layout="L0",
        data_format=data_format,
        scales={d0: 1},
        strides={d0: 1},
        offsets={},
        max_dim_sizes={d0: -1},
        allocation={},
        start_address=0,
        backGap={},
    )
    spec = SDSCSpec(
        opfunc="identity",
        execution_unit="sfp",
        data_format=data_format,
        num_inputs=0,
        iteration_space={d0: 4},
        num_cores=2,
        work_slices={d0: 2},
        core_id_to_work_slice={d0: core_id % 2},
        core_id_to_work_slice_override={
            "0": {"d0": 1},
            "1": {"d0": 0},
        },
        padding={},
        layouts={"L0": {"dim_order": [d0], "stick_dim_order": d0, "stick_size": 64}},
        args=[arg],
        constants={},
        coordinate_masking={},
    )

    sdsc = generate_sdsc(0, spec)["0_identity"]

    assert sdsc["coreIdToWkSlice_"] == {
        "0": {"d0": 1},
        "1": {"d0": 0},
    }
