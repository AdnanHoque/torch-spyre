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

from sympy import Integer, Symbol

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
    RestickifyLocalityCertificate,
    RestickifyRingEstimate,
    build_symbol_correspondence,
    estimate_byte_hops_from_mappings,
    materialize_default_core_mapping,
    materialize_k_fast_core_mapping,
    producer_aligned_dim_order,
    ring_distance,
    source_kind_from_buffer,
    _stride_map_from_layout,
)
from torch_spyre._inductor.pass_utils import stick_compatible
from torch_spyre._inductor.core_continuity_telemetry import (
    CoreContinuityEstimate,
    _estimate_to_json as _core_continuity_estimate_to_json,
)
from torch_spyre._inductor.input_fanout_telemetry import (
    InputFanoutEstimate,
    _estimate_to_json as _input_fanout_estimate_to_json,
)
from torch_spyre._inductor.restickify_telemetry import _estimate_to_json


def test_ring_distance_bidirectional():
    assert ring_distance(0, 31, 32) == 1
    assert ring_distance(0, 16, 32) == 16
    assert ring_distance(7, 3, 32) == ring_distance(3, 7, 32)
    assert ring_distance(0, 5, 8) == 3


def test_stick_compatible_accepts_same_stick_variable():
    d0 = Symbol("d0")
    d1 = Symbol("d1")

    assert stick_compatible([[d0, d1], [d0 + 1, d1]])


def test_stick_compatible_accepts_broadcast_stick_when_nonstick_dims_are_safe():
    d0 = Symbol("d0")
    d1 = Symbol("d1")

    assert stick_compatible([[d0, d1], [d0, Integer(0)]])


def test_stick_compatible_rejects_multiple_stick_variables():
    d0 = Symbol("d0")
    d1 = Symbol("d1")

    assert not stick_compatible([[d0, d1], [d1, d0]])


def test_stick_compatible_rejects_stick_variable_used_as_nonstick_dimension():
    d0 = Symbol("d0")
    d1 = Symbol("d1")

    assert not stick_compatible([[d0, d1], [d1, Integer(0)]])


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


def test_source_kind_from_buffer_classifies_graph_inputs_by_name():
    assert (
        source_kind_from_buffer("arg0_1", object(), graph_input_names=["arg0_1"])
        == "graph_input_or_weight"
    )


def test_source_kind_from_buffer_classifies_constants_by_type_name():
    fake_constant = type("SpyreConstantFallback", (), {})()

    assert source_kind_from_buffer("constant0", fake_constant) == "constant_or_extern"


def test_stride_map_from_layout_extracts_device_stride_map():
    class DeviceLayout:
        stride_map = [128, 1, -1]

    class Layout:
        device_layout = DeviceLayout()

    assert _stride_map_from_layout(Layout()) == [128, 1, -1]


def test_restickify_telemetry_json_includes_source_fields():
    estimate = RestickifyRingEstimate(
        restickify_name="buf4",
        producer_name="<none>",
        consumer_names=["buf5"],
        bytes_moved=128,
        byte_hops=0,
        avg_hops=0.0,
        max_hops=0,
        producer_splits={},
        restickify_splits={},
        symbol_map={},
        source_name="arg0_1",
        source_kind="graph_input_or_weight",
        consumer_name="buf5",
        consumer_kind="computed",
        target_stride_map=[64, 1],
        source_stride_map=[1, 64],
        skip_reason="graph-input-or-missing-producer",
    )

    payload = _estimate_to_json(estimate)

    assert payload["source_name"] == "arg0_1"
    assert payload["source_kind"] == "graph_input_or_weight"
    assert payload["consumer"] == "buf5"
    assert payload["consumer_kind"] == "computed"
    assert payload["target_stride_map"] == [64, 1]
    assert payload["source_stride_map"] == [1, 64]


def test_restickify_telemetry_json_includes_locality_certificate_fields():
    estimate = RestickifyRingEstimate(
        restickify_name="buf4",
        producer_name="buf3",
        consumer_names=["buf5"],
        bytes_moved=128,
        byte_hops=0,
        avg_hops=0.0,
        max_hops=0,
        producer_splits={"d1": 32},
        restickify_splits={"d0": 32},
        symbol_map={"d0": "d1"},
        locality_certified=True,
        locality_assertion="passed",
        locality_skip_reason=None,
        certified_byte_hops=0,
        certified_bytes_moved=128,
        certified_max_hops=0,
        certified_core_count=32,
    )

    payload = _estimate_to_json(estimate)

    assert payload["locality_certified"] is True
    assert payload["locality_assertion"] == "passed"
    assert payload["locality_skip_reason"] is None
    assert payload["certified_byte_hops"] == 0
    assert payload["certified_bytes_moved"] == 128
    assert payload["certified_max_hops"] == 0
    assert payload["certified_core_count"] == 32


def test_core_continuity_telemetry_json_includes_edge_fields():
    estimate = CoreContinuityEstimate(
        source_name="buf3",
        producer_name="buf3",
        consumer_name="buf4",
        producer_kind="computed",
        consumer_kind="computed",
        bytes_moved=256,
        byte_hops=512,
        avg_hops=2.0,
        max_hops=4,
        producer_splits={"d1": 32},
        consumer_splits={"d1": 32},
        symbol_map={"d1": "d1"},
        skip_reason=None,
    )

    payload = _core_continuity_estimate_to_json(estimate)

    assert payload["source_name"] == "buf3"
    assert payload["producer"] == "buf3"
    assert payload["consumer"] == "buf4"
    assert payload["producer_kind"] == "computed"
    assert payload["consumer_kind"] == "computed"
    assert payload["byte_hops"] == 512
    assert payload["avg_hops"] == 2.0
    assert payload["max_hops"] == 4
    assert payload["producer_splits"] == {"d1": 32}
    assert payload["consumer_splits"] == {"d1": 32}
    assert payload["symbol_map"] == {"d1": "d1"}


def test_input_fanout_telemetry_json_includes_source_fields():
    estimate = InputFanoutEstimate(
        source_name="arg1_1",
        source_kind="graph_input_or_weight",
        consumer_count=2,
        consumers=["buf2", "buf4"],
        consumer_kinds={"restickify": 1, "computed": 1},
        restickify_consumers=["buf2"],
        restickify_bytes_moved=1024,
        approximate_consumer_bytes=4096,
        source_stride_map=[1, 64],
        target_stride_maps=[[64, 1], [1, 64]],
    )

    payload = _input_fanout_estimate_to_json(estimate)

    assert payload["source_name"] == "arg1_1"
    assert payload["source_kind"] == "graph_input_or_weight"
    assert payload["consumer_count"] == 2
    assert payload["consumers"] == ["buf2", "buf4"]
    assert payload["consumer_kinds"] == {"restickify": 1, "computed": 1}
    assert payload["restickify_consumers"] == ["buf2"]
    assert payload["restickify_bytes_moved"] == 1024
    assert payload["source_stride_map"] == [1, 64]
    assert payload["target_stride_maps"] == [[64, 1], [1, 64]]


def test_locality_certificate_represents_failed_nonzero_byte_hops():
    certificate = RestickifyLocalityCertificate(
        locality_certified=False,
        locality_assertion="failed",
        locality_skip_reason="nonzero-byte-hops",
        certified_byte_hops=8,
        certified_bytes_moved=8,
        certified_max_hops=1,
        certified_core_count=4,
        producer_splits={"d0": 4},
        restickify_splits={"d0": 4},
        symbol_map={"d0": "d0"},
    )

    assert certificate.locality_assertion == "failed"
    assert certificate.locality_skip_reason == "nonzero-byte-hops"
    assert certificate.certified_byte_hops == 8


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
