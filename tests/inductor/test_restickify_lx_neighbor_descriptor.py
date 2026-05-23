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

import json

from sympy import Symbol

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.lx_neighbor_descriptor import (
    BRIDGE_CANDIDATE_FILENAME_TEMPLATE,
    DESCRIPTOR_FILENAME,
    LOCALITY_CERTIFICATE_OP_INFO_KEY,
    build_lx_neighbor_descriptor,
    maybe_emit_lx_neighbor_descriptor,
)
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.restickify_ring import CORE_MAPPING_OVERRIDE_OP_INFO_KEY


def _arg(
    is_input: bool,
    *,
    arg_index: int = -1,
    device_size: list[int] | None = None,
) -> TensorArg:
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=device_size or [2048, 2048],
        device_coordinates=[d0, d1],
        allocation={},
    )


def _op(op: str, op_info=None, args=None) -> OpSpec:
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    return OpSpec(
        op=op,
        is_reduction=False,
        iteration_space={d0: (2048, 32), d1: (2048, 1)},
        args=args or [_arg(True), _arg(False)],
        op_info=op_info or {},
    )


def _op_with_args(op: str, args: list[TensorArg], op_info=None) -> OpSpec:
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    return OpSpec(
        op=op,
        is_reduction=False,
        iteration_space={d0: (512, 32), d1: (512, 1)},
        args=args,
        op_info=op_info or {},
    )


def _files() -> list[str]:
    return [
        "sdsc_0_add.json",
        "sdsc_1_ReStickifyOpHBM.json",
        "sdsc_2_add.json",
    ]


def _sdsc_payload(
    name: str,
    opfunc: str,
    *,
    num_work_slices: dict | None = None,
    core_mapping: dict | None = None,
    input_labels: list[str] | None = None,
    output_labels: list[str] | None = None,
    primary_ds_info: dict | None = None,
    labeled_ds: list[dict] | None = None,
) -> dict:
    return {
        name: {
            "numCoresUsed_": 2,
            "numWkSlicesPerDim_": num_work_slices or {"d0": 2},
            "coreIdToWkSlice_": core_mapping or {
                "0": {"d0": 0},
                "1": {"d0": 1},
            },
            "dscs_": [
                {
                    opfunc: {
                        "numCoresUsed_": 2,
                        "primaryDsInfo_": primary_ds_info
                        or {
                            "INPUT": {
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [64],
                            },
                            "OUTPUT": {
                                "layoutDimOrder_": ["out", "mb"],
                                "stickDimOrder_": ["mb"],
                                "stickSize_": [64],
                            },
                            "KERNEL": {
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "stickSize_": [64],
                            },
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": "allocate-Tensor0_lx",
                                "ldsIdx_": 0,
                                "component_": "lx",
                                "layoutDimOrder_": ["mb", "out"],
                                "maxDimSizes_": [2048, 2048],
                                "startAddressCoreCorelet_": {
                                    "data_": {
                                        "[0, 0, 0]": "1024",
                                        "[1, 0, 0]": "1024",
                                    }
                                },
                            }
                        ],
                        "labeledDs_": labeled_ds
                        or [
                            {
                                "ldsIdx_": 0,
                                "dsName_": "Tensor0",
                                "dsType_": "INPUT",
                                "scale_": [1, 1],
                                "wordLength": 2,
                                "dataFormat_": "SEN169_FP16",
                                "memOrg_": {"lx": {"isPresent": 1}},
                            },
                            {
                                "ldsIdx_": 1,
                                "dsName_": "Tensor1",
                                "dsType_": "OUTPUT",
                                "scale_": [1, 1],
                                "wordLength": 2,
                                "dataFormat_": "SEN169_FP16",
                                "memOrg_": {"lx": {"isPresent": 1}},
                            },
                        ],
                        "computeOp_": [
                            {
                                "exUnit": "sfp",
                                "opFuncName": opfunc,
                                "inputLabeledDs": input_labels or ["Tensor0-idx0"],
                                "outputLabeledDs": output_labels
                                or ["Tensor1-idx1"],
                            }
                        ],
                    }
                }
            ],
        }
    }


def _payloads() -> list[dict]:
    return [
        _sdsc_payload("0_add", "add"),
        _sdsc_payload(
            "1_ReStickifyOpHBM",
            "ReStickifyOpHBM",
            input_labels=["Tensor0-idx0"],
            output_labels=["Tensor1-idx1"],
            labeled_ds=[
                {
                    "ldsIdx_": 0,
                    "dsName_": "Tensor0",
                    "dsType_": "OUTPUT",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
                {
                    "ldsIdx_": 1,
                    "dsName_": "Tensor1",
                    "dsType_": "KERNEL",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
            ],
        ),
        _sdsc_payload("2_add", "add"),
    ]


def _row_to_col_payloads() -> list[dict]:
    producer_mapping = {
        str(core): {"mb": core, "out": 0}
        for core in range(32)
    }
    consumer_mapping = {
        str(core): {"mb": 0, "out": core}
        for core in range(32)
    }
    return [
        _sdsc_payload(
            "0_add",
            "add",
            num_work_slices={"mb": 32, "out": 1},
            core_mapping=producer_mapping,
            primary_ds_info={
                "INPUT": {
                    "layoutDimOrder_": ["mb", "out"],
                    "stickDimOrder_": ["out"],
                    "stickSize_": [64],
                },
                "OUTPUT": {
                    "layoutDimOrder_": ["mb", "out"],
                    "stickDimOrder_": ["out"],
                    "stickSize_": [64],
                },
                "KERNEL": {
                    "layoutDimOrder_": ["mb", "out"],
                    "stickDimOrder_": ["out"],
                    "stickSize_": [64],
                },
            },
        ),
        _sdsc_payload(
            "1_ReStickifyOpHBM",
            "ReStickifyOpHBM",
            num_work_slices={"mb": 1, "out": 32},
            core_mapping=consumer_mapping,
            input_labels=["Tensor0-idx0"],
            output_labels=["Tensor1-idx1"],
            labeled_ds=[
                {
                    "ldsIdx_": 0,
                    "dsName_": "Tensor0",
                    "dsType_": "OUTPUT",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
                {
                    "ldsIdx_": 1,
                    "dsName_": "Tensor1",
                    "dsType_": "KERNEL",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
            ],
        ),
        _sdsc_payload(
            "2_add",
            "add",
            num_work_slices={"mb": 1, "out": 32},
            core_mapping=consumer_mapping,
        ),
    ]


def _candidate_specs() -> list[OpSpec]:
    return [
        _op("add"),
        _op(
            RESTICKIFY_OP,
            op_info={
                "restickify_source_name": "buf0",
                "restickify_source_kind": "in_graph_computed",
                CORE_MAPPING_OVERRIDE_OP_INFO_KEY: {
                    "0": {"d0": 0, "d1": 0},
                    "1": {"d0": 1, "d1": 0},
                },
                LOCALITY_CERTIFICATE_OP_INFO_KEY: {
                    "locality_certified": True,
                    "certified_byte_hops": 0,
                },
            },
        ),
        _op("add"),
    ]


def _coordinate_changing_specs() -> list[OpSpec]:
    c0 = Symbol("c0")
    c1 = Symbol("c1")
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    producer_output = TensorArg(
        is_input=False,
        arg_index=3,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 512, 64],
        device_coordinates=[c1 // 64, c0, c1 % 64],
        allocation={},
    )
    restickify_input = TensorArg(
        is_input=True,
        arg_index=3,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 512, 64],
        device_coordinates=[d0 // 64, d1, d0 % 64],
        allocation={},
    )
    restickify_output = TensorArg(
        is_input=False,
        arg_index=4,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 512, 64],
        device_coordinates=[d1 // 64, d0, d1 % 64],
        allocation={},
    )
    consumer_input = TensorArg(
        is_input=True,
        arg_index=4,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 512, 64],
        device_coordinates=[c1 // 64, c0, c1 % 64],
        allocation={},
    )
    return [
        _op_with_args("add", [_arg(True), _arg(True), producer_output]),
        _op_with_args(
            RESTICKIFY_OP,
            [restickify_input, restickify_output],
            op_info={
                "restickify_source_name": "buf0",
                "restickify_source_kind": "in_graph_computed",
                CORE_MAPPING_OVERRIDE_OP_INFO_KEY: {
                    "0": {"d0": 0, "d1": 0},
                    "1": {"d0": 1, "d1": 0},
                },
                LOCALITY_CERTIFICATE_OP_INFO_KEY: {
                    "locality_certified": True,
                    "certified_byte_hops": 0,
                },
            },
        ),
        _op_with_args("add", [_arg(True), consumer_input, _arg(False)]),
    ]


def test_builds_candidate_descriptor_for_adjacent_certified_restickify():
    descriptor = build_lx_neighbor_descriptor(
        "sdsc_fused_add",
        _files(),
        _candidate_specs(),
    )

    assert descriptor["schema_version"] == 5
    assert descriptor["kind"] == "torch_spyre.restickify_lx_neighbor_edges"
    assert descriptor["skipped"] == []
    assert len(descriptor["edges"]) == 1

    edge = descriptor["edges"][0]
    assert edge["producer"]["file"] == "sdsc_0_add.json"
    assert edge["restickify"]["file"] == "sdsc_1_ReStickifyOpHBM.json"
    assert edge["consumer"]["file"] == "sdsc_2_add.json"
    assert edge["same_bundle_internal_edge"] is True
    assert edge["source_kind"] == "in_graph_computed"
    assert edge["locality_certificate"]["certified_byte_hops"] == 0
    assert (
        edge["input_fetch_neighbor"]["path"]
        == "producer-output-lx-to-consumer-input-lx"
    )
    assert edge["input_fetch_neighbor"]["requires_single_runtime_bundle"] is True
    assert edge["packaging_requirements"]["preserve_producer_lx_core_state"]
    contract = edge["source_view_contract"]
    assert contract["producer_physical_output"]["device_coordinates"] == ["d0", "d1"]
    assert contract["restickify_logical_source_view"]["device_size"] == [2048, 2048]
    assert (
        contract["coordinate_relations"]["producer_output_to_restickify_input"][
            "same_coordinate_strings"
        ]
        == [True, True]
    )
    endpoint_contract = edge["lx_endpoint_contract"]
    assert endpoint_contract["kind"] == "torch_spyre.restickify_lx_endpoint_contract"
    assert endpoint_contract["memory_space"] == "lx"
    assert endpoint_contract["post_hoc_hbm_alias_allowed"] is False
    assert endpoint_contract["requires_deeptools_contract_work"] is True
    assert endpoint_contract["candidate_deeptools_ops"] == [
        "ReStickifyOpLx",
        "STCDPOpLx",
    ]
    assert set(endpoint_contract["endpoints"]) == {
        "producer_lx_source",
        "restickify_lx_input",
        "restickify_lx_output",
        "consumer_lx_sink",
    }
    assert endpoint_contract["endpoints"]["producer_lx_source"]["sdsc_index"] == 0
    assert endpoint_contract["endpoints"]["consumer_lx_sink"]["sdsc_index"] == 2
    assert endpoint_contract["requirements"][
        "preserve_producer_lx_allocation_identity"
    ]
    assert endpoint_contract["requirements"][
        "single_runtime_lifetime_or_explicit_cross_bundle_handoff"
    ]
    materialization_contract = edge["lx_materialization_contract"]
    assert (
        materialization_contract["kind"]
        == "torch_spyre.restickify_lx_materialization_contract"
    )
    assert materialization_contract["memory_space"] == "lx"
    assert materialization_contract["source_role"] == "producer_physical_output"
    assert (
        materialization_contract["destination_role"]
        == "consumer_restickified_input"
    )
    assert materialization_contract["intended_deeptools_sequence"] == [
        "ReStickifyOpLx",
        "STCDPOpLx",
    ]
    assert (
        materialization_contract["requires_producer_primary_to_match_bridge_input"]
        is False
    )
    assert materialization_contract["requires_remote_lx_materialization"] is True


def test_includes_sdsc_contract_when_payloads_are_provided():
    descriptor = build_lx_neighbor_descriptor(
        "sdsc_fused_add",
        _files(),
        _candidate_specs(),
        sdsc_payloads=_payloads(),
    )

    edge = descriptor["edges"][0]
    contract = edge["sdsc_contract"]
    assert contract["producer"]["opfunc"] == "add"
    assert contract["restickify"]["opfunc"] == "ReStickifyOpHBM"
    assert contract["consumer"]["opfunc"] == "add"
    assert contract["producer_output_role"]["lds_idx"] == 1
    assert contract["producer_output_role"]["ds_type"] == "OUTPUT"
    assert contract["restickify_edge_roles"]["source_ds_type"] == "OUTPUT"
    assert contract["restickify_edge_roles"]["destination_ds_type"] == "KERNEL"
    assert contract["consumer_input_role"]["lds_idx"] == 0
    assert contract["consumer_input_role"]["ds_type"] == "INPUT"
    assert contract["restickify_edge_roles"]["source_primary"][
        "layoutDimOrder_"
    ] == ["out", "mb"]
    assert contract["restickify_edge_roles"]["destination_primary"][
        "layoutDimOrder_"
    ] == ["mb", "out"]
    assert contract["restickify"]["allocate_nodes"][0]["component"] == "lx"
    endpoint_contract = edge["lx_endpoint_contract"]
    assert endpoint_contract["endpoints"]["producer_lx_source"]["sdsc_endpoint"][
        "lds_idx"
    ] == 1
    assert endpoint_contract["endpoints"]["restickify_lx_input"]["sdsc_endpoint"][
        "lds_idx"
    ] == 0
    assert endpoint_contract["endpoints"]["restickify_lx_output"]["sdsc_endpoint"][
        "lds_idx"
    ] == 1
    assert endpoint_contract["endpoints"]["consumer_lx_sink"]["sdsc_endpoint"][
        "lds_idx"
    ] == 0
    materialization_contract = edge["lx_materialization_contract"]
    assert materialization_contract["sdsc_endpoints"]["producer_source"][
        "lds_idx"
    ] == 1
    assert materialization_contract["sdsc_endpoints"]["restickify_source"][
        "lds_idx"
    ] == 0
    assert materialization_contract["sdsc_endpoints"]["restickify_destination"][
        "lds_idx"
    ] == 1
    assert materialization_contract["sdsc_endpoints"]["consumer_sink"][
        "lds_idx"
    ] == 0
    assert materialization_contract["streaming_ptlx"]["available"] is False
    assert (
        materialization_contract["streaming_ptlx"]["reason"]
        == "producer-work-slices-missing-layout-dims"
    )


def test_materialization_contract_includes_streaming_tile_plan_for_row_to_col():
    descriptor = build_lx_neighbor_descriptor(
        "sdsc_fused_add",
        _files(),
        _candidate_specs(),
        sdsc_payloads=_row_to_col_payloads(),
    )

    streaming = descriptor["edges"][0]["lx_materialization_contract"][
        "streaming_ptlx"
    ]

    assert streaming["available"] is True
    assert streaming["row_dim"] == "mb"
    assert streaming["col_dim"] == "out"
    assert streaming["producer_work_slices"] == {"mb": 32, "out": 1}
    assert streaming["destination_work_slices"] == {"mb": 1, "out": 32}
    assert streaming["destination_sdsc"] == "2_add"
    assert streaming["destination_role"] == "consumer_sink_contract"
    assert streaming["bounded_workspace_ok"] is True
    assert streaming["requires_remote_lx_gather"] is True
    assert streaming["requires_remote_lx_scatter"] is False
    assert streaming["contract"]["tile_size"] == 64
    assert streaming["contract"]["bounded_workspace_bytes"] == 24576
    assert streaming["summary"]["size"] == 2048
    assert streaming["summary"]["total_tiles"] == 1024
    assert streaming["summary"]["local_tiles"] == 32
    assert streaming["summary"]["moving_tiles"] == 992
    assert streaming["summary"]["tile_buffer_bytes"] == 8192
    assert streaming["summary"]["total_byte_hops"] > 0
    assert streaming["summary"]["sample_tiles"][0]["tile_row"] == 0
    assert streaming["summary"]["sample_tiles"][0]["tile_col"] == 0
    assert streaming["summary"]["sample_tiles"][1]["source_cores"] == [0]
    assert streaming["summary"]["sample_tiles"][1]["dest_cores"] == [1]
    assert streaming["producer_core_mapping_sample"]["entries"]["0"] == {
        "mb": 0,
        "out": 0,
    }
    assert streaming["destination_core_mapping_sample"]["entries"]["1"] == {
        "mb": 0,
        "out": 1,
    }


def test_streaming_tile_plan_accepts_tiled_3d_restickify_shape():
    specs = _candidate_specs()
    specs[1].args = [
        _arg(True, device_size=[32, 2048, 64]),
        _arg(False, device_size=[32, 2048, 64]),
    ]

    descriptor = build_lx_neighbor_descriptor(
        "sdsc_fused_add",
        _files(),
        specs,
        sdsc_payloads=_row_to_col_payloads(),
    )

    streaming = descriptor["edges"][0]["lx_materialization_contract"][
        "streaming_ptlx"
    ]

    assert streaming["available"] is True
    assert streaming["summary"]["size"] == 2048
    assert streaming["summary"]["total_tiles"] == 1024


def test_skips_graph_input_sources():
    specs = _candidate_specs()
    specs[1].op_info["restickify_source_kind"] = "graph_input_or_weight"

    descriptor = build_lx_neighbor_descriptor("k", _files(), specs)

    assert descriptor["edges"] == []
    assert descriptor["skipped"][0]["reason"] == "source-kind-graph_input_or_weight"


def test_skips_without_core_mapping_override():
    specs = _candidate_specs()
    del specs[1].op_info[CORE_MAPPING_OVERRIDE_OP_INFO_KEY]

    descriptor = build_lx_neighbor_descriptor("k", _files(), specs)

    assert descriptor["edges"] == []
    assert descriptor["skipped"][0]["reason"] == "missing-producer-aligned-core-mapping"


def test_skips_without_locality_certificate():
    specs = _candidate_specs()
    del specs[1].op_info[LOCALITY_CERTIFICATE_OP_INFO_KEY]

    descriptor = build_lx_neighbor_descriptor("k", _files(), specs)

    assert descriptor["edges"] == []
    assert descriptor["skipped"][0]["reason"] == "missing-locality-certificate"


def test_skips_when_locality_certificate_failed():
    specs = _candidate_specs()
    specs[1].op_info[LOCALITY_CERTIFICATE_OP_INFO_KEY] = {
        "locality_certified": False,
        "locality_skip_reason": "nonzero-byte-hops",
    }

    descriptor = build_lx_neighbor_descriptor("k", _files(), specs)

    assert descriptor["edges"] == []
    assert descriptor["skipped"][0]["reason"] == "locality-not-certified"


def test_maybe_emit_descriptor_writes_sidecar_when_flag_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "restickify_lx_neighbor_descriptor", True)

    maybe_emit_lx_neighbor_descriptor(
        "sdsc_fused_add",
        str(tmp_path),
        _files(),
        _candidate_specs(),
    )

    descriptor_path = tmp_path / DESCRIPTOR_FILENAME
    assert descriptor_path.exists()
    payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
    assert len(payload["edges"]) == 1


def test_maybe_emit_streaming_bridge_candidate_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "restickify_lx_neighbor_descriptor", True)
    monkeypatch.setattr(config, "restickify_lx_neighbor_streaming_bridge", True)
    specs = _candidate_specs()
    for spec in specs:
        for arg in spec.args:
            arg.device_size = [512, 512]
    producer_mapping = {
        str(core): {"mb": core, "out": 0}
        for core in range(32)
    }
    destination_mapping = {
        str(core): {"mb": core % 4, "out": core // 4}
        for core in range(32)
    }
    payloads = [
        _sdsc_payload(
            "0_add",
            "add",
            num_work_slices={"mb": 32, "out": 1},
            core_mapping=producer_mapping,
            primary_ds_info={
                "INPUT": {
                    "layoutDimOrder_": ["mb", "out"],
                    "stickDimOrder_": ["out"],
                    "stickSize_": [64],
                },
                "OUTPUT": {
                    "layoutDimOrder_": ["mb", "out"],
                    "stickDimOrder_": ["out"],
                    "stickSize_": [64],
                },
                "KERNEL": {
                    "layoutDimOrder_": ["mb", "out"],
                    "stickDimOrder_": ["out"],
                    "stickSize_": [64],
                },
            },
        ),
        _sdsc_payload(
            "1_ReStickifyOpHBM",
            "ReStickifyOpHBM",
            num_work_slices={"mb": 4, "out": 8},
            core_mapping=destination_mapping,
            input_labels=["Tensor0-idx0"],
            output_labels=["Tensor1-idx1"],
            labeled_ds=[
                {
                    "ldsIdx_": 0,
                    "dsName_": "Tensor0",
                    "dsType_": "OUTPUT",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
                {
                    "ldsIdx_": 1,
                    "dsName_": "Tensor1",
                    "dsType_": "KERNEL",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
            ],
        ),
        _sdsc_payload(
            "2_add",
            "add",
            num_work_slices={"mb": 4, "out": 8},
            core_mapping=destination_mapping,
        ),
    ]

    descriptor = maybe_emit_lx_neighbor_descriptor(
        "sdsc_fused_add",
        str(tmp_path),
        _files(),
        specs,
        sdsc_payloads=payloads,
    )

    assert descriptor is not None
    candidate = descriptor["streaming_bridge_candidates"][0]
    assert candidate["status"] == "emitted"
    assert candidate["bundle_mlir_unchanged"] is True
    assert candidate["executable_in_bundle"] is False
    assert candidate["fallback"] == "ReStickifyOpHBM"
    assert candidate["bridge_lowering"] == "same-layout-lx-ownership-remap"
    assert candidate["size"] == 512
    assert candidate["total_tiles"] == 64
    assert candidate["tile_records_materialized"] == 64
    assert candidate["streaming_summary"]["max_fan_in"] == 4
    assert candidate["streaming_summary"]["max_fan_out"] == 1
    assert candidate["bridge_kind"] == "same-layout-lx-ownership-remap"
    assert candidate["bridge_endpoint_contract_valid"] is True
    assert candidate["production_valid"] is True
    assert candidate["production_blocker"] is None
    production = candidate["production_contract"]
    assert production["semantic_transform_certified"] is True
    assert production["tile_contract"]["all_tiles_materialized"] is True
    assert production["tile_contract"]["max_fan_in"] == 4
    assert production["required_primitive"] is None
    bridge_path = tmp_path / BRIDGE_CANDIDATE_FILENAME_TEMPLATE.format(idx=1)
    assert bridge_path.exists()
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    root = next(iter(bridge.values()))
    assert root["streamingLXRemapFull_"]["fallback"] == "ReStickifyOpHBM"
    assert root["streamingLXRemapFull_"]["coalescing"] == (
        "same-layout-lx-ownership-remap-64x64-tiles"
    )
    assert "ReStickifyOpHBM" not in candidate["op_funcs_used"]
    assert set(candidate["op_funcs_used"]) == {"STCDPOpLx"}


def test_streaming_bridge_does_not_treat_coordinate_change_as_same_layout_remap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(config, "restickify_lx_neighbor_descriptor", True)
    monkeypatch.setattr(config, "restickify_lx_neighbor_streaming_bridge", True)
    specs = _coordinate_changing_specs()
    producer_mapping = {str(core): {"mb": core, "out": 0} for core in range(32)}
    destination_mapping = {
        str(core): {"mb": core % 4, "out": core // 4} for core in range(32)
    }
    same_primary = {
        "INPUT": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    payloads = [
        _sdsc_payload(
            "0_add",
            "add",
            num_work_slices={"mb": 32, "out": 1},
            core_mapping=producer_mapping,
            primary_ds_info=same_primary,
            input_labels=["Tensor0-idx0", "Tensor1-idx1"],
            output_labels=["Tensor2-idx2"],
            labeled_ds=[
                {
                    "ldsIdx_": 0,
                    "dsName_": "Tensor0",
                    "dsType_": "INPUT",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
                {
                    "ldsIdx_": 1,
                    "dsName_": "Tensor1",
                    "dsType_": "INPUT",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
                {
                    "ldsIdx_": 2,
                    "dsName_": "Tensor2",
                    "dsType_": "OUTPUT",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
            ],
        ),
        _sdsc_payload(
            "1_ReStickifyOpHBM",
            "ReStickifyOpHBM",
            num_work_slices={"mb": 4, "out": 8},
            core_mapping=destination_mapping,
            input_labels=["Tensor0-idx0"],
            output_labels=["Tensor1-idx1"],
            labeled_ds=[
                {
                    "ldsIdx_": 0,
                    "dsName_": "Tensor0",
                    "dsType_": "OUTPUT",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
                {
                    "ldsIdx_": 1,
                    "dsName_": "Tensor1",
                    "dsType_": "KERNEL",
                    "scale_": [1, 1],
                    "wordLength": 2,
                    "dataFormat_": "SEN169_FP16",
                    "memOrg_": {"lx": {"isPresent": 1}},
                },
            ],
        ),
        _sdsc_payload(
            "2_add",
            "add",
            num_work_slices={"mb": 4, "out": 8},
            core_mapping=destination_mapping,
            primary_ds_info=same_primary,
        ),
    ]

    descriptor = maybe_emit_lx_neighbor_descriptor(
        "sdsc_fused_add",
        str(tmp_path),
        _files(),
        specs,
        sdsc_payloads=payloads,
    )

    assert descriptor is not None
    relation = descriptor["edges"][0]["source_view_contract"]["coordinate_relations"]
    assert relation["producer_output_to_restickify_input"][
        "same_coordinate_strings"
    ] == [False, False, False]
    candidate = descriptor["streaming_bridge_candidates"][0]
    assert candidate["status"] == "emitted"
    assert candidate["bridge_kind"] == "direct-ptlx-layout-transform"
    assert candidate["bridge_lowering"] == "direct-ptlx-diagnostic"
    assert candidate["direction"] == "output-to-kernel"
    assert candidate["bridge_endpoint_contract_valid"] is True
    assert candidate["production_valid"] is False
    assert candidate["production_blocker"] == (
        "missing-three-stage-remote-fragment-ptlx-lowering"
    )
    assert candidate["production_contract"]["required_primitive"] == (
        "remote-fragment-aware-ptlx-coordinate-remap"
    )
    assert candidate["production_contract"]["tile_contract"][
        "all_tiles_materialized"
    ] is True
    assert candidate["production_contract"]["required_lowering"] == [
        "STCDPOpLx/InputFetchNeighbor gather producer LX fragments into "
        "bounded per-core tile workspace",
        "local PT/interslice tile transform changes stick/layout semantics",
        "STCDPOpLx/InputFetchNeighbor writes or scatters the consumer-owned "
        "LX tile",
    ]
    assert candidate["bridge_metadata"]["coalescing"] == "direct-64x64-tiles"
    assert candidate["bridge_metadata"]["direction"] == "output-to-kernel"
    assert candidate["bridge_metadata"]["semantic_transform_certified"] is False


def test_streaming_bridge_uses_three_stage_for_kernel_to_output_transform(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(config, "restickify_lx_neighbor_descriptor", True)
    monkeypatch.setattr(config, "restickify_lx_neighbor_streaming_bridge", True)
    specs = _candidate_specs()
    c0 = Symbol("c0")
    c1 = Symbol("c1")
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    for spec in specs:
        for arg in spec.args:
            arg.device_size = [512, 512]
    specs[0].args[-1].device_coordinates = [c0, c1]
    specs[1].args[0].device_coordinates = [d1, d0]
    specs[1].args[-1].device_coordinates = [d0, d1]
    specs[2].args[0].device_coordinates = [c1, c0]
    producer_primary = {
        "INPUT": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["mb", "out"],
            "stickDimOrder_": ["out"],
            "stickSize_": [64],
        },
    }
    destination_primary = {
        "INPUT": {
            "layoutDimOrder_": ["out", "mb"],
            "stickDimOrder_": ["mb"],
            "stickSize_": [64],
        },
        "OUTPUT": {
            "layoutDimOrder_": ["out", "mb"],
            "stickDimOrder_": ["mb"],
            "stickSize_": [64],
        },
        "KERNEL": {
            "layoutDimOrder_": ["out", "mb"],
            "stickDimOrder_": ["mb"],
            "stickSize_": [64],
        },
    }
    producer_mapping = {str(core): {"mb": core, "out": 0} for core in range(32)}
    destination_mapping = {
        str(core): {"mb": core % 4, "out": core // 4} for core in range(32)
    }
    payloads = [
        _sdsc_payload(
            "0_add",
            "add",
            num_work_slices={"mb": 32, "out": 1},
            core_mapping=producer_mapping,
            primary_ds_info=producer_primary,
        ),
        _sdsc_payload(
            "1_ReStickifyOpHBM",
            "ReStickifyOpHBM",
            num_work_slices={"mb": 4, "out": 8},
            core_mapping=destination_mapping,
            primary_ds_info={
                **producer_primary,
                "OUTPUT": destination_primary["OUTPUT"],
                "KERNEL": destination_primary["KERNEL"],
            },
            input_labels=["Tensor0-idx0"],
            output_labels=["Tensor1-idx1"],
        ),
        _sdsc_payload(
            "2_add",
            "add",
            num_work_slices={"mb": 4, "out": 8},
            core_mapping=destination_mapping,
            primary_ds_info=destination_primary,
        ),
    ]

    descriptor = maybe_emit_lx_neighbor_descriptor(
        "sdsc_fused_add",
        str(tmp_path),
        _files(),
        specs,
        sdsc_payloads=payloads,
    )

    assert descriptor is not None
    candidate = descriptor["streaming_bridge_candidates"][0]
    assert candidate["status"] == "emitted"
    assert candidate["bridge_kind"] == "direct-ptlx-layout-transform"
    assert candidate["bridge_lowering"] == "three-stage-gather-transform-scatter"
    assert candidate["direction"] == "kernel-to-output"
    assert candidate["bridge_endpoint_contract_valid"] is False
    assert candidate["bridge_endpoint_contract"]["reason"] == (
        "native-ptlx-output-needs-consumer-endpoint-adapter"
    )
    assert candidate["bridge_endpoint_contract"][
        "native_endpoint_adapter_required"
    ] is True
    assert candidate["production_valid"] is False
    assert candidate["production_blocker"] == (
        "native-ptlx-output-needs-consumer-endpoint-adapter"
    )
    assert candidate["production_contract"]["required_primitive"] == (
        "consumer-lx-endpoint-adapter"
    )
    adapter = candidate["consumer_endpoint_adapter"]
    assert adapter["available"] is True
    assert adapter["executable"] is False
    assert adapter["coordinate_map"] == {
        "destination_out": "native_out",
        "destination_mb": "native_j",
    }
    assert adapter["dropped_singleton_dims"] == ["native_i", "native_mb"]
    assert adapter["lowering_helper"] == (
        "generate_native_ptlx_consumer_endpoint_adapter_tile_sdsc"
    )
    assert adapter["required_stick_transform"] == {
        "from": "native_j",
        "to": "destination_mb",
    }
    diagnostic_candidates = {
        entry["name"]: entry for entry in adapter["diagnostic_candidates"]
    }
    assert diagnostic_candidates["native-64x64-tiles"]["role"] == (
        "local-ptlx-transform"
    )
    assert diagnostic_candidates["direct-64x64-tiles"]["production_blocker"] == (
        "direct-ptlx-tile-lacks-proven-remote-fragment-coordinate-map"
    )
    assert diagnostic_candidates["validgap-consumer-64x64-tiles"][
        "production_blocker"
    ] == "validgap-consumer-tile-lacks-hardware-value-proof"
    assert candidate["production_contract"]["consumer_endpoint_adapter"] == adapter
    assert set(candidate["op_funcs_used"]) == {"STCDPOpLx", "ReStickifyOpWithPTLx"}
    assert candidate["datadsc_count"] == candidate["total_tiles"] * 3
    assert candidate["production_contract"]["tile_contract"][
        "all_tiles_materialized"
    ] is True
    assert candidate["bridge_metadata"]["fallback"] == "ReStickifyOpHBM"
    assert candidate["bridge_metadata"]["coalescing"] == "native-64x64-tiles"
    assert candidate["bridge_metadata"]["native_local_transform_contract"] is True
    assert candidate["bridge_metadata"]["semantic_transform_certified"] is False
