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
    DESCRIPTOR_FILENAME,
    LOCALITY_CERTIFICATE_OP_INFO_KEY,
    build_lx_neighbor_descriptor,
    maybe_emit_lx_neighbor_descriptor,
)
from torch_spyre._inductor.constants import RESTICKIFY_OP
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.restickify_ring import CORE_MAPPING_OVERRIDE_OP_INFO_KEY


def _arg(is_input: bool, *, arg_index: int = -1) -> TensorArg:
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2048, 2048],
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
    input_labels: list[str] | None = None,
    output_labels: list[str] | None = None,
    primary_ds_info: dict | None = None,
    labeled_ds: list[dict] | None = None,
) -> dict:
    return {
        name: {
            "numCoresUsed_": 2,
            "numWkSlicesPerDim_": {"d0": 2},
            "coreIdToWkSlice_": {
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


def test_builds_candidate_descriptor_for_adjacent_certified_restickify():
    descriptor = build_lx_neighbor_descriptor(
        "sdsc_fused_add",
        _files(),
        _candidate_specs(),
    )

    assert descriptor["schema_version"] == 3
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
