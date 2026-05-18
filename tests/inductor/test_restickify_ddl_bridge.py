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
from torch_spyre._inductor.codegen.restickify_ddl_bridge import (
    generate_restickify_ddl_bridge_sdsc,
    restickify_ddl_bridge_skip_reason,
)
from torch_spyre._inductor.codegen.superdsc import SDSCArgs, SDSCSpec
from torch_spyre._inductor.constants import RESTICKIFY_OP, SEGMENT_OFFSETS
from torch_spyre._inductor.op_spec import OpSpec


def _core_mapping(dims, split_dim, num_cores):
    return {
        str(core): {
            str(dim): core if dim == split_dim else 0
            for dim in dims
        }
        for core in range(num_cores)
    }


def _op_spec_stub(source_kind="in_graph_computed") -> OpSpec:
    d0 = Symbol("d0")
    return OpSpec(
        RESTICKIFY_OP,
        False,
        {d0: (128, 1)},
        [],
        {
            "restickify_source_kind": source_kind,
            "restickify_source_name": "buf0",
        },
    )


def _spec(
    *,
    size=2048,
    num_cores=32,
    split_dim_name="d0",
    input_stick_name="d1",
    output_stick_name="d0",
) -> SDSCSpec:
    d0 = Symbol("d0")
    d1 = Symbol("d1")
    dims = {"d0": d0, "d1": d1}
    split_dim = dims[split_dim_name]
    input_stick = dims[input_stick_name]
    output_stick = dims[output_stick_name]
    data_format = DataFormats.SEN169_FP16
    work_slices = {d0: 1, d1: 1}
    work_slices[split_dim] = num_cores
    args = [
        SDSCArgs(
            layout="INPUT",
            data_format=data_format,
            scales={d0: 1, d1: 1},
            strides={d0: size, d1: 1},
            offsets={},
            max_dim_sizes={d0: -1, d1: -1},
            allocation={},
            start_address=SEGMENT_OFFSETS[0],
            backGap={},
        ),
        SDSCArgs(
            layout="OUTPUT",
            data_format=data_format,
            scales={d0: 1, d1: 1},
            strides={d0: 1, d1: size},
            offsets={},
            max_dim_sizes={d0: -1, d1: -1},
            allocation={},
            start_address=SEGMENT_OFFSETS[2],
            backGap={},
        ),
    ]
    return SDSCSpec(
        opfunc=RESTICKIFY_OP,
        execution_unit="sfp",
        data_format=data_format,
        num_inputs=1,
        iteration_space={d0: size, d1: size},
        num_cores=num_cores,
        work_slices=work_slices,
        core_id_to_work_slice={},
        core_id_to_work_slice_override=_core_mapping([d0, d1], split_dim, num_cores),
        padding={},
        layouts={
            "INPUT": {
                "dim_order": [d0, d1],
                "stick_dim_order": input_stick,
                "stick_size": 64,
            },
            "OUTPUT": {
                "dim_order": [d1, d0],
                "stick_dim_order": output_stick,
                "stick_size": 64,
            },
        },
        args=args,
        constants={},
        coordinate_masking={},
    )


def _dsc(payload):
    root = next(iter(payload.values()))
    return next(iter(root["dscs_"][0].values()))


def test_restickify_ddl_bridge_generates_compact_lx_contract():
    spec = _spec()
    compute_payload = generate_sdsc(0, spec)

    reason = restickify_ddl_bridge_skip_reason(_op_spec_stub(), spec)
    payload = generate_restickify_ddl_bridge_sdsc(0, spec, compute_payload)
    root_name, root = next(iter(payload.items()))
    dsc = _dsc(payload)

    assert reason is None
    assert root_name == "0_ReStickifyOpHBM_ddl_bridge"
    assert root.get("target_") is None
    assert root["numWkSlicesPerDim_"] == {"d0": 32, "d1": 1}
    assert root["coreIdToWkSlice_"]["31"] == {"d0": 31, "d1": 0}
    assert dsc.get("target_") is None
    assert set(dsc["primaryDsInfo_"]) == {"INPUT", "OUTPUT"}
    assert len(dsc["dataStageParam_"]) == 2
    assert [node["component_"] for node in dsc["scheduleTree_"][:2]] == ["lx", "lx"]
    assert [lds["dsType_"] for lds in dsc["labeledDs_"]] == ["INPUT", "OUTPUT"]
    assert [lds["segment_"] for lds in dsc["labeledDs_"]] == ["output", "model"]
    assert all(set(lds["memOrg_"]) == {"lx"} for lds in dsc["labeledDs_"])
    assert dsc["computeOp_"][0]["inputLabeledDs"] == ["Tensor0-idx0"]
    assert dsc["computeOp_"][0]["outputLabeledDs"] == ["Tensor1-idx1"]


def test_restickify_ddl_bridge_allows_mirrored_2048_direction():
    spec = _spec(input_stick_name="d0", output_stick_name="d1")
    compute_payload = generate_sdsc(0, spec)

    reason = restickify_ddl_bridge_skip_reason(_op_spec_stub(), spec)
    payload = generate_restickify_ddl_bridge_sdsc(0, spec, compute_payload)
    dsc = _dsc(payload)

    assert reason is None
    assert [node["component_"] for node in dsc["scheduleTree_"][:2]] == ["lx", "lx"]
    assert all(set(lds["memOrg_"]) == {"lx"} for lds in dsc["labeledDs_"])


def test_restickify_ddl_bridge_preserves_original_labeled_ds_roles():
    spec = _spec(input_stick_name="d0", output_stick_name="d1")
    compute_payload = generate_sdsc(0, spec)
    source_dsc = _dsc(compute_payload)
    source_dsc["primaryDsInfo_"]["KERNEL"] = source_dsc["primaryDsInfo_"].pop(
        "OUTPUT"
    )
    source_dsc["primaryDsInfo_"]["OUTPUT"] = source_dsc["primaryDsInfo_"].pop("INPUT")
    for lds in source_dsc["labeledDs_"]:
        if lds["ldsIdx_"] == 0:
            lds["dsType_"] = "OUTPUT"
        elif lds["ldsIdx_"] == 1:
            lds["dsType_"] = "KERNEL"

    payload = generate_restickify_ddl_bridge_sdsc(0, spec, compute_payload)
    dsc = _dsc(payload)

    assert set(dsc["primaryDsInfo_"]) == {"OUTPUT", "KERNEL"}
    assert [lds["dsType_"] for lds in dsc["labeledDs_"]] == ["OUTPUT", "KERNEL"]
    assert [lds["segment_"] for lds in dsc["labeledDs_"]] == ["output", "model"]


def test_restickify_ddl_bridge_skips_graph_input_sources():
    spec = _spec()

    reason = restickify_ddl_bridge_skip_reason(
        _op_spec_stub(source_kind="graph_input_or_weight"), spec
    )

    assert reason == "source-not-in-graph-computed"


def test_restickify_ddl_bridge_skips_unknown_sources():
    spec = _spec()
    d0 = Symbol("d0")

    reason = restickify_ddl_bridge_skip_reason(
        OpSpec(RESTICKIFY_OP, False, {d0: (128, 1)}, [], {}), spec
    )

    assert reason == "source-kind-unknown"


def test_restickify_ddl_bridge_skips_large_per_core_lx_contract():
    spec = _spec(size=4096)

    reason = restickify_ddl_bridge_skip_reason(_op_spec_stub(), spec)

    assert reason == "lx-bytes-per-core-too-large"


def test_restickify_ddl_bridge_skips_unknown_runtime_segments():
    spec = _spec()
    spec.args[-1].start_address = 1024

    reason = restickify_ddl_bridge_skip_reason(_op_spec_stub(), spec)

    assert reason == "unsupported-runtime-segment"
