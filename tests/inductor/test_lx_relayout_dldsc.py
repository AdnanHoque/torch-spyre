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

import json

from sympy import Integer, Mod, Symbol, floor
from torch._inductor.ir import ComputedBuffer

from torch_spyre._C import DataFormats
from torch_spyre._inductor import config
from torch_spyre._inductor.codegen.bundle import generate_bundle
from torch_spyre._inductor.lx_relayout import (
    LXRelayoutTopology,
    _matmul_operand_contract_exceeds_budget,
    _prefer_matmul_operand_contract,
)
from torch_spyre._inductor.lx_relayout_contracts import (
    COMM_CLASS_ALL_TO_ALL_SHUFFLE,
    COMM_CLASS_ALL_GATHER,
    COMM_PATTERN_ALL_GATHER,
    LAYOUT_MODIFIER_OPERAND_STAGED,
    LAYOUT_TRANSFORM_STICK_RELAYOUT,
    REALIZATION_GATHER_THEN_RESTICKIFY,
    make_matmul_operand_allgather_contract,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg


def _tensor_arg(
    *,
    is_input: bool,
    arg_index: int,
    name: str,
    allocation: dict,
    coords,
    residency=None,
) -> TensorArg:
    return TensorArg(
        is_input=is_input,
        arg_index=arg_index,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[8, 64, 64],
        device_coordinates=list(coords),
        allocation=allocation,
        name=name,
        lx_residency_core_id_to_wk_slice=residency,
    )


def test_matmul_operand_contract_is_all_gather_only(monkeypatch):
    monkeypatch.setattr(config, "lx_planner_relayout", True)
    all_gather = LXRelayoutTopology(
        COMM_CLASS_ALL_GATHER,
        "many_to_many",
        max_fanout=32,
        max_fanin=32,
        transfer_count=1024,
    )
    all_to_all_shuffle = LXRelayoutTopology(
        COMM_CLASS_ALL_TO_ALL_SHUFFLE,
        COMM_CLASS_ALL_TO_ALL_SHUFFLE,
        max_fanout=1,
        max_fanin=1,
        transfer_count=32,
    )

    assert _prefer_matmul_operand_contract(1, all_gather)
    assert not _prefer_matmul_operand_contract(1, all_to_all_shuffle)
    assert not _prefer_matmul_operand_contract(None, all_gather)

    contract = make_matmul_operand_allgather_contract(
        producer_op="mul",
        consumer_op="batchmatmul",
        read_index=1,
        producer_work_slice_dims={"0": 32},
        consumer_tensor_work_slice_dims={"0": 1},
        consumer_compute_work_slice_dims={"0": 32},
    )

    assert contract["kind"] == COMM_CLASS_ALL_GATHER
    assert contract["classification"] == COMM_CLASS_ALL_GATHER
    assert contract["communication_class"] == COMM_CLASS_ALL_GATHER
    assert contract["communication_pattern"] == COMM_PATTERN_ALL_GATHER
    assert contract["layout_transform"] == LAYOUT_TRANSFORM_STICK_RELAYOUT
    assert contract["layout_modifier"] == LAYOUT_MODIFIER_OPERAND_STAGED
    assert contract["realization_strategy"] == REALIZATION_GATHER_THEN_RESTICKIFY
    assert contract["operand_role"] == "rhs"
    assert contract["requires_staged_realization"]


def test_matmul_operand_contract_budget_classifies_oversized_tensors(monkeypatch):
    class FakeDType:
        itemsize = 2

    class FakeProducer:
        def __init__(self, size):
            self._size = size

        def get_size(self):
            return self._size

        def get_dtype(self):
            return FakeDType()

    class FakeGraph:
        pass

    monkeypatch.setattr(config, "lx_planner_relayout_max_matmul_operand_bytes", 1024)

    assert _matmul_operand_contract_exceeds_budget(
        FakeGraph(), FakeProducer([33, 16])
    ) == (True, 1056, 1024)
    assert _matmul_operand_contract_exceeds_budget(
        FakeGraph(), FakeProducer([32, 16])
    ) == (False, 1024, 1024)


def test_bundle_enriches_matmul_operand_contract_with_source_target_layouts(tmp_path):
    mb = Symbol("c0")
    out = Symbol("c1")
    in_dim = Symbol("c2")
    producer_residency = {str(core): {"0": core} for core in range(4)}
    contract = make_matmul_operand_allgather_contract(
        producer_op="mul",
        consumer_op="batchmatmul",
        read_index=1,
        producer_work_slice_dims={"0": 4},
        consumer_tensor_work_slice_dims={"0": 1},
        consumer_compute_work_slice_dims={"0": 4},
    )
    contract.update(
        {
            "source_name": "buf0",
            "producer_name": "buf0",
            "consumer_name": "buf1",
            "producer_core_id_to_device_slice": producer_residency,
            "consumer_core_id_to_device_slice": {
                str(core): {"0": 0} for core in range(4)
            },
        }
    )

    producer = OpSpec(
        op="mul",
        is_reduction=False,
        iteration_space={mb: (Integer(64), 1), out: (Integer(512), 4)},
        op_info={},
        args=[
            _tensor_arg(
                is_input=True,
                arg_index=0,
                name="arg0",
                allocation={"hbm": 0},
                coords=(floor(out / 64), mb, Mod(out, 64)),
            ),
            _tensor_arg(
                is_input=True,
                arg_index=1,
                name="arg1",
                allocation={"hbm": 0x1000},
                coords=(floor(out / 64), mb, Mod(out, 64)),
            ),
            _tensor_arg(
                is_input=False,
                arg_index=-1,
                name="buf0",
                allocation={"lx": 0},
                coords=(floor(out / 64), mb, Mod(out, 64)),
            ),
        ],
    )
    consumer = OpSpec(
        op="batchmatmul",
        is_reduction=True,
        iteration_space={
            mb: (Integer(64), 4),
            out: (Integer(512), 1),
            in_dim: (Integer(64), 1),
        },
        op_info={"lx_relayout_classifications": [contract]},
        args=[
            _tensor_arg(
                is_input=True,
                arg_index=2,
                name="arg2",
                allocation={"hbm": 0x2000},
                coords=(floor(in_dim / 64), mb, Mod(in_dim, 64)),
            ),
            _tensor_arg(
                is_input=True,
                arg_index=-1,
                name="buf0",
                allocation={"lx": 0},
                coords=(floor(out / 64), in_dim, Mod(out, 64)),
                residency=producer_residency,
            ),
            _tensor_arg(
                is_input=False,
                arg_index=3,
                name="buf1",
                allocation={"hbm": 0x3000},
                coords=(floor(out / 64), mb, Mod(out, 64)),
            ),
        ],
    )

    generate_bundle("lx_relayout_contract", str(tmp_path), [producer, consumer])

    sdsc_1 = json.loads((tmp_path / "sdsc_1.json").read_text())
    classification = next(iter(sdsc_1.values()))["lxRelayoutClassifications_"][0]

    assert classification["source_lx_tensor"]["buffer_name"] == "buf0"
    assert classification["source_lx_tensor"]["component_"] == "lx"
    assert classification["source_lx_tensor"]["coreIdToWkSlice_"] == {
        "0": {"mb": 0, "out": 0},
        "1": {"mb": 0, "out": 1},
        "2": {"mb": 0, "out": 2},
        "3": {"mb": 0, "out": 3},
    }
    assert classification["target_kernel_tensor"]["dsType_"] == "KERNEL"
    assert classification["target_kernel_tensor"]["component_"] == "lx"


def test_bundle_drops_incomplete_matmul_operand_contract(tmp_path):
    mb = Symbol("c0")
    out = Symbol("c1")
    in_dim = Symbol("c2")
    contract = make_matmul_operand_allgather_contract(
        producer_op="clone",
        consumer_op="batchmatmul",
        read_index=0,
        producer_work_slice_dims={"0": 4},
        consumer_tensor_work_slice_dims={"0": 1},
        consumer_compute_work_slice_dims={"0": 4},
    )
    contract.update({"source_name": "missing_source", "producer_name": "missing_source"})
    consumer = OpSpec(
        op="batchmatmul",
        is_reduction=True,
        iteration_space={
            mb: (Integer(64), 4),
            out: (Integer(512), 1),
            in_dim: (Integer(64), 1),
        },
        op_info={"lx_relayout_classifications": [contract]},
        args=[
            _tensor_arg(
                is_input=True,
                arg_index=-1,
                name="missing_source",
                allocation={"lx": 0},
                coords=(floor(in_dim / 64), mb, Mod(in_dim, 64)),
            ),
            _tensor_arg(
                is_input=True,
                arg_index=2,
                name="arg1",
                allocation={"hbm": 0x2000},
                coords=(floor(out / 64), in_dim, Mod(out, 64)),
            ),
            _tensor_arg(
                is_input=False,
                arg_index=3,
                name="buf1",
                allocation={"hbm": 0x3000},
                coords=(floor(out / 64), mb, Mod(out, 64)),
            ),
        ],
    )

    generate_bundle("missing_source_contract", str(tmp_path), [consumer])

    sdsc_0 = json.loads((tmp_path / "sdsc_0.json").read_text())
    assert next(iter(sdsc_0.values()))["lxRelayoutClassifications_"] == []


def test_computed_source_clone_is_lx_relayout_eligible(monkeypatch):
    class FakeReadWrites:
        reads = [type("Dep", (), {"name": "producer"})()]

    class FakeComputedBuffer(ComputedBuffer):
        pass

    class FakeGraph:
        name_to_buffer = {"producer": FakeComputedBuffer.__new__(FakeComputedBuffer)}

    class FakeTarget:
        __name__ = "clone"

    op = FakeComputedBuffer.__new__(FakeComputedBuffer)
    op.origin_node = type("Origin", (), {"target": FakeTarget()})()
    op.get_read_writes = lambda: FakeReadWrites()

    monkeypatch.setattr(config, "lx_planner_relayout", True)

    from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator

    assert ScratchpadAllocator.__new__(
        ScratchpadAllocator
    )._clone_output_good_for_lx_relayout(FakeGraph(), op)
