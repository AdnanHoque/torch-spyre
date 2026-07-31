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

from dataclasses import replace
from types import SimpleNamespace

from sympy import Integer, Mod, Symbol, floor

from torch_spyre._inductor import config
import torch_spyre._inductor.lx_relayout as lx_relayout_module
import torch_spyre._inductor.scratchpad.allocator as allocator_module
import torch_spyre._inductor.scratchpad.graph_editor as graph_editor_module

from torch_spyre._C import DataFormats
from torch_spyre._inductor.codegen.superdsc import (
    _map_core_id_to_wk_slice_dims,
    _map_device_dim_splits,
    compile_op_spec,
)
from torch_spyre._inductor.constants import BATCH_MATMUL_OP
from torch_spyre._inductor.lx_relayout import (
    LX_RELAYOUT_ATTR,
    LXCollectiveKind,
    LXRelayoutPlan,
    _classify_lx_relayout,
    _destination_size_ratio,
    collect_lx_relayout_plans,
)
from torch_spyre._inductor.op_spec import OpSpec, TensorArg
from torch_spyre._inductor.pass_utils import PerCoreView, copy_fx_custom_meta
from torch_spyre._inductor.propagate_hints import get_gather_dim
from torch_spyre._inductor.scratchpad.allocator import ScratchpadAllocator
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    FirstFitLayoutSolver,
)
from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
from torch_spyre._inductor.scratchpad.graph_editor import GraphEditor
from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.spyre_kernel import (
    _materialize_explicit_lx_shuffle,
    _materialize_lx_relayout_inputs,
    _repair_granite_p14_dim_labels_after_alignment,
    simplify_op_spec,
)
from torch_spyre._inductor.work_division import (
    _oracle_work_div_hint_by_name,
    _resolve_work_div_hint,
)


class _DummyOp:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class _DummyGraph:
    def __init__(self, *names):
        self.operations = [_DummyOp(name) for name in names]
        self.name_to_buffer = {op.name: op for op in self.operations}

    def try_get_buffer(self, name):
        return self.name_to_buffer.get(name)


def _all_gather_plan() -> LXRelayoutPlan:
    producer_map = {str(core): {"0": core // 8, "1": core % 8} for core in range(32)}
    consumer_map = {str(core): {"0": core // 8, "1": 0} for core in range(32)}
    return LXRelayoutPlan(
        source_name="buf_k",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 4, "1": 1},
        collective_kind=LXCollectiveKind.ALL_GATHER,
        destination_size_ratio=8,
    )


def _granite_mlp_all_to_all_plan() -> LXRelayoutPlan:
    producer_map = {str(core): {"0": core // 8, "1": core % 8} for core in range(32)}
    consumer_map = {str(core): {"0": core, "1": 0} for core in range(32)}
    return LXRelayoutPlan(
        source_name="buf_mlp",
        consumer_name="pointwise",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 4, "1": 8},
        destination_device_dim_splits={"0": 32, "1": 1},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )


def _broadcast_plan() -> LXRelayoutPlan:
    producer_map = {"0": {}}
    consumer_map = {str(core): {} for core in range(32)}
    return LXRelayoutPlan(
        source_name="broadcast_source",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={},
        destination_device_dim_splits={},
        collective_kind=LXCollectiveKind.BROADCAST,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )


def _overlap(lhs, rhs):
    return lhs.address < rhs.address + rhs.size and rhs.address < lhs.address + lhs.size


def _compile_shuffle(shuffle_spec):
    simplify_op_spec(shuffle_spec)
    sdsc, *_ = compile_op_spec(0, shuffle_spec, [])
    root = next(iter(sdsc.values()))
    shuffle_dsc = next(iter(root["dscs_"][0].values()))
    allocations = [
        row for row in shuffle_dsc["scheduleTree_"] if row["nodeType_"] == "allocate"
    ]
    return root, allocations


def test_gather_dim_hint_survives_metadata_copy():
    src = SimpleNamespace(meta={"custom": {"_hint_1": {"gather_dim": "Lq"}}})
    dst = SimpleNamespace(meta={"custom": {"_hint_2": {"work_div": {"H": 4}}}})
    copy_fx_custom_meta(src, dst)

    lq = Symbol("lq")
    op = SimpleNamespace(origins=[dst], work_div_loop_info={lq: ["Lq"]})
    assert get_gather_dim(op) == lq
    assert set(dst.meta["custom"]) == {"_hint_1", "_hint_2"}


def test_prefill_mlp_oracle_aligns_down_projection_with_swiglu(monkeypatch):
    class DummyBuffer:
        def get_name(self):
            return "buf57"

        def get_size(self):
            return [1, 512, 4096]

    op = DummyBuffer()
    mb, out, reduction = Symbol("mb"), Symbol("out"), Symbol("reduction")

    monkeypatch.setattr(config, "relayout_oracle_prefill_mlp_inputs", True)
    monkeypatch.setattr(config, "relayout_oracle_prefill_mlp_down_projection", True)

    assert _oracle_work_div_hint_by_name(op) == {"mb": 32, "out": 1}
    assert _resolve_work_div_hint(
        op,
        {mb: Integer(512), out: Integer(4096), reduction: Integer(12800)},
    ) == {mb: 32, out: 1}

    monkeypatch.setattr(
        config, "relayout_oracle_prefill_mlp_down_projection_mb", 16
    )
    monkeypatch.setattr(
        config, "relayout_oracle_prefill_mlp_down_projection_out", 2
    )
    assert _oracle_work_div_hint_by_name(op) == {"mb": 16, "out": 2}
    assert _resolve_work_div_hint(
        op,
        {mb: Integer(512), out: Integer(4096), reduction: Integer(12800)},
    ) == {mb: 16, out: 2}


def test_prefill_mlp_normalization_oracle_replays_p05_p10_p11(monkeypatch):
    class DummyBuffer:
        def __init__(self, name, size):
            self.name = name
            self.size = size

        def get_name(self):
            return self.name

        def get_size(self):
            return self.size

    mb, out = Symbol("mb"), Symbol("out")
    full = {mb: Integer(512), out: Integer(4096)}
    scalar = {mb: Integer(512)}

    monkeypatch.setattr(config, "relayout_oracle_prefill_mlp_normalization", False)
    assert _oracle_work_div_hint_by_name(
        DummyBuffer("buf47", [1, 512, 4096])
    ) == {}

    monkeypatch.setattr(config, "relayout_oracle_prefill_mlp_normalization", True)
    for name in ("buf46", "buf47", "buf51", "buf52"):
        op = DummyBuffer(name, [1, 512, 4096])
        assert _oracle_work_div_hint_by_name(op) == {"mb": 8, "out": 4}
        assert _resolve_work_div_hint(op, full) == {mb: 8, out: 4}
        assert op._spyre_oracle_gather_dim_symbol == out
    for name in ("buf48", "buf49", "buf50"):
        op = DummyBuffer(name, [1, 512, 1])
        iteration_space = full if name == "buf48" else scalar
        assert _oracle_work_div_hint_by_name(op) == {"mb": 8, "out": 1}
        expected = {mb: 8, out: 1} if name == "buf48" else {mb: 8}
        assert _resolve_work_div_hint(op, iteration_space) == expected


def test_granite_last_token_head_oracle_uses_output_only_split(monkeypatch):
    class DummyBuffer:
        def __init__(self, name="buf0", size=(1, 1, 50176), reduction_type=None):
            self.name = name
            self.size = list(size)
            self.data = SimpleNamespace(reduction_type=reduction_type)

        def get_name(self):
            return self.name

        def get_size(self):
            return self.size

    op = DummyBuffer(reduction_type=BATCH_MATMUL_OP)
    monkeypatch.setattr(config, "work_div_oracle_granite_last_token_head", False)
    assert _oracle_work_div_hint_by_name(op) == {}

    monkeypatch.setattr(config, "work_div_oracle_granite_last_token_head", True)
    assert _oracle_work_div_hint_by_name(op) == {"in": 1, "out": 28}
    # The head is buf0 when compiled alone and buf6 in the fused final-stage
    # graph. Shape + reduction kind are the stable exact-graph predicate.
    assert _oracle_work_div_hint_by_name(
        DummyBuffer(name="buf6", reduction_type=BATCH_MATMUL_OP)
    ) == {"in": 1, "out": 28}
    assert (
        _oracle_work_div_hint_by_name(
            DummyBuffer(size=(1, 512, 49280), reduction_type=BATCH_MATMUL_OP)
        )
        == {}
    )
    assert _oracle_work_div_hint_by_name(DummyBuffer(reduction_type="sum")) == {}

    out, reduction = Symbol("out"), Symbol("reduction")
    assert _resolve_work_div_hint(
        op,
        {out: Integer(784), reduction: Integer(64)},
    ) == {reduction: 1, out: 28}
    assert 784 % 28 == 0

    legacy = DummyBuffer(size=(1, 1, 49280), reduction_type=BATCH_MATMUL_OP)
    assert _oracle_work_div_hint_by_name(legacy) == {"in": 1, "out": 22}
    assert _resolve_work_div_hint(
        legacy,
        {out: Integer(770), reduction: Integer(64)},
    ) == {reduction: 1, out: 22}


def test_granite_p14_oracle_uses_token_hidden_grid(monkeypatch):
    class DummyBuffer:
        def __init__(self, name="buf5", size=(1, 512, 4096)):
            self.name = name
            self.size = list(size)

        def get_name(self):
            return self.name

        def get_size(self):
            return self.size

    op = DummyBuffer()
    mb, out = Symbol("mb"), Symbol("out")

    monkeypatch.setattr(config, "relayout_oracle_granite_p14", False)
    assert _oracle_work_div_hint_by_name(op) == {}

    monkeypatch.setattr(config, "relayout_oracle_granite_p14", True)
    assert _oracle_work_div_hint_by_name(op) == {"mb": 8, "out": 4}
    assert _resolve_work_div_hint(
        op,
        {mb: Integer(512), out: Integer(4096)},
    ) == {mb: 8, out: 4}
    assert op._spyre_oracle_gather_dim_symbol == out

    consumer = DummyBuffer("buf6", (1, 1, 4096))
    assert _oracle_work_div_hint_by_name(consumer) == {"out": 32}
    assert _resolve_work_div_hint(
        consumer,
        {out: Integer(4096)},
    ) == {out: 32}


def test_granite_last_token_subset_all_gather_is_exactly_gated():
    source_shards = {str(core): {"0": core} for core in range(32)}
    head_owners = {str(core): {"0": 0} for core in range(28)}

    assert (
        _classify_lx_relayout(
            source_shards,
            {"0": 32},
            head_owners,
            {"0": 1},
        )
        is None
    )
    classification = _classify_lx_relayout(
        source_shards,
        {"0": 32},
        head_owners,
        {"0": 1},
        allow_grouped_collectives=True,
    )
    assert classification is not None
    assert classification.collective_kind is LXCollectiveKind.ALL_GATHER
    assert classification.destination_size_ratio == 32
    deliveries = 32 * 28
    local_deliveries = 28
    assert deliveries == 896
    assert (deliveries - local_deliveries) * 256 == 222_208


def test_granite_p14_emits_sparse_last_cohort_to_hidden_shards(monkeypatch):
    hidden, token = Symbol("c0"), Symbol("z0")
    plan = LXRelayoutPlan(
        source_name="buf5",
        consumer_name="buf6",
        source_core_id_to_device_slice={
            str(core): {"0": 0, "1": core - 28}
            for core in range(28, 32)
        },
        destination_core_id_to_device_slice={
            str(core): {"0": 0, "1": core} for core in range(32)
        },
        source_device_dim_splits={"0": 1, "1": 4},
        destination_device_dim_splits={"0": 1, "1": 32},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=1,
        destination_size_divisor=512,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 64, 64],
        device_coordinates=[
            token + 511,
            floor(hidden / 64),
            Mod(hidden, 64),
        ],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="identity",
        is_reduction=False,
        # This is the actual post-slice rank-1 consumer shape.  The selected
        # singleton token symbol exists only on the source TensorArg.
        iteration_space={hidden: (Integer(4096), 32)},
        args=[source_arg, source_arg],
        op_info={},
    )

    monkeypatch.setattr(config, "relayout_oracle_granite_p14", True)
    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, _ = result
    assert shuffle_spec.iteration_space == {hidden: (Integer(4096), 32)}
    assert shuffle_spec.dim_labels_override == ["out"]
    # align_tensors adds this source-only singleton in the real rank-1 graph.
    # Reproduce its post-alignment state and exercise the exact-gated repair.
    shuffle_spec.iteration_space = {
        hidden: (Integer(4096), 32),
        token: (Integer(1), 1),
    }
    _repair_granite_p14_dim_labels_after_alignment(shuffle_spec, (hidden,))
    assert shuffle_spec.iteration_space == {
        hidden: (Integer(4096), 32),
        token: (Integer(1), 1),
    }
    assert shuffle_spec.dim_labels_override == ["out", "mb"]
    assert shuffle_spec.args[0].allocation == {"lx": 0x24000 + 129_024}
    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32
    assert allocations[0]["coordinates_"]["coreIdToWkSlice_"] == {
        str(28 + hidden): {"mb": 0, "out": hidden}
        for hidden in range(4)
    }
    assert allocations[1]["coordinates_"]["coreIdToWkSlice_"] == {
        str(core): {"mb": 0, "out": core} for core in range(32)
    }
    deliveries = 32
    local_deliveries = 1
    assert (deliveries - local_deliveries) * 256 == 7_936


def test_granite_p14_final_norm_emits_token_major_owners(monkeypatch):
    token, hidden = Symbol("c0"), Symbol("c1")
    input_arg = TensorArg(
        is_input=True,
        arg_index=0,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 64, 64],
        device_coordinates=[token, floor(hidden / 64), Mod(hidden, 64)],
        allocation={"hbm": 0},
    )
    output_arg = replace(
        input_arg,
        is_input=False,
        arg_index=-1,
        allocation={"lx": 0x100},
    )
    spec = OpSpec(
        op="mul",
        is_reduction=False,
        iteration_space={
            token: (Integer(512), 8),
            hidden: (Integer(4096), 4),
        },
        args=[input_arg, output_arg],
        op_info={},
        # The fused final-normalization chain currently loses this oracle
        # symbol during dimension-label normalization.
        gather_dim=Symbol("d1"),
    )

    monkeypatch.setattr(config, "relayout_oracle_granite_p14", True)
    sdsc, *_ = compile_op_spec(0, spec, [])
    root = next(iter(sdsc.values()))

    assert root["numWkSlicesPerDim_"] == {"out": 4, "mb": 8}
    assert root["coreIdToWkSlice_"]["0"] == {"out": 0, "mb": 0}
    assert root["coreIdToWkSlice_"]["1"] == {"out": 1, "mb": 0}
    assert root["coreIdToWkSlice_"]["28"] == {"out": 0, "mb": 7}
    assert root["coreIdToWkSlice_"]["31"] == {"out": 3, "mb": 7}


def test_granite_last_token_hidden_axis_bridge(monkeypatch):
    hidden = Symbol("in")
    source_map = {str(core): {"1": core} for core in range(32)}

    monkeypatch.setattr(config, "work_div_oracle_granite_last_token_head", True)

    for source_name in ("buf0", "buf6"):
        assert _map_core_id_to_wk_slice_dims(
            source_map,
            {"0": hidden},
            [hidden],
            source_name,
        ) == {str(core): {"in": core} for core in range(32)}
        assert _map_device_dim_splits(
            {"1": 32},
            {"0": hidden},
            f"__spyre_lx_relayout_destination__:{source_name}",
        ) == {hidden: 32}


def test_prefill_mlp_normalization_grouped_collectives():
    source_8x4 = {
        str(core): {"0": core // 4, "1": core % 4} for core in range(32)
    }
    reduction_owners = {
        str(core): {"0": core, "1": 0} for core in range(8)
    }
    p05 = _classify_lx_relayout(
        source_8x4,
        {"0": 8, "1": 4},
        reduction_owners,
        {"0": 8, "1": 1},
        allow_grouped_collectives=True,
    )
    assert p05 is not None
    assert p05.collective_kind is LXCollectiveKind.ALL_GATHER
    assert p05.destination_size_ratio == 4

    scalar_owners = {str(core): {"0": core} for core in range(8)}
    scalar_cohorts = {str(core): {"0": core // 4} for core in range(32)}
    p10_p11 = _classify_lx_relayout(
        scalar_owners,
        {"0": 8},
        scalar_cohorts,
        {"0": 8},
        allow_grouped_collectives=True,
    )
    assert p10_p11 is not None
    assert p10_p11.collective_kind is LXCollectiveKind.BROADCAST
    assert p10_p11.destination_size_ratio == 1


def test_prefill_mlp_normalization_p05_emits_sparse_all_gather(monkeypatch):
    mb, out = Symbol("mb"), Symbol("out")
    source_map = {
        str(core): {"0": core // 4, "1": core % 4} for core in range(32)
    }
    destination_map = {
        str(core): {"0": core, "1": 0} for core in range(8)
    }
    plan = LXRelayoutPlan(
        source_name="buf47",
        consumer_name="buf48",
        source_core_id_to_device_slice=source_map,
        destination_core_id_to_device_slice=destination_map,
        source_device_dim_splits={"0": 8, "1": 4},
        destination_device_dim_splits={"0": 8, "1": 1},
        collective_kind=LXCollectiveKind.ALL_GATHER,
        destination_size_ratio=4,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 64, 64],
        device_coordinates=[mb, floor(out / 64), Mod(out, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="mean",
        is_reduction=True,
        iteration_space={mb: (Integer(512), 8), out: (Integer(4096), 1)},
        args=[source_arg],
        op_info={},
    )

    monkeypatch.setattr(config, "relayout_oracle_prefill_mlp_normalization", True)
    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, _ = result
    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32
    assert allocations[0]["coordinates_"]["coreIdToWkSlice_"]["31"] == {
        "mb": 7,
        "out": 3,
    }
    assert allocations[1]["coordinates_"]["coreIdToWkSlice_"]["7"] == {
        "mb": 7,
        "out": 0,
    }


def test_prefill_residual_add_p12_classifies_sparse_repartition():
    source_map = {
        str(2 * token_shard): {"0": token_shard, "1": 0}
        for token_shard in range(16)
    }
    destination_map = {
        str(core): {"0": core // 4, "1": core % 4} for core in range(32)
    }

    classification = _classify_lx_relayout(
        source_map,
        {"0": 16, "1": 1},
        destination_map,
        {"0": 8, "1": 4},
        allow_grouped_collectives=True,
    )

    assert classification is not None
    assert classification.collective_kind is LXCollectiveKind.ALL_TO_ALL
    assert classification.destination_size_ratio == 1
    assert classification.destination_size_divisor == 2


def test_prefill_residual_add_p12_emits_exact_piece_maps():
    mb, out = Symbol("mb"), Symbol("out")
    source_map = {
        str(2 * token_shard): {"0": token_shard, "1": 0}
        for token_shard in range(16)
    }
    destination_map = {
        str(core): {"0": core // 4, "1": core % 4} for core in range(32)
    }
    plan = LXRelayoutPlan(
        source_name="buf45",
        consumer_name="buf46",
        source_core_id_to_device_slice=source_map,
        destination_core_id_to_device_slice=destination_map,
        source_device_dim_splits={"0": 16, "1": 1},
        destination_device_dim_splits={"0": 8, "1": 4},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=1,
        destination_size_divisor=2,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 64, 64],
        device_coordinates=[mb, floor(out / 64), Mod(out, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={mb: (Integer(512), 8), out: (Integer(4096), 4)},
        args=[source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, _ = result
    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32
    assert allocations[0]["coordinates_"]["coreIdToWkSlice_"] == {
        str(2 * token_shard): {"mb": token_shard, "out": 0}
        for token_shard in range(16)
    }
    assert allocations[1]["coordinates_"]["coreIdToWkSlice_"] == {
        str(core): {"mb": core // 4, "out": core % 4} for core in range(32)
    }


def test_prefill_mlp_normalization_scalar_axis_bridge(monkeypatch):
    mb, x = Symbol("mb"), Symbol("x")
    device_dim_to_sdsc_dim = {"1": mb, "0": x}

    monkeypatch.setattr(config, "relayout_oracle_prefill_mlp_normalization", True)

    assert _map_core_id_to_wk_slice_dims(
        {"0": {"3": 0}, "4": {"3": 1}},
        device_dim_to_sdsc_dim,
        [mb, x],
        "buf50",
    ) == {
        "0": {"mb": 0, "x": 0},
        "4": {"mb": 1, "x": 0},
    }
    assert _map_device_dim_splits(
        {"3": 8}, device_dim_to_sdsc_dim, "buf50"
    ) == {mb: 8}

    # The shuffle itself has only the normalized mb dimension.  It must use
    # the same bridge as the downstream [mb, x] consumer.
    assert _map_core_id_to_wk_slice_dims(
        {"0": {"3": 0}, "4": {"3": 1}},
        {"1": mb},
        [mb],
        "buf50",
    ) == {"0": {"mb": 0}, "4": {"mb": 1}}
    assert _map_device_dim_splits({"3": 8}, {"1": mb}, "buf50") == {mb: 8}


def test_prefill_mlp_normalization_broadcast_is_exactly_scoped(monkeypatch):
    mb, out = Symbol("mb"), Symbol("out")
    plan = LXRelayoutPlan(
        source_name="buf50",
        consumer_name="buf51",
        source_core_id_to_device_slice={
            str(core): {"3": core} for core in range(8)
        },
        destination_core_id_to_device_slice={
            str(core): {"3": core // 4} for core in range(32)
        },
        source_device_dim_splits={"3": 8},
        destination_device_dim_splits={"3": 8},
        collective_kind=LXCollectiveKind.BROADCAST,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[1, 1, 1, 512],
        device_coordinates=[Integer(0), Integer(0), Integer(0), mb],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="mul",
        is_reduction=False,
        iteration_space={mb: (Integer(512), 8), out: (Integer(4096), 4)},
        args=[source_arg],
        op_info={},
    )

    monkeypatch.setattr(config, "relayout_oracle_prefill_mlp_normalization", True)
    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    assert result[0].num_cores_override == 32
    assert result[0].replicas_contiguous

    generic = replace(plan, source_name="other", consumer_name="other_consumer")
    generic_result = _materialize_explicit_lx_shuffle(
        replace(source_arg, name="other"), consumer_spec, generic
    )
    assert generic_result is not None
    assert not generic_result[0].replicas_contiguous


def test_explicit_shuffle_rejects_sparse_participant_holes():
    x = Symbol("x")
    plan = LXRelayoutPlan(
        source_name="sparse",
        consumer_name="consumer",
        source_core_id_to_device_slice={"0": {"0": 0}, "2": {"0": 1}},
        destination_core_id_to_device_slice={"0": {"0": 1}, "2": {"0": 0}},
        source_device_dim_splits={"0": 2},
        destination_device_dim_splits={"0": 2},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 64],
        device_coordinates=[floor(x / 64), Mod(x, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={x: (Integer(128), 2)},
        args=[source_arg],
        op_info={},
    )

    assert _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan) is None


def test_all_to_all_geometry_and_emission():
    producer_map = {"0": {"0": 0}, "1": {"0": 1}}
    consumer_map = {"0": {"0": 1}, "1": {"0": 0}}
    all_to_all = _classify_lx_relayout(
        producer_map,
        {"0": 2},
        consumer_map,
        {"0": 2},
    )
    all_gather = _classify_lx_relayout(
        {"0": {"0": 0}, "1": {"0": 1}},
        {"0": 2},
        {"0": {"0": 0}, "1": {"0": 0}},
        {"0": 1},
    )

    assert all_to_all is not None
    assert all_to_all.collective_kind is LXCollectiveKind.ALL_TO_ALL
    assert all_to_all.destination_size_ratio == 1
    assert all_gather is not None
    assert all_gather.collective_kind is LXCollectiveKind.ALL_GATHER
    assert all_gather.destination_size_ratio == 2

    x = Symbol("x")
    plan = LXRelayoutPlan(
        source_name="buf_x",
        consumer_name="consumer",
        source_core_id_to_device_slice=producer_map,
        destination_core_id_to_device_slice=consumer_map,
        source_device_dim_splits={"0": 2},
        destination_device_dim_splits={"0": 2},
        collective_kind=LXCollectiveKind.ALL_TO_ALL,
        destination_size_ratio=1,
        destination_lx_address=0x44000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 64],
        device_coordinates=[floor(x / 64), Mod(x, 64)],
        allocation={"lx": 0x24000},
        name="buf_x",
    )
    consumer_spec = OpSpec(
        op="neg",
        is_reduction=False,
        iteration_space={x: (Integer(128), 2)},
        args=[source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)
    assert result is not None
    _, allocations = _compile_shuffle(result[0])
    assert allocations[0]["coordinates_"]["coreIdToWkSlice_"]["0"] == {"out": 0}
    assert allocations[1]["coordinates_"]["coreIdToWkSlice_"]["0"] == {"out": 1}

    node = SimpleNamespace(**{LX_RELAYOUT_ATTR: {"buf_x": plan}})
    current_node = SimpleNamespace(get_nodes=lambda: [SimpleNamespace(node=node)])
    args = [source_arg, source_arg]
    prefix = _materialize_lx_relayout_inputs(
        current_node,
        args,
        [(0, source_arg), (1, source_arg)],
        consumer_spec,
    )
    assert len(prefix) == 1
    assert args[0].name == args[1].name == plan.destination_name

    # Reduction scheduler wrappers may not carry the buffer's custom plan
    # metadata.  The consumer ComputedBuffer remains an authoritative fallback.
    unstamped_current_node = SimpleNamespace(get_nodes=lambda: [])
    stamped_consumer = SimpleNamespace(**{LX_RELAYOUT_ATTR: {"buf_x": plan}})
    args = [source_arg]
    prefix = _materialize_lx_relayout_inputs(
        unstamped_current_node,
        args,
        [(0, source_arg)],
        consumer_spec,
        stamped_consumer,
    )
    assert len(prefix) == 1
    assert args[0].name == plan.destination_name


def test_broadcast_geometry_requires_one_physical_source_owner():
    producer_map = {"0": {}}
    consumer_map = {str(core): {} for core in range(32)}

    classification = _classify_lx_relayout(
        producer_map,
        {},
        consumer_map,
        {},
    )

    assert classification is not None
    assert classification.collective_kind is LXCollectiveKind.BROADCAST
    assert classification.destination_size_ratio == 1

    # Repeated producer owners would falsely claim the value already exists on
    # every core. They are neither a partition nor a valid broadcast source.
    padded_producer_map = {str(core): {} for core in range(32)}
    assert (
        _classify_lx_relayout(
            padded_producer_map,
            {},
            consumer_map,
            {},
        )
        is None
    )

    # Multiple independently replicated source shards are multicast cohorts,
    # which are not yet supported by this planner.
    multicast_producer_map = {"0": {"0": 0}, "1": {"0": 1}}
    multicast_consumer_map = {
        str(core): {"0": 0 if core < 16 else 1} for core in range(32)
    }
    assert (
        _classify_lx_relayout(
            multicast_producer_map,
            {"0": 2},
            multicast_consumer_map,
            {"0": 2},
        )
        is None
    )


def test_planner_keeps_equal_view_edge_when_core_counts_require_broadcast(monkeypatch):
    class _FakePointwise:
        pass

    class _FakeBuffer:
        def __init__(self, name):
            self.name = name
            self.data = _FakePointwise()

        def get_name(self):
            return self.name

    class _FakeDep:
        def __init__(self, name):
            self.name = name

    producer = _FakeBuffer("broadcast_source")
    consumer = _FakeBuffer("consumer")
    source_write = _FakeDep(producer.name)
    source_read = _FakeDep(producer.name)
    consumer_write = _FakeDep(consumer.name)
    graph = SimpleNamespace(operations=[producer, consumer])
    whole_buffer_view = PerCoreView(work_slice_dims=(), core_to_slot=())

    def read_writes(op):
        if op is producer:
            return SimpleNamespace(reads=[], writes=[source_write])
        return SimpleNamespace(reads=[source_read], writes=[consumer_write])

    monkeypatch.setattr(lx_relayout_module.config, "lx_planner_relayout", True)
    monkeypatch.setattr(lx_relayout_module, "ComputedBuffer", _FakeBuffer)
    monkeypatch.setattr(lx_relayout_module, "Pointwise", _FakePointwise)
    monkeypatch.setattr(lx_relayout_module, "MemoryDep", _FakeDep)
    monkeypatch.setattr(lx_relayout_module, "op_read_writes", read_writes)
    monkeypatch.setattr(lx_relayout_module, "_is_matmul_op", lambda _: False)
    monkeypatch.setattr(
        lx_relayout_module,
        "_per_core_view_on_buf",
        lambda *_: (whole_buffer_view, False, True),
    )
    monkeypatch.setattr(
        lx_relayout_module,
        "_op_num_cores",
        lambda op: 1 if op is producer else 32,
    )

    plans = collect_lx_relayout_plans(graph)

    assert len(plans) == 1
    plan = plans[0]
    assert plan.collective_kind is LXCollectiveKind.BROADCAST
    assert plan.source_core_id_to_device_slice == {"0": {}}
    assert len(plan.destination_core_id_to_device_slice) == 32
    assert all(
        not per_core for per_core in plan.destination_core_id_to_device_slice.values()
    )


def test_broadcast_emits_one_to_all_standard_shuffle():
    m = Symbol("m")
    n = Symbol("n")
    plan = _broadcast_plan()
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[2, 64],
        device_coordinates=[floor(n / 64), Mod(n, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="add",
        is_reduction=False,
        iteration_space={m: (Integer(32), 32), n: (Integer(128), 1)},
        args=[source_arg, source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, consumer_arg = result
    assert shuffle_spec.op == "shuffle"
    assert shuffle_spec.num_cores_override == 32
    assert consumer_arg.allocation == {"lx": plan.destination_lx_address}
    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32

    producer_geometry = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    consumer_geometry = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert producer_geometry == {"0": {"out": 0}}
    assert len(consumer_geometry) == 32
    assert all(per_core == {"out": 0} for per_core in consumer_geometry.values())


def test_dense_common_refinement_geometries():
    producer_8x4 = {str(core): {"0": core // 4, "1": core % 4} for core in range(32)}
    partition_4x8 = {str(core): {"0": core // 8, "1": core % 8} for core in range(32)}
    partition_32x1 = {str(core): {"0": core, "1": 0} for core in range(32)}

    assert (
        _destination_size_ratio(
            producer_8x4,
            {"0": 8, "1": 4},
            partition_4x8,
            {"0": 4, "1": 8},
        )
        == 1
    )
    assert (
        _destination_size_ratio(
            partition_4x8,
            {"0": 4, "1": 8},
            partition_32x1,
            {"0": 32, "1": 1},
        )
        == 1
    )

    incomplete_partition = {
        str(core): {"0": core // 8, "1": core % 8} for core in range(32)
    }
    assert (
        _destination_size_ratio(
            incomplete_partition,
            {"0": 8, "1": 8},
            partition_32x1,
            {"0": 32, "1": 1},
        )
        is None
    )


def test_granite_mlp_common_refinement_emits_standard_shuffle():
    m = Symbol("m")
    n = Symbol("n")
    plan = _granite_mlp_all_to_all_plan()
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 200, 64],
        device_coordinates=[m, floor(n / 64), Mod(n, 64)],
        allocation={"lx": 0x24000},
        name=plan.source_name,
    )
    consumer_spec = OpSpec(
        op="silu",
        is_reduction=False,
        iteration_space={m: (Integer(512), 32), n: (Integer(12800), 1)},
        args=[source_arg],
        op_info={},
    )

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, consumer_arg = result
    root, allocations = _compile_shuffle(shuffle_spec)
    assert root["numCoresUsed_"] == 32
    assert consumer_arg.name == plan.destination_name
    assert consumer_arg.allocation == {"lx": plan.destination_lx_address}

    producer_geometry = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    consumer_geometry = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert producer_geometry["0"] == {"mb": 0, "out": 0}
    assert producer_geometry["31"] == {"mb": 3, "out": 7}
    assert consumer_geometry["0"] == {"mb": 0, "out": 0}
    assert consumer_geometry["31"] == {"mb": 31, "out": 0}


def test_expanding_geometry_is_allocated_atomically_or_falls_back():
    graph = _DummyGraph("producer", "independent_1", "independent_2", "consumer")
    # GraphEditor may insert operations without updating GraphLowering's buffer map.
    graph.name_to_buffer.pop("consumer")
    plan = _all_gather_plan()
    source = LifetimeBoundBuffer("buf_k", size=128 * 1024, uses=[0, 3])
    consumer_output = LifetimeBoundBuffer(
        "scores", size=512 * 1024, uses=[3, 4], in_place_parents=["buf_k"]
    )
    barred = LifetimeBoundBuffer(
        "barred", size=128, uses=[0, 4], residency_reason="op not allowed"
    )
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    buffers = [barred, source, consumer_output]
    allocator._append_lx_relayout_destinations(graph, buffers)

    destination = {buffer.name: buffer for buffer in buffers}[plan.destination_name]
    assert destination.size == 1024 * 1024
    assert source.uses == [0, 3]
    assert destination.uses == [3, 4]
    assert consumer_output.uses == [4, 5]
    assert consumer_output.in_place_parents == []
    layout_planning = allocator._layout_planner_for_buffers(buffers)
    assert isinstance(layout_planning, FirstFitLayoutSolver)
    allocation = allocator._plan_layout_with_atomic_relayouts(layout_planning, buffers)
    by_name = {buffer.name: buffer for buffer in allocation}
    assert by_name["barred"].address is None
    assert by_name["barred"].residency_reason == "op not allowed"
    assert all(
        buffer.address is not None for buffer in allocation if buffer.name != "barred"
    )
    assert not _overlap(by_name["buf_k"], by_name[plan.destination_name])
    assert _overlap(by_name["buf_k"], by_name["scores"])
    allocator._record_successful_lx_relayouts(graph, allocation)
    recorded = getattr(graph.operations[3], LX_RELAYOUT_ATTR)["buf_k"]
    assert recorded.destination_lx_address == by_name[plan.destination_name].address

    for buffer in buffers:
        buffer.address = None
    fallback_allocator = ScratchpadAllocator(GreedyLayoutSolver(1024 * 1024))
    fallback_allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    allocation = fallback_allocator._plan_layout_with_atomic_relayouts(
        FirstFitLayoutSolver(1024 * 1024), buffers
    )
    by_name = {buffer.name: buffer for buffer in allocation}
    assert by_name["buf_k"].address is None
    assert by_name[plan.destination_name].address is None
    assert by_name["scores"].address is not None
    fallback_allocator._record_successful_lx_relayouts(graph, allocation)
    assert not hasattr(graph.operations[3], LX_RELAYOUT_ATTR)


def test_broadcast_destination_is_full_sized_and_disjoint():
    graph = _DummyGraph("producer", "consumer")
    plan = _broadcast_plan()
    source = LifetimeBoundBuffer(plan.source_name, size=128 * 1024, uses=[0, 1])
    consumer_output = LifetimeBoundBuffer(
        "consumer_output",
        size=128 * 1024,
        uses=[1, 2],
        in_place_parents=[plan.source_name],
    )
    allocator = ScratchpadAllocator(GreedyLayoutSolver(384 * 1024))
    allocator._lx_relayout_plans_by_source = {plan.source_name: plan}
    buffers = [source, consumer_output]

    allocator._append_lx_relayout_destinations(graph, buffers)

    destination = {buffer.name: buffer for buffer in buffers}[plan.destination_name]
    assert destination.size == source.size
    assert source.uses == [0, 1]
    assert destination.uses == [1, 2]
    assert consumer_output.uses == [2, 3]
    assert consumer_output.in_place_parents == []

    layout_planning = allocator._layout_planner_for_buffers(buffers)
    allocation = allocator._plan_layout_with_atomic_relayouts(layout_planning, buffers)
    by_name = {buffer.name: buffer for buffer in allocation}
    assert all(buffer.address is not None for buffer in allocation)
    assert not _overlap(by_name[plan.source_name], by_name[plan.destination_name])

    allocator._record_successful_lx_relayouts(graph, allocation)
    recorded = getattr(graph.operations[1], LX_RELAYOUT_ATTR)[plan.source_name]
    assert recorded.collective_kind is LXCollectiveKind.BROADCAST
    assert recorded.destination_lx_address == destination.address


def test_relayout_keeps_producer_inputs_live_through_shuffle(monkeypatch):
    graph = _DummyGraph("producer_input", "buf_k", "consumer")
    plan = _all_gather_plan()
    producer_input = LifetimeBoundBuffer("producer_input", size=128, uses=[0, 1])
    source = LifetimeBoundBuffer("buf_k", size=128, uses=[1, 2])
    producer_child = LifetimeBoundBuffer(
        "producer_child", size=128, uses=[1, 2], in_place_parents=["producer_input"]
    )
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": plan}
    monkeypatch.setattr(
        allocator_module,
        "op_read_writes",
        lambda _: SimpleNamespace(reads=[SimpleNamespace(name="producer_input")]),
    )

    allocator._append_lx_relayout_destinations(
        graph, [producer_input, source, producer_child]
    )

    assert producer_input.uses == [0, 1, 2]
    assert producer_child.in_place_parents == []


def test_partial_relayout_fallback_keeps_complete_pairs():
    complete_plan = replace(
        _all_gather_plan(), source_name="complete", consumer_name="consumer_a"
    )
    incomplete_plan = replace(
        _all_gather_plan(), source_name="incomplete", consumer_name="consumer_b"
    )
    complete_source = LifetimeBoundBuffer("complete", size=128, uses=[0, 1])
    complete_destination = LifetimeBoundBuffer(
        complete_plan.destination_name, size=128, uses=[1, 2]
    )
    incomplete_source = LifetimeBoundBuffer("incomplete", size=128, uses=[2, 3])
    incomplete_destination = LifetimeBoundBuffer(
        incomplete_plan.destination_name, size=128, uses=[3, 4]
    )
    dependent = LifetimeBoundBuffer(
        "dependent",
        size=128,
        uses=[3, 4],
        in_place_parents=["incomplete"],
    )
    buffers = [
        complete_source,
        complete_destination,
        incomplete_source,
        incomplete_destination,
        dependent,
    ]

    class _PartialLayout:
        spill_reasons = {}

        def plan_layout(self, planned_buffers, log_lx_usage=False):
            del log_lx_usage
            addresses = {
                "complete": 0,
                complete_plan.destination_name: 256,
                "incomplete": 512,
                incomplete_plan.destination_name: None,
                "dependent": 640,
            }
            for buffer in planned_buffers:
                buffer.address = addresses[buffer.name]
            return list(planned_buffers)

    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {
        "complete": complete_plan,
        "incomplete": incomplete_plan,
    }
    allocation = allocator._plan_layout_with_atomic_relayouts(_PartialLayout(), buffers)
    by_name = {buffer.name: buffer for buffer in allocation}

    assert by_name["complete"].address == 0
    assert by_name[complete_plan.destination_name].address == 256
    assert by_name["incomplete"].address is None
    assert by_name[incomplete_plan.destination_name].address is None
    assert by_name["dependent"].address == 640
    assert by_name["dependent"].in_place_parents == []
    assert allocator._lx_relayout_plans_by_source == {"complete": complete_plan}


def test_planned_restickify_source_is_eligible_for_lx_reuse(monkeypatch):
    class _MutationLayout:
        pass

    class _ComputedBuffer:
        def __init__(self, name, layout=None):
            self.name = name
            self.layout = SimpleNamespace() if layout is None else layout

        def get_name(self):
            return self.name

    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": _all_gather_plan()}
    source = _ComputedBuffer("buf_k")
    unrelated = _ComputedBuffer("unrelated")
    mutated_source = _ComputedBuffer("buf_k", _MutationLayout())
    monkeypatch.setattr(allocator_module, "ComputedBuffer", _ComputedBuffer)
    monkeypatch.setattr(
        allocator_module,
        "MutationLayoutSHOULDREMOVE",
        _MutationLayout,
    )
    monkeypatch.setattr(
        allocator_module.config,
        "allow_all_ops_in_lx_planning",
        False,
    )
    monkeypatch.setattr(allocator, "_get_op_name", lambda _: "restickify")

    assert allocator._op_output_good_for_lx_reuse(source)
    assert not allocator._op_output_good_for_lx_reuse(unrelated)
    assert not allocator._op_output_good_for_lx_reuse(mutated_source)


def test_relayout_source_layout_checks_skip_only_rewritten_consumers():
    graph = _DummyGraph(
        "buf_k", "consumer_a", "unrelated_consumer", "consumer_b"
    )
    plan_a = replace(_all_gather_plan(), consumer_name="consumer_a")
    plan_b = replace(_all_gather_plan(), consumer_name="consumer_b")
    allocator = ScratchpadAllocator(GreedyLayoutSolver(1536 * 1024))
    allocator._lx_relayout_plans_by_source = {"buf_k": plan_b}
    allocator._lx_relayout_alias_plans_by_source = {
        "buf_k": [plan_a, plan_b]
    }

    assert allocator._uses_after_planned_lx_relayouts(
        graph, "buf_k", [0, 1, 2, 3]
    ) == [0, 2]
    assert allocator._uses_after_planned_lx_relayouts(
        graph, "not_a_relayout_source", [0, 1, 2, 3]
    ) == [0, 1, 2, 3]


def test_graph_output_replacement_preserves_reinterpret_view(monkeypatch):
    class _Buffer:
        def __init__(self, name):
            self.name = name

        def get_name(self):
            return self.name

    class _Wrapper:
        def __init__(self, data):
            self.data = data

    class _TensorBox(_Wrapper):
        pass

    class _StorageBox(_Wrapper):
        pass

    class _ReinterpretView(_Wrapper):
        def __init__(self, *, data, layout):
            super().__init__(data)
            self.layout = layout

    monkeypatch.setattr(graph_editor_module, "Buffer", _Buffer)
    monkeypatch.setattr(graph_editor_module, "TensorBox", _TensorBox)
    monkeypatch.setattr(graph_editor_module, "StorageBox", _StorageBox)
    monkeypatch.setattr(graph_editor_module, "ReinterpretView", _ReinterpretView)

    old = _Buffer("buf29")
    replacement = _Buffer("buf29_hbm_clone")
    view_layout = object()
    graph_output = _TensorBox(
        _StorageBox(
            _ReinterpretView(data=_StorageBox(old), layout=view_layout)
        )
    )
    editor = object.__new__(GraphEditor)
    editor.lowering = SimpleNamespace(graph_outputs=[graph_output])

    editor.change_graph_output(old, replacement)

    replaced = editor.lowering.graph_outputs[0]
    assert isinstance(replaced, _TensorBox)
    assert isinstance(replaced.data, _StorageBox)
    assert isinstance(replaced.data.data, _ReinterpretView)
    assert replaced.data.data.layout is view_layout
    assert isinstance(replaced.data.data.data, _StorageBox)
    assert replaced.data.data.data.data is replacement


def test_all_gather_emits_standard_shuffle_fold_geometry():
    h = Symbol("h")
    lq = Symbol("lq")
    lk = Symbol("lk")
    d = Symbol("d")
    plan = _all_gather_plan()
    plan = replace(plan, destination_lx_address=0x44000)
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[4, 4096, 2, 64],
        device_coordinates=[h, lk, floor(d / 64), Mod(d, 64)],
        allocation={"lx": 0x24000},
        name="buf_k",
    )
    consumer_spec = OpSpec(
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
        gather_dim=lq,
    )

    result = _materialize_explicit_lx_shuffle(
        source_arg,
        consumer_spec,
        plan,
    )
    assert result is not None
    shuffle_spec, consumer_arg = result
    assert shuffle_spec.gather_dim is None
    assert shuffle_spec.replicas_contiguous
    assert consumer_arg.allocation == {"lx": 0x44000}
    assert shuffle_spec.num_cores_override == 32
    root, allocations = _compile_shuffle(shuffle_spec)

    head_dim = next(
        dim for dim, splits in root["numWkSlicesPerDim_"].items() if splits == 4
    )
    assert [root["coreIdToWkSlice_"][str(core)][head_dim] for core in range(32)] == [
        head for head in range(4) for _ in range(8)
    ]

    input_map = allocations[0]["coordinates_"]["coreIdToWkSlice_"]
    output_map = allocations[1]["coordinates_"]["coreIdToWkSlice_"]
    assert len(input_map) == len(output_map) == 32
    assert input_map["0"] != input_map["4"]
    assert output_map["0"] == output_map["4"]

    input_out = allocations[0]["coordinates_"]["coordInfo"]["out"]["folds"]
    output_out = allocations[1]["coordinates_"]["coordInfo"]["out"]["folds"]
    assert (
        input_out["dim_prop_attr"][0]["factor_"],
        input_out["dim_prop_func"][0]["Affine"]["alpha_"],
    ) == (8, 512)
    assert (
        output_out["dim_prop_attr"][0]["factor_"],
        output_out["dim_prop_func"][0]["Affine"]["alpha_"],
    ) == (1, 4096)

    stick_folds = allocations[0]["coordinates_"]["coordInfo"]["in"]["folds"]
    assert (
        stick_folds["dim_prop_attr"][1]["factor_"],
        stick_folds["dim_prop_func"][1]["Affine"]["alpha_"],
    ) == (1, 0)


def test_compact_v_all_gather_maps_flattened_head_axis(monkeypatch):
    kv_head = Symbol("kv_head")
    query_group = Symbol("query_group")
    query_token = Symbol("query_token")
    head_dim = Symbol("head_dim")
    kv_token = Symbol("kv_token")
    plan = replace(
        _all_gather_plan(),
        source_name="buf29",
        source_core_id_to_device_slice={
            str(core): {"0": core // 4, "1": core % 4}
            for core in range(32)
        },
        destination_core_id_to_device_slice={
            str(core): {"0": 0, "1": 0} for core in range(32)
        },
        source_device_dim_splits={"0": 8, "1": 4},
        destination_device_dim_splits={"0": 1, "1": 1},
        destination_size_ratio=32,
        destination_lx_address=0x20000,
    )
    source_arg = TensorArg(
        is_input=True,
        arg_index=-1,
        device_dtype=DataFormats.SEN169_FP16,
        device_size=[512, 16, 1, 64],
        device_coordinates=[
            kv_token,
            2 * kv_head + floor(head_dim / 64),
            Integer(0),
            Mod(head_dim, 64),
        ],
        allocation={"lx": 0},
        name="buf29",
    )
    consumer_spec = OpSpec(
        op=BATCH_MATMUL_OP,
        is_reduction=True,
        iteration_space={
            kv_head: (Integer(8), 1),
            query_group: (Integer(4), 1),
            query_token: (Integer(512), 32),
            head_dim: (Integer(128), 1),
            kv_token: (Integer(512), 1),
        },
        args=[source_arg],
        op_info={},
    )
    monkeypatch.setattr(config, "relayout_oracle_compact_gqa", True)

    result = _materialize_explicit_lx_shuffle(source_arg, consumer_spec, plan)

    assert result is not None
    shuffle_spec, consumer_arg = result
    assert shuffle_spec.iteration_space == {
        kv_head: (Integer(8), 1),
        head_dim: (Integer(128), 1),
        kv_token: (Integer(512), 1),
    }
    assert shuffle_spec.args[0].allocation_device_dim_splits == {"0": 8, "1": 4}
    expected_source_map = {
        str(core): {"0": core % 8, "1": core // 8} for core in range(32)
    }
    assert (
        shuffle_spec.args[0].allocation_core_id_to_device_slice
        == expected_source_map
    )
    _, allocations = _compile_shuffle(shuffle_spec)
    assert allocations[0]["coordinates_"]["coreIdToWkSlice_"] == {
        str(core): {"in": core % 8, "out": 0, "y": core // 8}
        for core in range(32)
    }
    assert allocations[1]["coordinates_"]["coreIdToWkSlice_"] == {
        str(core): {"in": 0, "out": 0, "y": 0} for core in range(32)
    }
    assert consumer_arg.name == plan.destination_name
    assert consumer_arg.allocation == {"lx": 0x20000}
