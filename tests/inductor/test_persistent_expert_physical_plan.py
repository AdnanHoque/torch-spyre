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

import pytest

from torch_spyre._inductor.errors import Unsupported
from torch_spyre._inductor.expert_execution import physical_plan


class _Layout:
    def __init__(self, address=0):
        self.allocation = {"lx": address}


class _Computed:
    def __init__(self, name, info):
        self.name = name
        self.loop_info = info

    def get_layout(self):
        return _Layout(8192 if self.name == "accumulator" else 0)

    def get_name(self):
        return self.name


class _Empty(_Computed):
    pass


def _info(role, *, bindings=()):
    return SimpleNamespace(
        execution_role=role,
        loop_group_id=(7,),
        preheader_for_group=(7,),
        counted_loop_plan=SimpleNamespace(
            kind="persistent_dense_expert", trip_count=128
        ),
        loop_operand_bindings=list(bindings),
    )


def _graph(advancing=4):
    roles = ("gate_weight", "up_weight", "down_weight", "routing_weight")
    requirements = [
        SimpleNamespace(
            host_advance_per_level=(1,),
            operand_role=roles[index] if index < len(roles) else "extra",
            source_name=f"arg{index}",
        )
        for index in range(advancing)
    ]
    body = _Computed("body", _info("loop_body", bindings=requirements))
    accumulator = _Empty("accumulator", _info(None))
    accumulator.loop_storage_plan = SimpleNamespace(owner_group=(7,))
    return SimpleNamespace(
        operations=[
            _Computed("x", _info("invariant_preheader")),
            _Computed("fill", _info("accumulator_fill")),
            body,
            accumulator,
            _Computed("drain", _info("output_drain")),
        ]
    )


@pytest.fixture(autouse=True)
def _fake_ir(monkeypatch):
    monkeypatch.setattr(physical_plan, "ComputedBuffer", _Computed)
    monkeypatch.setattr(physical_plan, "SpyreEmptyFallback", _Empty)
    monkeypatch.setattr(physical_plan, "FixedTiledLayout", _Layout)
    monkeypatch.setattr(physical_plan, "is_loop_carried_lx_storage", lambda op: True)
    monkeypatch.setattr(
        physical_plan,
        "is_persistent_loop_preheader",
        lambda op: op.get_name() == "x",
    )
    monkeypatch.setattr(physical_plan, "op_short_name", lambda op: "body")


def test_accepts_complete_persistent_physical_plan():
    physical_plan.verify_persistent_expert_physical_plan(_graph())


def test_rejects_missing_advancing_operand():
    with pytest.raises(Unsupported, match="exactly one advancing"):
        physical_plan.verify_persistent_expert_physical_plan(_graph(advancing=3))


def test_rejects_duplicate_advancing_operand_role():
    graph = _graph()
    bindings = graph.operations[2].loop_info.loop_operand_bindings
    bindings[-1].operand_role = "gate_weight"
    with pytest.raises(Unsupported, match="exactly one advancing"):
        physical_plan.verify_persistent_expert_physical_plan(graph)


def test_physical_verifier_does_not_choose_the_expert_envelope():
    graph = _graph()
    graph.operations[2].loop_info.counted_loop_plan.trip_count = 2
    physical_plan.verify_persistent_expert_physical_plan(graph)


def test_rejects_invalid_counted_loop_plan():
    graph = _graph()
    graph.operations[2].loop_info.counted_loop_plan.trip_count = 1
    with pytest.raises(Unsupported, match="consistent counted-loop plan"):
        physical_plan.verify_persistent_expert_physical_plan(graph)


def test_ignores_graphs_without_persistent_loop_roles():
    physical_plan.verify_persistent_expert_physical_plan(SimpleNamespace(operations=[]))
