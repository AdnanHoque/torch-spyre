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

import torch
from torch import fx
from sympy import Symbol

from torch_spyre._inductor.constants import SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY
from torch_spyre._inductor.temp_passes import bmm_unflatten_pass, mm_to_bmm_pass

aten = torch.ops.aten


class _FakeVal:
    def __init__(self, shape):
        self.shape = shape


def _make_mm_to_bmm_graph(batch):
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((batch, 512, 4096), device="meta", dtype=torch.float16)
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty((4096, 12800), device="meta", dtype=torch.float16)

    flattened = graph.call_function(
        aten.reshape.default,
        args=(x, [batch * 512, 4096]),
    )
    flattened.meta["val"] = torch.empty(
        (batch * 512, 4096), device="meta", dtype=torch.float16
    )
    mm = graph.call_function(aten.mm.default, args=(flattened, y))
    mm.meta["val"] = torch.empty(
        (batch * 512, 12800), device="meta", dtype=torch.float16
    )
    output = graph.call_function(
        aten.reshape.default,
        args=(mm, [batch, 512, 12800]),
    )
    output.meta["val"] = torch.empty(
        (batch, 512, 12800), device="meta", dtype=torch.float16
    )
    graph.output(output)
    return fx.GraphModule({}, graph)


def _make_plain_bmm_graph(batch):
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((batch, 512, 4096), device="meta", dtype=torch.float16)
    y = graph.placeholder("y")
    y.meta["val"] = torch.empty(
        (batch, 4096, 12800), device="meta", dtype=torch.float16
    )
    bmm = graph.call_function(aten.bmm.default, args=(x, y))
    bmm.meta["val"] = torch.empty(
        (batch, 512, 12800), device="meta", dtype=torch.float16
    )
    graph.output(bmm)
    return fx.GraphModule({}, graph)


def _make_plain_bmm_graph_with_meta_shapes(lhs_shape, rhs_shape, out_shape):
    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = _FakeVal(lhs_shape)
    y = graph.placeholder("y")
    y.meta["val"] = _FakeVal(rhs_shape)
    bmm = graph.call_function(aten.bmm.default, args=(x, y))
    bmm.meta["val"] = _FakeVal(out_shape)
    graph.output(bmm)
    return fx.GraphModule({}, graph)


def _bmm_nodes(graph):
    return [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == aten.bmm.default
    ]


def test_unflatten_mm_to_bmm_marks_static_unit_batch_shared_weight():
    gm = _make_mm_to_bmm_graph(batch=1)

    assert mm_to_bmm_pass.apply(gm.graph) == 1
    gm.graph.lint()
    [bmm_node] = _bmm_nodes(gm.graph)

    assert bmm_node.meta["custom"][SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY] == {
        "batch_dim": 0
    }


def test_unflatten_mm_to_bmm_does_not_mark_non_unit_batch():
    gm = _make_mm_to_bmm_graph(batch=2)

    assert mm_to_bmm_pass.apply(gm.graph) == 1
    gm.graph.lint()
    [bmm_node] = _bmm_nodes(gm.graph)

    assert SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY not in (
        bmm_node.meta.get("custom") or {}
    )


def test_plain_bmm_marks_static_unit_batch_shared_weight():
    gm = _make_plain_bmm_graph(batch=1)

    bmm_unflatten_pass.apply(gm.graph)
    gm.graph.lint()
    [bmm_node] = _bmm_nodes(gm.graph)

    assert bmm_node.meta["custom"][SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY] == {
        "batch_dim": 0
    }


def test_plain_bmm_does_not_mark_non_unit_batch():
    gm = _make_plain_bmm_graph(batch=2)

    bmm_unflatten_pass.apply(gm.graph)
    gm.graph.lint()
    [bmm_node] = _bmm_nodes(gm.graph)

    assert SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY not in (
        bmm_node.meta.get("custom") or {}
    )


def test_plain_bmm_does_not_mark_dynamic_batch():
    batch = Symbol("batch")
    gm = _make_plain_bmm_graph_with_meta_shapes(
        (batch, 512, 4096),
        (batch, 4096, 12800),
        (batch, 512, 12800),
    )

    bmm_unflatten_pass.apply(gm.graph)
    gm.graph.lint()
    [bmm_node] = _bmm_nodes(gm.graph)

    assert SHARED_WEIGHT_UNIT_BMM_CUSTOM_META_KEY not in (
        bmm_node.meta.get("custom") or {}
    )
