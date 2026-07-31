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

from collections import Counter

import torch
from torch.fx import Graph

import torch_spyre._inductor.customops  # noqa: F401 -- register spyre ops
from torch_spyre._inductor.reuse_fp8_quantization import (
    reuse_fp8_activation_quantization,
)


def _emit_dynamic_quantization(
    graph: Graph,
    activation,
    clamp_max: float = 448.0,
    qfp8_target=torch.ops.spyre.qfp8mb.default,
):
    # This is the per-row scale structure emitted by the Granite FP8 frontend.
    viewed = graph.call_function(
        torch.ops.aten.view.default, args=(activation, [32, 64])
    )
    row_min = graph.call_function(
        torch.ops.aten.amin.default, args=(viewed, [-1], True)
    )
    row_max = graph.call_function(
        torch.ops.aten.amax.default, args=(viewed, [-1], True)
    )
    min_zero = graph.call_function(torch.ops.aten.zeros_like.default, args=(row_min,))
    max_zero = graph.call_function(torch.ops.aten.zeros_like.default, args=(row_max,))
    negative = graph.call_function(
        torch.ops.aten.minimum.default, args=(row_min, min_zero)
    )
    positive = graph.call_function(
        torch.ops.aten.maximum.default, args=(row_max, max_zero)
    )
    abs_min = graph.call_function(torch.ops.aten.neg.default, args=(negative,))
    abs_max = graph.call_function(
        torch.ops.aten.maximum.default, args=(abs_min, positive)
    )
    scale = graph.call_function(
        torch.ops.aten.div.Scalar, args=(abs_max, 448.0)
    )
    scale = graph.call_function(
        torch.ops.aten.clamp.default,
        args=(scale, torch.finfo(torch.float32).tiny, None),
    )

    # Use a separately-created view in the normalization branch, as FMS does.
    activation_for_quant = graph.call_function(
        torch.ops.aten.view.default, args=(activation, [32, 64])
    )
    inverse = graph.call_function(torch.ops.aten.reciprocal.default, args=(scale,))
    normalized = graph.call_function(
        torch.ops.aten.mul.Tensor, args=(activation_for_quant, inverse)
    )
    clamped = graph.call_function(
        torch.ops.spyre.clamp.default,
        args=(normalized, -448.0, clamp_max),
    )
    quantized = graph.call_function(qfp8_target, args=(clamped,))
    return quantized, scale


def _target_counts(graph: Graph) -> Counter:
    return Counter(str(node.target) for node in graph.nodes if node.op == "call_function")


def test_reuses_shared_granite_activation_quantization():
    graph = Graph()
    activation = graph.placeholder("activation")
    q, q_scale = _emit_dynamic_quantization(graph, activation)
    k, k_scale = _emit_dynamic_quantization(graph, activation)
    v, v_scale = _emit_dynamic_quantization(graph, activation)
    graph.output((q, q_scale, k, k_scale, v, v_scale))

    before = _target_counts(graph)
    assert before["spyre.qfp8mb.default"] == 3
    assert before["aten.amin.default"] == 3

    merged = reuse_fp8_activation_quantization(graph)

    after = _target_counts(graph)
    assert merged > 0
    assert after["spyre.qfp8mb.default"] == 1
    assert after["aten.amin.default"] == 1
    assert after["aten.amax.default"] == 1

    output = next(node for node in graph.nodes if node.op == "output").args[0]
    assert output[0] is output[2] is output[4]
    assert output[1] is output[3] is output[5]


def test_does_not_merge_different_activation_sources():
    graph = Graph()
    q_source = graph.placeholder("q_source")
    kv_source = graph.placeholder("kv_source")
    q, q_scale = _emit_dynamic_quantization(graph, q_source)
    kv, kv_scale = _emit_dynamic_quantization(graph, kv_source)
    graph.output((q, q_scale, kv, kv_scale))

    reuse_fp8_activation_quantization(graph)

    counts = _target_counts(graph)
    assert counts["spyre.qfp8mb.default"] == 2
    assert counts["aten.amin.default"] == 2


def test_does_not_merge_different_quantization_parameters():
    graph = Graph()
    activation = graph.placeholder("activation")
    first, _ = _emit_dynamic_quantization(graph, activation)
    second, _ = _emit_dynamic_quantization(graph, activation, clamp_max=240.0)
    graph.output((first, second))

    reuse_fp8_activation_quantization(graph)

    counts = _target_counts(graph)
    assert counts["spyre.qfp8mb.default"] == 2
    assert counts["spyre.clamp.default"] == 2


def test_reuses_decode_channel_packing_but_never_weight_packing():
    graph = Graph()
    activation = graph.placeholder("activation")
    decode0, _ = _emit_dynamic_quantization(
        graph, activation, qfp8_target=torch.ops.spyre.qfp8ch.default
    )
    decode1, _ = _emit_dynamic_quantization(
        graph, activation, qfp8_target=torch.ops.spyre.qfp8ch.default
    )
    weight0, _ = _emit_dynamic_quantization(
        graph, activation, qfp8_target=torch.ops.spyre.qfp8wt.default
    )
    weight1, _ = _emit_dynamic_quantization(
        graph, activation, qfp8_target=torch.ops.spyre.qfp8wt.default
    )
    graph.output((decode0, decode1, weight0, weight1))

    reuse_fp8_activation_quantization(graph)

    counts = _target_counts(graph)
    assert counts["spyre.qfp8ch.default"] == 1
    assert counts["spyre.qfp8wt.default"] == 2
