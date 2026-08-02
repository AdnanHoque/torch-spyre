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

"""Reuse repeated dynamic FP8 activation-quantization work.

Granite's Q/K/V projections and its gate/up projections can quantize the same
activation independently.  After Spyre decompositions, each copy has the form

    activation, scale -> reciprocal -> mul -> clamp -> qfp8ch/qfp8mb

and a dynamically-computed scale may itself contain a repeated amin/amax chain.
This pass performs common-subexpression elimination only inside those matched
activation-quantization slices.  It deliberately does not run general graph
CSE, does not touch qfp8wt weight packing, and uses the source activation node
as an identity boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch.fx import Graph, Node

from .logging_utils import get_inductor_logger


logger = get_inductor_logger("reuse_fp8_quantization")


_QFP8_ACTIVATION_TARGETS = {
    "spyre.qfp8ch.default",
    "spyre.qfp8mb.default",
}

_CLAMP_TARGETS = {
    "aten.clamp.default",
    "aten.clamp.Tensor",
    "spyre.clamp.default",
}

_MUL_TARGETS = {
    "aten.mul.Tensor",
    "aten.mul.Scalar",
}

_RECIPROCAL_TARGETS = {
    "aten.reciprocal.default",
}

# Operations which may sit between the logical activation and the normalization
# multiply.  Peeling these lets separately-created views of one activation use
# the same identity boundary without walking into the producer of that
# activation (for example, Granite's RMSNorm output).
_TRANSPARENT_ACTIVATION_TARGETS = {
    "aten._to_copy.default",
    "aten._unsafe_view.default",
    "aten.alias.default",
    "aten.clone.default",
    "aten.contiguous.default",
    "aten.detach.default",
    "aten.permute.default",
    "aten.reshape.default",
    "aten.squeeze.default",
    "aten.squeeze.dim",
    "aten.transpose.int",
    "aten.unsqueeze.default",
    "aten.view.default",
    "prims.convert_element_type.default",
}

# This is intentionally a closed allowlist derived from the dynamic per-row
# E4M3 scale chain emitted by FMS.  Keeping it closed prevents the pass from
# becoming an accidental whole-graph CSE implementation.
_SCALE_DERIVATION_TARGETS = _TRANSPARENT_ACTIVATION_TARGETS | {
    "aten.abs.default",
    "aten.amax.default",
    "aten.amin.default",
    "aten.clamp.default",
    "aten.clamp.Tensor",
    "aten.div.Scalar",
    "aten.div.Tensor",
    "aten.full_like.default",
    "aten.maximum.default",
    "aten.minimum.default",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.neg.default",
    "aten.reciprocal.default",
    "aten.zeros_like.default",
    "spyre.clamp.default",
    # The optimized DD2 path replaces the generic abs/max/divide/clamp slice
    # with this pure one-input reduction.  Treat it as the same kind of scale
    # derivation node so repeated Q/K/V consumers can share the specialized
    # result as well as the downstream normalized/packed activation.
    "spyre.quant_scale_per_token_fp8.default",
}


def _target_name(node: Node) -> str:
    return str(node.target) if node.op == "call_function" else ""


def _node_args(value: Any) -> Iterable[Node]:
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _node_args(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _node_args(item)


def _first_node_arg(node: Node) -> Node | None:
    return next(_node_args(node.args), None)


def _peel_activation_views(node: Node) -> tuple[Node, set[Node]]:
    wrappers: set[Node] = set()
    while _target_name(node) in _TRANSPARENT_ACTIVATION_TARGETS:
        next_node = _first_node_arg(node)
        if next_node is None:
            break
        wrappers.add(node)
        node = next_node
    return node, wrappers


def _collect_scale_derivation(
    node: Node,
    activation_source: Node,
    candidates: set[Node],
) -> None:
    """Collect the allowlisted scale slice, stopping at semantic leaves."""

    if node is activation_source or node in candidates:
        return
    if _target_name(node) not in _SCALE_DERIVATION_TARGETS:
        return

    candidates.add(node)
    for arg in _node_args((node.args, node.kwargs)):
        _collect_scale_derivation(arg, activation_source, candidates)


def _match_quantization_slice(root: Node) -> set[Node] | None:
    """Return the CSE-eligible nodes for a decomposed activation QFP8 chain."""

    if _target_name(root) not in _QFP8_ACTIVATION_TARGETS:
        return None

    clamp = _first_node_arg(root)
    if clamp is None or _target_name(clamp) not in _CLAMP_TARGETS:
        return None

    mul = _first_node_arg(clamp)
    if mul is None or _target_name(mul) not in _MUL_TARGETS:
        return None

    mul_inputs = list(_node_args(mul.args))
    reciprocal_inputs = [
        node for node in mul_inputs if _target_name(node) in _RECIPROCAL_TARGETS
    ]
    if len(reciprocal_inputs) != 1 or len(mul_inputs) != 2:
        return None

    reciprocal = reciprocal_inputs[0]
    activation = next(node for node in mul_inputs if node is not reciprocal)
    scale = _first_node_arg(reciprocal)
    if scale is None:
        return None

    activation_source, activation_wrappers = _peel_activation_views(activation)
    candidates = {root, clamp, mul, reciprocal, *activation_wrappers}
    _collect_scale_derivation(scale, activation_source, candidates)
    return candidates


def _arg_key(value: Any) -> Any:
    """Build a hashable exact-expression key for an FX argument."""

    if isinstance(value, Node):
        return ("node", id(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_arg_key(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_arg_key(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(sorted((str(key), _arg_key(item)) for key, item in value.items())),
        )
    if isinstance(value, slice):
        return (
            "slice",
            _arg_key(value.start),
            _arg_key(value.stop),
            _arg_key(value.step),
        )
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return ("value", value)


def _expression_key(node: Node) -> tuple[Any, ...]:
    return (
        node.op,
        node.target,
        _arg_key(node.args),
        _arg_key(node.kwargs),
    )


def reuse_fp8_activation_quantization(graph: Graph) -> int:
    """Share repeated dynamic activation quantization and return merge count.

    The rewrite is exact: nodes are merged only when their operator, arguments,
    keyword arguments, constants, and source-node identities match after earlier
    nodes in the same quantization slice have been canonicalized.
    """

    candidates: set[Node] = set()
    for node in graph.nodes:
        matched = _match_quantization_slice(node)
        if matched is not None:
            candidates.update(matched)

    canonical_by_expression: dict[tuple[Any, ...], Node] = {}
    merged = 0
    for node in list(graph.nodes):
        if node not in candidates:
            continue

        key = _expression_key(node)
        canonical = canonical_by_expression.get(key)
        if canonical is None:
            canonical_by_expression[key] = node
            continue

        node.replace_all_uses_with(canonical)
        merged += 1

    # Erase only dead nodes which were part of a matched quantization slice.
    # Avoid graph.eliminate_dead_code(), which could alter unrelated graph state.
    for node in reversed(list(graph.nodes)):
        if node in candidates and not node.users:
            graph.erase_node(node)

    if merged:
        logger.debug("reused %d FP8 activation-quantization nodes", merged)
    return merged
