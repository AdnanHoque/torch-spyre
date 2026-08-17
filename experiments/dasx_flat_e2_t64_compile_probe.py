#!/usr/bin/env python3
"""Reduced flat-expert-loop D-AS-X structure and correctness probe.

By default no generated kernel is launched. ``--device-correctness`` executes
the same compiled callable with two nonbinary alpha payloads after the exact
structural gate. The accepted structure is one static E=2 loop over the full
X[64,64] operand, a preheader X copy, expert weights and runtime alpha[E,T,1]
that advance per expert, an all-LX activation path, a fixed LX expert-sum
accumulator, and one post-loop drain to the final HBM output.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
import math
import pathlib
import re
import shutil
from unittest.mock import patch

import torch
from torch._inductor.utils import run_and_get_code

import torch_spyre
from torch_spyre._inductor import config, spyre_hint
import torch_spyre._inductor.wsr.propagate_named_dims as pnd
import torch_spyre.execution.async_compile as async_compile


def dense_activation_stationary_ffn(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    # One temporal loop only: experts. T is the full resident activation tile,
    # not a second temporal coarse-tile loop.
    with spyre_hint(num_tiles_per_dim={"E": 2}):
        gate = torch.ops.spyre.activation_stationary_shared_lhs_mm(x, gate_w)
        up = torch.ops.spyre.activation_stationary_shared_lhs_mm(x, up_w)
        hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
        down = torch.bmm(hidden, down_w)
        # Keep the mathematically required non-binary weight after down.  The
        # explicit singleton H dimension is a zero-stick scalar that broadcasts
        # over the native H-stick down output without restickification.
        return (down * alpha).sum(dim=0)


def _check_cpu_weighting_semantics() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    down = torch.randn((2, 7, 5), generator=generator)
    alpha = torch.rand((2, 7, 1), generator=generator) * 0.8 + 0.1
    actual = (down * alpha).sum(dim=0)
    expected = sum(down[expert] * alpha[expert] for expert in range(2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _keyword(call: ast.Call, name: str) -> ast.AST:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"{getattr(call.func, 'id', call.func)!r} has no {name=}")


def _has_keyword(call: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in call.keywords)


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _literal(node: ast.AST):
    return ast.literal_eval(node)


def _debug_chain(op: ast.Call) -> tuple[str, ...]:
    debug = _keyword(op, "debug_handle")
    assert isinstance(debug, ast.Call) and _call_name(debug) == "DebugHandle"
    return tuple(_literal(_keyword(debug, "ir_chain")))


def _debug_aten_op(op: ast.Call) -> str:
    debug = _keyword(op, "debug_handle")
    assert isinstance(debug, ast.Call) and _call_name(debug) == "DebugHandle"
    return _literal(_keyword(debug, "aten_op"))


def _tensor_args(op: ast.Call) -> list[dict]:
    args_node = _keyword(op, "args")
    assert isinstance(args_node, ast.List)
    parsed = []
    for tensor in args_node.elts:
        assert isinstance(tensor, ast.Call) and _call_name(tensor) == "TensorArg"
        parsed.append(
            {
                "is_input": _literal(_keyword(tensor, "is_input")),
                "arg_index": _literal(_keyword(tensor, "arg_index")),
                "device_size": list(_literal(_keyword(tensor, "device_size"))),
                "allocation": dict(_literal(_keyword(tensor, "allocation"))),
                "device_coordinates": [
                    ast.unparse(coordinate)
                    for coordinate in _keyword(tensor, "device_coordinates").elts
                ],
                "has_device_tile_advance": _has_keyword(
                    tensor, "device_tile_advance_expr"
                ),
            }
        )
    return parsed


def _lx_interval(tensor: dict) -> tuple[int, int] | None:
    allocation = tensor["allocation"]
    if "lx" not in allocation:
        return None
    # Spyre device_size's final coordinate is the 128-byte stick payload.
    size_bytes = math.prod(tensor["device_size"][:-1]) * 128
    start = allocation["lx"]
    return start, start + size_bytes


def _overlaps(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def _sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analyze_generated_structure(source: str) -> dict:
    """Fail closed on the C1 flat-E X-lifetime mechanism contract."""

    tree = ast.parse(source)
    sdsc_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sdsc"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "async_compile"
    ]
    assert len(sdsc_calls) == 1, (
        f"expected one async SDSC bundle, got {len(sdsc_calls)}"
    )
    sdsc_call = sdsc_calls[0]
    specs_node = sdsc_call.args[1]
    assert isinstance(specs_node, ast.List)
    top_specs = specs_node.elts

    loops = [
        spec
        for spec in top_specs
        if isinstance(spec, ast.Call) and _call_name(spec) == "LoopSpec"
    ]
    assert len(loops) == 1, f"expected one flat expert LoopSpec, got {len(loops)}"
    loop = loops[0]
    count = _keyword(loop, "count")
    assert (
        isinstance(count, ast.Call)
        and _call_name(count) == "sympify"
        and _literal(count.args[0]) == "2"
    ), "expert LoopSpec must have count 2"
    body_node = _keyword(loop, "body")
    assert isinstance(body_node, ast.List)
    assert not any(
        isinstance(node, ast.Call) and _call_name(node) == "LoopSpec"
        for body_spec in body_node.elts
        for node in ast.walk(body_spec)
    ), "temporal/nested loop found inside flat E loop"

    op_specs = [
        spec
        for spec in top_specs
        if isinstance(spec, ast.Call) and _call_name(spec) == "OpSpec"
    ]
    loop_ops = [
        spec
        for spec in body_node.elts
        if isinstance(spec, ast.Call) and _call_name(spec) == "OpSpec"
    ]
    local_sum_ops = [
        op
        for op in [*op_specs, *loop_ops]
        if _literal(_keyword(op, "op")).lower() in {"sum", "sumnonstick"}
    ]
    assert not local_sum_ops, (
        "the unit expert tile must not emit a local sum OpSpec; "
        f"found {[(_literal(_keyword(op, 'op')), _debug_chain(op)) for op in local_sum_ops]}"
    )

    x_copies = [
        op
        for op in op_specs
        if any(
            name.startswith("coarse_tile_read_copy_0_arg0") for name in _debug_chain(op)
        )
    ]
    assert len(x_copies) == 1, (
        f"expected one physical X preheader copy, got {len(x_copies)}"
    )
    x_copy = x_copies[0]
    assert top_specs.index(x_copy) < top_specs.index(loop)
    x_args = _tensor_args(x_copy)
    assert len(x_args) == 2
    assert x_args[0]["is_input"] and x_args[0]["arg_index"] == 0
    assert x_args[0]["allocation"].keys() == {"hbm"}
    assert not x_args[0]["has_device_tile_advance"]
    assert not x_args[1]["is_input"] and x_args[1]["allocation"].keys() == {"lx"}
    assert x_args[1]["arg_index"] == -1
    assert not x_args[1]["has_device_tile_advance"]
    assert x_args[1]["device_size"] == [1, 64, 64]
    x_interval = _lx_interval(x_args[1])
    assert x_interval is not None

    # Arg 0 is X.  It must cross HBM->LX exactly once in the whole bundle, and
    # unlike expert weights it must never carry a per-expert tile advance.
    x_hbm_sources = [
        (op, index, tensor)
        for op in [*op_specs, *loop_ops]
        for index, tensor in enumerate(_tensor_args(op))
        if tensor["is_input"]
        and tensor["arg_index"] == 0
        and tensor["allocation"].keys() == {"hbm"}
    ]
    assert len(x_hbm_sources) == 1
    assert x_hbm_sources[0][0] is x_copy and x_hbm_sources[0][1] == 0

    shared_lhs = [
        op
        for op in loop_ops
        if _call_name(op) == "OpSpec"
        and _literal(_keyword(op, "op")) == "batchmatmul"
        and _debug_chain(op)[0].startswith("activation_stationary_shared_lhs_mm")
    ]
    assert len(shared_lhs) == 2, (
        f"expected gate/up shared-LHS matmuls, got {len(shared_lhs)}"
    )
    for op in shared_lhs:
        tensors = _tensor_args(op)
        assert tensors[0]["is_input"]
        assert tensors[0]["arg_index"] == -1
        assert _lx_interval(tensors[0]) == x_interval
        assert not tensors[0]["has_device_tile_advance"]

    # The preheader allocation must remain reserved through every top-level op
    # after it as well as through the loop.  This catches a fusion-boundary or
    # allocator regression that aliases X outside the LoopSpec body.
    for op in op_specs:
        if op is x_copy:
            continue
        for index, tensor in enumerate(_tensor_args(op)):
            interval = _lx_interval(tensor)
            assert interval is None or not _overlaps(interval, x_interval), (
                f"top-level allocation {interval} aliases persistent X "
                f"{x_interval}; op={_debug_chain(op)} tensor_index={index}"
            )

    # Re-parse each loop TensorArg and reject every X-overlapping allocation
    # except the first input of the two shared-LHS matmuls.  This catches the
    # previous alpha-copy overwrite at exactly the X base address.
    for op in loop_ops:
        tensors = _tensor_args(op)
        for index, tensor in enumerate(tensors):
            interval = _lx_interval(tensor)
            if interval is None or not _overlaps(interval, x_interval):
                continue
            legitimate_x_read = (
                op in shared_lhs and index == 0 and interval == x_interval
            )
            assert legitimate_x_read, (
                f"loop allocation {interval} aliases persistent X {x_interval}; "
                f"op={_debug_chain(op)} tensor_index={index}"
            )

    fills = [
        op for op in op_specs if any("coarse_tile_fill" in n for n in _debug_chain(op))
    ]
    combines = [
        op
        for op in loop_ops
        if any("coarse_tile_combine" in n for n in _debug_chain(op))
    ]
    drains = [
        op
        for op in op_specs
        if any("coarse_tile_reduce_copy" in n for n in _debug_chain(op))
    ]
    assert len(fills) == 1 and top_specs.index(x_copy) < top_specs.index(
        fills[0]
    ) < top_specs.index(loop)
    assert len(combines) == 1
    assert len(drains) == 1 and top_specs.index(loop) < top_specs.index(drains[0])

    fill_args = _tensor_args(fills[0])
    combine_args = _tensor_args(combines[0])
    drain_args = _tensor_args(drains[0])
    assert len(fill_args) == 2
    assert len(combine_args) == 3
    assert len(drain_args) == 2
    assert not fill_args[1]["is_input"]
    accum_interval = _lx_interval(fill_args[1])
    assert accum_interval is not None
    assert combine_args[0]["is_input"]
    assert not combine_args[2]["is_input"]
    assert _lx_interval(combine_args[0]) == accum_interval
    assert _lx_interval(combine_args[2]) == accum_interval
    assert drain_args[0]["is_input"] and "lx" in drain_args[0]["allocation"]
    assert _lx_interval(drain_args[0]) == accum_interval
    assert not drain_args[1]["is_input"] and drain_args[1]["allocation"].keys() == {
        "hbm"
    }
    assert not _overlaps(x_interval, accum_interval), (
        f"persistent X {x_interval} overlaps accumulator {accum_interval}"
    )

    # No other tensor may share any byte of the fixed accumulator.  The fill
    # write, loop-carried combine read/write, and final drain read are the whole
    # intended lifetime.
    accumulator_users = {
        (id(fills[0]), 1),
        (id(combines[0]), 0),
        (id(combines[0]), 2),
        (id(drains[0]), 0),
    }
    observed_accumulator_users = set()
    for op in [*op_specs, *loop_ops]:
        for index, tensor in enumerate(_tensor_args(op)):
            interval = _lx_interval(tensor)
            if interval is None or not _overlaps(interval, accum_interval):
                continue
            key = (id(op), index)
            assert key in accumulator_users and interval == accum_interval, (
                f"allocation {interval} aliases accumulator {accum_interval}; "
                f"op={_debug_chain(op)} tensor_index={index}"
            )
            observed_accumulator_users.add(key)
    assert observed_accumulator_users == accumulator_users

    # The runtime contract is direct singleton alpha[E,T,1].  Verify the
    # wrapper ABI, then require the complete down -> alpha mul -> unit expert
    # contribution chain to remain in LX with no preprocessing of alpha.  The
    # enclosing expert LoopSpec plus the following accumulator add owns the
    # actual cross-expert sum; a local extent-one reduction is intentionally
    # emitted as an identity.
    asserted_shapes = [
        tuple(_literal(node.args[1]))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_size_stride"
        and len(node.args) >= 2
    ]
    assert asserted_shapes.count((2, 64, 1)) == 1

    down_ops = [
        op
        for op in loop_ops
        if _literal(_keyword(op, "op")) == "batchmatmul"
        and _debug_chain(op)[0] == "bmm"
        and "buf4" in _debug_chain(op)
    ]
    assert len(down_ops) == 1
    down = down_ops[0]
    down_args = _tensor_args(down)
    assert len(down_args) == 3
    down_interval = _lx_interval(down_args[2])
    assert down_interval is not None and not down_args[2]["is_input"]

    alpha_copies = [
        op for op in loop_ops if "coarse_tile_read_copy_0_arg4_1_4" in _debug_chain(op)
    ]
    assert len(alpha_copies) == 1
    alpha_copy = alpha_copies[0]
    alpha_args = _tensor_args(alpha_copy)
    assert len(alpha_args) == 2
    assert alpha_args[0]["is_input"] and alpha_args[0]["arg_index"] == 5
    assert alpha_args[0]["allocation"].keys() == {"hbm"}
    assert alpha_args[0]["has_device_tile_advance"]
    assert not alpha_args[1]["is_input"]
    assert alpha_args[1]["allocation"].keys() == {"lx"}
    alpha_interval = _lx_interval(alpha_args[1])
    assert alpha_interval is not None

    weighted_ops = [
        op
        for op in loop_ops
        if _literal(_keyword(op, "op")) == "mul"
        and _debug_chain(op)[0] == "mul_1"
        and "buf5" in _debug_chain(op)
    ]
    assert len(weighted_ops) == 1
    weighted = weighted_ops[0]
    weighted_args = _tensor_args(weighted)
    assert len(weighted_args) == 3
    assert all(tensor["allocation"].keys() == {"lx"} for tensor in weighted_args)
    assert _lx_interval(weighted_args[0]) == down_interval
    assert _lx_interval(weighted_args[1]) == alpha_interval
    assert _lx_interval(weighted_args[2]) == down_interval

    expert_contributions = [
        op
        for op in loop_ops
        if _literal(_keyword(op, "op")) == "identity"
        and _literal(_keyword(op, "is_reduction")) is False
        and _debug_chain(op)[0] == "sum_1"
        and "buf6" in _debug_chain(op)
    ]
    assert len(expert_contributions) == 1
    contribution = expert_contributions[0]
    assert _debug_aten_op(contribution) == "aten.sum.dim_IntList"
    assert _literal(_keyword(combines[0], "op")) == "add"
    assert _debug_aten_op(combines[0]) == "aten.sum.dim_IntList"
    assert (
        loop_ops.index(weighted)
        < loop_ops.index(contribution)
        < loop_ops.index(combines[0])
    ), "weighted-down -> identity contribution -> accumulator add order changed"
    contribution_args = _tensor_args(contribution)
    assert len(contribution_args) == 2
    assert all(tensor["allocation"].keys() == {"lx"} for tensor in contribution_args)
    assert _lx_interval(contribution_args[0]) == down_interval
    # This is the end-to-end post-retile load-index check.  The contribution
    # identity must read the exact elementwise coordinates just written by the
    # weighted-down mul; an early Reduction->Pointwise rewrite with a stale
    # pre-tile stride would differ here even though both allocations remain LX.
    assert contribution_args[0]["device_size"] == weighted_args[2]["device_size"]
    assert contribution_args[0]["allocation"] == weighted_args[2]["allocation"]
    assert (
        contribution_args[0]["device_coordinates"]
        == weighted_args[2]["device_coordinates"]
    ), (
        "expert contribution reads stale coordinates after retile: "
        f"weighted_down={weighted_args[2]['device_coordinates']} "
        f"contribution={contribution_args[0]['device_coordinates']}"
    )
    contribution_interval = _lx_interval(contribution_args[1])
    assert contribution_interval is not None
    assert _lx_interval(combine_args[1]) == contribution_interval

    # Every loop-internal compute tensor must be LX.  HBM is permitted only as
    # the source of one of the four per-expert copies (gate/up/down/alpha).
    expert_hbm_arg_indices = []
    for op in loop_ops:
        tensors = _tensor_args(op)
        for tensor in tensors:
            if tensor["allocation"].keys() == {"hbm"}:
                assert tensor["is_input"]
                assert tensor["has_device_tile_advance"]
                assert _literal(_keyword(op, "op")) == "identity"
                assert len(tensors) == 2
                assert tensors[1]["allocation"].keys() == {"lx"}
                expert_hbm_arg_indices.append(tensor["arg_index"])
                continue
            assert tensor["allocation"].keys() == {"lx"}, (
                f"loop tensor leaves LX: op={_debug_chain(op)} tensor={tensor}"
            )
    assert expert_hbm_arg_indices == [2, 3, 4, 5], (
        "expected gate/up/down/alpha HBM inputs in loop order; got "
        f"{expert_hbm_arg_indices}"
    )

    final_hbm_outputs = []
    hbm_pool_tensors = []
    for op in [*op_specs, *loop_ops]:
        for tensor in _tensor_args(op):
            if not tensor["is_input"] and tensor["allocation"].keys() == {"hbm"}:
                final_hbm_outputs.append((op, tensor))
            if "hbm_pool" in tensor["allocation"]:
                hbm_pool_tensors.append((op, tensor))
    assert len(final_hbm_outputs) == 1
    assert final_hbm_outputs[0][0] is drains[0]
    assert final_hbm_outputs[0][1]["allocation"] == drain_args[1]["allocation"]
    assert not hbm_pool_tensors, f"hbm_pool is forbidden: {hbm_pool_tensors}"
    restickify_ops = [
        op
        for op in [*op_specs, *loop_ops]
        if "restickify" in _literal(_keyword(op, "op")).lower()
        or any("restickify" in name.lower() for name in _debug_chain(op))
    ]
    assert not restickify_ops, f"restickify is forbidden: {restickify_ops}"

    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith("sdsc_")
    ]
    assert len(run_calls) == 1, (
        f"expected one wrapper bundle call, got {len(run_calls)}"
    )

    return {
        "one_bundle_call": True,
        "one_source_bundle": True,
        "one_flat_expert_loop": True,
        "expert_hbm_arg_indices": expert_hbm_arg_indices,
        "x_hbm_sources": 1,
        "x_loop_reads": 2,
        "x_has_tile_advance": False,
        "x_lx_interval_bytes": list(x_interval),
        "x_alias_free_through_loop": True,
        "accumulator_lx_interval_bytes": list(accum_interval),
        "accumulator_alias_free": True,
        "x_accumulator_disjoint": True,
        "runtime_alpha_shape": [2, 64, 1],
        "runtime_alpha_direct_hbm_to_lx": True,
        "runtime_alpha_lx_interval_bytes": list(alpha_interval),
        "down_output_lx_interval_bytes": list(down_interval),
        "post_down_weighting_lx": True,
        "expert_sum_lx": True,
        "local_sum_op_specs": 0,
        "contribution_coordinates_match_weighted_down": True,
        "final_hbm_writes": 1,
        "hbm_pool_allocations": 0,
        "restickify_ops": 0,
    }


def _assert_no_sum_sdscs(cache_dir: pathlib.Path) -> list[str]:
    """Require the local unit contribution to lower as identity, never sum."""

    sdsc_paths = sorted(cache_dir.rglob("sdsc_*.json"))
    assert sdsc_paths, f"expected emitted SDSC JSON files under {cache_dir}"
    keys = []
    sum_keys = []
    for path in sdsc_paths:
        payload = json.loads(path.read_text())
        assert len(payload) == 1, f"expected one top-level SDSC op in {path}"
        key = next(iter(payload))
        keys.append(key)
        op_name = key.split("_", 1)[1].lower() if "_" in key else key.lower()
        if re.search(r"(^|_)sum(nonstick)?($|_)", op_name):
            sum_keys.append(key)
    assert not sum_keys, f"unit local sum SDSCs remain: {sum_keys}"
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--backend-compile",
        action="store_true",
        help="run the real backend compiler while still mocking prepare/launch",
    )
    parser.add_argument(
        "--device-correctness",
        action="store_true",
        help="run the generated kernel twice with distinct runtime alpha",
    )
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    cache_dir = pathlib.Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _check_cpu_weighting_semantics()
    torch_spyre._autoload()
    pnd.reset()
    for name, size in {"E": 2, "T": 64, "H": 64, "F": 64, "K": 1}.items():
        pnd.declare_tensor_dim(name, size)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    cpu_inputs = [
        # Keep the three-matmul FP16 chain well above subnormal range.  The
        # earlier compile-only probe used 0.01 for every operand; that is fine
        # for structure but drives the final C1 output to zero on device.
        torch.randn((64, 64), dtype=torch.float16, generator=generator) * 0.25,
        torch.randn((2, 64, 64), dtype=torch.float16, generator=generator) * 0.1,
        torch.randn((2, 64, 64), dtype=torch.float16, generator=generator) * 0.1,
        torch.randn((2, 64, 64), dtype=torch.float16, generator=generator) * 0.1,
        # Non-binary runtime weights: both experts must contribute after down.
        torch.rand((2, 64, 1), dtype=torch.float16, generator=generator) * 0.8 + 0.1,
    ]
    alpha_b = (
        torch.rand((2, 64, 1), dtype=torch.float16, generator=generator) * 0.7 + 0.2
    )
    device_inputs = [tensor.to("spyre") for tensor in cpu_inputs]
    alpha_b_device = alpha_b.to("spyre")
    for tensor, dims in zip(
        device_inputs,
        (
            ["T", "H"],
            ["E", "H", "F"],
            ["E", "H", "F"],
            ["E", "F", "H"],
            ["E", "T", "K"],
        ),
    ):
        pnd.name_tensor_dims(tensor, dims)
    pnd.name_tensor_dims(alpha_b_device, ["E", "T", "K"])

    compiled = torch.compile(
        dense_activation_stationary_ffn,
        dynamic=False,
        fullgraph=True,
    )
    captures: list[pathlib.Path] = []
    real_generate_bundle = async_compile.generate_bundle

    def capture_generate_bundle(kernel_name, output_dir, specs):
        result = real_generate_bundle(kernel_name, output_dir, specs)
        captures.append(pathlib.Path(output_dir))
        return result

    async_compile.generate_bundle = capture_generate_bundle
    first_bundle_hash = ""
    second_bundle_hash = ""
    new_capture_paths: list[str] = []
    try:
        with (
            config.patch(
                {
                    "sencores": 1,
                    "lx_planning": True,
                    "allow_all_ops_in_lx_planning": True,
                }
            ),
            (
                nullcontext()
                if args.device_correctness
                else patch("torch_spyre.execution.kernel_runner.launch_jobplan")
            ),
            (
                nullcontext()
                if args.device_correctness
                else patch("torch_spyre.execution.kernel_runner.prepare_kernel")
            ),
            (
                nullcontext()
                if args.backend_compile or args.device_correctness
                else patch("torch_spyre.execution.async_compile.subprocess.run")
            ),
        ):
            actual_a, source_codes = run_and_get_code(compiled, *device_inputs)
            assert len(captures) == 1, (
                f"first alpha call emitted {len(captures)} bundles"
            )
            first_bundle = captures[0] / "bundle.mlir"
            assert first_bundle.is_file(), f"missing first-call bundle: {first_bundle}"
            first_bundle_hash = _sha256_file(first_bundle)
            capture_count_before_b = len(captures)
            actual_b = (
                compiled(*device_inputs[:-1], alpha_b_device)
                if args.device_correctness
                else None
            )
            if args.device_correctness:
                new_capture_paths = [
                    str(path) for path in captures[capture_count_before_b:]
                ]
                second_bundle_hash = _sha256_file(first_bundle)
    finally:
        async_compile.generate_bundle = real_generate_bundle

    if len(source_codes) != 1:
        raise RuntimeError(
            f"expected exactly one generated wrapper source, got {len(source_codes)}"
        )
    source_path = output_dir / "generated_module.py"
    source_path.write_text(source_codes[0])
    structural = _analyze_generated_structure(source_codes[0])
    sdsc_keys = _assert_no_sum_sdscs(cache_dir)
    structural["local_sum_sdscs"] = 0
    structural["sdsc_op_keys"] = sdsc_keys

    bundles = sorted(cache_dir.rglob("bundle.mlir"))
    if len(bundles) != 1 or len(captures) != 1:
        raise RuntimeError(
            f"expected one cache/captured bundle, got cache={len(bundles)} "
            f"captures={len(captures)}"
        )
    assert _sha256_file(bundles[0]) == first_bundle_hash
    copied_bundles = []
    for index, source_bundle in enumerate(bundles):
        destination = output_dir / f"bundle_{index}_{source_bundle.parent.name}.mlir"
        shutil.copy2(source_bundle, destination)
        copied_bundles.append(str(destination))

    payload = {
        "shape": {"E": 2, "T": 64, "H": 64, "F": 64, "cores": 1},
        "source": str(source_path),
        "bundles": copied_bundles,
        "launched_generated_kernel": bool(args.device_correctness),
        "structural": structural,
    }
    if args.device_correctness:
        assert first_bundle_hash == second_bundle_hash, (
            "bundle.mlir changed between runtime-alpha calls"
        )
        assert not new_capture_paths, (
            f"second alpha call emitted new bundle(s): {new_capture_paths}"
        )
        x, gate_w, up_w, down_w, alpha_a = cpu_inputs

        def reference(alpha: torch.Tensor) -> torch.Tensor:
            x32 = x.float()
            gate = torch.einsum("tk,ekn->etn", x32, gate_w.float())
            up = torch.einsum("tk,ekn->etn", x32, up_w.float())
            hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
            down = torch.einsum("etf,efh->eth", hidden, down_w.float())
            return (down * alpha.float()).sum(dim=0)

        actuals = [actual_a.cpu().float(), actual_b.cpu().float()]
        refs = [reference(alpha_a), reference(alpha_b)]

        def metrics(actual: torch.Tensor, ref: torch.Tensor) -> dict:
            diff = actual - ref
            return {
                "max_abs": float(diff.abs().max()),
                "rel_l2": float(
                    torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)
                ),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        actual.flatten(), ref.flatten(), dim=0
                    )
                ),
            }

        numeric = [metrics(actual, ref) for actual, ref in zip(actuals, refs)]
        delta = metrics(actuals[1] - actuals[0], refs[1] - refs[0])
        torch.save(
            {
                "schema_version": 1,
                "shape": {
                    "E": 2,
                    "T": 64,
                    "H": 64,
                    "F": 64,
                    "alpha_singleton": 1,
                },
                "inputs": {
                    "x": x,
                    "gate_w": gate_w,
                    "up_w": up_w,
                    "down_w": down_w,
                },
                "alphas": {"a": alpha_a, "b": alpha_b},
                "reference_route_alpha": {
                    "a": alpha_a.squeeze(-1),
                    "b": alpha_b.squeeze(-1),
                },
                "fp32_references": {"a": refs[0], "b": refs[1]},
                "device_outputs": {"a": actuals[0], "b": actuals[1]},
                "bundle_identity": {
                    "generated_source_sha256": _sha256_file(source_path),
                    "first_call_bundle_sha256": first_bundle_hash,
                    "second_call_bundle_sha256": second_bundle_hash,
                    "new_bundles_after_second_call": new_capture_paths,
                    "same_compiled_callable": True,
                },
                "timing_collected": False,
            },
            output_dir / "correctness_artifact.pt",
        )
        payload["correctness"] = {
            "payloads": {"alpha_a": numeric[0], "alpha_b": numeric[1]},
            "alpha_response_delta": delta,
            "same_callable_two_alphas": True,
        }
        payload["status"] = "pending_numeric_acceptance"
        (output_dir / "compile_result.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        for actual, ref in zip(actuals, refs):
            torch.testing.assert_close(actual, ref, rtol=0.03, atol=0.05)
        torch.testing.assert_close(
            actuals[1] - actuals[0], refs[1] - refs[0], rtol=0.03, atol=0.05
        )
        assert max(item["rel_l2"] for item in numeric) <= 0.03
        assert delta["rel_l2"] <= 0.03
        assert min(*(item["cosine"] for item in numeric), delta["cosine"]) >= 0.999
        payload["status"] = "passed"
    (output_dir / "compile_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
