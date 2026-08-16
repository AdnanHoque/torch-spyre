#!/usr/bin/env python3
"""Compile-only C32 work-division probe for the D-AS-X shared-LHS path.

This probe launches no generated kernel.  It asks one narrow question before
we add any LX fanout support: does the ordinary C32 work-division pass give the
one-time X preheader and both gate/up matmuls the same M16 x K2 ownership at
the real Gemma expert shape?  If it does, X can be reused directly and the
previously proposed M8xK4 -> replicated-M8 shuffle is unnecessary.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import nullcontext
import hashlib
import json
import math
import pathlib
import time
from unittest.mock import patch

import sympy
import torch
from torch._inductor.utils import run_and_get_code

import torch_spyre
from torch_spyre._inductor import config, spyre_hint
import torch_spyre._inductor.wsr.propagate_named_dims as pnd


E = 2
T = 512
H = 2816
F = 704
CORES = 32


def dense_activation_stationary_ffn(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """The accepted flat-E graph, now compiled with the real Gemma dimensions."""

    with spyre_hint(num_tiles_per_dim={"E": E}):
        gate = torch.ops.spyre.activation_stationary_shared_lhs_mm_prepacked(
            x, gate_w
        )
        up = torch.ops.spyre.activation_stationary_shared_lhs_mm_prepacked(x, up_w)
        hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
        down = torch.ops.spyre.activation_stationary_expert_mm_prepacked(
            hidden, down_w
        )
        return (down * alpha).sum(dim=0)


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _keyword(call: ast.Call, name: str) -> ast.AST:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    raise AssertionError(f"{_call_name(call)!r} has no keyword {name!r}")


def _optional_keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _literal(node: ast.AST):
    return ast.literal_eval(node)


def _sympify_call(node: ast.AST) -> sympy.Expr:
    assert isinstance(node, ast.Call) and _call_name(node) == "sympify"
    assert len(node.args) == 1
    return sympy.sympify(_literal(node.args[0]))


def _debug_chain(op: ast.Call) -> tuple[str, ...]:
    debug = _keyword(op, "debug_handle")
    assert isinstance(debug, ast.Call) and _call_name(debug) == "DebugHandle"
    return tuple(_literal(_keyword(debug, "ir_chain")))


def _op_name(op: ast.Call) -> str:
    return str(_literal(_keyword(op, "op")))


def _op_info(op: ast.Call) -> dict:
    info = _literal(_keyword(op, "op_info"))
    assert isinstance(info, dict), info
    return info


def _iteration_splits(op: ast.Call) -> dict[sympy.Symbol, int]:
    node = _keyword(op, "iteration_space")
    assert isinstance(node, ast.Dict)
    result: dict[sympy.Symbol, int] = {}
    for key, value in zip(node.keys, node.values):
        assert key is not None
        symbol = _sympify_call(key)
        assert isinstance(symbol, sympy.Symbol)
        assert isinstance(value, ast.Tuple) and len(value.elts) == 2
        result[symbol] = int(_literal(value.elts[1]))
    return result


def _tensor_args(op: ast.Call) -> list[dict]:
    args = _keyword(op, "args")
    assert isinstance(args, ast.List)
    result = []
    for tensor in args.elts:
        assert isinstance(tensor, ast.Call) and _call_name(tensor) == "TensorArg"
        coordinates = _keyword(tensor, "device_coordinates")
        assert isinstance(coordinates, ast.List)
        advance_node = _optional_keyword(tensor, "device_tile_advance_expr")
        advance = _sympify_call(advance_node) if advance_node is not None else None
        result.append(
            {
                "is_input": bool(_literal(_keyword(tensor, "is_input"))),
                "arg_index": int(_literal(_keyword(tensor, "arg_index"))),
                "device_size": list(_literal(_keyword(tensor, "device_size"))),
                "device_coordinates": [
                    _sympify_call(coordinate) for coordinate in coordinates.elts
                ],
                "allocation": dict(_literal(_keyword(tensor, "allocation"))),
                "has_tile_advance": advance is not None and advance != 0,
                "tile_advance": advance,
            }
        )
    return result


def _lx_interval(op: ast.Call, tensor: dict) -> tuple[int, int] | None:
    allocation = tensor["allocation"]
    if "lx" not in allocation:
        return None
    # LX addresses are per core.  ``device_size`` is the logical/global device
    # shape, so divide its 128-byte-stick payload by only the OpSpec splits that
    # actually occur in this TensorArg's coordinates.  A BMM K split, for
    # example, partitions X but not its output accumulator.
    splits = _iteration_splits(op)
    coordinate_symbols = set().union(
        *(coordinate.free_symbols for coordinate in tensor["device_coordinates"])
    )
    partition = math.prod(
        split for symbol, split in splits.items() if symbol in coordinate_symbols
    )
    global_bytes = math.prod(tensor["device_size"][:-1]) * 128
    assert partition >= 1 and global_bytes % partition == 0
    size_bytes = global_bytes // partition
    start = int(allocation["lx"])
    return start, start + size_bytes


def _matrix_ownership(op: ast.Call, tensor: dict) -> dict[str, int]:
    """Return row and stick-axis splits for a matrix TensorArg.

    For the X layout the physical coordinates are
    ``[floor(K/64), M, Mod(K,64)]``.  Recovering the two symbols from those
    coordinates makes this check independent of generated ``c0``/``d2`` names.
    """

    coordinates = tensor["device_coordinates"]
    assert len(coordinates) in (3, 4), coordinates
    if len(coordinates) == 4:
        leading = coordinates[0]
        assert len(leading.free_symbols) == 1, coordinates
        assert _iteration_splits(op).get(next(iter(leading.free_symbols)), 1) == 1
        coordinates = coordinates[1:]
    row = coordinates[1]
    assert isinstance(row, sympy.Symbol), coordinates
    stick_symbols = coordinates[-1].free_symbols
    assert len(stick_symbols) == 1, coordinates
    stick = next(iter(stick_symbols))
    assert coordinates[0].free_symbols == {stick}, coordinates
    splits = _iteration_splits(op)
    return {"M": splits.get(row, 1), "K": splits.get(stick, 1)}


def _analyze(source: str) -> dict:
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
    assert len(sdsc_calls) == 1, f"expected one source bundle, got {len(sdsc_calls)}"
    specs = sdsc_calls[0].args[1]
    assert isinstance(specs, ast.List)
    top_specs = specs.elts

    loops = [
        node
        for node in top_specs
        if isinstance(node, ast.Call) and _call_name(node) == "LoopSpec"
    ]
    assert len(loops) == 1, f"expected one flat E loop, got {len(loops)}"
    loop = loops[0]
    body = _keyword(loop, "body")
    assert isinstance(body, ast.List)
    loop_ops = [
        node
        for node in body.elts
        if isinstance(node, ast.Call) and _call_name(node) == "OpSpec"
    ]
    # C32 may need one explicit LX-only ownership clone between the down BMM
    # and router-weight multiply.  It is a valid on-chip bridge, not an HBM
    # spill; every later allocation/edge check still applies fail-closed.
    assert len(loop_ops) in (9, 10), (
        f"expected 9 direct-weight expert-loop OpSpecs plus at most one LX clone, "
        f"got {len(loop_ops)}"
    )
    top_ops = [
        node
        for node in top_specs
        if isinstance(node, ast.Call) and _call_name(node) == "OpSpec"
    ]

    x_copies = [
        op
        for op in top_ops
        if any(
            name.startswith("coarse_tile_read_copy_0_arg0") for name in _debug_chain(op)
        )
    ]
    assert len(x_copies) == 1, f"expected one X preheader, got {len(x_copies)}"
    x_copy = x_copies[0]
    assert top_specs.index(x_copy) < top_specs.index(loop)
    x_copy_args = _tensor_args(x_copy)
    assert len(x_copy_args) == 2
    assert x_copy_args[0]["arg_index"] == 0
    assert x_copy_args[0]["allocation"].keys() == {"hbm"}
    assert x_copy_args[1]["allocation"].keys() == {"lx"}
    assert not x_copy_args[0]["has_tile_advance"]
    assert not x_copy_args[1]["has_tile_advance"]
    assert x_copy_args[1]["device_size"] == [44, T, 64]
    x_interval = _lx_interval(x_copy, x_copy_args[1])
    assert x_interval is not None
    x_global_bytes = math.prod(x_copy_args[1]["device_size"][:-1]) * 128
    assert x_global_bytes == T * H * 2
    assert x_interval[1] - x_interval[0] == x_global_bytes // CORES
    x_copy_ownership = _matrix_ownership(x_copy, x_copy_args[1])
    x_symbols = tuple(_iteration_splits(x_copy))
    x_stick_symbols = x_copy_args[1]["device_coordinates"][-1].free_symbols
    assert len(x_stick_symbols) == 1
    x_stick_symbol = next(iter(x_stick_symbols))
    x_contiguous_dim = _op_info(x_copy).get("core_mapping_contiguous_dim")

    shared_lhs = [
        op
        for op in loop_ops
        if _literal(_keyword(op, "op")) == "batchmatmul"
        and _debug_chain(op)[0].startswith("activation_stationary_shared_lhs_mm")
    ]
    assert len(shared_lhs) == 2, f"expected gate/up BMMs, got {len(shared_lhs)}"
    bmm_ownership = []
    for op in shared_lhs:
        x_arg = _tensor_args(op)[0]
        assert x_arg["is_input"] and x_arg["arg_index"] == -1
        assert _lx_interval(op, x_arg) == x_interval
        bmm_ownership.append(_matrix_ownership(op, x_arg))

    expected = {"M": 32, "K": 1}
    assert x_copy_ownership == expected, x_copy_ownership
    assert bmm_ownership == [expected, expected], bmm_ownership
    assert math.prod(expected.values()) == CORES
    assert x_contiguous_dim is None
    assert _iteration_splits(x_copy).get(x_stick_symbol, 1) == 1

    # No second identity may read the resident X and produce another LX copy.
    fanout_copies = []
    for op in [*top_ops, *loop_ops]:
        if op is x_copy or _literal(_keyword(op, "op")) != "identity":
            continue
        tensors = _tensor_args(op)
        if any(
            _lx_interval(op, tensor) == x_interval
            for tensor in tensors
            if tensor["is_input"]
        ):
            fanout_copies.append(_debug_chain(op))
    assert not fanout_copies, (
        f"unexpected X fanout/relayout identities: {fanout_copies}"
    )

    x_hbm_reads = [
        (op, tensor)
        for op in [*top_ops, *loop_ops]
        for tensor in _tensor_args(op)
        if tensor["is_input"]
        and tensor["arg_index"] == 0
        and tensor["allocation"].keys() == {"hbm"}
    ]
    assert len(x_hbm_reads) == 1 and x_hbm_reads[0][0] is x_copy

    all_ops = [*top_ops, *loop_ops]

    def inputs(op: ast.Call) -> list[dict]:
        return [tensor for tensor in _tensor_args(op) if tensor["is_input"]]

    def outputs(op: ast.Call) -> list[dict]:
        return [tensor for tensor in _tensor_args(op) if not tensor["is_input"]]

    def output_interval(op: ast.Call) -> tuple[int, int]:
        tensors = outputs(op)
        assert len(tensors) == 1
        interval = _lx_interval(op, tensors[0])
        assert interval is not None, (_debug_chain(op), tensors[0]["allocation"])
        return interval

    def hbm_to_lx_copy(op: ast.Call) -> bool:
        op_inputs, op_outputs = inputs(op), outputs(op)
        return (
            _op_name(op) == "identity"
            and len(op_inputs) == len(op_outputs) == 1
            and op_inputs[0]["allocation"].keys() == {"hbm"}
            and op_outputs[0]["allocation"].keys() == {"lx"}
        )

    assert int(_sympify_call(_keyword(loop, "count"))) == E
    for op in all_ops:
        assert "restickify" not in _op_name(op).lower()
        assert not any("restickify" in name.lower() for name in _debug_chain(op))
        assert all(
            "hbm_pool" not in tensor["allocation"] for tensor in _tensor_args(op)
        )

    fills = [
        op
        for op in top_ops
        if any("coarse_tile_fill" in name for name in _debug_chain(op))
    ]
    drains = [
        op
        for op in top_ops
        if any("coarse_tile_reduce_copy" in name for name in _debug_chain(op))
    ]
    assert len(fills) == len(drains) == 1
    fill, drain = fills[0], drains[0]
    assert len(top_ops) == 3
    assert (
        top_specs.index(x_copy)
        < top_specs.index(fill)
        < top_specs.index(loop)
        < top_specs.index(drain)
    )
    assert hbm_to_lx_copy(fill)
    accumulator = output_interval(fill)
    assert not (accumulator[0] < x_interval[1] and x_interval[0] < accumulator[1])
    drain_inputs, drain_outputs = inputs(drain), outputs(drain)
    assert len(drain_inputs) == len(drain_outputs) == 1
    assert _lx_interval(drain, drain_inputs[0]) == accumulator
    assert drain_outputs[0]["allocation"].keys() == {"hbm"}

    hbm_writes = [
        (op, tensor)
        for op in all_ops
        for tensor in outputs(op)
        if tensor["allocation"].keys() == {"hbm"}
    ]
    assert len(hbm_writes) == 1 and hbm_writes[0][0] is drain

    loop_copies = [op for op in loop_ops if hbm_to_lx_copy(op)]
    assert len(loop_copies) == 1
    alpha_copy = loop_copies[0]
    assert inputs(alpha_copy)[0]["has_tile_advance"]

    bmms = [op for op in loop_ops if _op_name(op) == "batchmatmul"]
    assert len(bmms) == 3 and all(op in bmms for op in shared_lhs)
    for op in loop_ops:
        if op in loop_copies:
            continue
        allocations = [set(tensor["allocation"]) for tensor in _tensor_args(op)]
        if op in bmms:
            assert allocations.count({"hbm"}) == 1
            assert all(keys in ({"hbm"}, {"lx"}) for keys in allocations)
        else:
            assert all(keys == {"lx"} for keys in allocations), (
                _debug_chain(op),
                _tensor_args(op),
            )

    down_ops = [op for op in bmms if op not in shared_lhs]
    assert len(down_ops) == 1
    down = down_ops[0]
    down_output = output_interval(down)
    direct_weight_args = []
    for op in bmms:
        streamed = [
            tensor
            for tensor in inputs(op)
            if tensor["allocation"].keys() == {"hbm"}
            and tensor["arg_index"] in {2, 3, 4}
        ]
        assert len(streamed) == 1 and streamed[0]["has_tile_advance"]
        direct_weight_args.extend(streamed)
    assert len(direct_weight_args) == 3

    muls = [op for op in loop_ops if _op_name(op) == "mul"]
    assert len(muls) == 2
    route_muls = [op for op in muls if loop_ops.index(op) > loop_ops.index(down)]
    assert len(route_muls) == 1
    route_mul = route_muls[0]
    route_inputs = [_lx_interval(route_mul, tensor) for tensor in inputs(route_mul)]
    assert down_output in route_inputs
    alpha_interval = output_interval(alpha_copy)
    assert alpha_interval in route_inputs
    route_output = output_interval(route_mul)

    combine_ops = [
        op
        for op in loop_ops
        if any("coarse_tile_combine" in name for name in _debug_chain(op))
    ]
    assert len(combine_ops) == 1 and _op_name(combine_ops[0]) == "add"
    combine = combine_ops[0]
    combine_inputs = [_lx_interval(combine, tensor) for tensor in inputs(combine)]
    assert accumulator in combine_inputs and output_interval(combine) == accumulator
    contribution_interval = (
        combine_inputs[1] if combine_inputs[0] == accumulator else combine_inputs[0]
    )
    contribution_ops = [
        op
        for op in loop_ops[loop_ops.index(route_mul) + 1 : loop_ops.index(combine)]
        if _op_name(op) == "identity" and _debug_chain(op)[0].startswith("sum")
    ]
    assert len(contribution_ops) == 1
    assert (
        _lx_interval(contribution_ops[0], inputs(contribution_ops[0])[0])
        == route_output
    )
    assert output_interval(contribution_ops[0]) == contribution_interval
    assert not any(_op_name(op) == "sum" for op in loop_ops)

    # Persistent X and the fixed accumulator must remain disjoint from all
    # loop outputs. The only X aliases are the read-only first operands of the
    # two shared-LHS BMMs; the only accumulator output is the in-place combine.
    for op in loop_ops:
        for index, tensor in enumerate(_tensor_args(op)):
            interval = _lx_interval(op, tensor)
            if (
                interval is not None
                and interval[0] < x_interval[1]
                and x_interval[0] < interval[1]
            ):
                assert op in shared_lhs and tensor["is_input"] and index == 0
                assert interval == x_interval
        for tensor in outputs(op):
            interval = _lx_interval(op, tensor)
            if interval is None or op is combine and interval == accumulator:
                continue
            assert not (interval[0] < accumulator[1] and accumulator[0] < interval[1])

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
    asserted_shapes = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_size_stride"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
        ):
            asserted_shapes[node.args[0].id] = tuple(_literal(node.args[1]))
    alpha_arg_index = inputs(alpha_copy)[0]["arg_index"]
    assert 0 <= alpha_arg_index < len(run_calls[0].args)
    alpha_runtime_arg = run_calls[0].args[alpha_arg_index]
    assert isinstance(alpha_runtime_arg, ast.Name)
    assert asserted_shapes[alpha_runtime_arg.id] == (E, T, 1)

    return {
        "shape": {"E": E, "T": T, "H": H, "F": F, "cores": CORES},
        "one_flat_expert_loop": True,
        "one_x_hbm_to_lx_preheader": True,
        "x_logical_global_bytes": x_global_bytes,
        "x_per_core_lx_interval_bytes": list(x_interval),
        "x_copy_ownership": x_copy_ownership,
        "x_core_mapping_contiguous_dim": None,
        "gate_up_x_ownership": bmm_ownership,
        "gate_up_share_exact_x_interval": True,
        "x_fanout_identities": 0,
        "internal_compute_allocations": "LX-only",
        "fixed_accumulator_lx_interval_bytes": list(accumulator),
        "down_output_lx_interval_bytes": list(down_output),
        "runtime_alpha_shape": [E, T, 1],
        "router_weighting_after_down": True,
        "expert_hbm_operands_advance": 4,
        "hbm_pool_allocations": 0,
        "restickify_ops": 0,
        "final_hbm_outputs": 1,
        "one_source_bundle": True,
        "one_wrapper_bundle_call": True,
    }


def _analyze_backend_bundle(bundle_dir: pathlib.Path) -> dict:
    """Validate the physical C32 map in emitted SuperDSC JSON, if compiled."""

    bundle_mlir = bundle_dir / "bundle.mlir"
    assert bundle_mlir.is_file(), bundle_mlir
    roots = []
    for path in sorted(bundle_dir.glob("sdsc_*.json")):
        payload = json.loads(path.read_text())
        assert len(payload) == 1, path
        root = next(iter(payload.values()))
        debug = root.get("debug_handle_") or {}
        chain = tuple(debug.get("ir_chain") or ())
        roots.append((path, root, chain))

    x_roots = [
        item
        for item in roots
        if any(name.startswith("coarse_tile_read_copy_0_arg0") for name in item[2])
    ]
    assert len(x_roots) == 1
    x_path, x_root, _ = x_roots[0]
    expected_x_splits = {"mb": 32, "out": 1}
    expected_x_map = {
        str(core): {"mb": core, "out": 0} for core in range(CORES)
    }
    assert x_root["numWkSlicesPerDim_"] == expected_x_splits
    assert x_root["coreIdToWkSlice_"] == expected_x_map

    shared_bmm_roots = [
        item
        for item in roots
        if item[2]
        and item[2][0].startswith("activation_stationary_shared_lhs_mm")
        and not any("coarse_tile_read_copy" in name for name in item[2])
        and set(item[1].get("numWkSlicesPerDim_", {}))
        == {"x", "mb", "out", "in"}
    ]
    assert len(shared_bmm_roots) == 2
    # The private shared-LHS DDL names the externally selected parallel
    # non-reduction dimension ``x``.  In this one-expert-per-LoopSpec body it
    # carries the row-32 partition; the expert bank advances in the enclosing
    # affine HBM address, not in this local SDSC dimension.
    expected_bmm_splits = {"x": 32, "mb": 1, "out": 1, "in": 1}
    expected_bmm_map = {
        str(core): {"x": core, "mb": 0, "out": 0, "in": 0}
        for core in range(CORES)
    }
    for path, root, _ in shared_bmm_roots:
        assert root["numWkSlicesPerDim_"] == expected_bmm_splits, path
        assert root["coreIdToWkSlice_"] == expected_bmm_map, path

    return {
        "bundle_dir": str(bundle_dir),
        "bundle_mlir_sha256": hashlib.sha256(bundle_mlir.read_bytes()).hexdigest(),
        "sdsc_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path, _, _ in roots
        },
        "x_sdsc": x_path.name,
        "x_num_work_slices": expected_x_splits,
        "x_core_id_to_work_slice_all_32": True,
        "gate_up_sdscs": [path.name for path, _, _ in shared_bmm_roots],
        "gate_up_core_id_to_work_slice_matches_x_all_32": True,
    }


def main() -> None:
    global E
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--backend-compile",
        action="store_true",
        help="run backend compilation while still suppressing prepare/launch",
    )
    parser.add_argument(
        "--device-correctness",
        action="store_true",
        help="execute the compiled C32 program twice; never collects timing",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="run the registered 5/50/10x5/3 protocol after correctness",
    )
    parser.add_argument(
        "--bundle-dir",
        help="backend bundle directory; otherwise uniquely discovered under cache-dir",
    )
    parser.add_argument("--experts", type=int, default=E)
    args = parser.parse_args()
    if args.experts < 2:
        raise ValueError("--experts must be at least 2 for the temporal loop gate")
    E = args.experts

    output_dir = pathlib.Path(args.output_dir)
    cache_dir = pathlib.Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch_spyre._autoload()
    pnd.reset()
    for name, size in {"E": E, "T": T, "H": H, "F": F, "K": 1}.items():
        pnd.declare_tensor_dim(name, size)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    cpu_inputs = [
        torch.randn((T, H), dtype=torch.float16, generator=generator) * 0.1,
        torch.randn((H, E, F), dtype=torch.float16, generator=generator) * 0.1,
        torch.randn((H, E, F), dtype=torch.float16, generator=generator) * 0.1,
        torch.randn((F, E, H), dtype=torch.float16, generator=generator) * 0.1,
    ]
    if E >= 8:
        # Fair full-bank workload: exactly top-8 nonzero router weights per
        # token, balanced over all experts.  The second payload permutes expert
        # IDs while preserving the same nonbinary weights and sparsity.
        tokens = torch.arange(T)
        lanes = torch.arange(8)
        expert_ids = (tokens[:, None] * 8 + lanes[None, :]) % E
        weights = torch.linspace(0.05, 0.20, 8, dtype=torch.float32)
        weights = (weights / weights.sum()).to(torch.float16)
        alpha_a = torch.zeros((E, T, 1), dtype=torch.float16)
        alpha_a[expert_ids, tokens[:, None], 0] = weights[None, :]
        permutation = torch.randperm(E, generator=generator)
        alpha_b = torch.zeros_like(alpha_a)
        alpha_b[permutation[expert_ids], tokens[:, None], 0] = weights[None, :]
        alpha_hot = torch.zeros_like(alpha_a)
        alpha_hot[lanes[:, None], tokens[None, :], 0] = weights[:, None]
    else:
        alpha_a = (
            torch.rand((E, T, 1), dtype=torch.float16, generator=generator) * 0.8
            + 0.1
        )
        alpha_b = (
            torch.rand((E, T, 1), dtype=torch.float16, generator=generator) * 0.7
            + 0.2
        )
        alpha_hot = alpha_b.clone()
    cpu_inputs.append(alpha_a)
    device_inputs = [tensor.to("spyre") for tensor in cpu_inputs]
    alpha_b_device = alpha_b.to("spyre")
    alpha_hot_device = alpha_hot.to("spyre")
    for tensor, dims in zip(
        device_inputs,
        (
            ["T", "H"],
            ["H", "E", "F"],
            ["H", "E", "F"],
            ["F", "E", "H"],
            ["E", "T", "K"],
        ),
    ):
        pnd.name_tensor_dims(tensor, dims)
    pnd.name_tensor_dims(alpha_b_device, ["E", "T", "K"])
    pnd.name_tensor_dims(alpha_hot_device, ["E", "T", "K"])

    compiled = torch.compile(
        dense_activation_stationary_ffn,
        dynamic=False,
        fullgraph=True,
    )
    with (
        config.patch(
            {
                "sencores": CORES,
                "lx_planning": True,
                "allow_all_ops_in_lx_planning": True,
            }
        ),
        (
            nullcontext()
            if args.device_correctness or args.timing
            else patch("torch_spyre.execution.kernel_runner.launch_jobplan")
        ),
        (
            nullcontext()
            if args.device_correctness or args.timing
            else patch("torch_spyre.execution.kernel_runner.prepare_kernel")
        ),
        (
            nullcontext()
            if args.backend_compile or args.device_correctness or args.timing
            else patch("torch_spyre.execution.async_compile.subprocess.run")
        ),
    ):
        actual_a, source_codes = run_and_get_code(compiled, *device_inputs)
        actual_b = (
            compiled(*device_inputs[:-1], alpha_b_device)
            if args.device_correctness or args.timing
            else None
        )
        actual_hot = (
            compiled(*device_inputs[:-1], alpha_hot_device) if args.timing else None
        )

    if len(source_codes) != 1:
        raise RuntimeError(f"expected one generated source, got {len(source_codes)}")
    source = source_codes[0]
    source_path = output_dir / "generated_module.py"
    source_path.write_text(source)
    result = _analyze(source)
    bundle_dir = pathlib.Path(args.bundle_dir) if args.bundle_dir else None
    if (args.backend_compile or args.device_correctness or args.timing) and bundle_dir is None:
        bundle_dirs = sorted({path.parent for path in cache_dir.rglob("bundle.mlir")})
        assert len(bundle_dirs) == 1, (
            f"expected exactly one backend bundle under {cache_dir}, got {bundle_dirs}"
        )
        bundle_dir = bundle_dirs[0]
    if bundle_dir is not None:
        result["backend_core_mapping"] = _analyze_backend_bundle(bundle_dir)
    result.update(
        {
            "source": str(source_path),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "launched_generated_kernel": bool(args.device_correctness or args.timing),
            "backend_compile": bool(
                args.backend_compile or args.device_correctness or args.timing
            ),
        }
    )
    if args.device_correctness or args.timing:
        x, gate_w, up_w, down_w, alpha_a = cpu_inputs

        def reference(alpha: torch.Tensor) -> torch.Tensor:
            x32 = x.float()
            gate = torch.einsum("th,hef->etf", x32, gate_w.float())
            up = torch.einsum("th,hef->etf", x32, up_w.float())
            hidden = torch.nn.functional.gelu(gate, approximate="tanh") * up
            down = torch.einsum("etf,feh->eth", hidden, down_w.float())
            return (down * alpha.float()).sum(dim=0)

        alpha_payloads = [alpha_a, alpha_b]
        actual_payloads = [actual_a, actual_b]
        labels = ["identity", "permutation"]
        if args.timing:
            alpha_payloads.append(alpha_hot)
            actual_payloads.append(actual_hot)
            labels.append("hot8")
        actuals = [actual.cpu().float() for actual in actual_payloads]
        refs = [reference(alpha) for alpha in alpha_payloads]
        torch.save(
            {
                "inputs": cpu_inputs,
                "alpha_b": alpha_b,
                "alpha_hot": alpha_hot,
                "actuals": actuals,
                "references": refs,
            },
            output_dir / "correctness_artifact.pt",
        )

        def metrics(actual: torch.Tensor, ref: torch.Tensor) -> dict:
            diff = actual - ref
            return {
                "max_abs": float(diff.abs().max()),
                "rel_l2": float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(ref)),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(
                        actual.flatten(), ref.flatten(), dim=0
                    )
                ),
            }

        numeric = [metrics(actual, ref) for actual, ref in zip(actuals, refs)]
        delta_metrics = metrics(actuals[1] - actuals[0], refs[1] - refs[0])
        result["correctness"] = {
            "payloads": dict(zip(labels, numeric)),
            "alpha_response_delta": delta_metrics,
            "same_callable_two_alphas": True,
        }
        (output_dir / "compile_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        for actual, ref in zip(actuals, refs):
            torch.testing.assert_close(actual, ref, rtol=0.03, atol=0.05)
        torch.testing.assert_close(
            actuals[1] - actuals[0], refs[1] - refs[0], rtol=0.03, atol=0.05
        )
        assert max(item["rel_l2"] for item in numeric) <= 0.03
        assert delta_metrics["rel_l2"] <= 0.03
        assert min(*(item["cosine"] for item in numeric), delta_metrics["cosine"]) >= 0.999
        if args.timing:
            timing_inputs = {
                "identity": device_inputs[-1],
                "permutation": alpha_b_device,
                "hot8": alpha_hot_device,
            }
            for label in labels:
                for _ in range(5):
                    compiled(*device_inputs[:-1], timing_inputs[label])
                torch.spyre.synchronize()
            orders = [
                ["identity", "permutation", "hot8"],
                ["hot8", "identity", "permutation"],
                ["permutation", "hot8", "identity"],
            ]
            samples = []
            for round_index, order in enumerate(orders):
                for label in order:
                    call_args = (*device_inputs[:-1], timing_inputs[label])
                    for sample_index in range(50):
                        torch.spyre.synchronize()
                        start = time.perf_counter_ns()
                        compiled(*call_args)
                        torch.spyre.synchronize()
                        elapsed = (time.perf_counter_ns() - start) / 1_000_000
                        samples.append(
                            {
                                "selector": label,
                                "round": round_index,
                                "kind": "single",
                                "sample": sample_index,
                                "calls": 1,
                                "per_call_ms": elapsed,
                            }
                        )
                    for block_index in range(10):
                        torch.spyre.synchronize()
                        start = time.perf_counter_ns()
                        for _ in range(5):
                            compiled(*call_args)
                        torch.spyre.synchronize()
                        elapsed = (time.perf_counter_ns() - start) / 1_000_000
                        samples.append(
                            {
                                "selector": label,
                                "round": round_index,
                                "kind": "block",
                                "sample": block_index,
                                "calls": 5,
                                "per_call_ms": elapsed / 5,
                            }
                        )
            result["timing"] = {
                "protocol": {"warmups": 5, "singles": 50, "blocks": 10, "block_iters": 5, "rounds": 3},
                "orders": orders,
                "samples": samples,
            }
    (output_dir / "compile_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
