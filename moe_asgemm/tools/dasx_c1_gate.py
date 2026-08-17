#!/usr/bin/env python3
"""Fail-closed reduced-C1 gate for the executable D-AS-X comparator.

This checker consumes the generated Torch-Spyre wrapper source and its
``bundle.mlir``.  It does not compile or launch a kernel.  The structural gate
accepts only a single-bundle, single-expert-loop activation-stationary dense
FFN whose internal activations remain in LX.

The optional ``accept`` subcommand additionally validates a two-runtime-alpha
device correctness artifact produced by
``dense_activation_stationary_c1_correctness_gate.py``.  That second gate
proves that two non-binary alpha payloads used the exact same emitted bundle
and both match an independently recomputed FP32 post-down-weighting reference.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1


_BUNDLE_MAP_RE = re.compile(
    r"^\s*#map_(?P<id>[0-9]+)\s*=\s*"
    r"affine_map<\s*\(\s*d0\s*\)\s*\[\s*s0\s*\]\s*->\s*"
    r"\(\s*s0\s*\+\s*128\s*\*\s*d0\s*\)\s*>\s*$",
    re.MULTILINE,
)
_BUNDLE_ANY_MAP_RE = re.compile(r"^\s*#map_[0-9]+\s*=\s*affine_map<.*$", re.MULTILINE)
_BUNDLE_APPLY_RE = re.compile(
    r"^\s*(?P<addr>%addr_[A-Za-z0-9_]+)\s*=\s*affine\.apply\s+"
    r"#map_(?P<map>[0-9]+)\s*\(\s*(?P<loop>%[A-Za-z0-9_]+)\s*\)"
    r"\s*\[\s*(?P<base>%arg_[0-9]+)\s*\]\s*$",
    re.MULTILINE,
)
_BUNDLE_ANY_APPLY_RE = re.compile(r"\baffine\.apply\b")
_BUNDLE_EXEC_RE = re.compile(
    r"^\s*sdscbundle\.sdsc_execute\s*\((?P<operands>[^)]*)\)\s*"
    r"\{\s*sdsc_filename=\"sdsc_(?P<id>[0-9]+)\.json\"\s*,\s*"
    r"\"symbol_ids\"=\[(?P<symbols>[^]]*)\]\s*\}\s*$",
    re.MULTILINE,
)
_BUNDLE_CONST_RE = re.compile(
    r"^\s*(?P<name>%[A-Za-z0-9_]+)\s*=\s*arith\.constant\s+"
    r"(?P<value>-?[0-9]+)\s*:\s*index\s*$",
    re.MULTILINE,
)
_BUNDLE_LOOP_RE = re.compile(
    r"\bscf\.for\s+(?P<var>%[A-Za-z0-9_]+)\s*=\s*"
    r"(?P<lower>%[A-Za-z0-9_]+)\s+to\s+(?P<upper>%[A-Za-z0-9_]+)\s+"
    r"step\s+(?P<step>%[A-Za-z0-9_]+)\s*\{"
)


class GateFailure(AssertionError):
    """A fail-closed D-AS-X acceptance failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    raise GateFailure(f"{_call_name(call)!r} has no {name!r} keyword")


def _optional_keyword(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError) as error:
        raise GateFailure(f"expected literal, got {ast.dump(node)}") from error


def _debug_chain(call: ast.Call) -> tuple[str, ...]:
    debug = _keyword(call, "debug_handle")
    require(
        isinstance(debug, ast.Call) and _call_name(debug) == "DebugHandle",
        "OpSpec debug_handle is not a DebugHandle call",
    )
    chain = _literal(_keyword(debug, "ir_chain"))
    require(
        isinstance(chain, tuple)
        and chain
        and all(isinstance(item, str) for item in chain),
        f"invalid or empty debug ir_chain: {chain!r}",
    )
    return chain


def _aten_op(call: ast.Call) -> str | None:
    debug = _keyword(call, "debug_handle")
    value = _literal(_keyword(debug, "aten_op"))
    require(value is None or isinstance(value, str), f"invalid aten_op: {value!r}")
    return value


def _dtype_name(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ast.unparse(node)


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    is_input: bool
    arg_index: int
    device_size: tuple[int, ...]
    allocation: dict[str, int]
    device_dtype: str
    has_tile_advance: bool

    @property
    def lx_interval(self) -> tuple[int, int] | None:
        if set(self.allocation) != {"lx"}:
            return None
        require(self.device_size, "LX TensorArg has an empty device_size")
        # The final device coordinate is one 128-byte stick payload.
        size_bytes = math.prod(self.device_size[:-1]) * 128
        start = self.allocation["lx"]
        return start, start + size_bytes


@dataclasses.dataclass(frozen=True)
class Op:
    call: ast.Call
    scope: str
    index: int
    op: str
    is_reduction: bool
    chain: tuple[str, ...]
    aten_op: str | None
    tensors: tuple[TensorSpec, ...]

    @property
    def inputs(self) -> tuple[TensorSpec, ...]:
        return tuple(tensor for tensor in self.tensors if tensor.is_input)

    @property
    def outputs(self) -> tuple[TensorSpec, ...]:
        return tuple(tensor for tensor in self.tensors if not tensor.is_input)

    @property
    def label(self) -> str:
        return f"{self.scope}[{self.index}] {self.op} {'/'.join(self.chain)}"


def _tensor_args(call: ast.Call) -> tuple[TensorSpec, ...]:
    args_node = _keyword(call, "args")
    require(isinstance(args_node, ast.List), "OpSpec args is not a list")
    tensors: list[TensorSpec] = []
    for node in args_node.elts:
        require(
            isinstance(node, ast.Call) and _call_name(node) == "TensorArg",
            f"non-TensorArg entry in OpSpec args: {ast.dump(node)}",
        )
        allocation = _literal(_keyword(node, "allocation"))
        device_size = _literal(_keyword(node, "device_size"))
        require(isinstance(allocation, dict), f"invalid allocation: {allocation!r}")
        require(
            all(
                isinstance(key, str) and isinstance(value, int)
                for key, value in allocation.items()
            ),
            f"non-literal allocation: {allocation!r}",
        )
        require(
            isinstance(device_size, list)
            and device_size
            and all(isinstance(value, int) and value >= 1 for value in device_size),
            f"invalid device_size: {device_size!r}",
        )
        tensors.append(
            TensorSpec(
                is_input=bool(_literal(_keyword(node, "is_input"))),
                arg_index=int(_literal(_keyword(node, "arg_index"))),
                device_size=tuple(device_size),
                allocation=dict(allocation),
                device_dtype=_dtype_name(_keyword(node, "device_dtype")),
                has_tile_advance=_optional_keyword(node, "device_tile_advance_expr")
                is not None,
            )
        )
    return tuple(tensors)


def _parse_op(call: ast.Call, *, scope: str, index: int) -> Op:
    require(_call_name(call) == "OpSpec", f"expected OpSpec, got {_call_name(call)!r}")
    op = _literal(_keyword(call, "op"))
    require(isinstance(op, str), f"OpSpec op is not a string: {op!r}")
    is_reduction = _literal(_keyword(call, "is_reduction"))
    require(
        isinstance(is_reduction, bool),
        f"OpSpec is_reduction is not a bool: {is_reduction!r}",
    )
    return Op(
        call=call,
        scope=scope,
        index=index,
        op=op,
        is_reduction=is_reduction,
        chain=_debug_chain(call),
        aten_op=_aten_op(call),
        tensors=_tensor_args(call),
    )


def _overlaps(lhs: tuple[int, int], rhs: tuple[int, int]) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


def _only_alloc(tensor: TensorSpec, kind: str) -> bool:
    return set(tensor.allocation) == {kind}


def _is_hbm_to_lx_copy(op: Op) -> bool:
    return (
        op.op == "identity"
        and len(op.inputs) == 1
        and len(op.outputs) == 1
        and _only_alloc(op.inputs[0], "hbm")
        and _only_alloc(op.outputs[0], "lx")
    )


def _is_chain(op: Op, fragment: str) -> bool:
    return any(fragment in name for name in op.chain)


def _sympify_literal(node: ast.AST) -> str:
    require(
        isinstance(node, ast.Call)
        and _call_name(node) == "sympify"
        and len(node.args) == 1,
        f"expected sympify literal, got {ast.dump(node)}",
    )
    value = _literal(node.args[0])
    require(isinstance(value, str), f"sympify argument is not a string: {value!r}")
    return value


@dataclasses.dataclass
class ParsedWrapper:
    source_path: pathlib.Path
    source_text: str
    tree: ast.Module
    sdsc_name: str
    loop_call: ast.Call
    loop_index: int
    loop_count: str
    top_ops: list[Op]
    loop_ops: list[Op]
    run_call: ast.Call
    asserted_shapes: dict[str, tuple[int, ...]]


def parse_wrapper(path: pathlib.Path) -> ParsedWrapper:
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise GateFailure(f"generated wrapper does not parse: {error}") from error

    sdsc_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sdsc"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "async_compile"
    ]
    require(
        len(sdsc_calls) == 1, f"expected one async_compile.sdsc, got {len(sdsc_calls)}"
    )
    sdsc_call = sdsc_calls[0]
    require(len(sdsc_call.args) >= 2, "async_compile.sdsc is missing its spec list")
    specs = sdsc_call.args[1]
    require(
        isinstance(specs, ast.List), "async_compile.sdsc specs is not a literal list"
    )

    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.value is sdsc_call
    ]
    require(len(assignments) == 1, "could not identify the single SDSC assignment")
    assignment = assignments[0]
    require(
        len(assignment.targets) == 1 and isinstance(assignment.targets[0], ast.Name),
        "SDSC assignment does not have one name target",
    )
    sdsc_name = assignment.targets[0].id

    loops = [
        node
        for node in specs.elts
        if isinstance(node, ast.Call) and _call_name(node) == "LoopSpec"
    ]
    require(
        len(loops) == 1, f"expected exactly one top-level LoopSpec, got {len(loops)}"
    )
    loop = loops[0]
    body = _keyword(loop, "body")
    require(isinstance(body, ast.List), "LoopSpec body is not a literal list")
    nested_loops = [
        node
        for body_spec in body.elts
        for node in ast.walk(body_spec)
        if isinstance(node, ast.Call) and _call_name(node) == "LoopSpec"
    ]
    require(not nested_loops, f"found {len(nested_loops)} nested/temporal LoopSpecs")

    top_ops: list[Op] = []
    for index, node in enumerate(specs.elts):
        require(
            isinstance(node, ast.Call) and _call_name(node) in {"OpSpec", "LoopSpec"},
            f"unexpected top-level SDSC spec: {ast.dump(node)}",
        )
        if _call_name(node) == "OpSpec":
            top_ops.append(_parse_op(node, scope="top", index=index))
    loop_ops: list[Op] = []
    for index, node in enumerate(body.elts):
        require(
            isinstance(node, ast.Call) and _call_name(node) == "OpSpec",
            f"unexpected expert-loop spec: {ast.dump(node)}",
        )
        loop_ops.append(_parse_op(node, scope="loop", index=index))

    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == sdsc_name
    ]
    all_sdsc_runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id.startswith("sdsc_")
    ]
    require(
        len(run_calls) == 1, f"expected one {sdsc_name}.run call, got {len(run_calls)}"
    )
    require(
        len(all_sdsc_runs) == 1,
        f"expected one wrapper SDSC run, got {len(all_sdsc_runs)}",
    )

    asserted_shapes: dict[str, tuple[int, ...]] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_size_stride"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
        ):
            continue
        shape = _literal(node.args[1])
        require(
            isinstance(shape, tuple) and all(isinstance(value, int) for value in shape),
            f"invalid assert_size_stride shape: {shape!r}",
        )
        asserted_shapes[node.args[0].id] = tuple(shape)

    return ParsedWrapper(
        source_path=path,
        source_text=source,
        tree=tree,
        sdsc_name=sdsc_name,
        loop_call=loop,
        loop_index=specs.elts.index(loop),
        loop_count=_sympify_literal(_keyword(loop, "count")),
        top_ops=top_ops,
        loop_ops=loop_ops,
        run_call=run_calls[0],
        asserted_shapes=asserted_shapes,
    )


def _runtime_shape_for_hbm_arg(
    wrapper: ParsedWrapper, hbm_arg_index: int
) -> tuple[int, ...]:
    # TensorArg ``hbm`` indices address runtime arguments directly when no HBM
    # scratch pool exists.  A spill-bearing wrapper inserts the pool first;
    # structure validation rejects those allocations separately.
    require(wrapper.run_call.args, "SDSC run has no scratch-pool argument")
    first = wrapper.run_call.args[0]
    pool_offset = int(isinstance(first, ast.Name) and first.id.startswith("_pool_"))
    position = hbm_arg_index + pool_offset
    require(
        0 <= position < len(wrapper.run_call.args),
        f"HBM arg index {hbm_arg_index} is outside the SDSC run argument list",
    )
    argument = wrapper.run_call.args[position]
    require(
        isinstance(argument, ast.Name),
        f"HBM arg index {hbm_arg_index} is not passed as a named tensor",
    )
    require(
        argument.id in wrapper.asserted_shapes,
        f"runtime tensor {argument.id} has no assert_size_stride shape",
    )
    return wrapper.asserted_shapes[argument.id]


def _single(items: Sequence[Any], message: str) -> Any:
    require(len(items) == 1, f"{message}; got {len(items)}")
    return items[0]


def _output_interval(op: Op) -> tuple[int, int]:
    require(len(op.outputs) == 1, f"{op.label}: expected one output")
    interval = op.outputs[0].lx_interval
    require(interval is not None, f"{op.label}: output is not LX")
    return interval


def _input_intervals(op: Op) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for tensor in op.inputs:
        interval = tensor.lx_interval
        require(
            interval is not None,
            f"{op.label}: non-LX compute input {tensor.allocation}",
        )
        intervals.append(interval)
    return intervals


def _trace_lx_path(
    *,
    start: tuple[int, int],
    target: tuple[int, int],
    candidates: Iterable[Op],
    allowed_ops: set[str],
) -> tuple[bool, list[Op]]:
    """Trace an ordered LX-only data path by exact allocation intervals."""

    reachable = {start}
    used: list[Op] = []
    for op in candidates:
        if op.op not in allowed_ops:
            continue
        inputs = [tensor.lx_interval for tensor in op.inputs]
        outputs = [tensor.lx_interval for tensor in op.outputs]
        if not any(
            interval in reachable for interval in inputs if interval is not None
        ):
            continue
        require(
            all(interval is not None for interval in [*inputs, *outputs]),
            f"{op.label}: alpha/combine path leaves LX",
        )
        reachable.update(interval for interval in outputs if interval is not None)
        used.append(op)
    return target in reachable, used


def _validate_unit_contribution_contract(
    *, loop_ops: Sequence[Op], contribution_path: Sequence[Op], combine: Op
) -> Op:
    """Require the collapsed unit reduction to be identity then accumulator add."""

    actual_sum_ops = [op for op in loop_ops if op.op == "sum"]
    require(
        not actual_sum_ops,
        "collapsed unit reduction must emit zero actual local sum OpSpecs; got "
        f"{[op.label for op in actual_sum_ops]}",
    )
    require(
        len(loop_ops) == 12,
        f"exact reduced-C1 expert loop must map to sdsc_2..sdsc_13; got {len(loop_ops)} OpSpecs",
    )
    identity_contribution = loop_ops[10]
    require(
        list(contribution_path) == [identity_contribution]
        and identity_contribution.op == "identity"
        and not identity_contribution.is_reduction,
        "sdsc_12 must be exactly one non-reduction identity contribution between "
        f"router weighting and accumulation; got {[op.label for op in contribution_path]}",
    )
    require(
        identity_contribution.aten_op == "aten.sum.dim_IntList"
        and identity_contribution.chain[0].startswith("sum"),
        f"sdsc_12 identity lacks collapsed unit-sum provenance: {identity_contribution.label}",
    )
    require(
        combine is loop_ops[11] and combine.op == "add" and not combine.is_reduction,
        f"sdsc_13 must be the non-reduction loop-carried add; got {combine.label}",
    )
    return identity_contribution


@dataclasses.dataclass(frozen=True)
class BundleExecute:
    sdsc_id: int
    operands: tuple[str, ...]
    symbols: tuple[int, ...]


def _split_bundle_operands(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    values = tuple(piece.strip() for piece in raw.split(","))
    require(
        all(re.fullmatch(r"%[A-Za-z0-9_]+", value) for value in values),
        f"invalid SDSC operand list: {raw!r}",
    )
    return values


def _split_bundle_symbols(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    try:
        return tuple(int(piece.strip()) for piece in raw.split(","))
    except ValueError as error:
        raise GateFailure(f"invalid symbol_ids list: {raw!r}") from error


def _bundle_executes(text: str) -> list[BundleExecute]:
    return [
        BundleExecute(
            sdsc_id=int(match.group("id")),
            operands=_split_bundle_operands(match.group("operands")),
            symbols=_split_bundle_symbols(match.group("symbols")),
        )
        for match in _BUNDLE_EXEC_RE.finditer(text)
    ]


def _matching_bundle_brace(text: str, opening: int) -> int:
    """Return the matching brace while ignoring strings and line comments."""

    require(text[opening] == "{", "internal error: loop opening is not a brace")
    depth = 0
    in_string = False
    escaped = False
    in_comment = False
    for index in range(opening, len(text)):
        char = text[index]
        nxt = text[index + 1] if index + 1 < len(text) else ""
        if in_comment:
            if char == "\n":
                in_comment = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "/" and nxt == "/":
            in_comment = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
            require(depth >= 0, "unbalanced closing brace in bundle.mlir")
    raise GateFailure("unterminated scf.for body in bundle.mlir")


def _single_bundle_execute(
    executions: Sequence[BundleExecute], sdsc_id: int, scope: str
) -> BundleExecute:
    matches = [entry for entry in executions if entry.sdsc_id == sdsc_id]
    require(
        len(matches) == 1,
        f"{scope}: expected one sdsc_{sdsc_id}, got {len(matches)}",
    )
    return matches[0]


def _json_values(value: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key == key:
                values.append(child)
            values.extend(_json_values(child, key))
    elif isinstance(value, list):
        for child in value:
            values.extend(_json_values(child, key))
    return values


def _validate_bundle_sdsc_ops(bundle_dir: pathlib.Path) -> dict[int, tuple[str, ...]]:
    expected_ids = set(range(15))
    sdsc_paths = list(bundle_dir.glob("sdsc_*.json"))
    found: dict[int, pathlib.Path] = {}
    for path in sdsc_paths:
        match = re.fullmatch(r"sdsc_([0-9]+)\.json", path.name)
        require(match is not None, f"unrecognized SDSC filename: {path.name}")
        sdsc_id = int(match.group(1))
        require(sdsc_id not in found, f"duplicate SDSC id {sdsc_id}")
        found[sdsc_id] = path
    require(
        set(found) == expected_ids,
        "bundle directory must contain exactly sdsc_0.json through sdsc_14.json; "
        f"got {sorted(found)}",
    )

    functions: dict[int, tuple[str, ...]] = {}
    for sdsc_id, path in sorted(found.items()):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise GateFailure(f"{path.name} is not valid JSON: {error}") from error
        op_functions = tuple(_json_values(payload, "opFuncName"))
        require(
            len(op_functions) == 1 and isinstance(op_functions[0], str),
            f"{path.name} must contain exactly one string opFuncName; got {op_functions}",
        )
        functions[sdsc_id] = op_functions

    require(
        not any(
            "sum" in function.lower()
            for values in functions.values()
            for function in values
        ),
        f"actual local sum SDSC is forbidden: {functions}",
    )
    require(
        functions[12] == ("identity",),
        f"sdsc_12 must be the unit-reduction identity contribution; got {functions[12]}",
    )
    require(
        functions[13] == ("add",),
        f"sdsc_13 must be the loop-carried accumulator add; got {functions[13]}",
    )
    return functions


def _validate_bundle_mlir(path: pathlib.Path, expected_e: int) -> dict[str, Any]:
    text = path.read_text()
    require(
        expected_e == 2, f"exact reduced-C1 bundle requires E=2, got E={expected_e}"
    )
    require(
        len(re.findall(r"\bfunc\.func\s+@sdsc_bundle\b", text)) == 1,
        "bundle.mlir must contain exactly one @sdsc_bundle function",
    )

    loop_matches = list(_BUNDLE_LOOP_RE.finditer(text))
    require(
        len(loop_matches) == 1,
        f"bundle.mlir must contain one parsed scf.for, got {len(loop_matches)}",
    )
    require(
        len(re.findall(r"\bscf\.for\b", text)) == 1,
        "bundle.mlir contains an unparsed or nested scf.for",
    )
    loop = loop_matches[0]
    opening = loop.end() - 1
    closing = _matching_bundle_brace(text, opening)
    prefix = text[: loop.start()]
    body = text[opening + 1 : closing]
    suffix = text[closing + 1 :]
    require(not re.search(r"\bscf\.for\b", body), "nested scf.for is forbidden")

    constants = {
        match.group("name"): int(match.group("value"))
        for match in _BUNDLE_CONST_RE.finditer(text)
    }
    lower, upper, step = (loop.group(name) for name in ("lower", "upper", "step"))
    require(constants.get(lower) == 0, f"loop lower bound {lower} is not constant zero")
    require(constants.get(upper) == 2, f"loop upper bound {upper} is not constant two")
    require(constants.get(step) == 1, f"loop step {step} is not constant one")
    loop_var = loop.group("var")

    all_maps = list(_BUNDLE_ANY_MAP_RE.finditer(text))
    exact_maps = list(_BUNDLE_MAP_RE.finditer(text))
    require(
        len(all_maps) == 1, f"expected one affine map declaration, got {len(all_maps)}"
    )
    require(
        len(exact_maps) == 1,
        "the sole affine map must be exactly (d0)[s0] -> (s0 + 128*d0)",
    )
    map_id = exact_maps[0].group("id")

    all_apply_count = len(_BUNDLE_ANY_APPLY_RE.findall(text))
    body_applies = list(_BUNDLE_APPLY_RE.finditer(body))
    require(
        all_apply_count == 4,
        f"expected exactly four affine.apply operations, got {all_apply_count}",
    )
    require(
        len(body_applies) == 4,
        "all four affine.apply operations must be inside the expert loop",
    )
    require(
        not _BUNDLE_ANY_APPLY_RE.search(prefix + suffix),
        "affine.apply outside the expert loop is forbidden",
    )

    expected_bases = {f"%arg_{index}" for index in range(2, 6)}
    bases = [match.group("base") for match in body_applies]
    require(
        set(bases) == expected_bases and len(bases) == len(set(bases)),
        f"affine bases must be exactly arg_2/3/4/5 once each, got {bases}",
    )
    require(
        all(match.group("map") == map_id for match in body_applies),
        "every affine.apply must use the sole deduplicated map",
    )
    require(
        all(match.group("loop") == loop_var for match in body_applies),
        "every affine.apply must use the expert loop induction variable",
    )
    addr_by_base = {match.group("base"): match.group("addr") for match in body_applies}
    require(
        len(set(addr_by_base.values())) == 4,
        "affine.apply SSA results must be distinct",
    )

    parsed_executes = _bundle_executes(text)
    require(
        len(parsed_executes) == len(re.findall(r"\bsdscbundle\.sdsc_execute\b", text)),
        "bundle.mlir contains an unparsed sdsc_execute operation",
    )
    loop_executes = _bundle_executes(body)
    require(
        [entry.sdsc_id for entry in loop_executes] == list(range(2, 14)),
        "expert-loop SDSCs must be exactly 2..13 in order; got "
        f"{[entry.sdsc_id for entry in loop_executes]}",
    )
    expected_consumers = {2: "%arg_2", 5: "%arg_3", 8: "%arg_4", 10: "%arg_5"}
    for sdsc_id, base in expected_consumers.items():
        execute = _single_bundle_execute(loop_executes, sdsc_id, "expert loop")
        expected_addr = addr_by_base[base]
        require(
            execute.operands == (expected_addr,),
            f"sdsc_{sdsc_id} must consume only {expected_addr} derived from {base}; "
            f"got {execute.operands}",
        )
        require(
            len(execute.symbols) == 1,
            f"sdsc_{sdsc_id} must expose exactly one HBM base symbol; got {execute.symbols}",
        )

    for execute in loop_executes:
        if execute.sdsc_id in expected_consumers:
            continue
        require(
            execute.operands == (),
            f"LX-only sdsc_{execute.sdsc_id} unexpectedly has operands {execute.operands}",
        )
        require(
            execute.symbols == (),
            f"LX-only sdsc_{execute.sdsc_id} unexpectedly has symbols {execute.symbols}",
        )

    for base in expected_bases:
        require(
            len(re.findall(rf"{re.escape(base)}\b", body)) == 1,
            f"{base} must occur inside the loop only as its affine.apply base",
        )
    for fixed in ("%arg_0", "%arg_1", "%arg_6"):
        require(
            not re.search(rf"{re.escape(fixed)}\b", body),
            f"fixed operand {fixed} appears inside the expert loop",
        )

    prefix_executes = _bundle_executes(prefix)
    suffix_executes = _bundle_executes(suffix)
    require(
        [entry.sdsc_id for entry in prefix_executes] == [0, 1],
        f"preheader SDSCs must be exactly [0,1], got {[e.sdsc_id for e in prefix_executes]}",
    )
    require(
        prefix_executes[0].operands == ("%arg_0",),
        "sdsc_0 must consume fixed X arg_0",
    )
    require(
        prefix_executes[1].operands == ("%arg_1",),
        "sdsc_1 must consume fixed accumulator-fill arg_1",
    )
    require(
        [entry.sdsc_id for entry in suffix_executes] == [14],
        f"post-loop SDSC must be exactly [14], got {[e.sdsc_id for e in suffix_executes]}",
    )
    require(
        suffix_executes[0].operands == ("%arg_6",),
        "sdsc_14 must consume fixed output arg_6",
    )

    for address in addr_by_base.values():
        require(
            len(re.findall(rf"{re.escape(address)}\b", body)) == 2,
            f"{address} must be defined once and consumed once",
        )

    sdsc_functions = _validate_bundle_sdsc_ops(path.parent)
    return {
        "sha256": sha256_file(path),
        "single_sdsc_bundle_function": True,
        "single_scf_for": True,
        "loop_count": 2,
        "affine_map": f"#map_{map_id}: s0 + 128*d0",
        "expert_hbm_loop_advances_bytes": {base: 128 for base in sorted(addr_by_base)},
        "expert_hbm_consumers": expected_consumers,
        "fixed_hbm_args": ["%arg_0", "%arg_1", "%arg_6"],
        "sdsc_12_op": sdsc_functions[12][0],
        "sdsc_13_op": sdsc_functions[13][0],
        "actual_sum_sdscs": 0,
    }


def validate_structure(
    *,
    generated_module: pathlib.Path,
    bundle_mlir: pathlib.Path,
    expected_e: int,
    expected_t: int,
    expected_h: int,
    expected_f: int,
) -> dict[str, Any]:
    expected_shape = (2, 64, 64, 64)
    require(
        (expected_e, expected_t, expected_h, expected_f) == expected_shape,
        "canonical reduced-C1 gate accepts only E=2,T=64,H=64,F=64; got "
        f"E={expected_e},T={expected_t},H={expected_h},F={expected_f}",
    )
    wrapper = parse_wrapper(generated_module)
    require(
        wrapper.loop_count == str(expected_e),
        f"wrapper expert loop count is {wrapper.loop_count!r}, expected {expected_e}",
    )
    bundle = _validate_bundle_mlir(bundle_mlir, expected_e)

    all_ops = [*wrapper.top_ops, *wrapper.loop_ops]
    for op in all_ops:
        require(
            "restickify" not in op.op.lower(), f"{op.label}: restickify is forbidden"
        )
        require(
            not any("restickify" in name.lower() for name in op.chain),
            f"{op.label}: restickify provenance is forbidden",
        )
        for tensor in op.tensors:
            require(
                "hbm_pool" not in tensor.allocation,
                f"{op.label}: hbm_pool allocation is forbidden: {tensor.allocation}",
            )

    x_copy = _single(
        [
            op
            for op in wrapper.top_ops
            if _is_hbm_to_lx_copy(op)
            and _is_chain(op, "coarse_tile_read_copy")
            and _is_chain(op, "arg0")
            and _is_chain(op, "activation_stationary_shared_lhs_mm")
        ],
        "expected one physical X preheader HBM-to-LX copy",
    )
    fills = [op for op in wrapper.top_ops if _is_chain(op, "coarse_tile_fill")]
    drains = [op for op in wrapper.top_ops if _is_chain(op, "coarse_tile_reduce_copy")]
    fill = _single(fills, "expected one pre-loop accumulator fill")
    drain = _single(drains, "expected one post-loop output drain")
    require(
        len(wrapper.top_ops) == 3,
        f"top level must contain only X-copy, accumulator-fill, and drain; got {[op.label for op in wrapper.top_ops]}",
    )
    require(
        x_copy.index < fill.index < wrapper.loop_index < drain.index,
        "required top-level order is X-copy, accumulator-fill, expert-loop, final-drain",
    )

    x_source, x_lx = x_copy.inputs[0], x_copy.outputs[0]
    require(x_source.arg_index >= 0, "X preheader source is not a runtime graph input")
    require(
        not x_source.has_tile_advance,
        "X preheader source advances with the expert loop",
    )
    require(not x_lx.has_tile_advance, "X LX destination advances with the expert loop")
    require(
        x_source.device_dtype == "SEN169_FP16" and x_lx.device_dtype == "SEN169_FP16",
        f"X must be FP16, got {x_source.device_dtype}/{x_lx.device_dtype}",
    )
    expected_x_device_size = (1, expected_t, expected_h)
    require(
        x_source.device_size == expected_x_device_size
        and x_lx.device_size == expected_x_device_size,
        f"X must be one compact [1,T,H] tile {expected_x_device_size}, got "
        f"{x_source.device_size}/{x_lx.device_size}",
    )
    x_interval = x_lx.lx_interval
    require(x_interval is not None, "X destination has no LX interval")
    require(
        x_interval[1] - x_interval[0] == expected_t * expected_h * 2,
        f"X LX payload is {x_interval[1] - x_interval[0]} bytes, expected {expected_t * expected_h * 2}",
    )

    require(
        _is_hbm_to_lx_copy(fill) and len(fill.inputs) == 1 and len(fill.outputs) == 1,
        f"{fill.label}: accumulator fill must be one HBM scalar-to-LX operation",
    )
    accumulator = _output_interval(fill)
    require(
        not _overlaps(accumulator, x_interval),
        f"accumulator {accumulator} overlaps persistent X {x_interval}",
    )
    require(
        len(drain.inputs) == 1
        and len(drain.outputs) == 1
        and _only_alloc(drain.inputs[0], "lx")
        and _only_alloc(drain.outputs[0], "hbm")
        and drain.inputs[0].lx_interval == accumulator,
        f"{drain.label}: final drain does not consume the fixed LX accumulator",
    )
    require(drain.outputs[0].arg_index >= 0, "final drain is not a wrapper output")

    hbm_writes = [
        (op, tensor)
        for op in all_ops
        for tensor in op.outputs
        if _only_alloc(tensor, "hbm")
    ]
    require(
        len(hbm_writes) == 1 and hbm_writes[0][0] is drain,
        f"expected only the final HBM output; got {[op.label for op, _ in hbm_writes]}",
    )

    loop_copies = [op for op in wrapper.loop_ops if _is_hbm_to_lx_copy(op)]
    require(
        len(loop_copies) == 4,
        f"expert loop must have exactly gate/up/down weight and alpha HBM-to-LX copies; got {[op.label for op in loop_copies]}",
    )
    for copy in loop_copies:
        require(
            copy.inputs[0].arg_index >= 0,
            f"{copy.label}: HBM source is not a runtime input",
        )
        require(
            copy.inputs[0].has_tile_advance,
            f"{copy.label}: source does not advance by expert",
        )

    shared_weight_copies = [
        op
        for op in loop_copies
        if op.chain[0].startswith("activation_stationary_shared_lhs_mm")
    ]
    require(
        len(shared_weight_copies) == 2,
        f"expected two shared-LHS weight copies, got {[op.label for op in shared_weight_copies]}",
    )
    down_weight_copy = _single(
        [op for op in loop_copies if op.chain[0] == "bmm"],
        "expected one down-weight copy",
    )
    alpha_copy = _single(
        [
            op
            for op in loop_copies
            if op not in [*shared_weight_copies, down_weight_copy]
        ],
        "expected one runtime-alpha copy",
    )
    alpha_runtime_shape = _runtime_shape_for_hbm_arg(
        wrapper, alpha_copy.inputs[0].arg_index
    )
    require(
        alpha_runtime_shape == (expected_e, expected_t, 1),
        f"runtime alpha ABI must be [E,T,1]={(expected_e, expected_t, 1)}, got {alpha_runtime_shape}",
    )

    copy_set = {id(op) for op in loop_copies}
    for op in wrapper.loop_ops:
        if id(op) in copy_set:
            continue
        for tensor in op.tensors:
            require(
                _only_alloc(tensor, "lx"),
                f"{op.label}: internal compute tensor is not LX: {tensor.allocation}",
            )

    allowed_loop_ops = {"identity", "batchmatmul", "gelufwd", "mul", "sum", "add"}
    require(
        all(op.op in allowed_loop_ops for op in wrapper.loop_ops),
        f"unexpected expert-loop operation(s): {[op.label for op in wrapper.loop_ops if op.op not in allowed_loop_ops]}",
    )

    bmms = [op for op in wrapper.loop_ops if op.op == "batchmatmul"]
    shared_bmms = [
        op
        for op in bmms
        if op.chain[0].startswith("activation_stationary_shared_lhs_mm")
    ]
    require(
        len(shared_bmms) == 2,
        f"expected gate/up shared-LHS BMMs, got {len(shared_bmms)}",
    )
    down = _single(
        [op for op in bmms if op not in shared_bmms], "expected one down BMM"
    )
    require(len(bmms) == 3, f"expected exactly three FFN BMMs, got {len(bmms)}")

    shared_by_root = {op.chain[0]: op for op in shared_bmms}
    copies_by_root = {op.chain[0]: op for op in shared_weight_copies}
    require(
        set(shared_by_root) == set(copies_by_root),
        f"shared-LHS BMM/copy roots differ: {set(shared_by_root)} vs {set(copies_by_root)}",
    )
    for root, bmm in shared_by_root.items():
        require(
            len(bmm.inputs) == 2 and len(bmm.outputs) == 1,
            f"{bmm.label}: invalid BMM arity",
        )
        bmm_inputs = _input_intervals(bmm)
        require(
            bmm_inputs[0] == x_interval,
            f"{bmm.label}: first operand is not persistent X",
        )
        require(
            bmm_inputs[1] == _output_interval(copies_by_root[root]),
            f"{bmm.label}: second operand does not consume its streamed weight slab",
        )

    # Reject every use or allocation overlapping persistent X except the two
    # read-only first operands above.  This catches cross-iteration aliasing.
    for op in wrapper.loop_ops:
        for index, tensor in enumerate(op.tensors):
            interval = tensor.lx_interval
            if interval is None or not _overlaps(interval, x_interval):
                continue
            legitimate = (
                op in shared_bmms
                and index == 0
                and tensor.is_input
                and interval == x_interval
            )
            require(
                legitimate,
                f"{op.label}: tensor {index} interval {interval} aliases persistent X {x_interval}",
            )

    gelu = _single(
        [op for op in wrapper.loop_ops if op.op == "gelufwd"], "expected one GELU"
    )
    muls = [op for op in wrapper.loop_ops if op.op == "mul"]
    require(
        len(muls) == 2,
        f"expected gated and router-weight muls, got {[op.label for op in muls]}",
    )
    gated_candidates = [op for op in muls if op.index < down.index]
    route_candidates = [op for op in muls if op.index > down.index]
    gated = _single(gated_candidates, "expected one gated-activation mul before down")
    route_mul = _single(route_candidates, "expected one router-weight mul after down")
    require(
        gelu.index < gated.index < down.index < route_mul.index,
        "required dataflow order is GELU, gated mul, down BMM, router-weight mul",
    )

    require(
        len(gelu.inputs) == 1 and len(gelu.outputs) == 1,
        f"{gelu.label}: invalid GELU arity",
    )
    gelu_in = _input_intervals(gelu)[0]
    shared_outputs = {_output_interval(op) for op in shared_bmms}
    require(
        gelu_in in shared_outputs, f"{gelu.label}: input does not come from gate BMM"
    )
    gelu_out = _output_interval(gelu)
    gated_inputs = _input_intervals(gated)
    require(
        gelu_out in gated_inputs
        and any(
            interval in shared_outputs and interval != gelu_in
            for interval in gated_inputs
        ),
        f"{gated.label}: does not combine GELU(gate) with up",
    )
    hidden = _output_interval(gated)

    require(
        len(down.inputs) == 2 and len(down.outputs) == 1,
        f"{down.label}: invalid down BMM arity",
    )
    down_inputs = _input_intervals(down)
    require(
        down_inputs[0] == hidden, f"{down.label}: first operand is not gated hidden"
    )
    require(
        down_inputs[1] == _output_interval(down_weight_copy),
        f"{down.label}: second operand does not consume the streamed down weight",
    )
    down_output = _output_interval(down)

    require(
        len(route_mul.inputs) == 2 and len(route_mul.outputs) == 1,
        f"{route_mul.label}: invalid route mul arity",
    )
    route_inputs = _input_intervals(route_mul)
    require(
        down_output in route_inputs,
        f"{route_mul.label}: does not directly consume the LX down-projection output {down_output}",
    )
    route_operand = (
        route_inputs[1] if route_inputs[0] == down_output else route_inputs[0]
    )
    alpha_start = _output_interval(alpha_copy)
    alpha_candidates = [
        op
        for op in wrapper.loop_ops
        if alpha_copy.index < op.index < route_mul.index
        and op not in [*shared_bmms, gelu, gated, down]
        and id(op) not in copy_set
    ]
    alpha_reaches_route, alpha_path = _trace_lx_path(
        start=alpha_start,
        target=route_operand,
        candidates=alpha_candidates,
        allowed_ops={"identity", "sum"},
    )
    require(
        alpha_reaches_route,
        f"runtime alpha LX interval {alpha_start} does not feed post-down route operand {route_operand}",
    )
    require(
        alpha_start == route_operand and not alpha_path,
        "runtime [E,T,1] alpha must feed the post-down mul directly; "
        f"found preprocessing path {[op.label for op in alpha_path]}",
    )

    adds = [op for op in wrapper.loop_ops if op.op == "add"]
    combine = _single(adds, "expected one accumulator combine add")
    require(
        combine.index > route_mul.index,
        f"{combine.label}: combine precedes route weighting",
    )
    require(
        len(combine.inputs) == 2 and len(combine.outputs) == 1,
        f"{combine.label}: invalid add arity",
    )
    combine_inputs = _input_intervals(combine)
    combine_output = _output_interval(combine)
    require(
        accumulator in combine_inputs and combine_output == accumulator,
        f"{combine.label}: add is not in-place into fixed accumulator {accumulator}",
    )
    contribution = (
        combine_inputs[1] if combine_inputs[0] == accumulator else combine_inputs[0]
    )
    route_output = _output_interval(route_mul)
    contribution_candidates = [
        op
        for op in wrapper.loop_ops
        if route_mul.index < op.index < combine.index and op not in [route_mul]
    ]
    route_reaches_combine, contribution_path = _trace_lx_path(
        start=route_output,
        target=contribution,
        candidates=contribution_candidates,
        allowed_ops={"identity", "sum"},
    )
    require(
        route_reaches_combine,
        f"router-weighted output {route_output} does not feed combine contribution {contribution}",
    )
    _validate_unit_contribution_contract(
        loop_ops=wrapper.loop_ops,
        contribution_path=contribution_path,
        combine=combine,
    )

    permitted_ids = {
        *(id(op) for op in loop_copies),
        *(id(op) for op in shared_bmms),
        id(gelu),
        id(gated),
        id(down),
        id(route_mul),
        id(combine),
        *(id(op) for op in alpha_path),
        *(id(op) for op in contribution_path),
    }
    unaccounted = [op.label for op in wrapper.loop_ops if id(op) not in permitted_ids]
    require(not unaccounted, f"unaccounted expert-loop operations: {unaccounted}")

    # The fixed accumulator and X are loop-carried live state.  No temporary
    # output may overlap either interval.
    for op in wrapper.loop_ops:
        for tensor in op.outputs:
            interval = tensor.lx_interval
            if interval is None:
                continue
            if op is combine and interval == accumulator:
                continue
            require(
                not _overlaps(interval, accumulator),
                f"{op.label}: output {interval} aliases loop-carried accumulator {accumulator}",
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "structural-pass",
        "scope": "reduced C1 D-AS-X compile-only structural acceptance",
        "shape": {
            "E": expected_e,
            "T": expected_t,
            "H": expected_h,
            "F": expected_f,
            "cores": 1,
        },
        "generated_module": {
            "path": str(generated_module),
            "sha256": sha256_file(generated_module),
            "single_async_sdsc": True,
            "single_wrapper_run": True,
        },
        "bundle_mlir": {"path": str(bundle_mlir), **bundle},
        "structural": {
            "one_static_expert_loop": True,
            "no_temporal_token_loop": True,
            "x_preheader_hbm_to_lx_copies": 1,
            "x_lx_interval_bytes": list(x_interval),
            "x_alias_free_through_loop": True,
            "internal_compute_allocations": "LX-only",
            "down_output_lx_interval_bytes": list(down_output),
            "runtime_alpha_hbm_arg_index": alpha_copy.inputs[0].arg_index,
            "runtime_alpha_logical_shape": list(alpha_runtime_shape),
            "runtime_alpha_lx_interval_bytes": list(alpha_start),
            "runtime_alpha_preprocess_ops": [],
            "router_weighting_after_down": True,
            "router_weighted_lx_interval_bytes": list(route_output),
            "fixed_accumulator_lx_interval_bytes": list(accumulator),
            "combine_path_ops": [op.op for op in contribution_path],
            "identity_contribution_ops": 1,
            "actual_local_sum_opspecs": 0,
            "sdsc_12_op": bundle["sdsc_12_op"],
            "sdsc_13_op": bundle["sdsc_13_op"],
            "final_hbm_outputs": 1,
            "hbm_pool_allocations": 0,
            "restickify_ops": 0,
        },
        "timing_collected": False,
        "kernel_launched_by_checker": False,
    }


def _tensor_metrics(actual: Any, reference: Any) -> dict[str, float]:
    import torch

    actual_f = actual.detach().cpu().float()
    reference_f = reference.detach().cpu().float()
    delta = actual_f - reference_f
    ref_norm = float(torch.linalg.vector_norm(reference_f))
    delta_norm = float(torch.linalg.vector_norm(delta))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f.reshape(1, -1), reference_f.reshape(1, -1), dim=1
        ).item()
    )
    return {
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": delta_norm / max(ref_norm, 1e-12),
        "cosine": cosine,
    }


def _cpu_reference(payload: dict[str, Any], alpha: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    x = payload["x"].float()
    gate_w = payload["gate_w"].float()
    up_w = payload["up_w"].float()
    down_w = payload["down_w"].float()
    require(
        alpha.ndim == 3 and alpha.shape[-1] == 1,
        f"alpha must have direct singleton shape [E,T,1], got {tuple(alpha.shape)}",
    )
    route = alpha.float().squeeze(-1)
    output = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32)
    for expert in range(gate_w.shape[0]):
        gate = x @ gate_w[expert]
        up = x @ up_w[expert]
        hidden = functional.gelu(gate, approximate="tanh") * up
        down = hidden @ down_w[expert]
        output.add_(down * route[expert].unsqueeze(-1))
    return output


def validate_correctness_artifact(
    *,
    artifact_path: pathlib.Path,
    structural: dict[str, Any],
    rtol: float,
    atol: float,
    min_cosine: float,
    max_relative_l2: float,
) -> dict[str, Any]:
    import torch

    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    require(isinstance(artifact, dict), "correctness artifact is not a dictionary")
    require(
        artifact.get("schema_version") == SCHEMA_VERSION, "correctness schema mismatch"
    )
    require(
        artifact.get("timing_collected") is False,
        "correctness artifact contains timing",
    )

    identity = artifact.get("bundle_identity")
    require(isinstance(identity, dict), "missing bundle_identity")
    source_sha = structural["generated_module"]["sha256"]
    bundle_sha = structural["bundle_mlir"]["sha256"]
    require(
        identity.get("generated_source_sha256") == source_sha, "source hash mismatch"
    )
    require(
        identity.get("first_call_bundle_sha256") == bundle_sha,
        "first-call bundle hash mismatch",
    )
    require(
        identity.get("second_call_bundle_sha256") == bundle_sha,
        "second-call bundle hash mismatch",
    )
    require(
        identity.get("new_bundles_after_second_call") == [],
        "second alpha call emitted another bundle",
    )
    require(
        identity.get("same_compiled_callable") is True,
        "calls did not use one compiled callable",
    )

    inputs = artifact.get("inputs")
    alphas = artifact.get("alphas")
    expected_route_alpha = artifact.get("reference_route_alpha")
    outputs = artifact.get("device_outputs")
    saved_refs = artifact.get("fp32_references")
    require(
        all(
            isinstance(item, dict)
            for item in (inputs, alphas, expected_route_alpha, outputs, saved_refs)
        ),
        "missing tensor sections",
    )
    for key in ("x", "gate_w", "up_w", "down_w"):
        require(
            key in inputs and isinstance(inputs[key], torch.Tensor),
            f"missing input tensor {key}",
        )
    for label in ("a", "b"):
        require(
            label in alphas and isinstance(alphas[label], torch.Tensor),
            f"missing alpha {label}",
        )
        require(
            label in expected_route_alpha
            and isinstance(expected_route_alpha[label], torch.Tensor),
            f"missing expected route alpha {label}",
        )
        require(
            label in outputs and isinstance(outputs[label], torch.Tensor),
            f"missing device output {label}",
        )
        require(
            label in saved_refs and isinstance(saved_refs[label], torch.Tensor),
            f"missing FP32 reference {label}",
        )

    expected_shape = structural["shape"]
    e, t, h, f = (
        expected_shape["E"],
        expected_shape["T"],
        expected_shape["H"],
        expected_shape["F"],
    )
    require(tuple(inputs["x"].shape) == (t, h), "X shape mismatch")
    require(tuple(inputs["gate_w"].shape) == (e, h, f), "gate weight shape mismatch")
    require(tuple(inputs["up_w"].shape) == (e, h, f), "up weight shape mismatch")
    require(tuple(inputs["down_w"].shape) == (e, f, h), "down weight shape mismatch")
    require(
        all(inputs[key].dtype == torch.float16 for key in inputs), "inputs are not FP16"
    )

    for label, alpha in alphas.items():
        require(alpha.dtype == torch.float16, f"alpha {label} is not FP16")
        require(tuple(alpha.shape) == (e, t, 1), f"alpha {label} must be [E,T,1]")
        require(
            bool(torch.isfinite(alpha).all()),
            f"alpha {label} contains non-finite values",
        )
        require(
            bool((alpha != 0).all() and (alpha != 1).all()),
            f"alpha {label} contains binary 0/1 values",
        )
        require(int(torch.unique(alpha).numel()) > 1, f"alpha {label} is constant")
        derived_route = alpha.float().squeeze(-1)
        expected_route = expected_route_alpha[label].float()
        require(
            tuple(expected_route.shape) == (e, t), f"route alpha {label} shape mismatch"
        )
        torch.testing.assert_close(derived_route, expected_route, rtol=0.0, atol=0.0)
        require(
            bool(
                torch.isfinite(expected_route).all()
                and (expected_route != 0).all()
                and (expected_route != 1).all()
            ),
            f"route alpha {label} is not finite and strictly non-binary",
        )
    require(
        not torch.equal(alphas["a"], alphas["b"]),
        "alpha payloads A and B are identical",
    )

    references = {label: _cpu_reference(inputs, alphas[label]) for label in ("a", "b")}
    reports: dict[str, Any] = {}
    for label in ("a", "b"):
        torch.testing.assert_close(
            saved_refs[label].float(), references[label], rtol=1e-6, atol=1e-6
        )
        actual = outputs[label].float()
        reference = references[label]
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        metrics = _tensor_metrics(actual, reference)
        require(
            metrics["cosine"] >= min_cosine, f"alpha {label} cosine failed: {metrics}"
        )
        require(
            metrics["relative_l2"] <= max_relative_l2,
            f"alpha {label} relative-L2 failed: {metrics}",
        )
        reports[label] = metrics

    reference_delta = references["b"] - references["a"]
    actual_delta = outputs["b"].float() - outputs["a"].float()
    delta_norm = float(torch.linalg.vector_norm(reference_delta))
    base_norm = float(torch.linalg.vector_norm(references["a"]))
    require(
        delta_norm >= 1e-4,
        f"alpha payloads have negligible output effect: {delta_norm}",
    )
    require(
        delta_norm / max(base_norm, 1e-12) >= 0.05,
        "alpha payloads do not change the reference output by at least 5% relative L2",
    )
    delta_metrics = _tensor_metrics(actual_delta, reference_delta)
    require(
        delta_metrics["cosine"] >= 0.995,
        f"runtime-alpha delta cosine failed: {delta_metrics}",
    )
    require(
        delta_metrics["relative_l2"] <= max(0.05, max_relative_l2),
        f"runtime-alpha delta relative-L2 failed: {delta_metrics}",
    )

    return {
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "same_compiled_callable": True,
        "same_emitted_bundle_for_two_alpha_payloads": True,
        "nonbinary_runtime_alpha_payloads": 2,
        "per_payload": reports,
        "alpha_response_delta": delta_metrics,
        "reference_delta_l2": delta_norm,
        "passed": True,
    }


def _write_json(path: pathlib.Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--generated-module", type=pathlib.Path, required=True)
    parser.add_argument("--bundle-mlir", type=pathlib.Path, required=True)
    parser.add_argument("--e", type=int, default=2)
    parser.add_argument("--t", type=int, default=64)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--f", type=int, default=64)
    parser.add_argument("--output-json", type=pathlib.Path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    structure_parser = subparsers.add_parser(
        "structure", help="run compile-only structural gate"
    )
    _add_common_arguments(structure_parser)
    accept_parser = subparsers.add_parser(
        "accept", help="run structure and two-alpha correctness gates"
    )
    _add_common_arguments(accept_parser)
    accept_parser.add_argument(
        "--correctness-artifact", type=pathlib.Path, required=True
    )
    accept_parser.add_argument("--rtol", type=float, default=0.03)
    accept_parser.add_argument("--atol", type=float, default=0.05)
    accept_parser.add_argument("--min-cosine", type=float, default=0.999)
    accept_parser.add_argument("--max-relative-l2", type=float, default=0.03)
    args = parser.parse_args()

    try:
        structural = validate_structure(
            generated_module=args.generated_module,
            bundle_mlir=args.bundle_mlir,
            expected_e=args.e,
            expected_t=args.t,
            expected_h=args.h,
            expected_f=args.f,
        )
        payload: dict[str, Any] = structural
        if args.command == "accept":
            correctness = validate_correctness_artifact(
                artifact_path=args.correctness_artifact,
                structural=structural,
                rtol=args.rtol,
                atol=args.atol,
                min_cosine=args.min_cosine,
                max_relative_l2=args.max_relative_l2,
            )
            payload = {
                **structural,
                "status": "accepted",
                "scope": "reduced C1 D-AS-X structure plus same-bundle two-alpha correctness",
                "correctness": correctness,
            }
        _write_json(args.output_json, payload)
        return 0
    except (GateFailure, AssertionError, OSError, ValueError, KeyError) as error:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "rejected",
            "error_type": type(error).__name__,
            "reason": str(error),
            "kernel_launched_by_checker": False,
            "timing_collected": False,
        }
        _write_json(args.output_json, failure)
        return 2


if __name__ == "__main__":
    sys.exit(main())
