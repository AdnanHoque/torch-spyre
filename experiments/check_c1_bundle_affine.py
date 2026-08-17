#!/usr/bin/env python3
"""Fail-closed structural checker for the reduced C1 D-AS-X bundle.

This checker performs no compilation, device launch, or timing.  It accepts
only the exact flat-loop addressing contract needed by the E=2/T=64/H=64/F=64
C1 correctness gate:

* one ``scf.for`` with trip count two;
* one deduplicated affine map ``s0 + 128*d0``;
* exactly four ``affine.apply`` operations inside that loop;
* bases ``arg_2`` through ``arg_5`` advance once each;
* those addresses feed ``sdsc_2``, ``sdsc_5``, ``sdsc_8``, and ``sdsc_10``;
* X/fill/output (``arg_0``, ``arg_1``, ``arg_6``) remain raw and fixed; and
* every other loop SDSC is LX-only and therefore has no bundle operand.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass


class GateFailure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


_MAP_RE = re.compile(
    r"^\s*#map_(?P<id>[0-9]+)\s*=\s*"
    r"affine_map<\s*\(\s*d0\s*\)\s*\[\s*s0\s*\]\s*->\s*"
    r"\(\s*s0\s*\+\s*128\s*\*\s*d0\s*\)\s*>\s*$",
    re.MULTILINE,
)
_ANY_MAP_RE = re.compile(r"^\s*#map_[0-9]+\s*=\s*affine_map<.*$", re.MULTILINE)
_APPLY_RE = re.compile(
    r"^\s*(?P<addr>%addr_[A-Za-z0-9_]+)\s*=\s*affine\.apply\s+"
    r"#map_(?P<map>[0-9]+)\s*\(\s*(?P<loop>%[A-Za-z0-9_]+)\s*\)"
    r"\s*\[\s*(?P<base>%arg_[0-9]+)\s*\]\s*$",
    re.MULTILINE,
)
_ANY_APPLY_RE = re.compile(r"\baffine\.apply\b")
_EXEC_RE = re.compile(
    r"^\s*sdscbundle\.sdsc_execute\s*\((?P<operands>[^)]*)\)\s*"
    r"\{\s*sdsc_filename=\"sdsc_(?P<id>[0-9]+)\.json\"\s*,\s*"
    r"\"symbol_ids\"=\[(?P<symbols>[^]]*)\]\s*\}\s*$",
    re.MULTILINE,
)
_CONST_RE = re.compile(
    r"^\s*(?P<name>%[A-Za-z0-9_]+)\s*=\s*arith\.constant\s+"
    r"(?P<value>-?[0-9]+)\s*:\s*index\s*$",
    re.MULTILINE,
)
_LOOP_RE = re.compile(
    r"\bscf\.for\s+(?P<var>%[A-Za-z0-9_]+)\s*=\s*"
    r"(?P<lower>%[A-Za-z0-9_]+)\s+to\s+(?P<upper>%[A-Za-z0-9_]+)\s+"
    r"step\s+(?P<step>%[A-Za-z0-9_]+)\s*\{"
)


@dataclass(frozen=True)
class Execute:
    sdsc_id: int
    operands: tuple[str, ...]
    symbols: tuple[int, ...]


def _split_operands(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    values = tuple(piece.strip() for piece in raw.split(","))
    require(all(re.fullmatch(r"%[A-Za-z0-9_]+", value) for value in values),
            f"invalid SDSC operand list: {raw!r}")
    return values


def _split_symbols(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    try:
        return tuple(int(piece.strip()) for piece in raw.split(","))
    except ValueError as error:
        raise GateFailure(f"invalid symbol_ids list: {raw!r}") from error


def _executes(text: str) -> list[Execute]:
    return [
        Execute(
            sdsc_id=int(match.group("id")),
            operands=_split_operands(match.group("operands")),
            symbols=_split_symbols(match.group("symbols")),
        )
        for match in _EXEC_RE.finditer(text)
    ]


def _matching_brace(text: str, opening: int) -> int:
    """Return the closing brace while ignoring quoted text and line comments."""
    require(text[opening] == "{", "internal error: opening offset is not a brace")
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
            require(depth >= 0, "unbalanced closing brace")
    raise GateFailure("unterminated scf.for body")


def _single_execute(executions: list[Execute], sdsc_id: int, scope: str) -> Execute:
    matches = [entry for entry in executions if entry.sdsc_id == sdsc_id]
    require(len(matches) == 1, f"{scope}: expected one sdsc_{sdsc_id}, got {len(matches)}")
    return matches[0]


def validate(text: str) -> dict[str, object]:
    require(
        len(re.findall(r"\bfunc\.func\s+@sdsc_bundle\b", text)) == 1,
        "expected exactly one @sdsc_bundle function",
    )
    loop_matches = list(_LOOP_RE.finditer(text))
    require(len(loop_matches) == 1, f"expected exactly one scf.for, got {len(loop_matches)}")
    require(len(re.findall(r"\bscf\.for\b", text)) == 1, "found an unparsed or nested scf.for")
    loop = loop_matches[0]
    opening = loop.end() - 1
    closing = _matching_brace(text, opening)
    prefix = text[: loop.start()]
    body = text[opening + 1 : closing]
    suffix = text[closing + 1 :]
    require(not re.search(r"\bscf\.for\b", body), "nested scf.for is forbidden")

    constants = {match.group("name"): int(match.group("value")) for match in _CONST_RE.finditer(text)}
    lower, upper, step = (loop.group(name) for name in ("lower", "upper", "step"))
    require(constants.get(lower) == 0, f"loop lower bound {lower} is not constant zero")
    require(constants.get(upper) == 2, f"loop upper bound {upper} is not constant two")
    require(constants.get(step) == 1, f"loop step {step} is not constant one")
    loop_var = loop.group("var")

    all_map_lines = list(_ANY_MAP_RE.finditer(text))
    exact_maps = list(_MAP_RE.finditer(text))
    require(len(all_map_lines) == 1, f"expected one affine map declaration, got {len(all_map_lines)}")
    require(len(exact_maps) == 1, "the sole affine map is not exactly (d0)[s0] -> (s0 + 128*d0)")
    map_id = exact_maps[0].group("id")

    all_apply_count = len(_ANY_APPLY_RE.findall(text))
    body_applies = list(_APPLY_RE.finditer(body))
    require(all_apply_count == 4, f"expected exactly four affine.apply operations, got {all_apply_count}")
    require(len(body_applies) == 4, "all four affine.apply operations must be inside the expert loop")
    require(not _ANY_APPLY_RE.search(prefix + suffix), "affine.apply outside the expert loop is forbidden")

    expected_bases = {f"%arg_{index}" for index in range(2, 6)}
    bases = [match.group("base") for match in body_applies]
    require(set(bases) == expected_bases and len(bases) == len(set(bases)),
            f"affine bases must be exactly arg_2/3/4/5 once each, got {bases}")
    require(all(match.group("map") == map_id for match in body_applies),
            "every affine.apply must use the sole deduplicated map")
    require(all(match.group("loop") == loop_var for match in body_applies),
            "every affine.apply must use the expert loop induction variable")
    addr_by_base = {match.group("base"): match.group("addr") for match in body_applies}
    require(len(set(addr_by_base.values())) == 4, "affine.apply SSA results must be distinct")

    loop_executes = _executes(body)
    require([entry.sdsc_id for entry in loop_executes] == list(range(2, 14)),
            f"expert-loop SDSCs must be exactly 2..13 in order, got {[e.sdsc_id for e in loop_executes]}")
    expected_consumers = {2: "%arg_2", 5: "%arg_3", 8: "%arg_4", 10: "%arg_5"}
    for sdsc_id, base in expected_consumers.items():
        execute = _single_execute(loop_executes, sdsc_id, "expert loop")
        expected_addr = addr_by_base[base]
        require(execute.operands == (expected_addr,),
                f"sdsc_{sdsc_id} must consume only {expected_addr} derived from {base}; got {execute.operands}")
        require(len(execute.symbols) == 1,
                f"sdsc_{sdsc_id} must expose exactly one HBM base symbol; got {execute.symbols}")

    for execute in loop_executes:
        if execute.sdsc_id in expected_consumers:
            continue
        require(execute.operands == (),
                f"LX-only sdsc_{execute.sdsc_id} unexpectedly has bundle operands {execute.operands}")
        require(execute.symbols == (),
                f"LX-only sdsc_{execute.sdsc_id} unexpectedly has symbol_ids {execute.symbols}")

    for base in expected_bases:
        require(len(re.findall(rf"{re.escape(base)}\b", body)) == 1,
                f"{base} must appear inside the loop only as its affine.apply base")
    for forbidden in ("%arg_0", "%arg_1", "%arg_6"):
        require(not re.search(rf"{re.escape(forbidden)}\b", body),
                f"fixed operand {forbidden} appears inside the expert loop")

    prefix_executes = _executes(prefix)
    suffix_executes = _executes(suffix)
    require([entry.sdsc_id for entry in prefix_executes] == [0, 1],
            f"preheader SDSCs must be exactly [0,1], got {[e.sdsc_id for e in prefix_executes]}")
    require(prefix_executes[0].operands == ("%arg_0",), "sdsc_0 must use fixed X arg_0")
    require(prefix_executes[1].operands == ("%arg_1",), "sdsc_1 must use fixed fill arg_1")
    require([entry.sdsc_id for entry in suffix_executes] == [14],
            f"post-loop SDSC must be exactly [14], got {[e.sdsc_id for e in suffix_executes]}")
    require(suffix_executes[0].operands == ("%arg_6",), "sdsc_14 must use fixed output arg_6")

    for addr in addr_by_base.values():
        require(len(re.findall(rf"{re.escape(addr)}\b", body)) == 2,
                f"{addr} must be defined once and consumed once")

    return {
        "status": "structural-pass",
        "loop_var": loop_var,
        "loop_count": 2,
        "affine_map": f"#map_{map_id}: s0 + 128*d0",
        "advances": {base: addr_by_base[base] for base in sorted(addr_by_base)},
        "consumers": expected_consumers,
        "fixed_args": ["%arg_0", "%arg_1", "%arg_6"],
    }


def _fixture() -> str:
    lines = [
        "module {",
        "  #map_0 = affine_map<(d0)[s0] -> (s0 + 128*d0)>",
        "  func.func @sdsc_bundle() {",
        "    %c0 = arith.constant 0 : index",
        "    %c1 = arith.constant 1 : index",
        "    %loop_bound_0 = arith.constant 2 : index",
        '    sdscbundle.sdsc_execute (%arg_0) {sdsc_filename="sdsc_0.json", "symbol_ids"=[-1]}',
        '    sdscbundle.sdsc_execute (%arg_1) {sdsc_filename="sdsc_1.json", "symbol_ids"=[-2]}',
        "    scf.for %i_0 = %c0 to %loop_bound_0 step %c1 {",
    ]
    target = {2: (2, 0), 5: (3, 1), 8: (4, 2), 10: (5, 3)}
    for sdsc_id in range(2, 14):
        if sdsc_id in target:
            base, addr = target[sdsc_id]
            lines.append(f"      %addr_{addr} = affine.apply #map_0(%i_0)[%arg_{base}]")
            lines.append(
                f'      sdscbundle.sdsc_execute (%addr_{addr}) '
                f'{{sdsc_filename="sdsc_{sdsc_id}.json", "symbol_ids"=[-{sdsc_id + 1}]}}'
            )
        else:
            lines.append(
                f'      sdscbundle.sdsc_execute () '
                f'{{sdsc_filename="sdsc_{sdsc_id}.json", "symbol_ids"=[]}}'
            )
    lines.extend(
        [
            "    }",
            '    sdscbundle.sdsc_execute (%arg_6) {sdsc_filename="sdsc_14.json", "symbol_ids"=[-7]}',
            "    return",
            "  }",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    good = _fixture()
    validate(good)
    negatives = {
        "raw iteration-0 weight address": good.replace(
            "sdscbundle.sdsc_execute (%addr_0)",
            "sdscbundle.sdsc_execute (%arg_2)",
            1,
        ),
        "wrong stride": good.replace("s0 + 128*d0", "s0 + 8192*d0", 1),
        "wrong consumer": good.replace(
            "sdscbundle.sdsc_execute (%addr_1)",
            "sdscbundle.sdsc_execute (%addr_0)",
            1,
        ),
        "extra fixed-X advance": good.replace(
            "    scf.for %i_0",
            "    %addr_extra = affine.apply #map_0(%c0)[%arg_0]\n    scf.for %i_0",
            1,
        ),
        "nested loop": good.replace(
            "      %addr_0",
            "      scf.for %j = %c0 to %c1 step %c1 {\n      }\n      %addr_0",
            1,
        ),
        "LX SDSC gains address": good.replace(
            'sdscbundle.sdsc_execute () {sdsc_filename="sdsc_3.json"',
            'sdscbundle.sdsc_execute (%addr_0) {sdsc_filename="sdsc_3.json"',
            1,
        ),
        "duplicate map": good.replace(
            "  func.func",
            "  #map_1 = affine_map<(d0)[s0] -> (s0 + 128*d0)>\n  func.func",
            1,
        ),
    }
    for label, bad in negatives.items():
        try:
            validate(bad)
        except GateFailure:
            continue
        raise AssertionError(f"negative fixture unexpectedly passed: {label}")
    print(f"self-test: PASS ({len(negatives)} negative fixtures rejected)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.bundle is not None:
        try:
            result = validate(args.bundle.read_text())
        except (OSError, GateFailure) as error:
            print(f"REJECT: {error}", file=sys.stderr)
            return 1
        print(result)
    elif not args.self_test:
        parser.error("provide a bundle path and/or --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
