#!/usr/bin/env python3
"""Patch post-DDC restickify LX addresses and rerun Deeptools.

Stage 46 reduced the mirrored DDL failure to production-like per-core input LX
start addresses. This helper takes a post-DDC SDSC, rewrites selected LX
allocate-node start addresses to compact local starts, and reruns DCC/DXP. It is
diagnostic only; it does not model a safe runtime aliasing contract.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


_BOUNDARY_RE = re.compile(
    r"Register initialization out of boundary:\s*([^\n]+?:\s*[^\n]+?)\s*:\s*(\d+)"
)
_UNIT_RE = re.compile(r'name = "([^"]+)"')
_WORK_OP_RE = re.compile(
    r"\b("
    r"sentient\.load_and_send|sentient\.receive_and_store|"
    r"sentient\.vector_binary|agen\.vector_load|agen\.vector_store|"
    r"dataflow\.send|dataflow\.receive"
    r")\b"
)


def _read_json_with_comments(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(text)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC")
    return next(iter(payload.items()))


def _single_dsc(root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dscs = root.get("dscs_", [])
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DSC inside the SDSC")
    return next(iter(dscs[0].items()))


def _tool_path(bin_dir: Path, tool: str) -> str:
    candidate = bin_dir / tool
    if candidate.exists():
        return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    stdout: Path,
    stderr: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    stdout.write_text(proc.stdout, encoding="utf-8")
    if stderr is not None:
        stderr.write_text(proc.stderr, encoding="utf-8")
    elif proc.stderr:
        stdout.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


def _start_data(node: dict[str, Any]) -> dict[str, str]:
    return (node.get("startAddressCoreCorelet_") or {}).get("data_", {}) or {}


def _set_compact_start_data(node: dict[str, Any], *, num_cores: int, value: int) -> None:
    start = node.setdefault(
        "startAddressCoreCorelet_",
        {
            "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
            "dim_prop_attr": [
                {"factor_": num_cores, "label_": "core"},
                {"factor_": 1, "label_": "corelet"},
                {"factor_": 1, "label_": "time"},
            ],
            "data_": {},
        },
    )
    start["data_"] = {f"[{core}, 0, 0]": str(value) for core in range(num_cores)}


def _patch_allocations(
    payload: dict[str, Any],
    *,
    alloc_patterns: list[str],
    value: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, root = _single_root(payload)
    _, dsc = _single_dsc(root)
    num_cores = int(root.get("numCoresUsed_") or dsc.get("numCoresUsed_") or 1)
    regexes = [re.compile(pattern) for pattern in alloc_patterns]
    patched_allocs: list[dict[str, Any]] = []
    patched_offsets: list[dict[str, Any]] = []
    for node in dsc.get("scheduleTree_", []):
        if not isinstance(node, dict):
            continue
        name = str(node.get("name_", ""))
        if node.get("nodeType_") == "allocate" and any(regex.search(name) for regex in regexes):
            before = list(_start_data(node).items())
            _set_compact_start_data(node, num_cores=num_cores, value=value)
            after = list(_start_data(node).items())
            patched_allocs.append(
                {
                    "name": name,
                    "component": node.get("component_"),
                    "layout": node.get("layoutDimOrder_"),
                    "before_first": before[:3],
                    "before_last": before[-3:],
                    "after_first": after[:3],
                    "after_last": after[-3:],
                }
            )
        patched_offsets.extend(
            _patch_lxlu_start_offsets(node, node_name=name, num_cores=num_cores, value=value)
        )
    return patched_allocs, patched_offsets


def _patch_lxlu_start_offsets(
    node: dict[str, Any],
    *,
    node_name: str,
    num_cores: int,
    value: int,
) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    offsets = []
    src = node.get("srcLdsAndLoopOffsets_")
    if isinstance(src, dict):
        offsets.append(("srcLdsAndLoopOffsets_", src))
    for index, dst in enumerate(node.get("dstLdsAndLoopOffsets_", []) or []):
        if isinstance(dst, dict):
            offsets.append((f"dstLdsAndLoopOffsets_[{index}]", dst))
    for field, offset in offsets:
        start = offset.get("startAddr_")
        if not isinstance(start, dict) or not isinstance(start.get("data_"), dict):
            continue
        if offset.get("dataConnect_") != "lxlu_input":
            continue
        before = list(start["data_"].items())
        start["data_"] = {f"[{core}, 0, 0]": str(value) for core in range(num_cores)}
        after = list(start["data_"].items())
        patched.append(
            {
                "node": node_name,
                "field": field,
                "data_connect": offset.get("dataConnect_"),
                "before_first": before[:3],
                "before_last": before[-3:],
                "after_first": after[:3],
                "after_last": after[-3:],
            }
        )
    return patched


def _summarize_ir(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"unit_counts": {}, "has_hbm_or_l3_units": False, "work_op_count": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    units = Counter(_UNIT_RE.findall(text))
    return {
        "unit_counts": dict(sorted(units.items())),
        "has_hbm_or_l3_units": any(
            unit == "hbm" or unit.startswith("l3") for unit in units
        ),
        "work_op_count": len(_WORK_OP_RE.findall(text)),
    }


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "  func.func @sdsc_bundle() {\n"
        f"    sdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def _summarize_senprog(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    tokens = ["LXLU", "LXSU", "SFP", "PT", "L3LU", "L3SU", "HBM"]
    return {
        "exists": path.exists(),
        "bytes": len(text),
        "token_counts": {token: text.count(token) for token in tokens},
        "contains_lx": "lx" in text.lower(),
        "contains_hbm": "hbm" in text.lower(),
    }


def _parse_boundary(*paths: Path) -> dict[str, Any] | None:
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        match = _BOUNDARY_RE.search(text)
        if match:
            return {
                "register": match.group(1).strip(),
                "value": int(match.group(2)),
                "path": str(path),
            }
    return None


def _tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _run_dcc(path: Path, *, output_dir: Path, bin_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    dcc = _tool_path(bin_dir, "dcc_standalone")
    dcc_dir = output_dir / "dcc"
    dcc_dir.mkdir(parents=True, exist_ok=True)
    rc = _run(
        [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(path)],
        cwd=dcc_dir,
        stdout=dcc_dir / "dcc_prog.out",
        stderr=dcc_dir / "dcc_prog.err",
        env=env,
    )
    return {
        "rc": rc,
        "stderr_tail": _tail(dcc_dir / "dcc_prog.err", 24),
        **_summarize_ir(dcc_dir / "dcc_prog.out"),
    }


def _run_dxp(path: Path, *, output_dir: Path, bin_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    dxp = _tool_path(bin_dir, "dxp_standalone")
    bundle_dir = output_dir / "dxp_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_sdsc = bundle_dir / "sdsc.json"
    shutil.copy2(path, bundle_sdsc)
    (bundle_dir / "bundle.mlir").write_text(_bundle_mlir("sdsc.json"), encoding="utf-8")
    rc = _run(
        [dxp, "--bundle", "-d", str(bundle_dir)],
        cwd=bundle_dir,
        stdout=bundle_dir / "dxp.log",
        env=env,
    )
    debug_dir = bundle_dir / "debug" / "sdsc"
    return {
        "rc": rc,
        "ok": rc == 0,
        "log_tail": _tail(bundle_dir / "dxp.log", 24),
        "senprog": _summarize_senprog(debug_dir / "senprog.txt"),
    }


def _failure_kind(summary: dict[str, Any]) -> str:
    dcc = summary.get("dcc", {})
    dxp = summary.get("dxp", {})
    boundary = summary.get("boundary")
    if dcc.get("rc") not in (None, 0):
        return "dcc-lrf-boundary" if boundary else "dcc-fail"
    if dxp and dxp.get("rc") != 0:
        return "dxp-lrf-boundary" if boundary else "dxp-fail"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--alloc-pattern",
        action="append",
        default=[],
        help="Regex for allocate node names to compact. Repeatable.",
    )
    parser.add_argument("--compact-value", type=int, default=0)
    parser.add_argument("--run-dxp", action="store_true")
    parser.add_argument(
        "--deeptools-bin",
        default=Path("/opt/ibm/spyre/deeptools/bin"),
        type=Path,
    )
    parser.add_argument("--senarch", default="rcudd1a")
    args = parser.parse_args()

    if not args.alloc_pattern:
        args.alloc_pattern = [r"allocate_Tensor0_lx(?:_internalInput)?$"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_json_with_comments(args.sdsc)
    patched_allocs, patched_offsets = _patch_allocations(
        payload,
        alloc_patterns=args.alloc_pattern,
        value=args.compact_value,
    )
    patched_path = args.output_dir / "patched_sdsc.json"
    _write_json(patched_path, payload)

    env = os.environ.copy()
    env["SENARCH"] = args.senarch
    summary: dict[str, Any] = {
        "input_sdsc": str(args.sdsc),
        "patched_sdsc": str(patched_path),
        "alloc_patterns": args.alloc_pattern,
        "compact_value": args.compact_value,
        "patched_allocations": patched_allocs,
        "patched_offsets": patched_offsets,
    }
    summary["dcc"] = _run_dcc(
        patched_path,
        output_dir=args.output_dir,
        bin_dir=args.deeptools_bin,
        env=env,
    )
    if args.run_dxp:
        summary["dxp"] = _run_dxp(
            patched_path,
            output_dir=args.output_dir,
            bin_dir=args.deeptools_bin,
            env=env,
        )
    summary["boundary"] = _parse_boundary(
        args.output_dir / "dcc" / "dcc_prog.err",
        args.output_dir / "dcc" / "dcc_prog.out",
        args.output_dir / "dxp_bundle" / "dxp.log",
    )
    summary["failure_kind"] = _failure_kind(summary)

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
