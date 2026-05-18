#!/usr/bin/env python3
"""Probe the DDC/DDL contract for restickify SDSCs.

Stage 38 showed that directly rewriting a post-lowered ``ReStickifyOpHBM``
DLDSc JSON into ``ReStickifyOpLx`` is not enough: DXP may accept the bundle,
but DCC lowers it to empty L3 units. This tool checks the next layer down. It
feeds an SDSC to ``ddc_standalone`` so Deeptools can apply the
``restickify_sen1p5.ddl`` template, then compares the original and DDC-produced
SDSCs through DCC/DXP.

The probe is deliberately diagnostic. It does not attempt to execute the
kernel, and a DXP failure is reported as data instead of treated as a Python
failure.
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


_UNIT_RE = re.compile(r'name = "([^"]+)"')
_WORK_OP_RE = re.compile(
    r"\b("
    r"sentient\.load_and_send|sentient\.receive_and_store|"
    r"sentient\.vector_binary|agen\.vector_load|agen\.vector_store|"
    r"dataflow\.send|dataflow\.receive"
    r")\b"
)


def _tool_path(bin_dir: Path, tool: str) -> str:
    candidate = bin_dir / tool
    if candidate.exists():
        return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")


def _strip_json_comments(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("//")]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _load_single_root(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise ValueError(f"{path} must contain exactly one top-level SDSC")
    return next(iter(payload.items()))


def _dsc(root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dscs = root.get("dscs_", [])
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("SDSC root must contain exactly one dscs_ entry")
    return next(iter(dscs[0].items()))


def _summarize_sdsc(path: Path) -> dict[str, Any]:
    name, root = _load_single_root(path)
    dsc_name, dsc = _dsc(root)
    schedule = dsc.get("scheduleTree_", [])
    alloc_components = Counter(
        str(node.get("component_"))
        for node in schedule
        if isinstance(node, dict) and node.get("nodeType_") == "allocate"
    )
    node_types = Counter(
        str(node.get("nodeType_"))
        for node in schedule
        if isinstance(node, dict) and node.get("nodeType_") is not None
    )
    transfer_names = [
        str(node.get("name_"))
        for node in schedule
        if isinstance(node, dict) and node.get("nodeType_") == "transfer"
    ]
    data_connects: list[str] = []
    for node in schedule:
        if not isinstance(node, dict):
            continue
        for dst in node.get("dstLdsAndLoopOffsets_", []) or []:
            value = dst.get("dataConnect_")
            if value:
                data_connects.append(str(value))
        src = node.get("srcLdsAndLoopOffsets_") or {}
        value = src.get("dataConnect_")
        if value:
            data_connects.append(str(value))
    op_funcs = [
        str(op.get("opFuncName"))
        for op in dsc.get("computeOp_", [])
        if isinstance(op, dict) and op.get("opFuncName")
    ]
    return {
        "sdsc_name": name,
        "dsc_name": dsc_name,
        "op_funcs": op_funcs,
        "num_cores_used": root.get("numCoresUsed_"),
        "schedule_node_count": len(schedule),
        "schedule_node_types": dict(sorted(node_types.items())),
        "allocate_components": dict(sorted(alloc_components.items())),
        "labeled_ds_count": len(dsc.get("labeledDs_", [])),
        "primary_ds_count": len(dsc.get("primaryDsInfo_", {})),
        "data_stage_param_count": len(dsc.get("dataStageParam_", {})),
        "transfer_count": len(transfer_names),
        "transfer_name_sample": transfer_names[:8],
        "data_connects": sorted(set(data_connects)),
    }


def _summarize_ir(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "unit_counts": {},
            "has_hbm_or_l3_units": False,
            "work_op_count": 0,
        }
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


def _run_dcc(path: Path, *, output_dir: Path, bin_dir: Path) -> dict[str, Any]:
    dcc = _tool_path(bin_dir, "dcc_standalone")
    output_dir.mkdir(parents=True, exist_ok=True)
    rc = _run(
        [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(path)],
        cwd=output_dir,
        stdout=output_dir / "dcc_prog.out",
        stderr=output_dir / "dcc_prog.err",
    )
    return {
        "dcc_rc": rc,
        **_summarize_ir(output_dir / "dcc_prog.out"),
    }


def _run_dxp(path: Path, *, output_dir: Path, bin_dir: Path) -> dict[str, Any]:
    dxp = _tool_path(bin_dir, "dxp_standalone")
    bundle_dir = output_dir / "dxp_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    sdsc_path = bundle_dir / path.name
    shutil.copy2(path, sdsc_path)
    (bundle_dir / "bundle.mlir").write_text(_bundle_mlir(path.name), encoding="utf-8")
    rc = _run(
        [dxp, "--bundle", "-d", str(bundle_dir)],
        cwd=bundle_dir,
        stdout=bundle_dir / "dxp.log",
    )
    log = (bundle_dir / "dxp.log").read_text(encoding="utf-8", errors="replace")
    return {
        "dxp_rc": rc,
        "dxp_ok": rc == 0,
        "dxp_log_tail": "\n".join(log.splitlines()[-12:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--deeptools-bin",
        type=Path,
        default=Path("/opt/ibm/spyre/deeptools/bin"),
    )
    parser.add_argument("--senarch", default="rcudd1a")
    parser.add_argument("--run-deeptools", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_sdsc = args.output_dir / "input.json"
    _strip_json_comments(args.sdsc, input_sdsc)

    summary: dict[str, Any] = {
        "input_sdsc": str(args.sdsc),
        "normalized_input": str(input_sdsc),
        "senarch": args.senarch,
        "input_summary": _summarize_sdsc(input_sdsc),
    }

    if args.run_deeptools:
        env = os.environ.copy()
        env["SENARCH"] = args.senarch
        ddc = _tool_path(args.deeptools_bin, "ddc_standalone")
        ddc_rc = _run(
            [ddc, "-s", str(input_sdsc), "-d"],
            cwd=args.output_dir,
            stdout=args.output_dir / "ddc.out",
            stderr=args.output_dir / "ddc.err",
            env=env,
        )
        ddc_out = input_sdsc.with_suffix(".out.json")
        summary["ddc"] = {
            "ddc_rc": ddc_rc,
            "output_sdsc": str(ddc_out),
            "output_exists": ddc_out.exists(),
        }
        summary["dcc_input"] = _run_dcc(
            input_sdsc,
            output_dir=args.output_dir / "dcc_input",
            bin_dir=args.deeptools_bin,
        )
        if ddc_out.exists():
            summary["ddc_output_summary"] = _summarize_sdsc(ddc_out)
            summary["dcc_ddc_output"] = _run_dcc(
                ddc_out,
                output_dir=args.output_dir / "dcc_ddc_output",
                bin_dir=args.deeptools_bin,
            )
            summary["dxp_ddc_output"] = _run_dxp(
                ddc_out,
                output_dir=args.output_dir,
                bin_dir=args.deeptools_bin,
            )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
