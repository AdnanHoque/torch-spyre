#!/usr/bin/env python3
"""Probe DXP's pre-DDC corelet-split blocker for restickify SDSCs.

The Deeptools restickify DDL fixture can be expanded by ``ddc_standalone``, and
the expanded SDSC lowers through DCC to real LX/SFP/PT work. DXP, however, runs
``Dsm::doCoreletSplitSdsc`` before its DDC pass. That splitter currently
assumes exactly one ``dataStageParam_`` entry, while the restickify fixture uses
two.

This tool generates a tiny variant matrix to locate the boundary:

* raw: original fixture
* one_ds_repoint_loops: collapse to one data stage and repoint loop ids
* output_scale_out_zero: force the splitter's psum early-return heuristic
* fake_restickify_reduction_after: append a fake reduction-like restickify op

The variants are intentionally diagnostic. They are not candidate production
lowerings.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any


_FAIL_RE = re.compile(r"what\(\):\s+DtException:\s+(.*)")


def _strip_json_comments(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("//")]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def _single_dsc(payload: dict[str, Any]) -> dict[str, Any]:
    root = next(iter(payload.values()))
    return next(iter(root["dscs_"][0].values()))


def _walk(obj: Any, fn) -> None:
    if isinstance(obj, dict):
        fn(obj)
        for value in obj.values():
            _walk(value, fn)
    elif isinstance(obj, list):
        for value in obj:
            _walk(value, fn)


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "  func.func @sdsc_bundle() {\n"
        f"    sdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def _write_variant(output_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "sdsc.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (variant_dir / "bundle.mlir").write_text(_bundle_mlir("sdsc.json"), encoding="utf-8")
    return variant_dir


def _variants(base: dict[str, Any]) -> dict[str, dict[str, Any]]:
    variants: dict[str, dict[str, Any]] = {"raw": deepcopy(base)}

    one_ds = deepcopy(base)
    dsc = _single_dsc(one_ds)
    dsc["dataStageParam_"] = {"0": dsc["dataStageParam_"]["0"]}

    def repoint_loop_ids(node: dict[str, Any]) -> None:
        if node.get("numId_") == 1:
            node["numId_"] = 0
        if node.get("denId_") == 1:
            node["denId_"] = 0

    _walk(one_ds, repoint_loop_ids)
    variants["one_ds_repoint_loops"] = one_ds

    output_scale = deepcopy(base)
    dsc = _single_dsc(output_scale)
    # OUTPUT layout is [out, y, mb, j, x] in the Deeptools fixture. Marking
    # out broadcast-like forces the psum early return in the corelet splitter,
    # but it also breaks DDL matching. That is useful boundary evidence.
    dsc["labeledDs_"][1]["scale_"][0] = 0
    variants["output_scale_out_zero"] = output_scale

    fake_reduction = deepcopy(base)
    dsc = _single_dsc(fake_reduction)
    fake_compute = deepcopy(dsc["computeOp_"][0])
    fake_compute["outputLabeledDs"] = []
    dsc["computeOp_"].append(fake_compute)
    variants["fake_restickify_reduction_after"] = fake_reduction

    return variants


def _tail(path: Path, lines: int = 16) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _failure(log_text: str) -> str:
    match = _FAIL_RE.search(log_text)
    if match:
        return match.group(1).strip()
    if "DDL found but not suitable" in log_text:
        return "DDL found but not suitable"
    if not log_text.strip():
        return ""
    return log_text.strip().splitlines()[-1]


def _run_variant(
    variant_dir: Path,
    *,
    bin_dir: Path,
    senarch: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["SENARCH"] = senarch
    ddc = _tool_path(bin_dir, "ddc_standalone")
    dxp = _tool_path(bin_dir, "dxp_standalone")
    ddc_rc = _run(
        [ddc, "-s", "sdsc.json"],
        cwd=variant_dir,
        stdout=variant_dir / "ddc.out",
        stderr=variant_dir / "ddc.err",
        env=env,
    )
    dxp_rc = _run(
        [dxp, "--bundle", "-d", str(variant_dir)],
        cwd=variant_dir,
        stdout=variant_dir / "dxp.log",
        env=env,
    )
    ddc_log = _tail(variant_dir / "ddc.out") + "\n" + _tail(variant_dir / "ddc.err")
    dxp_log = _tail(variant_dir / "dxp.log")
    ddc_failure = "" if ddc_rc == 0 else _failure(ddc_log)
    dxp_failure = "" if dxp_rc == 0 else _failure(dxp_log)
    return {
        "ddc_rc": ddc_rc,
        "ddc_ok": ddc_rc == 0,
        "ddc_failure": ddc_failure,
        "ddc_log_tail": ddc_log.strip(),
        "dxp_rc": dxp_rc,
        "dxp_ok": dxp_rc == 0,
        "dxp_failure": dxp_failure,
        "dxp_log_tail": dxp_log.strip(),
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
    normalized = args.output_dir / "input.json"
    _strip_json_comments(args.sdsc, normalized)
    base = json.loads(normalized.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    for name, payload in _variants(base).items():
        variant_dir = _write_variant(args.output_dir, name, payload)
        row: dict[str, Any] = {
            "variant": name,
            "path": str(variant_dir),
        }
        if args.run_deeptools:
            row.update(
                _run_variant(
                    variant_dir,
                    bin_dir=args.deeptools_bin,
                    senarch=args.senarch,
                )
            )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    summary = {
        "input_sdsc": str(args.sdsc),
        "normalized_input": str(normalized),
        "senarch": args.senarch,
        "rows": rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
