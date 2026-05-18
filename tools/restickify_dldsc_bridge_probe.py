#!/usr/bin/env python3
"""Probe DLDSc-shaped ReStickifyOpLx bundle variants.

This is intentionally a diagnostic tool, not production lowering.  It starts
from a generated ``ReStickifyOpHBM`` SDSC because that JSON already has the
``dscs_`` / ``coreIdToDscSchedule`` shape accepted by ``dxp_standalone``.  The
tool then writes a small matrix of ``ReStickifyOpLx`` variants and, when
requested, runs Deeptools to classify each variant:

* Does DCC lower it to non-empty ProgIR?
* Does the lowered IR still mention HBM/L3 units?
* Does DXP accept the normal bundle path?

The goal is to distinguish a real LX-local DLDSc bridge from a JSON rewrite
that only changes names.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from copy import deepcopy
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


def _read_single_root(path: Path) -> tuple[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload) != 1:
        raise ValueError(f"{path} must contain exactly one root SDSC")
    name, root = next(iter(payload.items()))
    return name, root


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _dsc_entry(root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dscs = root.get("dscs_", [])
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("seed must contain exactly one dscs_ entry")
    return next(iter(dscs[0].items()))


def _rename_to_lx(root: dict[str, Any]) -> dict[str, Any]:
    root = deepcopy(root)
    _, dsc = _dsc_entry(root)
    root["dscs_"][0] = {"ReStickifyOpLx": dsc}
    for compute_op in dsc.get("computeOp_", []):
        if compute_op.get("opFuncName") == "ReStickifyOpHBM":
            compute_op["opFuncName"] = "ReStickifyOpLx"
    return root


def _allocation_nodes(dsc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in dsc.get("scheduleTree_", [])
        if node.get("nodeType_") == "allocate"
    ]


def _mark_allocations_lx(root: dict[str, Any], *, output_only: bool) -> None:
    _, dsc = _dsc_entry(root)
    allocs = _allocation_nodes(dsc)
    output_lds_idx = max((node.get("ldsIdx_", -1) for node in allocs), default=-1)
    for node in allocs:
        if output_only and node.get("ldsIdx_") != output_lds_idx:
            continue
        node["component_"] = "lx"
        node["name_"] = str(node.get("name_", "")).replace("_hbm", "_lx")


def _mark_memorg_lx_only(root: dict[str, Any]) -> None:
    _, dsc = _dsc_entry(root)
    for labeled_ds in dsc.get("labeledDs_", []):
        labeled_ds["memOrg_"] = {"lx": {"isPresent": 1}}


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "\tfunc.func @sdsc_bundle() {\n"
        f"\t\tsdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "\t\treturn\n"
        "\t}\n"
        "}\n"
    )


def _write_variant(output_dir: Path, name: str, root: dict[str, Any]) -> Path:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    sdsc_name = "sdsc_0_ReStickifyOpLx.json"
    _write_json(variant_dir / sdsc_name, {"0_ReStickifyOpLx": root})
    (variant_dir / "bundle.mlir").write_text(_bundle_mlir(sdsc_name), encoding="utf-8")
    return variant_dir


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    stdout: Path,
    stderr: Path | None = None,
) -> int:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout.write_text(proc.stdout, encoding="utf-8")
    if stderr is not None:
        stderr.write_text(proc.stderr, encoding="utf-8")
    elif proc.stderr:
        stdout.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return proc.returncode


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


def _tool_path(bin_dir: Path, tool: str) -> str:
    if bin_dir:
        candidate = bin_dir / tool
        if candidate.exists():
            return str(candidate)
    found = shutil.which(tool)
    if not found:
        raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")
    return found


def _run_deeptools(variant_dir: Path, bin_dir: Path) -> dict[str, Any]:
    dcc = _tool_path(bin_dir, "dcc_standalone")
    dxp = _tool_path(bin_dir, "dxp_standalone")
    sdsc = variant_dir / "sdsc_0_ReStickifyOpLx.json"
    dcc_rc = _run(
        [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(sdsc)],
        cwd=variant_dir,
        stdout=variant_dir / "dcc_prog.out",
        stderr=variant_dir / "dcc_prog.err",
    )
    dxp_rc = _run(
        [dxp, "--bundle", "-d", str(variant_dir)],
        cwd=variant_dir,
        stdout=variant_dir / "dxp.log",
    )
    return {
        "dcc_rc": dcc_rc,
        "dxp_rc": dxp_rc,
        "dxp_ok": dxp_rc == 0,
        **_summarize_ir(variant_dir / "dcc_prog.out"),
    }


def _run_dcc_only(path: Path, bin_dir: Path, output_dir: Path) -> dict[str, Any]:
    dcc = _tool_path(bin_dir, "dcc_standalone")
    output_dir.mkdir(parents=True, exist_ok=True)
    rc = _run(
        [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(path)],
        cwd=output_dir,
        stdout=output_dir / "dcc_prog.out",
        stderr=output_dir / "dcc_prog.err",
    )
    return {"dcc_rc": rc, **_summarize_ir(output_dir / "dcc_prog.out")}


def _variants(seed_root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rename_only = _rename_to_lx(seed_root)

    output_lx = _rename_to_lx(seed_root)
    _mark_allocations_lx(output_lx, output_only=True)

    output_lx_memorg_lx = _rename_to_lx(seed_root)
    _mark_allocations_lx(output_lx_memorg_lx, output_only=True)
    _mark_memorg_lx_only(output_lx_memorg_lx)

    all_lx_memorg_lx = _rename_to_lx(seed_root)
    _mark_allocations_lx(all_lx_memorg_lx, output_only=False)
    _mark_memorg_lx_only(all_lx_memorg_lx)

    return {
        "dldsc_rename_only": rename_only,
        "dldsc_output_lx": output_lx,
        "dldsc_output_lx_memorg_lx": output_lx_memorg_lx,
        "dldsc_all_lx_memorg_lx": all_lx_memorg_lx,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-sdsc", required=True, type=Path)
    parser.add_argument(
        "--dataop-sdsc",
        type=Path,
        help="Optional Stage 36 data-op SDSC for DCC-only comparison.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--deeptools-bin",
        type=Path,
        default=Path("/opt/ibm/spyre/deeptools/bin"),
    )
    parser.add_argument("--run-deeptools", action="store_true")
    args = parser.parse_args()

    _, seed_root = _read_single_root(args.seed_sdsc)
    rows: list[dict[str, Any]] = []
    for name, root in _variants(seed_root).items():
        variant_dir = _write_variant(args.output_dir, name, root)
        row: dict[str, Any] = {
            "variant": name,
            "path": str(variant_dir),
        }
        if args.run_deeptools:
            row.update(_run_deeptools(variant_dir, args.deeptools_bin))
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    if args.dataop_sdsc is not None:
        row = {
            "variant": "dataop_lx_reference",
            "path": str(args.dataop_sdsc),
        }
        if args.run_deeptools:
            row.update(
                _run_dcc_only(
                    args.dataop_sdsc,
                    args.deeptools_bin,
                    args.output_dir / "dataop_lx_reference",
                )
            )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    _write_json(args.output_dir / "summary.json", {"rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

