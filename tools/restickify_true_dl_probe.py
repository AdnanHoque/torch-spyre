#!/usr/bin/env python3
"""Probe true-DL restickify carrier candidates.

This is a diagnostic tool.  It starts from a normal Torch-Spyre
``ReStickifyOpHBM`` SDSC because that artifact already has the DLDSc bundle
shape accepted by ``dxp_standalone``.  It then rewrites only the top-level DL
op function and optional memory markings, and asks Deeptools whether the normal
DCC/DXP path emits real non-HBM restickify work.

The question is intentionally narrow:

* Does any existing top-level DL opfunc behave like a true LX-local
  restickify bridge?
* Or are the useful LX/PT/SFP restickify schedules only generated through the
  data-op/PCFG path?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


_UNIT_RE = re.compile(r'name = "([^"]+)"|Program for unit ([A-Za-z0-9_]+):')
_WORK_OP_RE = re.compile(
    r"\b("
    r"sentient\.load_and_send|sentient\.receive_and_store|"
    r"sentient\.vector_binary|sentient\.matmul|sentient\.sfp|"
    r"agen\.vector_load|agen\.vector_store|"
    r"dataflow\.send|dataflow\.receive"
    r")\b"
)
_DEFAULT_OPFUNCS = [
    "ReStickifyOpHBM",
    "ReStickifyOpLx",
    "ReStickifyOpWithPTLx",
    "interslicetranspose_fp16",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC")
    return next(iter(payload.items()))


def _single_dsc(root: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dscs = root.get("dscs_", [])
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DL DSC in seed")
    return next(iter(dscs[0].items()))


def _tool_path(bin_dir: Path, tool: str) -> str:
    candidate = bin_dir / tool
    if candidate.exists():
        return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "\tfunc.func @sdsc_bundle() {\n"
        f"\t\tsdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "\t\treturn\n"
        "\t}\n"
        "}\n"
    )


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False, env=env)


def _allocation_nodes(dsc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in dsc.get("scheduleTree_", [])
        if isinstance(node, dict) and node.get("nodeType_") == "allocate"
    ]


def _set_labeled_ds_lx_only(dsc: dict[str, Any], *, output_only: bool) -> None:
    output_idx = _output_lds_idx(dsc)
    for lds in dsc.get("labeledDs_", []):
        if output_only and int(lds.get("ldsIdx_", -1)) != output_idx:
            continue
        lds["memOrg_"] = {"lx": {"isPresent": 1}}
        lds["hbmStartAddress_"] = -1
        if "hbmSize_" in lds:
            lds["hbmSize_"] = 0


def _set_allocations_lx(dsc: dict[str, Any], *, output_only: bool) -> None:
    output_idx = _output_lds_idx(dsc)
    for node in _allocation_nodes(dsc):
        if output_only and int(node.get("ldsIdx_", -1)) != output_idx:
            continue
        node["component_"] = "lx"
        node["name_"] = str(node.get("name_", "")).replace("_hbm", "_lx")


def _output_lds_idx(dsc: dict[str, Any]) -> int:
    compute_ops = dsc.get("computeOp_", [])
    if compute_ops:
        outs = compute_ops[0].get("outputLabeledDs", [])
        if outs:
            match = re.search(r"idx(\d+)$", str(outs[-1]))
            if match:
                return int(match.group(1))
    alloc_idxs = [int(node.get("ldsIdx_", -1)) for node in _allocation_nodes(dsc)]
    return max(alloc_idxs) if alloc_idxs else -1


def _rewrite_candidate(
    seed_payload: dict[str, Any],
    *,
    opfunc: str,
    memory_mode: str,
    set_opfuncs_used: bool,
) -> dict[str, Any]:
    seed_name, seed_root = _single_root(seed_payload)
    root = deepcopy(seed_root)
    _, dsc = _single_dsc(root)

    dsc_name = opfunc
    if memory_mode != "original":
        dsc_name = f"{opfunc}_{memory_mode}"
    root["dscs_"][0] = {dsc_name: dsc}
    for compute_op in dsc.get("computeOp_", []):
        compute_op["opFuncName"] = opfunc
    if set_opfuncs_used:
        root["opFuncsUsed_"] = [opfunc]

    if memory_mode == "output_lx":
        _set_allocations_lx(dsc, output_only=True)
        _set_labeled_ds_lx_only(dsc, output_only=True)
    elif memory_mode == "all_lx":
        _set_allocations_lx(dsc, output_only=False)
        _set_labeled_ds_lx_only(dsc, output_only=False)
    elif memory_mode != "original":
        raise ValueError(f"unknown memory mode {memory_mode!r}")

    return {seed_name.replace("ReStickifyOpHBM", dsc_name): root}


def _summarize_text(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "unit_counts": {},
            "work_op_count": 0,
            "has_hbm": False,
            "has_l3": False,
            "has_lxlu": False,
            "has_lxsu": False,
            "has_pt_or_sfp": False,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    units = Counter((match[0] or match[1]).lower() for match in _UNIT_RE.findall(text))
    return {
        "unit_counts": dict(sorted(units.items())),
        "work_op_count": len(_WORK_OP_RE.findall(text)),
        "has_hbm": "hbm" in text.lower(),
        "has_l3": any(unit.startswith("l3") for unit in units),
        "has_lxlu": any(unit.startswith("lxlu") for unit in units),
        "has_lxsu": any(unit.startswith("lxsu") for unit in units),
        "has_pt_or_sfp": any(unit.startswith("pt") or unit.startswith("sfp") for unit in units),
    }


def _run_deeptools(variant_dir: Path, sdsc_path: Path, bin_dir: Path) -> dict[str, Any]:
    dcc = _tool_path(bin_dir, "dcc_standalone")
    dxp = _tool_path(bin_dir, "dxp_standalone")

    dcc_proc = _run([dcc, "--input-mode=sdsc", "--kEmitProgIR", str(sdsc_path)], cwd=variant_dir)
    (variant_dir / "dcc.out").write_text(dcc_proc.stdout, encoding="utf-8")
    (variant_dir / "dcc.err").write_text(dcc_proc.stderr, encoding="utf-8")

    dxp_proc = _run([dxp, "--bundle", "-d", str(variant_dir)], cwd=variant_dir)
    (variant_dir / "dxp.out").write_text(dxp_proc.stdout, encoding="utf-8")
    (variant_dir / "dxp.err").write_text(dxp_proc.stderr, encoding="utf-8")

    return {
        "dcc_rc": dcc_proc.returncode,
        "dxp_rc": dxp_proc.returncode,
        "dcc_error_tail": "\n".join(dcc_proc.stderr.splitlines()[-8:]),
        "dxp_error_tail": "\n".join(dxp_proc.stderr.splitlines()[-8:]),
        **_summarize_text(variant_dir / "dcc.out"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-sdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deeptools-bin", type=Path, default=Path("/opt/ibm/spyre/deeptools/bin"))
    parser.add_argument("--run-deeptools", action="store_true")
    parser.add_argument("--opfunc", action="append", choices=_DEFAULT_OPFUNCS)
    parser.add_argument(
        "--memory-mode",
        action="append",
        choices=["original", "output_lx", "all_lx"],
        help="May be repeated. Defaults to original, output_lx, all_lx.",
    )
    parser.add_argument("--set-opfuncs-used", action="store_true")
    args = parser.parse_args()

    opfuncs = args.opfunc or _DEFAULT_OPFUNCS
    memory_modes = args.memory_mode or ["original", "output_lx", "all_lx"]
    seed_payload = _read_json(args.seed_sdsc)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for opfunc in opfuncs:
        for memory_mode in memory_modes:
            variant = f"{opfunc}_{memory_mode}"
            variant_dir = args.output_dir / variant
            sdsc_name = f"sdsc_0_{variant}.json"
            sdsc_path = variant_dir / sdsc_name
            payload = _rewrite_candidate(
                seed_payload,
                opfunc=opfunc,
                memory_mode=memory_mode,
                set_opfuncs_used=args.set_opfuncs_used,
            )
            _write_json(sdsc_path, payload)
            (variant_dir / "bundle.mlir").write_text(_bundle_mlir(sdsc_name), encoding="utf-8")
            row: dict[str, Any] = {
                "variant": variant,
                "opfunc": opfunc,
                "memory_mode": memory_mode,
                "path": str(variant_dir),
                "sdsc": str(sdsc_path),
            }
            if args.run_deeptools:
                row.update(_run_deeptools(variant_dir, sdsc_path, args.deeptools_bin))
            rows.append(row)
            print(json.dumps(row, sort_keys=True))

    summary = {
        "seed_sdsc": str(args.seed_sdsc),
        "rows": rows,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
