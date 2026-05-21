#!/usr/bin/env python3
"""Probe whether an LX restickify bridge can be carried as DLDSc+PCFG.

Stage199 proved that a mixed SuperDsc containing one consumer DL DSC and two
PT/LX data ops can lower to non-HBM ProgIR through the DCC path.  DXP rejects
that shape today because bundle SDSCs are expected to contain DL DSCs, not
``datadscs_``.

This probe retries the "DLDSc-only" route from a different angle:

* generate/export PCFG from a known-good mixed SDSC;
* construct DL-only carrier SDSCs with the mixed PCFG embedded;
* run DCC/DXP to see whether the PCFG is honored or regenerated away;
* optionally run the standalone skip-PCFG path as a control.

It is intentionally diagnostic.  A successful production route would need the
normal Torch-Spyre bundle path to preserve the bridge semantics without the
mixed ``datadscs_`` import.
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
    r"sentient\.vector_binary|agen\.vector_load|agen\.vector_store|"
    r"dataflow\.send|dataflow\.receive|sentient\.matmul|sentient\.sfp"
    r")\b"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected one top-level SDSC root")
    return next(iter(payload.items()))


def _tool_path(bin_dir: Path, tool: str) -> str:
    candidate = bin_dir / tool
    if candidate.exists():
        return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False, env=env)


def _summarize_text(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "unit_counts": {},
            "work_op_count": 0,
            "has_hbm": False,
            "has_lxlu": False,
            "has_lxsu": False,
            "has_l3": False,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    units = Counter(
        (match[0] or match[1]).lower() for match in _UNIT_RE.findall(text)
    )
    upper_instrs = re.findall(r"\b(?:L3|LX|SFP|PT|PE)_[A-Z0-9_]+\b", text)
    return {
        "unit_counts": dict(sorted(units.items())),
        "work_op_count": len(_WORK_OP_RE.findall(text)) + len(upper_instrs),
        "has_hbm": "hbm" in text.lower(),
        "has_lxlu": any(unit.startswith("lxlu") for unit in units),
        "has_lxsu": any(unit.startswith("lxsu") for unit in units),
        "has_l3": any(unit.startswith("l3") for unit in units),
    }


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "\tfunc.func @sdsc_bundle() {\n"
        f"\t\tsdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "\t\treturn\n"
        "\t}\n"
        "}\n"
    )


def _expand_pcfg(pcfg_payload: dict[str, Any]) -> dict[str, Any]:
    _, pcfg_root = _single_root(pcfg_payload)
    if "pcfg_" in pcfg_root:
        return deepcopy(pcfg_root["pcfg_"])

    pcfg_map = pcfg_root.get("pcfgMap_", {}) or {}
    pcfg_pool = pcfg_root.get("pcfgPool_", {}) or {}
    if not pcfg_map or not pcfg_pool:
        raise ValueError("PCFG payload must contain pcfg_ or pcfgMap_/pcfgPool_")

    expanded: dict[str, dict[str, Any]] = {}
    for core_id, comp_map in pcfg_map.items():
        expanded[str(core_id)] = {}
        for comp, pool_id in comp_map.items():
            expanded[str(core_id)][str(comp)] = deepcopy(pcfg_pool[str(pool_id)])
    return expanded


def _dl_only_schedule(root: dict[str, Any]) -> dict[str, list[list[int]]]:
    num_cores = int(root.get("numCoresUsed_", 0) or 0)
    if num_cores <= 0:
        num_cores = len(root.get("coreIdToDsc_", {}) or {})
    return {str(core): [[-1, 0, 1, 0]] for core in range(num_cores)}


def _strip_dataops(root: dict[str, Any], *, schedule_mode: str) -> dict[str, Any]:
    out = deepcopy(root)
    out["datadscs_"] = []
    if schedule_mode == "clear":
        out["coreIdToDscSchedule"] = {}
    elif schedule_mode == "dl_only":
        out["coreIdToDscSchedule"] = _dl_only_schedule(out)
    elif schedule_mode != "keep":
        raise ValueError(f"unknown schedule mode {schedule_mode!r}")
    op_funcs = out.get("opFuncsUsed_", []) or []
    dataop_names = {"ReStickifyOpWithPTLx", "STCDPOpLx"}
    out["opFuncsUsed_"] = [name for name in op_funcs if name not in dataop_names]
    return out


def _make_carrier_variants(
    mixed_sdsc: dict[str, Any],
    exported_sdsc: dict[str, Any] | None,
    pcfg_payload: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    mixed_name, mixed_root = _single_root(mixed_sdsc)
    variants: dict[str, dict[str, Any]] = {}

    variants["dldsc_only_no_pcfg_keep_schedule"] = {
        mixed_name: _strip_dataops(mixed_root, schedule_mode="keep")
    }
    variants["dldsc_only_no_pcfg_clear_schedule"] = {
        mixed_name: _strip_dataops(mixed_root, schedule_mode="clear")
    }
    variants["dldsc_only_no_pcfg_dl_schedule"] = {
        mixed_name: _strip_dataops(mixed_root, schedule_mode="dl_only")
    }

    if pcfg_payload is not None:
        _, pcfg_root = _single_root(pcfg_payload)
        expanded_pcfg = _expand_pcfg(pcfg_payload)

        full = _strip_dataops(mixed_root, schedule_mode="dl_only")
        full["pcfg_"] = expanded_pcfg
        full["target_"] = "SENPCFG"
        variants["dldsc_only_full_pcfg_dl_schedule"] = {mixed_name: full}

        compressed = _strip_dataops(mixed_root, schedule_mode="dl_only")
        if "pcfgMap_" in pcfg_root:
            compressed["pcfgMap_"] = deepcopy(pcfg_root["pcfgMap_"])
        if "pcfgPool_" in pcfg_root:
            compressed["pcfgPool_"] = deepcopy(pcfg_root["pcfgPool_"])
        compressed["target_"] = "SENPCFG"
        variants["dldsc_only_compressed_pcfg_dl_schedule"] = {
            mixed_name: compressed
        }

    if exported_sdsc is not None:
        exported_name, exported_root = _single_root(exported_sdsc)
        variants["dcg_exported_strip_dataops"] = {
            exported_name: _strip_dataops(exported_root, schedule_mode="dl_only")
        }
        if pcfg_payload is not None:
            exported_full = _strip_dataops(exported_root, schedule_mode="dl_only")
            exported_full["pcfg_"] = _expand_pcfg(pcfg_payload)
            exported_full["target_"] = "SENPCFG"
            variants["dcg_exported_full_pcfg"] = {exported_name: exported_full}

    return variants


def _run_dcg_export(
    mixed_sdsc: Path, output_dir: Path, bin_dir: Path, *, emit_senprog: bool
) -> dict[str, Any]:
    dcg = _tool_path(bin_dir, "dcg_standalone")
    dcg_out = output_dir / "dcg_out"
    dcg_out.mkdir(parents=True, exist_ok=True)
    cmd = [dcg, "-d", str(dcg_out), "-initSdsc", str(mixed_sdsc)]
    if emit_senprog:
        cmd.append("-s")
    proc = _run(cmd, cwd=output_dir)
    (output_dir / "dcg_export.out").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "dcg_export.err").write_text(proc.stderr, encoding="utf-8")
    return {
        "dcg_export_rc": proc.returncode,
        "dcg_export_stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
        "dcg_export_stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
        "exported_sdsc": str(dcg_out / "sdsc.json"),
        "exported_pcfg": str(dcg_out / "pcfg.json"),
    }


def _run_dcc_dxp(variant_dir: Path, sdsc_path: Path, bin_dir: Path) -> dict[str, Any]:
    dcc = _tool_path(bin_dir, "dcc_standalone")
    dxp = _tool_path(bin_dir, "dxp_standalone")
    dcc_proc = _run(
        [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(sdsc_path)],
        cwd=variant_dir,
    )
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


def _run_skip_pcfg_control(
    mixed_sdsc: Path,
    pcfg: Path,
    output_dir: Path,
    bin_dir: Path,
) -> dict[str, Any]:
    dcg = _tool_path(bin_dir, "dcg_standalone")
    control_dir = output_dir / "skip_pcfg_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    proc = _run(
        [
            dcg,
            "-d",
            str(control_dir / "out"),
            "-initSdsc",
            str(mixed_sdsc),
            "-initPcfg",
            str(pcfg),
            "-skip_pcfggen",
            "-s",
        ],
        cwd=control_dir,
    )
    (control_dir / "dcg_skip.out").write_text(proc.stdout, encoding="utf-8")
    (control_dir / "dcg_skip.err").write_text(proc.stderr, encoding="utf-8")
    return {
        "variant": "dcg_skip_pcfg_control",
        "path": str(control_dir),
        "dcg_skip_rc": proc.returncode,
        "dcg_skip_stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
        "dcg_skip_stderr_tail": "\n".join(proc.stderr.splitlines()[-8:]),
        **_summarize_text(control_dir / "dataDSC" / "senprog.txt_ir"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-sdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--deeptools-bin",
        type=Path,
        default=Path("/opt/ibm/spyre/deeptools/bin"),
    )
    parser.add_argument("--run-deeptools", action="store_true")
    parser.add_argument(
        "--skip-dcg-export",
        action="store_true",
        help="Use only --mixed-sdsc and any existing --pcfg/--exported-sdsc inputs.",
    )
    parser.add_argument("--pcfg", type=Path)
    parser.add_argument("--exported-sdsc", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mixed_payload = _read_json(args.mixed_sdsc)

    setup: dict[str, Any] = {}
    pcfg_path = args.pcfg
    exported_sdsc_path = args.exported_sdsc
    if args.run_deeptools and not args.skip_dcg_export:
        setup = _run_dcg_export(
            args.mixed_sdsc, args.output_dir, args.deeptools_bin, emit_senprog=False
        )
        if setup["dcg_export_rc"] == 0:
            pcfg_path = Path(setup["exported_pcfg"])
            exported_sdsc_path = Path(setup["exported_sdsc"])

    pcfg_payload = _read_json(pcfg_path) if pcfg_path and pcfg_path.exists() else None
    exported_payload = (
        _read_json(exported_sdsc_path)
        if exported_sdsc_path and exported_sdsc_path.exists()
        else None
    )

    rows: list[dict[str, Any]] = []
    for name, payload in _make_carrier_variants(
        mixed_payload, exported_payload, pcfg_payload
    ).items():
        variant_dir = args.output_dir / name
        sdsc_name = f"sdsc_0_{name}.json"
        sdsc_path = variant_dir / sdsc_name
        _write_json(sdsc_path, payload)
        (variant_dir / "bundle.mlir").write_text(_bundle_mlir(sdsc_name), encoding="utf-8")
        row: dict[str, Any] = {
            "variant": name,
            "path": str(variant_dir),
            "sdsc": str(sdsc_path),
        }
        if args.run_deeptools:
            row.update(_run_dcc_dxp(variant_dir, sdsc_path, args.deeptools_bin))
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    if args.run_deeptools and pcfg_path and pcfg_path.exists():
        row = _run_skip_pcfg_control(args.mixed_sdsc, pcfg_path, args.output_dir, args.deeptools_bin)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    summary = {
        "mixed_sdsc": str(args.mixed_sdsc),
        "setup": setup,
        "pcfg": str(pcfg_path) if pcfg_path else None,
        "exported_sdsc": str(exported_sdsc_path) if exported_sdsc_path else None,
        "rows": rows,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
