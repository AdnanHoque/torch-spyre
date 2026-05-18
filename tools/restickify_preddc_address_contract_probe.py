#!/usr/bin/env python3
"""Probe pre-DDC address contracts for mirrored restickify DDL.

Stage 47 showed that post-DDC DCC verification passes when both the input
allocation start maps and the generated ``lxlu_input`` transfer source start map
are compact/local. This tool asks the next question: can a pre-DDC SDSC express
that contract so DDC emits the compact ``lxlu_input`` map naturally?

The variants here intentionally mutate only the compact diagnostic SDSC shape;
they are not production lowering strategies.
"""

from __future__ import annotations

import argparse
import csv
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
_WORK_OP_RE = re.compile(
    r"\b("
    r"sentient\.load_and_send|sentient\.receive_and_store|"
    r"sentient\.vector_binary|agen\.vector_load|agen\.vector_store|"
    r"dataflow\.send|dataflow\.receive"
    r")\b"
)
_UNIT_RE = re.compile(r'name = "([^"]+)"')


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


def _start_map(num_cores: int, *, base: int = 0, stride: int = 0) -> dict[str, str]:
    return {f"[{core}, 0, 0]": str(base + core * stride) for core in range(num_cores)}


def _set_start_addr_map(offset: dict[str, Any], num_cores: int, *, base: int = 0, stride: int = 0) -> None:
    offset["startAddr_"] = {
        "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
        "dim_prop_attr": [
            {"factor_": num_cores, "label_": "core"},
            {"factor_": 1, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": _start_map(num_cores, base=base, stride=stride),
    }


def _set_allocate_start(node: dict[str, Any], num_cores: int, *, base: int = 0, stride: int = 0) -> None:
    node["startAddressCoreCorelet_"] = {
        "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
        "dim_prop_attr": [
            {"factor_": num_cores, "label_": "core"},
            {"factor_": 1, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": _start_map(num_cores, base=base, stride=stride),
    }


def _mutate(payload: dict[str, Any], variant: str) -> list[str]:
    _, root = _single_root(payload)
    _, dsc = _single_dsc(root)
    num_cores = int(root.get("numCoresUsed_") or dsc.get("numCoresUsed_") or 1)
    mutations: list[str] = []
    input_alloc = None
    input_transfer = None
    for node in dsc.get("scheduleTree_", []):
        if node.get("nodeType_") == "allocate" and node.get("name_") == "allocate_Tensor0_lx":
            input_alloc = node
        if (
            node.get("nodeType_") == "transfer"
            and node.get("name_") == "transfer_lds0_src:no_component_dst:lx_lx_local"
        ):
            input_transfer = node
    if input_alloc is None or input_transfer is None:
        raise ValueError("expected allocate_Tensor0_lx and input transfer")

    if variant == "baseline":
        return mutations
    if variant == "compact-input-allocation":
        _set_allocate_start(input_alloc, num_cores, base=0, stride=0)
        mutations.append("allocate_Tensor0_lx.startAddressCoreCorelet_=compact")
        return mutations
    if variant == "input-transfer-dst-compact":
        dst = input_transfer["dstLdsAndLoopOffsets_"][0]
        _set_start_addr_map(dst, num_cores, base=0, stride=0)
        mutations.append("input_transfer.dst.startAddr_=compact")
        return mutations
    if variant == "input-transfer-dst-compact-lxlu-connect":
        dst = input_transfer["dstLdsAndLoopOffsets_"][0]
        _set_start_addr_map(dst, num_cores, base=0, stride=0)
        dst["dataConnect_"] = "lxlu_input"
        mutations.append("input_transfer.dst.startAddr_=compact")
        mutations.append("input_transfer.dst.dataConnect_=lxlu_input")
        return mutations
    if variant == "input-transfer-dst-connect-only":
        input_transfer["dstLdsAndLoopOffsets_"][0]["dataConnect_"] = "lxlu_input"
        mutations.append("input_transfer.dst.dataConnect_=lxlu_input")
        return mutations
    if variant == "compact-alloc-and-transfer-hint":
        _set_allocate_start(input_alloc, num_cores, base=0, stride=0)
        dst = input_transfer["dstLdsAndLoopOffsets_"][0]
        _set_start_addr_map(dst, num_cores, base=0, stride=0)
        dst["dataConnect_"] = "lxlu_input"
        mutations.append("allocate_Tensor0_lx.startAddressCoreCorelet_=compact")
        mutations.append("input_transfer.dst.startAddr_=compact")
        mutations.append("input_transfer.dst.dataConnect_=lxlu_input")
        return mutations
    raise ValueError(f"unknown variant {variant}")


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


def _extract_post_ddc_address_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = _read_json_with_comments(path)
    _, root = _single_root(payload)
    _, dsc = _single_dsc(root)
    out: dict[str, Any] = {}
    for index, node in enumerate(dsc.get("scheduleTree_", [])):
        if not isinstance(node, dict):
            continue
        name = str(node.get("name_", ""))
        if name in {
            "allocate_Tensor0_lx_internalInput",
            "allocate_Tensor0_lx",
            "transfer_lds0_src:lxlu_dst:sfp",
        }:
            out[f"{index}:{name}"] = _extract_address_fields(node)
    return out


def _extract_address_fields(node: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    start = ((node.get("startAddressCoreCorelet_") or {}).get("data_") or {})
    if start:
        fields["startAddressCoreCorelet_first"] = list(start.items())[:3]
        fields["startAddressCoreCorelet_last"] = list(start.items())[-3:]
    src = node.get("srcLdsAndLoopOffsets_")
    if isinstance(src, dict) and isinstance(src.get("startAddr_"), dict):
        data = src["startAddr_"].get("data_") or {}
        fields["src_start_data_connect"] = src.get("dataConnect_")
        fields["src_start_first"] = list(data.items())[:3]
        fields["src_start_last"] = list(data.items())[-3:]
    for idx, dst in enumerate(node.get("dstLdsAndLoopOffsets_", []) or []):
        if isinstance(dst, dict) and isinstance(dst.get("startAddr_"), dict):
            data = dst["startAddr_"].get("data_") or {}
            fields[f"dst{idx}_start_data_connect"] = dst.get("dataConnect_")
            fields[f"dst{idx}_start_first"] = list(data.items())[:3]
            fields[f"dst{idx}_start_last"] = list(data.items())[-3:]
    return fields


def _run_variant(
    *,
    source: Path,
    variant: str,
    output_dir: Path,
    bin_dir: Path,
    senarch: str,
) -> dict[str, Any]:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_json_with_comments(source)
    mutations = _mutate(payload, variant)
    pre_path = variant_dir / "pre_ddc.json"
    _write_json(pre_path, payload)

    env = os.environ.copy()
    env["SENARCH"] = senarch
    ddc = _tool_path(bin_dir, "ddc_standalone")
    ddc_rc = _run(
        [ddc, "-s", str(pre_path), "-d"],
        cwd=variant_dir,
        stdout=variant_dir / "ddc.out",
        stderr=variant_dir / "ddc.err",
        env=env,
    )
    post_path = pre_path.with_suffix(".out.json")
    row: dict[str, Any] = {
        "variant": variant,
        "mutations": mutations,
        "pre_ddc": str(pre_path),
        "post_ddc": str(post_path),
        "ddc": {
            "rc": ddc_rc,
            "output_exists": post_path.exists(),
            "stderr_tail": _tail(variant_dir / "ddc.err", 20),
        },
        "post_ddc_addresses": _extract_post_ddc_address_summary(post_path),
    }
    if post_path.exists():
        dcc = _tool_path(bin_dir, "dcc_standalone")
        dcc_dir = variant_dir / "dcc"
        dcc_dir.mkdir(exist_ok=True)
        dcc_rc = _run(
            [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(post_path)],
            cwd=dcc_dir,
            stdout=dcc_dir / "dcc_prog.out",
            stderr=dcc_dir / "dcc_prog.err",
            env=env,
        )
        row["dcc"] = {
            "rc": dcc_rc,
            "stderr_tail": _tail(dcc_dir / "dcc_prog.err", 24),
            **_summarize_ir(dcc_dir / "dcc_prog.out"),
        }
    row["boundary"] = _parse_boundary(
        variant_dir / "ddc.err",
        variant_dir / "dcc" / "dcc_prog.err",
        variant_dir / "dcc" / "dcc_prog.out",
    )
    row["failure_kind"] = _failure_kind(row)
    return row


def _failure_kind(row: dict[str, Any]) -> str:
    ddc = row.get("ddc", {})
    dcc = row.get("dcc", {})
    boundary = row.get("boundary")
    if ddc.get("rc") != 0:
        return "ddc-fail"
    if dcc.get("rc") not in (None, 0):
        return "dcc-lrf-boundary" if boundary else "dcc-fail"
    return "ok"


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    dcc = row.get("dcc") or {}
    boundary = row.get("boundary") or {}
    lxlu = None
    for key, fields in row.get("post_ddc_addresses", {}).items():
        if key.endswith("transfer_lds0_src:lxlu_dst:sfp"):
            lxlu = fields
            break
    return {
        "variant": row["variant"],
        "failure_kind": row.get("failure_kind"),
        "ddc_rc": row.get("ddc", {}).get("rc"),
        "dcc_rc": dcc.get("rc", ""),
        "boundary_register": boundary.get("register", ""),
        "boundary_value": boundary.get("value", ""),
        "dcc_work_ops": dcc.get("work_op_count", ""),
        "lxlu_src_first": json.dumps((lxlu or {}).get("src_start_first", [])),
        "lxlu_src_last": json.dumps((lxlu or {}).get("src_start_last", [])),
        "mutations": "; ".join(row.get("mutations", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdsc", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        choices=(
            "baseline",
            "compact-input-allocation",
            "input-transfer-dst-compact",
            "input-transfer-dst-compact-lxlu-connect",
            "input-transfer-dst-connect-only",
            "compact-alloc-and-transfer-hint",
        ),
    )
    parser.add_argument(
        "--deeptools-bin",
        default=Path("/opt/ibm/spyre/deeptools/bin"),
        type=Path,
    )
    parser.add_argument("--senarch", default="rcudd1a")
    args = parser.parse_args()

    if not args.variant:
        args.variant = [
            "baseline",
            "compact-input-allocation",
            "input-transfer-dst-compact",
            "input-transfer-dst-compact-lxlu-connect",
            "input-transfer-dst-connect-only",
            "compact-alloc-and-transfer-hint",
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _run_variant(
            source=args.sdsc,
            variant=variant,
            output_dir=args.output_dir,
            bin_dir=args.deeptools_bin,
            senarch=args.senarch,
        )
        for variant in args.variant
    ]
    jsonl_path = args.output_dir / "results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "variant": row["variant"],
                        "failure": row["failure_kind"],
                        "boundary": row.get("boundary"),
                    },
                    sort_keys=True,
                )
            )
    csv_rows = [_csv_row(row) for row in rows]
    csv_path = args.output_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "failure_counts": dict(Counter(row["failure_kind"] for row in rows)),
                "results_jsonl": str(jsonl_path),
                "results_csv": str(csv_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
