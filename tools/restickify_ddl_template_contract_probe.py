#!/usr/bin/env python3
"""Probe restickify DDL template variants without modifying Deeptools.

Stage 48 showed that the generated ``lxlu_input`` source address comes from the
external input LX allocation. This tool asks whether an alternate DDL template
can express a local LXLU source while preserving the restickify dataflow shape.

The tool copies the installed Deeptools ``share`` tree into a per-variant temp
directory, patches only the copied ``restickify.ddl``, sets ``DEEPTOOLS_PATH`` to
that copy, and runs DDC followed by DCC. It is a contract probe, not a production
lowering path.
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


_SOURCE_WITH_ALLOCATION = (
    '%src_inp_lxsfp = ddl.unit(%inptensor, %inptensor_lx_allocation) '
    '{unit="lxlu", data_connect="lxlu_input"}'
)
_EXTERNAL_INPUT_ALLOCATION = (
    '%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation '
    '(%inptensor) {memory="lx", data_connect="lxlu_input"}'
)


def _read_json_with_comments(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
    return json.loads(text)


def _write_json(path: Path, payload: Any) -> None:
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
    stderr: Path,
    env: dict[str, str],
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
    stderr.write_text(proc.stderr, encoding="utf-8")
    return proc.returncode


def _tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _parse_boundary(*paths: Path) -> dict[str, Any] | None:
    for path in paths:
        if not path.exists():
            continue
        match = _BOUNDARY_RE.search(path.read_text(encoding="utf-8", errors="replace"))
        if match:
            return {
                "register": match.group(1).strip(),
                "value": int(match.group(2)),
                "path": str(path),
            }
    return None


def _summarize_ir(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"unit_counts": {}, "has_hbm_or_l3_units": False, "work_op_count": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    units = Counter(_UNIT_RE.findall(text))
    return {
        "unit_counts": dict(sorted(units.items())),
        "has_hbm_or_l3_units": any(unit == "hbm" or unit.startswith("l3") for unit in units),
        "work_op_count": len(_WORK_OP_RE.findall(text)),
    }


def _data_first_last(data: dict[str, Any]) -> dict[str, Any]:
    def key(item: tuple[str, Any]) -> tuple[int, str]:
        match = re.match(r"\[(\d+),", item[0])
        return (int(match.group(1)) if match else 1_000_000, item[0])

    items = sorted(data.items(), key=key)
    return {"first": items[:3], "last": items[-3:]}


def _start_data(offset: dict[str, Any]) -> dict[str, Any]:
    start = offset.get("startAddr_", {}) if isinstance(offset, dict) else {}
    if not isinstance(start, dict):
        return {}
    data = start.get("data_", {})
    return data if isinstance(data, dict) else {}


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
        name = node.get("name_")
        node_type = node.get("nodeType_")
        if not name:
            continue
        key = f"{index}:{name}"
        if node_type == "allocate" and name in {
            "allocate_Tensor0_lx_internalInput",
            "allocate_Tensor0_lx",
        }:
            start = node.get("startAddressCoreCorelet_", {}).get("data_", {})
            out[key] = {
                "node_type": node_type,
                "startAddressCoreCorelet": _data_first_last(start),
            }
        if node_type == "transfer" and name == "transfer_lds0_src:lxlu_dst:sfp":
            src_offsets = node.get("srcLdsAndLoopOffsets_", [{}])
            dst_offsets = node.get("dstLdsAndLoopOffsets_", [{}])
            src = src_offsets[0] if isinstance(src_offsets, list) else src_offsets
            dst = dst_offsets[0] if isinstance(dst_offsets, list) else dst_offsets
            src_start = _start_data(src)
            dst_start = _start_data(dst)
            out[key] = {
                "node_type": node_type,
                "src": node.get("src_"),
                "dst_vias": node.get("dstVias_"),
                "relevant_components": sorted(node.get("relevantComps_", {}).keys()),
                "src_data_connect": src.get("dataConnect_") if isinstance(src, dict) else None,
                "dst0_data_connect": dst.get("dataConnect_") if isinstance(dst, dict) else None,
                "src_start": _data_first_last(src_start),
                "dst0_start": _data_first_last(dst_start),
            }
    return out


def _patch_template(original: str, variant: str) -> tuple[str, list[str]]:
    mutations: list[str] = []
    if variant == "baseline":
        return original, mutations

    text = original
    if variant == "source-unit-no-allocation":
        text = text.replace(
            _SOURCE_WITH_ALLOCATION,
            '%src_inp_lxsfp = ddl.unit(%inptensor) {unit="lxlu", data_connect="lxlu_input"}',
        )
        mutations.append("src_inp_lxsfp=ddl.unit(%inptensor)")
        return text, mutations

    if variant == "source-unit-no-allocation-sfp-connect":
        text = text.replace(
            _SOURCE_WITH_ALLOCATION,
            '%src_inp_lxsfp = ddl.unit(%inptensor) {unit="lxlu", data_connect="sfp_input"}',
        )
        mutations.append("src_inp_lxsfp=ddl.unit(%inptensor), data_connect=sfp_input")
        return text, mutations

    if variant == "external-connect-l3":
        text = text.replace(
            _EXTERNAL_INPUT_ALLOCATION,
            '%inptensor_lx_allocation = ddl.get_external_data_transfer_allocation '
            '(%inptensor) {memory="lx", data_connect="l3_lx_input"}',
        )
        text = text.replace(
            _SOURCE_WITH_ALLOCATION,
            '%src_inp_lxsfp = ddl.unit(%inptensor, %inptensor_lx_allocation) '
            '{unit="lxlu", data_connect="l3_lx_input"}',
        )
        mutations.append("external input allocation data_connect=l3_lx_input")
        mutations.append("src_inp_lxsfp data_connect=l3_lx_input")
        return text, mutations

    if variant == "local-lx-allocation-source":
        text = text.replace(
            _EXTERNAL_INPUT_ALLOCATION,
            '%inptensor_lx_allocation = ddl.allocate(%inptensor) {memory="lx"}',
        )
        mutations.append("input allocation=ddl.allocate(memory=lx)")
        return text, mutations

    if variant == "dual-external-and-local-source":
        text = text.replace(
            _EXTERNAL_INPUT_ALLOCATION,
            '%inptensor_external_lx_allocation = ddl.get_external_data_transfer_allocation '
            '(%inptensor) {memory="lx", data_connect="lxlu_input"}\n'
            '%inptensor_lx_allocation = ddl.allocate(%inptensor) {memory="lx"}',
        )
        mutations.append("keep external input allocation under alternate name")
        mutations.append("src allocation=ddl.allocate(memory=lx)")
        return text, mutations

    raise ValueError(f"unknown variant {variant}")


def _failure_kind(ddc_rc: int, dcc_rc: int | None, boundary: dict[str, Any] | None) -> str:
    if ddc_rc != 0:
        return "ddc-failed"
    if dcc_rc is None:
        return "dcc-not-run"
    if dcc_rc != 0 and boundary is not None:
        return "dcc-lrf-boundary"
    if dcc_rc != 0:
        return "dcc-failed"
    return "ok"


def run_variant(
    *,
    sdsc: Path,
    output_dir: Path,
    deeptools_share: Path,
    deeptools_bin: Path,
    variant: str,
    template_name: str,
) -> dict[str, Any]:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    share_dir = variant_dir / "share"
    shutil.copytree(deeptools_share, share_dir)

    template_path = share_dir / "ddc" / "ddl_templates" / template_name
    original = template_path.read_text(encoding="utf-8")
    patched, mutations = _patch_template(original, variant)
    template_path.write_text(patched, encoding="utf-8")
    (variant_dir / f"{template_name}.patched").write_text(patched, encoding="utf-8")

    pre_ddc = variant_dir / "pre_ddc.json"
    shutil.copyfile(sdsc, pre_ddc)

    env = os.environ.copy()
    env["DEEPTOOLS_PATH"] = str(share_dir)
    ddc_rc = _run(
        [_tool_path(deeptools_bin, "ddc_standalone"), "-s", str(pre_ddc), "-d"],
        cwd=variant_dir,
        stdout=variant_dir / "ddc.out",
        stderr=variant_dir / "ddc.err",
        env=env,
    )

    post_ddc = variant_dir / "pre_ddc.out.json"
    dcc_rc: int | None = None
    if post_ddc.exists():
        dcc_dir = variant_dir / "dcc"
        dcc_dir.mkdir(exist_ok=True)
        dcc_rc = _run(
            [
                _tool_path(deeptools_bin, "dcc_standalone"),
                "--input-mode=sdsc",
                "--kEmitProgIR",
                str(post_ddc),
            ],
            cwd=dcc_dir,
            stdout=dcc_dir / "dcc.out",
            stderr=dcc_dir / "dcc.err",
            env=env,
        )

    dcc_dir = variant_dir / "dcc"
    boundary = _parse_boundary(variant_dir / "ddc.err", dcc_dir / "dcc.err")
    senprog = dcc_dir / "senprog.txt"
    dcc_ir = senprog if senprog.exists() else dcc_dir / "dcc.out"
    result = {
        "variant": variant,
        "template": template_name,
        "mutations": mutations,
        "pre_ddc": str(pre_ddc),
        "post_ddc": str(post_ddc) if post_ddc.exists() else None,
        "ddc": {
            "rc": ddc_rc,
            "stderr_tail": _tail(variant_dir / "ddc.err", 40),
            "output_exists": post_ddc.exists(),
        },
        "dcc": {
            "rc": dcc_rc,
            "stderr_tail": _tail(dcc_dir / "dcc.err", 40),
            **_summarize_ir(dcc_ir),
        },
        "boundary": boundary,
        "failure_kind": _failure_kind(ddc_rc, dcc_rc, boundary),
        "post_ddc_addresses": _extract_post_ddc_address_summary(post_ddc),
    }
    _write_json(variant_dir / "result.json", result)
    print(json.dumps({k: result[k] for k in ("variant", "failure_kind", "boundary")}))
    return result


def _csv_summary(result: dict[str, Any]) -> dict[str, Any]:
    transfer = next(
        (
            value
            for key, value in result.get("post_ddc_addresses", {}).items()
            if key.endswith("transfer_lds0_src:lxlu_dst:sfp")
        ),
        {},
    )
    src = transfer.get("src_start", {})
    return {
        "variant": result["variant"],
        "failure_kind": result["failure_kind"],
        "ddc_rc": result["ddc"]["rc"],
        "dcc_rc": result["dcc"]["rc"],
        "boundary_register": (result.get("boundary") or {}).get("register", ""),
        "boundary_value": (result.get("boundary") or {}).get("value", ""),
        "dcc_work_ops": result["dcc"].get("work_op_count", ""),
        "has_hbm_or_l3_units": result["dcc"].get("has_hbm_or_l3_units", ""),
        "src": json.dumps(transfer.get("src", {})),
        "relevant_components": json.dumps(transfer.get("relevant_components", [])),
        "lxlu_src_first": json.dumps(src.get("first", [])),
        "lxlu_src_last": json.dumps(src.get("last", [])),
        "mutations": "; ".join(result.get("mutations", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdsc", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--deeptools-share",
        type=Path,
        default=Path("/opt/ibm/spyre/deeptools/share"),
    )
    parser.add_argument(
        "--deeptools-bin",
        type=Path,
        default=Path("/opt/ibm/spyre/deeptools/bin"),
    )
    parser.add_argument("--template-name", default="restickify.ddl")
    parser.add_argument(
        "--variant",
        action="append",
        choices=[
            "baseline",
            "source-unit-no-allocation",
            "source-unit-no-allocation-sfp-connect",
            "external-connect-l3",
            "local-lx-allocation-source",
            "dual-external-and-local-source",
        ],
        help="Variant to run. May be repeated. Defaults to all variants.",
    )
    args = parser.parse_args()

    variants = args.variant or [
        "baseline",
        "source-unit-no-allocation",
        "source-unit-no-allocation-sfp-connect",
        "external-connect-l3",
        "local-lx-allocation-source",
        "dual-external-and-local-source",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_variant(
            sdsc=args.sdsc,
            output_dir=args.output_dir,
            deeptools_share=args.deeptools_share,
            deeptools_bin=args.deeptools_bin,
            variant=variant,
            template_name=args.template_name,
        )
        for variant in variants
    ]
    _write_json(args.output_dir / "summary.json", results)
    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, sort_keys=True) + "\n")
    rows = [_csv_summary(result) for result in results]
    with (args.output_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
