#!/usr/bin/env python3
"""Reduce the mirrored ReStickify DDL bridge contract.

This diagnostic generates small synthetic ``ReStickifyOpHBM`` SDSCs through the
same Torch-Spyre SDSCSpec/codegen path used by the default-off DDL bridge, then
sweeps the contract dimensions through Deeptools.  The goal is narrow: identify
which direction, size, split dimension, or loop-order variant trips the
DDC/DCC/DXP boundary seen for the mirrored in-graph 2048x2048 restickify.

The generated SDSCs are not production lowering artifacts.  They are small
contract probes for Deeptools.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SHIM_SRC = r'''
#include <iostream>
class SuperDsc;
class Dsm {
 public:
  static void doCoreletSplitSdsc(SuperDsc* sdsc);
};
class L3DlOpsScheduler {
 public:
  void run(SuperDsc& sdsc);
};
void Dsm::doCoreletSplitSdsc(SuperDsc*) {
  std::cerr << "[restickify-reducer] skipped Dsm::doCoreletSplitSdsc via LD_PRELOAD\n";
}
void L3DlOpsScheduler::run(SuperDsc&) {
  std::cerr << "[restickify-reducer] skipped L3DlOpsScheduler::run via LD_PRELOAD\n";
}
'''

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


@dataclass(frozen=True)
class Candidate:
    direction: str
    size_mb: int
    size_out: int
    split_dim: str
    loop_order: str
    address_mode: str
    stick_size: int
    num_cores: int

    @property
    def name(self) -> str:
        shape = (
            f"{self.size_mb}"
            if self.size_mb == self.size_out
            else f"mb{self.size_mb}_out{self.size_out}"
        )
        return (
            f"{self.direction}_s{shape}_split-{self.split_dim}_"
            f"loop-{self.loop_order}_addr-{self.address_mode}_stick{self.stick_size}"
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


def _contiguous_strides(layout: list[Any], sizes: dict[Any, int]) -> dict[Any, int]:
    stride = 1
    out: dict[Any, int] = {}
    for dim in reversed(layout):
        out[dim] = stride
        stride *= sizes[dim]
    return out


def _core_mapping(dims: list[Any], split_dim: Any, num_cores: int) -> dict[str, dict[Any, int]]:
    return {
        str(core): {str(dim): core if dim == split_dim else 0 for dim in dims}
        for core in range(num_cores)
    }


def _make_sdsc_spec(candidate: Candidate):
    # Keep Torch-Spyre imports lazy so --help/py_compile work in lightweight envs.
    from sympy import Symbol

    from torch_spyre._C import DataFormats
    from torch_spyre._inductor.codegen.superdsc import SDSCArgs, SDSCSpec
    from torch_spyre._inductor.constants import RESTICKIFY_OP

    mb = Symbol("mb")
    out = Symbol("out")
    dims = {"mb": mb, "out": out}
    split_dim = dims[candidate.split_dim]
    all_dims = [mb, out]
    sizes = {mb: candidate.size_mb, out: candidate.size_out}

    if candidate.direction == "forward":
        input_layout = [mb, out]
        output_layout = [out, mb]
        input_stick = out
        output_stick = split_dim
    elif candidate.direction == "mirrored":
        input_layout = [out, mb]
        output_layout = [mb, out]
        input_stick = split_dim
        output_stick = out if split_dim == mb else mb
    else:
        raise ValueError(f"unknown direction {candidate.direction}")

    data_format = DataFormats.SEN169_FP16
    work_slices = {mb: 1, out: 1}
    work_slices[split_dim] = candidate.num_cores

    args = [
        SDSCArgs(
            layout="INPUT",
            data_format=data_format,
            scales={mb: 1, out: 1},
            strides=_contiguous_strides(input_layout, sizes),
            offsets={},
            max_dim_sizes={mb: -1, out: -1},
            allocation={},
            start_address=0,
            backGap={},
        ),
        SDSCArgs(
            layout="OUTPUT",
            data_format=data_format,
            scales={mb: 1, out: 1},
            strides=_contiguous_strides(output_layout, sizes),
            offsets={},
            max_dim_sizes={mb: -1, out: -1},
            allocation={},
            start_address=1024,
            backGap={},
        ),
    ]
    return SDSCSpec(
        opfunc=RESTICKIFY_OP,
        execution_unit="sfp",
        data_format=data_format,
        num_inputs=1,
        iteration_space={mb: candidate.size_mb, out: candidate.size_out},
        num_cores=candidate.num_cores,
        work_slices=work_slices,
        core_id_to_work_slice={},
        core_id_to_work_slice_override=_core_mapping(all_dims, split_dim, candidate.num_cores),
        padding={},
        layouts={
            "INPUT": {
                "dim_order": input_layout,
                "stick_dim_order": input_stick,
                "stick_size": candidate.stick_size,
            },
            "OUTPUT": {
                "dim_order": output_layout,
                "stick_dim_order": output_stick,
                "stick_size": candidate.stick_size,
            },
        },
        args=args,
        constants={},
        coordinate_masking={},
    )


def _op_spec_stub():
    from sympy import Symbol

    from torch_spyre._inductor.constants import RESTICKIFY_OP
    from torch_spyre._inductor.op_spec import OpSpec

    d0 = Symbol("d0")
    return OpSpec(
        RESTICKIFY_OP,
        False,
        {d0: (128, 1)},
        [],
        {
            "restickify_source_kind": "in_graph_computed",
            "restickify_source_name": "reducer",
        },
    )


def _generate_bridge_payload(candidate: Candidate) -> tuple[dict[str, Any], str | None]:
    from torch_spyre._inductor.codegen.compute_ops import generate_sdsc
    from torch_spyre._inductor.codegen.restickify_ddl_bridge import (
        generate_restickify_ddl_bridge_sdsc,
        restickify_ddl_bridge_skip_reason,
    )

    spec = _make_sdsc_spec(candidate)
    compute_payload = generate_sdsc(0, spec)
    skip_reason = restickify_ddl_bridge_skip_reason(_op_spec_stub(), spec)
    bridge_payload = generate_restickify_ddl_bridge_sdsc(0, spec, compute_payload)
    _rewrite_loop_order(bridge_payload, candidate.loop_order)
    _rewrite_address_mode(bridge_payload, candidate.address_mode)
    return bridge_payload, skip_reason


def _rewrite_loop_order(payload: dict[str, Any], loop_order: str) -> None:
    _, root = _single_root(payload)
    _, dsc = _single_dsc(root)
    primary = dsc["primaryDsInfo_"]
    input_layout = list(primary["INPUT"]["layoutDimOrder_"])
    output_layout = list(primary["OUTPUT"]["layoutDimOrder_"])
    if loop_order == "input-reversed":
        dims = list(reversed(input_layout))
    elif loop_order == "input":
        dims = input_layout
    elif loop_order == "output-reversed":
        dims = list(reversed(output_layout))
    elif loop_order == "output":
        dims = output_layout
    else:
        raise ValueError(f"unknown loop order {loop_order}")

    schedule = dsc["scheduleTree_"]
    head = schedule[:3]
    tail_transfer = schedule[-1]
    loops = _loop_skeleton(dims)
    block = {
        "nodeType_": "block",
        "name_": "lx_below_schedule",
        "prev_": loops[-1]["name_"] if loops else "",
        "relevantComps_": {},
        "next_": [],
    }
    dsc["scheduleTree_"] = [*head, *loops, block, tail_transfer]


def _rewrite_address_mode(payload: dict[str, Any], address_mode: str) -> None:
    if address_mode == "generated":
        return
    _, root = _single_root(payload)
    _, dsc = _single_dsc(root)
    num_cores = int(root.get("numCoresUsed_") or dsc.get("numCoresUsed_") or 1)
    allocs = [
        node
        for node in dsc.get("scheduleTree_", [])
        if isinstance(node, dict) and node.get("nodeType_") == "allocate"
    ]
    lds_by_idx = {int(lds["ldsIdx_"]): lds for lds in dsc.get("labeledDs_", [])}
    if len(allocs) < 2:
        raise ValueError("expected input/output allocate nodes")

    input_lx = int(lds_by_idx[int(allocs[0]["ldsIdx_"])]["lxSize_"])
    output_lx = int(lds_by_idx[int(allocs[1]["ldsIdx_"])]["lxSize_"])
    if address_mode == "input-strided":
        _set_start_addresses(allocs[0], num_cores, base=0, stride=input_lx)
    elif address_mode == "both-strided":
        _set_start_addresses(allocs[0], num_cores, base=0, stride=input_lx)
        _set_start_addresses(allocs[1], num_cores, base=0, stride=output_lx)
    elif address_mode == "stage44-like":
        _set_start_addresses(allocs[0], num_cores, base=0, stride=input_lx)
        _set_start_addresses(
            allocs[1],
            num_cores,
            base=34_359_738_368,
            stride=max(128, output_lx // max(1, num_cores)),
        )
    else:
        raise ValueError(f"unknown address mode {address_mode}")


def _set_start_addresses(
    allocate_node: dict[str, Any],
    num_cores: int,
    *,
    base: int,
    stride: int,
) -> None:
    start = allocate_node.setdefault(
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
    start["data_"] = {
        f"[{core}, 0, 0]": str(base + core * stride) for core in range(num_cores)
    }


def _loop_skeleton(loop_dims: list[str]) -> list[dict[str, Any]]:
    loops: list[dict[str, Any]] = []
    for index, dim in enumerate(loop_dims):
        prev = "" if index == 0 else f"loop_ds0_ds1_{loop_dims[index - 1]}"
        next_name = (
            f"loop_ds0_ds1_{loop_dims[index + 1]}"
            if index + 1 < len(loop_dims)
            else "lx_below_schedule"
        )
        loops.append(
            {
                "nodeType_": "loop",
                "name_": f"loop_ds0_ds1_{dim}",
                "prev_": prev,
                "relevantComps_": {},
                "next_": [next_name],
                "numId_": 0,
                "denId_": 1,
                "parametricLoop_": 0,
                "parametricIterCount_": -1,
                "parametricLdsIdx_": -1,
                "dims_": [{"dim_": dim, "kind_": "unpadded"}],
                "loopCountSymbolIds_": {},
            }
        )
    return loops


def _summarize_sdsc(path: Path) -> dict[str, Any]:
    try:
        payload = _read_json_with_comments(path)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}
    name, root = _single_root(payload)
    dsc_name, dsc = _single_dsc(root)
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
    labeled_ds = []
    for lds in dsc.get("labeledDs_", []):
        labeled_ds.append(
            {
                "idx": lds.get("ldsIdx_"),
                "name": lds.get("dsName_"),
                "type": lds.get("dsType_"),
                "mem_org": sorted((lds.get("memOrg_") or {}).keys()),
                "lx_size": lds.get("lxSize_"),
            }
        )
    return {
        "sdsc_name": name,
        "dsc_name": dsc_name,
        "num_cores_used": root.get("numCoresUsed_"),
        "num_wk_slices_per_dim": root.get("numWkSlicesPerDim_"),
        "core_id_to_wk_slice_sample": dict(
            list((root.get("coreIdToWkSlice_") or {}).items())[:4]
        ),
        "primary_ds": dsc.get("primaryDsInfo_"),
        "labeled_ds": labeled_ds,
        "schedule_node_count": len(schedule),
        "schedule_node_types": dict(sorted(node_types.items())),
        "allocate_components": dict(sorted(alloc_components.items())),
        "transfer_count": len(transfer_names),
        "transfer_name_sample": transfer_names[:8],
        "data_connects": sorted(set(data_connects)),
        "op_funcs": [
            str(op.get("opFuncName"))
            for op in dsc.get("computeOp_", [])
            if isinstance(op, dict) and op.get("opFuncName")
        ],
    }


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


def _compile_shim(output_dir: Path) -> Path:
    src = output_dir / "restickify_reducer_skip_dxp_preddc_shim.cpp"
    lib = output_dir / "librestickify_reducer_skip_dxp_preddc.so"
    src.write_text(_SHIM_SRC, encoding="utf-8")
    cxx = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not cxx:
        raise FileNotFoundError("could not find g++, c++, or clang++")
    rc = _run(
        [cxx, "-shared", "-fPIC", "-std=c++17", str(src), "-o", str(lib)],
        cwd=output_dir,
        stdout=output_dir / "compile_shim.out",
        stderr=output_dir / "compile_shim.err",
    )
    if rc != 0:
        raise RuntimeError((output_dir / "compile_shim.err").read_text(encoding="utf-8"))
    return lib


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


def _run_deeptools(
    sdsc_path: Path,
    *,
    output_dir: Path,
    deeptools_bin: Path,
    senarch: str,
    run_dxp_preload: bool,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["SENARCH"] = senarch
    summary: dict[str, Any] = {}

    ddc = _tool_path(deeptools_bin, "ddc_standalone")
    ddc_rc = _run(
        [ddc, "-s", str(sdsc_path), "-d"],
        cwd=output_dir,
        stdout=output_dir / "ddc.out",
        stderr=output_dir / "ddc.err",
        env=env,
    )
    ddc_out = sdsc_path.with_suffix(".out.json")
    summary["ddc"] = {
        "rc": ddc_rc,
        "output": str(ddc_out),
        "output_exists": ddc_out.exists(),
        "stderr_tail": _tail(output_dir / "ddc.err", 20),
    }

    if ddc_out.exists():
        summary["ddc_output_summary"] = _summarize_sdsc(ddc_out)
        dcc = _tool_path(deeptools_bin, "dcc_standalone")
        dcc_dir = output_dir / "dcc_ddc_output"
        dcc_dir.mkdir(exist_ok=True)
        dcc_rc = _run(
            [dcc, "--input-mode=sdsc", "--kEmitProgIR", str(ddc_out)],
            cwd=dcc_dir,
            stdout=dcc_dir / "dcc_prog.out",
            stderr=dcc_dir / "dcc_prog.err",
            env=env,
        )
        summary["dcc_ddc_output"] = {
            "rc": dcc_rc,
            "stderr_tail": _tail(dcc_dir / "dcc_prog.err", 24),
            **_summarize_ir(dcc_dir / "dcc_prog.out"),
        }

    if run_dxp_preload:
        bundle_dir = output_dir / "dxp_bundle"
        bundle_dir.mkdir(exist_ok=True)
        bundle_sdsc = bundle_dir / "sdsc.json"
        shutil.copy2(sdsc_path, bundle_sdsc)
        (bundle_dir / "bundle.mlir").write_text(_bundle_mlir("sdsc.json"), encoding="utf-8")
        shim = _compile_shim(output_dir)
        dxp_env = env.copy()
        dxp_env["LD_PRELOAD"] = str(shim)
        dxp_env["DXP_VERBOSE"] = "0"
        dxp = _tool_path(deeptools_bin, "dxp_standalone")
        dxp_rc = _run(
            [dxp, "--bundle", "-d", str(bundle_dir)],
            cwd=bundle_dir,
            stdout=bundle_dir / "dxp_preload.log",
            env=dxp_env,
        )
        debug_dir = bundle_dir / "debug" / "sdsc"
        summary["dxp_preload"] = {
            "rc": dxp_rc,
            "ok": dxp_rc == 0,
            "log_tail": _tail(bundle_dir / "dxp_preload.log", 24),
            "senprog": _summarize_senprog(debug_dir / "senprog.txt"),
        }

    boundary = _parse_boundary(
        output_dir / "ddc.err",
        output_dir / "dcc_ddc_output" / "dcc_prog.err",
        output_dir / "dcc_ddc_output" / "dcc_prog.out",
        output_dir / "dxp_bundle" / "dxp_preload.log",
    )
    summary["boundary"] = boundary
    summary["failure_kind"] = _failure_kind(summary)
    return summary


def _tail(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def _failure_kind(summary: dict[str, Any]) -> str:
    ddc = summary.get("ddc", {})
    dcc = summary.get("dcc_ddc_output", {})
    dxp = summary.get("dxp_preload", {})
    boundary = summary.get("boundary")
    if ddc.get("rc") not in (None, 0):
        return "ddc-fail"
    if dcc.get("rc") not in (None, 0):
        return "dcc-lrf-boundary" if boundary else "dcc-fail"
    if dxp and dxp.get("rc") != 0:
        return "dxp-lrf-boundary" if boundary else "dxp-fail"
    return "ok"


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    deeptools = row.get("deeptools", {})
    ddc = deeptools.get("ddc", {})
    dcc = deeptools.get("dcc_ddc_output", {})
    dxp = deeptools.get("dxp_preload", {})
    boundary = deeptools.get("boundary") or {}
    ddc_summary = deeptools.get("ddc_output_summary", {})
    return {
        "candidate": row["candidate"],
        "direction": row["direction"],
        "size_mb": row["size_mb"],
        "size_out": row["size_out"],
        "split_dim": row["split_dim"],
        "loop_order": row["loop_order"],
        "address_mode": row["address_mode"],
        "stick_size": row["stick_size"],
        "bridge_skip_reason": row["bridge_skip_reason"] or "",
        "ddc_rc": ddc.get("rc", ""),
        "dcc_rc": dcc.get("rc", ""),
        "dxp_rc": dxp.get("rc", ""),
        "failure_kind": deeptools.get("failure_kind", ""),
        "boundary_register": boundary.get("register", ""),
        "boundary_value": boundary.get("value", ""),
        "ddc_transfers": ",".join(ddc_summary.get("transfer_name_sample", [])[:6]),
        "dcc_work_ops": dcc.get("work_op_count", ""),
        "dcc_units": json.dumps(dcc.get("unit_counts", {}), sort_keys=True),
        "senprog_tokens": json.dumps(
            (dxp.get("senprog") or {}).get("token_counts", {}),
            sort_keys=True,
        ),
    }


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    counts = Counter((row.get("deeptools") or {}).get("failure_kind", "not-run") for row in rows)
    lines = [
        "# Mirrored ReStickify DDL Reducer Summary",
        "",
        "This file is generated by `tools/restickify_mirrored_ddl_reducer.py`.",
        "",
        "## Failure Counts",
        "",
    ]
    for kind, count in sorted(counts.items()):
        lines.append(f"- `{kind}`: {count}")
    lines.extend(["", "## Rows", ""])
    lines.append(
        "| candidate | skip reason | DDC | DCC | DXP | failure | boundary |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for row in rows:
        dt = row.get("deeptools", {})
        ddc = dt.get("ddc", {})
        dcc = dt.get("dcc_ddc_output", {})
        dxp = dt.get("dxp_preload", {})
        boundary = dt.get("boundary") or {}
        boundary_text = (
            f"{boundary.get('register')}={boundary.get('value')}"
            if boundary
            else ""
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['candidate']}`",
                    f"`{row['bridge_skip_reason'] or ''}`",
                    str(ddc.get("rc", "")),
                    str(dcc.get("rc", "")),
                    str(dxp.get("rc", "")),
                    f"`{dt.get('failure_kind', '')}`",
                    boundary_text,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_shape(value: str) -> tuple[int, int]:
    if "x" in value:
        left, right = value.split("x", 1)
        return int(left), int(right)
    size = int(value)
    return size, size


def _candidates(args: argparse.Namespace) -> list[Candidate]:
    shapes = [_parse_shape(value) for value in args.size]
    return [
        Candidate(
            direction=direction,
            size_mb=size_mb,
            size_out=size_out,
            split_dim=split_dim,
            loop_order=loop_order,
            address_mode=address_mode,
            stick_size=stick_size,
            num_cores=args.num_cores,
        )
        for direction in args.direction
        for size_mb, size_out in shapes
        for split_dim in args.split_dim
        for loop_order in args.loop_order
        for address_mode in args.address_mode
        for stick_size in args.stick_size
        if not (
            direction == "forward"
            and address_mode in {"input-strided", "both-strided", "stage44-like"}
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--size",
        action="append",
        default=[],
        help="Square size N or rectangular mbxout. Repeatable.",
    )
    parser.add_argument(
        "--direction",
        action="append",
        choices=("forward", "mirrored"),
        default=[],
    )
    parser.add_argument(
        "--split-dim",
        action="append",
        choices=("mb", "out"),
        default=[],
    )
    parser.add_argument(
        "--loop-order",
        action="append",
        choices=("input-reversed", "input", "output-reversed", "output"),
        default=[],
    )
    parser.add_argument(
        "--address-mode",
        action="append",
        choices=("generated", "input-strided", "both-strided", "stage44-like"),
        default=[],
    )
    parser.add_argument("--stick-size", action="append", type=int, default=[])
    parser.add_argument("--num-cores", type=int, default=32)
    parser.add_argument("--run-deeptools", action="store_true")
    parser.add_argument("--run-dxp-preload", action="store_true")
    parser.add_argument(
        "--deeptools-bin",
        default=Path("/opt/ibm/spyre/deeptools/bin"),
        type=Path,
    )
    parser.add_argument("--senarch", default="rcudd1a")
    args = parser.parse_args()

    if not args.size:
        args.size = ["64", "128", "256", "512", "1024", "2048"]
    if not args.direction:
        args.direction = ["forward", "mirrored"]
    if not args.split_dim:
        args.split_dim = ["mb"]
    if not args.loop_order:
        args.loop_order = ["input-reversed"]
    if not args.address_mode:
        args.address_mode = ["generated"]
    if not args.stick_size:
        args.stick_size = [64]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    jsonl_path = args.output_dir / "results.jsonl"
    csv_path = args.output_dir / "results.csv"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for candidate in _candidates(args):
            candidate_dir = args.output_dir / candidate.name
            candidate_dir.mkdir(parents=True, exist_ok=True)
            payload, skip_reason = _generate_bridge_payload(candidate)
            sdsc_path = candidate_dir / "restickify_ddl_bridge.json"
            _write_json(sdsc_path, payload)
            row: dict[str, Any] = {
                "candidate": candidate.name,
                "direction": candidate.direction,
                "size_mb": candidate.size_mb,
                "size_out": candidate.size_out,
                "split_dim": candidate.split_dim,
                "loop_order": candidate.loop_order,
                "address_mode": candidate.address_mode,
                "stick_size": candidate.stick_size,
                "num_cores": candidate.num_cores,
                "bridge_skip_reason": skip_reason,
                "sdsc": str(sdsc_path),
                "input_summary": _summarize_sdsc(sdsc_path),
            }
            if args.run_deeptools or args.run_dxp_preload:
                row["deeptools"] = _run_deeptools(
                    sdsc_path,
                    output_dir=candidate_dir,
                    deeptools_bin=args.deeptools_bin,
                    senarch=args.senarch,
                    run_dxp_preload=args.run_dxp_preload,
                )
            rows.append(row)
            jsonl.write(json.dumps(row, sort_keys=True) + "\n")
            jsonl.flush()
            print(
                json.dumps(
                    {
                        "candidate": candidate.name,
                        "skip": skip_reason,
                        "failure": (row.get("deeptools") or {}).get("failure_kind"),
                    },
                    sort_keys=True,
                )
            )

    csv_rows = [_csv_row(row) for row in rows]
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
    _write_markdown(rows, args.output_dir / "summary.md")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "row_count": len(rows),
                "failure_counts": dict(
                    Counter(
                        (row.get("deeptools") or {}).get("failure_kind", "not-run")
                        for row in rows
                    )
                ),
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
