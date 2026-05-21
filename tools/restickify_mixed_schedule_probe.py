#!/usr/bin/env python3
"""Build mixed DL+data-op restickify bridge schedule variants.

This is a diagnostic bridge between the proven PT-aware LX data-op splice and
normal Torch-Spyre bundle lowering.  It starts from a real generated bundle,
finds an adjacent ``ReStickifyOpHBM`` and its consumer, then emits a single
SuperDsc that contains:

* the consumer DL DSC under ``dscs_``;
* the two PT-aware LX bridge data ops under ``datadscs_``;
* a few candidate ``coreIdToDscSchedule`` spellings.

``dcc_standalone`` can consume this shape through Deeptools'
``runDcgForDataOpsDlOps`` path.  The normal DXP bundle path currently rejects
it at import time because it disallows ``datadscs_`` in bundle SDSCs; that is
the integration boundary this tool is meant to make explicit.
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

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from torch_spyre._inductor.codegen.restickify_lx_dataop import (
    generate_ptlx_restickify_bridge_sdsc,
)
from torch_spyre._inductor.codegen.restickify_ptlx_boundary import (
    _constant_lx_start_payload,
    _first_compute_input_index,
    _force_consumer_corelets,
    _infer_size_and_cores,
    _patch_consumer_input_lx_map,
)

_UNIT_RE = re.compile(r'name = "([^"]+)"')
_WORK_OP_RE = re.compile(
    r"\b("
    r"sentient\.load_and_send|sentient\.receive_and_store|"
    r"sentient\.vector_binary|agen\.vector_load|agen\.vector_store|"
    r"dataflow\.send|dataflow\.receive|sentient\.matmul|sentient\.sfp"
    r")\b"
)
_SDSC_FILENAME_RE = re.compile(r'sdsc_filename\s*=\s*"([^"]+)"')


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _single_root(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC root")
    return next(iter(payload.items()))


def _single_dsc(root: dict[str, Any]) -> dict[str, Any]:
    dscs = root.get("dscs_", []) or []
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one DL DSC")
    return next(iter(dscs[0].values()))


def _compute_op_name(payload: dict[str, Any]) -> str | None:
    try:
        _, root = _single_root(payload)
        dsc = _single_dsc(root)
    except ValueError:
        return None
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return None
    return str(ops[0].get("opFuncName"))


def _bundle_order(bundle_dir: Path) -> list[Path]:
    mlir = bundle_dir / "bundle.mlir"
    if not mlir.exists():
        return sorted(bundle_dir.glob("sdsc_*.json"))
    names = _SDSC_FILENAME_RE.findall(mlir.read_text(encoding="utf-8"))
    return [bundle_dir / name for name in names]


def _find_restickify(order: list[Path], explicit_index: int | None) -> int:
    if explicit_index is not None:
        return explicit_index
    for idx, path in enumerate(order):
        if _compute_op_name(_read_json(path)) == "ReStickifyOpHBM":
            return idx
    raise ValueError("could not find ReStickifyOpHBM in bundle order")


def _lds_name(dsc: dict[str, Any], lds_idx: int) -> str:
    for lds in dsc.get("labeledDs_", []) or []:
        if int(lds.get("ldsIdx_", -1)) == int(lds_idx):
            return str(lds.get("dsName_", f"lds{lds_idx}"))
    raise ValueError(f"could not find LDS index {lds_idx}")


def _patch_consumer_input_to_lx(
    consumer_payload: dict[str, Any],
    *,
    lds_idx: int,
    num_cores: int,
    base: int,
) -> None:
    _, root = _single_root(consumer_payload)
    dsc = _single_dsc(root)
    input_name = _lds_name(dsc, lds_idx)
    start_payload = _constant_lx_start_payload(num_cores=num_cores, base=base)
    _patch_consumer_input_lx_map(
        consumer_payload,
        input_name=input_name,
        lds_idx=lds_idx,
        start_payload=start_payload,
    )
    _force_consumer_corelets(consumer_payload, factor=1)


def _mixed_root(
    consumer_payload: dict[str, Any],
    bridge_payload: dict[str, Any],
    *,
    schedule: dict[str, list[list[int]]],
    root_name: str,
) -> dict[str, Any]:
    _, consumer_root = _single_root(consumer_payload)
    _, bridge_root = _single_root(bridge_payload)
    root = deepcopy(consumer_root)
    root["datadscs_"] = deepcopy(bridge_root.get("datadscs_", []) or [])
    root["coreIdToDscSchedule"] = schedule
    dataop_names = {
        str(next(iter(datadsc.values())).get("op", {}).get("name"))
        for datadsc in root["datadscs_"]
        if next(iter(datadsc.values())).get("op", {}).get("name") is not None
    }
    root["opFuncsUsed_"] = sorted(set(root.get("opFuncsUsed_", []) or []) | dataop_names)
    return {root_name: root}


def _schedule(num_cores: int, kind: str) -> dict[str, list[list[int]]]:
    if kind == "bridge_then_dl":
        steps = [[0, -1, 0, 1], [1, -1, 1, 1], [-1, 0, 1, 0]]
    elif kind == "bridge_then_paired_dl":
        steps = [[0, -1, 0, 1], [1, 0, 1, 1], [-1, 0, 1, 0]]
    elif kind == "paired_only":
        steps = [[0, 0, 0, 0]]
    elif kind == "paired_then_dl":
        steps = [[0, 0, 0, 1], [-1, 0, 1, 0]]
    else:
        raise ValueError(f"unknown schedule kind {kind!r}")
    return {str(core): deepcopy(steps) for core in range(num_cores)}


def _bundle_mlir(sdsc_name: str) -> str:
    return (
        "module {\n"
        "\tfunc.func @sdsc_bundle() {\n"
        f"\t\tsdscbundle.sdsc_execute () {{sdsc_filename=\"{sdsc_name}\"}}\n"
        "\t\treturn\n"
        "\t}\n"
        "}\n"
    )


def _tool_path(bin_dir: Path, tool: str) -> str:
    candidate = bin_dir / tool
    if candidate.exists():
        return str(candidate)
    found = shutil.which(tool)
    if found:
        return found
    raise FileNotFoundError(f"could not find {tool}; pass --deeptools-bin")


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def _summarize_text(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"unit_counts": {}, "work_op_count": 0, "has_hbm": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    units = Counter(_UNIT_RE.findall(text))
    return {
        "unit_counts": dict(sorted(units.items())),
        "work_op_count": len(_WORK_OP_RE.findall(text)),
        "has_hbm": "hbm" in text.lower(),
    }


def _run_deeptools(variant_dir: Path, bin_dir: Path, sdsc_path: Path) -> dict[str, Any]:
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
        "dxp_error_tail": "\n".join(dxp_proc.stderr.splitlines()[-8:]),
        **_summarize_text(variant_dir / "dcc.out"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--restickify-index", type=int)
    parser.add_argument("--consumer-input-index", type=int)
    parser.add_argument("--producer-base", type=int, default=16 * 1024)
    parser.add_argument("--consumer-base", type=int, default=8 * 1024)
    parser.add_argument(
        "--deeptools-bin",
        type=Path,
        default=Path("/opt/ibm/spyre/deeptools/bin"),
    )
    parser.add_argument("--run-deeptools", action="store_true")
    args = parser.parse_args()

    order = _bundle_order(args.bundle_dir)
    ridx = _find_restickify(order, args.restickify_index)
    if ridx + 1 >= len(order):
        raise ValueError("restickify has no adjacent consumer in bundle order")

    restickify_payload = _read_json(order[ridx])
    consumer_payload_seed = _read_json(order[ridx + 1])
    _, restickify_root = _single_root(restickify_payload)
    restickify_dsc = _single_dsc(restickify_root)
    _, consumer_root = _single_root(consumer_payload_seed)
    consumer_dsc = _single_dsc(consumer_root)
    size, num_cores = _infer_size_and_cores(restickify_root, restickify_dsc)
    consumer_input_idx = (
        args.consumer_input_index
        if args.consumer_input_index is not None
        else _first_compute_input_index(consumer_dsc)
    )

    bridge_payload = generate_ptlx_restickify_bridge_sdsc(
        "ptlx_bridge",
        size=size,
        num_cores=num_cores,
        mode="stage3b",
        direction="kernel-to-output",
        input_start_address=args.producer_base,
        output_start_address=args.consumer_base,
        restickify_op_name="ReStickifyOpWithPTLx",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for kind in [
        "bridge_then_dl",
        "bridge_then_paired_dl",
        "paired_only",
        "paired_then_dl",
    ]:
        consumer_payload = deepcopy(consumer_payload_seed)
        _patch_consumer_input_to_lx(
            consumer_payload,
            lds_idx=consumer_input_idx,
            num_cores=num_cores,
            base=args.consumer_base,
        )
        root_name = f"mixed_{kind}"
        payload = _mixed_root(
            consumer_payload,
            bridge_payload,
            schedule=_schedule(num_cores, kind),
            root_name=root_name,
        )
        variant_dir = args.output_dir / kind
        sdsc_name = f"sdsc_{root_name}.json"
        sdsc_path = variant_dir / sdsc_name
        _write_json(sdsc_path, payload)
        (variant_dir / "bundle.mlir").write_text(_bundle_mlir(sdsc_name))
        row: dict[str, Any] = {
            "variant": kind,
            "path": str(variant_dir),
            "sdsc": str(sdsc_path),
            "source_bundle": str(args.bundle_dir),
            "restickify_index": ridx,
            "restickify_sdsc": str(order[ridx]),
            "consumer_sdsc": str(order[ridx + 1]),
            "consumer_input_index": consumer_input_idx,
            "size": size,
            "num_cores": num_cores,
        }
        if args.run_deeptools:
            row.update(_run_deeptools(variant_dir, args.deeptools_bin, sdsc_path))
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    _write_json(args.output_dir / "summary.json", {"rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
