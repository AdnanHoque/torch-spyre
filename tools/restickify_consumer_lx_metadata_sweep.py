#!/usr/bin/env python3
# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile-only sweep for consumer-side LX input metadata.

This is intentionally hardware-safe: it generates single-SDSC consumer variants
and runs ``dxp_standalone --bundle`` only. It does not call launch_kernel.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_TOKENS = ("HBM", "L3", "L3LU", "L3SU", "LXLU", "LXSU", "LX", "SFP", "PT")


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("//")
    )
    return json.loads(text)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _single_payload_dsc(
    payload: dict[str, Any],
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    if len(payload) != 1:
        raise ValueError("expected exactly one top-level SDSC")
    sdsc_name, root = next(iter(payload.items()))
    dscs = root.get("dscs_", []) or []
    if len(dscs) != 1 or len(dscs[0]) != 1:
        raise ValueError("expected exactly one dscs_ entry")
    dsc_name, dsc = next(iter(dscs[0].items()))
    return sdsc_name, root, dsc_name, dsc


def _compute_input_indices(dsc: dict[str, Any]) -> list[int]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return []
    return [
        _lds_index_from_label(token)
        for token in ops[0].get("inputLabeledDs", [])
    ]


def _lds_index_from_label(label: str) -> int:
    match = re.search(r"-idx(\d+)$", str(label))
    if not match:
        raise ValueError(f"could not parse LDS index from {label!r}")
    return int(match.group(1))


def _labeled_ds_by_idx(dsc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(lds.get("ldsIdx_", -1)): lds
        for lds in dsc.get("labeledDs_", []) or []
    }


def _alloc_node_by_idx(dsc: dict[str, Any], lds_idx: int) -> dict[str, Any] | None:
    for node in dsc.get("scheduleTree_", []) or []:
        if node.get("nodeType_") != "allocate":
            continue
        if int(node.get("ldsIdx_", -1)) == int(lds_idx):
            return node
    return None


def _core_count(root: dict[str, Any], dsc: dict[str, Any]) -> int:
    return int(root.get("numCoresUsed_", dsc.get("numCoresUsed_", 32)))


def _constant_lx_start_payload(*, num_cores: int, base: int) -> dict[str, Any]:
    return {
        "dim_prop_func": [{"Map": {}}, {"Const": {}}, {"Const": {}}],
        "dim_prop_attr": [
            {"factor_": int(num_cores), "label_": "core"},
            {"factor_": 1, "label_": "corelet"},
            {"factor_": 1, "label_": "time"},
        ],
        "data_": {
            f"[{core}, 0, 0]": str(int(base))
            for core in range(int(num_cores))
        },
    }


def _core_state_init(start_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = start_payload.get("data_", {}) or {}
    for key in sorted(data, key=lambda item: int(item.strip("[]").split(",")[0])):
        rows.append(
            {
                "ebrInit_": -1,
                "gtr_": {
                    "type": "multicast",
                    "id": 18446744073709551615,
                    "count": 0,
                    "sharers": 0,
                    "groupInfo_": {},
                },
                "condGtr_": [],
                "lbrInit_": [int(data[key])],
                "gapPerDim_": {},
                "lxSizeWithGaps_": 2_147_483_647,
                "lbrInitForwardGap_": 0,
            }
        )
    return rows


def _copy_primary_for_role(
    dsc: dict[str, Any],
    *,
    source_role: str,
    target_role: str,
) -> None:
    primary = dsc.setdefault("primaryDsInfo_", {})
    if target_role not in primary and source_role in primary:
        primary[target_role] = copy.deepcopy(primary[source_role])


def _patch_lx_input(
    payload: dict[str, Any],
    *,
    lds_idx: int,
    ds_type: str | None,
    keep_hbm_memorg: bool,
    hbm_fields: str,
    include_core_state: bool,
    base: int,
    primary_target_role: str | None,
) -> dict[str, Any]:
    _, root, _, dsc = _single_payload_dsc(payload)
    lds = _labeled_ds_by_idx(dsc).get(int(lds_idx))
    if lds is None:
        raise ValueError(f"LDS {lds_idx} not found")
    original_type = str(lds.get("dsType_", "OUTPUT"))
    if ds_type is not None:
        lds["dsType_"] = ds_type
    if primary_target_role is not None:
        _copy_primary_for_role(
            dsc,
            source_role=original_type,
            target_role=primary_target_role,
        )

    name = str(lds.get("dsName_", f"Tensor{lds_idx}"))
    alloc_name = f"allocate-{name}_lx"
    old_mem = lds.get("memOrg_", {}) or {}
    lx_meta = dict(old_mem.get("lx", {}))
    lx_meta.update(
        {
            "isPresent": 1,
            "isPadded": 0,
            "isZeroPadded": 0,
            "zpadGapFront": [0, 0],
            "gapPerDim": {},
            "dsOffset": 0,
            "allocateNode_": alloc_name,
        }
    )
    new_mem = {"lx": lx_meta}
    if keep_hbm_memorg:
        new_mem["hbm"] = dict(old_mem.get("hbm", {"isPresent": 1}))
    lds["memOrg_"] = new_mem

    if hbm_fields == "clear":
        lds["hbmStartAddress_"] = -1
        lds["hbmSize_"] = 0
    elif hbm_fields == "remove":
        lds.pop("hbmStartAddress_", None)
        lds.pop("hbmSize_", None)
    elif hbm_fields != "preserve":
        raise ValueError(f"unknown hbm_fields={hbm_fields}")

    lds["lxSize_"] = max(int(lds.get("lxSize_", 0) or 0), 2_147_483_647)
    lds["lxBufferSize_"] = max(
        int(lds.get("lxBufferSize_", 0) or 0),
        2_147_483_647,
    )
    start = _constant_lx_start_payload(num_cores=_core_count(root, dsc), base=base)
    if include_core_state:
        lds["coreStateInit_"] = _core_state_init(start)
    else:
        lds.pop("coreStateInit_", None)

    root["coreletFoldProp_"] = {"factor_": 1, "label_": "corelet"}
    dsc["numCoreletsUsed_"] = 1
    dsc["numCoreletsUsed_DSC2_"] = 1
    node = _alloc_node_by_idx(dsc, lds_idx)
    if node is None:
        raise ValueError(f"allocation node for LDS {lds_idx} not found")
    node["name_"] = alloc_name
    node["component_"] = "lx"
    node["startAddressCoreCorelet_"] = start
    return {
        "target_lds_idx": lds_idx,
        "target_ds_name": name,
        "original_ds_type": original_type,
        "final_ds_type": lds.get("dsType_"),
        "keep_hbm_memorg": keep_hbm_memorg,
        "hbm_fields": hbm_fields,
        "include_core_state": include_core_state,
        "base": base,
        "primary_target_role": primary_target_role or "",
    }


def _bundle_text(sdsc_file: str) -> str:
    return (
        "module {\n"
        "\tfunc.func @sdsc_bundle() {\n"
        f'\t\tsdscbundle.sdsc_execute () {{sdsc_filename="{sdsc_file}"}}\n'
        "\t\treturn\n"
        "\t}\n"
        "}\n"
    )


def _tool_path(explicit: str | None, name: str) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    deeptools = os.environ.get("DEEPTOOLS_INSTALL_DIR")
    if deeptools:
        candidates.append(str(Path(deeptools) / "bin" / name))
    candidates.append(f"/opt/ibm/spyre/deeptools/bin/{name}")
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(name)


def _run_dxp(bundle_dir: Path, *, dxp: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [dxp, "--bundle", "-d", str(bundle_dir)],
        cwd=bundle_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (bundle_dir / "dxp.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (bundle_dir / "dxp.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    return proc.returncode, proc.stdout, proc.stderr


def _count_tokens(bundle_dir: Path) -> dict[str, int]:
    counts = {token: 0 for token in _TOKENS}
    for path in bundle_dir.rglob("*"):
        if not path.is_file() or path.suffix not in {".txt", ".mlir", ".json"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lower = text.lower()
        for token in _TOKENS:
            counts[token] += lower.count(token.lower())
    return counts


def _compile_variant(
    *,
    name: str,
    payload: dict[str, Any],
    output_dir: Path,
    dxp: str,
) -> dict[str, Any]:
    variant_dir = output_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    sdsc_file = "sdsc_0_consumer.json"
    _write_json(variant_dir / sdsc_file, payload)
    (variant_dir / "bundle.mlir").write_text(_bundle_text(sdsc_file), encoding="utf-8")
    rc, stdout, stderr = _run_dxp(variant_dir, dxp=dxp)
    init_files = [str(path) for path in variant_dir.rglob("init.txt")]
    return {
        "variant": name,
        "returncode": rc,
        "compiled": rc == 0,
        "variant_dir": str(variant_dir),
        "init_count": len(init_files),
        "init_files": init_files[:4],
        "token_counts": _count_tokens(variant_dir),
        "stdout_tail": stdout[-1200:],
        "stderr_tail": stderr[-1600:],
    }


def _variants(
    original: dict[str, Any],
    *,
    lds_idx: int,
    base: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    rows.append(("original_hbm", copy.deepcopy(original), {"kind": "baseline"}))

    specs = [
        (
            "lx_only_output_corestate",
            {
                "ds_type": None,
                "keep_hbm_memorg": False,
                "hbm_fields": "clear",
                "include_core_state": True,
                "primary_target_role": None,
            },
        ),
        (
            "lx_only_input_corestate_primary",
            {
                "ds_type": "INPUT",
                "keep_hbm_memorg": False,
                "hbm_fields": "clear",
                "include_core_state": True,
                "primary_target_role": "INPUT",
            },
        ),
        (
            "lx_hbm_present_output_corestate",
            {
                "ds_type": None,
                "keep_hbm_memorg": True,
                "hbm_fields": "preserve",
                "include_core_state": True,
                "primary_target_role": None,
            },
        ),
        (
            "lx_only_output_no_corestate",
            {
                "ds_type": None,
                "keep_hbm_memorg": False,
                "hbm_fields": "clear",
                "include_core_state": False,
                "primary_target_role": None,
            },
        ),
        (
            "lx_hbm_present_input_primary",
            {
                "ds_type": "INPUT",
                "keep_hbm_memorg": True,
                "hbm_fields": "preserve",
                "include_core_state": True,
                "primary_target_role": "INPUT",
            },
        ),
        (
            "lx_only_input_no_corestate_primary",
            {
                "ds_type": "INPUT",
                "keep_hbm_memorg": False,
                "hbm_fields": "clear",
                "include_core_state": False,
                "primary_target_role": "INPUT",
            },
        ),
    ]
    for name, kwargs in specs:
        payload = copy.deepcopy(original)
        metadata = _patch_lx_input(payload, lds_idx=lds_idx, base=base, **kwargs)
        rows.append((name, payload, metadata))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-sdsc", required=True)
    parser.add_argument("--output-dir", default="/tmp/restickify-consumer-lx-sweep")
    parser.add_argument("--target-lds-idx", type=int, default=None)
    parser.add_argument("--lx-base", type=int, default=8192)
    parser.add_argument("--dxp", default=None)
    args = parser.parse_args()

    input_path = Path(args.consumer_sdsc).resolve()
    output_dir = Path(args.output_dir).resolve()
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = _load_json(input_path)
    _, _, _, dsc = _single_payload_dsc(original)
    input_indices = _compute_input_indices(dsc)
    if not input_indices:
        raise ValueError("consumer SDSC has no compute inputs")
    target_lds_idx = args.target_lds_idx
    if target_lds_idx is None:
        target_lds_idx = input_indices[1] if len(input_indices) > 1 else input_indices[0]

    dxp = _tool_path(args.dxp, "dxp_standalone")
    rows = []
    for name, payload, metadata in _variants(
        original,
        lds_idx=target_lds_idx,
        base=args.lx_base,
    ):
        result = _compile_variant(
            name=name,
            payload=payload,
            output_dir=output_dir,
            dxp=dxp,
        )
        result.update(
            {
                "input_path": str(input_path),
                "target_lds_idx": target_lds_idx,
                "metadata": metadata,
            }
        )
        rows.append(result)

    _write_json(output_dir / "summary.json", rows)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "variant",
                "returncode",
                "compiled",
                "init_count",
                "target_lds_idx",
                "metadata",
                "token_counts",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant": row["variant"],
                    "returncode": row["returncode"],
                    "compiled": row["compiled"],
                    "init_count": row["init_count"],
                    "target_lds_idx": row["target_lds_idx"],
                    "metadata": json.dumps(row["metadata"], sort_keys=True),
                    "token_counts": json.dumps(row["token_counts"], sort_keys=True),
                }
            )

    print(json.dumps(rows, indent=2, sort_keys=True))
    failed = [row for row in rows if row["returncode"] != 0]
    return 1 if len(failed) == len(rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
