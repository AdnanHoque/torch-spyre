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

"""Audit runtime allocation maps around a restickify bundle.

This is a diagnostic helper for the PT-LX restickify prototype. It compares the
normal Torch-Spyre bundle allocations around ``ReStickifyOpHBM`` with the LX
addresses used by emitted PT-LX sidecar chunks.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sdsc_index(path: Path) -> int:
    if not path.name.startswith("sdsc_"):
        return 10**9
    try:
        return int(path.name.split("_", 2)[1])
    except Exception:
        return 10**9


def _unwrap_sdsc(path: Path) -> tuple[str, dict[str, Any]]:
    root_name, root = next(iter(_read_json(path).items()))
    dscs = root.get("dscs_", []) or []
    if not dscs:
        return root_name, root
    inner_name, inner = next(iter(dscs[0].items()))
    return inner_name, inner


def _lds_label_index(token: str) -> int:
    match = re.search(r"-idx(\d+)$", str(token))
    if not match:
        raise ValueError(f"could not parse LDS index from {token!r}")
    return int(match.group(1))


def _compute_indices(dsc: dict[str, Any]) -> tuple[list[int], list[int]]:
    ops = dsc.get("computeOp_", []) or []
    if not ops:
        return [], []
    op = ops[0]
    inputs = [_lds_label_index(token) for token in op.get("inputLabeledDs", []) or []]
    outputs = [_lds_label_index(token) for token in op.get("outputLabeledDs", []) or []]
    return inputs, outputs


def _parse_core_key(key: str) -> int | None:
    parts = [part.strip() for part in key.strip("[]").split(",") if part.strip()]
    if not parts:
        return None
    return int(parts[0])


def _alloc_maps_by_lds(dsc: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for node in dsc.get("scheduleTree_", []) or []:
        if not isinstance(node, dict) or node.get("nodeType_") != "allocate":
            continue
        lds_idx = str(int(node.get("ldsIdx_", -1)))
        component = str(node.get("component_", ""))
        data = ((node.get("startAddressCoreCorelet_") or {}).get("data_") or {})
        starts = {
            str(core): int(value)
            for key, value in data.items()
            if (core := _parse_core_key(str(key))) is not None
        }
        if starts:
            out.setdefault(lds_idx, {})[component] = starts
    return out


def _piece_lx_starts(labeled_ds: dict[str, Any]) -> list[int]:
    starts: set[int] = set()
    for piece in labeled_ds.get("PieceInfo", []) or []:
        for placement in piece.get("PlacementInfo", []) or []:
            if placement.get("type") != "lx":
                continue
            starts.update(int(start) for start in placement.get("startAddr", []) or [])
    return sorted(starts)


def _sidecar_lx_summary(path: Path) -> dict[str, Any]:
    root_name, root = next(iter(_read_json(path).items()))
    datadscs = root.get("datadscs_", []) or []
    gather_source: set[int] = set()
    scatter_dest: set[int] = set()
    op_rows = []
    for datadsc in datadscs:
        name, dataop = next(iter(datadsc.items()))
        labeled = dataop.get("labeledDs_", []) or []
        first = _piece_lx_starts(labeled[0]) if labeled else []
        last = _piece_lx_starts(labeled[-1]) if labeled else []
        if "STCDPOpLx_native_gather" in name:
            gather_source.update(first)
        if "STCDPOpLx_validgap_endpoint_scatter" in name:
            scatter_dest.update(last)
        op_rows.append(
            {
                "name": name,
                "first_lx_starts": first,
                "last_lx_starts": last,
            }
        )
    return {
        "path": str(path),
        "root_name": root_name,
        "chunk_index": (root.get("streamingPTLXFull_", {}) or {}).get("chunk_index"),
        "dataop_count": len(datadscs),
        "gather_source_lx_starts": sorted(gather_source),
        "scatter_dest_lx_starts": sorted(scatter_dest),
        "ops": op_rows,
    }


def _sdsc_summary(path: Path) -> dict[str, Any]:
    name, dsc = _unwrap_sdsc(path)
    inputs, outputs = _compute_indices(dsc)
    return {
        "path": str(path),
        "file": path.name,
        "name": name,
        "inputs": inputs,
        "outputs": outputs,
        "allocations": _alloc_maps_by_lds(dsc),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("allocation_audit.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    code_dir = args.code_dir.resolve()
    sdscs = sorted(code_dir.glob("sdsc_*.json"), key=_sdsc_index)
    sdsc_rows = [_sdsc_summary(path) for path in sdscs]
    restickify_pos = next(
        (
            index
            for index, row in enumerate(sdsc_rows)
            if "ReStickify" in row["file"] or "ReStickify" in row["name"]
        ),
        None,
    )
    sidecars = [
        _sidecar_lx_summary(path)
        for path in sorted(
            code_dir.glob("restickify_lx_neighbor_streaming_bridge_edge_*_chunk*.json")
        )
    ]

    comparison: dict[str, Any] = {"status": "no-restickify"}
    if restickify_pos is not None:
        producer = sdsc_rows[restickify_pos - 1] if restickify_pos > 0 else None
        restickify = sdsc_rows[restickify_pos]
        consumer = (
            sdsc_rows[restickify_pos + 1]
            if restickify_pos + 1 < len(sdsc_rows)
            else None
        )
        producer_output_idx = (
            str(producer["outputs"][0])
            if producer and producer.get("outputs")
            else ""
        )
        consumer_input_indices = [
            str(idx) for idx in (consumer.get("inputs") if consumer else []) or []
        ]
        producer_output_allocs = (
            (producer.get("allocations", {}) or {}).get(producer_output_idx, {})
            if producer_output_idx
            else {}
        )
        consumer_input_allocs = {
            idx: (consumer.get("allocations", {}) or {}).get(idx, {})
            for idx in consumer_input_indices
        } if consumer else {}
        comparison = {
            "status": "ok",
            "restickify_position": restickify_pos,
            "producer": producer,
            "restickify": restickify,
            "consumer": consumer,
            "producer_output_idx": producer_output_idx,
            "producer_output_components": sorted(producer_output_allocs),
            "producer_output_has_lx": "lx" in producer_output_allocs,
            "consumer_input_components": {
                idx: sorted(allocs) for idx, allocs in consumer_input_allocs.items()
            },
            "consumer_any_input_has_lx": any(
                "lx" in allocs for allocs in consumer_input_allocs.values()
            ),
            "sidecar_gather_lx_starts": sorted(
                {
                    start
                    for row in sidecars
                    for start in row["gather_source_lx_starts"]
                }
            ),
            "sidecar_scatter_lx_starts": sorted(
                {
                    start
                    for row in sidecars
                    for start in row["scatter_dest_lx_starts"]
                }
            ),
        }

    payload = {
        "code_dir": str(code_dir),
        "sdscs": sdsc_rows,
        "sidecars": sidecars,
        "comparison": comparison,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
