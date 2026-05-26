#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
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

"""Inspect mixed flash-pipeline sidecar artifacts in an Inductor cache.

This is a post-run debugging aid for ``tools/onchip_sdpa_sweep.py``.  The sweep
records whether a run compiled and executed, but it intentionally keeps its
cache summary small.  This script reads the cached source SDSC and DXP debug
SDSC JSON files and reports the parts that matter while debugging
``warp_overlap_probe``:

* source ``STCDPOpLx`` ``coreletId`` requests;
* generated ``LXLU/LXSU/PE`` PCFG component routing;
* LX source/destination addresses; and
* mixed-SDSC schedule rows, including overlapped data-op/compute rows.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


_COMPONENT_RE = re.compile(r"^(lxlu|lxsu|pe)([0-9]+)$")


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _first_item(mapping: dict[str, Any]) -> tuple[str, Any]:
    return next(iter(mapping.items()))


def _piece_summary(labeled_ds: dict[str, Any], index: int) -> dict[str, Any]:
    pieces = labeled_ds.get("PieceInfo") or []
    placements = []
    starts = []
    mem_ids = []
    for piece in pieces:
        placement = (piece.get("PlacementInfo") or [{}])[0]
        start_addr = placement.get("startAddr")
        mem_id = placement.get("memId")
        placements.append(
            {
                "coord": piece.get("dimToStartCordinate"),
                "size": piece.get("dimToSize_"),
                "memId": mem_id,
                "startAddr": start_addr,
            }
        )
        if start_addr is not None:
            starts.append(start_addr)
        if mem_id is not None:
            mem_ids.append(mem_id)
    return {
        "index": index,
        "ldsName": labeled_ds.get("ldsName_"),
        "segment": labeled_ds.get("segment_"),
        "layout": labeled_ds.get("layoutDimOrder_"),
        "stick": labeled_ds.get("stickDimOrder_"),
        "first": placements[0] if placements else None,
        "uniqueStartAddr": _unique(starts),
        "uniqueMemId": _unique(mem_ids),
    }


def _unique(values: list[Any]) -> list[Any]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _source_dataop_summary(dataop: dict[str, Any], index: int) -> dict[str, Any]:
    name, body = _first_item(dataop)
    op = body.get("op") or {}
    labeled = body.get("labeledDs_") or []
    return {
        "index": index,
        "name": name,
        "opName": op.get("name"),
        "opCoreletId": op.get("coreletId"),
        "labeledDs": [
            _piece_summary(ld, ld_index)
            for ld_index, ld in enumerate(labeled)
            if isinstance(ld, dict)
        ],
    }


def _source_compute_summary(dscs: list[Any]) -> dict[str, Any]:
    if not dscs:
        return {}
    name, body = _first_item(dscs[0])
    schedule_tree = body.get("scheduleTree_") or []
    components = Counter(
        node.get("component_")
        for node in schedule_tree
        if isinstance(node, dict) and node.get("component_") is not None
    )
    return {
        "name": name,
        "numCoreletsUsed": body.get("numCoreletsUsed_"),
        "numCoreletsUsedDSC2": body.get("numCoreletsUsed_DSC2_"),
        "coreIdsUsed": body.get("coreIdsUsed_"),
        "scheduleTreeComponents": dict(sorted(components.items())),
    }


def _overlap_rows(schedule: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for core_id, core_rows in sorted(schedule.items(), key=lambda kv: int(kv[0])):
        for row_index, row in enumerate(core_rows):
            if len(row) >= 2 and int(row[0]) >= 0 and int(row[1]) >= 0:
                rows.append(
                    {
                        "core": str(core_id),
                        "rowIndex": row_index,
                        "row": list(row),
                    }
                )
    return rows


def _source_tile_summary(path: Path, cache_dir: Path) -> dict[str, Any]:
    data = _load_json(path)
    name, body = _first_item(data)
    schedule = body.get("coreIdToDscSchedule") or {}
    dataops = [
        _source_dataop_summary(dataop, index)
        for index, dataop in enumerate(body.get("datadscs_") or [])
        if isinstance(dataop, dict)
    ]
    return {
        "name": name,
        "sourceFile": str(path.relative_to(cache_dir)),
        "graphDir": str(path.parent.relative_to(cache_dir)),
        "meta": body.get("flashAttentionPipeline_") or {},
        "scheduleRows": schedule,
        "overlapRows": _overlap_rows(schedule),
        "dataops": dataops,
        "compute": _source_compute_summary(body.get("dscs_") or []),
    }


def _debug_path_for_source(source_path: Path) -> Path | None:
    stem = source_path.stem
    debug_dir = source_path.parent / "debug" / stem
    candidates = [
        debug_dir / f"{stem}.out.out.out.json",
        debug_dir / f"{stem}.out.out.json",
        debug_dir / f"{stem}.out.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _iter_transfer_nodes(pcfg: list[Any]) -> list[dict[str, Any]]:
    transfers = []
    for core_index, core_pcfg in enumerate(pcfg):
        if not isinstance(core_pcfg, dict):
            continue
        for component, nodes in core_pcfg.items():
            if not isinstance(nodes, list):
                continue
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                node_type = node.get("type")
                if node_type not in {"datatransfer", "ptsfpdatatransfer"}:
                    continue
                transfers.append(
                    {
                        "core": core_index,
                        "component": component,
                        "name": node.get("name"),
                        "type": node_type,
                        "self": node.get("self"),
                        "coreletId": node.get("coreletId"),
                        "srcDest": node.get("srcDest"),
                        "srcStartAddr": node.get("srcStartAddr"),
                        "destStartAddr": node.get("destStartAddr"),
                        "dimLayoutOrder": node.get("dimLayoutOrder"),
                    }
                )
    return transfers


def _debug_dataop_summary(dataop: dict[str, Any], index: int) -> dict[str, Any]:
    name, body = _first_item(dataop)
    pcfg = body.get("pcfg_") or []
    components = Counter()
    corelet_components = Counter()
    for core_pcfg in pcfg:
        if not isinstance(core_pcfg, dict):
            continue
        for component in core_pcfg:
            components[component] += 1
            match = _COMPONENT_RE.match(component)
            if match:
                corelet_components[match.group(2)] += 1
    transfers = _iter_transfer_nodes(pcfg)
    return {
        "index": index,
        "name": name,
        "opCoreletId": (body.get("op") or {}).get("coreletId"),
        "components": dict(sorted(components.items())),
        "componentCorelets": dict(sorted(corelet_components.items())),
        "transfers": transfers,
    }


def _debug_tile_summary(path: Path, cache_dir: Path) -> dict[str, Any]:
    data = _load_json(path)
    name, body = _first_item(data)
    dataops = [
        _debug_dataop_summary(dataop, index)
        for index, dataop in enumerate(body.get("datadscs_") or [])
        if isinstance(dataop, dict)
    ]
    component_totals = Counter()
    transfer_corelets = Counter()
    for dataop in dataops:
        component_totals.update(dataop["components"])
        for transfer in dataop["transfers"]:
            corelet = transfer.get("coreletId")
            if corelet is not None:
                transfer_corelets[str(corelet)] += 1
    return {
        "name": name,
        "debugFile": str(path.relative_to(cache_dir)),
        "scheduleRows": body.get("coreIdToDscSchedule") or {},
        "dataops": dataops,
        "componentTotals": dict(sorted(component_totals.items())),
        "transferCoreletIds": dict(sorted(transfer_corelets.items())),
    }


def _senprog_summary(debug_path: Path) -> dict[str, int]:
    senprog = debug_path.parent / "senprog.txt"
    if not senprog.exists():
        return {}
    text = senprog.read_text(errors="ignore")
    tokens = ["LXLU:", "LXSU:", "PE:", "LXLU", "LXSU", "PE"]
    return {token: text.count(token) for token in tokens if text.count(token)}


def _tile_summary(path: Path, cache_dir: Path) -> dict[str, Any]:
    tile = _source_tile_summary(path, cache_dir)
    debug_path = _debug_path_for_source(path)
    if debug_path is not None:
        tile["debug"] = _debug_tile_summary(debug_path, cache_dir)
        tile["senprog"] = _senprog_summary(debug_path)
    else:
        tile["debug"] = None
        tile["senprog"] = {}
    return tile


def inspect_cache(cache_dir: Path) -> dict[str, Any]:
    cache_dir = cache_dir.resolve()
    tiles = []
    for path in sorted(cache_dir.rglob("sdsc_mixed_flash_pipeline_tile_*.json")):
        if "/debug/" in str(path):
            continue
        try:
            tiles.append(_tile_summary(path, cache_dir))
        except (OSError, json.JSONDecodeError, StopIteration) as exc:
            tiles.append(
                {
                    "sourceFile": str(path.relative_to(cache_dir)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "cacheDir": str(cache_dir),
        "tiles": tiles,
    }


def _cache_dirs_from_sweep_json(path: Path) -> list[Path]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of sweep result rows")
    cache_dirs = []
    for row in rows:
        if isinstance(row, dict) and row.get("cache_dir"):
            cache_dirs.append(Path(row["cache_dir"]))
    return cache_dirs


def inspect_inputs(cache_dirs: list[Path], sweep_json: Path | None) -> dict[str, Any]:
    requested = list(cache_dirs)
    if sweep_json is not None:
        requested.extend(_cache_dirs_from_sweep_json(sweep_json))
    reports = [inspect_cache(path) for path in requested]
    return {"reports": reports}


def _debug_components(tile: dict[str, Any]) -> Counter:
    components = Counter()
    debug = tile.get("debug") or {}
    for dataop in debug.get("dataops") or []:
        components.update(dataop.get("components") or {})
    return components


def validate_report(
    report: dict[str, Any],
    expect_prefetch_corelet: int | None,
    require_overlap_prefix: bool,
) -> list[str]:
    diagnostics = []
    expected_tiles = []
    routed_debug_tiles = []
    for cache_report in report["reports"]:
        tiles = cache_report.get("tiles") or []
        if not tiles:
            diagnostics.append(f"{cache_report['cacheDir']}: no mixed flash tile artifacts")
            continue
        for tile in tiles:
            if tile.get("error"):
                diagnostics.append(
                    f"{cache_report['cacheDir']}/{tile['sourceFile']}: {tile['error']}"
                )
                continue
            meta = tile.get("meta") or {}
            is_overlap_prefix = bool(meta.get("overlap_prefix"))
            if require_overlap_prefix and not is_overlap_prefix:
                continue
            if require_overlap_prefix:
                expected_tiles.append(tile)
            if expect_prefetch_corelet is None or not is_overlap_prefix:
                continue
            if meta.get("prefetch_corelet_id") != expect_prefetch_corelet:
                diagnostics.append(
                    f"{tile['sourceFile']}: meta prefetch_corelet_id="
                    f"{meta.get('prefetch_corelet_id')} expected {expect_prefetch_corelet}"
                )
            op_corelets = {
                dataop.get("opCoreletId")
                for dataop in tile.get("dataops") or []
            }
            if op_corelets != {expect_prefetch_corelet}:
                diagnostics.append(
                    f"{tile['sourceFile']}: source op corelets "
                    f"{sorted(op_corelets, key=str)} expected [{expect_prefetch_corelet}]"
                )
            if not tile.get("overlapRows"):
                diagnostics.append(f"{tile['sourceFile']}: no overlapped schedule row")
            if tile.get("debug") is None:
                continue
            components = _debug_components(tile)
            required = {
                f"lxlu{expect_prefetch_corelet}",
                f"lxsu{expect_prefetch_corelet}",
                f"pe{expect_prefetch_corelet}",
            }
            missing = sorted(required - set(components))
            if missing:
                diagnostics.append(
                    f"{tile['sourceFile']}: missing routed debug components {missing}"
                )
            else:
                routed_debug_tiles.append(tile)
    if require_overlap_prefix and not expected_tiles:
        diagnostics.append("no overlap-prefix mixed flash tile artifact found")
    if (
        expect_prefetch_corelet is not None
        and require_overlap_prefix
        and expected_tiles
        and not routed_debug_tiles
    ):
        diagnostics.append("no overlap-prefix DXP debug JSON with routed components found")
    return diagnostics


def _format_addr(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def _compact_overlap_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "none"
    first = rows[0]
    if all(
        row["rowIndex"] == first["rowIndex"] and row["row"] == first["row"]
        for row in rows
    ):
        cores = [int(row["core"]) for row in rows]
        cores.sort()
        if len(cores) > 1 and cores == list(range(cores[0], cores[-1] + 1)):
            return (
                f"c{cores[0]}..c{cores[-1]}#{first['rowIndex']}="
                f"{first['row']} ({len(cores)} cores)"
            )
    return ", ".join(
        f"c{row['core']}#{row['rowIndex']}={row['row']}"
        for row in rows
    )


def _print_human(report: dict[str, Any]) -> None:
    for cache_report in report["reports"]:
        print(f"cache: {cache_report['cacheDir']}")
        for tile in cache_report.get("tiles") or []:
            if tile.get("error"):
                print(f"  {tile['sourceFile']}: ERROR {tile['error']}")
                continue
            meta = tile.get("meta") or {}
            print(
                "  "
                f"{tile['name']} tile={meta.get('tile_index')} "
                f"overlap_prefix={meta.get('overlap_prefix')} "
                f"candidate={meta.get('overlap_candidate')} "
                f"prefetch_corelet={meta.get('prefetch_corelet_id')} "
                f"replaces={meta.get('replaces_sdsc')}"
            )
            if meta.get("overlap_prefix_rejection_reasons"):
                print(f"    rejection={meta['overlap_prefix_rejection_reasons']}")
            print(f"    overlap_rows: {_compact_overlap_rows(tile.get('overlapRows') or [])}")
            source_rows = tile.get("scheduleRows", {}).get("0")
            debug_rows = (tile.get("debug") or {}).get("scheduleRows", {}).get("0")
            if source_rows is not None:
                print(f"    source_schedule_core0: {source_rows}")
            if debug_rows is not None and debug_rows != source_rows:
                print(f"    debug_schedule_core0: {debug_rows}")
            corelets = sorted(
                {str(dataop.get("opCoreletId")) for dataop in tile.get("dataops") or []}
            )
            dst_addrs = []
            for dataop in tile.get("dataops") or []:
                labeled = dataop.get("labeledDs") or []
                if len(labeled) > 1 and labeled[1].get("first"):
                    dst_addrs.append(_format_addr(labeled[1]["first"].get("startAddr")))
            print(f"    source_op_corelets: {corelets}")
            if dst_addrs:
                print(f"    source_dst_start_addrs: {dst_addrs}")
            compute = tile.get("compute") or {}
            if compute:
                print(
                    "    compute: "
                    f"name={compute.get('name')} "
                    f"numCoreletsUsed={compute.get('numCoreletsUsed')} "
                    f"scheduleTreeComponents={compute.get('scheduleTreeComponents')}"
                )
            debug = tile.get("debug")
            if debug is None:
                print("    debug: missing")
                continue
            print(f"    debug_components: {debug.get('componentTotals')}")
            print(f"    debug_transfer_corelet_ids: {debug.get('transferCoreletIds')}")
            first_transfers = []
            for dataop in debug.get("dataops") or []:
                first_transfers.extend((dataop.get("transfers") or [])[:2])
                if len(first_transfers) >= 4:
                    break
            for transfer in first_transfers[:4]:
                print(
                    "    transfer: "
                    f"core={transfer.get('core')} "
                    f"component={transfer.get('component')} "
                    f"type={transfer.get('type')} "
                    f"coreletId={transfer.get('coreletId')} "
                    f"{transfer.get('srcDest')} "
                    f"{transfer.get('srcStartAddr')}->{transfer.get('destStartAddr')}"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_dir",
        nargs="*",
        type=Path,
        help="TORCHINDUCTOR_CACHE_DIR path(s) to inspect",
    )
    parser.add_argument(
        "--sweep-json",
        type=Path,
        help="tools/onchip_sdpa_sweep.py --output-json file to read cache dirs from",
    )
    parser.add_argument("--json", action="store_true", help="print JSON report")
    parser.add_argument(
        "--expect-prefetch-corelet",
        type=int,
        choices=(0, 1),
        help="validate overlap-prefix source ops and debug PCFG components",
    )
    parser.add_argument(
        "--require-overlap-prefix",
        action="store_true",
        help="fail validation if no overlap-prefix tile is found",
    )
    args = parser.parse_args(argv)

    if not args.cache_dir and args.sweep_json is None:
        parser.error("provide at least one cache_dir or --sweep-json")

    report = inspect_inputs(args.cache_dir, args.sweep_json)
    diagnostics = validate_report(
        report,
        args.expect_prefetch_corelet,
        args.require_overlap_prefix,
    )
    report["diagnostics"] = diagnostics
    report["ok"] = not diagnostics

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human(report)
        if diagnostics:
            print("\ndiagnostics:")
            for diagnostic in diagnostics:
                print(f"  - {diagnostic}")

    return 1 if diagnostics else 0


if __name__ == "__main__":
    raise SystemExit(main())
