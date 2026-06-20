#!/usr/bin/env python3
"""Emit Jamie-style SDSC summaries for before/after layout movement reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import sdsc_artifact_summary as base


FIELDS = [
    "Op",
    "cores",
    "alloc_tensor {i}_{loc}",
    "Role",
    "Layout* extent/wkSlices",
    "Address",
    "coreIdToWkSlice",
    "schedule",
    "json files",
]


def _md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _sdsc_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except ValueError:
        return 1 << 30


def _short_json_path(path: str) -> str:
    return Path(path).stem


def _role_sets(dsc: dict[str, Any]) -> tuple[set[str], set[str]]:
    inputs: set[str] = set()
    outputs: set[str] = set()
    for op in dsc.get("computeOp_", []) or []:
        inputs.update(op.get("inputLabeledDs") or [])
        outputs.update(op.get("outputLabeledDs") or [])
    return inputs, outputs


def _tensor_label(tensor: dict[str, Any]) -> str:
    return f"{tensor.get('dsName_', '')}-idx{tensor.get('ldsIdx_', '')}"


def _display_role(dsc: dict[str, Any], tensor: dict[str, Any]) -> str:
    inputs, outputs = _role_sets(dsc)
    label = _tensor_label(tensor)
    if label in inputs and label in outputs:
        return "INPUT/OUTPUT"
    if label in inputs:
        return "INPUT"
    if label in outputs:
        return "OUTPUT"
    ds_type = str(tensor.get("dsType_", ""))
    if ds_type == "KERNEL":
        return "INPUT"
    return ds_type


def _role_for_layout(dsc: dict[str, Any], tensor: dict[str, Any], role: str) -> str:
    primary = dsc.get("primaryDsInfo_") or {}
    if role in primary:
        return role
    ds_type = str(tensor.get("dsType_", ""))
    if ds_type in primary:
        return ds_type
    if primary:
        return next(iter(primary.keys()))
    return role


def _display_loc(raw_loc: str) -> str:
    if raw_loc == "lx":
        return "lx"
    if raw_loc == "lx->lx":
        return "lx->lx"
    if "hbm" in raw_loc:
        return "hbm"
    return raw_loc


def _alloc_label(tensor: dict[str, Any], loc: str) -> str:
    return f"{tensor.get('ldsIdx_', '')}_{_display_loc(loc)}"


def _input_output_summary(rows: list[dict[str, str]]) -> str:
    pieces = []
    for row in rows:
        if row["Role"] in {"INPUT", "OUTPUT", "INPUT/OUTPUT", "MOVE"}:
            pieces.append(f"{row['Role']} ({row['alloc_tensor {i}_{loc}'].split('_')[-1]})")
    return ", ".join(pieces)


def _row_from_dataop(
    sdsc: Path,
    root_name: str,
    root: dict[str, Any],
    index: int,
    dataop: dict[str, Any],
) -> dict[str, str]:
    op_name, payload = base._dataop_payload(dataop)
    movements = base._movement_range_count(payload)
    byte_count = base._movement_range_bytes(payload)
    ranges = base._movement_ranges(payload)
    lowering = payload.get("lowering") or {}
    coverage = payload.get("coverage") or {}
    core_ids = payload.get("coreIdsUsed_") or []
    src_addrs = sorted(
        {addr for addr in base._movement_range_addresses(payload, "source") if addr is not None}
    )
    dst_addrs = sorted(
        {addr for addr in base._movement_range_addresses(payload, "destination") if addr is not None}
    )
    address = []
    if src_addrs:
        address.append(f"src={hex(src_addrs[0])}..{hex(src_addrs[-1])}")
    if dst_addrs:
        address.append(f"dst={hex(dst_addrs[0])}..{hex(dst_addrs[-1])}")
    return {
        "Op": op_name,
        "cores": str(len(core_ids) or root.get("numCoresUsed_", "")),
        "alloc_tensor {i}_{loc}": f"dataop_{index}_lx->lx",
        "Role": "MOVE",
        "Layout* extent/wkSlices": (
            f"coverage={base._short(coverage.get('device_sizes'))}; "
            f"ranges={len(ranges)}; movements={movements}; bytes={byte_count}; "
            f"coalesced={lowering.get('coalescedMovements', '')}"
        ),
        "Address": "; ".join(address),
        "coreIdToWkSlice": base._short(core_ids),
        "schedule": base._schedule_summary(root.get("coreIdToDscSchedule")),
        "json files": sdsc.stem,
        "_root": root_name,
        "_sdsc": str(sdsc),
    }


def _rows_for_sdsc(sdsc: Path) -> list[dict[str, str]]:
    root_name, root = base._unwrap_sdsc(sdsc)
    rows: list[dict[str, str]] = []
    for index, dataop in enumerate(base._root_dataops(root)):
        rows.append(_row_from_dataop(sdsc, root_name, root, index, dataop))
    for dsc_map in root.get("dscs_", []) or []:
        op_name, dsc = base._op_name_from_dsc(dsc_map)
        root_wk = base._wk_slice_summary(root.get("coreIdToWkSlice_"))
        root_schedule = base._schedule_summary(root.get("coreIdToDscSchedule"))
        for tensor in dsc.get("labeledDs_", []) or []:
            raw_loc = base._mem_locs(tensor.get("memOrg_"))
            role = _display_role(dsc, tensor)
            layout_role = _role_for_layout(dsc, tensor, role)
            rows.append(
                {
                    "Op": op_name,
                    "cores": str(dsc.get("numCoresUsed_", root.get("numCoresUsed_", ""))),
                    "alloc_tensor {i}_{loc}": _alloc_label(tensor, raw_loc),
                    "Role": role,
                    "Layout* extent/wkSlices": base._layout_summary(dsc, layout_role),
                    "Address": base._address_summary(dsc, tensor.get("ldsIdx_")),
                    "coreIdToWkSlice": root_wk,
                    "schedule": root_schedule,
                    "json files": sdsc.stem,
                    "_root": root_name,
                    "_sdsc": str(sdsc),
                }
            )
    return rows


def collect_rows(sdsc_dir: Path) -> list[dict[str, str]]:
    sdscs = sorted(sdsc_dir.rglob("sdsc_*.json"), key=lambda p: (_sdsc_number(p), str(p)))
    rows: list[dict[str, str]] = []
    for sdsc in sdscs:
        rows.extend(_rows_for_sdsc(sdsc))
    return rows


def _write_table_md(path: Path, title: str, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(FIELDS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in FIELDS) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_md_escape(row.get(field, "")) for field in FIELDS) + " |\n")


def _write_table_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def _group_by_sdsc(rows: list[dict[str, str]]) -> OrderedDict[str, list[dict[str, str]]]:
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["json files"], []).append(row)
    return grouped


def _write_summary(path: Path, title: str, sdsc_dir: Path, rows: list[dict[str, str]]) -> None:
    grouped = _group_by_sdsc(rows)
    op_summaries: OrderedDict[str, OrderedDict[str, None]] = OrderedDict()
    for these_rows in grouped.values():
        rows_by_op: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        for row in these_rows:
            rows_by_op.setdefault(row["Op"], []).append(row)
        for op, op_rows in rows_by_op.items():
            summary = _input_output_summary(op_rows)
            op_summaries.setdefault(op, OrderedDict()).setdefault(summary, None)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("SDSC Operations Summary - Batch Report\n")
        handle.write(f"Directory: {sdsc_dir}\n")
        handle.write(f"Total sdsc.json files found: {len(grouped)}\n\n")
        handle.write("Operations Summary:\n\n")
        for op, summaries in op_summaries.items():
            handle.write(f"{op:<20} - {'; '.join(summaries)}\n")
        handle.write("\nTensor Details:\n\n")
        for group_index, (sdsc_name, these_rows) in enumerate(grouped.items()):
            op_names = []
            for row in these_rows:
                if row["Op"] not in op_names:
                    op_names.append(row["Op"])
            core_counts = [int(row["cores"]) for row in these_rows if row["cores"].isdigit()]
            cores = max(core_counts) if core_counts else ""
            handle.write(f"{sdsc_name}: {' + '.join(op_names)} ({cores} cores)\n")
            for row in these_rows:
                layout = row["Layout* extent/wkSlices"]
                if layout and not layout.startswith(("layout=", "coverage=")):
                    layout = f"layout={layout}"
                handle.write(
                    f"  - {row['alloc_tensor {i}_{loc}']}: "
                    f"role={row['Role']}, "
                    f"{layout}, "
                    f"wkSlice={row['coreIdToWkSlice']}, "
                    f"address={row['Address']}\n"
                )
            if group_index != len(grouped) - 1:
                handle.write("\n")


def _find_first(rows: list[dict[str, str]], op: str, role: str, alloc_prefix: str) -> dict[str, str] | None:
    for row in rows:
        if row["Op"] == op and row["Role"] == role and row["alloc_tensor {i}_{loc}"].startswith(alloc_prefix):
            return row
    return None


def _metric(summary: dict[str, Any], key: str) -> Any:
    return summary.get(key, "")


def _load_summary(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_roundtrip_comparison(
    path: Path,
    baseline_rows: list[dict[str, str]],
    current_rows: list[dict[str, str]],
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
) -> None:
    checks = [
        ("Projection output", "batchmatmul", "OUTPUT", "2_"),
        ("SiLU neg first-half input", "neg", "INPUT", "0_"),
        ("SiLU realdiv first-half input", "realdiv", "INPUT", "0_"),
        ("Gate mul second-half input", "mul", "INPUT", "1_"),
        ("Gate mul output", "mul", "OUTPUT", "2_"),
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Fused SwiGLU HBM Round-Trip Comparison\n\n")
        handle.write(
            "This file is the direct before/after readout for an FMS fused "
            "SwiGLU run. The important signal is whether pointwise "
            "consumers read the projection halves from HBM or from LX after "
            "coordinate remap.\n\n"
        )
        handle.write("| edge | baseline alloc/addr | coordinate-remap alloc/addr | interpretation |\n")
        handle.write("| --- | --- | --- | --- |\n")
        for label, op, role, alloc_prefix in checks:
            before = _find_first(baseline_rows, op, role, alloc_prefix)
            after = _find_first(current_rows, op, role, alloc_prefix)
            before_text = "missing" if before is None else f"{before['alloc_tensor {i}_{loc}']} @ {before['Address']}"
            after_text = "missing" if after is None else f"{after['alloc_tensor {i}_{loc}']} @ {after['Address']}"
            before_lx = before is not None and before["alloc_tensor {i}_{loc}"].endswith("_lx")
            after_lx = after is not None and after["alloc_tensor {i}_{loc}"].endswith("_lx")
            if after_lx and not before_lx:
                interp = "HBM read eliminated for this input."
            elif after_lx and before_lx:
                interp = "Already LX-resident in both variants."
            elif before is None or after is None:
                interp = "Could not classify from SDSC rows."
            else:
                interp = "Still HBM-backed."
            handle.write(
                f"| {label} | {_md_escape(before_text)} | {_md_escape(after_text)} | {interp} |\n"
            )
        handle.write("\n## Structural Counters\n\n")
        handle.write("| metric | baseline | coordinate-remap |\n")
        handle.write("| --- | ---: | ---: |\n")
        for key in [
            "sdsc_count",
            "row_count",
            "sdsc_with_dataops",
            "remap_chunks",
            "remap_movements",
            "remap_bytes",
        ]:
            handle.write(f"| {key} | {_metric(baseline_summary, key)} | {_metric(current_summary, key)} |\n")
        handle.write("\n## Interpretation\n\n")
        if _metric(current_summary, "remap_chunks"):
            handle.write(
                "- The first projection output moves from HBM-backed SDSC rows to LX output in the coordinate-remap run.\n"
            )
            handle.write(
                "- `neg` and `realdiv` consume the first half from LX at `0x100000` after the remap carrier runs.\n"
            )
            handle.write(
                "- `mul` consumes the second half from LX at `0x100000` after the second remap carrier runs.\n"
            )
            handle.write(
                "- The pointwise chain still writes its final product to HBM for the downstream matmul; the weight restickifies also remain.\n"
            )
        else:
            handle.write(
                "- No `LXCoordinateRemapOp` chunks were emitted for this run, so the pointwise consumers remain HBM-backed.\n"
            )
            handle.write(
                "- This is expected to have little or no pass-driven speedup; any timing delta is benchmark noise or secondary compiler effects.\n"
            )


def emit_variant(sdsc_dir: Path, output_dir: Path, title: str) -> list[dict[str, str]]:
    rows = collect_rows(sdsc_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary(output_dir / "sdsc_jamie_summary.md", title, sdsc_dir, rows)
    _write_table_md(output_dir / "sdsc_jamie_table.md", title, rows)
    _write_table_csv(output_dir / "sdsc_jamie_table.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdsc-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="SDSC Jamie-Style Report")
    parser.add_argument("--baseline-sdsc-dir", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument("--current-summary", type=Path)
    args = parser.parse_args()

    current_rows = emit_variant(args.sdsc_dir, args.output_dir, args.title)
    if args.baseline_sdsc_dir:
        baseline_rows = collect_rows(args.baseline_sdsc_dir)
        _write_roundtrip_comparison(
            args.output_dir / "sdsc_hbm_roundtrip_comparison.md",
            baseline_rows,
            current_rows,
            _load_summary(args.baseline_summary) if args.baseline_summary else {},
            _load_summary(args.current_summary) if args.current_summary else {},
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
