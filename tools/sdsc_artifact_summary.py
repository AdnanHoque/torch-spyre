#!/usr/bin/env python3
"""Generate benchmark artifacts from Spyre SDSC and Kineto trace outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


_UNIT_RE = re.compile(
    r"Unit: ([A-Za-z0-9_]+) Program START|"
    r"Program for unit ([A-Za-z0-9_]+):"
)
_OPCODE_RE = re.compile(r"^([A-Z][A-Z0-9_]+)\b", re.MULTILINE)


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _unwrap_sdsc(path: Path) -> tuple[str, dict[str, Any]]:
    obj = _load_json(path)
    if isinstance(obj, dict) and len(obj) == 1:
        name, value = next(iter(obj.items()))
        if isinstance(value, dict):
            return name, value
    if isinstance(obj, dict):
        return path.stem, obj
    raise ValueError(f"{path} does not contain an SDSC object")


def _find_sdscs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(path.rglob("sdsc_*.json"))
    return sorted(set(found), key=lambda p: str(p))


def _short(value: Any, *, limit: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _fmt_addr(value: Any) -> str:
    as_int = _as_int(value)
    if as_int is not None:
        return hex(as_int)
    return str(value)


def _mem_locs(mem_org: dict[str, Any] | None) -> str:
    if not mem_org:
        return ""
    locs = []
    for loc, payload in mem_org.items():
        if isinstance(payload, dict):
            present = payload.get("isPresent", True)
        else:
            present = bool(payload)
        if present:
            locs.append(loc)
    return "+".join(sorted(locs))


def _layout_summary(dsc: dict[str, Any], role: str) -> str:
    info = (dsc.get("primaryDsInfo_") or {}).get(role) or {}
    n_dims = dsc.get("N_") or {}
    dim_order = info.get("layoutDimOrder_") or []
    pieces = []
    if dim_order:
        pieces.append("layout=" + ",".join(dim_order))
    if info.get("stickDimOrder_"):
        pieces.append("stick=" + ",".join(info["stickDimOrder_"]))
    if info.get("stickSize_"):
        pieces.append("stick_size=" + _short(info["stickSize_"]))
    extents = {
        dim: n_dims[dim]
        for dim in dim_order
        if isinstance(n_dims, dict) and dim in n_dims
    }
    if extents:
        pieces.append("extent=" + _short(extents))
    return "; ".join(pieces)


def _address_summary(dsc: dict[str, Any], lds_idx: int | None) -> str:
    if lds_idx is None:
        return ""
    for node in dsc.get("scheduleTree_", []) or []:
        if node.get("ldsIdx_") != lds_idx:
            continue
        data = (node.get("startAddressCoreCorelet_") or {}).get("data_") or {}
        values = list(data.values())
        if not values:
            return ""
        parsed = [_as_int(v) for v in values]
        if all(v is not None for v in parsed):
            ints = [v for v in parsed if v is not None]
            unique = sorted(set(ints))
            if len(unique) == 1:
                return hex(unique[0])
            return f"{hex(unique[0])}..{hex(unique[-1])} ({len(unique)} unique)"
        unique_text = sorted({str(v) for v in values})
        if len(unique_text) == 1:
            return unique_text[0]
        return f"{unique_text[0]}..{unique_text[-1]} ({len(unique_text)} unique)"
    return ""


def _wk_slice_summary(wk_slice: dict[str, Any] | None) -> str:
    if not wk_slice:
        return ""
    by_core: dict[int, dict[str, Any]] = {}
    for core, value in wk_slice.items():
        core_int = _as_int(core)
        if core_int is not None and isinstance(value, dict):
            by_core[core_int] = value
    if not by_core:
        return _short(wk_slice)
    dims = sorted({dim for value in by_core.values() for dim in value})
    parts = []
    for dim in dims:
        values = {core: by_core[core].get(dim) for core in by_core}
        if all(values[core] == core for core in values):
            parts.append(f"{dim}=core_id")
            continue
        unique = sorted({value for value in values.values() if value is not None})
        if len(unique) == 1:
            parts.append(f"{dim}={unique[0]}")
        elif unique:
            parts.append(f"{dim}={unique[0]}:{unique[-1]} ({len(unique)} unique)")
    return " ".join(parts) if parts else _short(wk_slice)


def _schedule_summary(schedule: dict[str, Any] | None) -> str:
    if not schedule:
        return ""
    values = [_short(value, limit=90) for _, value in sorted(schedule.items())]
    unique = sorted(set(values))
    if len(unique) == 1:
        return "all " + unique[0]
    first_core = sorted(schedule, key=lambda value: int(value))[0]
    return f"{len(unique)} schedules; core {first_core} {values[0]}"


def _op_name_from_dsc(dsc_map: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if len(dsc_map) == 1:
        name, dsc = next(iter(dsc_map.items()))
        if isinstance(dsc, dict):
            return name, dsc
    return "unknown", dsc_map


def _root_dataops(root: dict[str, Any]) -> list[dict[str, Any]]:
    rows = root.get("datadscs_")
    if rows is None:
        rows = root.get("dataOpdscs_")
    return rows or []


def _dataop_payload(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name, payload = _op_name_from_dsc(row)
    op_name = ((payload.get("op") or {}).get("name")) or name
    return op_name, payload


def _movement_ranges(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload.get("movementRanges") or []


def _movement_range_count(payload: dict[str, Any]) -> int:
    ranges = _movement_ranges(payload)
    if ranges:
        return sum(_as_int(item.get("count")) or 0 for item in ranges)
    return len(payload.get("movements") or [])


def _movement_range_bytes(payload: dict[str, Any]) -> int:
    ranges = _movement_ranges(payload)
    if ranges:
        return sum(
            (_as_int(item.get("count")) or 0)
            * (_as_int(item.get("bytesPerMove")) or 0)
            for item in ranges
        )
    return sum(_as_int(move.get("bytes")) or 0 for move in payload.get("movements") or [])


def _movement_range_addresses(payload: dict[str, Any], side: str) -> list[int | None]:
    ranges = _movement_ranges(payload)
    if ranges:
        addresses: list[int | None] = []
        for item in ranges:
            side_payload = item.get(side) or {}
            start = _as_int(side_payload.get("lxAddress"))
            count = _as_int(item.get("count")) or 0
            stride = _as_int(item.get(f"{side}StrideBytes")) or 0
            if start is None:
                addresses.append(None)
                continue
            addresses.append(start)
            if count > 1:
                addresses.append(start + (count - 1) * stride)
        return addresses
    return [
        _as_int((move.get(side) or {}).get("lxAddress"))
        for move in payload.get("movements") or []
    ]

def sdsc_rows(sdsc: Path) -> list[dict[str, str]]:
    root_name, root = _unwrap_sdsc(sdsc)
    rows: list[dict[str, str]] = []
    root_schedule = _schedule_summary(root.get("coreIdToDscSchedule"))
    root_wk = _wk_slice_summary(root.get("coreIdToWkSlice_"))
    rel = str(sdsc)

    for index, dataop in enumerate(_root_dataops(root)):
        op_name, payload = _dataop_payload(dataop)
        movements = payload.get("movements") or []
        movement_ranges = _movement_ranges(payload)
        logical_movement_count = _movement_range_count(payload)
        byte_count = _movement_range_bytes(payload)
        src_addrs = _movement_range_addresses(payload, "source")
        dst_addrs = _movement_range_addresses(payload, "destination")
        src_unique = sorted({addr for addr in src_addrs if addr is not None})
        dst_unique = sorted({addr for addr in dst_addrs if addr is not None})
        address = []
        if src_unique:
            address.append(f"src={hex(src_unique[0])}..{hex(src_unique[-1])}")
        if dst_unique:
            address.append(f"dst={hex(dst_unique[0])}..{hex(dst_unique[-1])}")
        lowering = payload.get("lowering") or {}
        coverage = payload.get("coverage") or {}
        core_ids = payload.get("coreIdsUsed_") or []
        rows.append(
            {
                "json_file": rel,
                "root": root_name,
                "op": op_name,
                "cores": str(len(core_ids) or root.get("numCoresUsed_", "")),
                "tensor": f"dataop_{index}",
                "role": "MOVE",
                "loc": "lx->lx",
                "layout_wkslices": (
                    f"coverage={_short(coverage.get('device_sizes'))}; "
                    f"ranges={len(movement_ranges)}; "
                    f"movements={logical_movement_count}; bytes={byte_count}; "
                    f"coalesced={lowering.get('coalescedMovements', '')}"
                ),
                "address": "; ".join(address),
                "coreIdToWkSlice": _short(core_ids),
                "schedule": root_schedule,
            }
        )

    for dsc_map in root.get("dscs_", []) or []:
        op_name, dsc = _op_name_from_dsc(dsc_map)
        labeled = dsc.get("labeledDs_") or []
        if not labeled:
            rows.append(
                {
                    "json_file": rel,
                    "root": root_name,
                    "op": op_name,
                    "cores": str(dsc.get("numCoresUsed_", root.get("numCoresUsed_", ""))),
                    "tensor": "",
                    "role": "",
                    "loc": "",
                    "layout_wkslices": "",
                    "address": "",
                    "coreIdToWkSlice": root_wk,
                    "schedule": root_schedule,
                }
            )
            continue
        for tensor in labeled:
            role = tensor.get("dsType_", "")
            lds_idx = tensor.get("ldsIdx_")
            loc = _mem_locs(tensor.get("memOrg_"))
            rows.append(
                {
                    "json_file": rel,
                    "root": root_name,
                    "op": op_name,
                    "cores": str(dsc.get("numCoresUsed_", root.get("numCoresUsed_", ""))),
                    "tensor": f"{lds_idx}_{tensor.get('dsName_', '')}",
                    "role": role,
                    "loc": loc,
                    "layout_wkslices": _layout_summary(dsc, role),
                    "address": _address_summary(dsc, lds_idx),
                    "coreIdToWkSlice": root_wk,
                    "schedule": root_schedule,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "json_file",
        "root",
        "op",
        "cores",
        "tensor",
        "role",
        "loc",
        "layout_wkslices",
        "address",
        "coreIdToWkSlice",
        "schedule",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _write_markdown(path: Path, rows: list[dict[str, str]], title: str) -> None:
    fields = [
        "json_file",
        "op",
        "cores",
        "tensor",
        "role",
        "loc",
        "layout_wkslices",
        "address",
        "coreIdToWkSlice",
        "schedule",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(_md_escape(row.get(field, "")) for field in fields)
                + " |\n"
            )


def _summary(sdscs: list[Path]) -> dict[str, Any]:
    op_counts: Counter[str] = Counter()
    loc_counts: Counter[str] = Counter()
    remap_chunks = 0
    remap_movements = 0
    remap_bytes = 0
    sdsc_with_dataops = 0
    rows = []
    for sdsc in sdscs:
        root_name, root = _unwrap_sdsc(sdsc)
        dataops = _root_dataops(root)
        if dataops:
            sdsc_with_dataops += 1
        for dataop in dataops:
            op_name, payload = _dataop_payload(dataop)
            op_counts[op_name] += 1
            if op_name == "LXCoordinateRemapOp":
                remap_chunks += 1
                remap_movements += _movement_range_count(payload)
                remap_bytes += _movement_range_bytes(payload)
        for dsc_map in root.get("dscs_", []) or []:
            op_name, _ = _op_name_from_dsc(dsc_map)
            op_counts[op_name] += 1
        for row in sdsc_rows(sdsc):
            rows.append(row)
            if row["loc"]:
                loc_counts[row["loc"]] += 1
        if "ReStickifyOpHBM" in root_name:
            op_counts["ReStickifyOpHBM"] += 1
    return {
        "sdsc_count": len(sdscs),
        "row_count": len(rows),
        "sdsc_with_dataops": sdsc_with_dataops,
        "op_counts": dict(sorted(op_counts.items())),
        "loc_counts": dict(sorted(loc_counts.items())),
        "remap_chunks": remap_chunks,
        "remap_movements": remap_movements,
        "remap_bytes": remap_bytes,
    }


def _write_diff(
    path: Path,
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    senprog_rows: list[dict[str, Any]] | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# SDSC Structural Diff\n\n")
        if baseline is None:
            handle.write("No baseline SDSC directory was provided.\n\n")
        handle.write("| metric | baseline | current |\n")
        handle.write("| --- | ---: | ---: |\n")

        def get(summary: dict[str, Any] | None, key: str) -> Any:
            if summary is None:
                return ""
            return summary.get(key, "")

        for key in [
            "sdsc_count",
            "row_count",
            "sdsc_with_dataops",
            "remap_chunks",
            "remap_movements",
            "remap_bytes",
        ]:
            handle.write(f"| {key} | {get(baseline, key)} | {get(current, key)} |\n")
        handle.write("\n## Operation Counts\n\n")
        ops = set(current["op_counts"])
        if baseline:
            ops.update(baseline["op_counts"])
        handle.write("| op | baseline | current |\n")
        handle.write("| --- | ---: | ---: |\n")
        for op in sorted(ops):
            b = baseline["op_counts"].get(op, 0) if baseline else ""
            c = current["op_counts"].get(op, 0)
            handle.write(f"| {op} | {b} | {c} |\n")
        handle.write("\n## Tensor Location Counts\n\n")
        locs = set(current["loc_counts"])
        if baseline:
            locs.update(baseline["loc_counts"])
        handle.write("| loc | baseline | current |\n")
        handle.write("| --- | ---: | ---: |\n")
        for loc in sorted(locs):
            b = baseline["loc_counts"].get(loc, 0) if baseline else ""
            c = current["loc_counts"].get(loc, 0)
            handle.write(f"| {loc} | {b} | {c} |\n")
        if senprog_rows is not None:
            handle.write("\n## Senprog Status\n\n")
            handle.write("| sdsc | returncode | unit_counts | hbm_text_count | stderr_tail |\n")
            handle.write("| --- | ---: | --- | ---: | --- |\n")
            for row in senprog_rows:
                sdsc = Path(row["sdsc"]).name
                stderr_tail = _md_escape(str(row.get("stderr_tail", "")).replace("\n", " "))
                handle.write(
                    f"| {sdsc} | {row['returncode']} | "
                    f"{_md_escape(_short(row.get('unit_counts', {}), limit=120))} | "
                    f"{row.get('hbm_text_count', '')} | {stderr_tail[:240]} |\n"
                )


def _write_senprog_status(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Senprog Status\n\n")
        handle.write(
            "This is best-effort by default. Use `--fail-on-senprog-error` when "
            "a nonzero DCC return code should fail the artifact run.\n\n"
        )
        handle.write("| sdsc | returncode | senprog | stderr | unit_counts | opcode_prefix_counts | hbm_text_count |\n")
        handle.write("| --- | ---: | --- | --- | --- | --- | ---: |\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        _md_escape(Path(row["sdsc"]).name),
                        str(row["returncode"]),
                        _md_escape(row["senprog"]),
                        _md_escape(row["stderr"]),
                        _md_escape(_short(row.get("unit_counts", {}), limit=120)),
                        _md_escape(_short(row.get("opcode_prefix_counts", {}), limit=120)),
                        str(row.get("hbm_text_count", "")),
                    ]
                )
                + " |\n"
            )


def _find_trace(paths: list[Path]) -> Path | None:
    traces: list[Path] = []
    for path in paths:
        if path.is_file():
            traces.append(path)
        elif path.is_dir():
            traces.extend(path.rglob("*.pt.trace.json"))
    if not traces:
        return None
    return max(traces, key=lambda path: path.stat().st_mtime)


def _trace_summary(trace: Path, active_iters: int | None) -> dict[str, Any]:
    events = _load_json(trace).get("traceEvents", [])
    kernel_us = 0.0
    mem_us = 0.0
    kernel_events: Counter[str] = Counter()
    kernel_durations: Counter[str] = Counter()
    for event in events:
        cat = event.get("cat")
        dur = float(event.get("dur") or 0.0)
        if cat == "kernel":
            kernel_us += dur
            name = event.get("name") or "<unnamed>"
            kernel_events[name] += 1
            kernel_durations[name] += dur
        elif cat in {"gpu_memcpy", "gpu_memset"}:
            mem_us += dur
    denom = active_iters or 1
    return {
        "trace": str(trace),
        "active_iters": active_iters,
        "kernel_ms_total": kernel_us / 1000.0,
        "kernel_ms_per_iter": (kernel_us / denom) / 1000.0,
        "memory_ms_total": mem_us / 1000.0,
        "memory_ms_per_iter": (mem_us / denom) / 1000.0,
        "kernel_event_counts": dict(sorted(kernel_events.items())),
        "kernel_durations_ms": {
            name: dur / 1000.0 for name, dur in sorted(kernel_durations.items())
        },
    }


def _summarize_senprog(text: str) -> dict[str, Any]:
    units: Counter[str] = Counter()
    for match in _UNIT_RE.finditer(text):
        units[match.group(1) or match.group(2)] += 1
    opcodes = Counter(_OPCODE_RE.findall(text))
    return {
        "unit_counts": dict(sorted(units.items())),
        "opcode_counts": dict(sorted(opcodes.items())),
        "opcode_prefix_counts": dict(
            sorted(Counter(opcode.split("_")[0] for opcode in opcodes).items())
        ),
        "hbm_text_count": text.lower().count("hbm"),
    }


def _emit_senprog(sdscs: list[Path], output_dir: Path, dcc: str) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sdsc in sdscs:
        stdout_path = output_dir / f"{sdsc.stem}.senprog.txt"
        stderr_path = output_dir / f"{sdsc.stem}.stderr.txt"
        proc = subprocess.run(
            [
                dcc,
                "--input-mode=sdsc",
                "--kEmitProgIR=dump-progir progir-format=senprog",
                str(sdsc),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        rows.append(
            {
                "sdsc": str(sdsc),
                "returncode": proc.returncode,
                "senprog": str(stdout_path),
                "stderr": str(stderr_path),
                "stderr_tail": proc.stderr[-1000:],
                **_summarize_senprog(proc.stdout),
            }
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return rows


def _run_legacy_sdsc_senprog_summary(
    script: Path,
    sdscs: list[Path],
    output_dir: Path,
    dcc: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script), "--output-dir", str(output_dir), "--dcc", dcc]
    for sdsc in sdscs:
        command.extend(["--sdsc", str(sdsc)])
    (output_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    (output_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (output_dir / "returncode.txt").write_text(
        f"{proc.returncode}\n",
        encoding="utf-8",
    )
    return proc.returncode


def _write_env(path: Path, args: argparse.Namespace, sdscs: list[Path]) -> None:
    keys = [
        "PYTHONPATH",
        "DEEPTOOLS_PATH",
        "PATH",
        "LD_LIBRARY_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "SPYRE_ONCHIP_MOVE_PLANNER",
        "SPYRE_ONCHIP_MOVE_REALIZE",
        "SPYRE_ONCHIP_MOVE_CARRIER",
        "SPYRE_ONCHIP_MOVE_COORDINATE_REMAP_CHUNK_CELLS",
        "SPYRE_ONCHIP_MOVE_RANGE_ENCODING",
        "SPYRE_ONCHIP_MOVE_MAX_CELLS",
        "SPYRE_ONCHIP_MOVE_DEBUG_CELLS",
    ]
    lines = [
        "command_args=" + _short(vars(args), limit=4000),
        "sdsc_count=" + str(len(sdscs)),
    ]
    for key in keys:
        if key in os.environ:
            lines.append(f"{key}={os.environ[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*", help="SDSC files or directories")
    parser.add_argument("--sdsc-dir", type=Path, action="append", default=[])
    parser.add_argument("--baseline-sdsc-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--trace-dir", type=Path, action="append", default=[])
    parser.add_argument("--active-iters", type=int, default=None)
    parser.add_argument("--emit-senprog", action="store_true")
    parser.add_argument(
        "--sdsc-senprog-summary",
        type=Path,
        help=(
            "Optional legacy sdsc_senprog_summary.py helper to run and archive "
            "alongside this tool's built-in senprog output."
        ),
    )
    parser.add_argument("--dcc", default="dcc_standalone")
    parser.add_argument("--fail-on-senprog-error", action="store_true")
    args = parser.parse_args(argv)

    sdscs = _find_sdscs(args.paths + args.sdsc_dir)
    if not sdscs:
        raise SystemExit("no sdsc_*.json files found")
    baseline_sdscs = _find_sdscs(args.baseline_sdsc_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sdsc in sdscs:
        rows.extend(sdsc_rows(sdsc))
    _write_csv(args.output_dir / "sdsc_table.csv", rows)
    _write_markdown(args.output_dir / "sdsc_table.md", rows, "SDSC Operation Table")

    current_summary = _summary(sdscs)
    baseline_summary = _summary(baseline_sdscs) if baseline_sdscs else None
    (args.output_dir / "sdsc_summary.json").write_text(
        json.dumps(current_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_env(args.output_dir / "env.txt", args, sdscs)

    trace = _find_trace(args.trace + args.trace_dir)
    if trace is not None:
        (args.output_dir / "trace_summary.json").write_text(
            json.dumps(_trace_summary(trace, args.active_iters), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    senprog_rows = None
    if args.emit_senprog:
        senprog_rows = _emit_senprog(sdscs, args.output_dir / "senprog", args.dcc)
        _write_senprog_status(args.output_dir / "senprog_status.md", senprog_rows)
        if args.fail_on_senprog_error and any(
            row["returncode"] != 0 for row in senprog_rows
        ):
            _write_diff(
                args.output_dir / "sdsc_diff.md",
                current_summary,
                baseline_summary,
                senprog_rows,
            )
            return 1

    legacy_rc = 0
    if args.sdsc_senprog_summary:
        legacy_rc = _run_legacy_sdsc_senprog_summary(
            args.sdsc_senprog_summary,
            sdscs,
            args.output_dir / "sdsc_senprog_summary",
            args.dcc,
        )

    _write_diff(
        args.output_dir / "sdsc_diff.md",
        current_summary,
        baseline_summary,
        senprog_rows,
    )
    if args.fail_on_senprog_error and legacy_rc != 0:
        return legacy_rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
