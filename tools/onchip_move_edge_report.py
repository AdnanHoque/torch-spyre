#!/usr/bin/env python3
"""Summarize coordinate-remap planner edges by communication class."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _view_summary(view: dict[str, Any] | None) -> str:
    if not view:
        return ""
    dims = view.get("work_slice_dims") or []
    slots = view.get("core_to_slot") or []
    dim_text = ",".join(
        f"d{item.get('device_dim')}:{item.get('split')}" for item in dims
    )
    slot_text = ",".join(
        f"d{item.get('device_dim')}={item.get('slot_expr')}" for item in slots
    )
    if dim_text and slot_text:
        return f"{dim_text}; {slot_text}"
    return dim_text or slot_text


def _classify(row: dict[str, Any]) -> str:
    if row.get("status") == "planned":
        return "exact-reshard"

    reason = str(row.get("fallback_reason") or row.get("reason") or "")
    if reason == "same-per-core-view-owned-by-lx-planner":
        return "same-view-lx-planner"
    if "consumer-duplicate-owner" in reason:
        return "fanout-multicast-unsupported"
    if "producer-duplicate-owner" in reason:
        return "producer-duplicate-owner-unsupported"
    if "k-split" in reason or "partial" in reason:
        return "reduction-or-split-k-unsupported"
    if "stick" in reason or "restickify" in reason:
        return "layout-or-stick-unsupported"
    if "capacity" in reason or "exceeds" in reason:
        return "capacity-unsupported"
    if not reason:
        return "unknown"
    return "other-fallback"


def _edge_name(row: dict[str, Any]) -> tuple[str, str]:
    producer = row.get("producer_op") or row.get("producer_op_name") or row.get("producer")
    consumer = row.get("consumer_op") or row.get("consumer_op_name") or row.get("consumer")
    return str(producer or ""), str(consumer or "")


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        producer_op, consumer_op = _edge_name(row)
        comm_class = _classify(row)
        reason = str(row.get("fallback_reason") or "")
        key = (producer_op, consumer_op, comm_class, reason)
        item = summary.setdefault(
            key,
            {
                "producer_op": producer_op,
                "consumer_op": consumer_op,
                "communication_class": comm_class,
                "status": row.get("status", ""),
                "fallback_reason": reason,
                "edge_count": 0,
                "bytes_moved": 0,
                "cell_count": 0,
                "producer_view": _view_summary(row.get("producer_view")),
                "consumer_view": _view_summary(row.get("consumer_view")),
                "example_producer": row.get("producer", ""),
                "example_consumer": row.get("consumer", ""),
            },
        )
        item["edge_count"] += 1
        item["bytes_moved"] += int(row.get("bytes_moved") or 0)
        item["cell_count"] += int(row.get("cell_count") or 0)
    return sorted(
        summary.values(),
        key=lambda item: (
            item["communication_class"],
            item["producer_op"],
            item["consumer_op"],
            item["fallback_reason"],
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "producer_op",
        "consumer_op",
        "communication_class",
        "status",
        "fallback_reason",
        "edge_count",
        "bytes_moved",
        "cell_count",
        "producer_view",
        "consumer_view",
        "example_producer",
        "example_consumer",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    class_counts = Counter(row["communication_class"] for row in rows)
    planned_bytes = sum(
        int(row["bytes_moved"])
        for row in rows
        if row["communication_class"] == "exact-reshard"
    )
    lines = [
        f"# {title}",
        "",
        f"Total summarized edge groups: {len(rows)}",
        f"Planned exact-reshard bytes: {planned_bytes}",
        "",
        "## Communication Classes",
        "",
    ]
    if class_counts:
        for name, count in sorted(class_counts.items()):
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- No on-chip move rows were found.")
    lines.extend(
        [
            "",
            "## Edge Groups",
            "",
            "| Producer op | Consumer op | Class | Status | Fallback | Edges | Bytes | Cells | Producer view | Consumer view |",
            "|---|---|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {producer_op} | {consumer_op} | `{communication_class}` | {status} | "
            "{fallback_reason} | {edge_count} | {bytes_moved} | {cell_count} | "
            "{producer_view} | {consumer_view} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="On-Chip Move Edge Report")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _summarize(_load_rows(args.jsonl))
    _write_csv(args.output_dir / "onchip_move_edge_report.csv", rows)
    _write_md(args.output_dir / "onchip_move_edge_report.md", rows, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
