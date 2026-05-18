#!/usr/bin/env python3
"""Probe Deeptools GTR multicast lowering for SDSC JSON files.

The script optionally runs L3DlOpsScheduler_standalone for each input SDSC and
then summarizes any schedule-tree nodes carrying coreIdToGTRInfo_.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _iter_sdsc_files(inputs: Iterable[str], pattern: str) -> list[Path]:
    files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob(pattern)))
        elif path.is_file():
            files.append(path)
    return sorted(dict.fromkeys(files))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _summarize_gtr(path: Path, scheduled_path: Path | None) -> dict[str, Any]:
    parsed = _load_json(scheduled_path or path)
    gtr_nodes: list[dict[str, Any]] = []
    for node in _walk(parsed):
        info = node.get("coreIdToGTRInfo_")
        if isinstance(info, dict) and info:
            gtr_nodes.append(node)

    multicast_nodes = 0
    group_ids: set[int] = set()
    max_sharers = 0
    total_core_entries = 0
    multicast_core_entries = 0
    for node in gtr_nodes:
        info = node["coreIdToGTRInfo_"]
        node_has_multicast = False
        for entry in info.values():
            if not isinstance(entry, dict):
                continue
            total_core_entries += 1
            group_id = int(entry.get("groupId_", -1))
            num_sharers = int(entry.get("numSharers_", 0))
            max_sharers = max(max_sharers, num_sharers)
            if group_id >= 0:
                group_ids.add(group_id)
            if group_id >= 0 and num_sharers > 1:
                node_has_multicast = True
                multicast_core_entries += 1
        if node_has_multicast:
            multicast_nodes += 1

    root_name = next(iter(parsed.keys()), path.stem) if parsed else path.stem
    return {
        "input_path": str(path),
        "scheduled_path": str(scheduled_path or path),
        "root_name": root_name,
        "gtr_nodes": len(gtr_nodes),
        "multicast_nodes": multicast_nodes,
        "total_core_entries": total_core_entries,
        "multicast_core_entries": multicast_core_entries,
        "max_sharers": max_sharers,
        "group_ids": sorted(group_ids),
    }


def _run_scheduler(
    scheduler: str, input_path: Path, output_dir: Path, verbose: str | None
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(input_path).encode("utf-8")).hexdigest()[:10]
    output_path = output_dir / f"{input_path.stem}.{digest}.scheduled.json"
    log_path = output_dir / f"{input_path.stem}.{digest}.scheduled.log"
    cmd = [scheduler, "-s", str(input_path), "-o", str(output_path)]
    if verbose is not None:
        cmd.extend(["-v", verbose])
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{scheduler} failed for {input_path} with exit {proc.returncode}; "
            f"see {log_path}"
        )
    return output_path, str(log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="SDSC JSON files or directories containing SDSC JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="sdsc_*.json",
        help="File glob used when an input is a directory.",
    )
    parser.add_argument(
        "--run-scheduler",
        action="store_true",
        help="Run L3DlOpsScheduler_standalone before summarizing.",
    )
    parser.add_argument(
        "--scheduler",
        default="L3DlOpsScheduler_standalone",
        help="Scheduler executable.",
    )
    parser.add_argument(
        "--scheduler-verbose",
        default=None,
        help="Optional value passed as -v to the scheduler.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/sdsc-gtr-probe",
        help="Directory for scheduled JSON, logs, and summaries.",
    )
    parser.add_argument("--jsonl-name", default="sdsc_gtr_probe.jsonl")
    parser.add_argument("--csv-name", default="sdsc_gtr_probe.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    files = _iter_sdsc_files(args.inputs, args.pattern)
    if not files:
        raise SystemExit("no SDSC JSON files matched")

    rows: list[dict[str, Any]] = []
    for path in files:
        scheduled_path = None
        scheduler_log = ""
        if args.run_scheduler:
            scheduled_path, scheduler_log = _run_scheduler(
                args.scheduler, path, output_dir / "scheduled", args.scheduler_verbose
            )
        row = _summarize_gtr(path, scheduled_path)
        row["scheduler_log"] = scheduler_log
        rows.append(row)
        print(
            f"{path.name}: gtr_nodes={row['gtr_nodes']} "
            f"multicast_nodes={row['multicast_nodes']} "
            f"max_sharers={row['max_sharers']} group_ids={row['group_ids']}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / args.jsonl_name
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = output_dir / args.csv_name
    csv_fields = [
        "input_path",
        "scheduled_path",
        "root_name",
        "gtr_nodes",
        "multicast_nodes",
        "total_core_entries",
        "multicast_core_entries",
        "max_sharers",
        "group_ids",
        "scheduler_log",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["group_ids"] = json.dumps(csv_row["group_ids"])
            writer.writerow(csv_row)

    total_multicast_nodes = sum(int(row["multicast_nodes"]) for row in rows)
    print(
        f"wrote {len(rows)} rows to {jsonl_path}; "
        f"total_multicast_nodes={total_multicast_nodes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
