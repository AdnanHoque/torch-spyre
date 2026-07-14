#!/usr/bin/env python3

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


METRICS = (
    "wall_clock_ms.mean_ms",
    "spyre_ms.mean_ms",
    "kernel_ms.mean_ms",
    "memory_transfer_ms.mean_ms",
    "compiler_ms.mean_ms",
    "pt_util%",
)


def read_metrics(report: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    text = report.read_text(errors="replace")
    for metric in METRICS:
        match = re.search(rf"^{re.escape(metric)}\s+([0-9.]+)", text, re.MULTILINE)
        if match:
            result[metric] = float(match.group(1))
    return result


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def inspect_sdscs(cache: Path) -> dict[str, int]:
    op_counts: Counter[str] = Counter()
    explicit_lx_distributions = 0
    sdsc_count = 0
    for path in cache.rglob("sdsc_*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sdsc_count += 1
        if isinstance(payload, dict):
            for name in payload:
                op = name.split("_", 1)[1] if "_" in name else name
                op_counts[op] += 1
        for node in walk(payload):
            coordinates = node.get("coordinates_")
            if not isinstance(coordinates, dict):
                continue
            mapping = coordinates.get("coreIdToWkSlice_")
            if node.get("component_") == "lx" and isinstance(mapping, dict) and mapping:
                explicit_lx_distributions += 1
    return {
        "sdsc_count": sdsc_count,
        "restickify_hbm": op_counts["ReStickifyOpHBM"],
        "restickify_lx": op_counts["ReStickifyOpLx"],
        "explicit_lx_distributions": explicit_lx_distributions,
    }


def collect(root: Path) -> list[dict]:
    rows = []
    for report in sorted(root.rglob("report.txt")):
        run = report.parent
        rel = run.relative_to(root)
        if len(rel.parts) != 3:
            continue
        operation, runner, run_name = rel.parts
        match = re.fullmatch(r"lq(\d+)_mask([01])_(.+)", run_name)
        if not match:
            continue
        lq, masked, variant = match.groups()
        exit_code_path = run / "exit_code.txt"
        exit_code = exit_code_path.read_text().strip() if exit_code_path.exists() else "?"
        metrics = read_metrics(report)
        structure = inspect_sdscs(run / "cache")
        fired = structure["restickify_lx"] > 0 and structure["explicit_lx_distributions"] > 0
        rows.append(
            {
                "operation": operation,
                "runner": runner,
                "lq": int(lq),
                "masked": int(masked),
                "variant": variant,
                "exit_code": exit_code,
                "wall_ms": metrics.get("wall_clock_ms.mean_ms"),
                "kernel_ms": metrics.get("kernel_ms.mean_ms"),
                "spyre_ms": metrics.get("spyre_ms.mean_ms"),
                "transfer_ms": metrics.get("memory_transfer_ms.mean_ms"),
                "compiler_ms": metrics.get("compiler_ms.mean_ms"),
                "pt_util": metrics.get("pt_util%"),
                "relayout_fired": fired,
                **structure,
            }
        )
    variant_order = {"off0p2": 0, "off0p6": 1, "split": 2}
    rows.sort(
        key=lambda row: (
            row["operation"],
            row["lq"],
            row["masked"],
            variant_order.get(row["variant"], 99),
        )
    )
    return rows


def speedup(base: float | None, candidate: float | None) -> str:
    if not base or not candidate:
        return ""
    return f"{base / candidate:.3f}"


def render_markdown(rows: list[dict]) -> str:
    lookup = {
        (row["operation"], row["lq"], row["masked"], row["variant"]): row
        for row in rows
    }
    lines = [
        "# Flash PR81 Relayout Matrix",
        "",
        "`split` is the benchmark-only configuration where Torch sees full LX and DXP sees 0.6 backend workspace. It is not a production-safe allocation policy.",
        "",
        "| Operation | Lq | Mask | Variant | Kernel ms | Wall ms | ReStickify HBM | ReStickify LX | Explicit LX distributions | Fired | Kernel speedup vs 0.2 | Wall speedup vs 0.2 | Kernel speedup vs 0.6 | Wall speedup vs 0.6 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        key = (row["operation"], row["lq"], row["masked"])
        base02 = lookup.get((*key, "off0p2"), {})
        base06 = lookup.get((*key, "off0p6"), {})
        lines.append(
            "| {operation} | {lq} | {masked} | {variant} | {kernel} | {wall} | "
            "{hbm} | {lx} | {coords} | {fired} | {ks02} | {ws02} | {ks06} | {ws06} |".format(
                operation=row["operation"],
                lq=row["lq"],
                masked=row["masked"],
                variant=row["variant"],
                kernel="" if row["kernel_ms"] is None else f"{row['kernel_ms']:.3f}",
                wall="" if row["wall_ms"] is None else f"{row['wall_ms']:.3f}",
                hbm=row["restickify_hbm"],
                lx=row["restickify_lx"],
                coords=row["explicit_lx_distributions"],
                fired="yes" if row["relayout_fired"] else "no",
                ks02=speedup(base02.get("kernel_ms"), row["kernel_ms"]),
                ws02=speedup(base02.get("wall_ms"), row["wall_ms"]),
                ks06=speedup(base06.get("kernel_ms"), row["kernel_ms"]),
                ws06=speedup(base06.get("wall_ms"), row["wall_ms"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    rows = collect(args.root)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    args.markdown.write_text(render_markdown(rows))


if __name__ == "__main__":
    main()
