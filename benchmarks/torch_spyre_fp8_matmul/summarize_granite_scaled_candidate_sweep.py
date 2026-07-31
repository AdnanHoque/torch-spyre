#!/usr/bin/env python3
"""Summarize the four serialized Granite FP8 candidate sweeps.

The input directory contains one subdirectory per projection family.  Every
case is self-describing through ``output/result.json``; failed cases retain an
``exit_code.txt`` and ``stderr.log`` instead.  This script never infers a timing
for a failed or incomplete case.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECTIONS = ("kv", "qo", "gate_up", "down")
VARIANTS = (
    "fp16",
    "fp8_scaled",
    "fp8_raw_dynamic",
    "fp8_raw_prepacked",
)
CONDENSED_M = (1, 512, 2048)


def _first_error_line(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _load_case(case_dir: Path, projection: str, m: int, variant: str) -> dict[str, Any]:
    result_path = case_dir / "output" / "result.json"
    exit_path = case_dir / "exit_code.txt"
    row: dict[str, Any] = {
        "projection": projection,
        "M": m,
        "variant": variant,
        "status": "incomplete",
        "exit_code": None,
        "K": None,
        "N": None,
        "kernel_us": None,
        "tflops": None,
        "correctness_passed": None,
        "error": None,
    }

    if result_path.is_file():
        result = json.loads(result_path.read_text())
        shape = result["logical_shape"]
        row.update(
            {
                "status": "passed" if result["correctness"]["passed"] else "failed",
                "exit_code": 0,
                "K": shape["K"],
                "N": shape["N"],
                "kernel_us": result["kernel_mean_us_per_iteration"],
                "tflops": result["effective_matmul_tflops"],
                "correctness_passed": result["correctness"]["passed"],
            }
        )
        return row

    if exit_path.is_file():
        row["exit_code"] = int(exit_path.read_text().strip())
        row["status"] = "failed"
        row["error"] = _first_error_line(case_dir / "stderr.log")
    return row


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for projection in PROJECTIONS:
        projection_dir = root / projection
        if not projection_dir.is_dir():
            continue
        for m_dir in sorted(
            projection_dir.glob("m*"),
            key=lambda path: int(path.name.removeprefix("m")),
        ):
            m = int(m_dir.name.removeprefix("m"))
            for variant in VARIANTS:
                case_dir = m_dir / variant
                if case_dir.is_dir():
                    rows.append(_load_case(case_dir, projection, m, variant))
    return rows


def add_comparisons(rows: list[dict[str, Any]]) -> None:
    by_case = {(row["projection"], row["M"], row["variant"]): row for row in rows}
    for row in rows:
        row["speedup_over_fp16"] = None
        row["speedup_over_full_scaled_fp8"] = None
        if row["status"] != "passed":
            continue
        fp16 = by_case.get((row["projection"], row["M"], "fp16"))
        scaled = by_case.get((row["projection"], row["M"], "fp8_scaled"))
        if fp16 is not None and fp16["status"] == "passed":
            row["speedup_over_fp16"] = fp16["kernel_us"] / row["kernel_us"]
        if scaled is not None and scaled["status"] == "passed":
            row["speedup_over_full_scaled_fp8"] = scaled["kernel_us"] / row["kernel_us"]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    by_case = {(row["projection"], row["M"], row["variant"]): row for row in rows}
    lines = [
        "# Granite scaled FP8 candidate sweep",
        "",
        "Kernel timing is the mean Kineto `kernel` duration per iteration. "
        "Compilation, host work, host/device transfers, and dynamic scale "
        "derivation are excluded. On-device LX/HBM movement remains included. "
        "The full FP8 case includes supplied-scale activation normalization and "
        "packing, FP8 matmul, and both scale applications; weights are packed "
        "outside the timed graph.",
        "",
        "## Condensed results",
        "",
        "| Projection | M | FP16 us | Full FP8 us | FP8 / FP16 | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for projection in PROJECTIONS:
        for m in CONDENSED_M:
            fp16 = by_case.get((projection, m, "fp16"))
            fp8 = by_case.get((projection, m, "fp8_scaled"))
            if fp16 is None and fp8 is None:
                continue
            status = fp8["status"] if fp8 is not None else "missing"
            error = fp8.get("error") if fp8 is not None else None
            if error:
                status = f"{status}: {error}"
            lines.append(
                "| "
                + " | ".join(
                    (
                        projection,
                        str(m),
                        _fmt(fp16.get("kernel_us") if fp16 else None),
                        _fmt(fp8.get("kernel_us") if fp8 else None),
                        _fmt(fp8.get("speedup_over_fp16") if fp8 else None),
                        status.replace("|", "\\|"),
                    )
                )
                + " |"
            )

    lines.extend(
        (
            "",
            "## All measured cases",
            "",
            "| Projection | M | Variant | Kernel us | TFLOP/s | vs FP16 | Correct |",
            "|---|---:|---|---:|---:|---:|---|",
        )
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    row["projection"],
                    str(row["M"]),
                    row["variant"],
                    _fmt(row["kernel_us"]),
                    _fmt(row["tflops"]),
                    _fmt(row["speedup_over_fp16"]),
                    str(row["correctness_passed"]),
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    output_dir = args.output_dir or args.result_root
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.result_root)
    if not rows:
        raise SystemExit(f"no sweep cases found below {args.result_root}")
    add_comparisons(rows)

    (output_dir / "sweep_summary.json").write_text(
        json.dumps({"schema_version": 1, "rows": rows}, indent=2, sort_keys=True) + "\n"
    )
    fields = list(rows[0])
    with (output_dir / "sweep_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "sweep_summary.md").write_text(render_markdown(rows))


if __name__ == "__main__":
    main()
