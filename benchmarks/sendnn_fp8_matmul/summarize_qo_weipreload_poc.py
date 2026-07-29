#!/usr/bin/env python3
"""Validate and summarize the Q/O weight-preload FP8 PoC sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
K = 4096
N = 4096
VARIANTS = {
    "fp16": {
        "mode": "fp16",
        "dt_opt": "autopilot=1",
        "kernel": "fp16_bmm",
        "label": "FP16 matmul",
    },
    "fp8_baseline": {
        "mode": "fp8",
        "dt_opt": "autopilot=1",
        "kernel": "fp8_scaled_bmm-Qfp8",
        "label": "Baseline FP8 matmul",
    },
    "fp8_weipreload0": {
        "mode": "fp8",
        "dt_opt": "autopilot=1,weipreload=0",
        "kernel": "fp8_scaled_bmm-Qfp8",
        "label": "Optimized FP8 matmul",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def effective_tflops(m: int, mean_us: float) -> float:
    return 2 * m * K * N / (mean_us * 1_000_000)


def validate_provenance(run_root: Path) -> str:
    path = run_root / "provenance.txt"
    text = path.read_text(errors="replace")
    lowered = text.lower()
    if "ibm-senlib-dd2-" not in lowered:
        raise ValueError(f"{path}: missing DD2 package evidence")
    if "1p5" in lowered:
        raise ValueError(f"{path}: forbidden 1p5 provenance")
    for expected in (
        "benchmark_sha256="
        "3536cfcb912779e2f04013df04d534e0c11b2d38a43152ca235e21b713bbd046",
        "wrapper_sha256="
        "a5873a426ba376ca2f97172567d09e6402923f3281a40ecfb53e7949bdb174f9",
        "runner_sha256="
        "906743b1e274d46272f7ecd8fcf929d654bcb9da35e614d49a161ee8e70c0c45",
        "ibm-deeptools-2.0.0-0.main.1+1401.ee2f97a_0.el10.x86_64",
        "ibm-flex-2.0.0-0.main.1+388.81385a4_0.el10.x86_64",
        "ibm-senlib-dd2-2.0.0-0.main.1+194.951e4c4_0.el10.x86_64",
        "torch 2.10.0+aiu.kineto.1.1.1",
        "torch_sendnn 1.3.0+main.1.1bef083.0",
        "warmups=5",
        "repetitions=20",
    ):
        if expected not in text:
            raise ValueError(f"{path}: missing {expected!r}")
    return text


def validate_status(run_root: Path) -> list[dict[str, str]]:
    path = run_root / "status.tsv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"M", "variant", "mode", "dt_opt", "exit_status", "result"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"{path}: unexpected columns")
    expected = {(str(m), variant) for m in M_VALUES for variant in VARIANTS}
    actual = {(row["M"], row["variant"]) for row in rows}
    if actual != expected:
        raise ValueError(
            f"{path}: case mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    for row in rows:
        variant = VARIANTS[row["variant"]]
        if row["exit_status"] != "0":
            raise ValueError(f"{path}: failed row {row}")
        if row["mode"] != variant["mode"]:
            raise ValueError(f"{path}: mode mismatch {row}")
        if row["dt_opt"] != variant["dt_opt"]:
            raise ValueError(f"{path}: DT_OPT mismatch {row}")
    return rows


def read_case(run_root: Path, m: int, variant_name: str) -> dict:
    variant = VARIANTS[variant_name]
    path = (
        run_root
        / f"m{m}_k{K}_n{N}"
        / variant_name
        / "profile"
        / "result.json"
    )
    result = json.loads(path.read_text())
    errors = []
    if result.get("mode") != variant["mode"]:
        errors.append(f"mode={result.get('mode')!r}")
    if result.get("logical_shape") != {"M": m, "K": K, "N": N}:
        errors.append(f"logical_shape={result.get('logical_shape')!r}")
    if result.get("warmups") != 5 or result.get("repetitions") != 20:
        errors.append(
            f"warmups/repetitions={result.get('warmups')}/"
            f"{result.get('repetitions')}"
        )
    if not result.get("correctness", {}).get("passed"):
        errors.append("correctness failed")
    if result.get("environment", {}).get("DT_OPT") != variant["dt_opt"]:
        errors.append(
            f"DT_OPT={result.get('environment', {}).get('DT_OPT')!r}"
        )
    statuses = {
        **result.get("compile_statuses", {}),
        **result.get("lifecycle_statuses", {}),
    }
    if not statuses or any(value != "Status OK" for value in statuses.values()):
        errors.append(f"statuses={statuses!r}")

    events = result.get("trace_summary", {}).get("kernel_events", [])
    durations = [float(event["duration_us"]) for event in events]
    names = {str(event["name"]) for event in events}
    categories = {str(event["category"]) for event in events}
    if len(durations) != 20:
        errors.append(f"kernel event count={len(durations)}")
    if names != {variant["kernel"]}:
        errors.append(f"kernel names={sorted(names)!r}")
    if categories != {"kernel"}:
        errors.append(f"kernel categories={sorted(categories)!r}")
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        errors.append("invalid kernel duration")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))

    mean_us = statistics.fmean(durations)
    if not math.isclose(
        mean_us,
        float(result["kernel_mean_us_per_predict"]),
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{path}: recorded mean does not match events")
    correctness = result["correctness"]
    return {
        "M": m,
        "K": K,
        "N": N,
        "variant": variant_name,
        "mode": variant["mode"],
        "dt_opt": variant["dt_opt"],
        "kernel_name": variant["kernel"],
        "kernel_mean_us": mean_us,
        "kernel_p05_us": percentile(durations, 0.05),
        "kernel_p50_us": statistics.median(durations),
        "kernel_p95_us": percentile(durations, 0.95),
        "kernel_stdev_us": statistics.stdev(durations),
        "effective_matmul_tflops": effective_tflops(m, mean_us),
        "correctness_relative_l2": float(correctness["relative_l2_error"]),
        "correctness_max_abs": float(correctness["max_abs_error"]),
        "result_path": str(path.relative_to(run_root)),
    }


def build_comparisons(rows: list[dict]) -> list[dict]:
    by_key = {(row["M"], row["variant"]): row for row in rows}
    comparisons = []
    for m in M_VALUES:
        fp16 = by_key[(m, "fp16")]
        baseline = by_key[(m, "fp8_baseline")]
        treatment = by_key[(m, "fp8_weipreload0")]
        comparisons.append(
            {
                "M": m,
                "K": K,
                "N": N,
                "fp16_kernel_mean_us": fp16["kernel_mean_us"],
                "fp8_baseline_kernel_mean_us": baseline["kernel_mean_us"],
                "fp8_weipreload0_kernel_mean_us": treatment["kernel_mean_us"],
                "fp16_effective_tflops": fp16["effective_matmul_tflops"],
                "fp8_baseline_effective_tflops": baseline[
                    "effective_matmul_tflops"
                ],
                "fp8_weipreload0_effective_tflops": treatment[
                    "effective_matmul_tflops"
                ],
                "fp8_baseline_over_fp16": (
                    fp16["kernel_mean_us"] / baseline["kernel_mean_us"]
                ),
                "fp8_weipreload0_over_fp16": (
                    fp16["kernel_mean_us"] / treatment["kernel_mean_us"]
                ),
                "treatment_over_baseline": (
                    baseline["kernel_mean_us"] / treatment["kernel_mean_us"]
                ),
            }
        )
    return comparisons


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(comparisons: list[dict], path: Path) -> None:
    lines = [
        "# Q/O scaled-FP8 weight-preload PoC",
        "",
        "Logical operation: `[M,4096] @ [4096,4096]`. Each number is the "
        "mean of 20 Kineto device-kernel events after five warmups on DD2.",
        "",
        "| M | FP16 us | Stock FP8 us | FP8 `weipreload=0` us "
        "| Stock FP8 / FP16 | PoC FP8 / FP16 | PoC / stock FP8 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['M']} "
            f"| {row['fp16_kernel_mean_us']:.3f} "
            f"| {row['fp8_baseline_kernel_mean_us']:.3f} "
            f"| {row['fp8_weipreload0_kernel_mean_us']:.3f} "
            f"| {row['fp8_baseline_over_fp16']:.3f}x "
            f"| {row['fp8_weipreload0_over_fp16']:.3f}x "
            f"| {row['treatment_over_baseline']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "The FP8 kernel includes FP16-to-FP8 `Qfp8`, any inserted "
            "relayout, FP8 BatchMatMul, both scale-recovery stages, and FP16 "
            "output production. Scale derivation is outside this standalone "
            "graph; per-row and per-output-channel scale inputs are fixed at "
            "one for correctness isolation.",
            "",
            "The PoC changes one DeepTools option only: "
            "`DT_OPT=autopilot=1,weipreload=0`. It is an experimental "
            "whole-graph switch, not a production recommendation.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def rounded_axis_max(maximum: float) -> tuple[float, float]:
    rough_step = maximum / 5
    magnitude = 10 ** math.floor(math.log10(rough_step))
    for multiplier in (1, 2, 5, 10):
        step = multiplier * magnitude
        if step >= rough_step:
            break
    return math.ceil(maximum * 1.08 / step) * step, step


def write_chart(rows: list[dict], path: Path) -> None:
    width, height = 980, 680
    left, right, top, bottom = 88, 946, 118, 592
    plot_width = right - left
    plot_height = bottom - top
    y_max, y_step = rounded_axis_max(
        max(row["effective_matmul_tflops"] for row in rows)
    )

    def xpos(m: int) -> float:
        return left + math.log2(m) / math.log2(M_VALUES[-1]) * plot_width

    def ypos(value: float) -> float:
        return bottom - value / y_max * plot_height

    styles = {
        "fp16": ("#3568b8", "circle"),
        "fp8_baseline": ("#d35434", "square"),
        "fp8_weipreload0": ("#23845b", "triangle"),
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">Q/O matmul performance</title>',
        '<desc id="description">Effective TFLOP per second over M for FP16, '
        "baseline FP8, and optimized FP8 matmul.</desc>",
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "fill:#202124}.small{font-size:13px;fill:#5f6368}"
        ".tick{font-size:11px;fill:#5f6368}.axis{stroke:#80868b;"
        "stroke-width:1.1}.grid{stroke:#e4e7eb;stroke-width:1}"
        ".series{fill:none;stroke-width:2.8;stroke-linejoin:round;"
        "stroke-linecap:round}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        '<text x="42" y="43" font-size="25" font-weight="600">'
        "Q/O Matmul Performance</text>",
        '<text x="42" y="70" class="small">'
        "Granite 3 8B · K=N=4096</text>",
    ]

    legend_x = 42
    for variant_name in ("fp16", "fp8_baseline", "fp8_weipreload0"):
        color, _ = styles[variant_name]
        label = VARIANTS[variant_name]["label"]
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="91" x2="{legend_x + 30}" '
                f'y2="91" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 38}" y="96" font-size="13">'
                f"{escape(label)}</text>",
            ]
        )
        legend_x += 210 if variant_name != "fp8_weipreload0" else 0

    tick = 0.0
    while tick <= y_max + 1e-9:
        y = ypos(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{right}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" '
                f'text-anchor="end">{tick:g}</text>',
            ]
        )
        tick += y_step
    for index, m in enumerate(M_VALUES):
        x = xpos(m)
        parts.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top}" '
            f'x2="{x:.2f}" y2="{bottom}"/>'
        )
        if index % 2 == 0 or m == M_VALUES[-1]:
            parts.append(
                f'<text class="tick" x="{x:.2f}" y="{bottom + 21}" '
                f'text-anchor="middle">{m}</text>'
            )
    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{bottom}" '
            f'x2="{right}" y2="{bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" '
            f'x2="{left}" y2="{bottom}"/>',
            f'<text x="{(left + right) / 2:.2f}" y="638" class="small" '
            'text-anchor="middle">M (log scale)</text>',
            f'<text x="23" y="{(top + bottom) / 2:.2f}" class="small" '
            f'text-anchor="middle" transform="rotate(-90 23 '
            f'{(top + bottom) / 2:.2f})">TFLOP/s</text>',
        ]
    )

    for variant_name in ("fp16", "fp8_baseline", "fp8_weipreload0"):
        color, marker = styles[variant_name]
        selected = [row for row in rows if row["variant"] == variant_name]
        points = " ".join(
            f"{xpos(row['M']):.2f},{ypos(row['effective_matmul_tflops']):.2f}"
            for row in selected
        )
        parts.append(
            f'<polyline class="series" points="{points}" stroke="{color}"/>'
        )
        for row in selected:
            x = xpos(row["M"])
            y = ypos(row["effective_matmul_tflops"])
            if marker == "circle":
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.4" '
                    f'fill="#fff" stroke="{color}" stroke-width="2"/>'
                )
            elif marker == "square":
                parts.append(
                    f'<rect x="{x - 4.2:.2f}" y="{y - 4.2:.2f}" '
                    f'width="8.4" height="8.4" fill="#fff" stroke="{color}" '
                    'stroke-width="2"/>'
                )
            else:
                points = (
                    f"{x:.2f},{y - 5:.2f} "
                    f"{x - 4.8:.2f},{y + 4:.2f} "
                    f"{x + 4.8:.2f},{y + 4:.2f}"
                )
                parts.append(
                    f'<polygon points="{points}" fill="#fff" '
                    f'stroke="{color}" stroke-width="2"/>'
                )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = validate_provenance(run_root)
    status_rows = validate_status(run_root)
    rows = [
        read_case(run_root, m, variant)
        for m in M_VALUES
        for variant in VARIANTS
    ]
    comparisons = build_comparisons(rows)
    write_csv(rows, output_dir / "qo_weipreload_poc_rows.csv")
    write_csv(comparisons, output_dir / "qo_weipreload_poc_comparison.csv")
    write_markdown(comparisons, output_dir / "qo_weipreload_poc_summary.md")
    write_chart(rows, output_dir / "qo_weipreload_poc_tflops.svg")
    summary = {
        "schema_version": 1,
        "logical_operation": "[M,4096]@[4096,4096]",
        "m_values": list(M_VALUES),
        "timing": "mean Kineto cat=kernel duration; 5 warmups, 20 iterations",
        "hardware_scope": "DD2 only; 1p5 provenance rejected",
        "variants": VARIANTS,
        "status_rows": status_rows,
        "rows": rows,
        "comparisons": comparisons,
        "provenance_text": provenance,
    }
    (output_dir / "qo_weipreload_poc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
