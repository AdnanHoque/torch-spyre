#!/usr/bin/env python3
"""Validate and summarize a Granite KV SenDNN FP16/scaled-FP8 M sweep."""

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
N = 1024
MODES = ("fp16", "fp8")
EXPECTED_KERNEL_NAMES = {"fp16": "fp16_bmm", "fp8": "fp8_scaled_bmm-Qfp8"}
AIU1_NOMINAL_GHZ = 1.5
AIU1_ACTIVE_CORES = 32
AIU1_CORELETS_PER_CORE = 2
AIU1_MAC_LANES_PER_CORELET = 512
AIU1_FP16_PEAK_TFLOPS = (
    AIU1_ACTIVE_CORES
    * AIU1_CORELETS_PER_CORE
    * AIU1_MAC_LANES_PER_CORELET
    * 2
    * AIU1_NOMINAL_GHZ
    / 1000
)
AIU1_FP8_PEAK_TFLOPS = AIU1_FP16_PEAK_TFLOPS * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def effective_tflops(m: int, mean_us: float) -> float:
    return 2 * m * K * N / (mean_us * 1_000_000)


def read_case(run_root: Path, m: int, mode: str) -> dict:
    result_path = run_root / f"m{m}_k{K}_n{N}" / mode / "profile" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())

    errors = []
    expected_shape = {"M": m, "K": K, "N": N}
    if result.get("mode") != mode:
        errors.append(f"mode={result.get('mode')!r}")
    if result.get("logical_shape") != expected_shape:
        errors.append(f"logical_shape={result.get('logical_shape')!r}")
    if result.get("warmups") != 5:
        errors.append(f"warmups={result.get('warmups')!r}")
    repetitions = result.get("repetitions")
    if repetitions != 20:
        errors.append(f"repetitions={repetitions!r}")
    if not result.get("correctness", {}).get("passed"):
        errors.append("correctness did not pass")
    expected_contract = (
        "FP16 PrimaryInput -> Identity(output FP8) -> BatchScaledMatmul"
        if mode == "fp8"
        else "FP16 PrimaryInput -> BatchMatMul"
    )
    if result.get("graph_contract") != expected_contract:
        errors.append(f"graph_contract={result.get('graph_contract')!r}")
    for section in ("compile_statuses", "lifecycle_statuses"):
        statuses = result.get(section, {})
        if not statuses or any(value != "Status OK" for value in statuses.values()):
            errors.append(f"{section}={statuses!r}")

    kernel_events = result.get("trace_summary", {}).get("kernel_events", [])
    durations = [float(event["duration_us"]) for event in kernel_events]
    names = sorted({str(event["name"]) for event in kernel_events})
    if len(durations) != repetitions:
        errors.append(
            f"kernel_event_count={len(durations)!r}, expected {repetitions!r}"
        )
    expected_name = EXPECTED_KERNEL_NAMES[mode]
    if names != [expected_name]:
        errors.append(f"kernel_names={names!r}, expected {[expected_name]!r}")
    if not durations or any(duration <= 0 for duration in durations):
        errors.append("kernel durations must all be positive")
    if errors:
        raise ValueError(f"{result_path}: " + "; ".join(errors))

    mean_us = statistics.fmean(durations)
    recorded_mean_us = float(result["kernel_mean_us_per_predict"])
    if not math.isclose(mean_us, recorded_mean_us, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"{result_path}: recomputed mean {mean_us} != {recorded_mean_us}"
        )

    correctness = result["correctness"]
    return {
        "M": m,
        "K": K,
        "N": N,
        "mode": "fp8_scaled" if mode == "fp8" else "fp16",
        "graph_contract": result["graph_contract"],
        "warmups": result["warmups"],
        "repetitions": repetitions,
        "kernel_name": expected_name,
        "kernel_event_count": len(durations),
        "kernel_mean_us": mean_us,
        "kernel_p50_us": statistics.median(durations),
        "kernel_p05_us": percentile(durations, 0.05),
        "kernel_p95_us": percentile(durations, 0.95),
        "kernel_stdev_us": statistics.stdev(durations),
        "kernel_min_us": min(durations),
        "kernel_max_us": max(durations),
        "effective_matmul_tflops": effective_tflops(m, mean_us),
        "correctness_passed": True,
        "correctness_relative_l2": float(correctness["relative_l2_error"]),
        "correctness_max_abs": float(correctness["max_abs_error"]),
        "benchmark_sha256": result["benchmark_script"]["sha256"],
        "wrapper_sha256": result["wrapper"]["sha256"],
        "result_path": str(result_path.relative_to(run_root)),
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(rows: list[dict]) -> list[dict]:
    by_key = {(row["M"], row["mode"]): row for row in rows}
    pairs = []
    for m in M_VALUES:
        fp16 = by_key[(m, "fp16")]
        fp8 = by_key[(m, "fp8_scaled")]
        pairs.append(
            {
                "M": m,
                "K": K,
                "N": N,
                "fp16_kernel_mean_us": fp16["kernel_mean_us"],
                "fp8_scaled_kernel_mean_us": fp8["kernel_mean_us"],
                "fp16_effective_matmul_tflops": fp16["effective_matmul_tflops"],
                "fp8_scaled_effective_matmul_tflops": fp8["effective_matmul_tflops"],
                "fp8_over_fp16_speedup": fp16["kernel_mean_us"] / fp8["kernel_mean_us"],
            }
        )
    return pairs


def write_markdown(pairs: list[dict], output_path: Path) -> None:
    lines = [
        "# Granite KV standalone SenDNN M sweep",
        "",
        "Fixed logical operation: `[M, 4096] @ [4096, 1024]`.",
        "",
        "| M | FP16 mean kernel (us) | Scaled FP8 mean kernel (us) "
        "| FP16 effective TFLOP/s | Scaled FP8 effective TFLOP/s "
        "| FP8 / FP16 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['M']} "
            f"| {pair['fp16_kernel_mean_us']:.3f} "
            f"| {pair['fp8_scaled_kernel_mean_us']:.3f} "
            f"| {pair['fp16_effective_matmul_tflops']:.3f} "
            f"| {pair['fp8_scaled_effective_matmul_tflops']:.3f} "
            f"| {pair['fp8_over_fp16_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "TFLOP/s counts only the logical matmul FLOPs (`2*M*K*N`) and "
            "uses the mean Kineto device-kernel duration. The scaled-FP8 "
            "kernel includes on-device FP16-to-FP8 Qfp8 conversion, FP8 "
            "matmul, and both scale-recovery stages. It uses unit-valued "
            "per-row activation scales and per-output-channel weight scales; "
            "scale derivation and activation normalization are not included.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def plot_rows(rows: list[dict], output_path: Path) -> None:
    by_mode = {
        mode: [row for row in rows if row["mode"] == mode]
        for mode in ("fp16", "fp8_scaled")
    }
    styles = {
        "fp16": {
            "label": "FP16 BatchMatMul",
            "color": "#3568b8",
            "marker": "o",
        },
        "fp8_scaled": {
            "label": ("Scaled FP8, fixed unit scales (Qfp8 + matmul + 2x recovery)"),
            "color": "#d35434",
            "marker": "s",
        },
    }
    width = 1280
    height = 760
    left = 104
    right = 45
    top = 150
    bottom = 132
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(row["effective_matmul_tflops"] for row in rows)
    rough_step = maximum / 6
    step = 10 ** math.floor(math.log10(rough_step))
    for candidate in (1, 2, 5, 10):
        y_step = candidate * step
        if y_step >= rough_step:
            break
    y_max = math.ceil(maximum * 1.12 / y_step) * y_step

    def x_position(m: int) -> float:
        return left + math.log2(m) / math.log2(M_VALUES[-1]) * plot_width

    def y_position(value: float) -> float:
        return top + plot_height - value / y_max * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        '<title id="title">FP16 versus scaled FP8 throughput by M</title>',
        '<desc id="description">Effective matmul TFLOP per second for the '
        "Granite 3 8B KV projection at M values from 1 to 2048.</desc>",
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "fill:#202124} .small{font-size:14px;fill:#5f6368} "
        ".tick{font-size:13px;fill:#5f6368} .axis{stroke:#80868b;"
        "stroke-width:1.2} .grid{stroke:#dadce0;stroke-width:1} "
        ".series{fill:none;stroke-width:3;stroke-linejoin:round;"
        "stroke-linecap:round}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<text x="52" y="54" font-size="27" font-weight="600">'
        "Granite 3 8B KV projection: [M, 4096] @ [4096, 1024]</text>",
        '<text x="52" y="84" class="small">Mean Kineto device-kernel time; '
        "5 warmups and 20 measured iterations per point</text>",
    ]

    legend_x = 52
    for mode in ("fp16", "fp8_scaled"):
        style = styles[mode]
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="116" x2="{legend_x + 34}" '
                f'y2="116" stroke="{style["color"]}" stroke-width="3"/>',
                f'<text x="{legend_x + 43}" y="121" font-size="14">'
                f"{escape(style['label'])}</text>",
            ]
        )
        legend_x += 225 if mode == "fp16" else 0

    tick_value = 0.0
    while tick_value <= y_max + 1e-9:
        y = y_position(tick_value)
        parts.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                f'x2="{left + plot_width}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 14}" y="{y + 5:.2f}" '
                f'text-anchor="end">{tick_value:g}</text>',
            ]
        )
        tick_value += y_step

    for m in M_VALUES:
        x = x_position(m)
        parts.extend(
            [
                f'<line class="grid" x1="{x:.2f}" y1="{top}" '
                f'x2="{x:.2f}" y2="{top + plot_height}"/>',
                f'<text class="tick" x="{x:.2f}" '
                f'y="{top + plot_height + 26}" text-anchor="middle">{m}</text>',
            ]
        )

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + plot_height}"/>',
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 66}" '
            'font-size="17" text-anchor="middle">M (log2 scale)</text>',
            f'<text x="28" y="{top + plot_height / 2:.2f}" font-size="17" '
            'text-anchor="middle" transform="rotate(-90 28 '
            f'{top + plot_height / 2:.2f})">Effective matmul throughput '
            "(TFLOP/s)</text>",
        ]
    )

    for mode in ("fp16", "fp8_scaled"):
        style = styles[mode]
        mode_rows = by_mode[mode]
        points = " ".join(
            f"{x_position(row['M']):.2f},"
            f"{y_position(row['effective_matmul_tflops']):.2f}"
            for row in mode_rows
        )
        parts.append(
            f'<polyline class="series" points="{points}" stroke="{style["color"]}"/>'
        )
        for row in mode_rows:
            x = x_position(row["M"])
            y = y_position(row["effective_matmul_tflops"])
            if mode == "fp16":
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.2" '
                    f'fill="#ffffff" stroke="{style["color"]}" '
                    'stroke-width="2.5"/>'
                )
            else:
                parts.append(
                    f'<rect x="{x - 5:.2f}" y="{y - 5:.2f}" width="10" '
                    f'height="10" fill="#ffffff" stroke="{style["color"]}" '
                    'stroke-width="2.5"/>'
                )

    parts.extend(
        [
            f'<text x="{width - right}" y="{height - 27}" class="small" '
            'text-anchor="end">AIU 1.0 nominal arithmetic peaks at '
            f"{AIU1_NOMINAL_GHZ:.1f} GHz: FP16 "
            f"{AIU1_FP16_PEAK_TFLOPS:.1f}, FP8 "
            f"{AIU1_FP8_PEAK_TFLOPS:.1f} TFLOP/s</text>",
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [read_case(args.run_root, m, mode) for m in M_VALUES for mode in MODES]
    benchmark_hashes = {row["benchmark_sha256"] for row in rows}
    wrapper_hashes = {row["wrapper_sha256"] for row in rows}
    if len(benchmark_hashes) != 1 or len(wrapper_hashes) != 1:
        raise ValueError(
            "all cases must use one benchmark and wrapper hash: "
            f"{benchmark_hashes=}, {wrapper_hashes=}"
        )
    pairs = paired_rows(rows)

    write_csv(rows, args.output_dir / "m_sweep_summary.csv")
    summary = {
        "schema_version": 1,
        "measurement": {
            "logical_operation": "[M,4096]@[4096,1024]",
            "m_values": list(M_VALUES),
            "timing": "mean Kineto device-kernel duration",
            "effective_tflops_numerator": "2*M*K*N",
            "fp8_scope": (
                "on-device FP16-to-FP8 Qfp8 + FP8 BatchMatMul + two "
                "scale-recovery stages; unit per-row activation and "
                "per-output-channel weight scales; no scale derivation or "
                "activation normalization"
            ),
            "validation": (
                "24/24 cases: correctness, graph lifecycle, exact shape, "
                "20 positive kernel events, and expected kernel name"
            ),
            "benchmark_sha256": next(iter(benchmark_hashes)),
            "wrapper_sha256": next(iter(wrapper_hashes)),
        },
        "aiu_1_0_nominal_arithmetic_peak": {
            "frequency_ghz": AIU1_NOMINAL_GHZ,
            "active_cores": AIU1_ACTIVE_CORES,
            "corelets_per_core": AIU1_CORELETS_PER_CORE,
            "mac_lanes_per_corelet": AIU1_MAC_LANES_PER_CORELET,
            "fp16_tflops": AIU1_FP16_PEAK_TFLOPS,
            "fp8_tflops": AIU1_FP8_PEAK_TFLOPS,
        },
        "rows": rows,
        "pairs": pairs,
    }
    (args.output_dir / "m_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(pairs, args.output_dir / "m_sweep_summary.md")
    plot_rows(rows, args.output_dir / "m_sweep_tflops.svg")
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
