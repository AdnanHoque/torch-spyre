#!/usr/bin/env python3
"""Validate and summarize all Granite 3 8B linear-layer SenDNN M sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from xml.sax.saxutils import escape


M_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
MODES = ("fp16", "fp8")
EXPECTED_KERNEL_NAMES = {"fp16": "fp16_bmm", "fp8": "fp8_scaled_bmm-Qfp8"}
EXPECTED_BENCHMARK_SHA256 = (
    "3536cfcb912779e2f04013df04d534e0c11b2d38a43152ca235e21b713bbd046"
)
EXPECTED_WRAPPER_SHA256 = (
    "a5873a426ba376ca2f97172567d09e6402923f3281a40ecfb53e7949bdb174f9"
)
EXPECTED_RUNNER_SHA256 = (
    "7c69ee115d521b184c5f8c974232c4c5c30fbe9105e98f2ccb6cfa1fda9ae5d1"
)
EXPECTED_TORCH = "2.10.0+aiu.kineto.1.1.1"
EXPECTED_TORCH_SENDNN = "1.3.0+main.1.1bef083.0"
SHAPES = {
    "kv": {
        "projection": "K/V",
        "K": 4096,
        "N": 1024,
        "operation": "[M,4096]@[4096,1024]",
    },
    "qo": {
        "projection": "Q/O",
        "K": 4096,
        "N": 4096,
        "operation": "[M,4096]@[4096,4096]",
    },
    "mlp_up": {
        "projection": "gate/up",
        "K": 4096,
        "N": 12800,
        "operation": "[M,4096]@[4096,12800]",
    },
    "mlp_down": {
        "projection": "down",
        "K": 12800,
        "N": 4096,
        "operation": "[M,12800]@[12800,4096]",
    },
}
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
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="one local run root for each of kv, qo, mlp_up, and mlp_down",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_run_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or label not in SHAPES or not raw_path:
            raise ValueError(f"invalid --run-root {value!r}")
        if label in roots:
            raise ValueError(f"duplicate --run-root label {label!r}")
        roots[label] = Path(raw_path).resolve()
    if set(roots) != set(SHAPES):
        raise ValueError(f"need run roots for {sorted(SHAPES)}, got {sorted(roots)}")
    return roots


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def effective_tflops(m: int, k: int, n: int, mean_us: float) -> float:
    return 2 * m * k * n / (mean_us * 1_000_000)


def parse_provenance(run_root: Path, label: str) -> dict:
    path = run_root / "provenance.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw_text = path.read_text(errors="replace")
    lines = raw_text.splitlines()
    values: dict[str, str] = {}
    if len(lines) >= 2:
        values["timestamp"] = lines[0]
        values["hostname"] = lines[1]
    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value

    shape = SHAPES[label]
    expected_values = {
        "shape_label": label,
        "logical_shape": shape["operation"],
        "m_values": ",".join(str(m) for m in M_VALUES),
        "warmups": "5",
        "repetitions": "20",
        "failure_count": "0",
    }
    errors = [
        f"{key}={values.get(key)!r}, expected {expected!r}"
        for key, expected in expected_values.items()
        if values.get(key) != expected
    ]
    expected_hashes = {
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
    }
    for key, expected in expected_hashes.items():
        actual = values.get(key, "").split(maxsplit=1)[0]
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    lowered = raw_text.lower()
    if "ibm-senlib-dd2-" not in lowered:
        errors.append("missing ibm-senlib-dd2 package evidence")
    if re.search(r"(sen|aiu)[_-]?1p5|1p5", lowered):
        errors.append("provenance contains forbidden 1p5 architecture evidence")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))

    package_lines = sorted(
        line
        for line in lines
        if line.startswith(("ibm-deeptools-", "ibm-flex-", "ibm-senlib-"))
    )
    version_lines = {
        prefix.rstrip(): next(
            (line for line in lines if line.startswith(prefix)),
            "",
        )
        for prefix in ("python ", "torch ", "torch_sendnn ", "sendnn ")
    }
    return {
        "path": f"raw/{label}/provenance.txt",
        "timestamp": values["timestamp"],
        "hostname": values["hostname"],
        "shape_label": label,
        "logical_shape": values["logical_shape"],
        "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
        "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
        "backend_packages": package_lines,
        "software_lines": version_lines,
    }


def validate_status(run_root: Path) -> None:
    path = run_root / "status.tsv"
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text().splitlines()
    if not lines or lines[0] != "M\tmode\texit_status\tresult":
        raise ValueError(f"{path}: unexpected header")
    if len(lines) != 1 + len(M_VALUES) * len(MODES):
        raise ValueError(f"{path}: expected 24 cases, found {len(lines) - 1}")
    failures = []
    cases = set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4 or fields[2] != "0":
            failures.append(line)
            continue
        cases.add((int(fields[0]), fields[1]))
    if failures:
        raise ValueError(f"{path}: failed cases: {failures}")
    expected_cases = {(m, mode) for m in M_VALUES for mode in MODES}
    if cases != expected_cases:
        raise ValueError(
            f"{path}: case set mismatch: missing={sorted(expected_cases - cases)}, "
            f"extra={sorted(cases - expected_cases)}"
        )


def read_case(
    run_root: Path,
    label: str,
    m: int,
    mode: str,
) -> dict:
    shape = SHAPES[label]
    k = int(shape["K"])
    n = int(shape["N"])
    result_path = run_root / f"m{m}_k{k}_n{n}" / mode / "profile" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    result = json.loads(result_path.read_text())

    errors = []
    expected_shape = {"M": m, "K": k, "N": n}
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
    categories = sorted({str(event["category"]) for event in kernel_events})
    if len(durations) != repetitions:
        errors.append(
            f"kernel_event_count={len(durations)!r}, expected {repetitions!r}"
        )
    expected_name = EXPECTED_KERNEL_NAMES[mode]
    if names != [expected_name]:
        errors.append(f"kernel_names={names!r}, expected {[expected_name]!r}")
    if categories != ["kernel"]:
        errors.append(f"kernel_categories={categories!r}")
    if not durations or any(
        not math.isfinite(duration) or duration <= 0 for duration in durations
    ):
        errors.append("kernel durations must all be positive")
    if result.get("kernel_event_count") != repetitions:
        errors.append(
            f"top-level kernel_event_count={result.get('kernel_event_count')!r}"
        )

    benchmark_sha = result.get("benchmark_script", {}).get("sha256")
    wrapper_sha = result.get("wrapper", {}).get("sha256")
    if benchmark_sha != EXPECTED_BENCHMARK_SHA256:
        errors.append(f"benchmark_sha256={benchmark_sha!r}")
    if wrapper_sha != EXPECTED_WRAPPER_SHA256:
        errors.append(f"wrapper_sha256={wrapper_sha!r}")
    software = result.get("software", {})
    if software.get("torch") != EXPECTED_TORCH:
        errors.append(f"torch={software.get('torch')!r}")
    if software.get("torch_sendnn_distribution") != EXPECTED_TORCH_SENDNN:
        errors.append(f"torch_sendnn={software.get('torch_sendnn_distribution')!r}")

    correctness = result.get("correctness", {})
    expected_policy = (
        {"rtol": 0.08, "atol": 0.5, "relative_l2_limit": 0.08}
        if mode == "fp8"
        else {"rtol": 0.02, "atol": 0.25, "relative_l2_limit": 0.06}
    )
    for key, expected in expected_policy.items():
        if correctness.get(key) != expected:
            errors.append(f"correctness.{key}={correctness.get(key)!r}")
    if errors:
        raise ValueError(f"{result_path}: " + "; ".join(errors))

    mean_us = statistics.fmean(durations)
    recorded_mean_us = float(result["kernel_mean_us_per_predict"])
    if not math.isclose(mean_us, recorded_mean_us, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            f"{result_path}: recomputed mean {mean_us} != {recorded_mean_us}"
        )
    if not math.isclose(
        sum(durations),
        float(result["kernel_total_us"]),
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"{result_path}: kernel_total_us mismatch")

    return {
        "shape_label": label,
        "projection": shape["projection"],
        "M": m,
        "K": k,
        "N": n,
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
        "effective_matmul_tflops": effective_tflops(m, k, n, mean_us),
        "correctness_passed": True,
        "correctness_relative_l2": float(correctness["relative_l2_error"]),
        "correctness_max_abs": float(correctness["max_abs_error"]),
        "benchmark_sha256": benchmark_sha,
        "wrapper_sha256": wrapper_sha,
        "torch": result["software"]["torch"],
        "torch_sendnn": result["software"]["torch_sendnn_distribution"],
        "result_path": str(result_path.relative_to(run_root)),
    }


def paired_rows(rows: list[dict]) -> list[dict]:
    by_key = {(row["shape_label"], row["M"], row["mode"]): row for row in rows}
    pairs = []
    for label, shape in SHAPES.items():
        for m in M_VALUES:
            fp16 = by_key[(label, m, "fp16")]
            fp8 = by_key[(label, m, "fp8_scaled")]
            pairs.append(
                {
                    "shape_label": label,
                    "projection": shape["projection"],
                    "M": m,
                    "K": shape["K"],
                    "N": shape["N"],
                    "fp16_kernel_mean_us": fp16["kernel_mean_us"],
                    "fp8_scaled_kernel_mean_us": fp8["kernel_mean_us"],
                    "fp16_effective_matmul_tflops": fp16["effective_matmul_tflops"],
                    "fp8_scaled_effective_matmul_tflops": fp8[
                        "effective_matmul_tflops"
                    ],
                    "fp8_over_fp16_speedup": (
                        fp16["kernel_mean_us"] / fp8["kernel_mean_us"]
                    ),
                }
            )
    return pairs


def write_csv(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(pairs: list[dict], output_path: Path) -> None:
    lines = [
        "# Granite 3 8B TP1 linear-layer standalone SenDNN M sweeps",
        "",
        "All unique single-device transformer-block linear projection shapes "
        "were measured.",
        "",
        "| Projection | Logical operation |",
        "|---|---|",
    ]
    for shape in SHAPES.values():
        lines.append(f"| {shape['projection']} | `{shape['operation']}` |")

    for label, shape in SHAPES.items():
        lines.extend(
            [
                "",
                f"## {shape['projection']}: `{shape['operation']}`",
                "",
                "| M | FP16 mean kernel (us) | Scaled FP8 mean kernel (us) "
                "| FP16 effective TFLOP/s | Scaled FP8 effective TFLOP/s "
                "| FP8 / FP16 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for pair in pairs:
            if pair["shape_label"] != label:
                continue
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
            "TFLOP/s counts the logical matmul FLOPs (`2*M*K*N`) and uses "
            "the mean Kineto device-kernel duration. The scaled-FP8 kernel "
            "includes on-device FP16-to-FP8 Qfp8 conversion, relayout, FP8 "
            "matmul, and two scale-recovery stages. It uses fixed unit-valued "
            "per-row activation scales and per-output-channel weight scales; "
            "scale derivation and activation normalization are not included.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines))


def rounded_axis_max(maximum: float) -> tuple[float, float]:
    rough_step = max(maximum / 5, 1e-9)
    magnitude = 10 ** math.floor(math.log10(rough_step))
    for multiplier in (1, 2, 5, 10):
        step = multiplier * magnitude
        if step >= rough_step:
            break
    return math.ceil(maximum * 1.10 / step) * step, step


def chart_styles() -> dict[str, dict[str, str]]:
    return {
        "fp16": {
            "label": "FP16 BatchMatMul",
            "color": "#3568b8",
            "marker": "circle",
        },
        "fp8_scaled": {
            "label": "Scaled FP8 pipeline (fixed unit per-axis scales)",
            "color": "#d35434",
            "marker": "square",
        },
    }


def add_panel(
    parts: list[str],
    panel_rows: list[dict],
    shape: dict,
    x0: float,
    y0: float,
    width: float,
    height: float,
    show_title: bool = True,
) -> None:
    left = x0 + 65
    right = x0 + width - 18
    top = y0 + (53 if show_title else 22)
    bottom = y0 + height - 58
    plot_width = right - left
    plot_height = bottom - top
    maximum = max(row["effective_matmul_tflops"] for row in panel_rows)
    y_max, y_step = rounded_axis_max(maximum)

    def x_position(m: int) -> float:
        return left + math.log2(m) / math.log2(M_VALUES[-1]) * plot_width

    def y_position(value: float) -> float:
        return bottom - value / y_max * plot_height

    if show_title:
        parts.append(
            f'<text x="{x0 + 12:.2f}" y="{y0 + 25:.2f}" font-size="18" '
            f'font-weight="600">{escape(str(shape["projection"]))}: '
            f"{escape(str(shape['operation']))}</text>"
        )

    tick = 0.0
    while tick <= y_max + 1e-9:
        y = y_position(tick)
        parts.extend(
            [
                f'<line class="grid" x1="{left:.2f}" y1="{y:.2f}" '
                f'x2="{right:.2f}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 9:.2f}" y="{y + 4:.2f}" '
                f'text-anchor="end">{tick:g}</text>',
            ]
        )
        tick += y_step

    for index, m in enumerate(M_VALUES):
        x = x_position(m)
        parts.append(
            f'<line class="grid" x1="{x:.2f}" y1="{top:.2f}" '
            f'x2="{x:.2f}" y2="{bottom:.2f}"/>'
        )
        if index % 2 == 0 or m == M_VALUES[-1]:
            parts.append(
                f'<text class="tick" x="{x:.2f}" y="{bottom + 20:.2f}" '
                f'text-anchor="middle">{m}</text>'
            )

    parts.extend(
        [
            f'<line class="axis" x1="{left:.2f}" y1="{bottom:.2f}" '
            f'x2="{right:.2f}" y2="{bottom:.2f}"/>',
            f'<line class="axis" x1="{left:.2f}" y1="{top:.2f}" '
            f'x2="{left:.2f}" y2="{bottom:.2f}"/>',
            f'<text x="{(left + right) / 2:.2f}" y="{y0 + height - 14:.2f}" '
            'class="small" text-anchor="middle">M (log2 scale)</text>',
            f'<text x="{x0 + 18:.2f}" y="{(top + bottom) / 2:.2f}" '
            'class="small" text-anchor="middle" '
            f'transform="rotate(-90 {x0 + 18:.2f} '
            f'{(top + bottom) / 2:.2f})">Effective TFLOP/s</text>',
        ]
    )

    styles = chart_styles()
    for mode in ("fp16", "fp8_scaled"):
        mode_rows = [row for row in panel_rows if row["mode"] == mode]
        style = styles[mode]
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
            if style["marker"] == "circle":
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" '
                    f'fill="#fff" stroke="{style["color"]}" stroke-width="2"/>'
                )
            else:
                parts.append(
                    f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" '
                    f'height="8" fill="#fff" stroke="{style["color"]}" '
                    'stroke-width="2"/>'
                )


def svg_header(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="description">{escape(description)}</desc>',
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "fill:#202124}.small{font-size:13px;fill:#5f6368}"
        ".tick{font-size:11px;fill:#5f6368}.axis{stroke:#80868b;"
        "stroke-width:1.1}.grid{stroke:#e4e7eb;stroke-width:1}"
        ".series{fill:none;stroke-width:2.7;stroke-linejoin:round;"
        "stroke-linecap:round}",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
    ]


def write_combined_chart(rows: list[dict], output_path: Path) -> None:
    width, height = 1500, 1080
    parts = svg_header(
        width,
        height,
        "Granite 3 8B FP16 versus scaled FP8 throughput",
        "Four panels show effective TFLOP per second over M for all unique "
        "single-device Granite linear projection shapes. Each panel has its "
        "own vertical scale.",
    )
    parts.extend(
        [
            '<text x="55" y="50" font-size="28" font-weight="600">'
            "Granite 3 8B TP1 linear projections: FP16 vs scaled FP8</text>",
            '<text x="55" y="80" class="small">Mean Kineto device-kernel '
            "time; 5 warmups and 20 measured iterations; "
            "each panel uses its own y-axis scale</text>",
        ]
    )
    styles = chart_styles()
    legend_x = 55
    for mode in ("fp16", "fp8_scaled"):
        style = styles[mode]
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="110" x2="{legend_x + 34}" '
                f'y2="110" stroke="{style["color"]}" stroke-width="3"/>',
                f'<text x="{legend_x + 43}" y="115" font-size="14">'
                f"{escape(style['label'])}</text>",
            ]
        )
        legend_x += 215 if mode == "fp16" else 0

    panel_width, panel_height = 705, 425
    origins = ((40, 140), (755, 140), (40, 575), (755, 575))
    for (label, shape), (x0, y0) in zip(SHAPES.items(), origins):
        panel_rows = [row for row in rows if row["shape_label"] == label]
        add_panel(parts, panel_rows, shape, x0, y0, panel_width, panel_height)

    parts.extend(
        [
            '<text x="1460" y="1050" class="small" text-anchor="end">'
            "AIU 1.0 raw PT arithmetic roofs at 1.5 GHz "
            "(not scaled-pipeline roofs): FP16 "
            f"{AIU1_FP16_PEAK_TFLOPS:.1f}, FP8 "
            f"{AIU1_FP8_PEAK_TFLOPS:.1f} TFLOP/s</text>",
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts) + "\n")


def write_individual_charts(rows: list[dict], output_dir: Path) -> None:
    for label, shape in SHAPES.items():
        width, height = 920, 650
        parts = svg_header(
            width,
            height,
            f"{shape['projection']} FP16 versus scaled FP8 throughput",
            f"Effective TFLOP per second over M for {shape['operation']}.",
        )
        parts.extend(
            [
                f'<text x="42" y="43" font-size="24" font-weight="600">'
                f"Granite 3 8B {escape(str(shape['projection']))}: "
                f"{escape(str(shape['operation']))}</text>",
                '<text x="42" y="71" class="small">Mean Kineto '
                "device-kernel time; 5 warmups and 20 iterations</text>",
            ]
        )
        styles = chart_styles()
        legend_x = 42
        for mode in ("fp16", "fp8_scaled"):
            style = styles[mode]
            parts.extend(
                [
                    f'<line x1="{legend_x}" y1="99" x2="{legend_x + 30}" '
                    f'y2="99" stroke="{style["color"]}" stroke-width="3"/>',
                    f'<text x="{legend_x + 38}" y="104" font-size="13">'
                    f"{escape(style['label'])}</text>",
                ]
            )
            legend_x += 200 if mode == "fp16" else 0
        panel_rows = [row for row in rows if row["shape_label"] == label]
        add_panel(parts, panel_rows, shape, 25, 115, 870, 485, show_title=False)
        parts.extend(
            [
                '<text x="890" y="633" class="small" text-anchor="end">'
                "Effective throughput = 2*M*K*N / kernel time</text>",
                "</svg>",
            ]
        )
        (output_dir / f"{label}_m_sweep_tflops.svg").write_text("\n".join(parts) + "\n")


def main() -> None:
    args = parse_args()
    run_roots = parse_run_roots(args.run_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {}
    rows = []
    for label in SHAPES:
        run_root = run_roots[label]
        validate_status(run_root)
        provenance[label] = parse_provenance(run_root, label)
        rows.extend(
            read_case(run_root, label, m, mode) for m in M_VALUES for mode in MODES
        )

    benchmark_hashes = {row["benchmark_sha256"] for row in rows}
    wrapper_hashes = {row["wrapper_sha256"] for row in rows}
    torch_versions = {row["torch"] for row in rows}
    torch_sendnn_versions = {row["torch_sendnn"] for row in rows}
    if benchmark_hashes != {EXPECTED_BENCHMARK_SHA256} or wrapper_hashes != {
        EXPECTED_WRAPPER_SHA256
    }:
        raise ValueError(
            "all cases must use one benchmark and wrapper hash: "
            f"{benchmark_hashes=}, {wrapper_hashes=}"
        )
    if torch_versions != {EXPECTED_TORCH} or torch_sendnn_versions != {
        EXPECTED_TORCH_SENDNN
    }:
        raise ValueError(
            "all cases must use one Torch and torch_sendnn version: "
            f"{torch_versions=}, {torch_sendnn_versions=}"
        )
    backend_package_sets = {
        tuple(item["backend_packages"]) for item in provenance.values()
    }
    if len(backend_package_sets) != 1:
        raise ValueError(
            "all run roots must use one DeepTools/Flex/SenLib package set: "
            f"{backend_package_sets!r}"
        )

    pairs = paired_rows(rows)
    write_csv(rows, args.output_dir / "linear_shape_m_sweep_rows.csv")
    write_csv(pairs, args.output_dir / "linear_shape_m_sweep_pairs.csv")
    write_markdown(pairs, args.output_dir / "linear_shape_m_sweep_summary.md")
    write_combined_chart(
        rows,
        args.output_dir / "linear_shape_m_sweep_tflops.svg",
    )
    write_individual_charts(rows, args.output_dir)

    summary = {
        "schema_version": 1,
        "measurement": {
            "shapes": SHAPES,
            "m_values": list(M_VALUES),
            "timing": "mean Kineto device-kernel duration",
            "effective_tflops_numerator": "2*M*K*N",
            "fp8_scope": (
                "on-device FP16-to-FP8 Qfp8 + relayout + FP8 BatchMatMul "
                "+ two scale-recovery stages; fixed unit per-row activation "
                "and per-output-channel weight scales; no scale derivation "
                "or activation normalization"
            ),
            "validation": (
                "96/96 cases: correctness, graph lifecycle, exact shape, "
                "20 positive kernel events, and expected kernel name"
            ),
            "hardware_scope": (
                "AIU 1.0 only, proven by ibm-senlib-dd2 package provenance; "
                "any 1p5 provenance is rejected"
            ),
            "benchmark_sha256": EXPECTED_BENCHMARK_SHA256,
            "wrapper_sha256": EXPECTED_WRAPPER_SHA256,
            "runner_sha256": EXPECTED_RUNNER_SHA256,
            "torch": EXPECTED_TORCH,
            "torch_sendnn": EXPECTED_TORCH_SENDNN,
        },
        "provenance": provenance,
        "aiu_1_0_nominal_arithmetic_peak": {
            "scope": (
                "raw PT arithmetic roof; not directly comparable to the "
                "measured fixed-scale FP8 pipeline"
            ),
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
    (args.output_dir / "linear_shape_m_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
