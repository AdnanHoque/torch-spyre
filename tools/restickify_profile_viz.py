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

"""Render a compact SVG/HTML profile summary for restickify counter runs.

The report is intentionally dependency-free so it can run inside lean Spyre pods.
It consumes the JSONL rows written by the restickify probes and produces an
Nsight-style visual summary: hardware context, compiler locality metrics,
runtime timing, AIU SMI bandwidth counters, and restickify source attribution.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


WIDTH = 1480
MARGIN = 28
CARD = "#151a22"
CARD_2 = "#11161d"
GRID = "#2a3340"
TEXT = "#e7edf6"
MUTED = "#9aa7b5"
GREEN = "#49d17d"
BLUE = "#58a6ff"
CYAN = "#51d1e6"
YELLOW = "#f4c542"
ORANGE = "#ff9f43"
RED = "#ff6b6b"
PURPLE = "#b58cff"

MODE_COLORS = {
    "baseline": BLUE,
    "stage3b": GREEN,
}

SOURCE_COLORS = {
    "in_graph_computed": GREEN,
    "graph_input_or_weight": ORANGE,
    "constant_or_extern": PURPLE,
    "mutation_target": RED,
    "unknown": MUTED,
}


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _num(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_int(value: Any) -> str:
    return f"{int(_num(value)):,}"


def _fmt_float(value: Any, digits: int = 3) -> str:
    return f"{_num(value):.{digits}f}"


def _fmt_bytes(value: Any) -> str:
    value_f = _num(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(value_f) < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(value_f):,} {unit}"
            return f"{value_f:.2f} {unit}"
        value_f /= 1024.0
    return f"{value_f:.2f} GiB"


def _fmt_ms(value: Any) -> str:
    return f"{_num(value):.4f} ms"


def _fmt_pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str = "none",
    rx: float = 10,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
        f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}"/>'
    )


def _text(
    x: float,
    y: float,
    value: Any,
    size: int = 16,
    fill: str = TEXT,
    weight: int = 400,
    anchor: str = "start",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{_esc(value)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, color: str = GRID, width: float = 1.0) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{width:.1f}"/>'
    )


def _short(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _bar(
    x: float,
    y: float,
    w: float,
    h: float,
    value: float,
    max_value: float,
    color: str,
    label: str,
    value_label: str,
) -> list[str]:
    max_value = max(max_value, 1e-9)
    fill_w = max(0.0, min(w, w * value / max_value))
    return [
        _rect(x, y, w, h, "#202936", rx=5),
        _rect(x, y, fill_w, h, color, rx=5),
        _text(x, y - 6, label, 12, MUTED),
        _text(x + w + 10, y + h - 3, value_label, 12, TEXT),
    ]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> list[tuple[tuple[str, int], list[dict[str, Any]]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("case", "unknown")), int(_num(row.get("size"))))].append(row)
    ordered = []
    for key in sorted(grouped, key=lambda item: (item[0], item[1])):
        group = sorted(grouped[key], key=lambda row: (row.get("mode") != "baseline", row.get("mode", "")))
        ordered.append((key, group))
    return ordered


def _source_bytes(row: dict[str, Any]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    entries = row.get("ring_entries") or row.get("entries") or []
    for entry in entries:
        kind = entry.get("source_kind") or entry.get("source") or "unknown"
        totals[str(kind)] += _num(entry.get("bytes_moved") or entry.get("num_bytes") or entry.get("bytes"))
    return dict(totals)


def _paired_summary(group: list[dict[str, Any]]) -> tuple[str, str]:
    by_mode = {str(row.get("mode")): row for row in group}
    baseline = by_mode.get("baseline")
    stage3b = by_mode.get("stage3b")
    if not baseline or not stage3b:
        return "paired comparison unavailable", MUTED
    base_hops = _num(baseline.get("ring_total_byte_hops"))
    stage_hops = _num(stage3b.get("ring_total_byte_hops"))
    base_ms = _num(baseline.get("median_ms"))
    stage_ms = _num(stage3b.get("median_ms"))
    hop_reduction = 0.0 if base_hops == 0 else 1.0 - stage_hops / base_hops
    speedup = 0.0 if stage_ms == 0 else base_ms / stage_ms
    color = GREEN if hop_reduction > 0.5 and speedup >= 1.0 else YELLOW
    return f"byte-hop reduction {_fmt_pct(hop_reduction)} | speedup {speedup:.4f}x", color


def _draw_hardware_panel(x: float, y: float, w: float, h: float) -> list[str]:
    out = [_rect(x, y, w, h, CARD, "#263241", rx=14)]
    out.append(_text(x + 18, y + 30, "AIU Memory/Fabric Context", 18, TEXT, 700))
    out.append(_text(x + 18, y + 55, "32 cores on RIU data BiRing; HBM/off-chip and cross-core LX-LX both use ring-facing data paths", 13, MUTED))

    ring_x = x + 32
    ring_y = y + 82
    ring_w = 390
    ring_h = 118
    out.append(
        f'<rect x="{ring_x:.1f}" y="{ring_y:.1f}" width="{ring_w:.1f}" height="{ring_h:.1f}" '
        f'rx="54" fill="none" stroke="{CYAN}" stroke-width="6"/>'
    )
    out.append(
        f'<rect x="{ring_x + 12:.1f}" y="{ring_y + 12:.1f}" width="{ring_w - 24:.1f}" height="{ring_h - 24:.1f}" '
        f'rx="44" fill="none" stroke="{BLUE}" stroke-width="2" stroke-dasharray="7 6" opacity="0.8"/>'
    )
    out.append(_text(ring_x + ring_w / 2, ring_y + 27, "RIU data BiRing", 14, CYAN, 700, "middle"))
    out.append(_text(ring_x + ring_w / 2, ring_y + 47, "166 GB/s per direction | 333 GB/s aggregate model", 11, MUTED, 400, "middle"))

    core_positions = [
        (ring_x + 38, ring_y + 29, "C0"),
        (ring_x + 110, ring_y + 80, "C1"),
        (ring_x + 194, ring_y + 92, "C16"),
        (ring_x + 288, ring_y + 80, "C30"),
        (ring_x + 342, ring_y + 29, "C31"),
    ]
    for cx, cy, label in core_positions:
        out.append(_rect(cx - 22, cy - 14, 44, 28, "#222c39", "#3f526b", rx=6))
        out.append(_text(cx, cy + 5, label, 12, TEXT, 700, "middle"))

    hbm_x = x + w - 258
    hbm_y = y + 78
    out.append(_rect(hbm_x, hbm_y, 216, 70, "#2a1d13", ORANGE, rx=12))
    out.append(_text(hbm_x + 108, hbm_y + 29, "HBM / off-chip", 16, ORANGE, 700, "middle"))
    out.append(_text(hbm_x + 108, hbm_y + 51, "device memory traffic", 12, MUTED, 400, "middle"))

    lx_x = x + w - 258
    lx_y = y + 164
    out.append(_rect(lx_x, lx_y, 216, 54, "#172819", GREEN, rx=12))
    out.append(_text(lx_x + 108, lx_y + 23, "Per-core LX", 15, GREEN, 700, "middle"))
    out.append(_text(lx_x + 108, lx_y + 42, "private scratchpad ownership", 11, MUTED, 400, "middle"))

    out.append(_line(ring_x + ring_w, ring_y + 58, hbm_x, hbm_y + 35, ORANGE, 2))
    out.append(_line(ring_x + ring_w, ring_y + 92, lx_x, lx_y + 27, GREEN, 2))
    out.append(_text(x + 456, y + 107, "What the visual compares", 14, TEXT, 700))
    out.append(_text(x + 456, y + 130, "1. compiler-modeled ring byte-hops", 12, MUTED))
    out.append(_text(x + 456, y + 150, "2. measured kernel loop latency", 12, MUTED))
    out.append(_text(x + 456, y + 170, "3. aiu-smi device read/write counters", 12, MUTED))
    out.append(_text(x + 456, y + 190, "4. source class: in-graph vs graph-input/weight", 12, MUTED))
    return out


def _draw_group(x: float, y: float, w: float, group_key: tuple[str, int], group: list[dict[str, Any]]) -> list[str]:
    case, size = group_key
    h = 254
    out = [_rect(x, y, w, h, CARD_2, "#263241", rx=14)]
    out.append(_text(x + 18, y + 30, f"{case} | size {size}", 18, TEXT, 700))
    summary, summary_color = _paired_summary(group)
    out.append(_text(x + 18, y + 52, summary, 13, summary_color, 700))

    max_hops = max((_num(row.get("ring_total_byte_hops")) for row in group), default=1.0)
    max_ms = max((_num(row.get("median_ms")) for row in group), default=1.0)
    max_bw = max(
        (
            _num(row.get("aiusmi_avg_nonzero_rdmem_GiB_per_s"))
            + _num(row.get("aiusmi_avg_nonzero_wrmem_GiB_per_s"))
            for row in group
        ),
        default=1.0,
    )

    col1 = x + 18
    col2 = x + 390
    col3 = x + 750
    col4 = x + 1090
    top = y + 83
    out.append(_text(col1, top - 18, "Compiler locality", 13, TEXT, 700))
    out.append(_text(col2, top - 18, "Timed loop", 13, TEXT, 700))
    out.append(_text(col3, top - 18, "AIU SMI bandwidth", 13, TEXT, 700))
    out.append(_text(col4, top - 18, "Restickify source bytes", 13, TEXT, 700))

    for index, row in enumerate(group):
        mode = str(row.get("mode", f"row{index}"))
        color = MODE_COLORS.get(mode, MUTED)
        row_y = top + index * 74
        out.append(_text(col1, row_y + 11, mode, 13, color, 700))
        out.extend(
            _bar(
                col1 + 92,
                row_y,
                220,
                15,
                _num(row.get("ring_total_byte_hops")),
                max_hops,
                color,
                "",
                _fmt_int(row.get("ring_total_byte_hops")),
            )
        )
        out.append(
            _text(
                col1 + 92,
                row_y + 35,
                f"{_fmt_bytes(row.get('total_bytes'))} moved | avg hops {_fmt_float(row.get('ring_avg_hops'))} | max {_fmt_int(row.get('ring_max_hops'))}",
                11,
                MUTED,
            )
        )

        ms = _num(row.get("median_ms"))
        p10 = _num(row.get("p10_ms"))
        p90 = _num(row.get("p90_ms"))
        bar_x = col2 + 10
        bar_w = 215
        out.extend(_bar(bar_x, row_y, bar_w, 15, ms, max_ms, color, "", _fmt_ms(ms)))
        if max_ms > 0:
            p10_x = bar_x + bar_w * p10 / max_ms
            p90_x = bar_x + bar_w * p90 / max_ms
            out.append(_line(p10_x, row_y - 4, p10_x, row_y + 19, YELLOW, 1.5))
            out.append(_line(p90_x, row_y - 4, p90_x, row_y + 19, YELLOW, 1.5))
        out.append(_text(bar_x, row_y + 35, f"p10 {_fmt_ms(p10)} | p90 {_fmt_ms(p90)}", 11, MUTED))

        rd = _num(row.get("aiusmi_avg_nonzero_rdmem_GiB_per_s"))
        wr = _num(row.get("aiusmi_avg_nonzero_wrmem_GiB_per_s"))
        peak_rd = _num(row.get("aiusmi_peak_rdmem_GiB_per_s"))
        peak_wr = _num(row.get("aiusmi_peak_wrmem_GiB_per_s"))
        bw_x = col3 + 10
        bw_w = 210
        denom = max_bw
        rd_w = bw_w * rd / max(denom, 1e-9)
        wr_w = bw_w * wr / max(denom, 1e-9)
        out.append(_rect(bw_x, row_y, bw_w, 15, "#202936", rx=5))
        out.append(_rect(bw_x, row_y, rd_w, 15, BLUE, rx=5))
        out.append(_rect(bw_x + rd_w, row_y, wr_w, 15, ORANGE, rx=5))
        out.append(_text(bw_x + bw_w + 10, row_y + 12, f"{rd + wr:.1f} GiB/s", 12, TEXT))
        out.append(_text(bw_x, row_y + 35, f"rd {rd:.1f} avg/{peak_rd:.1f} peak | wr {wr:.1f} avg/{peak_wr:.1f} peak", 11, MUTED))

        src = _source_bytes(row)
        total_src = sum(src.values()) or _num(row.get("total_bytes")) or 1.0
        sx = col4 + 8
        sw = 256
        cursor = sx
        out.append(_rect(sx, row_y, sw, 15, "#202936", rx=5))
        for kind, bytes_value in sorted(src.items(), key=lambda item: item[0]):
            frac_w = sw * bytes_value / max(total_src, 1e-9)
            out.append(_rect(cursor, row_y, frac_w, 15, SOURCE_COLORS.get(kind, MUTED), rx=3))
            cursor += frac_w
        label_parts = []
        for kind, bytes_value in sorted(src.items(), key=lambda item: item[0]):
            short = kind.replace("_or_weight", "/weight").replace("_computed", "").replace("_", " ")
            label_parts.append(f"{short}: {_fmt_bytes(bytes_value)}")
        out.append(_text(sx, row_y + 35, " | ".join(label_parts) if label_parts else "no source rows", 11, MUTED))

    legend_y = y + h - 18
    out.append(_rect(x + 18, legend_y - 11, 10, 10, BLUE, rx=2))
    out.append(_text(x + 34, legend_y - 2, "read or baseline", 11, MUTED))
    out.append(_rect(x + 150, legend_y - 11, 10, 10, GREEN, rx=2))
    out.append(_text(x + 166, legend_y - 2, "Stage 3B or in-graph", 11, MUTED))
    out.append(_rect(x + 332, legend_y - 11, 10, 10, ORANGE, rx=2))
    out.append(_text(x + 348, legend_y - 2, "write or graph-input/weight", 11, MUTED))
    return out


def render_svg(rows: list[dict[str, Any]], title: str = "Spyre Restickify Counter Report") -> str:
    groups = _group_rows(rows)
    height = 420 + 278 * max(1, len(groups))
    display_title = _short(title, 62)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}">',
        "<defs>",
        "<style>",
        "text { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }",
        "</style>",
        "</defs>",
        _rect(0, 0, WIDTH, height, "#0b0f14", rx=0),
        _text(MARGIN, 46, display_title, 26, TEXT, 800),
        _text(MARGIN, 72, "Compiler ring locality + AIU SMI counters + source attribution", 14, MUTED),
    ]

    ok_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    total_hops = sum(_num(row.get("ring_total_byte_hops")) for row in ok_rows)
    total_bytes = sum(_num(row.get("total_bytes")) for row in ok_rows)
    max_median = max((_num(row.get("median_ms")) for row in ok_rows), default=0.0)
    out.append(_rect(WIDTH - 462, 24, 410, 64, CARD, "#263241", rx=12))
    out.append(_text(WIDTH - 442, 50, f"{len(ok_rows)} rows", 14, TEXT, 700))
    out.append(_text(WIDTH - 330, 50, f"{_fmt_bytes(total_bytes)} restickified", 14, TEXT, 700))
    out.append(_text(WIDTH - 442, 74, f"{_fmt_int(total_hops)} byte-hops | slowest median {_fmt_ms(max_median)}", 12, MUTED))

    out.extend(_draw_hardware_panel(MARGIN, 108, WIDTH - 2 * MARGIN, 238))

    y = 374
    if not groups:
        out.append(_text(MARGIN, y + 40, "No rows available", 18, MUTED))
    for group_key, group in groups:
        out.extend(_draw_group(MARGIN, y, WIDTH - 2 * MARGIN, group_key, group))
        y += 278

    out.append(_text(MARGIN, height - 26, "Note: byte-hops are a compiler locality model; aiu-smi read/write counters measure full workload device-memory traffic.", 12, MUTED))
    out.append("</svg>")
    return "\n".join(out)


def render_html(svg: str, title: str) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            f"<title>{_esc(title)}</title>",
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<style>",
            "body{margin:0;background:#0b0f14;color:#e7edf6;font-family:Inter,system-ui,sans-serif;}",
            ".wrap{padding:24px;}",
            "svg{max-width:100%;height:auto;display:block;margin:0 auto;}",
            "</style>",
            "</head>",
            "<body><div class=\"wrap\">",
            svg,
            "</div></body></html>",
        ]
    )


def write_report(
    rows: list[dict[str, Any]],
    svg_path: Path,
    html_path: Path | None = None,
    title: str = "Spyre Restickify Counter Report",
) -> tuple[Path, Path | None]:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg = render_svg(rows, title=title)
    svg_path.write_text(svg, encoding="utf-8")
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(svg, title=title), encoding="utf-8")
    return svg_path, html_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Probe JSONL file to visualize.")
    parser.add_argument("--output-svg", type=Path, help="Output SVG path.")
    parser.add_argument("--output-html", type=Path, help="Output HTML path.")
    parser.add_argument("--title", default="Spyre Restickify Counter Report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = _load_rows(args.jsonl)
    svg_path = args.output_svg or args.jsonl.with_suffix(".svg")
    html_path = args.output_html or args.jsonl.with_suffix(".html")
    write_report(rows, svg_path, html_path, title=args.title)
    print(f"Wrote {svg_path}")
    print(f"Wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
