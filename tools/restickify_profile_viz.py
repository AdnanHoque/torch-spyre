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


def _json_script(value: Any) -> str:
    return json.dumps(value, sort_keys=True).replace("</", "<\\/")


def render_html(rows: list[dict[str, Any]], svg: str, title: str) -> str:
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {
  color-scheme: dark;
  --bg: #090d12;
  --panel: #111822;
  --panel-2: #151d28;
  --line: #263241;
  --line-2: #334155;
  --text: #e7edf6;
  --muted: #9aa7b5;
  --blue: #58a6ff;
  --green: #49d17d;
  --cyan: #51d1e6;
  --yellow: #f4c542;
  --orange: #ff9f43;
  --red: #ff6b6b;
  --purple: #b58cff;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: radial-gradient(circle at top left, #111827 0, var(--bg) 360px);
  color: var(--text);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
button, select, input {
  font: inherit;
}
.app {
  max-width: 1440px;
  margin: 0 auto;
  padding: 28px;
}
.topbar {
  display: flex;
  gap: 18px;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 18px;
}
h1 {
  margin: 0;
  font-size: 34px;
  line-height: 1.08;
  letter-spacing: 0;
}
.subtitle {
  margin-top: 8px;
  color: var(--muted);
  font-size: 15px;
}
.summary {
  display: grid;
  grid-template-columns: repeat(4, minmax(145px, 1fr));
  gap: 10px;
  min-width: 560px;
}
.stat {
  background: rgba(21, 29, 40, 0.92);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px 14px;
}
.stat .label {
  color: var(--muted);
  font-size: 12px;
}
.stat .value {
  margin-top: 4px;
  font-size: 18px;
  font-weight: 750;
}
.grid {
  display: grid;
  grid-template-columns: 360px minmax(0, 1fr);
  gap: 14px;
}
.panel {
  background: rgba(17, 24, 34, 0.96);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px;
  min-width: 0;
}
.panel h2 {
  font-size: 15px;
  margin: 0 0 12px 0;
}
.hardware {
  display: grid;
  gap: 12px;
}
.ring {
  position: relative;
  height: 168px;
  border: 5px solid var(--cyan);
  border-radius: 82px;
  background: #121a24;
}
.ring::after {
  content: "";
  position: absolute;
  inset: 16px;
  border: 2px dashed var(--blue);
  border-radius: 70px;
  opacity: 0.75;
}
.ring-title {
  position: absolute;
  left: 50%;
  top: 18px;
  transform: translateX(-50%);
  color: var(--cyan);
  font-weight: 800;
}
.ring-subtitle {
  position: absolute;
  left: 50%;
  top: 45px;
  transform: translateX(-50%);
  color: var(--muted);
  font-size: 12px;
  white-space: nowrap;
}
.core {
  position: absolute;
  z-index: 2;
  width: 52px;
  height: 32px;
  border: 1px solid #4c6280;
  border-radius: 8px;
  background: #202b3a;
  display: grid;
  place-items: center;
  font-weight: 800;
}
.core.c0 { left: 32px; top: 35px; }
.core.c1 { left: 86px; top: 103px; }
.core.c16 { left: 154px; top: 118px; }
.core.c30 { right: 86px; top: 103px; }
.core.c31 { right: 32px; top: 35px; }
.hw-note {
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}
.hw-paths {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.path-card {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px;
  background: #101720;
}
.path-card strong {
  display: block;
  margin-bottom: 5px;
}
.path-card.orange strong { color: var(--orange); }
.path-card.green strong { color: var(--green); }
.controls {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.control {
  display: grid;
  gap: 6px;
}
.control label {
  color: var(--muted);
  font-size: 12px;
}
select, input[type="search"] {
  width: 100%;
  border: 1px solid var(--line-2);
  border-radius: 8px;
  background: #0d131b;
  color: var(--text);
  padding: 9px 10px;
}
.tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 14px;
}
.tab {
  border: 1px solid var(--line);
  background: #0d131b;
  color: var(--muted);
  border-radius: 999px;
  padding: 8px 12px;
  cursor: pointer;
}
.tab.active {
  color: var(--text);
  border-color: var(--blue);
  background: #132033;
}
.view { display: none; }
.view.active { display: block; }
.pairs {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}
.pair {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px;
  background: #101720;
}
.pair .name { font-weight: 800; }
.pair .good { color: var(--green); }
.pair .warn { color: var(--yellow); }
.pair .muted { color: var(--muted); }
.chart {
  display: grid;
  gap: 10px;
}
.chart-row {
  display: grid;
  grid-template-columns: 220px minmax(180px, 1fr) 135px;
  align-items: center;
  gap: 10px;
  padding: 8px;
  border-radius: 9px;
  background: #0d131b;
  border: 1px solid transparent;
  cursor: pointer;
}
.chart-row:hover, .chart-row.selected {
  border-color: var(--blue);
  background: #111b28;
}
.bar-track {
  position: relative;
  height: 18px;
  border-radius: 6px;
  background: #202936;
  overflow: hidden;
}
.bar {
  height: 100%;
  border-radius: 6px;
}
.bar.baseline { background: var(--blue); }
.bar.stage3b { background: var(--green); }
.bar.other { background: var(--purple); }
.value {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.table-wrap {
  overflow: auto;
  border: 1px solid var(--line);
  border-radius: 10px;
}
table {
  width: 100%;
  border-collapse: collapse;
  min-width: 980px;
}
th, td {
  text-align: left;
  padding: 10px;
  border-bottom: 1px solid var(--line);
  font-size: 13px;
}
th {
  color: var(--muted);
  background: #101720;
  position: sticky;
  top: 0;
}
tr {
  cursor: pointer;
}
tbody tr:hover, tbody tr.selected {
  background: #132033;
}
.detail-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
  gap: 14px;
}
.detail-card {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px;
  background: #101720;
}
.kv {
  display: grid;
  grid-template-columns: 180px minmax(0, 1fr);
  gap: 8px;
  margin: 6px 0;
}
.kv span:first-child {
  color: var(--muted);
}
.source-row {
  display: grid;
  grid-template-columns: 170px minmax(0, 1fr) 92px;
  gap: 10px;
  align-items: center;
  margin: 8px 0;
}
.source-bar {
  height: 14px;
  border-radius: 6px;
  background: #202936;
  overflow: hidden;
}
.source-fill {
  height: 100%;
  border-radius: 6px;
}
.entry {
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 10px;
  margin-top: 10px;
  background: #0d131b;
}
.entry-title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-weight: 800;
  margin-bottom: 8px;
}
pre {
  white-space: pre-wrap;
  overflow: auto;
  background: #06090d;
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 12px;
  color: #d5dde8;
  max-height: 520px;
}
.snapshot svg {
  width: 100%;
  height: auto;
}
.pill {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--line);
  background: #0d131b;
  border-radius: 999px;
  padding: 3px 8px;
  color: var(--muted);
  font-size: 12px;
}
.footer-note {
  margin-top: 14px;
  color: var(--muted);
  font-size: 12px;
}
@media (max-width: 980px) {
  .topbar, .grid, .detail-grid { display: block; }
  .summary { min-width: 0; grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 14px; }
  .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .panel { margin-bottom: 14px; }
  .chart-row { grid-template-columns: 1fr; }
  .value { text-align: left; }
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div>
      <h1>__TITLE__</h1>
      <div class="subtitle">Interactive Spyre restickify report: compiler locality, AIU SMI counters, source attribution, and raw rows.</div>
    </div>
    <div class="summary" id="summary"></div>
  </header>

  <div class="grid">
    <aside class="panel hardware">
      <div>
        <h2>AIU Memory/Fabric Context</h2>
        <div class="hw-note">From the Spyre Knowledgebase: AIU has 32 cores connected by a bidirectional RIU data ring. Per-core LX scratchpad ownership and off-chip device-memory traffic both meet the ring-facing data path.</div>
      </div>
      <div class="ring" aria-label="AIU ring diagram">
        <div class="ring-title">RIU data BiRing</div>
        <div class="ring-subtitle">166 GB/s per direction | 333 GB/s aggregate model</div>
        <div class="core c0">C0</div>
        <div class="core c1">C1</div>
        <div class="core c16">C16</div>
        <div class="core c30">C30</div>
        <div class="core c31">C31</div>
      </div>
      <div class="hw-paths">
        <div class="path-card orange"><strong>HBM/off-chip</strong><span class="hw-note">Device memory read/write counters can stay high even after compiler byte-hops improve.</span></div>
        <div class="path-card green"><strong>Per-core LX</strong><span class="hw-note">Stage 3B models ownership alignment for eligible in-graph restickify edges.</span></div>
      </div>
    </aside>

    <main class="panel">
      <div class="controls">
        <div class="control">
          <label for="metric">Metric</label>
          <select id="metric">
            <option value="ring_total_byte_hops">Byte-hops</option>
            <option value="median_ms">Median latency</option>
            <option value="bandwidth_sum">Avg read+write bandwidth</option>
            <option value="aiusmi_avg_nonzero_rdmem_GiB_per_s">Avg read bandwidth</option>
            <option value="aiusmi_avg_nonzero_wrmem_GiB_per_s">Avg write bandwidth</option>
            <option value="total_bytes">Restickify bytes moved</option>
            <option value="ring_avg_hops">Avg hops</option>
          </select>
        </div>
        <div class="control">
          <label for="mode">Mode</label>
          <select id="mode">
            <option value="all">All modes</option>
            <option value="baseline">Baseline</option>
            <option value="stage3b">Stage 3B</option>
          </select>
        </div>
        <div class="control">
          <label for="search">Search</label>
          <input id="search" type="search" placeholder="case, size, source...">
        </div>
        <div class="control">
          <label for="sort">Sort</label>
          <select id="sort">
            <option value="metric-desc">Metric high to low</option>
            <option value="case">Case / size / mode</option>
            <option value="latency-desc">Latency high to low</option>
          </select>
        </div>
      </div>

      <div class="tabs">
        <button class="tab active" data-view="overview">Overview</button>
        <button class="tab" data-view="table">Rows</button>
        <button class="tab" data-view="detail">Selected Detail</button>
        <button class="tab" data-view="snapshot">Static Snapshot</button>
        <button class="tab" data-view="raw">Raw JSON</button>
      </div>

      <section class="view active" id="view-overview">
        <div class="pairs" id="pairs"></div>
        <div class="chart" id="chart"></div>
      </section>

      <section class="view" id="view-table">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Case</th>
                <th>Size</th>
                <th>Mode</th>
                <th>Restickifies</th>
                <th>Bytes</th>
                <th>Byte-hops</th>
                <th>Median</th>
                <th>Avg rd</th>
                <th>Avg wr</th>
                <th>Sources</th>
              </tr>
            </thead>
            <tbody id="rows"></tbody>
          </table>
        </div>
      </section>

      <section class="view" id="view-detail">
        <div id="detail"></div>
      </section>

      <section class="view snapshot" id="view-snapshot">
        __SVG__
      </section>

      <section class="view" id="view-raw">
        <pre id="raw"></pre>
      </section>

      <div class="footer-note">Byte-hops are compiler-modeled locality. AIU SMI counters measure full workload device-memory traffic for the measured loop.</div>
    </main>
  </div>
</div>

<script id="profile-data" type="application/json">__DATA__</script>
<script>
const rows = JSON.parse(document.getElementById("profile-data").textContent);
let selectedId = rows.length ? rowId(rows[0]) : "";

const colors = {
  baseline: "#58a6ff",
  stage3b: "#49d17d",
  other: "#b58cff",
  in_graph_computed: "#49d17d",
  graph_input_or_weight: "#ff9f43",
  constant_or_extern: "#b58cff",
  mutation_target: "#ff6b6b",
  unknown: "#9aa7b5",
};

const metricLabels = {
  ring_total_byte_hops: "byte-hops",
  median_ms: "ms",
  bandwidth_sum: "GiB/s",
  aiusmi_avg_nonzero_rdmem_GiB_per_s: "GiB/s",
  aiusmi_avg_nonzero_wrmem_GiB_per_s: "GiB/s",
  total_bytes: "bytes",
  ring_avg_hops: "hops",
};

function num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function rowId(row) {
  return `${row.case || "case"}|${row.size || 0}|${row.mode || "mode"}`;
}

function fmtInt(v) {
  return Math.round(num(v)).toLocaleString();
}

function fmtFloat(v, digits = 3) {
  return num(v).toFixed(digits);
}

function fmtMs(v) {
  return `${num(v).toFixed(4)} ms`;
}

function fmtBytes(v) {
  let n = num(v);
  const units = ["B", "KiB", "MiB", "GiB"];
  for (const unit of units) {
    if (Math.abs(n) < 1024 || unit === "GiB") {
      return unit === "B" ? `${Math.round(n).toLocaleString()} ${unit}` : `${n.toFixed(2)} ${unit}`;
    }
    n /= 1024;
  }
  return `${n.toFixed(2)} GiB`;
}

function fmtMetric(metric, value) {
  if (metric === "median_ms") return fmtMs(value);
  if (metric === "total_bytes") return fmtBytes(value);
  if (metric.includes("GiB_per_s") || metric === "bandwidth_sum") return `${num(value).toFixed(2)} GiB/s`;
  if (metric === "ring_avg_hops") return `${num(value).toFixed(3)} hops`;
  return fmtInt(value);
}

function metricValue(row, metric) {
  if (metric === "bandwidth_sum") {
    return num(row.aiusmi_avg_nonzero_rdmem_GiB_per_s) + num(row.aiusmi_avg_nonzero_wrmem_GiB_per_s);
  }
  return num(row[metric]);
}

function sourceBytes(row) {
  const totals = {};
  const entries = row.ring_entries || row.entries || [];
  for (const entry of entries) {
    const kind = entry.source_kind || entry.source || "unknown";
    const bytes = num(entry.bytes_moved || entry.num_bytes || entry.bytes);
    totals[kind] = (totals[kind] || 0) + bytes;
  }
  return totals;
}

function sourceLabel(row) {
  const src = sourceBytes(row);
  const parts = Object.entries(src).map(([kind, bytes]) => `${kind.replaceAll("_", " ")} ${fmtBytes(bytes)}`);
  return parts.length ? parts.join(" | ") : "none";
}

function filteredRows() {
  const mode = document.getElementById("mode").value;
  const q = document.getElementById("search").value.trim().toLowerCase();
  const metric = document.getElementById("metric").value;
  const sort = document.getElementById("sort").value;
  let out = rows.filter(row => mode === "all" || row.mode === mode);
  if (q) {
    out = out.filter(row => {
      const text = `${row.case || ""} ${row.size || ""} ${row.mode || ""} ${row.source_hint || ""} ${sourceLabel(row)}`.toLowerCase();
      return text.includes(q);
    });
  }
  out.sort((a, b) => {
    if (sort === "case") {
      return `${a.case}|${a.size}|${a.mode}`.localeCompare(`${b.case}|${b.size}|${b.mode}`);
    }
    if (sort === "latency-desc") {
      return num(b.median_ms) - num(a.median_ms);
    }
    return metricValue(b, metric) - metricValue(a, metric);
  });
  return out;
}

function renderSummary() {
  const ok = rows.filter(row => (row.status || "ok") === "ok");
  const totalBytes = ok.reduce((acc, row) => acc + num(row.total_bytes), 0);
  const totalHops = ok.reduce((acc, row) => acc + num(row.ring_total_byte_hops), 0);
  const maxMedian = ok.reduce((acc, row) => Math.max(acc, num(row.median_ms)), 0);
  const maxBw = ok.reduce((acc, row) => Math.max(acc, metricValue(row, "bandwidth_sum")), 0);
  const stats = [
    ["Rows", ok.length.toLocaleString()],
    ["Restickified", fmtBytes(totalBytes)],
    ["Byte-hops", fmtInt(totalHops)],
    ["Peak avg bandwidth", `${maxBw.toFixed(2)} GiB/s`],
    ["Slowest median", fmtMs(maxMedian)],
  ];
  document.getElementById("summary").innerHTML = stats.map(([label, value]) => `
    <div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>
  `).join("");
}

function groupedPairs() {
  const groups = new Map();
  for (const row of rows) {
    const key = `${row.case || "case"}|${row.size || 0}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

function renderPairs() {
  const el = document.getElementById("pairs");
  el.innerHTML = groupedPairs().map(([key, group]) => {
    const [caseName, size] = key.split("|");
    const byMode = Object.fromEntries(group.map(row => [row.mode, row]));
    const base = byMode.baseline;
    const stage = byMode.stage3b;
    let summary = "paired comparison unavailable";
    let klass = "muted";
    if (base && stage) {
      const baseHops = num(base.ring_total_byte_hops);
      const stageHops = num(stage.ring_total_byte_hops);
      const reduction = baseHops ? (1 - stageHops / baseHops) : 0;
      const speedup = num(stage.median_ms) ? num(base.median_ms) / num(stage.median_ms) : 0;
      klass = reduction > 0.5 && speedup >= 1 ? "good" : "warn";
      summary = `byte-hop reduction ${(reduction * 100).toFixed(1)}% | speedup ${speedup.toFixed(4)}x`;
    }
    return `<div class="pair">
      <div class="name">${caseName} <span class="pill">size ${size}</span></div>
      <div class="${klass}" style="margin-top:8px">${summary}</div>
    </div>`;
  }).join("");
}

function renderChart() {
  const metric = document.getElementById("metric").value;
  const data = filteredRows();
  const max = Math.max(1e-9, ...data.map(row => metricValue(row, metric)));
  const el = document.getElementById("chart");
  el.innerHTML = data.map(row => {
    const value = metricValue(row, metric);
    const pct = Math.max(0, Math.min(100, value / max * 100));
    const mode = row.mode || "other";
    const klass = mode === "baseline" || mode === "stage3b" ? mode : "other";
    const selected = rowId(row) === selectedId ? " selected" : "";
    const title = `${row.case} size ${row.size} ${mode}: ${fmtMetric(metric, value)}`;
    return `<div class="chart-row${selected}" data-id="${rowId(row)}" title="${title}">
      <div><strong style="color:${colors[mode] || colors.other}">${mode}</strong><br><span class="pill">${row.case || ""} | ${row.size || ""}</span></div>
      <div class="bar-track"><div class="bar ${klass}" style="width:${pct}%"></div></div>
      <div class="value">${fmtMetric(metric, value)}</div>
    </div>`;
  }).join("") || `<div class="hw-note">No rows match the current filters.</div>`;
  el.querySelectorAll(".chart-row").forEach(node => {
    node.addEventListener("click", () => selectRow(node.dataset.id, "detail"));
  });
}

function renderTable() {
  const data = filteredRows();
  const tbody = document.getElementById("rows");
  tbody.innerHTML = data.map(row => {
    const selected = rowId(row) === selectedId ? " class=\"selected\"" : "";
    return `<tr data-id="${rowId(row)}"${selected}>
      <td>${row.case || ""}</td>
      <td>${row.size || ""}</td>
      <td><strong style="color:${colors[row.mode] || colors.other}">${row.mode || ""}</strong></td>
      <td>${fmtInt(row.restickify_count)}</td>
      <td>${fmtBytes(row.total_bytes)}</td>
      <td>${fmtInt(row.ring_total_byte_hops)}</td>
      <td>${fmtMs(row.median_ms)}</td>
      <td>${fmtFloat(row.aiusmi_avg_nonzero_rdmem_GiB_per_s)} GiB/s</td>
      <td>${fmtFloat(row.aiusmi_avg_nonzero_wrmem_GiB_per_s)} GiB/s</td>
      <td>${sourceLabel(row)}</td>
    </tr>`;
  }).join("");
  tbody.querySelectorAll("tr").forEach(node => {
    node.addEventListener("click", () => selectRow(node.dataset.id, "detail"));
  });
}

function selectedRow() {
  return rows.find(row => rowId(row) === selectedId) || rows[0];
}

function renderSourceBreakdown(row) {
  const src = sourceBytes(row);
  const total = Object.values(src).reduce((a, b) => a + b, 0) || num(row.total_bytes) || 1;
  return Object.entries(src).map(([kind, bytes]) => {
    const pct = bytes / total * 100;
    const color = colors[kind] || colors.unknown;
    return `<div class="source-row">
      <div>${kind.replaceAll("_", " ")}</div>
      <div class="source-bar"><div class="source-fill" style="width:${pct}%; background:${color}"></div></div>
      <div class="value">${fmtBytes(bytes)}</div>
    </div>`;
  }).join("") || `<div class="hw-note">No source rows.</div>`;
}

function renderEntries(row) {
  const entries = row.ring_entries || row.entries || [];
  if (!entries.length) return `<div class="hw-note">No restickify entries available.</div>`;
  return entries.map(entry => {
    const name = `${entry.producer || entry.source_name || "<none>"} -> ${entry.restickify || "<restickify>"}`;
    const kind = entry.source_kind || "unknown";
    const split = `producer ${JSON.stringify(entry.producer_splits || entry.producer_split_map || {})} | restickify ${JSON.stringify(entry.restickify_splits || entry.restickify_split_map || {})}`;
    return `<div class="entry">
      <div class="entry-title"><span>${name}</span><span style="color:${colors[kind] || colors.unknown}">${kind}</span></div>
      <div class="kv"><span>Bytes moved</span><strong>${fmtBytes(entry.bytes_moved)}</strong></div>
      <div class="kv"><span>Byte-hops</span><strong>${fmtInt(entry.byte_hops || entry.total_byte_hops)}</strong></div>
      <div class="kv"><span>Avg / max hops</span><strong>${fmtFloat(entry.avg_hops)} / ${fmtInt(entry.max_hops)}</strong></div>
      <div class="kv"><span>Consumer</span><strong>${entry.consumer || (entry.consumers || []).join(", ") || ""}</strong></div>
      <div class="kv"><span>Split maps</span><strong>${split}</strong></div>
      <div class="kv"><span>Skip</span><strong>${entry.skip_reason || entry.locality_skip_reason || "none"}</strong></div>
    </div>`;
  }).join("");
}

function renderDetail() {
  const row = selectedRow();
  const el = document.getElementById("detail");
  if (!row) {
    el.innerHTML = `<div class="hw-note">No row selected.</div>`;
    return;
  }
  el.innerHTML = `<div class="detail-grid">
    <div class="detail-card">
      <h2>${row.case || ""} <span class="pill">size ${row.size || ""}</span> <span class="pill">${row.mode || ""}</span></h2>
      <div class="kv"><span>Restickifies</span><strong>${fmtInt(row.restickify_count)}</strong></div>
      <div class="kv"><span>Bytes moved</span><strong>${fmtBytes(row.total_bytes)}</strong></div>
      <div class="kv"><span>Byte-hops</span><strong>${fmtInt(row.ring_total_byte_hops)}</strong></div>
      <div class="kv"><span>Avg / max hops</span><strong>${fmtFloat(row.ring_avg_hops)} / ${fmtInt(row.ring_max_hops)}</strong></div>
      <div class="kv"><span>Median</span><strong>${fmtMs(row.median_ms)}</strong></div>
      <div class="kv"><span>p10 / p90</span><strong>${fmtMs(row.p10_ms)} / ${fmtMs(row.p90_ms)}</strong></div>
      <div class="kv"><span>AIU SMI avg rd/wr</span><strong>${fmtFloat(row.aiusmi_avg_nonzero_rdmem_GiB_per_s)} / ${fmtFloat(row.aiusmi_avg_nonzero_wrmem_GiB_per_s)} GiB/s</strong></div>
      <div class="kv"><span>AIU SMI peak rd/wr</span><strong>${fmtFloat(row.aiusmi_peak_rdmem_GiB_per_s)} / ${fmtFloat(row.aiusmi_peak_wrmem_GiB_per_s)} GiB/s</strong></div>
      <h2 style="margin-top:16px">Source Breakdown</h2>
      ${renderSourceBreakdown(row)}
    </div>
    <div class="detail-card">
      <h2>Restickify Entries</h2>
      ${renderEntries(row)}
    </div>
  </div>`;
}

function renderRaw() {
  document.getElementById("raw").textContent = JSON.stringify(selectedRow() || rows, null, 2);
}

function selectRow(id, view) {
  selectedId = id;
  renderChart();
  renderTable();
  renderDetail();
  renderRaw();
  if (view) showView(view);
}

function showView(name) {
  document.querySelectorAll(".tab").forEach(tab => tab.classList.toggle("active", tab.dataset.view === name));
  document.querySelectorAll(".view").forEach(view => view.classList.toggle("active", view.id === `view-${name}`));
}

function renderAll() {
  renderSummary();
  renderPairs();
  renderChart();
  renderTable();
  renderDetail();
  renderRaw();
}

document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => showView(tab.dataset.view));
});
["metric", "mode", "search", "sort"].forEach(id => {
  document.getElementById(id).addEventListener("input", () => {
    renderChart();
    renderTable();
  });
});

renderAll();
</script>
</body>
</html>
"""
    return (
        template.replace("__TITLE__", _esc(title))
        .replace("__DATA__", _json_script(rows))
        .replace("__SVG__", svg)
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
        html_path.write_text(render_html(rows, svg, title=title), encoding="utf-8")
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
