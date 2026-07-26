#!/usr/bin/env python3
"""Build the SenDNN vs Torch-Spyre parity study PDF."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = ROOT / "tmp" / "pdfs" / "charts"
OUTPUT = ROOT / "output" / "pdf" / "torch_spyre_sendnn_parity_study.pdf"
CHART_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# Theme
NAVY = "#12233F"
BLUE = "#3266CC"
TEAL = "#16A39A"
ORANGE = "#F29F3D"
PURPLE = "#7157C8"
RED = "#D9544D"
GREEN = "#2E9D69"
SLATE = "#5C667A"
LIGHT = "#F4F7FB"
GRID = "#D8E0EA"
INK = "#172033"

RL_NAVY = colors.HexColor(NAVY)
RL_BLUE = colors.HexColor(BLUE)
RL_TEAL = colors.HexColor(TEAL)
RL_ORANGE = colors.HexColor(ORANGE)
RL_PURPLE = colors.HexColor(PURPLE)
RL_RED = colors.HexColor(RED)
RL_GREEN = colors.HexColor(GREEN)
RL_SLATE = colors.HexColor(SLATE)
RL_LIGHT = colors.HexColor(LIGHT)
RL_GRID = colors.HexColor(GRID)
RL_INK = colors.HexColor(INK)


PAGE_W, PAGE_H = landscape(letter)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def save_chart(fig, name: str) -> Path:
    path = CHART_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_charts() -> dict[str, Path]:
    configure_matplotlib()
    charts: dict[str, Path] = {}

    # 1. Device and wall baseline.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2))
    phases = ["Prefill", "Decode"]
    x = np.arange(2)
    w = 0.34
    tsp_device = [301.442, 153.592]
    sen_device = [190.406, 123.961]
    axes[0].bar(x - w / 2, tsp_device, w, color=BLUE, label="Torch-Spyre")
    axes[0].bar(x + w / 2, sen_device, w, color=TEAL, label="SenDNN")
    axes[0].set_xticks(x, phases)
    axes[0].set_ylabel("Device time (ms)")
    axes[0].set_title("Measured device-program latency")
    axes[0].legend(loc="upper right")
    for i, values in enumerate((tsp_device, sen_device)):
        for j, value in enumerate(values):
            axes[0].text(j + (-w / 2 if i == 0 else w / 2), value + 5, f"{value:.1f}", ha="center", fontsize=8)
    axes[0].set_ylim(0, 350)

    tsp_wall = [1293.896, 1105.272]
    sen_wall = [195.535, 132.884]
    axes[1].bar(x - w / 2, tsp_wall, w, color=BLUE)
    axes[1].bar(x + w / 2, sen_wall, w, color=TEAL)
    axes[1].set_xticks(x, phases)
    axes[1].set_ylabel("Profiled wall time (ms)")
    axes[1].set_title("Historical end-to-end wall latency")
    for i, values in enumerate((tsp_wall, sen_wall)):
        for j, value in enumerate(values):
            axes[1].text(j + (-w / 2 if i == 0 else w / 2), value + 24, f"{value:.0f}", ha="center", fontsize=8)
    axes[1].set_ylim(0, 1450)
    fig.tight_layout(w_pad=3)
    charts["baseline"] = save_chart(fig, "01_baseline.png")

    # 2. Wall gap budget.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.15))
    labels = ["Device\nprogram", "CPU\nembedding", "Compiled-region\noverhead", "Other host"]
    colors_ = [PURPLE, RED, ORANGE, SLATE]
    for ax, title, vals in zip(
        axes,
        ["Prefill wall gap: 1098.4 ms", "Decode wall gap: 972.4 ms"],
        [[111.036, 846.040, 63.799, 77.486], [29.631, 836.520, 92.249, 13.989]],
    ):
        left = 0
        for label, value, color in zip(labels, vals, colors_):
            ax.barh([0], [value], left=left, color=color, height=0.42, label=label.replace("\n", " "))
            if value > 45:
                ax.text(left + value / 2, 0, f"{value:.0f}", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
            left += value
        ax.set_title(title)
        ax.set_xlim(0, sum(vals) * 1.02)
        ax.set_yticks([])
        ax.set_xlabel("Milliseconds")
    axes[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0, -0.38), fontsize=7)
    fig.tight_layout(w_pad=2)
    charts["wall_budget"] = save_chart(fig, "02_wall_budget.png")

    # 3. Prefill family latency, target, and PT ideal.
    families = ["Q proj +\nnorm", "K + QK +\nsoftmax", "V + AV +\nout proj", "SwiGLU"]
    current = np.array([0.532906, 1.434337, 1.056483, 4.224541])
    target = np.array([0.400, 0.450, 0.720, 2.900])
    ideal = np.array([0.238313, 0.089367, 0.327680, 2.234182])
    fig, ax = plt.subplots(figsize=(10.4, 3.55))
    x = np.arange(len(families))
    ax.bar(x - 0.25, current, 0.25, color=BLUE, label="Current")
    ax.bar(x, target, 0.25, color=ORANGE, label="Parity budget")
    ax.bar(x + 0.25, ideal, 0.25, color=TEAL, label="PT ideal proxy")
    ax.set_xticks(x, families)
    ax.set_ylabel("Latency per layer (ms)")
    ax.set_title("Prefill: where 111 ms must be recovered")
    ax.legend(ncol=3, loc="upper left")
    for xi, c, t in zip(x, current, target):
        ax.text(xi - 0.25, c + 0.08, f"{c:.2f}", ha="center", fontsize=8)
        ax.text(xi, t + 0.08, f"{t:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 4.8)
    fig.tight_layout()
    charts["prefill"] = save_chart(fig, "03_prefill_budget.png")

    # 4. PT utilization current and target.
    current_util = [44.72, 6.23, 31.02, 52.89]
    target_util = [59.58, 19.86, 45.51, 77.04]
    fig, ax = plt.subplots(figsize=(10.4, 3.3))
    y = np.arange(4)
    ax.barh(y + 0.17, target_util, 0.33, color=ORANGE, label="Target proxy")
    ax.barh(y - 0.17, current_util, 0.33, color=BLUE, label="Current proxy")
    ax.set_yticks(y, families)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Compiler PT ideal / measured latency (%)")
    ax.set_title("Prefill PT realization - same arithmetic floor, different schedule")
    ax.legend(loc="lower right")
    for yi, c, t in zip(y, current_util, target_util):
        ax.text(c + 1.2, yi - 0.17, f"{c:.1f}%", va="center", fontsize=8)
        ax.text(t + 1.2, yi + 0.17, f"{t:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    charts["pt_util"] = save_chart(fig, "04_pt_util.png")

    # 5. Decode useful bandwidth.
    fig, ax = plt.subplots(figsize=(10.4, 3.45))
    names = ["Attention +\nprojections + KV", "Whole decoder\nblock", "SwiGLU", "Parity attention\nrange", "Modeled ceiling"]
    values = [60.84, 107.54, 136.45, 120.75, 140.0]
    bar_colors = [RED, BLUE, TEAL, ORANGE, NAVY]
    bars = ax.bar(np.arange(len(names)), values, color=bar_colors, width=0.62)
    ax.set_xticks(np.arange(len(names)), names)
    ax.set_ylabel("Useful payload GB/s")
    ax.set_ylim(0, 155)
    ax.set_title("Decode is an attention locality problem, not an MLP problem")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"{value:.1f}", ha="center", fontsize=8, fontweight="bold")
    ax.axhspan(116, 126, color=ORANGE, alpha=0.12)
    fig.tight_layout()
    charts["decode_bw"] = save_chart(fig, "05_decode_bandwidth.png")

    # 6. Decode latency decomposition and target range.
    fig, ax = plt.subplots(figsize=(10.4, 3.45))
    groups = ["Proj + KV", "QK + softmax", "AV + output", "SwiGLU"]
    curr = [0.1593, 0.8488, 0.4054, 2.3054]
    low_target = [0.120, 0.285, 0.280, 2.3054]
    high_target = [0.130, 0.320, 0.294, 2.2469]
    x = np.arange(4)
    ax.bar(x - 0.22, curr, 0.22, color=BLUE, label="Current")
    ax.bar(x, low_target, 0.22, color=TEAL, label="Attention-led parity")
    ax.bar(x + 0.22, high_target, 0.22, color=ORANGE, label="Relaxed attention + MLP floor")
    ax.set_xticks(x, groups)
    ax.set_ylabel("Latency per layer (ms)")
    ax.set_ylim(0, 2.6)
    ax.set_title("Decode parity budget - two non-overlapping routes")
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    fig.tight_layout()
    charts["decode_budget"] = save_chart(fig, "06_decode_budget.png")

    # 7. SMC memory vs peer transfer sites.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.45), sharey=True)
    path_names = ["Memory\nloads", "Peer\nloads", "Memory\nstores", "Peer\nstores"]
    tsp = [[204, 0, 28, 0], [170, 0, 32, 0]]
    sen = [[140.0, 83.8, 39.0, 12.1], [59.0, 217.2, 9.0, 29.2]]
    for ax, title, a, b in zip(axes, ["Prefill marginal layer", "Decode marginal layer"], tsp, sen):
        x = np.arange(4)
        ax.bar(x - 0.18, a, 0.36, color=BLUE, label="Torch-Spyre")
        ax.bar(x + 0.18, b, 0.36, color=TEAL, label="SenDNN")
        ax.set_xticks(x, path_names)
        ax.set_title(title)
        ax.set_ylabel("Static L3 opcode sites")
        for xi, value in enumerate(b):
            ax.text(xi + 0.18, value + 5, f"{value:g}", ha="center", fontsize=7)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Generated SMCs choose different physical data paths", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    charts["smc_paths"] = save_chart(fig, "07_smc_paths.png")

    # 8. Relayout family demand.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.65))
    prefill_vals = [4274.218, 128.975, 62.915, 39.322, 31.457, 3.154]
    prefill_names = ["BMM", "Exx2", "Mul", "Restickify", "LayerNorm", "Other"]
    decode_vals = [176.698, 58.884, 5.080, 1.905, 0.643, 0.600]
    decode_names = ["BMM", "Restickify", "Softmax", "Max/Sum", "Norm", "Other"]
    palette = [BLUE, TEAL, ORANGE, PURPLE, GREEN, SLATE]
    for ax, title, vals, names in zip(
        axes,
        ["Prefill: 4.540 GB logical remote demand", "Decode: 242.81 MB logical remote demand"],
        [prefill_vals, decode_vals],
        [prefill_names, decode_names],
    ):
        ax.pie(vals, labels=None, colors=palette, startangle=90, wedgeprops={"linewidth": 1, "edgecolor": "white"})
        ax.set_title(title)
        ax.legend(names, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7)
    fig.tight_layout()
    charts["relayout_demand"] = save_chart(fig, "08_relayout_demand.png")

    # 9. Ring proxy.
    fig, ax = plt.subplots(figsize=(10.4, 3.35))
    names = ["Prefill", "Decode"]
    service_ms = [3.211, 0.277]
    phase_ms = [190.406, 123.961]
    x = np.arange(2)
    ax.bar(x, phase_ms, color=LIGHT, edgecolor=GRID, label="Measured SenDNN phase")
    ax.bar(x, service_ms, color=PURPLE, label="Modeled hottest-link service")
    ax.set_xticks(x, names)
    ax.set_ylabel("Milliseconds")
    ax.set_title("The ring is an enabler, not the phase bottleneck")
    ax.legend(loc="upper right")
    for xi, s, p in zip(x, service_ms, phase_ms):
        ax.text(xi, s + 4, f"{s:.3f} ms = {100*s/p:.2f}%", ha="center", fontsize=9, fontweight="bold", color=PURPLE)
    ax.set_ylim(0, 215)
    fig.tight_layout()
    charts["ring"] = save_chart(fig, "09_ring_proxy.png")

    # 10. Opportunity waterfall.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.55))
    # Prefill waterfall
    start = 301.442
    changes = [-58.149, -52.982]
    labels = ["Current", "Attention +\nprojection", "SwiGLU", "SenDNN"]
    vals = [start, start + changes[0], start + sum(changes), 190.406]
    axes[0].bar([0], [vals[0]], color=BLUE)
    axes[0].bar([1], [abs(changes[0])], bottom=[vals[1]], color=TEAL)
    axes[0].bar([2], [abs(changes[1])], bottom=[vals[2]], color=ORANGE)
    axes[0].bar([3], [vals[3]], color=NAVY)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylabel("Device time (ms)")
    axes[0].set_title("Prefill gap reconciliation")
    axes[0].set_ylim(0, 330)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 7, f"{v:.1f}", ha="center", fontsize=8)

    start = 153.114
    vals = [start, start - 26.781, start - 26.781 - 2.339, 123.981]
    labels = ["Current", "Attention", "MLP floor", "SenDNN"]
    axes[1].bar([0], [vals[0]], color=BLUE)
    axes[1].bar([1], [26.781], bottom=[vals[1]], color=TEAL)
    axes[1].bar([2], [2.339], bottom=[vals[2]], color=ORANGE)
    axes[1].bar([3], [vals[3]], color=NAVY)
    axes[1].set_xticks(range(4), labels)
    axes[1].set_ylabel("Device time (ms)")
    axes[1].set_title("Decode gap reconciliation")
    axes[1].set_ylim(0, 170)
    for i, v in enumerate(vals):
        axes[1].text(i, v + 4, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout(w_pad=3)
    charts["waterfall"] = save_chart(fig, "10_opportunity_waterfall.png")

    # 11. Roadmap Gantt-style.
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    stages = [
        "M0 Permanent parity harness",
        "M1 Phase DAG + device embedding",
        "M2 Decode ownership chain",
        "M3 Prefill attention",
        "M4 Prefill true-BMM SwiGLU",
        "M5 One-offs and wall cleanup",
    ]
    starts = [0, 1, 3, 5, 5, 7]
    durations = [2, 3, 3, 3, 3, 2]
    stage_colors = [SLATE, BLUE, TEAL, PURPLE, ORANGE, GREEN]
    y = np.arange(len(stages))
    for yi, start_, dur, color in zip(y, starts, durations, stage_colors):
        ax.barh(yi, dur, left=start_, color=color, height=0.55)
    ax.set_yticks(y, stages)
    ax.invert_yaxis()
    ax.set_xticks(range(10), ["Gate", "Arch", "Arch", "Decode", "Decode", "Prefill", "Prefill", "Integrate", "Close", "Parity"])
    ax.set_xlabel("Dependency sequence, not calendar time")
    ax.set_title("Execution roadmap - fastest falsifiable path to parity")
    ax.grid(axis="x")
    fig.tight_layout()
    charts["roadmap"] = save_chart(fig, "11_roadmap.png")

    return charts


BASE_STYLES = getSampleStyleSheet()
styles = {
    "cover_title": ParagraphStyle(
        "CoverTitle", parent=BASE_STYLES["Title"], fontName="Helvetica-Bold", fontSize=30,
        leading=34, textColor=colors.white, alignment=TA_LEFT, spaceAfter=12,
    ),
    "cover_sub": ParagraphStyle(
        "CoverSub", parent=BASE_STYLES["Normal"], fontName="Helvetica", fontSize=14,
        leading=20, textColor=colors.HexColor("#DDE7F6"), alignment=TA_LEFT,
    ),
    "h1": ParagraphStyle(
        "H1", parent=BASE_STYLES["Heading1"], fontName="Helvetica-Bold", fontSize=21,
        leading=24, textColor=RL_NAVY, spaceAfter=8,
    ),
    "h2": ParagraphStyle(
        "H2", parent=BASE_STYLES["Heading2"], fontName="Helvetica-Bold", fontSize=13,
        leading=16, textColor=RL_BLUE, spaceBefore=4, spaceAfter=5,
    ),
    "body": ParagraphStyle(
        "Body", parent=BASE_STYLES["BodyText"], fontName="Helvetica", fontSize=9,
        leading=12.2, textColor=RL_INK, spaceAfter=5,
    ),
    "body_small": ParagraphStyle(
        "BodySmall", parent=BASE_STYLES["BodyText"], fontName="Helvetica", fontSize=7.5,
        leading=9.7, textColor=RL_INK, spaceAfter=3,
    ),
    "caption": ParagraphStyle(
        "Caption", parent=BASE_STYLES["BodyText"], fontName="Helvetica", fontSize=7,
        leading=9, textColor=RL_SLATE, alignment=TA_LEFT, spaceBefore=2,
    ),
    "callout": ParagraphStyle(
        "Callout", parent=BASE_STYLES["BodyText"], fontName="Helvetica-Bold", fontSize=10,
        leading=14, textColor=RL_NAVY, spaceAfter=0,
    ),
    "metric": ParagraphStyle(
        "Metric", parent=BASE_STYLES["BodyText"], fontName="Helvetica-Bold", fontSize=18,
        leading=20, textColor=RL_NAVY, alignment=TA_CENTER,
    ),
    "metric_label": ParagraphStyle(
        "MetricLabel", parent=BASE_STYLES["BodyText"], fontName="Helvetica", fontSize=7.3,
        leading=9, textColor=RL_SLATE, alignment=TA_CENTER,
    ),
    "table": ParagraphStyle(
        "Table", parent=BASE_STYLES["BodyText"], fontName="Helvetica", fontSize=7.1,
        leading=8.8, textColor=RL_INK,
    ),
    "table_head": ParagraphStyle(
        "TableHead", parent=BASE_STYLES["BodyText"], fontName="Helvetica-Bold", fontSize=7.1,
        leading=8.8, textColor=colors.white, alignment=TA_LEFT,
    ),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def section_header(number: str, title: str, subtitle: str | None = None):
    out = [P(f"{number}  {title}", "h1")]
    if subtitle:
        out.append(P(subtitle, "body"))
    out.append(HRFlowable(width="100%", thickness=1.2, color=RL_GRID, spaceBefore=1, spaceAfter=8))
    return out


def callout(text: str, color=RL_TEAL, bg=RL_LIGHT):
    table = Table([[P(text, "callout")]], colWidths=[7.15 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0.8, color),
                ("LINEBEFORE", (0, 0), (0, -1), 6, color),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def metric_cards(cards: list[tuple[str, str, str]], widths=None):
    widths = widths or [7.15 * inch / len(cards)] * len(cards)
    cells = []
    for value, label, accent in cards:
        cells.append([P(value, "metric"), P(label, "metric_label"), colors.HexColor(accent)])
    nested = []
    for value_p, label_p, accent in cells:
        t = Table([[value_p], [label_p]], colWidths=[widths[len(nested)]])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.8, RL_GRID),
                    ("LINEABOVE", (0, 0), (-1, 0), 4, accent),
                    ("TOPPADDING", (0, 0), (-1, 0), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                    ("TOPPADDING", (0, 1), (-1, 1), 2),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
                ]
            )
        )
        nested.append(t)
    outer = Table([nested], colWidths=widths, hAlign="LEFT")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return outer


def styled_table(data, col_widths, header=True, font_size=7.1, row_bgs=True):
    converted = []
    for r, row in enumerate(data):
        converted.append([P(str(v), "table_head" if header and r == 0 else "table") for v in row])
    t = Table(converted, colWidths=col_widths, repeatRows=1 if header else 0, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), RL_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, RL_GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if row_bgs:
        for r in range(1, len(data)):
            if r % 2 == 0:
                style.append(("BACKGROUND", (0, r), (-1, r), RL_LIGHT))
    t.setStyle(TableStyle(style))
    return t


def two_col(left, right, widths=(3.55 * inch, 3.55 * inch), gap=0.1 * inch):
    t = Table([[left, right]], colWidths=list(widths), hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), gap / 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), gap / 2),
            ]
        )
    )
    return t


def image(path: Path, width: float, height: float | None = None):
    im = Image(str(path), width=width, height=height or width * 0.34)
    im.hAlign = "CENTER"
    return im


def architecture_diagram() -> Drawing:
    d = Drawing(720, 240)
    d.add(String(15, 218, "Torch-Spyre today", fontName="Helvetica-Bold", fontSize=13, fillColor=RL_BLUE))
    d.add(String(390, 218, "SenDNN / parity architecture", fontName="Helvetica-Bold", fontSize=13, fillColor=RL_TEAL))

    # Torch boxes
    y = 165
    for i, label in enumerate(["Bundle A", "Bundle B", "Bundle C"]):
        x = 15 + i * 108
        d.add(Rect(x, y, 88, 38, 6, fillColor=colors.HexColor("#EAF0FC"), strokeColor=RL_BLUE))
        d.add(String(x + 44, y + 15, label, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8, fillColor=RL_NAVY))
        if i < 2:
            d.add(Line(x + 88, y + 19, x + 108, y + 19, strokeColor=RL_RED, strokeWidth=2))
            d.add(String(x + 98, y + 26, "HBM", textAnchor="middle", fontSize=6, fillColor=RL_RED))
    d.add(Rect(15, 91, 304, 43, 6, fillColor=colors.white, strokeColor=RL_GRID))
    d.add(String(167, 118, "Independent compile, allocation, correction, launch", textAnchor="middle", fontSize=8, fillColor=RL_SLATE))
    d.add(String(167, 102, "No phase-wide ownership; zero repeated-block peer L3 sites", textAnchor="middle", fontSize=8, fillColor=RL_RED))

    # SenDNN plan box
    d.add(Rect(390, 78, 315, 125, 8, fillColor=colors.HexColor("#E7F7F5"), strokeColor=RL_TEAL, strokeWidth=1.5))
    d.add(String(547, 187, "One optimizer-visible phase plan", textAnchor="middle", fontName="Helvetica-Bold", fontSize=10, fillColor=RL_NAVY))
    for i, label in enumerate(["Internal A", "Internal B", "Internal C"]):
        x = 410 + i * 94
        d.add(Rect(x, 133, 76, 31, 5, fillColor=colors.white, strokeColor=RL_TEAL))
        d.add(String(x + 38, 145, label, textAnchor="middle", fontSize=7.5, fillColor=RL_INK))
        if i < 2:
            d.add(Line(x + 76, 148, x + 94, 148, strokeColor=RL_PURPLE, strokeWidth=3))
            d.add(String(x + 85, 155, "LX", textAnchor="middle", fontSize=6, fillColor=RL_PURPLE))
    d.add(String(547, 108, "Global lifetimes + placement + peer relayout + prefetch + overlap", textAnchor="middle", fontSize=8, fillColor=RL_INK))
    d.add(String(547, 91, "Internal segmentation preserved; external submission collapsed", textAnchor="middle", fontSize=8, fillColor=RL_SLATE))

    d.add(Line(360, 65, 360, 210, strokeColor=RL_GRID, strokeWidth=1))
    d.add(String(15, 42, "Boundary cost", fontName="Helvetica-Bold", fontSize=8, fillColor=RL_RED))
    d.add(String(90, 42, "HBM materialization + barriers + correction + launch", fontSize=8, fillColor=RL_SLATE))
    d.add(String(390, 42, "Parity principle", fontName="Helvetica-Bold", fontSize=8, fillColor=RL_TEAL))
    d.add(String(470, 42, "Keep ownership explicit until the last consumer", fontSize=8, fillColor=RL_SLATE))
    return d


def topology_diagram() -> Drawing:
    d = Drawing(720, 250)
    panels = [
        (10, "Permutation", "32 singleton pieces", "all-to-all-like relayout", RL_PURPLE),
        (188, "Subgroup all-gather", "32 shards -> 8 groups x4", "QK / GQA ingress", RL_BLUE),
        (366, "Reduce + broadcast", "4-to-1 gather; 1-to-4 fanout", "softmax max and sum", RL_ORANGE),
        (544, "Gather-to-owner", "16 shards -> cache owner", "KV scatter", RL_GREEN),
    ]
    for x, title, line1, line2, color in panels:
        d.add(Rect(x, 30, 166, 188, 8, fillColor=colors.white, strokeColor=RL_GRID))
        d.add(Rect(x, 188, 166, 30, 8, fillColor=color, strokeColor=color))
        d.add(String(x + 83, 199, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8.5, fillColor=colors.white))
        # Nodes
        if title == "Permutation":
            for i in range(5):
                sy = 158 - i * 22
                dy = 158 - ((i * 2) % 5) * 22
                d.add(Rect(x + 18, sy, 12, 12, 3, fillColor=colors.HexColor("#EEEAFB"), strokeColor=color))
                d.add(Rect(x + 136, dy, 12, 12, 3, fillColor=colors.HexColor("#EEEAFB"), strokeColor=color))
                d.add(Line(x + 30, sy + 6, x + 136, dy + 6, strokeColor=color, strokeWidth=1.2))
        elif title == "Subgroup all-gather":
            for i in range(4):
                d.add(Rect(x + 18, 150 - i * 20, 12, 12, 3, fillColor=colors.HexColor("#EAF0FC"), strokeColor=color))
                d.add(Line(x + 30, 156 - i * 20, x + 95, 123, strokeColor=color))
            d.add(Rect(x + 95, 112, 48, 24, 4, fillColor=colors.HexColor("#EAF0FC"), strokeColor=color))
            d.add(String(x + 119, 121, "group x4", textAnchor="middle", fontSize=6.5, fillColor=RL_NAVY))
        elif title == "Reduce + broadcast":
            for i in range(4):
                d.add(Rect(x + 14 + i * 30, 154, 10, 10, 2, fillColor=colors.HexColor("#FFF0DC"), strokeColor=color))
                d.add(Line(x + 19 + i * 30, 154, x + 79, 127, strokeColor=color))
            d.add(Rect(x + 68, 116, 22, 22, 4, fillColor=colors.HexColor("#FFF0DC"), strokeColor=color))
            for i in range(4):
                d.add(Line(x + 79, 116, x + 19 + i * 30, 82, strokeColor=color))
                d.add(Rect(x + 14 + i * 30, 72, 10, 10, 2, fillColor=colors.HexColor("#FFF0DC"), strokeColor=color))
        else:
            for i in range(5):
                d.add(Rect(x + 15, 158 - i * 20, 10, 10, 2, fillColor=colors.HexColor("#E8F5EE"), strokeColor=color))
                d.add(Line(x + 25, 163 - i * 20, x + 121, 123, strokeColor=color))
            d.add(Rect(x + 121, 107, 27, 32, 4, fillColor=colors.HexColor("#E8F5EE"), strokeColor=color))
            d.add(String(x + 134, 120, "KV", textAnchor="middle", fontName="Helvetica-Bold", fontSize=7, fillColor=RL_GREEN))
        d.add(String(x + 83, 53, line1, textAnchor="middle", fontSize=7, fillColor=RL_INK))
        d.add(String(x + 83, 39, line2, textAnchor="middle", fontSize=7, fillColor=RL_SLATE))
    return d


def compiler_stack_diagram() -> Drawing:
    d = Drawing(720, 240)
    layers = [
        ("1. Capture", "One prefill FX graph + one dynamic decode graph", RL_BLUE),
        ("2. Phase scheduler", "Ordered internal-bundle DAG with stable edge identity", RL_TEAL),
        ("3. Joint planner", "Work division + physical placement + route + critical path", RL_PURPLE),
        ("4. LX allocator", "Atomic S1/S2 lifetimes under exact usable-LX contract", RL_ORANGE),
        ("5. Emitter/runtime", "Peer L3 opcodes + patch specialization + one phase launch", RL_GREEN),
    ]
    y = 195
    for i, (title, text, color) in enumerate(layers):
        width = 650 - i * 35
        x = 35 + i * 17.5
        d.add(Rect(x, y - i * 39, width, 30, 6, fillColor=colors.white, strokeColor=color, strokeWidth=1.3))
        d.add(Rect(x, y - i * 39, 126, 30, 6, fillColor=color, strokeColor=color))
        d.add(String(x + 63, y + 10 - i * 39, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.white))
        d.add(String(x + 138, y + 10 - i * 39, text, fontSize=8, fillColor=RL_INK))
    d.add(String(360, 16, "Invariant: every requested on-chip edge must be proven in the emitted SMC", textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=RL_NAVY))
    return d


def page_decor(canvas, doc):
    canvas.saveState()
    if doc.page == 1:
        canvas.setFillColor(RL_NAVY)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.setFillColor(RL_TEAL)
        canvas.rect(0, 0, 18, PAGE_H, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#1C3458"))
        canvas.circle(PAGE_W - 65, PAGE_H - 72, 92, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#24456F"))
        canvas.circle(PAGE_W - 20, 80, 125, fill=1, stroke=0)
    else:
        canvas.setStrokeColor(RL_GRID)
        canvas.setLineWidth(0.6)
        canvas.line(36, 28, PAGE_W - 36, 28)
        canvas.setFillColor(RL_SLATE)
        canvas.setFont("Helvetica", 6.7)
        canvas.drawString(36, 17, "Torch-Spyre performance parity study | Granite 3.3 8B | July 25, 2026")
        canvas.drawRightString(PAGE_W - 36, 17, f"{doc.page}")
        canvas.setFillColor(RL_NAVY)
        canvas.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    canvas.restoreState()


def build_story(charts: dict[str, Path]):
    story = []

    # Cover
    story += [Spacer(1, 0.68 * inch)]
    story += [P("Torch-Spyre to SenDNN<br/>Performance Parity", "cover_title")]
    story += [P("A first-principles study of full-model Granite prefill and decode", "cover_sub")]
    story += [Spacer(1, 0.34 * inch)]
    cover_box = Table(
        [[P("The diagnosis", "table_head"), P("The path", "table_head")],
         [P("SenDNN wins by preserving ownership across a phase-wide internal-bundle plan, routing selected edges directly through peer LX, and scheduling PT, HBM, and synchronization as one critical path.", "body"),
          P("Build a phase DAG, make peer-LX handoff first-class, close decode attention first, then prefill attention and true-BMM SwiGLU, and finally remove embedding and submission overhead.", "body")]],
        colWidths=[3.45 * inch, 3.45 * inch],
    )
    cover_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24456F")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F6F9FD")),
        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#4C6688")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD6E6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [cover_box, Spacer(1, 0.35 * inch)]
    story += [P("Measured contract: Granite 3.3 8B Instruct, 40 decoder layers, batch 1, prompt 512, FP16, unfused weights, SDPA, four generated-token phases, five profiled runs.", "cover_sub")]
    story += [Spacer(1, 0.17 * inch), P("Artifact branch: adnan/sendnn-granite-antoni-repro-20260725 | Study commit: b12dacfd", "cover_sub")]
    story += [PageBreak()]

    # Executive summary
    story += section_header("01", "Executive conclusion", "The performance gap has two causes: device-program realization and end-to-end execution boundaries.")
    story += [metric_cards([
        ("111.0 ms", "Prefill device gap", RED),
        ("29.6 ms", "Decode device gap", ORANGE),
        ("0 peer sites", "Torch repeated-block peer L3", BLUE),
        ("1 phase job", "SenDNN external execution", TEAL),
    ])]
    story += [Spacer(1, 8)]
    story += [callout("Primary finding: SenDNN does not perform materially less prefill matrix arithmetic. It realizes the same PT floor with better ownership, data paths, synchronization, and scheduling scope.")]
    story += [Spacer(1, 7)]
    left = [
        P("What SenDNN has", "h2"),
        P("- One phase-wide optimizer-visible plan with 169/180 internal bundles.<br/>- Exact LX producer-to-consumer relayouts using peer L3 instructions.<br/>- Patch-heavy core specialization and fewer static synchronization sites.<br/>- Compatible BMM work division and compact PT loop forms.", "body"),
    ]
    right = [
        P("What Torch-Spyre is missing", "h2"),
        P("- Cross-bundle tensor identity, lifetime, and ownership.<br/>- A joint work-division, placement, route, and spill cost model.<br/>- An LX-resident associative attention chain, not K-only retention.<br/>- Phase-level address binding, execution, and embedding lowering.", "body"),
    ]
    story += [two_col(left, right), Spacer(1, 6)]
    story += [P("The shortest path is decode-first because its causal chain is strongest: Torch attention realizes 60.84 useful GB/s, SwiGLU already realizes 136.45 GB/s, and SenDNN decode replaces most memory data motion with peer traffic.", "body")]
    story += [PageBreak()]

    # First principles
    story += section_header("02", "First-principles performance model", "Latency is the critical path through compute, memory, communication, synchronization, and execution boundaries.")
    story += [callout("T_phase = critical_path(PT compute, HBM service, peer-LX service, synchronization, legal overlap) + external execution overhead", RL_PURPLE, colors.HexColor("#F0EDFA"))]
    story += [Spacer(1, 7), architecture_diagram()]
    principles = [
        ["Principle", "Implication for this study"],
        ["Arithmetic floor", "If ideal PT work is equal, speed comes from realizing it with fewer exposed stalls and transfers."],
        ["Ownership", "A producer-consumer value stays cheap only if both ends agree on its exact per-core byte map."],
        ["Communication", "Peer LX is valuable because it avoids HBM materialization; raw ring time is not the main gap."],
        ["Scope", "Runtime batching cannot recover device time unless placement and lifetimes are optimized across bundles."],
        ["Cost model", "Rank complete legal regions, including boundary traffic, spill, synchronization, and overlap."],
    ]
    story += [styled_table(principles, [1.35 * inch, 5.8 * inch]), Spacer(1, 4)]
    story += [P("This model prevents a common analytical error: dividing delivered bytes by one link's bandwidth and calling the result ring utilization. The correct network lower bound uses the hottest directed link, source injection, and destination drain.", "caption")]
    story += [PageBreak()]

    # Baseline
    story += section_header("03", "Measured full-model gap", "Device-program time and profiled wall time must be treated as separate optimization problems.")
    story += [image(charts["baseline"], 7.15 * inch, 2.2 * inch), Spacer(1, 2)]
    baselines = [
        ["View", "Torch-Spyre", "SenDNN", "Torch gap", "Interpretation"],
        ["Prefill device", "301.442 ms", "190.406 ms", "111.036 ms", "Compiler and schedule realization"],
        ["Decode device", "153.592 ms", "123.961 ms", "29.631 ms", "Attention/KV memory service"],
        ["Prefill wall", "1293.896 ms", "195.535 ms", "1098.361 ms", "CPU embedding + region overhead + device"],
        ["Decode wall", "1105.272 ms", "132.884 ms", "972.388 ms", "CPU embedding dominates"],
        ["1 prefill + 3 decode device", "762.218 ms", "562.288 ms", "199.930 ms", "Device request proxy"],
        ["1 prefill + 3 decode wall", "4609.713 ms", "594.187 ms", "4015.526 ms", "Historical end-to-end request"],
    ]
    story += [styled_table(baselines, [1.37 * inch, 1.02 * inch, 1.02 * inch, 1.02 * inch, 2.7 * inch])]
    story += [Spacer(1, 5), P("Parity gates: prefill device <=192.310 ms, decode device <=125.200 ms, prefill wall <=197.490 ms, decode wall <=134.213 ms, with identical output tokens.", "callout")]
    story += [PageBreak()]

    # Wall
    story += section_header("04", "Why wall time is much worse", "Only 10.1% of the prefill wall gap and 3.0% of the decode wall gap is the device-program difference.")
    story += [image(charts["wall_budget"], 7.15 * inch, 2.25 * inch), Spacer(1, 3)]
    wall_table = [
        ["Lever", "Prefill opportunity", "Decode opportunity", "One-prefill / three-decode wall effect"],
        ["Device embedding", "~846 ms", "~837 ms/token", "~3356 ms, 72.8% of Torch request wall"],
        ["One phase submission", "~64 ms", "~92 ms/token", "~341 ms, 7.4%"],
        ["Device-program parity", "111 ms", "30 ms/token", "~200 ms, 4.3%"],
        ["Remaining host cleanup", "~77 ms", "~14 ms/token", "~119 ms, 2.6%"],
    ]
    story += [styled_table(wall_table, [1.6 * inch, 1.45 * inch, 1.45 * inch, 2.65 * inch])]
    story += [Spacer(1, 6), callout("Device and wall gates are both required. A faster host path must not hide a compiler regression; a fast kernel sum must not hide CPU fallbacks and launch gaps.", RL_ORANGE, colors.HexColor("#FFF6E8"))]
    story += [Spacer(1, 6), P("The run log reports aten.embedding.default falling back to CPU. Torch then launches one compiled region per layer and five to seven device kernels per layer. SenDNN includes embedding and the entire phase in one prepared device program.", "body")]
    story += [PageBreak()]

    # Prefill
    story += section_header("05", "Prefill: same PT work, different realization", "Torch and SenDNN report nearly identical ideal PT work: 115.582 ms versus 115.587 ms for the 40-layer computation.")
    story += [image(charts["prefill"], 7.15 * inch, 2.43 * inch)]
    story += [image(charts["pt_util"], 7.15 * inch, 2.05 * inch)]
    story += [P("Torch realizes 39.49% of the block PT proxy. SenDNN realizes 60.71% across the complete prefill phase. Holding Torch's 8.752 ms one-off work fixed, parity requires 63.63% aggregate block realization - not perfect PT efficiency.", "caption")]
    story += [PageBreak()]

    # Prefill actions
    story += section_header("06", "Prefill opportunity budget", "Attention ownership and true-BMM SwiGLU are both required; neither closes the gap alone.")
    prefill_actions = [
        ["Region", "Current/layer", "Target/layer", "40-layer saving", "Concrete missing ingredient"],
        ["Q projection + norm", "0.533 ms", "0.400 ms", "5.32 ms", "Carry normalized activation ownership into projection"],
        ["K + QK + softmax", "1.434 ms", "0.450 ms", "39.37 ms", "K peer relayout + tiled causal attention + LX softmax state"],
        ["V + AV + output", "1.056 ms", "0.720 ms", "13.46 ms", "V peer ingress + AV state retention + output handoff"],
        ["True-BMM SwiGLU", "4.225 ms", "2.900 ms", "52.98 ms", "Shared gate/up ownership + PT tiling + weight double buffer"],
        ["Total", "7.317 ms block", "4.539 ms block", "111.13 ms", "Matches the measured 111.04 ms gap"],
    ]
    story += [styled_table(prefill_actions, [1.22 * inch, 0.9 * inch, 0.9 * inch, 0.95 * inch, 3.18 * inch]), Spacer(1, 7)]
    story += [callout("A K-only optimization is insufficient. Prior isolated evidence improved 385.609 us to 320.294 us by retaining K, yet remained behind 227.143 us because V and generic max/sum/BMM state still materialized through HBM.")]
    story += [Spacer(1, 8)]
    story += [P("Production acceptance: combined Q/K/QK/V/AV/output <=1.570 ms/layer; SwiGLU <=2.900 ms/layer; no ReStickifyOpHBM on the selected attention edge; ideal PT work unchanged; full phase <=192.310 ms.", "body")]
    story += [PageBreak()]

    # Decode
    story += section_header("07", "Decode: attention is the only large device pool", "The compulsory-weight and KV model makes the bottleneck unambiguous.")
    story += [image(charts["decode_bw"], 7.15 * inch, 2.35 * inch)]
    story += [image(charts["decode_budget"], 7.15 * inch, 2.22 * inch)]
    story += [P("Torch attention/projection/KV moves 85.998 MB of useful payload per layer in 1.4135 ms, or 60.84 GB/s. Parity requires 0.685-0.744 ms/layer, or 116-126 GB/s. SwiGLU already reaches 136.45 GB/s and has only 2.34 ms of full-phase modeled headroom.", "caption")]
    story += [PageBreak()]

    # Decode exact edges
    story += section_header("08", "Decode attention: exact communication anatomy", "The post-LXOpt SDSCs prove the producer, tensor, consumer, input ordinal, core placement, and transfer kind.")
    story += [topology_diagram(), Spacer(1, 4)]
    edge_table = [
        ["Exact edge / role", "SenDNN topology", "Collective category", "Torch action"],
        ["K/cache tile -> QK Restickify", "32 singleton -> 32 singleton; 93.75% remote", "Permutation / all-to-all-like", "Relayout loaded tile in LX; no HBM restickified copy"],
        ["Q/head-break -> QK BMM", "32 shards -> 8 groups x4", "Four-way subgroup all-gather", "Choose GQA-compatible QK division"],
        ["Logits -> Max; Exp -> Sum", "32 shards -> 8 reducer cores", "Four-to-one gather feeding reduction", "Retain reduction operands and state in LX"],
        ["Max -> Sub; reciprocal -> Mul", "8 results -> 8 groups x4", "Four-way broadcast", "Emit peer multicast; preserve ownership"],
        ["Softmax and V -> AV", "32 shards -> 16 groups x2", "Two-way subgroup all-gather", "Stream V directly into AV consumers"],
        ["New KV -> cache Scatter", "16 pieces -> owner", "Gather-to-owner", "In-place persistent cache update"],
    ]
    story += [styled_table(edge_table, [1.75 * inch, 1.65 * inch, 1.45 * inch, 2.3 * inch])]
    story += [Spacer(1, 4), P("The Max/Sum paths are not network reductions. Peer transfer gathers shards to reducer cores; the Max or Sum op performs the arithmetic; the result is then broadcast. Together they implement an all-reduce-like pattern.", "caption")]
    story += [PageBreak()]

    # SMC
    story += section_header("09", "Generated SMCs confirm different physical paths", "The ISA distinguishes memory load/store opcodes from peer scratchpad load/store opcodes.")
    story += [image(charts["smc_paths"], 7.15 * inch, 2.36 * inch), Spacer(1, 4)]
    smc_table = [
        ["Static program property", "SenDNN prefill", "Torch prefill", "SenDNN decode", "Torch decode"],
        ["External device jobs", "1", "204", "1", "244"],
        ["Logical init payload", "15.616 MB", "25.329 MB", "14.116 MB", "28.881 MB"],
        ["Patch share", "58.9%", "27.6%", "62.9%", "28.2%"],
        ["SYNC sites / marginal layer", "200", "267", "150", "256"],
        ["PT FMA slots / marginal layer", "1932", "2960", "716", "3828"],
        ["PT loop-count slots", "263", "177", "261", "177"],
    ]
    story += [styled_table(smc_table, [1.72 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch])]
    story += [Spacer(1, 4), P("These are static delivered-program counts, not executed instructions or utilization counters. Their value is structural: SenDNN chooses peer paths, patch specialization, fewer synchronization sites, and a more compact looped PT form.", "caption")]
    story += [PageBreak()]

    # SDSC and ring
    story += section_header("10", "What the SDSCs say - and what they do not", "All 130 folded relayouts are STCDPOpLx and every source/destination map requires remote-core data.")
    story += [image(charts["relayout_demand"], 7.15 * inch, 2.5 * inch), Spacer(1, 3)]
    sdsc_metrics = [
        ["Phase", "Final SDSCs", "Folded LX relayouts", "Expanded instances", "Logical remote destination bytes"],
        ["Prefill", "227", "55", "647", "4,540,039,936"],
        ["Decode", "243", "75", "889", "242,810,112"],
    ]
    story += [styled_table(sdsc_metrics, [1.2 * inch, 1.25 * inch, 1.45 * inch, 1.4 * inch, 1.85 * inch]), Spacer(1, 6)]
    story += [callout("The SDSCs prove exact operation-level topology. They do not provide a final IBUFF range per SDSC, dynamic burst counts, measured link bytes, or executed stall time.", RL_ORANGE, colors.HexColor("#FFF6E8"))]
    story += [Spacer(1, 6), P("The largest prefill peer demand is BMM operand redistribution. Decode is 72.77% BMM input and 24.25% attention restickify. Softmax and KV scatter are semantically important but small in logical byte volume.", "body")]
    story += [PageBreak()]

    # Ring
    story += section_header("11", "Ring model: cheap enough to use, too small to explain the gap", "The correct proxy routes exact placement demand on a global 32-core bidirectional ring and examines the hottest directed link.")
    story += [image(charts["ring"], 7.15 * inch, 2.42 * inch), Spacer(1, 5)]
    ring_table = [
        ["Quantity", "Prefill", "Decode", "Status"],
        ["Logical remote destination demand", "4.540 GB", "242.81 MB", "Placement-derived"],
        ["Expanded-unicast hop bytes", "25.017 GB", "1.983 GB", "Modeled shortest paths"],
        ["Hottest directed link", "438.21 MB", "37.83 MB", "Modeled proxy"],
        ["Service at 136.457 GB/s", "3.211 ms", "0.277 ms", "Lower-bound-like proxy"],
        ["Phase-average hot-link occupancy", "1.69%", "0.22%", "Not a hardware counter"],
    ]
    story += [styled_table(ring_table, [1.95 * inch, 1.25 * inch, 1.25 * inch, 2.7 * inch])]
    story += [Spacer(1, 6), callout("Do not optimize ring bandwidth first. Use the ring to eliminate HBM materialization and expose PT/HBM overlap. Peak active-window utilization may be high, but phase-average modeled occupancy is low.", RL_PURPLE, colors.HexColor("#F0EDFA"))]
    story += [PageBreak()]

    # Waterfall
    story += section_header("12", "Non-overlapping opportunity accounting", "The proposed budgets reconcile to the measured device gaps without assigning the same millisecond twice.")
    story += [image(charts["waterfall"], 7.15 * inch, 2.42 * inch), Spacer(1, 4)]
    opp = [
        ["Phase", "Opportunity", "Saving", "Region effect", "Phase effect"],
        ["Prefill", "Attention + projections", "58.15 ms", "48.1% lower; 1.93x", "19.29%"],
        ["Prefill", "True-BMM SwiGLU", "52.98 ms", "31.4% lower; 1.46x", "17.58%"],
        ["Decode A", "Attention, MLP unchanged", "29.14 ms", "51.5% attention reduction", "19.0%"],
        ["Decode B", "Attention relaxed", "26.78 ms", "47.4% attention reduction", "17.5%"],
        ["Decode B", "MLP to 140 GB/s floor", "2.34 ms", "2.5% MLP reduction", "1.5%"],
    ]
    story += [styled_table(opp, [1.0 * inch, 2.05 * inch, 1.0 * inch, 1.75 * inch, 1.1 * inch])]
    story += [Spacer(1, 5), P("The phase-DAG, placement, synchronization, and patch levers are enabling mechanisms inside these budgets. They must not be added again as independent device savings.", "caption")]
    story += [PageBreak()]

    # Compiler architecture
    story += section_header("13", "Concrete Torch-Spyre architecture", "Build the capability as a legal, typed, emitted-path-verified compiler feature - not an attention-specific environment-variable heuristic.")
    story += [compiler_stack_diagram(), Spacer(1, 4)]
    seams = [
        ["Seam", "Required change"],
        ["scheduler.py", "Retain ordered internal bundles and cross-bundle dependencies instead of call_kernel per node."],
        ["bundle.py", "Emit a phase artifact with stable tensor and edge identities."],
        ["lx_relayout.py", "Make source, consumer, operand ordinal, and target view first-class in LXRelayoutPlan."],
        ["scratchpad/allocator.py", "Allocate S1/S2 atomically over phase lifetimes; preserve DXP_LX_FRAC_AVAIL=0.2."],
        ["work division / placement", "Enumerate producer-compatible, consumer-compatible, multicast, unicast, and HBM alternatives."],
        ["superdsc.py / SMC", "Serialize exact ownership and require realized peer transport proof."],
        ["async_compile.py / kernel_runner.py", "Compile, bind, cache, and launch one prepared phase plan."],
        ["fallbacks.py", "Replace CPU embedding fallback with a real Gather/Embedding lowering."],
    ]
    story += [styled_table(seams, [1.75 * inch, 5.4 * inch])]
    story += [PageBreak()]

    # Cost model
    story += section_header("14", "Cost-model policy", "Jointly choose work division, physical placement, transport, and lifetime for a closed dependence region.")
    story += [callout("Cost(P) = critical_path(compute, HBM, LX link/injection/drain, sync, legal overlap) + spill penalty + uncertainty guardband", RL_BLUE, colors.HexColor("#EAF0FC")), Spacer(1, 8)]
    policy = [
        ["Step", "Concrete rule", "Failure mode prevented"],
        ["1. Form closed region", "Include real producer, aliases, relayout, consumers, shared allocations, and boundary traffic.", "Semantically incoherent ownership"],
        ["2. Enumerate bounded candidates", "Default, dimension-major, cohort-contiguous, rotations/reflection, neighboring placements.", "Unbounded search and policy churn"],
        ["3. Route exact demands", "Count CW/CCW hot-link bytes, injection, drain, multicast realization, and HBM alternative.", "Aggregate-byte utilization errors"],
        ["4. Score critical path", "Include compute, service, barriers, spill, and only DAG-proven overlap.", "Summing hidden work or assuming overlap"],
        ["5. Commit transactionally", "Validate capacity and backend capability before mutating graph; retry default on failure.", "Half-applied placement or LX fallback"],
        ["6. Prove realized path", "Export requested and emitted placement/transport plus per-edge SMC range.", "Planner telemetry mistaken for hardware realization"],
    ]
    story += [styled_table(policy, [1.3 * inch, 3.95 * inch, 1.9 * inch]), Spacer(1, 7)]
    story += [P("Initial measured coefficients: one-way fixed 0.14925 us, one-way effective 136.457 GB/s, and 512 KiB balanced-duplex aggregate 255.439 GB/s. Native multicast, matched HBM read/write, bank pressure, sync, and overlap require calibration or guardbands.", "body")]
    story += [PageBreak()]

    # Roadmap
    story += section_header("15", "Plan of action", "The milestones are ordered by dependency and falsifiability, not by calendar estimate.")
    story += [image(charts["roadmap"], 7.15 * inch, 2.48 * inch), Spacer(1, 5)]
    roadmap_table = [
        ["Milestone", "Build", "Hard acceptance gate"],
        ["M0", "Permanent 40-layer parity harness", "Device, span, wall, launch counts, tokens, allocations, relayouts"],
        ["M1", "Device embedding + optimizer-visible phase DAG", "One plan/launch; cross-bundle LX probe has no HBM roundtrip"],
        ["M2", "Decode Q/K/V -> softmax -> AV -> output -> KV scatter", "Attention <=0.744 then <=0.685 ms/layer; phase <=125.200 ms"],
        ["M3", "Production tiled causal prefill attention", "Combined attention <=1.570 ms/layer; emitted peer proof"],
        ["M4", "Production true-BMM SwiGLU", "<=2.900 ms/layer; no decode regression from 136.45 GB/s"],
        ["M5", "One-offs, address binding, token feedback", "Prefill <=192.310 ms and both wall parity gates"],
    ]
    story += [styled_table(roadmap_table, [0.62 * inch, 3.0 * inch, 3.53 * inch])]
    story += [PageBreak()]

    # Acceptance
    story += section_header("16", "Acceptance scorecard", "No candidate passes because a planner log looks good or because the raw SMC is smaller.")
    acceptance = [
        ["Gate", "Required evidence", "Pass condition"],
        ["Correctness", "Isolated CPU comparison + exact full-model tokens", "No numerical or token regression"],
        ["Phase visibility", "One dependency/lifetime plan", "Not runtime-only batching"],
        ["On-chip realization", "Matched peer L3 opcodes + SDSC/IBUFF edge mapping", "No HBM pair on selected edge"],
        ["Decode locality", "Useful service and physical HBM counters", "116-126 GB/s equivalent; no traffic amplification"],
        ["Synchronization", "Executed wait/stall time", "Lower dynamic stalls; static SYNC is support only"],
        ["Prefill arithmetic", "Ideal PT cycles", "Unchanged while latency falls"],
        ["Device parity", "40-layer phase timings", "Prefill <=192.310; decode <=125.200 ms"],
        ["Wall parity", "Python wall and accelerator span", "Prefill <=197.490; decode <=134.213 ms"],
    ]
    story += [styled_table(acceptance, [1.35 * inch, 3.55 * inch, 2.25 * inch]), Spacer(1, 8)]
    story += [metric_cards([
        ("<=192.31", "Prefill device ms", TEAL),
        ("<=125.20", "Decode device ms", TEAL),
        ("<=197.49", "Prefill wall ms", BLUE),
        ("<=134.21", "Decode wall ms", BLUE),
    ])]
    story += [Spacer(1, 8), callout("Stop rule: reject any change that improves wall time while regressing either device phase, or claims LX ownership without emitted-path and post-run evidence.", RL_RED, colors.HexColor("#FCEDEC"))]
    story += [PageBreak()]

    # Evidence and references
    story += section_header("17", "Evidence boundaries and sources", "Measured facts, derived calculations, and modeled proxies are intentionally separated.")
    evidence = [
        ["Evidence type", "Examples", "What it supports"],
        ["Measured", "Trace kernel duration, phase wall time, generated tokens", "Latency and correctness"],
        ["Compiler proxy", "Ideal PT cycles, allocation nodes, folded SDSCs", "Schedule structure and opportunity location"],
        ["Static SMC", "Opcode sites, patches, SYNC, PT loop form", "Chosen physical path and program construction"],
        ["Derived", "PT proxy, useful GB/s, opportunity percentages", "Comparable accounting under stated assumptions"],
        ["Modeled", "Ring hot-link load, 140 GB/s floor, phase-equivalent SenDNN service", "Directional planning, not counters"],
        ["Unavailable", "Per-SDSC final IBUFF range, dynamic peer bytes/stalls", "Prevents exact instruction-level edge timing"],
    ]
    story += [styled_table(evidence, [1.25 * inch, 2.9 * inch, 3.0 * inch]), Spacer(1, 8)]
    refs = [
        "[1] results/2026-07-25/full_model_comparison/metrics.json - trace-derived full-model metrics.",
        "[2] outputs/sendnn_vs_torch_spyre_gap_analysis.md - PT, bandwidth, latency, and wall-gap study.",
        "[3] runbooks/sendnn_vs_torch_spyre_smc_study.md - generated-SMC and post-LXOpt analysis.",
        "[4] results/.../sdsc/sendnn_sdsc_lx_attribution.json and .csv - 130 exact relayout records.",
        "[5] results/.../smc/generated_smc_study_summary.json - decoded ISA packet and opcode aggregates.",
        "[6] AIU 1.0 Rapid Core ISA Specification, revision 2026-01-21 - packet and L3 opcode interpretation.",
        "[7] ah/communication-cost-model study - measured LX coefficients, joint-placement legality, and owner-compute experiments.",
    ]
    story += [P("<b>Primary sources</b>", "h2")]
    for ref in refs:
        story += [P(ref, "body_small")]
    story += [Spacer(1, 6), P("Prepared July 25, 2026. All modeled figures are explicitly labeled and should be refreshed when dynamic L3/ring counters or final SDSC-to-IBUFF mapping become available.", "caption")]
    return story


def main() -> None:
    charts = make_charts()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(letter),
        rightMargin=36,
        leftMargin=36,
        topMargin=34,
        bottomMargin=34,
        title="Torch-Spyre to SenDNN Performance Parity",
        author="OpenAI Codex",
        subject="Granite 3.3 8B full-model performance analysis and action plan",
    )
    doc.build(build_story(charts), onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    main()
