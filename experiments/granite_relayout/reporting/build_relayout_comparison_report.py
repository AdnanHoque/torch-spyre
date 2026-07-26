#!/usr/bin/env python3
"""Build the focused SenDNN vs Torch-Spyre relayout analysis PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
CHART_DIR = ROOT / "tmp" / "pdfs" / "relayout_charts"
OUTPUT = ROOT / "output" / "pdf" / "granite_sendnn_torch_spyre_relayout_analysis.pdf"
CHART_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

PAGE_W, PAGE_H = landscape(letter)

# Visual system shared with the broader parity study.
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


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
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
    fig.savefig(path, dpi=230, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_charts() -> dict[str, Path]:
    configure_matplotlib()
    charts: dict[str, Path] = {}

    # Full-model latency matrix.
    configs = ["SenDNN\nLXOpt off", "SenDNN\nLXOpt on", "Torch\nrelayout off", "Torch\nall-gather", "Torch\nall relayouts"]
    cfg_colors = [SLATE, TEAL, BLUE, PURPLE, ORANGE]
    prefill = [360.694, 190.406, 378.584, 368.216, 403.744]
    decode = [129.879, 123.961, 160.360, 159.741, 157.925]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.2))
    for ax, title, values, ylim in zip(
        axes,
        ["Prefill: phase-wide planning changes the result", "Decode: current Torch coverage is too narrow"],
        [prefill, decode],
        [(0, 445), (0, 180)],
    ):
        bars = ax.bar(np.arange(len(values)), values, color=cfg_colors, width=0.68)
        ax.set_xticks(np.arange(len(values)), configs)
        ax.set_ylabel("Device-program time (ms)")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + (8 if ylim[1] > 400 else 3), f"{value:.1f}", ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    charts["latency"] = save_chart(fig, "01_latency_matrix.png")

    # Measured changes from relayout policies.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.15))
    pre_names = ["SenDNN full\nLXOpt plan", "Torch\nall-gather", "Torch added\ndense A2A"]
    pre_delta = [170.288, 10.368, -35.528]
    dec_names = ["SenDNN full\nLXOpt plan", "Torch\nall-gather", "Torch added\nsmall A2A"]
    dec_delta = [5.919, 0.619, 1.816]
    for ax, title, names, values in zip(
        axes,
        ["Prefill latency saved", "Decode latency saved"],
        [pre_names, dec_names],
        [pre_delta, dec_delta],
    ):
        bar_colors = [TEAL, PURPLE, GREEN if values[2] > 0 else RED]
        bars = ax.bar(np.arange(3), values, color=bar_colors, width=0.62)
        ax.axhline(0, color=NAVY, linewidth=0.8)
        ax.set_xticks(np.arange(3), names)
        ax.set_ylabel("Milliseconds saved; negative is regression")
        ax.set_title(title)
        pad = max(abs(v) for v in values) * 0.06
        for bar, value in zip(bars, values):
            va = "bottom" if value >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, value + (pad if value >= 0 else -pad), f"{value:+.3f}", ha="center", va=va, fontsize=8, fontweight="bold")
    axes[0].set_ylim(-55, 190)
    axes[1].set_ylim(-0.5, 7.0)
    fig.tight_layout(w_pad=2.4)
    charts["delta"] = save_chart(fig, "02_policy_deltas.png")

    # Torch relayout payload inventory.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.25))
    pre_names = ["BMM\nall-gather", "Post-BMM\nA2A", "SiLU\nA2A", "Mul\nA2A", "Post-BMM\nA2A"]
    pre_sizes = [4.0, 4.0, 12.5, 12.5, 4.0]
    pre_colors = [PURPLE, ORANGE, ORANGE, ORANGE, ORANGE]
    dec_names = ["BMM\nall-gather", "Attention\npost-BMM A2A", "MLP\npost-BMM A2A"]
    dec_sizes = [0.5, 0.5, 0.5]
    dec_colors = [PURPLE, GREEN, GREEN]
    for ax, title, names, values, cols, ylim in zip(
        axes,
        ["Torch prefill: five relayout tensors per layer", "Torch decode: three small tensors per layer"],
        [pre_names, dec_names],
        [pre_sizes, dec_sizes],
        [pre_colors, dec_colors],
        [(0, 14.5), (0, 0.7)],
    ):
        bars = ax.bar(np.arange(len(values)), values, color=cols, width=0.62)
        ax.set_xticks(np.arange(len(values)), names)
        ax.set_ylabel("FP16 tensor size (MiB)")
        ax.set_title(title)
        ax.set_ylim(*ylim)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ylim[1] * 0.03, f"{value:g}", ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout(w_pad=2.2)
    charts["payloads"] = save_chart(fig, "03_torch_payloads.png")

    # Coverage and PT realization.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.1))
    x = np.arange(2)
    w = 0.34
    axes[0].bar(x - w / 2, [5.0, 3.0], w, color=BLUE, label="Torch")
    axes[0].bar(x + w / 2, [647 / 40, 889 / 40], w, color=TEAL, label="SenDNN")
    axes[0].set_xticks(x, ["Prefill", "Decode"])
    axes[0].set_ylabel("Expanded relayout instances per layer")
    axes[0].set_title("Relayout coverage")
    axes[0].legend(loc="upper left")
    for xpos, value in zip([x[0] - w / 2, x[1] - w / 2, x[0] + w / 2, x[1] + w / 2], [5, 3, 647 / 40, 889 / 40]):
        axes[0].text(xpos, value + 0.7, f"{value:.1f}", ha="center", fontsize=8, fontweight="bold")
    axes[0].set_ylim(0, 25.5)

    values = [32.05, 60.71]
    bars = axes[1].bar([0, 1], values, color=[SLATE, TEAL], width=0.58)
    axes[1].set_xticks([0, 1], ["SenDNN\nLXOpt off", "SenDNN\nLXOpt on"])
    axes[1].set_ylabel("Ideal PT time / measured prefill (%)")
    axes[1].set_title("Same 115.587 ms PT floor, different realization")
    axes[1].set_ylim(0, 72)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 2.1, f"{value:.2f}%", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout(w_pad=2.4)
    charts["coverage_pt"] = save_chart(fig, "04_coverage_and_pt.png")

    # SenDNN relayout destination demand by consumer family.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.25))
    pre_vals = [4274.218, 128.975, 62.915, 39.322, 31.457, 3.153]
    pre_names = ["BMM", "Exx2", "Mul", "Restickify", "LayerNorm", "Other"]
    dec_vals = [176.698, 58.884, 5.080, 1.905, 0.643, 0.600]
    dec_names = ["BMM", "Restickify", "Softmax", "Max/Sum", "Norm", "Other"]
    palette = [BLUE, TEAL, ORANGE, PURPLE, GREEN, SLATE]
    for ax, title, values, names in zip(
        axes,
        ["Prefill: 4.540 GB logical remote demand", "Decode: 242.810 MB logical remote demand"],
        [pre_vals, dec_vals],
        [pre_names, dec_names],
    ):
        ax.pie(values, colors=palette, startangle=90, wedgeprops={"linewidth": 1, "edgecolor": "white"})
        ax.set_title(title)
        ax.legend(names, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7)
    fig.tight_layout()
    charts["demand"] = save_chart(fig, "05_sendnn_demand.png")

    # Cross-stack accounting, explicitly not a causal decomposition.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.0))
    for ax, title, total_gap, lx_gain, remain in [
        (axes[0], "Prefill current-stack gap", 188.178, 170.288, 17.890),
        (axes[1], "Decode current-stack gap", 36.399, 5.919, 30.480),
    ]:
        ax.barh([0], [lx_gain], color=TEAL, height=0.44, label="SenDNN LXOpt-plan gain")
        ax.barh([0], [remain], left=[lx_gain], color=SLATE, height=0.44, label="Residual cross-stack difference")
        ax.set_xlim(0, total_gap * 1.08)
        ax.set_yticks([])
        ax.set_xlabel("Milliseconds")
        ax.set_title(title)
        ax.text(lx_gain / 2, 0, f"{lx_gain:.1f} ms", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        if remain > total_gap * 0.15:
            ax.text(lx_gain + remain / 2, 0, f"{remain:.1f} ms", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, -0.42), fontsize=7)
    fig.tight_layout(w_pad=2.4)
    charts["accounting"] = save_chart(fig, "06_cross_stack_accounting.png")

    return charts


BASE = getSampleStyleSheet()
styles = {
    "cover_title": ParagraphStyle("CoverTitle", parent=BASE["Title"], fontName="Helvetica-Bold", fontSize=29, leading=33, textColor=colors.white, alignment=TA_LEFT, spaceAfter=11),
    "cover_sub": ParagraphStyle("CoverSub", parent=BASE["Normal"], fontName="Helvetica", fontSize=13, leading=18, textColor=colors.HexColor("#DDE7F6"), alignment=TA_LEFT),
    "h1": ParagraphStyle("H1", parent=BASE["Heading1"], fontName="Helvetica-Bold", fontSize=20, leading=23, textColor=RL_NAVY, spaceAfter=7),
    "h2": ParagraphStyle("H2", parent=BASE["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=15, textColor=RL_BLUE, spaceBefore=3, spaceAfter=4),
    "body": ParagraphStyle("Body", parent=BASE["BodyText"], fontName="Helvetica", fontSize=8.8, leading=11.7, textColor=RL_INK, spaceAfter=4),
    "body_small": ParagraphStyle("BodySmall", parent=BASE["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.4, textColor=RL_INK, spaceAfter=3),
    "caption": ParagraphStyle("Caption", parent=BASE["BodyText"], fontName="Helvetica", fontSize=6.8, leading=8.5, textColor=RL_SLATE, spaceBefore=2),
    "callout": ParagraphStyle("Callout", parent=BASE["BodyText"], fontName="Helvetica-Bold", fontSize=9.8, leading=13.2, textColor=RL_NAVY),
    "metric": ParagraphStyle("Metric", parent=BASE["BodyText"], fontName="Helvetica-Bold", fontSize=17, leading=19, textColor=RL_NAVY, alignment=TA_CENTER),
    "metric_label": ParagraphStyle("MetricLabel", parent=BASE["BodyText"], fontName="Helvetica", fontSize=7.1, leading=8.6, textColor=RL_SLATE, alignment=TA_CENTER),
    "table": ParagraphStyle("Table", parent=BASE["BodyText"], fontName="Helvetica", fontSize=6.9, leading=8.5, textColor=RL_INK),
    "table_head": ParagraphStyle("TableHead", parent=BASE["BodyText"], fontName="Helvetica-Bold", fontSize=6.9, leading=8.5, textColor=colors.white),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def section_header(number: str, title: str, subtitle: str | None = None):
    items = [P(f"{number}  {title}", "h1")]
    if subtitle:
        items.append(P(subtitle, "body"))
    items.append(HRFlowable(width="100%", thickness=1.1, color=RL_GRID, spaceBefore=1, spaceAfter=7))
    return items


def callout(text: str, accent=RL_TEAL, background=RL_LIGHT):
    t = Table([[P(text, "callout")]], colWidths=[7.15 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 6, accent),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def metric_cards(cards: list[tuple[str, str, str]]):
    width = 7.15 * inch / len(cards)
    nested = []
    for value, label, accent in cards:
        t = Table([[P(value, "metric")], [P(label, "metric_label")]], colWidths=[width])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.8, RL_GRID),
                    ("LINEABOVE", (0, 0), (-1, 0), 4, colors.HexColor(accent)),
                    ("TOPPADDING", (0, 0), (-1, 0), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
                    ("TOPPADDING", (0, 1), (-1, 1), 2),
                    ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
                ]
            )
        )
        nested.append(t)
    outer = Table([nested], colWidths=[width] * len(cards), hAlign="LEFT")
    outer.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return outer


def styled_table(data, widths, header=True):
    rows = []
    for row_idx, row in enumerate(data):
        rows.append([P(str(value), "table_head" if header and row_idx == 0 else "table") for value in row])
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), RL_NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, RL_GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]
    for row_idx in range(1, len(rows)):
        if row_idx % 2 == 0:
            commands.append(("BACKGROUND", (0, row_idx), (-1, row_idx), RL_LIGHT))
    t.setStyle(TableStyle(commands))
    return t


def two_col(left, right, widths=(3.53 * inch, 3.53 * inch)):
    t = Table([[left, right]], colWidths=list(widths), hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def image(path: Path, width: float, height: float):
    im = Image(str(path), width=width, height=height)
    im.hAlign = "CENTER"
    return im


def arrow(d: Drawing, x1: float, y1: float, x2: float, y2: float, color, width=1.6):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    if x2 >= x1:
        pts = [x2, y2, x2 - 7, y2 + 4, x2 - 7, y2 - 4]
    else:
        pts = [x2, y2, x2 + 7, y2 + 4, x2 + 7, y2 - 4]
    d.add(Polygon(pts, fillColor=color, strokeColor=color))


def planning_diagram() -> Drawing:
    d = Drawing(720, 220)
    d.add(String(18, 198, "Torch-Spyre: independently planned, repaired afterward", fontName="Helvetica-Bold", fontSize=11, fillColor=RL_BLUE))
    torch_boxes = [(18, "Producer\nwork division"), (120, "S1 LX\nallocation"), (222, "Standalone\nSHUFFLE"), (324, "S2 LX\nallocation"), (426, "Consumer\nwork division")]
    for x, label in torch_boxes:
        d.add(Rect(x, 135, 82, 42, 6, fillColor=colors.HexColor("#EAF0FC"), strokeColor=RL_BLUE))
        parts = label.split("\n")
        d.add(String(x + 41, 158, parts[0], textAnchor="middle", fontName="Helvetica-Bold", fontSize=7.2, fillColor=RL_NAVY))
        d.add(String(x + 41, 145, parts[1], textAnchor="middle", fontSize=7.0, fillColor=RL_SLATE))
    for x in [100, 202, 304, 406]:
        arrow(d, x, 156, x + 20, 156, RL_RED)
    d.add(String(540, 158, "Full edge barrier", fontName="Helvetica-Bold", fontSize=8, fillColor=RL_RED))
    d.add(String(540, 144, "Two LX buffers", fontSize=7.5, fillColor=RL_SLATE))

    d.add(String(18, 104, "SenDNN: phase-wide ownership and schedule", fontName="Helvetica-Bold", fontSize=11, fillColor=RL_TEAL))
    d.add(Rect(18, 26, 684, 62, 8, fillColor=colors.HexColor("#E7F7F5"), strokeColor=RL_TEAL, strokeWidth=1.4))
    labels = ["Producer", "Owner map", "Grouped peer route", "Consumer layout", "Next consumer"]
    xs = [38, 168, 302, 444, 576]
    for x, label in zip(xs, labels):
        d.add(Rect(x, 44, 96, 28, 5, fillColor=colors.white, strokeColor=RL_TEAL))
        d.add(String(x + 48, 55, label, textAnchor="middle", fontSize=7.3, fontName="Helvetica-Bold", fillColor=RL_INK))
    for x1, x2 in zip([134, 264, 398, 540], [168, 302, 444, 576]):
        arrow(d, x1, 58, x2, 58, RL_PURPLE, 2.3)
    d.add(String(360, 31, "One critical path: LX lifetime + route + HBM avoidance + overlap + local synchronization", textAnchor="middle", fontSize=7.5, fillColor=RL_SLATE))
    return d


def topology_diagram() -> Drawing:
    d = Drawing(720, 205)
    panels = [
        (12, "All-gather", "32 mb shards", "4 mb groups x 8 replicas", RL_PURPLE),
        (246, "Dense all-to-all", "8 out x 4 mb", "32 mb shards", RL_ORANGE),
        (480, "SenDNN owner groups", "Per-edge source pieces", "Grouped / multicast targets", RL_TEAL),
    ]
    for x, title, src, dst, color in panels:
        d.add(Rect(x, 18, 220, 172, 8, fillColor=colors.white, strokeColor=RL_GRID))
        d.add(Rect(x, 160, 220, 30, 8, fillColor=color, strokeColor=color))
        d.add(String(x + 110, 171, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.white))
        d.add(String(x + 48, 133, src, textAnchor="middle", fontSize=7.2, fillColor=RL_SLATE))
        d.add(String(x + 172, 133, dst, textAnchor="middle", fontSize=7.2, fillColor=RL_SLATE))
        if title == "All-gather":
            for i in range(8):
                yy = 112 - i * 10
                d.add(Rect(x + 22, yy, 8, 7, 1.5, fillColor=colors.HexColor("#EEEAFB"), strokeColor=color))
                if i < 4:
                    arrow(d, x + 31, yy + 3.5, x + 142, 106 - i * 18, color, 0.8)
            for i in range(4):
                d.add(Rect(x + 142, 96 - i * 18, 58, 11, 2, fillColor=colors.HexColor("#EEEAFB"), strokeColor=color))
        elif title == "Dense all-to-all":
            for i in range(6):
                sy = 112 - i * 13
                dy = 112 - ((i * 2) % 6) * 13
                d.add(Rect(x + 22, sy, 10, 9, 2, fillColor=colors.HexColor("#FFF0DC"), strokeColor=color))
                d.add(Rect(x + 184, dy, 10, 9, 2, fillColor=colors.HexColor("#FFF0DC"), strokeColor=color))
                arrow(d, x + 33, sy + 4.5, x + 184, dy + 4.5, color, 0.8)
        else:
            for i in range(4):
                d.add(Rect(x + 22, 107 - i * 18, 11, 11, 2, fillColor=colors.HexColor("#E7F7F5"), strokeColor=color))
                arrow(d, x + 34, 112 - i * 18, x + 132, 90, color, 1.0)
            d.add(Rect(x + 132, 72, 64, 36, 5, fillColor=colors.HexColor("#E7F7F5"), strokeColor=color))
            d.add(String(x + 164, 92, "owner", textAnchor="middle", fontName="Helvetica-Bold", fontSize=7.2, fillColor=RL_TEAL))
            d.add(String(x + 164, 80, "group", textAnchor="middle", fontSize=7.2, fillColor=RL_TEAL))
        d.add(String(x + 110, 32, "Source and destination are chosen together" if title == "SenDNN owner groups" else "Explicit S1 -> SHUFFLE -> S2", textAnchor="middle", fontSize=6.8, fillColor=RL_SLATE))
    return d


def decision_diagram() -> Drawing:
    d = Drawing(720, 190)
    steps = [
        (20, "Candidate edge", RL_BLUE),
        (155, "Compatible layout?", RL_TEAL),
        (300, "Peer route cost", RL_PURPLE),
        (445, "Avoided HBM + sync", RL_ORANGE),
        (590, "Net critical path", RL_GREEN),
    ]
    for x, label, color in steps:
        d.add(Rect(x, 112, 112, 38, 7, fillColor=colors.white, strokeColor=color, strokeWidth=1.3))
        d.add(String(x + 56, 127, label, textAnchor="middle", fontName="Helvetica-Bold", fontSize=7.5, fillColor=RL_INK))
    for x1, x2 in zip([132, 267, 412, 557], [155, 300, 445, 590]):
        arrow(d, x1, 131, x2, 131, RL_NAVY, 1.3)
    d.add(Rect(114, 28, 492, 48, 8, fillColor=RL_LIGHT, strokeColor=RL_GRID))
    d.add(String(360, 58, "benefit = avoided HBM + avoided restickify + avoided barriers", textAnchor="middle", fontName="Helvetica-Bold", fontSize=8.3, fillColor=RL_NAVY))
    d.add(String(360, 42, "            - peer bytes - ring contention - LX pressure - exposed synchronization", textAnchor="middle", fontName="Helvetica-Bold", fontSize=8.3, fillColor=RL_NAVY))
    d.add(String(360, 15, "Accept only when emitted SMC and device timing prove a positive critical-path change.", textAnchor="middle", fontSize=7.5, fillColor=RL_SLATE))
    return d


def page_decor(canvas, doc):
    canvas.saveState()
    if doc.page == 1:
        canvas.setFillColor(RL_NAVY)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.setFillColor(RL_TEAL)
        canvas.rect(0, 0, 18, PAGE_H, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#1C3458"))
        canvas.circle(PAGE_W - 70, PAGE_H - 72, 94, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#24456F"))
        canvas.circle(PAGE_W - 25, 75, 126, fill=1, stroke=0)
    else:
        canvas.setStrokeColor(RL_GRID)
        canvas.setLineWidth(0.6)
        canvas.line(36, 28, PAGE_W - 36, 28)
        canvas.setFillColor(RL_SLATE)
        canvas.setFont("Helvetica", 6.7)
        canvas.drawString(36, 17, "Granite relayout analysis | Torch-Spyre vs SenDNN | July 25, 2026")
        canvas.drawRightString(PAGE_W - 36, 17, f"{doc.page}")
        canvas.setFillColor(RL_NAVY)
        canvas.rect(0, PAGE_H - 8, PAGE_W, 8, fill=1, stroke=0)
    canvas.restoreState()


def build_story(charts: dict[str, Path]):
    story = []

    # Cover.
    story += [Spacer(1, 0.62 * inch)]
    story += [P("Granite Relayouts:<br/>Torch-Spyre vs SenDNN", "cover_title")]
    story += [P("Why the same core-to-core primitive produces opposite prefill behavior", "cover_sub")]
    story += [Spacer(1, 0.35 * inch)]
    cover_table = Table(
        [
            [P("Measured answer", "table_head"), P("Engineering conclusion", "table_head")],
            [
                P("SenDNN's full LXOpt plan saves 170.288 ms in prefill and 5.919 ms in decode. Torch's all-gather helps, but four post-hoc dense prefill all-to-alls lose 35.528 ms incrementally.", "body"),
                P("Core-to-core transfer is necessary but not sufficient. Parity requires joint producer-consumer placement, grouped routes, selective cost gating, and phase-wide scheduling.", "body"),
            ],
        ],
        colWidths=[3.45 * inch, 3.45 * inch],
    )
    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24456F")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#F6F9FD")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#4C6688")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD6E6")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story += [cover_table, Spacer(1, 0.34 * inch)]
    story += [P("Workload: Granite 3.3 8B Instruct, 40 decoder layers, B1/S512, FP16, unfused weights, SDPA, five measured generations.", "cover_sub")]
    story += [Spacer(1, 0.12 * inch), P("Analysis scope: device-program latency, final SDSCs, generated SMC topology, and compiler placement structure.", "cover_sub")]
    story += [PageBreak()]

    # Executive A/B.
    story += section_header("01", "Executive answer", "The newly measured SenDNN LXOpt-off counterfactual separates the value of its coordinated LX plan from the remaining stack advantages.")
    story += [metric_cards([
        ("170.288 ms", "SenDNN prefill LXOpt-plan gain", TEAL),
        ("10.368 ms", "Torch prefill all-gather gain", PURPLE),
        ("-35.528 ms", "Torch added prefill A2A regression", RED),
        ("2.435 ms", "Torch total decode relayout gain", GREEN),
    ]), Spacer(1, 7)]
    story += [image(charts["latency"], 7.15 * inch, 2.16 * inch), Spacer(1, 3)]
    summary = [
        ["Configuration", "Prefill", "Decode", "Change versus its off state"],
        ["SenDNN LXOpt off", "360.694 ms", "129.879 ms", "Counterfactual baseline"],
        ["SenDNN LXOpt on", "190.406 ms", "123.961 ms", "47.21% prefill; 4.56% decode reduction"],
        ["Torch relayout off", "378.584 ms", "160.360 ms", "Current-stack baseline"],
        ["Torch all-gather only", "368.216 ms", "159.741 ms", "2.74% prefill; 0.39% decode reduction"],
        ["Torch all relayouts", "403.744 ms", "157.925 ms", "6.65% prefill regression; 1.52% decode reduction"],
    ]
    story += [styled_table(summary, [1.65 * inch, 1.05 * inch, 1.05 * inch, 3.4 * inch]), Spacer(1, 5)]
    story += [callout("Bottom line: Torch now has a working peer-transfer primitive. The missing ingredient is SenDNN-style planning that makes the transfer part of a better ownership and schedule decision.")]
    story += [PageBreak()]

    # Planning structure.
    story += section_header("02", "Why the relayouts are different", "Torch repairs a mismatch after planning. SenDNN plans ownership, routing, lifetime, and the consumer together.")
    story += [planning_diagram(), Spacer(1, 4)]
    comparison = [
        ["Dimension", "Torch-Spyre current path", "SenDNN path"],
        ["Planning scope", "One bundle and one edge", "Complete prefill or decode phase"],
        ["Materialization", "Explicit S1 -> SHUFFLE -> S2", "Integrated STCDPOpLx handoff"],
        ["Work division", "Producer and consumer chosen independently", "Owner and consumer divisions co-optimized"],
        ["Topology", "Fixed all-gather or dense all-to-all classification", "Per-edge owner groups, replication, multicast, or local retention"],
        ["Synchronization", "Standalone edge becomes a full barrier", "Dependency-local synchronization inside one phase program"],
        ["Coverage", "5 prefill and 3 decode relayouts per layer", "About 16.2 prefill and 22.2 decode instances per layer"],
        ["Proof in SMC", "Peer sites appear only inside explicit shuffle SDSCs", "Peer LD/LDU/LDG/LDGU and ST/STU/STG/STGU across the phase"],
    ]
    story += [styled_table(comparison, [1.18 * inch, 2.78 * inch, 3.19 * inch]), Spacer(1, 5)]
    story += [P("SenDNN also uses 25% fewer static SYNC sites in prefill and 41% fewer in decode than the earlier Torch repeated-block program. Static counts are directional evidence, not executed-stall counters.", "caption")]
    story += [PageBreak()]

    # Exact current Torch layouts.
    story += section_header("03", "Exact Torch relayout inventory", "The beneficial edge is a grouped all-gather. The harmful prefill edges are large, dense, mostly remote all-to-alls.")
    story += [topology_diagram(), Spacer(1, 3)]
    story += [image(charts["payloads"], 7.15 * inch, 2.08 * inch), Spacer(1, 3)]
    inventory = [
        ["Phase", "Relayout", "Source -> destination", "Tensor", "Consumer / result"],
        ["Prefill", "All-gather", "32 mb shards -> 4 mb groups replicated 8x", "4 MiB", "Before BMM; saves 10.368 ms full-model"],
        ["Prefill", "Dense A2A", "8 out x 4 mb -> 32 mb", "4 MiB", "Post-BMM pointwise input"],
        ["Prefill", "Dense A2A", "8 out x 4 mb -> 32 mb", "12.5 MiB x2", "SiLU and Mul inputs"],
        ["Prefill", "Dense A2A", "4 out x 8 mb -> 32 mb", "4 MiB", "Post-BMM pointwise input"],
        ["Decode", "All-gather", "32 mb shards -> 4 mb groups replicated 8x", "512 KiB", "Before BMM"],
        ["Decode", "Dense A2A x2", "8 out x 4 mb -> 32 mb", "512 KiB x2", "Attention and MLP post-BMM Mul inputs"],
    ]
    story += [styled_table(inventory, [0.62 * inch, 0.8 * inch, 2.2 * inch, 0.87 * inch, 2.66 * inch])]
    story += [PageBreak()]

    # Prefill explanation.
    story += section_header("04", "Why Torch prefill gets worse", "Core-to-core is only a win when it removes more critical-path HBM, restickify, and synchronization than the peer route introduces.")
    story += [image(charts["delta"], 7.15 * inch, 2.1 * inch), Spacer(1, 3)]
    left = [
        P("Why the all-gather wins", "h2"),
        P("- It feeds the BMM layout directly.<br/>- It uses grouped replication rather than dense permutation.<br/>- Its avoided BMM ingress materialization exceeds its ring and synchronization cost.<br/>- Full-model result: 10.368 ms saved.", "body"),
        P("Why four dense all-to-alls lose", "h2"),
        P("- They move 33 MiB of source tensors per layer before protocol overhead.<br/>- Nearly every destination byte changes core.<br/>- They allocate separate S1 and S2 LX regions.<br/>- They serialize a full edge rather than overlapping with PT or HBM service.<br/>- Incremental result: 35.528 ms lost.", "body"),
    ]
    right = [image(charts["coverage_pt"], 3.43 * inch, 2.05 * inch), P("The SenDNN prefill PT ideal remains 115.587 ms with LXOpt on and off. The proxy rises from 32.05% to 60.71%, showing that the gain is realization of the same arithmetic, not arithmetic elimination.", "caption")]
    story += [two_col(left, right), Spacer(1, 4)]
    story += [callout("The correct cost question is not 'Can this tensor use core-to-core?' It is 'Does this route reduce the final phase critical path after accounting for placement, HBM, LX pressure, synchronization, and overlap?'", RL_PURPLE, colors.HexColor("#F0EDFA"))]
    story += [PageBreak()]

    # SenDNN coverage and decode.
    story += section_header("05", "What SenDNN relayouts cover", "SenDNN spends peer traffic on a much broader ownership chain, especially BMM ingress and decode attention restickify.")
    story += [image(charts["demand"], 7.15 * inch, 2.18 * inch), Spacer(1, 3)]
    coverage = [
        ["Consumer family", "Prefill folded / expanded", "Decode folded / expanded", "Why it matters"],
        ["BatchMatMulV2", "22 / 244", "23 / 245", "94.15% of prefill and 72.77% of decode remote demand"],
        ["Mul", "12 / 160", "15 / 200", "Maintains compatible pointwise ownership"],
        ["LayerNormNorm", "12 / 160", "14 / 162", "Carries normalization output into projections"],
        ["Exx2", "4 / 41", "7 / 81", "Keeps reduction state on chip"],
        ["Restickify", "3 / 40", "3 / 40", "24.25% of decode remote demand; key attention edge"],
        ["Softmax Max/Sub/Sum", "0 / 0", "9 / 120", "Avoids HBM materialization between reductions"],
        ["KV Scatter", "0 / 0", "3 / 40", "Routes the updated cache to its owner"],
        ["Total", "55 / 647", "75 / 889", "Every retained record is STCDPOpLx with remote delivery"],
    ]
    story += [styled_table(coverage, [1.2 * inch, 1.32 * inch, 1.32 * inch, 3.31 * inch]), Spacer(1, 5)]
    story += [callout("Decode implication: Torch's current three relayouts save only 2.435 ms because they do not cover the dominant attention restickify, softmax, and KV ownership chain. SenDNN's decode advantage is broader than its LXOpt toggle alone.", RL_ORANGE, colors.HexColor("#FFF6E8"))]
    story += [PageBreak()]

    # Causal interpretation.
    story += section_header("06", "What the SenDNN counterfactual proves", "The full LXOpt plan is causally large in prefill and material but not dominant in decode.")
    story += [image(charts["accounting"], 7.15 * inch, 1.9 * inch), Spacer(1, 4)]
    a_b = [
        ["Quantity", "Prefill", "Decode", "Interpretation"],
        ["SenDNN LXOpt off", "360.694 ms", "129.879 ms", "No retained LxRelayout records in final execution order"],
        ["SenDNN LXOpt on", "190.406 ms", "123.961 ms", "55 / 75 folded relayout records plus LX placement"],
        ["Plan gain", "170.288 ms; 1.894x", "5.919 ms; 1.048x", "Exact measured value of the full LXOpt plan"],
        ["Share of current Torch-off gap", "90.5%", "16.3%", "Cross-stack accounting only; not a causal decomposition"],
    ]
    story += [styled_table(a_b, [1.45 * inch, 1.35 * inch, 1.35 * inch, 3.0 * inch]), Spacer(1, 6)]
    story += [callout("Critical caveat: DT_OPT=autopilot=1,lxopt=0 disables more than STCDPOpLx insertion. It also changes LX retention and disables LX-dependent overlapped input fetch. The 170.288 / 5.919 ms values are an upper bound on pure relayout-only gain and the exact gain of the full SenDNN LXOpt plan.", RL_RED, colors.HexColor("#FCEDEC"))]
    story += [Spacer(1, 6), P("The output text remained identical across the five-run counterfactual. The prefill compiler ideal remained 127,146,112 cycles, confirming that the latency reduction is not fewer PT arithmetic cycles.", "body")]
    story += [PageBreak()]

    # Path to parity.
    story += section_header("07", "Concrete path to Torch-Spyre parity", "Use the current size gate as the safe interim policy, then replace post-hoc repair with a joint phase planner.")
    story += [decision_diagram(), Spacer(1, 4)]
    actions = [
        ["Priority", "Implementation", "Immediate rule", "Acceptance evidence"],
        ["P0", "Volume-aware selector", "Keep prefill all-gather; reject A2A >1 MiB; retain three decode routes", "Full-model correctness; prefill <=368.216 ms; decode <=157.925 ms or better"],
        ["P1", "Joint producer-consumer work division", "Rank compatible divisions before inserting a shuffle", "Fewer S1/S2 copies; same or lower ideal PT work; positive device delta"],
        ["P2", "Decode attention ownership chain", "Relayout restickify, softmax state, and KV scatter before more MLP work", "Remove ReStickifyOpHBM; attention 0.685-0.744 ms/layer"],
        ["P3", "Grouped multicast routes", "Choose owner groups and replication factor per edge", "Emitted LDG/LDGU and STG/STGU; lower unicast sites and ring demand"],
        ["P4", "Phase DAG and dependency-local sync", "Overlap peer transfer, HBM, and PT under exact LX lifetimes", "Lower executed stalls and phase latency; no capacity-contract change"],
        ["P5", "Parity closure", "Expand the proven policy to prefill BMMs, normalization, and true-BMM SwiGLU", "Prefill <=192.310 ms; decode <=125.200 ms; identical tokens"],
    ]
    story += [styled_table(actions, [0.42 * inch, 1.38 * inch, 2.43 * inch, 2.92 * inch]), Spacer(1, 5)]
    story += [callout("Do not generalize from 'all-gather is good' to 'all peer traffic is good.' The lever is coordinated ownership that removes a larger exposed cost than it introduces.")]
    story += [PageBreak()]

    # Evidence and measurement boundary.
    story += section_header("08", "Evidence, reproducibility, and measurement boundary", "All performance numbers below are device-program durations; modeled transport quantities are labeled separately.")
    evidence = [
        ["Artifact", "Location / identity"],
        ["Torch checkout", "9bea573e6f304fba5357656ce9122f6e4b587700"],
        ["DeepTools checkout", "406142afb9f080b9271e7c565a757ab8d8b5ed8f; all-to-all handoff included"],
        ["SenDNN normal result", "results/2026-07-25/full_model_comparison/metrics.json; 5 generations"],
        ["SenDNN LXOpt-off run", "/home/adnan/codex-isolated/sendnn_granite_antoni_20260725/runs/full_40_layer_b1_s512_5x4_lxopt0_relayout_study_a"],
        ["Torch relayout runs", "/home/adnan/codex-isolated/device_parity_pr2939_20260725/runs/full40_*"],
        ["SenDNN SDSC attribution", "55 prefill and 75 decode folded final LxRelayout records; tools/analyze_sendnn_sdsc_lx.py"],
        ["Analysis sources", "outputs/sendnn_vs_torch_spyre_gap_analysis.md and outputs/sendnn_vs_torch_spyre_smc_study.md"],
        ["Fixed allocator contract", "DXP_LX_FRAC_AVAIL=0.2"],
    ]
    story += [styled_table(evidence, [1.52 * inch, 5.63 * inch]), Spacer(1, 8)]
    boundaries = [
        P("Measured", "h2"),
        P("- Full-model kernel durations from Kineto device events.<br/>- Correct generated response across the five-run SenDNN counterfactual.<br/>- Final SDSC relayout counts and consumer families.<br/>- Static generated-SMC opcode families.", "body"),
    ]
    proxies = [
        P("Not physical counters", "h2"),
        P("- Logical remote destination bytes from source/destination piece placement.<br/>- Compiler ideal PT cycles divided by measured latency.<br/>- Static LD/ST/SYNC opcode-site counts.<br/>- Any inferred ring service or utilization without executed-unit counters.", "body"),
    ]
    story += [two_col(boundaries, proxies), Spacer(1, 6)]
    story += [callout("Final conclusion: SenDNN is faster because its relayouts are one component of a coordinated phase ownership plan. Torch should copy the decision process and coverage, not simply increase the number of SHUFFLE operations.", RL_TEAL, colors.HexColor("#E7F7F5"))]

    return story


def main() -> None:
    charts = make_charts()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(letter),
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.48 * inch,
        bottomMargin=0.46 * inch,
        title="Granite Relayouts: Torch-Spyre vs SenDNN",
        author="Codex performance analysis",
        subject="Full-model Granite relayout and LXOpt comparison",
    )
    doc.build(build_story(charts), onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT)


if __name__ == "__main__":
    main()
