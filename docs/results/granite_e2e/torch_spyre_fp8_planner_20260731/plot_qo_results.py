#!/usr/bin/env python3
"""Render the compact Q/O planner and path-comparison charts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def read_rows() -> list[dict[str, float]]:
    with (ROOT / "qo_automatic_sweep.csv").open(newline="") as handle:
        return [
            {key: float(value) for key, value in row.items() if key != "passed"}
            for row in csv.DictReader(handle)
        ]


def render(
    rows: list[dict[str, float]],
    series: list[tuple[str, str, str]],
    output: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    m_values = [row["M"] for row in rows]
    for label, field, color in series:
        axis.plot(
            m_values,
            [row[field] for row in rows],
            marker="o",
            linewidth=2,
            markersize=4,
            label=label,
            color=color,
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(m_values)
    axis.set_xticklabels([str(int(value)) for value in m_values])
    axis.set_xlabel("M")
    axis.set_ylabel("Effective TFLOP/s")
    axis.set_title("Granite Q/O matmul (K=N=4096)")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncols=2)
    figure.tight_layout()
    figure.savefig(ROOT / output, dpi=180)
    plt.close(figure)


def main() -> None:
    rows = read_rows()
    render(
        rows,
        [
            ("FP16", "fp16_tflops", "#4472C4"),
            ("Old FP8 planner", "old_fp8_tflops", "#C55A11"),
            ("Automatic FP8", "automatic_fp8_tflops", "#2E8B57"),
        ],
        "qo_planner_tflops.png",
    )
    render(
        rows,
        [
            ("FP16", "fp16_tflops", "#4472C4"),
            ("Automatic FP8", "automatic_fp8_tflops", "#2E8B57"),
            ("Raw FP8", "raw_fp8_tflops", "#7030A0"),
        ],
        "qo_path_tflops.png",
    )


if __name__ == "__main__":
    main()
