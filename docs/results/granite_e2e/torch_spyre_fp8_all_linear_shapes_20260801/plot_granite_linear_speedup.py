#!/usr/bin/env python3

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1440, 900
PLOT = (150, 210, 1360, 755)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def main() -> None:
    with (HERE / "granite_linear_shape_sweep.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (HERE / "granite_linear_weighted_sum.csv").open(newline="") as handle:
        weighted = list(csv.DictReader(handle))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["projection_family"]].append(row)

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = PLOT
    x_values = (512, 1024, 2048)
    x_by_m = {m: left + i * (right - left) / 2 for i, m in enumerate(x_values)}
    y_min, y_max = 0.5, 2.0

    def point(m: int, speedup: float) -> tuple[float, float]:
        x = x_by_m[m]
        y = bottom - ((speedup - y_min) / (y_max - y_min)) * (bottom - top)
        return x, y

    draw.text(
        (left, 38),
        "Granite Linear-Layer FP8 Speedup",
        fill="#202124",
        font=font(38, True),
    )
    draw.text(
        (left, 88),
        "Complete dynamic scaled matmul; static weight packing excluded",
        fill="#5f6368",
        font=font(22),
    )

    for tick in (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0):
        _, y = point(512, tick)
        color = "#9aa0a6" if tick == 1.0 else "#e5e7eb"
        width = 4 if tick == 1.0 else 2
        draw.line((left, y, right, y), fill=color, width=width)
        label = f"{tick:.2f}x" if tick % 1 else f"{tick:.0f}x"
        box = draw.textbbox((0, 0), label, font=font(18))
        draw.text(
            (left - 18 - (box[2] - box[0]), y - 10),
            label,
            fill="#5f6368",
            font=font(18),
        )

    for m in x_values:
        x, _ = point(m, y_min)
        draw.line((x, top, x, bottom), fill="#f0f1f2", width=2)
        label = str(m)
        box = draw.textbbox((0, 0), label, font=font(20))
        draw.text(
            (x - (box[2] - box[0]) / 2, bottom + 18),
            label,
            fill="#5f6368",
            font=font(20),
        )

    draw.line((left, top, left, bottom), fill="#80868b", width=3)
    draw.line((left, bottom, right, bottom), fill="#80868b", width=3)

    colors = {
        "K/V": "#4472C4",
        "Q/O": "#2E8B57",
        "gate/up": "#ED7D31",
        "down": "#A64AC9",
        "Weighted linear sum": "#202124",
    }
    series: list[tuple[str, list[tuple[int, float]]]] = []
    for family in ("K/V", "Q/O", "gate/up", "down"):
        values = sorted(
            (
                int(row["m"]),
                float(row["optimized_fp8_over_fp16"]),
            )
            for row in grouped[family]
        )
        series.append((family, values))
    series.append(
        (
            "Weighted linear sum",
            [
                (int(row["m"]), float(row["optimized_fp8_over_fp16"]))
                for row in weighted
            ],
        )
    )

    legend_x = 480
    for index, (label, values) in enumerate(series):
        color = colors[label]
        width = 7 if label == "Weighted linear sum" else 5
        points = [point(m, speedup) for m, speedup in values]
        draw.line(points, fill=color, width=width, joint="curve")
        for x, y in points:
            radius = 8 if label == "Weighted linear sum" else 6
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill="white",
                outline=color,
                width=4,
            )
        item_x = legend_x + (index % 3) * 285
        item_y = 132 + (index // 3) * 36
        draw.line(
            (item_x, item_y + 8, item_x + 38, item_y + 8),
            fill=color,
            width=width,
        )
        draw.text(
            (item_x + 50, item_y - 5),
            label,
            fill="#303134",
            font=font(18, label == "Weighted linear sum"),
        )

    x_label = "M"
    box = draw.textbbox((0, 0), x_label, font=font(23, True))
    draw.text(
        ((left + right - (box[2] - box[0])) / 2, 825),
        x_label,
        fill="#303134",
        font=font(23, True),
    )

    y_label = "Optimized FP8 / FP16"
    y_image = Image.new("RGBA", (330, 44), (255, 255, 255, 0))
    ImageDraw.Draw(y_image).text((0, 0), y_label, fill="#303134", font=font(23, True))
    y_image = y_image.rotate(90, expand=True)
    image.paste(y_image, (25, int((top + bottom - y_image.height) / 2)), y_image)

    image.save(HERE / "granite_linear_speedup.png")


if __name__ == "__main__":
    main()
