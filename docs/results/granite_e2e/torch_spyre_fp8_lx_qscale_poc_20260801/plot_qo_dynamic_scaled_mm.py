#!/usr/bin/env python3

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1440, 864
PLOT = (145, 125, 1370, 730)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def main() -> None:
    with (HERE / "qo_dynamic_scaled_mm.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = PLOT
    x_max, y_max = 2048.0, 90.0

    def point(m: float, tflops: float) -> tuple[float, float]:
        x = left + (m / x_max) * (right - left)
        y = bottom - (tflops / y_max) * (bottom - top)
        return x, y

    draw.text(
        (left, 38),
        "Granite Q/O Matmul Performance",
        fill="#202124",
        font=font(38, True),
    )
    draw.text((left, 84), "M x 4096 x 4096", fill="#5f6368", font=font(23))

    for tick in range(0, 91, 10):
        _, y = point(0, tick)
        draw.line((left, y, right, y), fill="#e5e7eb", width=2)
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=font(19))
        draw.text(
            (left - 18 - (box[2] - box[0]), y - 11),
            label,
            fill="#5f6368",
            font=font(19),
        )

    for tick in (0, 512, 1024, 1536, 2048):
        x, _ = point(tick, 0)
        draw.line((x, top, x, bottom), fill="#f0f1f2", width=2)
        label = str(tick)
        box = draw.textbbox((0, 0), label, font=font(19))
        draw.text(
            (x - (box[2] - box[0]) / 2, bottom + 18),
            label,
            fill="#5f6368",
            font=font(19),
        )

    draw.line((left, top, left, bottom), fill="#80868b", width=3)
    draw.line((left, bottom, right, bottom), fill="#80868b", width=3)

    series = (
        ("FP16 matmul", "fp16_tflops", "#4472C4"),
        ("Baseline FP8 matmul", "baseline_fp8_tflops", "#ED7D31"),
        ("Optimized FP8 matmul", "optimized_fp8_tflops", "#2E8B57"),
    )
    legend_x = 710
    for index, (label, field, color) in enumerate(series):
        points = [point(float(row["m"]), float(row[field])) for row in rows]
        draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse(
                (x - 7, y - 7, x + 7, y + 7), fill="white", outline=color, width=4
            )
        item_x = legend_x + index * 225
        draw.line((item_x, 92, item_x + 36, 92), fill=color, width=5)
        draw.text((item_x + 46, 78), label, fill="#303134", font=font(18))

    x_label = "M"
    box = draw.textbbox((0, 0), x_label, font=font(23, True))
    draw.text(
        ((left + right - (box[2] - box[0])) / 2, 790),
        x_label,
        fill="#303134",
        font=font(23, True),
    )

    y_label = "Effective TFLOP/s"
    y_image = Image.new("RGBA", (300, 44), (255, 255, 255, 0))
    ImageDraw.Draw(y_image).text((0, 0), y_label, fill="#303134", font=font(23, True))
    y_image = y_image.rotate(90, expand=True)
    image.paste(y_image, (24, int((top + bottom - y_image.height) / 2)), y_image)

    image.save(HERE / "qo_dynamic_scaled_mm_tflops.png")


if __name__ == "__main__":
    main()
