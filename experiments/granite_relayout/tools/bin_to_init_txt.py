#!/usr/bin/env python3
"""Convert a DeepTools binary init image to DIP's textual 128-byte flit form."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = args.input.read_bytes()
    if len(payload) % 128:
        raise SystemExit(f"input size {len(payload)} is not a multiple of 128")

    lines: list[str] = []
    for offset in range(0, len(payload), 128):
        words = struct.unpack("<32I", payload[offset : offset + 128])
        lines.append("".join(f"{word:08x}" for word in reversed(words)))
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
