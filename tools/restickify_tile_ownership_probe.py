#!/usr/bin/env python3
"""Diagnose tile ownership for transpose-shaped LX restickify edges.

This tool is intentionally hardware-free.  It models the 64-element stick/tile
grid used by the failing Stage126 fixture and answers one concrete question:
which logical tiles can stay on the same core, and which tiles must be fetched
from another core before the local restickification step can be correct?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TileMovement:
    tile_row: int
    tile_col: int
    source_core: int
    dest_core: int
    hops: int


@dataclass(frozen=True)
class ProbeSummary:
    size: int
    tile_size: int
    tiles_per_dim: int
    num_cores: int
    source_split_dim: str
    dest_split_dim: str
    total_tiles: int
    local_tiles: int
    moving_tiles: int
    total_tile_hops: int
    max_tile_hops: int
    hop_histogram: dict[int, int]
    sample_values: dict[str, dict[str, int]]


def ring_distance(a: int, b: int, n: int) -> int:
    distance = abs(a - b)
    return min(distance, n - distance)


def owner_for_tile(
    tile_row: int,
    tile_col: int,
    *,
    split_dim: str,
    tiles_per_dim: int,
    num_cores: int,
) -> int:
    if split_dim in {"row", "mb", "d0"}:
        return (tile_row * num_cores) // tiles_per_dim
    if split_dim in {"col", "out", "d1"}:
        return (tile_col * num_cores) // tiles_per_dim
    raise ValueError(f"unknown split dim {split_dim!r}; use row/col/mb/out/d0/d1")


def movements(
    *,
    size: int,
    tile_size: int,
    num_cores: int,
    source_split_dim: str,
    dest_split_dim: str,
) -> list[TileMovement]:
    if size % tile_size != 0:
        raise ValueError("size must be divisible by tile size for this diagnostic")
    tiles_per_dim = size // tile_size
    return [
        TileMovement(
            tile_row=tile_row,
            tile_col=tile_col,
            source_core=owner_for_tile(
                tile_row,
                tile_col,
                split_dim=source_split_dim,
                tiles_per_dim=tiles_per_dim,
                num_cores=num_cores,
            ),
            dest_core=owner_for_tile(
                tile_row,
                tile_col,
                split_dim=dest_split_dim,
                tiles_per_dim=tiles_per_dim,
                num_cores=num_cores,
            ),
            hops=ring_distance(
                owner_for_tile(
                    tile_row,
                    tile_col,
                    split_dim=source_split_dim,
                    tiles_per_dim=tiles_per_dim,
                    num_cores=num_cores,
                ),
                owner_for_tile(
                    tile_row,
                    tile_col,
                    split_dim=dest_split_dim,
                    tiles_per_dim=tiles_per_dim,
                    num_cores=num_cores,
                ),
                num_cores,
            ),
        )
        for tile_row in range(tiles_per_dim)
        for tile_col in range(tiles_per_dim)
    ]


def expected_value(row: int, col: int) -> int:
    return col


def no_tile_exchange_but_local_transpose(row: int, col: int, tile_size: int) -> int:
    return (row // tile_size) * tile_size + (col % tile_size)


def tile_exchange_but_no_local_transpose(row: int, col: int, tile_size: int) -> int:
    return (col // tile_size) * tile_size + (row % tile_size)


def tile_exchange_and_local_transpose(row: int, col: int, tile_size: int) -> int:
    return expected_value(row, col)


def sample_value_fingerprints(tile_size: int) -> dict[str, dict[str, int]]:
    samples = [(0, 64), (0, 1024), (64, 0), (127, 0), (128, 0), (128, 64)]
    out: dict[str, dict[str, int]] = {}
    for row, col in samples:
        key = f"{row},{col}"
        out[key] = {
            "expected": expected_value(row, col),
            "no_tile_exchange_local_transpose": no_tile_exchange_but_local_transpose(
                row, col, tile_size
            ),
            "tile_exchange_no_local_transpose": tile_exchange_but_no_local_transpose(
                row, col, tile_size
            ),
            "tile_exchange_local_transpose": tile_exchange_and_local_transpose(
                row, col, tile_size
            ),
        }
    return out


def summarize(
    *,
    size: int,
    tile_size: int,
    num_cores: int,
    source_split_dim: str,
    dest_split_dim: str,
) -> ProbeSummary:
    tile_movements = movements(
        size=size,
        tile_size=tile_size,
        num_cores=num_cores,
        source_split_dim=source_split_dim,
        dest_split_dim=dest_split_dim,
    )
    hop_histogram = Counter(movement.hops for movement in tile_movements)
    return ProbeSummary(
        size=size,
        tile_size=tile_size,
        tiles_per_dim=size // tile_size,
        num_cores=num_cores,
        source_split_dim=source_split_dim,
        dest_split_dim=dest_split_dim,
        total_tiles=len(tile_movements),
        local_tiles=sum(1 for movement in tile_movements if movement.hops == 0),
        moving_tiles=sum(1 for movement in tile_movements if movement.hops != 0),
        total_tile_hops=sum(movement.hops for movement in tile_movements),
        max_tile_hops=max((movement.hops for movement in tile_movements), default=0),
        hop_histogram=dict(sorted(hop_histogram.items())),
        sample_values=sample_value_fingerprints(tile_size),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--num-cores", type=int, default=32)
    parser.add_argument("--source-split-dim", default="row")
    parser.add_argument("--dest-split-dim", default="col")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(
        size=args.size,
        tile_size=args.tile_size,
        num_cores=args.num_cores,
        source_split_dim=args.source_split_dim,
        dest_split_dim=args.dest_split_dim,
    )
    payload = asdict(summary)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            f"{summary.size}x{summary.size}, tile={summary.tile_size}, "
            f"tiles={summary.tiles_per_dim}x{summary.tiles_per_dim}"
        )
        print(
            f"source split={summary.source_split_dim}, "
            f"dest split={summary.dest_split_dim}, cores={summary.num_cores}"
        )
        print(
            f"tiles: total={summary.total_tiles}, local={summary.local_tiles}, "
            f"moving={summary.moving_tiles}"
        )
        print(
            f"tile hops: total={summary.total_tile_hops}, "
            f"max={summary.max_tile_hops}, histogram={summary.hop_histogram}"
        )
        print("sample deterministic fingerprints:")
        for coord, values in summary.sample_values.items():
            print(f"  {coord}: {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
