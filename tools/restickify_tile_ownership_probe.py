#!/usr/bin/env python3
"""Diagnose tile ownership for transpose-shaped LX restickify edges.

This tool is intentionally hardware-free.  It models the 64-element stick/tile
grid used by the failing Stage126 fixture and answers one concrete question:
which logical tiles can stay on the same core, and which tiles must be fetched
from another core before the local restickification step can be correct?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass

_STREAMING_PLANNER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "torch_spyre",
    "_inductor",
    "codegen",
    "restickify_ptlx_streaming.py",
)
_STREAMING_PLANNER_SPEC = importlib.util.spec_from_file_location(
    "_restickify_ptlx_streaming",
    _STREAMING_PLANNER_PATH,
)
if _STREAMING_PLANNER_SPEC is None or _STREAMING_PLANNER_SPEC.loader is None:
    raise ImportError(f"cannot load streaming planner from {_STREAMING_PLANNER_PATH}")
_STREAMING_PLANNER = importlib.util.module_from_spec(_STREAMING_PLANNER_SPEC)
sys.modules[_STREAMING_PLANNER_SPEC.name] = _STREAMING_PLANNER
_STREAMING_PLANNER_SPEC.loader.exec_module(_STREAMING_PLANNER)

default_core_mapping = _STREAMING_PLANNER.default_core_mapping
plan_streaming_ptlx_tiles = _STREAMING_PLANNER.plan_streaming_ptlx_tiles
streaming_ptlx_contract = _STREAMING_PLANNER.streaming_ptlx_contract


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
    parser.add_argument(
        "--streaming-ptlx",
        action="store_true",
        help="Emit the Stage214 streaming PT-LX tile plan summary.",
    )
    parser.add_argument(
        "--source-work-slices",
        default="mb:32,out:1",
        help="Comma separated source split map, for example mb:32,out:1.",
    )
    parser.add_argument(
        "--dest-work-slices",
        default="mb:1,out:32",
        help="Comma separated destination split map, for example mb:1,out:32.",
    )
    parser.add_argument("--row-dim", default="mb")
    parser.add_argument("--col-dim", default="out")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.streaming_ptlx:
        source_work_slices = _parse_work_slices(args.source_work_slices)
        dest_work_slices = _parse_work_slices(args.dest_work_slices)
        streaming_summary = plan_streaming_ptlx_tiles(
            size=args.size,
            tile_size=args.tile_size,
            ring_size=args.num_cores,
            source_work_slices=source_work_slices,
            source_core_mapping=default_core_mapping(
                source_work_slices,
                row_dim=args.row_dim,
                col_dim=args.col_dim,
            ),
            dest_work_slices=dest_work_slices,
            dest_core_mapping=default_core_mapping(
                dest_work_slices,
                row_dim=args.row_dim,
                col_dim=args.col_dim,
            ),
            row_dim=args.row_dim,
            col_dim=args.col_dim,
        )
        payload = asdict(streaming_summary)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"{args.size}x{args.size}, tile={args.tile_size}, "
                f"tiles={streaming_summary.tiles_per_row}x"
                f"{streaming_summary.tiles_per_col}"
            )
            print(
                f"source={streaming_summary.source_work_slices}, "
                f"dest={streaming_summary.dest_work_slices}"
            )
            print(
                f"tiles: total={streaming_summary.total_tiles}, "
                f"local={streaming_summary.local_tiles}, "
                f"moving={streaming_summary.moving_tiles}, "
                f"gather={streaming_summary.gather_tiles}, "
                f"scatter={streaming_summary.scatter_tiles}"
            )
            print(
                f"fan-in max={streaming_summary.max_fan_in}, "
                f"fan-out max={streaming_summary.max_fan_out}, "
                f"byte-hops={streaming_summary.total_byte_hops}, "
                f"max hops={streaming_summary.max_tile_hops}"
            )
            print(
                f"workspace: tile buffer={streaming_summary.tile_buffer_bytes} B, "
                f"full tensor/core src="
                f"{streaming_summary.full_tensor_bytes_per_source_core} B, "
                f"dst={streaming_summary.full_tensor_bytes_per_dest_core} B"
            )
            contract = streaming_ptlx_contract(streaming_summary)
            print(
                f"contract: phases={contract['phases']}, "
                f"bounded_workspace={contract['bounded_workspace_bytes']} B, "
                f"fits_lx={contract['fits_lx_workspace']}"
            )
            print(f"notes: {streaming_summary.notes}")
        return 0

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


def _parse_work_slices(value: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid work-slice item {item!r}; expected dim:split")
        dim, split = item.split(":", 1)
        out[dim.strip()] = int(split)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
