# Copyright 2025 The Torch-Spyre Authors.
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

"""Hardware-free planner for streaming tiled PT-LX restickify bridges.

The current PT-LX prototype emits one full-tensor bridge. That is enough for
the 2048x2048 high-signal case, but it skips shapes where a full bridge would
need sub-stick pieces or too much LX workspace. This module models the next
contract: process one logical 64x64 tile at a time, gather any producer-owned
fragments to a bridge core, apply the local restickify, then scatter fragments
to consumer/restickify ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TileRect:
    core: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int


@dataclass(frozen=True)
class TileFragment:
    core: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    bytes: int
    hops: int


@dataclass(frozen=True)
class StreamingTileSample:
    tile_row: int
    tile_col: int
    bridge_core: int
    fan_in: int
    fan_out: int
    bytes_gathered: int
    bytes_scattered: int
    byte_hops: int
    source_cores: list[int]
    dest_cores: list[int]
    source_fragments: list[TileFragment]
    dest_fragments: list[TileFragment]


@dataclass(frozen=True)
class StreamingPTLXSummary:
    size: int
    tile_size: int
    tiles_per_row: int
    tiles_per_col: int
    total_tiles: int
    ring_size: int
    bytes_per_element: int
    source_work_slices: dict[str, int]
    dest_work_slices: dict[str, int]
    source_core_count: int
    dest_core_count: int
    local_tiles: int
    moving_tiles: int
    gather_tiles: int
    scatter_tiles: int
    max_fan_in: int
    max_fan_out: int
    total_transfer_bytes: int
    total_byte_hops: int
    max_tile_hops: int
    tile_buffer_bytes: int
    full_tensor_bytes: int
    full_tensor_bytes_per_source_core: int
    full_tensor_bytes_per_dest_core: int
    notes: list[str]
    sample_tiles: list[StreamingTileSample]


def streaming_ptlx_contract(
    summary: StreamingPTLXSummary,
    *,
    lx_limit_bytes: int = 2 * 1024 * 1024,
    tile_buffers: int = 3,
) -> dict[str, Any]:
    """Return the production-shaped lowering contract for a streaming plan.

    This is deliberately a contract, not a guarantee that lowering exists yet.
    It states which movement phases are required and whether the bounded tile
    workspace fits inside a single core's LX budget. Lowering may consume this
    contract to decide whether to emit a streaming PT-LX bridge or fall back to
    ``ReStickifyOpHBM``.
    """

    workspace_bytes = int(summary.tile_buffer_bytes) * int(tile_buffers)
    phases = []
    if summary.max_fan_in > 1:
        phases.append("gather-source-fragments")
    else:
        phases.append("read-source-tile")
    phases.append("local-ptlx-restickify")
    if summary.max_fan_out > 1:
        phases.append("scatter-dest-fragments")
    else:
        phases.append("write-dest-tile")

    fits_workspace = workspace_bytes <= int(lx_limit_bytes)
    return {
        "kind": "streaming_ptlx_contract",
        "tile_size": summary.tile_size,
        "tile_buffer_bytes": summary.tile_buffer_bytes,
        "tile_buffers": int(tile_buffers),
        "bounded_workspace_bytes": workspace_bytes,
        "lx_limit_bytes": int(lx_limit_bytes),
        "fits_lx_workspace": fits_workspace,
        "requires_gather": summary.max_fan_in > 1,
        "requires_scatter": summary.max_fan_out > 1,
        "requires_core_count_adapter": summary.source_core_count
        != summary.dest_core_count,
        "max_fan_in": summary.max_fan_in,
        "max_fan_out": summary.max_fan_out,
        "phases": phases,
        "fallback_required": not fits_workspace,
        "fallback_reason": None if fits_workspace else "tile-workspace-exceeds-lx",
    }


def generate_streaming_ptlx_artifact(
    name: str,
    summary: StreamingPTLXSummary,
    *,
    producer_base: int = 0,
    consumer_base: int = 256 * 1024,
    tile_workspace_base: int = 512 * 1024,
    tile_buffers: int = 3,
    max_tiles: int | None = None,
) -> dict[str, Any]:
    """Generate a codegen-only streaming PT-LX descriptor.

    The descriptor is intentionally not a Deeptools SuperDsc yet. It is the
    production-shaped contract lowering should consume next: a sequence of
    per-tile data-op stages with explicit source fragments, destination
    fragments, and bounded LX tile buffers.
    """

    contract = streaming_ptlx_contract(summary, tile_buffers=tile_buffers)
    tiles = []
    for sample in summary.sample_tiles:
        tiles.append(
            _tile_artifact(
                sample,
                producer_base=producer_base,
                consumer_base=consumer_base,
                tile_workspace_base=tile_workspace_base,
                tile_buffer_bytes=summary.tile_buffer_bytes,
            )
        )
    if max_tiles is not None:
        tiles = tiles[: int(max_tiles)]
    return {
        name: {
            "kind": "streaming_ptlx_restickify_descriptor",
            "version": 1,
            "status": "codegen-only",
            "size": summary.size,
            "tile_size": summary.tile_size,
            "tiles_per_row": summary.tiles_per_row,
            "tiles_per_col": summary.tiles_per_col,
            "total_tiles": summary.total_tiles,
            "source_core_count": summary.source_core_count,
            "dest_core_count": summary.dest_core_count,
            "contract": contract,
            "lx_buffers": {
                "producer_base": int(producer_base),
                "consumer_base": int(consumer_base),
                "tile_workspace_base": int(tile_workspace_base),
                "tile_buffer_bytes": summary.tile_buffer_bytes,
                "tile_buffers": int(tile_buffers),
            },
            "tile_records_materialized": len(tiles),
            "tiles": tiles,
        }
    }


def _tile_artifact(
    tile: StreamingTileSample,
    *,
    producer_base: int,
    consumer_base: int,
    tile_workspace_base: int,
    tile_buffer_bytes: int,
) -> dict[str, Any]:
    gather_buffer = tile_workspace_base
    restickified_buffer = tile_workspace_base + tile_buffer_bytes
    output_buffer = tile_workspace_base + 2 * tile_buffer_bytes
    return {
        "tile_row": tile.tile_row,
        "tile_col": tile.tile_col,
        "bridge_core": tile.bridge_core,
        "source_cores": tile.source_cores,
        "dest_cores": tile.dest_cores,
        "fan_in": tile.fan_in,
        "fan_out": tile.fan_out,
        "byte_hops": tile.byte_hops,
        "stages": [
            {
                "op": "STCDPOpLx",
                "role": "gather-source-fragments",
                "input_base": int(producer_base),
                "output_base": int(gather_buffer),
                "fragments": [_fragment_payload(fragment) for fragment in tile.source_fragments],
            },
            {
                "op": "ReStickifyOpWithPTLx",
                "role": "local-ptlx-restickify",
                "input_base": int(gather_buffer),
                "output_base": int(restickified_buffer),
                "core": tile.bridge_core,
            },
            {
                "op": "STCDPOpLx",
                "role": "write-dest-tile",
                "input_base": int(restickified_buffer),
                "output_base": int(consumer_base),
                "tile_output_base": int(output_buffer),
                "fragments": [_fragment_payload(fragment) for fragment in tile.dest_fragments],
            },
        ],
    }


def _fragment_payload(fragment: TileFragment) -> dict[str, int]:
    return {
        "core": fragment.core,
        "row_start": fragment.row_start,
        "row_end": fragment.row_end,
        "col_start": fragment.col_start,
        "col_end": fragment.col_end,
        "bytes": fragment.bytes,
        "hops": fragment.hops,
    }


def ring_distance(a: int, b: int, n: int) -> int:
    distance = abs(int(a) - int(b))
    return min(distance, int(n) - distance)


def default_core_mapping(
    work_slices: Mapping[str, int],
    *,
    row_dim: str = "mb",
    col_dim: str = "out",
) -> dict[str, dict[str, int]]:
    """Materialize the simple row-fastest mapping used by SuperDsc emission."""

    normalized = _normalize_work_slices(work_slices)
    row_splits = int(normalized.get(row_dim, 1))
    col_splits = int(normalized.get(col_dim, 1))
    mapping: dict[str, dict[str, int]] = {}
    for core in range(max(1, row_splits * col_splits)):
        mapping[str(core)] = {
            row_dim: core % row_splits if row_splits > 1 else 0,
            col_dim: (core // row_splits) % col_splits if col_splits > 1 else 0,
        }
    return mapping


def ownership_rectangles(
    *,
    size: int,
    work_slices: Mapping[Any, Any],
    core_mapping: Mapping[Any, Mapping[Any, Any]] | None,
    row_dim: str = "mb",
    col_dim: str = "out",
) -> list[TileRect]:
    normalized_slices = _normalize_work_slices(work_slices)
    mapping = _normalize_core_mapping(
        core_mapping
        or default_core_mapping(normalized_slices, row_dim=row_dim, col_dim=col_dim)
    )
    row_splits = max(1, int(normalized_slices.get(row_dim, 1)))
    col_splits = max(1, int(normalized_slices.get(col_dim, 1)))
    rectangles: list[TileRect] = []
    for core, per_dim in sorted(mapping.items()):
        row_idx = int(per_dim.get(row_dim, 0))
        col_idx = int(per_dim.get(col_dim, 0))
        row_start, row_end = _axis_interval(size, row_idx, row_splits)
        col_start, col_end = _axis_interval(size, col_idx, col_splits)
        rectangles.append(
            TileRect(
                core=int(core),
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
            )
        )
    return rectangles


def plan_streaming_ptlx_tiles(
    *,
    size: int,
    source_work_slices: Mapping[Any, Any],
    dest_work_slices: Mapping[Any, Any],
    source_core_mapping: Mapping[Any, Mapping[Any, Any]] | None = None,
    dest_core_mapping: Mapping[Any, Mapping[Any, Any]] | None = None,
    tile_size: int = 64,
    ring_size: int = 32,
    bytes_per_element: int = 2,
    row_dim: str = "mb",
    col_dim: str = "out",
    sample_limit: int = 8,
    sample_all_tiles: bool = False,
) -> StreamingPTLXSummary:
    """Return a tiled gather/restickify/scatter plan summary.

    ``source_*`` describes the producer-owned LX value. ``dest_*`` describes
    the ownership expected by the local restickify output or consumer input.
    The planner does not emit SDSC JSON yet; it makes the tile movement
    contract explicit so lowering can decide whether a streaming bridge is
    viable.
    """

    if size <= 0:
        raise ValueError("size must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if ring_size <= 0:
        raise ValueError("ring_size must be positive")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be positive")

    source_slices = _normalize_work_slices(source_work_slices)
    dest_slices = _normalize_work_slices(dest_work_slices)
    source_rects = ownership_rectangles(
        size=size,
        work_slices=source_slices,
        core_mapping=source_core_mapping,
        row_dim=row_dim,
        col_dim=col_dim,
    )
    dest_rects = ownership_rectangles(
        size=size,
        work_slices=dest_slices,
        core_mapping=dest_core_mapping,
        row_dim=row_dim,
        col_dim=col_dim,
    )
    tiles_per_row = _ceil_div(size, tile_size)
    tiles_per_col = _ceil_div(size, tile_size)

    local_tiles = 0
    moving_tiles = 0
    gather_tiles = 0
    scatter_tiles = 0
    max_fan_in = 0
    max_fan_out = 0
    total_transfer_bytes = 0
    total_byte_hops = 0
    max_tile_hops = 0
    sample_tiles: list[StreamingTileSample] = []

    for tile_row in range(tiles_per_row):
        for tile_col in range(tiles_per_col):
            tile = TileRect(
                core=-1,
                row_start=tile_row * tile_size,
                row_end=min(size, (tile_row + 1) * tile_size),
                col_start=tile_col * tile_size,
                col_end=min(size, (tile_col + 1) * tile_size),
            )
            source_intersections = _intersections(
                tile,
                source_rects,
                bytes_per_element=bytes_per_element,
                bridge_core=None,
                ring_size=ring_size,
            )
            dest_intersections = _intersections(
                tile,
                dest_rects,
                bytes_per_element=bytes_per_element,
                bridge_core=None,
                ring_size=ring_size,
            )
            if not source_intersections:
                raise ValueError(f"tile {tile_row},{tile_col} has no source owner")
            if not dest_intersections:
                raise ValueError(f"tile {tile_row},{tile_col} has no destination owner")

            bridge_core = _primary_core(dest_intersections)
            source_fragments = _intersections(
                tile,
                source_rects,
                bytes_per_element=bytes_per_element,
                bridge_core=bridge_core,
                ring_size=ring_size,
            )
            dest_fragments = _intersections(
                tile,
                dest_rects,
                bytes_per_element=bytes_per_element,
                bridge_core=bridge_core,
                ring_size=ring_size,
            )
            fan_in = len({fragment.core for fragment in source_fragments})
            fan_out = len({fragment.core for fragment in dest_fragments})
            bytes_gathered = sum(fragment.bytes for fragment in source_fragments)
            bytes_scattered = sum(fragment.bytes for fragment in dest_fragments)
            byte_hops = sum(
                fragment.bytes * fragment.hops for fragment in source_fragments
            ) + sum(fragment.bytes * fragment.hops for fragment in dest_fragments)
            tile_hops = max(
                [fragment.hops for fragment in source_fragments + dest_fragments]
                or [0]
            )

            max_fan_in = max(max_fan_in, fan_in)
            max_fan_out = max(max_fan_out, fan_out)
            max_tile_hops = max(max_tile_hops, tile_hops)
            total_transfer_bytes += bytes_gathered + bytes_scattered
            total_byte_hops += byte_hops
            if byte_hops == 0:
                local_tiles += 1
            else:
                moving_tiles += 1
            if fan_in > 1:
                gather_tiles += 1
            if fan_out > 1:
                scatter_tiles += 1

            if _should_sample_tile(
                byte_hops,
                fan_in,
                fan_out,
                sample_tiles,
                sample_limit,
                sample_all_tiles=sample_all_tiles,
            ):
                sample_tiles.append(
                    StreamingTileSample(
                        tile_row=tile_row,
                        tile_col=tile_col,
                        bridge_core=bridge_core,
                        fan_in=fan_in,
                        fan_out=fan_out,
                        bytes_gathered=bytes_gathered,
                        bytes_scattered=bytes_scattered,
                        byte_hops=byte_hops,
                        source_cores=sorted({f.core for f in source_fragments}),
                        dest_cores=sorted({f.core for f in dest_fragments}),
                        source_fragments=source_fragments,
                        dest_fragments=dest_fragments,
                    )
                )

    full_tensor_bytes = int(size) * int(size) * int(bytes_per_element)
    source_core_count = max(1, len(source_rects))
    dest_core_count = max(1, len(dest_rects))
    notes = _plan_notes(
        source_rects=source_rects,
        dest_rects=dest_rects,
        tile_size=tile_size,
    )
    return StreamingPTLXSummary(
        size=size,
        tile_size=tile_size,
        tiles_per_row=tiles_per_row,
        tiles_per_col=tiles_per_col,
        total_tiles=tiles_per_row * tiles_per_col,
        ring_size=ring_size,
        bytes_per_element=bytes_per_element,
        source_work_slices=source_slices,
        dest_work_slices=dest_slices,
        source_core_count=source_core_count,
        dest_core_count=dest_core_count,
        local_tiles=local_tiles,
        moving_tiles=moving_tiles,
        gather_tiles=gather_tiles,
        scatter_tiles=scatter_tiles,
        max_fan_in=max_fan_in,
        max_fan_out=max_fan_out,
        total_transfer_bytes=total_transfer_bytes,
        total_byte_hops=total_byte_hops,
        max_tile_hops=max_tile_hops,
        tile_buffer_bytes=min(tile_size, size) * min(tile_size, size) * bytes_per_element,
        full_tensor_bytes=full_tensor_bytes,
        full_tensor_bytes_per_source_core=_ceil_div(full_tensor_bytes, source_core_count),
        full_tensor_bytes_per_dest_core=_ceil_div(full_tensor_bytes, dest_core_count),
        notes=notes,
        sample_tiles=sample_tiles[:sample_limit],
    )


def _normalize_work_slices(work_slices: Mapping[Any, Any]) -> dict[str, int]:
    return {str(dim): int(split) for dim, split in (work_slices or {}).items()}


def _normalize_core_mapping(
    core_mapping: Mapping[Any, Mapping[Any, Any]],
) -> dict[int, dict[str, int]]:
    return {
        int(core): {str(dim): int(value) for dim, value in per_dim.items()}
        for core, per_dim in (core_mapping or {}).items()
    }


def _axis_interval(size: int, index: int, split_count: int) -> tuple[int, int]:
    if index < 0 or index >= split_count:
        raise ValueError(
            f"slice index {index} is outside split range 0..{split_count - 1}"
        )
    return (size * index) // split_count, (size * (index + 1)) // split_count


def _intersections(
    tile: TileRect,
    owners: list[TileRect],
    *,
    bytes_per_element: int,
    bridge_core: int | None,
    ring_size: int,
) -> list[TileFragment]:
    fragments: list[TileFragment] = []
    for owner in owners:
        row_start = max(tile.row_start, owner.row_start)
        row_end = min(tile.row_end, owner.row_end)
        col_start = max(tile.col_start, owner.col_start)
        col_end = min(tile.col_end, owner.col_end)
        if row_start >= row_end or col_start >= col_end:
            continue
        hops = 0 if bridge_core is None else ring_distance(owner.core, bridge_core, ring_size)
        fragments.append(
            TileFragment(
                core=owner.core,
                row_start=row_start,
                row_end=row_end,
                col_start=col_start,
                col_end=col_end,
                bytes=(row_end - row_start) * (col_end - col_start) * bytes_per_element,
                hops=hops,
            )
        )
    return fragments


def _primary_core(fragments: list[TileFragment]) -> int:
    by_core: dict[int, int] = {}
    for fragment in fragments:
        by_core[fragment.core] = by_core.get(fragment.core, 0) + fragment.bytes
    return min(
        by_core,
        key=lambda core: (-by_core[core], core),
    )


def _should_sample_tile(
    byte_hops: int,
    fan_in: int,
    fan_out: int,
    samples: list[StreamingTileSample],
    sample_limit: int,
    *,
    sample_all_tiles: bool = False,
) -> bool:
    if len(samples) >= sample_limit:
        return False
    if sample_all_tiles:
        return True
    return byte_hops != 0 or fan_in > 1 or fan_out > 1 or not samples


def _plan_notes(
    *,
    source_rects: list[TileRect],
    dest_rects: list[TileRect],
    tile_size: int,
) -> list[str]:
    notes: list[str] = []
    if len(source_rects) != len(dest_rects):
        notes.append("source-dest-core-count-mismatch")
    if any(_rect_smaller_than_tile(rect, tile_size) for rect in source_rects):
        notes.append("source-piece-smaller-than-tile")
    if any(_rect_smaller_than_tile(rect, tile_size) for rect in dest_rects):
        notes.append("dest-piece-smaller-than-tile")
    if not notes:
        notes.append("single-tile-bridge-contract-compatible")
    return notes


def _rect_smaller_than_tile(rect: TileRect, tile_size: int) -> bool:
    return (rect.row_end - rect.row_start) < tile_size or (
        rect.col_end - rect.col_start
    ) < tile_size


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)
