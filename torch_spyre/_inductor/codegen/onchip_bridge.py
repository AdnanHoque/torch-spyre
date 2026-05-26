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

"""Synthesize the data-op (datadscs_) blocks of a mixed DL+data-op SuperDSC.

A mixed SuperDSC keeps a producer->consumer activation handoff resident in LX:
the consumer DL op (in dscs_) is preceded by data-ops (in datadscs_) that move
the producer's LX-resident output to the consumer's input LX, scheduled by
coreIdToDscSchedule. This module builds those data-op blocks.

Two bridge shapes:
- same-layout (Tier 1): a single STCDPOpLx cross-core move (no stick change);
- layout-changing (Tier 2): ReStickifyOpWithPTLx (local stick transform) then
  STCDPOpLx (place on the consumer-owned core).

The block schema matches deeptools' SuperDsc JSON exactly (verified byte-for-byte
against a known-good reference for the 2048 case).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

DATA_FORMAT = "SEN169_FP16"
WORD_LENGTH = 2

# Per-core LX scratchpad capacity in bytes (2 MB, AIU 1.0). Source of truth:
# torch_spyre/_inductor/scratchpad.py ("scratch pad is 2MB = 2<<20 bytes").
LX_CAPACITY_BYTES = 2 << 20  # 2_097_152
# Stick alignment: a stick is 128 bytes (64 fp16 elements).
STICK_BYTES = 128
# Fixed per-side LX buffer for the streamed bridge. Each tile is <= this; in+out
# = 2 * STREAM_TILE_BYTES, leaving the rest of the 2 MB/core LX for the DL op.
STREAM_TILE_BYTES = 128 << 10  # 131_072
# Default tile footprint for the mixed-SDSC flash-attention pipeline proof.  The
# v0 proof uses the same conservative tile size as streamed same-stick handoffs:
# two K/V double buffers plus a small number of scratch regions still fit in the
# 2 MB/core LX budget.
FLASH_PIPELINE_TILE_BYTES = STREAM_TILE_BYTES
# The same-stick/same-split 512 proof used a 2048x2048 data-op frame even when
# the DL op's logical tensor was 512x512. Smaller frames can compile but hang on
# device, so preserve that proven lower bound for sub-stick same-stick chunks.
MIN_SAME_STICK_FRAME_DIM = 2048


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return ((value + alignment - 1) // alignment) * alignment


def per_core_slice_bytes(
    iter_sizes: Mapping[str, int], split_dim: str, stick_size: int,
    num_cores: int, word_length: int = WORD_LENGTH,
) -> int:
    """Per-core LX bytes for one bridge buffer (rows * stick-padded chunk cols).

    Each core owns rows = product of the non-split dims, cols = the split-dim
    chunk (split_dim / num_cores), padded up to a full stick. Sub-stick chunks
    still occupy a whole stick. Result is rounded up to the 128-byte stick.
    """
    chunk = iter_sizes[split_dim] // num_cores
    cols = max(chunk, stick_size)
    rows = 1
    for d, n in iter_sizes.items():
        if d != split_dim:
            rows *= n
    return _align_up(rows * cols * word_length, STICK_BYTES)


def per_core_same_stick_slice_bytes(
    iter_sizes: Mapping[str, int], split_dim: str, stick_dim: str, stick_size: int,
    num_cores: int, word_length: int = WORD_LENGTH,
) -> int:
    """Per-core bytes for same-stick bridges where split and stick may differ.

    ``per_core_slice_bytes`` is the original same-shard helper and assumes the
    split dimension is also the stick dimension.  Attention score handoffs split
    the query axis across cores while the key axis is the stick, so padding the
    split chunk to a whole stick overstates the LX footprint.  This helper pads
    only the actual stick dimension.
    """
    chunk = iter_sizes[split_dim] // num_cores
    if split_dim == stick_dim:
        chunk = max(chunk, stick_size)
    elems = chunk
    for dim, size in iter_sizes.items():
        if dim == split_dim:
            continue
        if dim == stick_dim:
            size = _align_up(size, stick_size)
        elems *= size
    return _align_up(elems * word_length, STICK_BYTES)


def allocate_lx_bases(
    num_regions: int, slice_bytes: int,
    capacity: int = LX_CAPACITY_BYTES, region0: int = 0,
) -> list[int]:
    """Non-overlapping, stick-aligned LX bases for ``num_regions`` buffers.

    Packs regions back-to-back: base[k] = region0 + k * aligned_slice. Each gets
    ``slice_bytes`` (already a stick multiple). Raises ValueError if the total
    footprint exceeds the per-core LX capacity -- e.g. a 3-region round trip at
    1 MB/slice needs 3 MB > 2 MB and cannot fit. region0 leaves headroom at the
    bottom for the DL op's own LX tensors.
    """
    aligned = _align_up(slice_bytes, STICK_BYTES)
    bases = [region0 + k * aligned for k in range(num_regions)]
    footprint = bases[-1] + aligned if bases else region0
    if footprint > capacity:
        raise ValueError(
            f"{num_regions} regions x {aligned} B + {region0} = {footprint} B "
            f"exceeds per-core LX capacity {capacity} B"
        )
    return bases


def num_stream_tiles(slice_bytes: int, tile_bytes: int = STREAM_TILE_BYTES) -> int:
    """K = ceil(slice / fixed-buffer). 1 MB / 128 KB = 8; 4 MB = 32; >=1."""
    aligned = _align_up(tile_bytes, STICK_BYTES)
    return max(1, -(-slice_bytes // aligned))


def tile_rows(rows: int, num_tiles: int) -> int:
    """Rows per tile along the windowed non-stick dim (split dim untouched)."""
    return -(-rows // num_tiles)


def allocate_stream_bases(
    tile_bytes: int = STREAM_TILE_BYTES,
    capacity: int = LX_CAPACITY_BYTES, region0: int = 0,
) -> list[int]:
    """Two FIXED tile-sized LX bases (in, out) that fit alongside the DL op's LX.

    Streaming pins one in-buffer + one out-buffer of ``tile_bytes`` each; the
    slice flows through them in K tiles, so the LX footprint stays at 2*T (not
    2*slice). Raises ValueError if even the two fixed tiles + region0 overflow.
    """
    return allocate_lx_bases(2, tile_bytes, capacity=capacity, region0=region0)


def allocate_flash_attention_pipeline_bases(
    num_lanes: int = 2,
    tile_bytes: int = FLASH_PIPELINE_TILE_BYTES,
    scratch_regions: int = 2,
    capacity: int = LX_CAPACITY_BYTES,
    region0: int = 0,
    include_source_regions: bool = True,
) -> dict:
    """Allocate LX regions for the flash-attention mixed-SDSC pipeline proof.

    ``num_lanes`` is the number of independent payload streams.  The intended
    attention use is two lanes, K and V, but the helper is generic so the first
    proof can exercise one-lane score staging as well.  Each lane gets two
    destination buffers for ping-pong/double buffering.  Optional source regions
    model LX-resident producer outputs in unit tests; production callers may use
    externally allocated producer bases instead.
    """
    if num_lanes <= 0:
        raise ValueError(f"num_lanes must be positive, got {num_lanes}")
    if tile_bytes <= 0:
        raise ValueError(f"tile_bytes must be positive, got {tile_bytes}")
    if scratch_regions < 0:
        raise ValueError(
            f"scratch_regions must be non-negative, got {scratch_regions}"
        )

    aligned_tile = _align_up(tile_bytes, STICK_BYTES)
    source_count = num_lanes if include_source_regions else 0
    region_count = source_count + num_lanes * 2 + scratch_regions
    bases = allocate_lx_bases(
        region_count,
        aligned_tile,
        capacity=capacity,
        region0=region0,
    )

    idx = 0
    source_bases = []
    if include_source_regions:
        source_bases = bases[:num_lanes]
        idx = num_lanes
    lane_bases = [bases[idx + 2 * lane: idx + 2 * lane + 2]
                  for lane in range(num_lanes)]
    idx += num_lanes * 2
    scratch_bases = bases[idx:]
    footprint = bases[-1] + aligned_tile if bases else region0
    return {
        "source_bases": source_bases,
        "lane_bases": lane_bases,
        "scratch_bases": scratch_bases,
        "tile_bytes": aligned_tile,
        "footprint": footprint,
    }


def _piece_info(
    layout_order: Sequence[str],
    split_dim: str,
    iter_sizes: Mapping[str, int],
    chunk: int,
    base: int,
    num_cores: int,
    reverse: bool = False,
) -> list[dict]:
    """Per-core PieceInfo: split_dim is chunked across cores, others are full.

    Piece i always covers logical slice i (dimToStartCordinate = i*chunk). When
    reverse=True, piece i is *placed* on core (num_cores-1-i) instead of core i,
    so the logical->physical core mapping is mirrored. Matching pieces by logical
    coordinate against a non-reversed endpoint then forces a genuine cross-core
    move (slice i lives on core i one side, core 31-i the other).
    """
    pieces = []
    for i in range(num_cores):
        start = {d: (i * chunk if d == split_dim else 0) for d in layout_order}
        size = {d: (chunk if d == split_dim else iter_sizes[d]) for d in layout_order}
        gap = {d: [[size[d], 0]] for d in layout_order}
        mem = (num_cores - 1 - i) if reverse else i
        pieces.append(
            {
                "key_": f"p{i + 1}",
                "dimToStartCordinate": start,
                "dimToSize_": size,
                "validGap_": gap,
                "PlacementInfo": [{"type": "lx", "memId": [mem], "startAddr": [base]}],
            }
        )
    return pieces


def _labeled_ds(
    pds_name: str,
    layout_order: Sequence[str],
    stick_dim: str,
    split_dim: str,
    iter_sizes: Mapping[str, int],
    stick_size: int,
    base: int,
    num_cores: int,
    lx_size: int,
    reverse: bool = False,
) -> dict:
    """One labeledDs (dataIN_L0 / dataOUT_L0) with its per-core PieceInfo."""
    chunk = iter_sizes[split_dim] // num_cores
    layout_sizes = dict(iter_sizes)
    if split_dim == stick_dim:
        layout_sizes = {
            dim: max(size, MIN_SAME_STICK_FRAME_DIM)
            for dim, size in layout_sizes.items()
        }
        chunk = max(chunk, stick_size)
        layout_sizes[split_dim] = max(layout_sizes[split_dim], chunk * num_cores)
    return {
        "ldsName_": f"{pds_name}_L0",
        "pdsName_": pds_name,
        "wordLength": WORD_LENGTH,
        "dataformat": DATA_FORMAT,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": list(layout_order),
        "stickDimOrder_": [stick_dim],
        "dimToLayoutSize_": {d: layout_sizes[d] for d in layout_order},
        "dimToStickSize_": {stick_dim: stick_size},
        "validGap_": {d: [[layout_sizes[d], 0]] for d in layout_order},
        "totElements": -1,
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": lx_size,
        "lxStartAddress_": {},
        "PieceInfo": _piece_info(
            layout_order, split_dim, layout_sizes, chunk, base, num_cores, reverse
        ),
    }


def _datadsc(name: str, op: dict, dim_pool: Sequence[str], in_ld: dict, out_ld: dict,
             num_cores: int) -> dict:
    return {
        name: {
            "coreIdsUsed_": list(range(num_cores)),
            "dimPool_": list(dim_pool),
            "outDimTodimRelation_": [],
            "primaryDs_": [
                {"name_": "dataIN", "dimNames": list(dim_pool)},
                {"name_": "dataOUT", "dimNames": list(dim_pool)},
            ],
            "labeledDs_": [in_ld, out_ld],
            "op": op,
        }
    }


# --- endpoint descriptor: (layoutDimOrder_, stickDim, splitDim, lxBase) ---
class Endpoint:
    def __init__(self, layout, stick_dim, split_dim, base, reverse=False):
        self.layout = layout
        self.stick_dim = stick_dim
        self.split_dim = split_dim
        self.base = base
        # reverse=True mirrors the logical->physical core mapping (piece i on
        # core num_cores-1-i), forcing genuine cross-core ring traffic.
        self.reverse = reverse


def _stcdp_op() -> dict:
    return {"name": "STCDPOpLx"}


def _restickify_op() -> dict:
    return {
        "name": "ReStickifyOpWithPTLx",
        "numClToUse": 1,
        "defaultClId": 0,
        "workSplitDim": "null_ptr",
        "cl0ToLxOffsetLU": 0,
        "cl0ToLxOffsetSU": 0,
        "useARF": 1,
        "doInPlace": 0,
    }


def make_datadsc(
    name: str, op: dict, dim_pool: Sequence[str],
    src: Endpoint, dst: Endpoint,
    iter_sizes: Mapping[str, int], stick_size: int, num_cores: int, lx_size: int,
) -> dict:
    in_ld = _labeled_ds("dataIN", src.layout, src.stick_dim, src.split_dim,
                        iter_sizes, stick_size, src.base, num_cores, lx_size,
                        src.reverse)
    out_ld = _labeled_ds("dataOUT", dst.layout, dst.stick_dim, dst.split_dim,
                         iter_sizes, stick_size, dst.base, num_cores, lx_size,
                         dst.reverse)
    return _datadsc(name, op, dim_pool, in_ld, out_ld, num_cores)


def mixed_schedule(num_dataops: int, num_cores: int) -> dict:
    """coreIdToDscSchedule rows: each data-op (before-sync), then the DL op."""
    rows = []
    for k in range(num_dataops):
        rows.append([k, -1, 1 if k > 0 else 0, 1])
    rows.append([-1, 0, 1, 0])
    return {str(c): [list(r) for r in rows] for c in range(num_cores)}


def _schedule_rows_for_all_cores(rows: Sequence[Sequence[int]],
                                 num_cores: int) -> dict:
    """Attach local dependency bits and duplicate the schedule to every core."""
    normalized = [
        [
            int(row[0]),
            int(row[1]),
            1 if idx > 0 else 0,
            1 if idx < len(rows) - 1 else 0,
        ]
        for idx, row in enumerate(rows)
    ]
    return {str(c): [list(r) for r in normalized] for c in range(num_cores)}


def flash_pipeline_schedule(
    num_tiles: int,
    num_lanes: int,
    num_cores: int,
    overlap: bool = False,
) -> dict:
    """Schedule rows for a tiled mixed-SDSC flash-attention pipeline proof.

    Serial mode is the conservative Foundation-safe control: prefetch all lanes
    for tile ``t`` and then run compute DSC ``t``.  Overlap mode is the
    warp-specialized candidate: prefetch tile 0 as a prologue, then pair compute
    DSC ``t`` with the first prefetch data-op for tile ``t + 1``.  The remaining
    lanes for the next tile are emitted as data-op-only rows, avoiding duplicate
    compute dispatch.
    """
    if num_tiles <= 0:
        raise ValueError(f"num_tiles must be positive, got {num_tiles}")
    if num_lanes <= 0:
        raise ValueError(f"num_lanes must be positive, got {num_lanes}")

    rows: list[list[int]] = []
    if not overlap:
        for tile in range(num_tiles):
            dataop_base = tile * num_lanes
            for lane in range(num_lanes):
                rows.append([dataop_base + lane, -1])
            rows.append([-1, tile])
        return _schedule_rows_for_all_cores(rows, num_cores)

    # Prologue: tile 0 must be resident before compute can start.
    for lane in range(num_lanes):
        rows.append([lane, -1])
    for tile in range(num_tiles - 1):
        next_dataop_base = (tile + 1) * num_lanes
        rows.append([next_dataop_base, tile])
        for lane in range(1, num_lanes):
            rows.append([next_dataop_base + lane, -1])
    rows.append([-1, num_tiles - 1])
    return _schedule_rows_for_all_cores(rows, num_cores)


def build_transpose_bridge(
    dim_pool: Sequence[str], iter_sizes: Mapping[str, int], stick_size: int,
    num_cores: int, lx_size: int,
    producer_base: int, scratch_base: int, consumer_base: int,
    out_dim: str, mb_dim: str,
) -> tuple[list[dict], list[str], dict]:
    """Tier-2 reference bridge: ReStickifyOpWithPTLx (out-stick->mb-stick) + STCDPOpLx.

    Reproduces the known-good 2048 reference. out_dim is the producer stick dim,
    mb_dim the consumer stick dim.
    """
    rs = make_datadsc(
        "0_ReStickifyOpWithPTLx_dataop", _restickify_op(), dim_pool,
        src=Endpoint([mb_dim, out_dim], out_dim, out_dim, producer_base),
        dst=Endpoint([out_dim, mb_dim], mb_dim, mb_dim, scratch_base),
        iter_sizes=iter_sizes, stick_size=stick_size, num_cores=num_cores,
        lx_size=lx_size,
    )
    stcdp = make_datadsc(
        "1_STCDPOpLx_dataop", _stcdp_op(), dim_pool,
        src=Endpoint([out_dim, mb_dim], mb_dim, mb_dim, scratch_base),
        dst=Endpoint([out_dim, mb_dim], mb_dim, out_dim, consumer_base),
        iter_sizes=iter_sizes, stick_size=stick_size, num_cores=num_cores,
        lx_size=lx_size,
    )
    datadscs = [rs, stcdp]
    return datadscs, ["ReStickifyOpWithPTLx", "STCDPOpLx"], mixed_schedule(2, num_cores)


def build_same_layout_bridge(
    dim_pool, iter_sizes, stick_size, num_cores, lx_size,
    src_base, dst_base, layout, stick_dim, src_split_dim, dst_split_dim,
):
    """Tier-1 bridge: a single STCDPOpLx pure same-stick cross-core move.

    No transpose -- src and dst share layout/stick; only the per-core ownership
    (which dim is split across cores) differs (src_split_dim -> dst_split_dim).
    This is the pure data move (no PT/compute op), the part proven HBM-free on
    the ring. Used to isolate the Compute-CB fault to the transpose.
    """
    stcdp = make_datadsc(
        "0_STCDPOpLx_dataop", _stcdp_op(), dim_pool,
        src=Endpoint(layout, stick_dim, src_split_dim, src_base),
        dst=Endpoint(layout, stick_dim, dst_split_dim, dst_base),
        iter_sizes=iter_sizes, stick_size=stick_size, num_cores=num_cores,
        lx_size=lx_size,
    )
    return [stcdp], ["STCDPOpLx"], mixed_schedule(1, num_cores)


def build_same_stick_bridge(
    dim_pool, iter_sizes, stick_size, num_cores, lx_size,
    src_base, dst_base, src_layout, dst_layout, stick_dim,
    src_split_dim, dst_split_dim,
):
    """Tier-1 bridge: one STCDPOpLx between same-stick endpoints.

    Unlike build_same_layout_bridge, this accepts different source/destination
    layoutDimOrder_ values. This is needed for stock SDPA score handoffs:
    QK^T and softmax both keep the score matrix on the same stick dim, but
    their non-stick dimension order and core split dim differ.
    """
    stcdp = make_datadsc(
        "0_STCDPOpLx_dataop", _stcdp_op(), dim_pool,
        src=Endpoint(src_layout, stick_dim, src_split_dim, src_base),
        dst=Endpoint(dst_layout, stick_dim, dst_split_dim, dst_base),
        iter_sizes=iter_sizes, stick_size=stick_size, num_cores=num_cores,
        lx_size=lx_size,
    )
    return [stcdp], ["STCDPOpLx"], mixed_schedule(1, num_cores)


def build_roundtrip_bridge(
    dim_pool, iter_sizes, stick_size, num_cores, lx_size,
    producer_base, scratch_base, consumer_base, layout, stick_dim, split_dim,
):
    """Genuine cross-core ring proof: two same-stick STCDPOpLx moves that mirror
    then un-mirror the per-core ownership.

    STCDP1: producer (linear @producer_base, piece i on core i)
            -> scratch  (reversed @scratch_base, piece i on core 31-i)
    STCDP2: scratch  (reversed @scratch_base, piece i on core 31-i)
            -> consumer (linear @consumer_base, piece i on core i)

    Each STCDP moves slice i between core i and core 31-i -> real ring traffic.
    The round trip lands data back in the consumer's native (linear) layout, so
    the result is value-correct WITHOUT any consumer-reshard surgery. Pure data
    moves (no PT/compute op), to test the ring path in isolation from the
    Compute-CB-faulting transpose.
    """
    stcdp1 = make_datadsc(
        "0_STCDPOpLx_dataop", _stcdp_op(), dim_pool,
        src=Endpoint(layout, stick_dim, split_dim, producer_base),
        dst=Endpoint(layout, stick_dim, split_dim, scratch_base, reverse=True),
        iter_sizes=iter_sizes, stick_size=stick_size, num_cores=num_cores,
        lx_size=lx_size,
    )
    stcdp2 = make_datadsc(
        "1_STCDPOpLx_dataop", _stcdp_op(), dim_pool,
        src=Endpoint(layout, stick_dim, split_dim, scratch_base, reverse=True),
        dst=Endpoint(layout, stick_dim, split_dim, consumer_base),
        iter_sizes=iter_sizes, stick_size=stick_size, num_cores=num_cores,
        lx_size=lx_size,
    )
    return ([stcdp1, stcdp2], ["STCDPOpLx", "STCDPOpLx"],
            mixed_schedule(2, num_cores))


def _tiled_piece_info(
    layout_order: Sequence[str],
    split_dim: str,
    row_dim: str,
    iter_sizes: Mapping[str, int],
    chunk: int,
    base: int,
    num_cores: int,
    row_start: int,
    n_rows: int,
    reverse: bool = False,
) -> list[dict]:
    """Per-core PieceInfo for one tile: row_dim windowed, split_dim full chunk.

    Same as _piece_info but dimToStartCordinate[row_dim] = row_start and
    dimToSize_[row_dim] = n_rows so each tile covers a horizontal slab of rows.
    The split dim keeps its full per-core chunk (sticks stay whole). The fixed
    base is reused every tile -- the SAME buffer; the dimToSize_ row count is the
    hardware ring loop bound (L3_MVLOOPCNT), so a smaller tile = smaller loop.
    """
    pieces = []
    for i in range(num_cores):
        start = {d: 0 for d in layout_order}
        start[split_dim] = i * chunk
        start[row_dim] = row_start
        size = {d: iter_sizes[d] for d in layout_order}
        size[split_dim] = chunk
        size[row_dim] = n_rows
        gap = {d: [[size[d], 0]] for d in layout_order}
        mem = (num_cores - 1 - i) if reverse else i
        pieces.append(
            {
                "key_": f"p{i + 1}",
                "dimToStartCordinate": start,
                "dimToSize_": size,
                "validGap_": gap,
                "PlacementInfo": [{"type": "lx", "memId": [mem], "startAddr": [base]}],
            }
        )
    return pieces


def _tiled_labeled_ds(
    pds_name: str, layout_order, stick_dim, split_dim, row_dim, iter_sizes,
    stick_size, base, num_cores, lx_size, row_start, n_rows, reverse=False,
) -> dict:
    """labeledDs for a streamed tile: full layout dims, windowed row dimToSize_."""
    chunk = iter_sizes[split_dim] // num_cores
    return {
        "ldsName_": f"{pds_name}_L0",
        "pdsName_": pds_name,
        "wordLength": WORD_LENGTH,
        "dataformat": DATA_FORMAT,
        "isExternal_": 0,
        "segment_": "output",
        "layoutDimOrder_": list(layout_order),
        "stickDimOrder_": [stick_dim],
        "dimToLayoutSize_": {d: iter_sizes[d] for d in layout_order},
        "dimToStickSize_": {stick_dim: stick_size},
        "validGap_": {d: [[iter_sizes[d], 0]] for d in layout_order},
        "totElements": -1,
        "hbmSize_": 0,
        "hbmStartAddress_": 0,
        "lxSize_": lx_size,
        "lxStartAddress_": {},
        "PieceInfo": _tiled_piece_info(
            layout_order, split_dim, row_dim, iter_sizes, chunk, base, num_cores,
            row_start, n_rows, reverse,
        ),
    }


def build_flash_attention_pipeline_bridge(
    dim_pool,
    iter_sizes,
    stick_size,
    num_cores,
    lx_size,
    src_bases,
    dst_lane_bases,
    layout,
    stick_dim,
    split_dim,
    row_dim,
    lane_names=None,
    tile_bytes=FLASH_PIPELINE_TILE_BYTES,
    overlap=False,
):
    """Build the data-op side of a double-buffered flash-attention pipeline.

    This is a descriptor/scheduler proof helper.  It stages one or more
    LX-resident payload lanes through ping-pong buffers using ``STCDPOpLx``.
    For attention, lanes are typically K and V.  The helper intentionally does
    not model HBM->LX loads; it only uses the same certified LX->LX primitive as
    the existing core-to-core handoff work.
    """
    if row_dim == split_dim:
        raise ValueError("row_dim must differ from split_dim for tiled staging")
    if row_dim not in iter_sizes:
        raise ValueError(f"row_dim {row_dim!r} missing from iter_sizes")
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        raise ValueError("split_dim must be present and divisible by num_cores")

    num_lanes = len(src_bases)
    if num_lanes == 0:
        raise ValueError("at least one pipeline lane is required")
    if len(dst_lane_bases) != num_lanes:
        raise ValueError("dst_lane_bases must match src_bases length")
    for bases in dst_lane_bases:
        if len(bases) != 2:
            raise ValueError("each lane needs exactly two destination bases")
    if lane_names is None:
        lane_names = [f"lane{lane}" for lane in range(num_lanes)]
    if len(lane_names) != num_lanes:
        raise ValueError("lane_names must match src_bases length")

    slice_bytes = per_core_same_stick_slice_bytes(
        iter_sizes,
        split_dim,
        stick_dim,
        stick_size,
        num_cores,
    )
    num_tiles = num_stream_tiles(slice_bytes, tile_bytes)
    rows_per_tile = tile_rows(iter_sizes[row_dim], num_tiles)

    datadscs = []
    for tile in range(num_tiles):
        row_start = tile * rows_per_tile
        n_rows = min(rows_per_tile, iter_sizes[row_dim] - row_start)
        for lane, lane_name in enumerate(lane_names):
            dataop_idx = tile * num_lanes + lane
            dst_base = dst_lane_bases[lane][tile % 2]
            in_ld = _tiled_labeled_ds(
                "dataIN",
                layout,
                stick_dim,
                split_dim,
                row_dim,
                iter_sizes,
                stick_size,
                src_bases[lane],
                num_cores,
                lx_size,
                row_start,
                n_rows,
            )
            out_ld = _tiled_labeled_ds(
                "dataOUT",
                layout,
                stick_dim,
                split_dim,
                row_dim,
                iter_sizes,
                stick_size,
                dst_base,
                num_cores,
                lx_size,
                row_start,
                n_rows,
            )
            datadscs.append(
                _datadsc(
                    f"{dataop_idx}_STCDPOpLx_prefetch_{lane_name}_tile{tile}",
                    _stcdp_op(),
                    dim_pool,
                    in_ld,
                    out_ld,
                    num_cores,
                )
            )

    return (
        datadscs,
        ["STCDPOpLx"] * len(datadscs),
        flash_pipeline_schedule(num_tiles, num_lanes, num_cores, overlap=overlap),
    )


def build_flash_attention_pipeline_mixed_sdsc(
    name: str,
    datadscs: Sequence[dict],
    opfuncs: Sequence[str],
    schedule: Mapping[str, Sequence[Sequence[int]]],
    compute_dscs: Sequence[dict],
    num_cores: int,
) -> dict:
    """Wrap tiled compute DSCs with flash pipeline data-ops as one SuperDSC.

    The bridge builder emits the data movement side; this helper gives the next
    compiler step a production-shaped artifact with ``datadscs_``, ``dscs_`` and
    ``coreIdToDscSchedule`` in one body.  It validates that every schedule row
    references an existing data-op or compute DSC before returning JSON.
    """
    if not name:
        raise ValueError("name is required")
    if not compute_dscs:
        raise ValueError("at least one compute DSC is required")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")

    data_count = len(datadscs)
    dsc_count = len(compute_dscs)
    overlap_candidate = False
    for core_id, rows in schedule.items():
        int(core_id)
        for row in rows:
            if len(row) != 4:
                raise ValueError(f"schedule row must have 4 fields, got {row}")
            data_idx, dsc_idx = int(row[0]), int(row[1])
            if data_idx == -1 and dsc_idx == -1:
                raise ValueError(f"empty schedule row is invalid: {row}")
            if data_idx < -1 or data_idx >= data_count:
                raise ValueError(f"data-op index out of range: {row}")
            if dsc_idx < -1 or dsc_idx >= dsc_count:
                raise ValueError(f"compute DSC index out of range: {row}")
            overlap_candidate = overlap_candidate or (
                data_idx >= 0 and dsc_idx >= 0
            )

    return {
        name: {
            "numCoresUsed_": num_cores,
            "coreIdToDscSchedule": {
                str(core_id): [list(row) for row in rows]
                for core_id, rows in schedule.items()
            },
            "datadscs_": [dict(dataop) for dataop in datadscs],
            "dscs_": [dict(dsc) for dsc in compute_dscs],
            "opFuncsUsed_": list(opfuncs),
            "flashAttentionPipeline_": {
                "tile_count": dsc_count,
                "dataop_count": data_count,
                "overlap_candidate": overlap_candidate,
            },
        }
    }


def build_streamed_bridge(
    dim_pool, iter_sizes, stick_size, num_cores, lx_size,
    src_base, dst_base, layout, stick_dim, src_split_dim, dst_split_dim, row_dim,
    slice_bytes, tile_bytes=STREAM_TILE_BYTES,
):
    """Tiled same-stick cross-core move for slices bigger than the LX buffers.

    The producer->consumer move is split into K = ceil(slice/tile) tiles along
    the non-stick row dim; each tile is one STCDPOpLx through ONE fixed in+out
    buffer pair (src_base/dst_base), windowing dimToSize_/dimToStartCordinate[row]
    so each tile is <= the buffer. src_split_dim -> dst_split_dim carries the same
    per-core ownership as build_same_layout_bridge (single move), so a streamed
    bundle's memId mapping mirrors the single-move structure tile by tile.

    The schedule is mixed_schedule(K): K data-op rows then the DL op. Tile k's
    before-sync + tile k+1's after-sync force buffer reuse -- tile k+1 cannot
    overwrite the buffer until tile k's consume drains.

    device-validate: single-buffer reuse; fallback = double-buffer (two tile pairs,
    ping-pong) -- a small change here: alternate two src/dst base pairs per k.
    """
    rows = iter_sizes[row_dim]
    k_tiles = num_stream_tiles(slice_bytes, tile_bytes)
    tr = tile_rows(rows, k_tiles)
    datadscs = []
    for k in range(k_tiles):
        r0 = k * tr
        nr = min(tr, rows - r0)
        in_ld = _tiled_labeled_ds(
            "dataIN", layout, stick_dim, src_split_dim, row_dim, iter_sizes,
            stick_size, src_base, num_cores, lx_size, r0, nr,
        )
        out_ld = _tiled_labeled_ds(
            "dataOUT", layout, stick_dim, dst_split_dim, row_dim, iter_sizes,
            stick_size, dst_base, num_cores, lx_size, r0, nr,
        )
        datadscs.append(
            _datadsc(f"{k}_STCDPOpLx_dataop", _stcdp_op(), dim_pool, in_ld, out_ld,
                     num_cores)
        )
    return datadscs, ["STCDPOpLx"] * k_tiles, mixed_schedule(k_tiles, num_cores)
