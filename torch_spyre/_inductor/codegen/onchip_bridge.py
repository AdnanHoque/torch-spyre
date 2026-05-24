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
        "PieceInfo": _piece_info(
            layout_order, split_dim, iter_sizes, chunk, base, num_cores, reverse
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


def _partition_pieces(
    stick_dim: str, owners: Sequence[int], starts: Sequence[int],
    lengths: Sequence[int], layout_order: Sequence[str],
    iter_sizes: Mapping[str, int], base: int,
) -> list[dict]:
    """Native PieceInfo: piece k owns [starts[k], starts[k]+lengths[k]) on owner k.

    A generalization of _piece_info for ASYMMETRIC reshard: pieces need not be
    equal-sized nor 1:1 with cores. Piece k covers stick_dim sub-range
    starts[k]..starts[k]+lengths[k]-1, full on every other dim, and is placed on
    core owners[k]. N (producer) and M (consumer) partitions need not match -- the
    same-stick STCDP overlap-cell engine (DCG createSubPieces) gathers each
    consumer piece from whichever producer pieces overlap it, riding the ring when
    owners differ. Equal-cell builders are the special case N==M, lengths uniform.
    """
    pieces = []
    for k, owner in enumerate(owners):
        start = {d: (starts[k] if d == stick_dim else 0) for d in layout_order}
        size = {d: (lengths[k] if d == stick_dim else iter_sizes[d])
                for d in layout_order}
        gap = {d: [[size[d], 0]] for d in layout_order}
        pieces.append(
            {
                "key_": f"p{k + 1}",
                "dimToStartCordinate": start,
                "dimToSize_": size,
                "validGap_": gap,
                "PlacementInfo": [
                    {"type": "lx", "memId": [owner], "startAddr": [base]}
                ],
            }
        )
    return pieces


def build_asymmetric_reshard_bridge(
    dim_pool: Sequence[str], iter_sizes: Mapping[str, int], stick_size: int,
    num_cores: int, lx_size: int, src_base: int, dst_base: int,
    layout: Sequence[str], stick_dim: str,
    prod_owners: Sequence[int], prod_starts: Sequence[int], prod_lens: Sequence[int],
    cons_owners: Sequence[int], cons_starts: Sequence[int], cons_lens: Sequence[int],
):
    """Single STCDPOpLx whose IN is N native producer pieces, OUT is M consumer pieces.

    Same-stick N->M cross-core redistribution where N != M and pieces may have
    unequal sizes/boundaries. The DCG STCDP overlap-cell engine loops every
    producer piece x consumer piece, intersects on stick_dim, and rides the ring
    for any src-owner != dst-owner cell -- no cell math is needed here. Pieces tile
    stick_dim disjointly on each side; emit producer NATIVE pieces in dataIN,
    consumer NATIVE pieces in dataOUT. The 8->25 granite bmm-out -> mul-in edge is
    this; the equal 32x32-cell builder is the special case.
    """
    in_ld = _labeled_ds("dataIN", layout, stick_dim, stick_dim, iter_sizes,
                        stick_size, src_base, num_cores, lx_size)
    in_ld["PieceInfo"] = _partition_pieces(
        stick_dim, prod_owners, prod_starts, prod_lens, layout, iter_sizes, src_base)
    out_ld = _labeled_ds("dataOUT", layout, stick_dim, stick_dim, iter_sizes,
                         stick_size, dst_base, num_cores, lx_size)
    out_ld["PieceInfo"] = _partition_pieces(
        stick_dim, cons_owners, cons_starts, cons_lens, layout, iter_sizes, dst_base)
    stcdp = _datadsc("0_STCDPOpLx_dataop", _stcdp_op(), dim_pool, in_ld, out_ld,
                     num_cores)
    return [stcdp], ["STCDPOpLx"], mixed_schedule(1, num_cores)


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
