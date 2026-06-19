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

"""Asymmetric piece builder for the on-chip core-to-core reduction reshard.

A *piece* is one core's owned sub-rectangle of a logical tensor: a 2-D tile
``[row-band, col-band]`` over ``(rows, cols)`` placed on one core. This module
builds the producer and consumer ``PieceInfo`` lists (native, unequal sizes) for
a same-stick redistribution and feeds them, unchanged, to the
``substrate.build_asymmetric_reshard_bridge`` / ``STCDPOpLx`` path. The overlap
cells are computed by DCG (``createSubPieces``), not here.

The worked default is the Granite SwiGLU ``mul -> down_proj`` reduction-input
edge (the genuine non-co-assignable reshard): the down-proj reduces over
``K=12800`` which IS the mul's split dim, so the mul output must be gathered
core-to-core (LX -> RIU ring -> LX) instead of round-tripping through HBM.

Worked it-space (prefill ``1x512x4096``, fused gate/up matmul ``[512, 25600]``):

  Producer ``mul`` output ``[512, 12800]`` (the gate*up product), co-split
  ``{mb:4, out:8}``; each core owns ``128 rows x 1600 cols``; owner(core) =
  ``mb + 4*out``.

  Consumer ``down_proj`` matmul reducing over ``K=12800`` (= the producer's
  split dim), ``{mb:32, out:1}`` over its own output; each core that holds a
  K-shard owns ``16 rows x 1600 cols`` of the activation it reduces.

The producer owner map is the caller's pin -- this module does NOT re-derive
owners.

Torch-free: imports only stdlib. Offline / compile-study safe.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence

# A stick is 128 bytes = 64 fp16 elements (AIU 1.0). Source of truth:
# CLAUDE.md "Spyre Hardware Basics" + onchip_bridge.STICK_BYTES.
STICK_ELEMS = 64


@dataclasses.dataclass(frozen=True)
class Band:
    """One contiguous sub-range ``[start, start + length)`` of a logical axis."""

    start: int
    length: int

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclasses.dataclass(frozen=True)
class Piece:
    """One core's owned 2-D tile: a row-band x a col-band, on ``owner``.

    ``key`` is the deeptools PieceInfo ``key_`` (``p1``, ``p2`` ...). ``owner`` is
    the LX core that holds the tile (PieceInfo ``PlacementInfo.memId``). The col
    band lives on the stick dim; the row band is the non-stick dim.
    """

    key: str
    owner: int
    rows: Band
    cols: Band


def _even_bands(extent: int, n: int) -> list[Band]:
    """Split ``[0, extent)`` into ``n`` equal contiguous bands.

    Requires ``extent % n == 0`` -- the SwiGLU splits are exact (512/4=128,
    12800/8=1600, 512/32=16). Raises ValueError otherwise so a bad split is
    caught offline, never silently truncated.
    """
    if n <= 0 or extent % n != 0:
        raise ValueError(f"cannot split extent {extent} into {n} even bands")
    step = extent // n
    return [Band(i * step, step) for i in range(n)]


def build_producer_pieces(
    m_rows: int,
    n_cols: int,
    m_split: int,
    n_split: int,
    owner_fn: Callable[[int, int], int],
) -> list[Piece]:
    """Producer pieces for a ``{mb:m_split, out:n_split}`` 2-D work division.

    ``owner_fn(mb_band, out_band) -> core`` is the caller's PINNED owner map (for
    SwiGLU: ``mb + 4*out``). Bands are ordered out-major then mb (``p`` index =
    ``mb + m_split*out``) so the key order matches the device ``coreIdToWkSlice_``
    walk; ``owner_fn`` -- not the index -- decides the physical core. Each piece
    is ``(m_rows/m_split) rows x (n_cols/n_split) cols`` at NATIVE size.
    """
    row_bands = _even_bands(m_rows, m_split)
    col_bands = _even_bands(n_cols, n_split)
    pieces: list[Piece] = []
    k = 0
    for out_b in range(n_split):
        for mb_b in range(m_split):
            pieces.append(
                Piece(
                    key=f"p{k + 1}",
                    owner=owner_fn(mb_b, out_b),
                    rows=row_bands[mb_b],
                    cols=col_bands[out_b],
                )
            )
            k += 1
    return pieces


def build_consumer_pieces(
    m_rows: int,
    n_extent: int,
    m_split: int,
    n_split: int,
    owner_fn: Callable[[int, int], int],
) -> list[Piece]:
    """Consumer pieces over the region it actually READS: ``cols in [0, n_extent)``.

    Rows split ``m_split`` ways; the read columns split ``n_split`` ways.
    ``owner_fn(mb_band, out_band) -> core``; for the down-proj K-reduction
    consumer ``{mb:32, out:1}`` it is ``lambda mb, out: mb``.
    """
    row_bands = _even_bands(m_rows, m_split)
    col_bands = _even_bands(n_extent, n_split)
    pieces: list[Piece] = []
    k = 0
    for out_b in range(n_split):
        for mb_b in range(m_split):
            pieces.append(
                Piece(
                    key=f"p{k + 1}",
                    owner=owner_fn(mb_b, out_b),
                    rows=row_bands[mb_b],
                    cols=col_bands[out_b],
                )
            )
            k += 1
    return pieces


def piece_to_dict(
    piece: Piece,
    layout_order: Sequence[str],
    row_dim: str,
    stick_dim: str,
    iter_sizes: Mapping[str, int],
    base: int,
) -> dict:
    """Render one :class:`Piece` as a deeptools ``PieceInfo`` entry.

    ``dimToStartCordinate`` and ``dimToSize_`` carry the row band on ``row_dim``
    and the col band on ``stick_dim``, full on any other layout dim; ``validGap_``
    is one ``[size, 0]`` per dim; placement is LX on ``owner`` at ``base``.
    """
    start = {}
    size = {}
    for d in layout_order:
        if d == row_dim:
            start[d], size[d] = piece.rows.start, piece.rows.length
        elif d == stick_dim:
            start[d], size[d] = piece.cols.start, piece.cols.length
        else:
            start[d], size[d] = 0, iter_sizes[d]
    gap = {d: [[size[d], 0]] for d in layout_order}
    return {
        "key_": piece.key,
        "dimToStartCordinate": start,
        "dimToSize_": size,
        "validGap_": gap,
        "PlacementInfo": [{"type": "lx", "memId": [piece.owner], "startAddr": [base]}],
    }


def pieces_to_pieceinfo(
    pieces: Sequence[Piece],
    layout_order: Sequence[str],
    row_dim: str,
    stick_dim: str,
    iter_sizes: Mapping[str, int],
    base: int,
) -> list[dict]:
    """Render a list of :class:`Piece` as a deeptools ``PieceInfo`` list."""
    return [
        piece_to_dict(p, layout_order, row_dim, stick_dim, iter_sizes, base)
        for p in pieces
    ]


# --- The pinned SwiGLU mul->down_proj reduction-input edge (do NOT re-derive) --
#
# Producer mul output is the gate*up product [512, 12800]; the down-proj reduces
# over the full K=12800 (= the mul split dim). The producer co-splits {mb:4,
# out:8}; the consumer is mb-banded {mb:32}, reading the FULL K it reduces.

SWIGLU_M_ROWS = 512
SWIGLU_K_EXTENT = 12800  # the reduction dim K = mul output cols = down-proj K
SWIGLU_PROD_SPLIT = {"mb": 4, "out": 8}
SWIGLU_CONS_SPLIT = {"mb": 32, "out": 1}


def swiglu_producer_owner(mb_band: int, out_band: int) -> int:
    """Producer mul owner: ``mb + 4*out`` (the {mb:4,out:8} co-split)."""
    return mb_band + SWIGLU_PROD_SPLIT["mb"] * out_band


def swiglu_consumer_owner(mb_band: int, out_band: int) -> int:
    """Consumer down-proj owner: ``c`` (mb-banded, out not split)."""
    return mb_band


def swiglu_reshard_sources(c: int) -> list[int]:
    """Pinned reshard map: consumer core ``c`` <- the producer cores it reads.

    The consumer reads the FULL ``K in [0, 12800)``, so it spans ALL EIGHT
    producer ``out``-bands (each 1600 cols; 8*1600 = 12800). mb-band ``c//8`` are
    the producer rows overlapping consumer ``c``'s 16-row band. Producer owner =
    ``mb + 4*out`` => sources ``c//8 + 4*{0..7}`` -- eight cores per consumer
    (32 * 8 = 256 single-source cells).
    """
    mb_band = c // 8
    return [mb_band + SWIGLU_PROD_SPLIT["mb"] * out_b for out_b in range(8)]


def build_swiglu_edge() -> tuple[list[Piece], list[Piece]]:
    """Producer + consumer pieces for the SwiGLU mul->down_proj reduction edge.

    Returns ``(producer_pieces, consumer_pieces)`` -- 32 producer pieces (4x8)
    over the full ``[512, 12800]`` mul output (owner ``mb + 4*out``) and 32
    consumer pieces (32x1) over the SAME full ``[512, 12800]`` K (owner ``c``).
    The consumer reads the WHOLE K, so the overlap engine intersects each
    consumer 16-row band against all eight producer out-bands -> 256
    single-source, whole-stick cells mapping ``c <- {c//8 + 4k : k=0..7}``.
    """
    producer = build_producer_pieces(
        SWIGLU_M_ROWS,
        SWIGLU_K_EXTENT,
        SWIGLU_PROD_SPLIT["mb"],
        SWIGLU_PROD_SPLIT["out"],
        swiglu_producer_owner,
    )
    consumer = build_consumer_pieces(
        SWIGLU_M_ROWS,
        SWIGLU_K_EXTENT,
        SWIGLU_CONS_SPLIT["mb"],
        SWIGLU_CONS_SPLIT["out"],
        swiglu_consumer_owner,
    )
    return producer, consumer


# --- Per-band decomposition: 8 single-column-band STCDPs (NO intra-row scatter) -
#
# The single-STCDP edge hands DCG the full 2-D scatter: 32 producer pieces (4x8
# co-split) -> 32 consumer pieces spanning the FULL [0,12800) row, so DCG must
# place each producer out-band at its column offset WITHIN the consumer row. That
# intra-row column placement is what the EBR packer mis-linearises (3200*core vs
# 3200*(core//4)).
#
# This decomposition instead emits ONE STCDP per out-band. STCDP ``b`` moves only
# column band ``[b*1600, +1600)``: BOTH producer and consumer pieces sit at the
# SAME logical column band (src_col == dst_col) -- a pure row (mb) redistribution
# at a FIXED column. No intra-row column placement for the packer to linearise.


def build_swiglu_perband_edges() -> list[tuple[list[Piece], list[Piece]]]:
    """Eight ``(producer, consumer)`` edges, one per producer ``out``-band.

    Edge ``b`` (band cols ``[b*1600, +1600)``) has 4 producer pieces (the
    ``mb=0..3`` row-bands of ``out=b``, owner ``mb + 4*b``) and 32 consumer
    sub-slice pieces (each consumer core ``c``'s 16-row band, restricted to band
    ``b``'s columns, owner ``c``). Producer and consumer cols are IDENTICAL within
    an edge -> no column re-placement. The 8 edges tile the full ``[512, 12800]``
    exactly (8 * 32 = 256 single-source cells).
    """
    m_split = SWIGLU_PROD_SPLIT["mb"]  # 4
    n_split = SWIGLU_PROD_SPLIT["out"]  # 8
    cons_m_split = SWIGLU_CONS_SPLIT["mb"]  # 32
    prod_row_bands = _even_bands(SWIGLU_M_ROWS, m_split)  # 4 x 128
    col_bands = _even_bands(SWIGLU_K_EXTENT, n_split)  # 8 x 1600
    cons_row_bands = _even_bands(SWIGLU_M_ROWS, cons_m_split)  # 32 x 16
    edges: list[tuple[list[Piece], list[Piece]]] = []
    for b, col in enumerate(col_bands):
        producer = [
            Piece(
                key=f"p{mb + 1}",
                owner=swiglu_producer_owner(mb, b),
                rows=prod_row_bands[mb],
                cols=col,
            )
            for mb in range(m_split)
        ]
        consumer = [
            Piece(
                key=f"p{c + 1}",
                owner=swiglu_consumer_owner(c, 0),
                rows=cons_row_bands[c],
                cols=col,
            )
            for c in range(cons_m_split)
        ]
        edges.append((producer, consumer))
    return edges
