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

"""Asymmetric piece builder for the on-chip core-to-core reshard.

A *piece* is one core's owned sub-rectangle of a logical tensor: a 2-D tile
``[row-band, col-band]`` over ``(rows, cols)`` placed on one core. This module
builds the producer and consumer ``PieceInfo`` lists (native, unequal sizes) for
a same-stick redistribution and feeds them, unchanged, to the same
``build_asymmetric_reshard_bridge`` / ``STCDPOpLx`` / ``createSubPieces`` path
proven on ``origin/attention-overlap`` -- the overlap cells are computed by DCG,
not here (see ``cells.py`` for the offline mirror that proves correctness).

The worked default is the Granite fused-SwiGLU ``matmul -> neg`` cross-division
edge (prefill ``1x512x4096``):

  Producer matmul ``{mb:4, out:8, in:1}``, output ``[512, 25600]``; each core
  owns ``128 rows x 3200 cols``; owner(core) = ``mb + 4*out`` (``in:1`` => no
  K-reduction, owners direct, no rep-core ambiguity).

  Consumer neg ``{mb:32, out:1}``, reads the gate half ``[0, 12800)``; each core
  owns ``16 rows x full-12800``; owner(core) = ``c``.

  Reshard map: consumer ``c`` <- producer ``{c//8, c//8+4, c//8+8, c//8+12}``
  (mb-band ``c//8``; the four ``out``-bands covering ``out in [0, 12800)``).

Generalizes to any ``N_p -> N_c`` (col split) and ``M_p -> M_c`` (row split),
same-stick, with the SwiGLU edge as the pinned, verified case. The producer
owner map is the caller's pin -- ``pieces.py`` does NOT re-derive owners (the
broken ``0b994bb`` failed by guessing them).

Torch-free: imports only ``regex`` and stdlib. Offline / compile-study only.
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
    25600/8=3200, 512/32=16). Uneven cases route through ``uniform_partition`` in
    ``substrate.py`` (whole-stick fail-closed). Raises ValueError otherwise so a
    bad split is caught offline, never silently truncated.
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

    The consumer may read a sub-range of the producer's columns (SwiGLU neg reads
    the gate half ``[0, 12800)`` of the combined ``25600``); ``n_extent`` is that
    read length, split ``n_split`` ways (gate-half: ``n_split=1`` => one full
    band). Rows split ``m_split`` ways. ``owner_fn(mb_band, out_band) -> core``;
    for the neg consumer ``{mb:32, out:1}`` it is ``lambda mb, out: mb``.
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

    Matches the ``_partition_pieces`` shape on ``origin/attention-overlap``
    (``codegen/onchip_bridge.py``) so the result is consumable by the same
    ``STCDPOpLx`` / ``createSubPieces`` path: ``dimToStartCordinate`` and
    ``dimToSize_`` carry the row band on ``row_dim`` and the col band on
    ``stick_dim``, full on any other layout dim; ``validGap_`` is one
    ``[size, 0]`` per dim; placement is LX on ``owner`` at ``base``.
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


# --- The pinned SwiGLU matmul->neg edge (do NOT re-derive owners) -------------

# Producer matmul output [512, 25600]; consumer neg reads gate half [0, 12800).
SWIGLU_M_ROWS = 512
SWIGLU_N_COLS = 25600
SWIGLU_GATE_EXTENT = 12800
SWIGLU_PROD_SPLIT = {"mb": 4, "out": 8}
SWIGLU_CONS_SPLIT = {"mb": 32, "out": 1}


def swiglu_producer_owner(mb_band: int, out_band: int) -> int:
    """Producer matmul owner: ``mb + 4*out`` (pinned, verified from device)."""
    return mb_band + SWIGLU_PROD_SPLIT["mb"] * out_band


def swiglu_consumer_owner(mb_band: int, out_band: int) -> int:
    """Consumer neg owner: ``c`` (mb-banded, out not split)."""
    return mb_band


def swiglu_reshard_sources(c: int) -> list[int]:
    """Pinned reshard map: consumer core ``c`` <- producer cores it reads from.

    ``{c//8, c//8+4, c//8+8, c//8+12}``: mb-band ``c//8`` (the producer rows that
    overlap consumer ``c``'s 16-row band) crossed with the four producer
    ``out``-bands ``{0,1,2,3}`` that cover the gate half ``out in [0, 12800)``
    (each out-band = 3200 cols; 4*3200 = 12800). Producer owner = ``mb + 4*out``
    => sources ``c//8 + 4*{0,1,2,3}``.
    """
    mb_band = c // 8
    return [mb_band + SWIGLU_PROD_SPLIT["mb"] * out_b for out_b in range(4)]


def build_swiglu_edge() -> tuple[list[Piece], list[Piece]]:
    """The worked default: producer + consumer pieces for the SwiGLU matmul->neg.

    Returns ``(producer_pieces, consumer_pieces)`` -- 32 producer pieces (4x8)
    over ``[512, 25600]`` and 32 consumer pieces (32x1) over the gate half
    ``[512, 12800)``. The consumer reads only ``out in [0, 12800)``; the producer
    pieces are emitted at full ``25600`` so the overlap engine intersects the gate
    half against the four covering producer out-bands.
    """
    producer = build_producer_pieces(
        SWIGLU_M_ROWS, SWIGLU_N_COLS,
        SWIGLU_PROD_SPLIT["mb"], SWIGLU_PROD_SPLIT["out"], swiglu_producer_owner,
    )
    consumer = build_consumer_pieces(
        SWIGLU_M_ROWS, SWIGLU_GATE_EXTENT,
        SWIGLU_CONS_SPLIT["mb"], SWIGLU_CONS_SPLIT["out"], swiglu_consumer_owner,
    )
    return producer, consumer
