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

"""Offline mirror of DCG ``createSubPieces`` + the structural correctness gate.

On device, ``DcgFE::createSubPieces(STCDPOpLx*)`` loops every consumer piece x
every producer piece, calls ``doesPiecesOverlap`` (a rectangle-intersection test
on ``dimToStartCordinate``/``dimToSize_``), and for each non-empty intersection
registers one LX->LX sub-move keyed by ``src memId -> dst memId``. This module
recomputes those cells in pure Python so we can PROVE the redistribution is
total and disjoint BEFORE any device run -- the safety net the broken ``0b994bb``
(``max_err 0.669``) lacked because it guessed owners and never checked coverage.

A *cell* is the intersection of one producer piece and one consumer piece:
rows = ``[max(starts), min(ends))``, cols = same on the stick dim. Non-empty
cells are the actual ring sub-moves (``src=producer.owner -> dst=consumer.owner``).

``assert_partition`` is the gate: the union of cells equals the consumer region
exactly (no gaps, no overlap), every cell is whole-stick, and every consumer
element is sourced from exactly one producer fragment. Torch-free.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .pieces import STICK_ELEMS, Band, Piece


@dataclasses.dataclass(frozen=True)
class Cell:
    """One producer->consumer overlap sub-move (a ring LX->LX copy on device).

    ``rows``/``cols`` are the intersection rectangle; ``src`` is the producer
    owner core, ``dst`` the consumer owner core. ``src != dst`` => the cell rides
    the ring (cross-core); ``src == dst`` => a local LX copy.
    """

    rows: Band
    cols: Band
    src: int
    dst: int

    @property
    def is_ring(self) -> bool:
        return self.src != self.dst

    @property
    def n_elems(self) -> int:
        return self.rows.length * self.cols.length


def _intersect(a: Band, b: Band) -> Band | None:
    """Intersection of two bands, or None if empty (the ``doesPiecesOverlap`` test)."""
    start = max(a.start, b.start)
    end = min(a.end, b.end)
    if start >= end:
        return None
    return Band(start, end - start)


def compute_cells(
    producer: Sequence[Piece], consumer: Sequence[Piece]
) -> list[Cell]:
    """All non-empty producer x consumer overlap cells (mirrors createSubPieces).

    Loops every consumer piece against every producer piece and keeps each
    non-empty 2-D (rows x cols) intersection. The result is the exact set of
    ring sub-moves DCG would emit for this STCDP. Each consumer piece is filled
    by the cells whose ``dst`` is its owner.
    """
    cells: list[Cell] = []
    for cons in consumer:
        for prod in producer:
            rows = _intersect(cons.rows, prod.rows)
            cols = _intersect(cons.cols, prod.cols)
            if rows is None or cols is None:
                continue
            cells.append(
                Cell(rows=rows, cols=cols, src=prod.owner, dst=cons.owner)
            )
    return cells


def consumer_region(consumer: Sequence[Piece]) -> tuple[Band, Band]:
    """The (rows, cols) bounding region the consumer reads, as a sanity bound.

    Returns the min-start/max-end over the consumer pieces on each axis. The gate
    proves the cells tile exactly this region; this just gives the target extent.
    """
    r0 = min(p.rows.start for p in consumer)
    r1 = max(p.rows.end for p in consumer)
    c0 = min(p.cols.start for p in consumer)
    c1 = max(p.cols.end for p in consumer)
    return Band(r0, r1 - r0), Band(c0, c1 - c0)


def assert_partition(
    producer: Sequence[Piece],
    consumer: Sequence[Piece],
    stick_elems: int = STICK_ELEMS,
) -> list[Cell]:
    """Structural correctness gate. Raises AssertionError on any violation.

    The MUST-PASS check before any device run (the ``0b994bb`` safety net).
    Verifies, on the overlap cells computed exactly as DCG would:

    1. every cell is whole-stick on the stick (col) dim -- start and length both
       multiples of ``stick_elems`` (DCG rejects sub-stick cells);
    2. for EACH consumer piece, its overlap cells tile its rectangle exactly --
       total area, no gaps, no double-cover (every element sourced ONCE);
    3. each consumer element comes from exactly one producer fragment (implied by
       2 since producer pieces are disjoint, asserted explicitly);
    4. the union of all cells equals the union of consumer pieces (total cover).

    Returns the cells so callers can inspect the ring map after the gate passes.
    """
    cells = compute_cells(producer, consumer)

    # (0) producer pieces must be mutually disjoint (else an element has two
    # sources). Consumer pieces likewise (else double-write).
    _assert_disjoint(producer, "producer")
    _assert_disjoint(consumer, "consumer")

    # (1) whole-stick on the col (stick) dim.
    for cell in cells:
        assert cell.cols.start % stick_elems == 0, (
            f"cell col start {cell.cols.start} not stick-aligned "
            f"({stick_elems})"
        )
        assert cell.cols.length % stick_elems == 0, (
            f"cell col length {cell.cols.length} not whole-stick "
            f"({stick_elems})"
        )

    # (2)+(3) per consumer piece, its cells tile its rectangle exactly and each
    # element is sourced once.
    for cons in consumer:
        owned = [c for c in cells if c.dst == cons.owner
                 and _within(c, cons)]
        target = cons.rows.length * cons.cols.length
        covered = _exact_cover_area(owned, cons)
        assert covered == target, (
            f"consumer core {cons.owner}: cells cover {covered} of {target} "
            f"elements (gap or overlap) over rows={cons.rows} cols={cons.cols}"
        )

    # (4) global: total cell area == total consumer area, and cells are pairwise
    # disjoint across the whole redistribution.
    total_cell_area = sum(c.n_elems for c in cells)
    total_cons_area = sum(p.rows.length * p.cols.length for p in consumer)
    assert total_cell_area == total_cons_area, (
        f"total cell area {total_cell_area} != consumer area {total_cons_area}"
    )
    return cells


def _within(cell: Cell, piece: Piece) -> bool:
    """True if the cell lies inside the consumer piece's rectangle."""
    return (
        piece.rows.start <= cell.rows.start
        and cell.rows.end <= piece.rows.end
        and piece.cols.start <= cell.cols.start
        and cell.cols.end <= piece.cols.end
    )


def _exact_cover_area(cells: Sequence[Cell], piece: Piece) -> int:
    """Area covered by ``cells`` within ``piece``, asserting no pairwise overlap.

    Cells of one consumer piece must be disjoint (an element sourced once). Sums
    their areas after asserting no two overlap, so the caller can compare against
    the piece area for an exact-tiling (no-gap) check.
    """
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            if _rect_overlap(cells[i], cells[j]):
                raise AssertionError(
                    f"overlapping cells on consumer core {piece.owner}: "
                    f"{cells[i]} and {cells[j]} (element double-sourced)"
                )
    return sum(c.n_elems for c in cells)


def _rect_overlap(a: Cell, b: Cell) -> bool:
    """True if two cells' rectangles overlap (positive-area intersection)."""
    return (
        _intersect(a.rows, b.rows) is not None
        and _intersect(a.cols, b.cols) is not None
    )


def _assert_disjoint(pieces: Sequence[Piece], side: str) -> None:
    """Assert the pieces' rectangles are pairwise disjoint (one source/dst each)."""
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            ri = _intersect(pieces[i].rows, pieces[j].rows)
            ci = _intersect(pieces[i].cols, pieces[j].cols)
            if ri is not None and ci is not None:
                raise AssertionError(
                    f"{side} pieces overlap: {pieces[i].key} and {pieces[j].key}"
                )


def ring_map(cells: Sequence[Cell]) -> dict[int, set[int]]:
    """``{consumer_core: {producer_cores it sources from}}`` -- the reshard map.

    Built from the overlap cells. For the SwiGLU edge this must equal the pinned
    ``c <- {c//8, c//8+4, c//8+8, c//8+12}`` (asserted in the tests).
    """
    out: dict[int, set[int]] = {}
    for cell in cells:
        out.setdefault(cell.dst, set()).add(cell.src)
    return out
