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

"""Offline unit tests for the asymmetric reshard core (pure Python, no torch).

Runs the structural gate (``cells.assert_partition``) on:
  - the pinned Granite SwiGLU ``matmul -> neg`` edge (4x8 -> 32x1, gate half),
    checking the reshard map equals ``c <- {c//8, c//8+4, c//8+8, c//8+12}``;
  - a synthetic 1-D ``8 -> 25`` reshard (the granite bmm->mul example shape);
  - an even ``32 -> 32`` reshard (the symmetric special case).

Run directly (``python ab/reshard/test_reshard.py``) for a pass/fail summary, or
under pytest. No device, no benchmark, no dxp -- compile-study only.
"""

from __future__ import annotations

import sys

# Allow ``python ab/reshard/test_reshard.py`` (script) AND pytest (package).
if __package__ in (None, ""):
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from reshard import pieces as pieces_mod
    from reshard.cells import assert_partition, ring_map
    from reshard.pieces import (
        Band,
        Piece,
        build_consumer_pieces,
        build_producer_pieces,
        build_swiglu_edge,
        build_swiglu_unfused_edge,
        swiglu_reshard_sources,
        swiglu_unfused_reshard_sources,
    )
    from reshard.substrate import build_asymmetric_reshard_bridge
else:
    from . import pieces as pieces_mod  # noqa: F401
    from .cells import assert_partition, ring_map
    from .pieces import (
        Band,
        Piece,
        build_consumer_pieces,
        build_producer_pieces,
        build_swiglu_edge,
        build_swiglu_unfused_edge,
        swiglu_reshard_sources,
        swiglu_unfused_reshard_sources,
    )
    from .substrate import build_asymmetric_reshard_bridge


def test_swiglu_edge_map_and_partition():
    """SwiGLU matmul->neg: gate passes + map == c <- {c//8 + 4k}."""
    producer, consumer = build_swiglu_edge()
    assert len(producer) == 32  # 4 mb-bands x 8 out-bands
    assert len(consumer) == 32  # 32 mb-bands x 1 out-band

    cells = assert_partition(producer, consumer)
    rmap = ring_map(cells)

    for c in range(32):
        expected = set(swiglu_reshard_sources(c))
        assert expected == {c // 8, c // 8 + 4, c // 8 + 8, c // 8 + 12}, c
        assert rmap[c] == expected, (
            f"core {c}: got {sorted(rmap[c])}, want {sorted(expected)}"
        )

    # Every consumer core gathers from exactly 4 producer cores; all are ring
    # moves (consumer c's owner c never equals any producer owner mb+4*out for
    # these bands except by coincidence -- count the genuine cross-core ones).
    ring_cells = [x for x in cells if x.is_ring]
    assert ring_cells, "expected cross-core ring traffic on the SwiGLU edge"


def test_swiglu_unfused_edge_map_and_partition():
    """Unfused SwiGLU gate-matmul->neg: gate passes + map == c <- {c//8 + 4k}.

    Full-out [512, 12800] on both sides (no gate-half sub-slice), so each consumer
    spans all EIGHT producer out-bands -> 8 sources per consumer, 256 cells.
    """
    producer, consumer = build_swiglu_unfused_edge()
    assert len(producer) == 32  # 4 mb-bands x 8 out-bands
    assert len(consumer) == 32  # 32 mb-bands x 1 out-band

    cells = assert_partition(producer, consumer)
    assert len(cells) == 256  # 32 consumers x 8 producer out-bands each
    rmap = ring_map(cells)

    for c in range(32):
        expected = set(swiglu_unfused_reshard_sources(c))
        assert expected == {c // 8 + 4 * k for k in range(8)}, c
        assert rmap[c] == expected, (
            f"core {c}: got {sorted(rmap[c])}, want {sorted(expected)}"
        )
        assert len(rmap[c]) == 8  # eight sources (full out), not four

    # Full [512, 12800] coverage, whole-stick, with genuine ring traffic.
    assert sum(x.n_elems for x in cells) == 512 * 12800
    for x in cells:
        assert x.cols.start % pieces_mod.STICK_ELEMS == 0
        assert x.cols.length % pieces_mod.STICK_ELEMS == 0
    assert any(x.is_ring for x in cells)


def test_swiglu_cells_whole_stick_and_total():
    """Each SwiGLU cell is whole-stick; cells tile the 512x12800 gate region."""
    producer, consumer = build_swiglu_edge()
    cells = assert_partition(producer, consumer)
    # gate region = 512 rows x 12800 cols.
    assert sum(c.n_elems for c in cells) == 512 * 12800
    for c in cells:
        assert c.cols.start % pieces_mod.STICK_ELEMS == 0
        assert c.cols.length % pieces_mod.STICK_ELEMS == 0


def test_synthetic_8_to_25():
    """1-D 8->25 reshard over a stick-aligned length passes the gate.

    Mirrors the documented granite bmm->mul example. 8 and 25 must each divide
    the length evenly into whole sticks: pick length = 8*25*64 = 12800 sticks too
    big; use lcm(8,25)=200 sticks -> 200*64 = 12800 elems so both even+whole-stick.
    """
    length = 200 * 64  # divisible by 8 and 25, each band whole-stick
    rows = 64
    producer, consumer = _one_d_pieces_split(length, rows, 8, 25)
    cells = assert_partition(producer, consumer)
    rmap = ring_map(cells)
    assert len(producer) == 8 and len(consumer) == 25
    assert sum(c.n_elems for c in cells) == length * rows
    # Cross-core: consumer owners 0..24, producer owners 0..7 -> most differ.
    assert any(c.is_ring for c in cells)
    assert set(rmap.keys()) == set(range(25))


def _one_d_pieces_split(length: int, rows: int, n_p: int, n_c: int):
    """Producer N_p col-bands + consumer N_c col-bands, 1 row-band, owner=out."""
    producer = build_producer_pieces(rows, length, 1, n_p, lambda mb, out: out)
    consumer = build_consumer_pieces(rows, length, 1, n_c, lambda mb, out: out)
    return producer, consumer


def test_even_32_to_32():
    """Even 32->32 reshard (symmetric special case) passes the gate.

    Producer and consumer split the col dim 32 ways identically but with mirrored
    owners (producer owner = out, consumer owner = 31-out) to force genuine ring
    traffic on every cell (the build_roundtrip_bridge reverse-map analogue).
    """
    length = 32 * 64  # 32 whole sticks
    rows = 16
    producer = build_producer_pieces(rows, length, 1, 32, lambda mb, out: out)
    consumer = build_consumer_pieces(rows, length, 1, 32, lambda mb, out: 31 - out)
    cells = assert_partition(producer, consumer)
    assert len(cells) == 32  # 1:1 aligned boundaries -> one cell per band
    assert all(c.is_ring for c in cells)  # mirrored owners -> every move rings


def test_gate_rejects_gap():
    """A consumer piece not covered by any producer piece must fail the gate."""
    rows = 16
    # Producer covers [0, 64); consumer reads [0, 128) -> [64,128) has no source.
    producer = [Piece("p1", 0, Band(0, rows), Band(0, 64))]
    consumer = [Piece("p1", 0, Band(0, rows), Band(0, 128))]
    try:
        assert_partition(producer, consumer)
    except AssertionError:
        return
    raise AssertionError("gate failed to reject an uncovered consumer region")


def test_gate_rejects_overlap():
    """Two producer pieces covering the same element must fail the gate."""
    rows = 16
    producer = [
        Piece("p1", 0, Band(0, rows), Band(0, 128)),
        Piece("p2", 1, Band(0, rows), Band(0, 128)),  # duplicate region
    ]
    consumer = [Piece("p1", 0, Band(0, rows), Band(0, 128))]
    try:
        assert_partition(producer, consumer)
    except AssertionError:
        return
    raise AssertionError("gate failed to reject double-sourced consumer region")


def test_substrate_emits_single_stcdp_with_2d_pieces():
    """The ported bridge builder renders the 2-D SwiGLU pieces into one STCDP.

    Shape-only check (no device): one STCDPOpLx datadsc, 32 producer PieceInfo in
    dataIN + 32 consumer PieceInfo in dataOUT, schedule = one data-op + DL op, and
    each piece carries BOTH a row band (mb_) and a col band (out_) in dimToSize_.
    """
    producer, consumer = build_swiglu_edge()
    layout = ["mb_", "out_"]
    iter_sizes = {"mb_": 512, "out_": 25600}
    datadscs, opfuncs, sched = build_asymmetric_reshard_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=pieces_mod.STICK_ELEMS,
        num_cores=32,
        lx_size=2 << 20,
        src_base=0,
        dst_base=1 << 20,
        layout=layout,
        row_dim="mb_",
        stick_dim="out_",
        producer_pieces=producer,
        consumer_pieces=consumer,
    )
    assert opfuncs == ["STCDPOpLx"] and len(datadscs) == 1
    dd = datadscs[0]["0_STCDPOpLx_dataop"]
    in_p = dd["labeledDs_"][0]["PieceInfo"]
    out_p = dd["labeledDs_"][1]["PieceInfo"]
    assert len(in_p) == 32 and len(out_p) == 32
    assert len(sched["0"]) == 2  # one data-op + DL op
    # producer piece p1 = mb-band 0 (128 rows) x out-band 0 (3200 cols) on core 0.
    p1 = in_p[0]
    assert p1["dimToSize_"] == {"mb_": 128, "out_": 3200}
    assert p1["PlacementInfo"][0]["memId"] == [0]
    # consumer piece p1 = mb-band 0 (16 rows) x gate (12800 cols) on core 0.
    c1 = out_p[0]
    assert c1["dimToSize_"] == {"mb_": 16, "out_": 12800}


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
