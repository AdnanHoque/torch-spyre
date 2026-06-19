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

"""K-chunked warp-spec (MPMD PT||SFP||L3) schedule for the reduction reshard.

The value-correct reshard (``substrate.splice_reshard`` + ``mixed_schedule``)
runs the LX -> RIU ring -> LX gather as a single barrier-fenced data-op step
that fully precedes the consumer's K-reduction compute. This module software-
pipelines that move against the compute by chunking the consumer's ``K`` (the
``down_proj`` reduction extent, ``K=12800``) into ``k_chunks`` tiles and
interleaving three logical warps in the per-core ``coreIdToDscSchedule``:

  - **L3** -- the ``STCDPOpLx`` data-ops that move chunk ``t`` from the
    producer's ``{mb, out}`` co-split LX layout onto the cores that reduce it
    (one data-op per K-chunk; the ring leg).
  - **PT** -- the consumer's ``down_proj`` reduction DL DSC, which accumulates
    over the K-chunks (the matmul / PSUM-accumulate engine).
  - **SFP** -- the producer ``silu * mul`` pointwise that materializes each
    chunk into LX before its L3 move; modelled here as an extra data-op lane
    when ``sfp_lanes_per_chunk > 1`` (the pointwise + the gather share the
    overlap window).

The schedules emit the device-proven 4-field ``importJsonStr`` rows
``[datadsc_idx, dldsc_idx, before_sync, after_sync]`` (``superdsc.cpp:744-762``),
identical in shape to :func:`substrate.mixed_schedule`. They reorder the SAME
data-op / DL DSC indices; the value-correct serial schedule is the control. The
chunked data-op list is built by :func:`build_warpspec_reshard_chunks`, which
slices the producer ``out`` (consumer ``K``) into per-chunk column bands and
reuses :func:`substrate.build_perband_reshard_bridge` per chunk -- no new device
op shape, only a finer pipeline granularity.

Torch-free, no device, no dxp. Default off; gated on
``config.onchip_reduction_reshard_warpspec`` on top of the value-correct
reshard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .pieces import Band, Piece
from .substrate import build_perband_reshard_bridge


def warpspec_serial_schedule(num_chunks: int, num_cores: int) -> dict:
    """Control schedule: all ``num_chunks`` L3 moves, then the single PT DL row.

    The conservative, barrier-correct ordering -- every reshard data-op completes
    (each fenced after itself) before the consumer's reduction DL DSC runs. This
    is the warp-spec serial control; it has the same value semantics as
    :func:`substrate.mixed_schedule` but exposes the chunk granularity so the
    overlap schedule below can be A/B'd against it on identical data-ops.

    Row layout (4-field ``importJsonStr``):
      - chunk ``k``: ``[k, -1, 1 if k > 0 else 0, 1]`` (data-op, fenced after),
      - PT: ``[-1, 0, 1, 0]`` (the consumer DL DSC, fenced before).
    """
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    rows = [[k, -1, 1 if k > 0 else 0, 1] for k in range(num_chunks)]
    rows.append([-1, 0, 1, 0])
    return {str(c): [list(r) for r in rows] for c in range(num_cores)}


def warpspec_overlap_schedule(num_chunks: int, num_cores: int) -> dict:
    """Warp-spec overlap schedule: prologue L3 chunk 0, then pair PT_t || L3_{t+1}.

    The MPMD pipeline DXP's current mixed-SuperDSC contract admits (one DL
    compute DSC): prefetch chunk 0 as a prologue (L3 warp), then for each chunk
    ``t`` pair the consumer's reduction DL DSC for chunk ``t`` (PT warp) with the
    L3 move of chunk ``t + 1``, so the ring leg of the next chunk overlaps the
    PSUM-accumulate of the current one. The final chunk's DL row drains the
    pipeline.

    Because the consumer is a single accumulating DL DSC (``dldsc_idx = 0``),
    the same DL index is reissued per chunk (the PSUM accumulates across the
    K-chunks); only the data-op (``L3``) index advances. The ``before_sync`` /
    ``after_sync`` bits fence each issued row against its predecessor, so the
    overlap is expressed by ROW ORDER (L3_{t+1} immediately after PT_t) within
    the gate DXP currently honours.

    Row layout, ``num_chunks = C``:
      - prologue: ``[0, -1, 0, 1]`` (L3 chunk 0, no before-sync),
      - for ``t`` in ``0 .. C-2``:
          ``[-1, 0, 1, 1]``       (PT chunk ``t``),
          ``[t + 1, -1, 1, 1]``   (L3 chunk ``t + 1``, overlapping PT_t),
      - drain: ``[-1, 0, 1, 0]``  (PT chunk ``C-1``, fenced after = pipeline end).
    """
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if num_cores <= 0:
        raise ValueError(f"num_cores must be positive, got {num_cores}")
    rows: list[list[int]] = []
    rows.append([0, -1, 0, 1])  # prologue: L3 chunk 0 resident before any PT
    for t in range(num_chunks - 1):
        rows.append([-1, 0, 1, 1])  # PT chunk t
        rows.append([t + 1, -1, 1, 1])  # L3 chunk t+1 overlaps PT chunk t
    rows.append([-1, 0, 1, 0])  # PT chunk C-1 drains the pipeline
    return {str(c): [list(r) for r in rows] for c in range(num_cores)}


def build_warpspec_reshard_chunks(
    iter_sizes: Mapping[str, int],
    layout: Sequence[str],
    row_dim: str,
    stick_dim: str,
    m_split: int,
    n_split: int,
    cons_m_split: int,
    stick_size: int,
    num_cores: int,
    src_base: int,
    dst_base: int,
    k_chunks: int,
    lx_size: int,
) -> tuple[list[dict], list[str]]:
    """K-chunk the 2-D reshard: one ``STCDPOpLx`` data-op per consumer-K chunk.

    Slices the producer ``out`` extent (== the consumer reduction ``K``) into
    ``k_chunks`` equal column ranges and emits one per-band-style STCDP datadsc
    per chunk, so the L3 warp can be pipelined chunk-by-chunk against the PT
    reduction. Within a chunk, each producer ``{mb, out}`` co-split tile that
    falls inside the chunk's column range moves to the ``mb``-banded consumer
    rows at the SAME columns (``src_col == dst_col``) -- a pure row
    redistribution per chunk (the per-band shape, sidestepping the DCG EBR
    dest-column packer).

    Returns ``(datadscs, opfuncs)`` with ``len(datadscs) == k_chunks``; pair with
    :func:`warpspec_serial_schedule` or :func:`warpspec_overlap_schedule` for the
    same ``num_chunks = k_chunks``. Fail-closed: raises ``ValueError`` if ``K`` is
    not evenly divisible by ``k_chunks`` or by ``n_split``, or if the per-chunk
    column step is not a whole number of producer ``out`` bands.
    """
    if k_chunks <= 0:
        raise ValueError(f"k_chunks must be positive, got {k_chunks}")
    m_rows = int(iter_sizes[row_dim])
    k_extent = int(iter_sizes[stick_dim])
    if k_extent % k_chunks:
        raise ValueError(
            f"K={k_extent} not divisible by k_chunks={k_chunks}"
        )
    if k_extent % n_split:
        raise ValueError(f"K={k_extent} not divisible by n_split={n_split}")
    chunk_cols = k_extent // k_chunks
    out_band = k_extent // n_split
    if chunk_cols % out_band:
        raise ValueError(
            f"chunk column step {chunk_cols} is not a whole number of producer "
            f"out bands (band={out_band})"
        )
    if m_rows % m_split or m_rows % cons_m_split:
        raise ValueError(
            f"m_rows={m_rows} not divisible by m_split={m_split} / "
            f"cons_m_split={cons_m_split}"
        )

    def _producer_owner(mb_band: int, out_band_idx: int) -> int:
        return mb_band + m_split * out_band_idx

    row_step = m_rows // m_split
    cons_row_step = m_rows // cons_m_split
    bridge_iter = {row_dim: m_rows, stick_dim: k_extent}

    datadscs: list[dict] = []
    opfuncs: list[str] = []
    for chunk in range(k_chunks):
        col0 = chunk * chunk_cols
        # Producer out-bands whose columns fall inside this chunk's K-range.
        out_lo = col0 // out_band
        out_hi = (col0 + chunk_cols) // out_band
        edges: list[tuple[list[Piece], list[Piece]]] = []
        for ob in range(out_lo, out_hi):
            producer = [
                Piece(
                    key=f"p{mb + 1}",
                    owner=_producer_owner(mb, ob),
                    rows=Band(mb * row_step, row_step),
                    cols=Band(ob * out_band, out_band),
                )
                for mb in range(m_split)
            ]
            consumer = [
                Piece(
                    key=f"p{c + 1}",
                    owner=c,
                    rows=Band(c * cons_row_step, cons_row_step),
                    cols=Band(ob * out_band, out_band),
                )
                for c in range(cons_m_split)
            ]
            edges.append((producer, consumer))
        chunk_datadscs, chunk_opfuncs, _ = build_perband_reshard_bridge(
            edges,
            dim_pool=layout,
            iter_sizes=bridge_iter,
            stick_size=stick_size,
            num_cores=num_cores,
            lx_size=lx_size,
            src_base=src_base,
            dst_base=dst_base,
            layout=layout,
            row_dim=row_dim,
            stick_dim=stick_dim,
        )
        # One STCDP data-op per K-chunk: re-key its name so the chunk index is
        # the datadsc index the warp-spec schedule references.
        merged: dict = {}
        for band_dd in chunk_datadscs:
            merged.update(band_dd)
        datadscs.append({f"{chunk}_STCDPOpLx_dataop": _coalesce_bands(merged)})
        opfuncs.append("STCDPOpLx")
    return datadscs, opfuncs


def _coalesce_bands(band_datadscs: dict) -> dict:
    """Merge the per-out-band datadsc bodies of one K-chunk into one data-op.

    ``build_perband_reshard_bridge`` emits one datadsc per out-band; a K-chunk
    spanning multiple out-bands therefore yields several. They share the same
    ``op`` / ``dimPool_`` / ``coreIdsUsed_`` and differ only in their
    ``PieceInfo`` row sets, so the chunk's single STCDP carries the concatenated
    pieces of all its bands (the DCG overlap-cell engine handles the union).
    """
    bodies = list(band_datadscs.values())
    base = {k: v for k, v in bodies[0].items()}
    in_pieces: list[dict] = []
    out_pieces: list[dict] = []
    for body in bodies:
        in_ld, out_ld = body["labeledDs_"]
        in_pieces.extend(in_ld["PieceInfo"])
        out_pieces.extend(out_ld["PieceInfo"])
    in_ld0 = {k: v for k, v in base["labeledDs_"][0].items()}
    out_ld0 = {k: v for k, v in base["labeledDs_"][1].items()}
    in_ld0["PieceInfo"] = in_pieces
    out_ld0["PieceInfo"] = out_pieces
    base["labeledDs_"] = [in_ld0, out_ld0]
    return base
