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

"""Emission layer for the asymmetric core-to-core reduction reshard.

Synthesizes the device data-movement program for the 2-D ``mul -> down_proj``
reduction-input edge: an ``STCDPOpLx`` (LX -> RIU ring -> LX) that gathers the
producer's ``{mb:4, out:8}`` co-split output onto the cores that reduce over
``K=12800``. Pure dict surgery -- torch-free, no device, no dxp.

Two emission shapes:
  - :func:`build_asymmetric_reshard_bridge` -- single ``STCDPOpLx``, N producer
    pieces in, M consumer pieces out (the DCG overlap-cell engine does the
    cells). Hands DCG the full 2-D scatter.
  - :func:`build_perband_reshard_bridge` -- one ``STCDPOpLx`` per column band
    (``src_col == dst_col`` within each band; pure row redistribution).

Two splice shapes:
  - :func:`splice_reshard` -- folds the STCDP into the consumer SDSC as a mixed
    (DL + data-op) SuperDSC. REJECTED by harvest dxp (``SdscTree.cpp:152``
    "Datadsc not allowed, use dldsc"); kept for reference / cross-check.
  - :func:`splice_reshard_standalone` -- emits the STCDP as its OWN pure-data-op
    SDSC step, routed through dxp's ``dxp.cpp:255`` (``dscs_==0 &&
    dataOpdscs_>0 -> dcg.runDcg``) branch. This is the live path.

All labeledDs are pure-LX (``isExternal_:0``, ``hbmSize_:0``) so the move stays
on the RIU ring (the ``L3_STU``/``L3_LDU`` legs). :func:`apply_lx_flip` flips a
DL labeledDs to LX-resident and clears the HBM-layout inter-core gap.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from .pieces import Piece, pieces_to_pieceinfo

# --- Constants (the device-proven STCDPOpLx schema) --------------------------
DATA_FORMAT = "SEN169_FP16"
WORD_LENGTH = 2
LX_CAPACITY_BYTES = 2 << 20  # 2 MB per-core LX (AIU 1.0)
STICK_BYTES = 128
# Per-core LX byte span declared inside each data-op labeledDs; DL-DSC LX size
# sentinel.
DATAOP_LX_SIZE = 2 << 20
DL_LX_SENTINEL = 2147483647


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@dataclasses.dataclass(frozen=True)
class LxFlip:
    """Flip one DL labeledDs to LX-resident: ``base`` per core, sentinel size."""

    ldsidx: int
    lx_base: int
    role: str  # "producer-output" | "consumer-input"


def _stcdp_op() -> dict:
    """The STCDPOpLx op stub."""
    return {"name": "STCDPOpLx"}


def mixed_schedule(num_dataops: int, num_cores: int) -> dict:
    """coreIdToDscSchedule rows: each data-op (before-sync), then the DL op.

    Row k for data-op k is ``[k, -1, 1 if k>0 else 0, 1]``; the final
    ``[-1, 0, 1, 0]`` is the DL op. Field order is
    ``[datadsc_idx, dldsc_idx, before_sync, after_sync]`` (the importJsonStr
    layout, ``superdsc.cpp:744-762`` -- the authoritative parser).
    """
    rows = []
    for k in range(num_dataops):
        rows.append([k, -1, 1 if k > 0 else 0, 1])
    rows.append([-1, 0, 1, 0])
    return {str(c): [list(r) for r in rows] for c in range(num_cores)}


def _labeled_ds(
    pds_name: str,
    layout_order: Sequence[str],
    stick_dim: str,
    iter_sizes: Mapping[str, int],
    stick_size: int,
    lx_size: int,
    piece_info: list[dict],
) -> dict:
    """One labeledDs (dataIN_L0 / dataOUT_L0) with caller-supplied PieceInfo.

    Pure-LX (``isExternal_:0``, ``hbmSize_:0``) so the move stays on the ring.
    """
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
        "PieceInfo": piece_info,
    }


def _datadsc(
    name: str,
    op: dict,
    dim_pool: Sequence[str],
    in_ld: dict,
    out_ld: dict,
    num_cores: int,
) -> dict:
    """One datadsc block."""
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


def build_asymmetric_reshard_bridge(
    dim_pool: Sequence[str],
    iter_sizes: Mapping[str, int],
    stick_size: int,
    num_cores: int,
    lx_size: int,
    src_base: int,
    dst_base: int,
    layout: Sequence[str],
    row_dim: str,
    stick_dim: str,
    producer_pieces: Sequence[Piece],
    consumer_pieces: Sequence[Piece],
) -> tuple[list[dict], list[str], dict]:
    """Single STCDPOpLx: N producer pieces in dataIN, M consumer pieces in dataOUT.

    Takes the 2-D :class:`Piece` lists (row-band x col-band per core) and renders
    them with ``pieces_to_pieceinfo``. The DCG overlap-cell engine
    (createSubPieces) does the cells; we feed native, unequal pieces. Every owner
    must be ``< num_cores`` (dxp rejects a cell sourced from a core outside the
    consumer's active corelet set).
    """
    owners = [p.owner for p in producer_pieces] + [p.owner for p in consumer_pieces]
    if any(o >= num_cores or o < 0 for o in owners):
        raise ValueError(
            f"owners out of range [0,{num_cores}): {sorted(set(owners))}"
        )
    in_pi = pieces_to_pieceinfo(
        producer_pieces, layout, row_dim, stick_dim, iter_sizes, src_base
    )
    out_pi = pieces_to_pieceinfo(
        consumer_pieces, layout, row_dim, stick_dim, iter_sizes, dst_base
    )
    in_ld = _labeled_ds(
        "dataIN", layout, stick_dim, iter_sizes, stick_size, lx_size, in_pi
    )
    out_ld = _labeled_ds(
        "dataOUT", layout, stick_dim, iter_sizes, stick_size, lx_size, out_pi
    )
    stcdp = _datadsc(
        "0_STCDPOpLx_dataop", _stcdp_op(), dim_pool, in_ld, out_ld, num_cores
    )
    return [stcdp], ["STCDPOpLx"], mixed_schedule(1, num_cores)


def build_perband_reshard_bridge(
    edges: Sequence[tuple[Sequence[Piece], Sequence[Piece]]],
    dim_pool: Sequence[str],
    iter_sizes: Mapping[str, int],
    stick_size: int,
    num_cores: int,
    lx_size: int,
    src_base: int,
    dst_base: int,
    layout: Sequence[str],
    row_dim: str,
    stick_dim: str,
) -> tuple[list[dict], list[str], dict]:
    """One STCDPOpLx per column band -- the per-band decomposition of the 2-D edge.

    ``edges[b] = (producer_pieces_b, consumer_pieces_b)`` covers only band ``b``'s
    columns; producer and consumer pieces share the same logical column band, so
    each STCDP is a pure row redistribution at a fixed column (no intra-row column
    re-placement). All bands read producer LX ``src_base`` and write consumer LX
    ``dst_base``; the per-band column offset is carried by the piece
    ``dimToStartCordinate`` (identical on src and dst), not by a base delta.
    """
    datadscs: list[dict] = []
    for b, (producer_pieces, consumer_pieces) in enumerate(edges):
        owners = [p.owner for p in producer_pieces] + [
            p.owner for p in consumer_pieces
        ]
        if any(o >= num_cores or o < 0 for o in owners):
            raise ValueError(
                f"band {b} owners out of range [0,{num_cores}): "
                f"{sorted(set(owners))}"
            )
        in_pi = pieces_to_pieceinfo(
            producer_pieces, layout, row_dim, stick_dim, iter_sizes, src_base
        )
        out_pi = pieces_to_pieceinfo(
            consumer_pieces, layout, row_dim, stick_dim, iter_sizes, dst_base
        )
        in_ld = _labeled_ds(
            "dataIN", layout, stick_dim, iter_sizes, stick_size, lx_size, in_pi
        )
        out_ld = _labeled_ds(
            "dataOUT", layout, stick_dim, iter_sizes, stick_size, lx_size, out_pi
        )
        datadscs.append(
            _datadsc(
                f"{b}_STCDPOpLx_dataop",
                _stcdp_op(),
                dim_pool,
                in_ld,
                out_ld,
                num_cores,
            )
        )
    opfuncs = ["STCDPOpLx"] * len(datadscs)
    return datadscs, opfuncs, mixed_schedule(len(datadscs), num_cores)


def allocate_lx_bases(
    num_regions: int,
    slice_bytes: int,
    capacity: int = LX_CAPACITY_BYTES,
    region0: int = 0,
) -> list[int]:
    """Non-overlapping stick-aligned LX bases.

    Packs regions back-to-back; raises ValueError (fail-closed) if the footprint
    exceeds the per-core LX capacity.
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


# --- Bundle-splice: fold the bridge into the consumer SDSC. Pure dict surgery. -


def _dl_op(sdsc_json: dict) -> dict:
    """Return the single DL op dict of an SDSC body's first dsc."""
    body = sdsc_json[next(iter(sdsc_json))]
    dsc = body["dscs_"][0]
    return dsc[next(iter(dsc))]


def _core_state_init_entry(lx_base: int) -> dict:
    """Per-core LX coreStateInit_ entry.

    ``ebrInit_:-1`` is the multicast sentinel; the per-band/per-piece dest column
    is derived by the DCG packer (see the EBR carrier note in P1/P2).
    """
    return {
        "ebrInit_": -1,
        "gtr_": {
            "type": "multicast",
            "id": 18446744073709551615,
            "count": 0,
            "sharers": 0,
            "groupInfo_": {},
        },
        "condGtr_": [],
        "lbrInit_": [lx_base],
        "gapPerDim_": {},
        "lxSizeWithGaps_": DL_LX_SENTINEL,
        "lbrInitForwardGap_": 0,
    }


def apply_lx_flip(sdsc_json: dict, flip: LxFlip) -> None:
    """Flip the DL labeledDs at ``flip.ldsidx`` to LX-resident @ ``flip.lx_base``.

    Rewrites the labeledDs (memOrg_ -> lx, HBM addr/size cleared, lx size
    sentinel, per-core coreStateInit_) and its scheduleTree allocate node. In
    place. Clears the HBM-layout inter-core gap (``backGapCore_``): an LX-resident
    tile is per-core contiguous, so the HBM-style gap (keyed by -1) is meaningless
    in LX and makes dxp codegen fail ("AllocNode has gap in Dim, but coreId not
    avail", dsc2.cpp:3867).
    """
    dl = _dl_op(sdsc_json)
    lds = next(e for e in dl["labeledDs_"] if e["ldsIdx_"] == flip.ldsidx)
    alloc_node = f"allocate-{lds['dsName_']}_lx"
    num_cores = dl["numCoresUsed_"]
    node = next(
        n
        for n in dl["scheduleTree_"]
        if n.get("nodeType_") == "allocate" and n.get("ldsIdx_") == flip.ldsidx
    )
    node["name_"] = alloc_node
    node["component_"] = "lx"
    node["startAddressCoreCorelet_"]["data_"] = {
        f"[{c}, 0, 0]": str(flip.lx_base) for c in range(num_cores)
    }
    node["backGapCore_"] = {}
    if "gapStickSpread_" in node:
        node["gapStickSpread_"] = {}
    lds["memOrg_"] = {"lx": {"isPresent": 1, "allocateNode_": alloc_node}}
    lds["hbmStartAddress_"] = -1
    lds["hbmSize_"] = 0
    lds["lxSize_"] = DL_LX_SENTINEL
    lds["lxBufferSize_"] = DL_LX_SENTINEL
    lds["coreStateInit_"] = [
        _core_state_init_entry(flip.lx_base) for _ in range(num_cores)
    ]


def splice_reshard(
    producer_sdsc: dict,
    consumer_sdsc: dict,
    producer_out_idx: int,
    consumer_in_idx: int,
    producer_base: int,
    consumer_base: int,
    datadscs: list[dict],
    opfuncs: list[str],
    schedule: dict,
) -> None:
    """Fold the reshard bridge into the consumer SDSC (mixed DL + data-op).

    REJECTED by harvest dxp (``SdscTree.cpp:152`` "Datadsc not allowed, use
    dldsc" -- ``SdscNode::importSdsc`` asserts every imported SDSC has an empty
    ``dataOpdscs_``). Superseded by :func:`splice_reshard_standalone`. Kept for
    cross-check / the non-bundle-import path. In place.
    """
    apply_lx_flip(
        producer_sdsc, LxFlip(producer_out_idx, producer_base, "producer-output")
    )
    apply_lx_flip(
        consumer_sdsc, LxFlip(consumer_in_idx, consumer_base, "consumer-input")
    )
    body = consumer_sdsc[next(iter(consumer_sdsc))]
    body["coreIdToDscSchedule"] = schedule
    body["datadscs_"] = datadscs
    body["opFuncsUsed_"] = opfuncs
    _dl_op(consumer_sdsc)["numCoreletsUsed_DSC2_"] = 1


# --- Option (b): standalone pure-data-op SDSC (its own sdsc_execute step) -------
# Avoids the mixed-fold assert (SdscTree.cpp:152) by emitting the STCDP as a
# self-contained SDSC: empty dscs_, populated datadscs_, data-op-only schedule.
# dxp's runCodegen has a pure-data-op branch (dxp.cpp:255: dscs_==0 &&
# dataOpdscs_>0 -> dcg.runDcg) that handles exactly this shape.


def dataop_schedule(num_dataops: int, num_cores: int) -> dict:
    """coreIdToDscSchedule rows for a PURE data-op SDSC (no DL row).

    Each row is ``[datadsc_idx, dldsc_idx, before_sync, after_sync]``. For a
    standalone STCDP there is only the data-op (``dldsc_idx = -1``); the first op
    syncs neither before nor after.
    """
    rows = [[k, -1, 1 if k > 0 else 0, 0] for k in range(num_dataops)]
    return {str(c): [list(r) for r in rows] for c in range(num_cores)}


def build_standalone_dataop_sdsc(
    name: str,
    datadscs: list[dict],
    opfuncs: list[str],
    num_cores: int,
) -> dict:
    """Wrap a STCDP datadsc into a STANDALONE pure-data-op SDSC dict.

    The structure dxp's pure-data-op codegen branch (dxp.cpp:255) expects:
      - ``dscs_`` PRESENT but EMPTY (``importJsonStr`` early-returns false if the
        key is absent; the pure-data-op branch keys off ``dscs_.size()==0``);
      - ``datadscs_`` -- the STCDP datadsc list (populates ``dataOpdscs_``);
      - ``coreIdToDscSchedule`` -- data-op-only rows (no DL row);
      - ``opFuncsUsed_`` -- ``["STCDPOpLx"]``;
      - ``numCoresUsed_`` -- the active core count.
    """
    return {
        name: {
            "numCoresUsed_": num_cores,
            "dscs_": [],
            "datadscs_": datadscs,
            "coreIdToDscSchedule": dataop_schedule(len(datadscs), num_cores),
            "opFuncsUsed_": list(opfuncs),
        }
    }


def splice_reshard_standalone(
    producer_sdsc: dict,
    consumer_sdsc: dict,
    producer_out_idx: int,
    consumer_in_idx: int,
    producer_base: int,
    consumer_base: int,
    datadscs: list[dict],
    opfuncs: list[str],
    num_cores: int,
    sdsc_name: str = "1b_STCDP_reshard",
) -> dict:
    """Option (b): flip producer-out/consumer-in to LX, return a standalone SDSC.

    Leaves both DL SDSCs as pure-DL-op (only their edge labeledDs flipped to
    LX-resident at bases A/B) and returns a SEPARATE pure-data-op SDSC. The caller
    writes that SDSC as its own ``sdsc_execute`` step between the producer and
    consumer in ``bundle.mlir``. Returns the standalone SDSC dict; in place for
    the two DL SDSCs.
    """
    apply_lx_flip(
        producer_sdsc, LxFlip(producer_out_idx, producer_base, "producer-output")
    )
    apply_lx_flip(
        consumer_sdsc, LxFlip(consumer_in_idx, consumer_base, "consumer-input")
    )
    return build_standalone_dataop_sdsc(sdsc_name, datadscs, opfuncs, num_cores)
