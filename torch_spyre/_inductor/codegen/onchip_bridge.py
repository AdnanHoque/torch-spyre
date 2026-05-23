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


def _piece_info(
    layout_order: Sequence[str],
    split_dim: str,
    iter_sizes: Mapping[str, int],
    chunk: int,
    base: int,
    num_cores: int,
) -> list[dict]:
    """Per-core PieceInfo: split_dim is chunked across cores, others are full."""
    pieces = []
    for i in range(num_cores):
        start = {d: (i * chunk if d == split_dim else 0) for d in layout_order}
        size = {d: (chunk if d == split_dim else iter_sizes[d]) for d in layout_order}
        gap = {d: [[size[d], 0]] for d in layout_order}
        pieces.append(
            {
                "key_": f"p{i + 1}",
                "dimToStartCordinate": start,
                "dimToSize_": size,
                "validGap_": gap,
                "PlacementInfo": [{"type": "lx", "memId": [i], "startAddr": [base]}],
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
            layout_order, split_dim, iter_sizes, chunk, base, num_cores
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
    def __init__(self, layout, stick_dim, split_dim, base):
        self.layout = layout
        self.stick_dim = stick_dim
        self.split_dim = split_dim
        self.base = base


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
                        iter_sizes, stick_size, src.base, num_cores, lx_size)
    out_ld = _labeled_ds("dataOUT", dst.layout, dst.stick_dim, dst.split_dim,
                         iter_sizes, stick_size, dst.base, num_cores, lx_size)
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
