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

"""Pure realization logic for the on-chip same-layout handoff (Tier 1).

This module is torch-free so it unit-tests in a bare worktree (mirrors the
restickify_cost.py split). Given an eligible same-stick edge -- the consumer's
sharding, the shared stick dim, iteration sizes, num_cores -- it produces a
structured ``OnChipRealization``: LX bases (non-overlapping, in-capacity), the
synthesized datadscs_/schedule/opFuncsUsed_ (via codegen.onchip_bridge), and
the producer/consumer LX-flip descriptors. Over-capacity or layout-changing
edges return None (fail-closed). The realize FIRST CUT targets the simplest
same-core same-shard case: a single STCDPOpLx, 2 LX regions, no ring.
"""

from __future__ import annotations

import dataclasses

from .codegen.onchip_bridge import (
    LX_CAPACITY_BYTES,
    STREAM_TILE_BYTES,
    allocate_lx_bases,
    allocate_stream_bases,
    build_same_layout_bridge,
    build_streamed_bridge,
    num_stream_tiles,
    per_core_slice_bytes,
)

# Per-core LX byte span declared inside each data-op labeledDs (2 MB, matches
# splice_2048_stcdp DATAOP_LX_SIZE). DL-DSC LX size sentinel.
DATAOP_LX_SIZE = 2 << 20
DL_LX_SENTINEL = 2147483647

# Device-proven LX address contract for the 2048 same-core add->add bridge
# (splice_2048_stcdp): producer output @16384, consumer input @8192.
PRODUCER_LX_BASE = 16384
CONSUMER_LX_BASE = 8192
STICK_SIZE = 64

# Tier select: stream once the single 2-region move would consume more than half
# the 2 MB LX, leaving no headroom for the DL op. Below it the single move fits.
STREAM_THRESHOLD = LX_CAPACITY_BYTES // 2


@dataclasses.dataclass(frozen=True)
class LxFlip:
    """Flip one DL labeledDs to LX-resident: ``base`` per core, sentinel size."""

    ldsidx: int
    lx_base: int
    role: str  # "producer-output" | "consumer-input"


@dataclasses.dataclass(frozen=True)
class OnChipRealization:
    """A realized same-layout handoff: LX bases + synthesized mixed-DSC parts."""

    producer_base: int
    consumer_base: int
    slice_bytes: int
    producer_flip: LxFlip
    consumer_flip: LxFlip
    datadscs: list
    opfuncs: list[str]
    schedule: dict
    realizable: bool = True


def realize_same_core_handoff(
    iter_sizes: dict[str, int],
    layout: list[str],
    stick_dim: str,
    split_dim: str,
    stick_size: int,
    num_cores: int,
    producer_ldsidx: int,
    consumer_ldsidx: int,
    capacity: int = LX_CAPACITY_BYTES,
    region0: int = 0,
) -> OnChipRealization | None:
    """Build a 2-region same-core single-STCDP realization, or None (fail-closed).

    Both producer-out and consumer-in split the SAME dim (same-shard), so a
    single STCDPOpLx copies each core's slice LX->LX with no ring. Bases are
    derived per-size and packed non-overlapping; if the 2 regions exceed
    per-core LX, return None. The fold is the simplest 18/40 case.
    """
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None
    slice_bytes = per_core_slice_bytes(iter_sizes, split_dim, stick_size, num_cores)
    try:
        bases = allocate_lx_bases(2, slice_bytes, capacity=capacity, region0=region0)
    except ValueError:
        return None
    producer_base, consumer_base = bases
    datadscs, opfuncs, sched = build_same_layout_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=stick_size,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_base=producer_base,
        dst_base=consumer_base,
        layout=layout,
        stick_dim=stick_dim,
        src_split_dim=split_dim,
        dst_split_dim=split_dim,
    )
    return OnChipRealization(
        producer_base=producer_base,
        consumer_base=consumer_base,
        slice_bytes=slice_bytes,
        producer_flip=LxFlip(producer_ldsidx, producer_base, "producer-output"),
        consumer_flip=LxFlip(consumer_ldsidx, consumer_base, "consumer-input"),
        datadscs=datadscs,
        opfuncs=opfuncs,
        schedule=sched,
    )


@dataclasses.dataclass(frozen=True)
class StreamedRealization:
    """A realized streamed handoff: 2 fixed tile bases + K-tile mixed-DSC parts."""

    producer_base: int
    consumer_base: int
    slice_bytes: int
    num_tiles: int
    tile_bytes: int
    producer_flip: LxFlip
    consumer_flip: LxFlip
    datadscs: list
    opfuncs: list[str]
    schedule: dict
    realizable: bool = True


def realize_streamed_handoff(
    iter_sizes: dict[str, int],
    layout: list[str],
    stick_dim: str,
    split_dim: str,
    stick_size: int,
    num_cores: int,
    producer_ldsidx: int,
    consumer_ldsidx: int,
    tile_bytes: int = STREAM_TILE_BYTES,
    capacity: int = LX_CAPACITY_BYTES,
    region0: int = 0,
) -> StreamedRealization | None:
    """Stream a >LX/2 slice through 2 fixed tile buffers, or None (fail-closed).

    The single move stays the same-shard same-stick cross-core copy; streaming
    just tiles it along the non-split row dim so only 2*tile_bytes of LX live at
    once (vs 2*slice). K = ceil(slice/tile). Returns None if even the 2 fixed
    tiles don't fit -- the DL op gets the rest of the 2 MB. Single-buffer reuse:
    device-validate (fallback = double-buffer).
    """
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None
    row_dims = [d for d in layout if d != split_dim]
    if len(row_dims) != 1:
        return None
    row_dim = row_dims[0]
    slice_bytes = per_core_slice_bytes(iter_sizes, split_dim, stick_size, num_cores)
    try:
        bases = allocate_stream_bases(
            tile_bytes,
            capacity=capacity,
            region0=region0,
        )
    except ValueError:
        return None
    producer_base, consumer_base = bases
    k_tiles = num_stream_tiles(slice_bytes, tile_bytes)
    datadscs, opfuncs, sched = build_streamed_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=stick_size,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_base=producer_base,
        dst_base=consumer_base,
        layout=layout,
        stick_dim=stick_dim,
        src_split_dim=split_dim,
        dst_split_dim=split_dim,
        row_dim=row_dim,
        slice_bytes=slice_bytes,
        tile_bytes=tile_bytes,
    )
    return StreamedRealization(
        producer_base=producer_base,
        consumer_base=consumer_base,
        slice_bytes=slice_bytes,
        num_tiles=k_tiles,
        tile_bytes=tile_bytes,
        producer_flip=LxFlip(producer_ldsidx, producer_base, "producer-output"),
        consumer_flip=LxFlip(consumer_ldsidx, consumer_base, "consumer-input"),
        datadscs=datadscs,
        opfuncs=opfuncs,
        schedule=sched,
    )


def is_same_shard(
    producer_splits: dict[str, int],
    consumer_splits: dict[str, int],
    symbol_map: dict[str, str],
) -> bool:
    """True when both sides split the same dim the same way (no ring needed)."""
    import math

    if math.prod(producer_splits.values()) != math.prod(consumer_splits.values()):
        return False
    for cons_sym, prod_sym in symbol_map.items():
        if consumer_splits.get(cons_sym, 1) != producer_splits.get(prod_sym, 1):
            return False
    return True


# --- In-memory SDSC transform: emit the mixed bundle DURING compilation. The
# functions below port splice_2048_stcdp into the codegen path; they are pure
# dict surgery (torch-free) so generate_bundle and the offline gate share them.


_POINTWISE_HANDOFF_OPS = {
    "add",
    "exp",
    "identity",
    "maximum",
    "mul",
    "realdiv",
    "sub",
}


def _dl_op(sdsc_json: dict) -> dict:
    """Return the single DL op dict of an SDSC body's first dsc."""
    body = sdsc_json[next(iter(sdsc_json))]
    dsc = body["dscs_"][0]
    return dsc[next(iter(dsc))]


def _op_name(sdsc_json: dict) -> str:
    dsc = sdsc_json[next(iter(sdsc_json))]["dscs_"][0]
    return next(iter(dsc))


def _core_state_init_entry(lx_base: int) -> dict:
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

    Mirrors splice_2048_stcdp._flip_tensor_to_lx exactly: rewrites the labeledDs
    (memOrg_ -> lx, hbm addr/size cleared, lx size sentinel, per-core
    coreStateInit_) and its scheduleTree allocate node (name/component/per-core
    LX base). In place.
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
    lds["memOrg_"] = {"lx": {"isPresent": 1, "allocateNode_": alloc_node}}
    lds["hbmStartAddress_"] = -1
    lds["hbmSize_"] = 0
    lds["lxSize_"] = DL_LX_SENTINEL
    lds["lxBufferSize_"] = DL_LX_SENTINEL
    lds["coreStateInit_"] = [
        _core_state_init_entry(flip.lx_base) for _ in range(num_cores)
    ]


def _hbm_base(dl: dict, lds_idx: int) -> str | None:
    """Per-core[0] HBM base for the labeledDs allocate node, else None."""
    for node in dl["scheduleTree_"]:
        if node.get("nodeType_") == "allocate" and node.get("ldsIdx_") == lds_idx:
            if node.get("component_") != "hbm":
                return None
            return next(iter(node["startAddressCoreCorelet_"]["data_"].values()), None)
    return None


def _label_indices(labels: list[str]) -> list[int]:
    return [int(lbl.rsplit("-idx", 1)[1]) for lbl in labels]


def _producer_output_indices(dl: dict) -> list[int]:
    return _label_indices(dl["computeOp_"][0]["outputLabeledDs"])


def _consumer_input_indices(dl: dict) -> list[int]:
    return _label_indices(dl["computeOp_"][0]["inputLabeledDs"])


def _future_consumers(sdscs_json: list[dict], start: int, hbm_addr: str):
    consumers = []
    for c in range(start + 1, len(sdscs_json)):
        cons = sdscs_json[c]
        cons_dl = _dl_op(cons)
        for in_idx in _consumer_input_indices(cons_dl):
            if _hbm_base(cons_dl, in_idx) == hbm_addr:
                consumers.append((c, cons, in_idx))
    return consumers


def _iter_sizes_from_dl(dl: dict, shard: dict[str, int]) -> dict[str, int] | None:
    sizes = dl.get("N_", {})
    iter_sizes: dict[str, int] = {}
    for dim in shard:
        key = f"{dim}_"
        if key not in sizes:
            return None
        iter_sizes[key] = int(sizes[key])
    return iter_sizes


def detect_onchip_edge(sdscs_json: list[dict]):
    """Find an eligible same-stick same-shard producer->consumer edge.

    The original proof matched add->add only.  Keep that production-shaped narrow
    contract, but allow the same pointwise shape for attention's Inductor-level
    online-softmax graph.  We require a single future consumer to avoid fanout
    values that still need the HBM materialization.
    """
    for p in range(len(sdscs_json)):
        prod = sdscs_json[p]
        if _op_name(prod) not in _POINTWISE_HANDOFF_OPS:
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue
        prod_shard = prod[next(iter(prod))].get("numWkSlicesPerDim_")
        consumers = _future_consumers(sdscs_json, p, prod_addr)
        if len(consumers) != 1:
            continue
        _c, cons, in_idx = consumers[0]
        if _op_name(cons) not in _POINTWISE_HANDOFF_OPS:
            continue
        if cons[next(iter(cons))].get("numWkSlicesPerDim_") != prod_shard:
            continue
        return prod, cons, out_idx, in_idx
    return None


def realize_onchip_handoff(sdscs_json: list[dict]) -> bool:
    """Realize the eligible same-core handoff edge in place; fail-closed.

    Detects the add->add edge, builds a same-layout bridge with the same
    size-aware LX allocation as the standalone realization helpers, flips
    producer-output + consumer-input to LX, and folds the bridge into the
    consumer (mixed DL+data-op SuperDSC).
    """
    edge = detect_onchip_edge(sdscs_json)
    if edge is None:
        return False
    prod, cons, out_idx, in_idx = edge
    shard = cons[next(iter(cons))]["numWkSlicesPerDim_"]
    split = [d for d, v in shard.items() if v > 1]
    if len(split) != 1:
        return False
    num_cores = shard[split[0]]
    layout = [f"{d}_" for d in shard]
    iter_sizes = _iter_sizes_from_dl(_dl_op(cons), shard)
    if iter_sizes is None:
        return False
    split_dim = f"{split[0]}_"
    slice_bytes = per_core_slice_bytes(iter_sizes, split_dim, STICK_SIZE, num_cores)
    # Tier branch: single 2-region move when 2*slice fits (slice <= half cap);
    # else stream through 2 fixed tile buffers; else fail-closed. The add->add
    # 2048 case stays the single move (slice == 256 KB << half cap), byte-identical.
    if slice_bytes <= STREAM_THRESHOLD:
        realization = realize_same_core_handoff(
            iter_sizes=iter_sizes,
            layout=layout,
            stick_dim=split_dim,
            split_dim=split_dim,
            stick_size=STICK_SIZE,
            num_cores=num_cores,
            producer_ldsidx=out_idx,
            consumer_ldsidx=in_idx,
            region0=PRODUCER_LX_BASE,
        )
    else:
        realization = realize_streamed_handoff(
            iter_sizes=iter_sizes,
            layout=layout,
            stick_dim=split_dim,
            split_dim=split_dim,
            stick_size=STICK_SIZE,
            num_cores=num_cores,
            producer_ldsidx=out_idx,
            consumer_ldsidx=in_idx,
            region0=PRODUCER_LX_BASE,
        )
    if realization is None:
        return False
    apply_lx_flip(prod, realization.producer_flip)
    apply_lx_flip(cons, realization.consumer_flip)
    body = cons[next(iter(cons))]
    body["coreIdToDscSchedule"] = realization.schedule
    body["datadscs_"] = realization.datadscs
    body["opFuncsUsed_"] = realization.opfuncs
    _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True
