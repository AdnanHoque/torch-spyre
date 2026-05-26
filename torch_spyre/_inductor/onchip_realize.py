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
import copy

from .codegen.onchip_bridge import (
    LX_CAPACITY_BYTES,
    STICK_BYTES,
    STREAM_TILE_BYTES,
    allocate_flash_attention_pipeline_bases,
    allocate_lx_bases,
    allocate_stream_bases,
    build_flash_attention_pipeline_bridge,
    build_flash_attention_pipeline_mixed_sdsc,
    flash_pipeline_overlap_prefix_schedule,
    mixed_schedule,
    build_roundtrip_bridge,
    build_same_layout_bridge,
    build_streamed_bridge,
    num_stream_tiles,
    per_core_slice_bytes,
    per_core_same_stick_slice_bytes,
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
# Device-proven 512x512 add and seq64 attention splices reserve at least 256 KiB
# per LX bridge region. Tighter packing can overlap DL-op private LX scratch even
# when the logical tensor slice is smaller.
MIN_BRIDGE_REGION_BYTES = 256 << 10
# Stage022 device sweep: score-scale PT->SFP handoff is value-correct through
# 128-wide score blocks, but 256-wide score blocks corrupt values. Keep larger
# blocks fail-closed to the HBM score-scale path while retaining later SFP
# pointwise handoffs.
FLASH_SCORE_SCALE_MAX_STICK_ELEMS = 128
INPUT_FETCH_NEIGHBOR_DISALLOWED_PINS = {"hbm", None}
INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX = 0
PINNED_COMPONENT_ORDER = (
    "hbm",
    "ring",
    "sfpring",
    "lx",
    "pt",
    "ptxrf",
    "ptarf",
    "sfplrf",
    "pelrf",
    "l0",
    "ptirf",
)


def _reserve_bridge_region_bytes(slice_bytes: int) -> int:
    return max(slice_bytes, MIN_BRIDGE_REGION_BYTES)


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
    slice_bytes = _reserve_bridge_region_bytes(
        per_core_slice_bytes(iter_sizes, split_dim, stick_size, num_cores)
    )
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


def realize_same_layout_handoff(
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
    """Build a same-layout STCDP handoff when split and stick can differ."""
    if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
        return None
    slice_bytes = _reserve_bridge_region_bytes(
        per_core_same_stick_slice_bytes(
            iter_sizes,
            split_dim,
            stick_dim,
            stick_size,
            num_cores,
        )
    )
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
    slice_bytes = _reserve_bridge_region_bytes(
        per_core_slice_bytes(iter_sizes, split_dim, stick_size, num_cores)
    )
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


def _body(sdsc_json: dict) -> dict:
    return sdsc_json[next(iter(sdsc_json))]


def _dl_op(sdsc_json: dict) -> dict:
    """Return the single DL op dict of an SDSC body's first dsc."""
    dsc = _body(sdsc_json)["dscs_"][0]
    return dsc[next(iter(dsc))]


def _op_name(sdsc_json: dict) -> str:
    dsc = _body(sdsc_json)["dscs_"][0]
    return next(iter(dsc))


def _symbol_dim(dim: str) -> str:
    return dim if dim.endswith("_") else f"{dim}_"


def _lds_by_idx(dl: dict, lds_idx: int) -> dict | None:
    for lds in dl.get("labeledDs_", []):
        if lds.get("ldsIdx_") == lds_idx:
            return lds
    return None


def _primary_ds_info(dl: dict, lds_idx: int) -> dict:
    lds = _lds_by_idx(dl, lds_idx)
    if lds is None:
        return {}
    role = lds.get("dsType_")
    return dl.get("primaryDsInfo_", {}).get(role, {})


def _stick_dim_for_lds(dl: dict, lds_idx: int) -> str | None:
    sticks = _primary_ds_info(dl, lds_idx).get("stickDimOrder_", [])
    if len(sticks) != 1:
        return None
    return _symbol_dim(sticks[0])


def _layout_for_lds(dl: dict, lds_idx: int) -> list[str] | None:
    layout = _primary_ds_info(dl, lds_idx).get("layoutDimOrder_", [])
    if not layout:
        return None
    return [_symbol_dim(d) for d in layout]


def _single_split_dim(shard: dict[str, int]) -> str | None:
    split = [d for d, v in shard.items() if v > 1]
    if len(split) != 1:
        return None
    return _symbol_dim(split[0])


def _iter_sizes_for_layout(dl: dict, layout: list[str]) -> dict[str, int] | None:
    sizes = dl.get("N_", {})
    out: dict[str, int] = {}
    for dim in layout:
        if dim not in sizes:
            return None
        out[dim] = int(sizes[dim])
    return out


def _dim_size(dl: dict, dim: str) -> int | None:
    value = dl.get("N_", {}).get(dim)
    return int(value) if value is not None else None


def _split_factor(shard: dict[str, int], split_dim: str) -> int:
    return int(shard.get(split_dim.removesuffix("_"), 1))


def _same_shard_on_layout(
    producer_shard: dict[str, int],
    consumer_shard: dict[str, int],
    layout: list[str],
) -> bool:
    layout_dims = {dim.removesuffix("_") for dim in layout}
    for dim in layout_dims:
        if int(producer_shard.get(dim, 1)) != int(consumer_shard.get(dim, 1)):
            return False
    for dim, factor in producer_shard.items():
        if dim not in layout_dims and int(factor) != 1:
            return False
    for dim, factor in consumer_shard.items():
        if dim not in layout_dims and int(factor) != 1:
            return False
    return True


def _same_physical_stick_layout(
    prod_dl: dict,
    prod_layout: list[str],
    prod_stick: str,
    cons_dl: dict,
    cons_layout: list[str],
    cons_stick: str,
) -> bool:
    """True when producer/consumer layouts differ only by stick dim naming.

    Matmul producer outputs name the hidden stick axis ``out_`` while a following
    matmul consumer names that same physical axis ``in_``.  Treat that as
    same-stick only when the stick appears in the same layout position, all
    non-stick dims have the same names, and paired extents match.
    """
    if len(prod_layout) != len(cons_layout):
        return False
    for p_dim, c_dim in zip(prod_layout, cons_layout):
        if p_dim == prod_stick and c_dim == cons_stick:
            pass
        elif p_dim != c_dim:
            return False
        p_size = _dim_size(prod_dl, p_dim)
        c_size = _dim_size(cons_dl, c_dim)
        if p_size is None or c_size is None or p_size != c_size:
            return False
    return True


def _handoff_bytes(iter_sizes: dict[str, int], word_length: int) -> int:
    size = word_length
    for n in iter_sizes.values():
        size *= n
    return size


def _memorg_is_present(mem_org: dict, component: str) -> bool:
    entry = mem_org.get(component) or mem_org.get(component.upper())
    if not isinstance(entry, dict):
        return False
    return bool(entry.get("isPresent", 0))


def _pinned_component(lds: dict) -> str | None:
    """Return Foundation's first present pinned component for a labeledDs."""
    mem_org = lds.get("memOrg_", {})
    if not isinstance(mem_org, dict):
        return None
    for component in PINNED_COMPONENT_ORDER:
        if _memorg_is_present(mem_org, component):
            return component
    return None


def _input_fetch_neighbor_compute_eligible(compute_dsc: dict) -> bool:
    """True when a compute DSC satisfies DXP's InputFetchNeighbor contract."""
    if not isinstance(compute_dsc, dict) or not compute_dsc:
        return False
    dl = next(iter(compute_dsc.values()))
    if not isinstance(dl, dict):
        return False
    labeled_ds = dl.get("labeledDs_", [])
    if not labeled_ds:
        return False
    if not all(
        _pinned_component(lds) not in INPUT_FETCH_NEIGHBOR_DISALLOWED_PINS
        for lds in labeled_ds
    ):
        return False

    compute_ops = dl.get("computeOp_", [])
    if not compute_ops:
        return False
    input_labels = compute_ops[0].get("inputLabeledDs", [])
    if not input_labels:
        return False
    first_input_idx = int(input_labels[0].rsplit("-idx", 1)[1])
    if first_input_idx != INPUT_FETCH_NEIGHBOR_INPUT_LDSIDX:
        return False
    input_lds = next(
        (lds for lds in labeled_ds if lds.get("ldsIdx_") == first_input_idx),
        None,
    )
    if input_lds is None or _pinned_component(input_lds) != "lx":
        return False
    if not _input_fetch_neighbor_ij_order_supported(dl, first_input_idx):
        return False
    return _has_input_fetch_neighbor_transfer(dl, first_input_idx)


def _input_fetch_neighbor_ij_order_supported(dl: dict, lds_idx: int) -> bool:
    """Current Foundation InputFetchNeighbor ordering assumes i/j coordinates."""
    layout = _layout_for_lds(dl, lds_idx)
    if layout is None:
        return False
    return {"i_", "j_"}.issubset(set(layout))


def _has_input_fetch_neighbor_transfer(dl: dict, lds_idx: int) -> bool:
    """DXP later expects a NO_COMPONENT -> LX transfer node for the neighbor."""
    for node in dl.get("scheduleTree_", []):
        if node.get("nodeType_") != "transfer":
            continue
        src = node.get("src_", {})
        if src.get("storage_") != "no_component":
            continue
        if not any(
            (via.get("loc_", {}) or {}).get("storage_") == "lx"
            for via in node.get("dstVias_", [])
        ):
            continue
        if any(
            dst.get("myLdsIdx_") == lds_idx
            for dst in node.get("dstLdsAndLoopOffsets_", [])
        ):
            return True
    return False


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


def apply_lx_flip(
    sdsc_json: dict,
    flip: LxFlip,
    *,
    core_state_init: bool = True,
    num_corelets: int | None = None,
) -> None:
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
    if num_corelets is not None:
        dl["numCoreletsUsed_"] = num_corelets
        dl["numCoreletsUsed_DSC2_"] = num_corelets
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
    if core_state_init:
        lds["coreStateInit_"] = [
            _core_state_init_entry(flip.lx_base) for _ in range(num_cores)
        ]
    else:
        lds.pop("coreStateInit_", None)


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
            if _hbm_base(cons_dl, in_idx) != hbm_addr:
                continue
            # Scratch HBM addresses are reused.  Treat an input as belonging to
            # this producer only when no later producer between start and the
            # consumer wrote the same address.
            latest = None
            for p in range(c - 1, -1, -1):
                prod_dl = _dl_op(sdscs_json[p])
                if any(
                    _hbm_base(prod_dl, out_idx) == hbm_addr
                    for out_idx in _producer_output_indices(prod_dl)
                ):
                    latest = p
                    break
            if latest == start:
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


def detect_attention_score_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
):
    """Find stock SDPA's same-stick QK^T score handoff fanout.

    The score matrix feeds both softmax max and sub.  Once the producer output
    is flipped to LX, both consumers must be fed from LX too; realizing only one
    leg leaves the other consumer reading a stale HBM address.

    Use the proven splice's same-stick roundtrip geometry for each fanout leg:
    split dim first, remaining non-stick dims next, stick dim last.  That shape
    exercises real cross-core L3 traffic and keeps the implementation out of the
    uncertified PT-LX/ReStickify path.
    """
    for p, prod in enumerate(sdscs_json):
        if _op_name(prod) != "batchmatmul":
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue

        consumers = _future_consumers(sdscs_json, p, prod_addr)
        max_edges = [edge for edge in consumers if _op_name(edge[1]) == "max"]
        sub_edges = [edge for edge in consumers if _op_name(edge[1]) == "sub"]
        if len(max_edges) != 1 or len(sub_edges) != 1:
            continue
        if any(_op_name(edge[1]) not in {"max", "sub"} for edge in consumers):
            continue

        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        if prod_stick is None:
            continue
        bridged_edges = [max_edges[0], sub_edges[0]]
        if any(
            _stick_dim_for_lds(_dl_op(cons), in_idx) != prod_stick
            for _c, cons, in_idx in bridged_edges
        ):
            continue

        # Use the proven splice's data-op geometry: split dim first, remaining
        # non-stick dims next, stick dim last.  This is not necessarily the DL
        # primaryDsInfo_ order.
        _sub_c, sub_cons, sub_in_idx = sub_edges[0]
        sub_dl = _dl_op(sub_cons)
        shard = _body(sub_cons).get("numWkSlicesPerDim_", {})
        split_dim = _single_split_dim(shard)
        if split_dim is None:
            continue
        layout = [split_dim]
        for dim in shard:
            sym = _symbol_dim(dim)
            if sym not in (split_dim, prod_stick):
                layout.append(sym)
        layout.append(prod_stick)
        num_cores = int(_body(sub_cons).get("numCoresUsed_", 0))
        if num_cores <= 0:
            continue
        iter_sizes = _iter_sizes_for_layout(sub_dl, layout)
        if iter_sizes is None:
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            continue
        lds = _lds_by_idx(prod_dl, out_idx)
        word_length = int((lds or {}).get("wordLength", 2))
        handoff_bytes = _handoff_bytes(iter_sizes, word_length)
        if handoff_bytes < min_handoff_bytes:
            continue
        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes, split_dim, prod_stick, STICK_SIZE, num_cores, word_length
            )
        )
        try:
            bases = allocate_lx_bases(3, slice_bytes, region0=0)
        except ValueError:
            continue
        return {
            "producer": prod,
            "producer_out_idx": out_idx,
            "consumers": bridged_edges,
            "iter_sizes": iter_sizes,
            "layout": layout,
            "stick_dim": prod_stick,
            "split_dim": split_dim,
            "num_cores": num_cores,
            "word_length": word_length,
            "handoff_bytes": handoff_bytes,
            "slice_bytes": slice_bytes,
            "producer_base": bases[0],
            "scratch_base": bases[1],
            "consumer_base": bases[2],
        }
    return None


def realize_attention_score_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
) -> bool:
    edge = detect_attention_score_handoff(sdscs_json, min_handoff_bytes)
    if edge is None:
        return False

    prod = edge["producer"]
    apply_lx_flip(
        prod,
        LxFlip(edge["producer_out_idx"], edge["producer_base"], "producer-output"),
    )
    for _c, cons, in_idx in edge["consumers"]:
        apply_lx_flip(
            cons,
            LxFlip(in_idx, edge["consumer_base"], "consumer-input"),
        )
        datadscs, opfuncs, sched = build_roundtrip_bridge(
            dim_pool=edge["layout"],
            iter_sizes=edge["iter_sizes"],
            stick_size=STICK_SIZE,
            num_cores=edge["num_cores"],
            lx_size=edge["slice_bytes"],
            producer_base=edge["producer_base"],
            scratch_base=edge["scratch_base"],
            consumer_base=edge["consumer_base"],
            layout=edge["layout"],
            stick_dim=edge["stick_dim"],
            split_dim=edge["split_dim"],
        )
        body = _body(cons)
        body["coreIdToDscSchedule"] = sched
        body["datadscs_"] = datadscs
        body["opFuncsUsed_"] = opfuncs
        _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True


def detect_static_matmul_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
):
    """Find a static same-stick ``batchmatmul -> batchmatmul`` handoff.

    This targets the MoE static routing proxy: ``(perm @ x) @ w`` and
    ``(perm_w @ y) @ w``.  The routed activation is a single HBM-backed producer
    output consumed by one later matmul input.  Producer and consumer may name
    the hidden stick axis differently (``out_`` vs ``in_``), but the physical
    layout must preserve the stick position and split the same token/slot dim.
    """
    for p, prod in enumerate(sdscs_json):
        if _op_name(prod) != "batchmatmul":
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue

        consumers = _future_consumers(sdscs_json, p, prod_addr)
        if len(consumers) != 1:
            continue
        _c, cons, in_idx = consumers[0]
        if _op_name(cons) != "batchmatmul":
            continue
        cons_dl = _dl_op(cons)

        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        if (
            prod_stick is None
            or cons_stick is None
            or prod_layout is None
            or cons_layout is None
        ):
            continue
        if not _same_physical_stick_layout(
            prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
        ):
            continue

        prod_shard = _body(prod).get("numWkSlicesPerDim_", {})
        cons_shard = _body(cons).get("numWkSlicesPerDim_", {})
        prod_split = _single_split_dim(prod_shard)
        cons_split = _single_split_dim(cons_shard)
        if prod_split is None or cons_split is None or prod_split != cons_split:
            continue
        num_cores = int(_body(cons).get("numCoresUsed_", 0))
        split_factor = _split_factor(cons_shard, cons_split)
        if num_cores <= 0 or split_factor != num_cores:
            continue
        if _split_factor(prod_shard, prod_split) != split_factor:
            continue

        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            continue
        if cons_split not in iter_sizes or iter_sizes[cons_split] % num_cores != 0:
            continue
        lds = _lds_by_idx(prod_dl, out_idx)
        word_length = int((lds or {}).get("wordLength", 2))
        handoff_bytes = _handoff_bytes(iter_sizes, word_length)
        if handoff_bytes < min_handoff_bytes:
            continue
        slice_bytes = per_core_same_stick_slice_bytes(
            iter_sizes, cons_split, cons_stick, STICK_SIZE, num_cores, word_length
        )
        try:
            bases = allocate_lx_bases(3, slice_bytes, region0=0)
        except ValueError:
            continue
        return {
            "producer": prod,
            "producer_out_idx": out_idx,
            "consumer": cons,
            "consumer_in_idx": in_idx,
            "iter_sizes": iter_sizes,
            "layout": cons_layout,
            "stick_dim": cons_stick,
            "split_dim": cons_split,
            "num_cores": num_cores,
            "word_length": word_length,
            "handoff_bytes": handoff_bytes,
            "slice_bytes": slice_bytes,
            "producer_base": bases[0],
            "scratch_base": bases[1],
            "consumer_base": bases[2],
        }
    return None


def realize_static_matmul_handoff(
    sdscs_json: list[dict],
    min_handoff_bytes: int = 1 << 20,
) -> bool:
    edge = detect_static_matmul_handoff(sdscs_json, min_handoff_bytes)
    if edge is None:
        return False

    prod = edge["producer"]
    cons = edge["consumer"]
    apply_lx_flip(
        prod,
        LxFlip(edge["producer_out_idx"], edge["producer_base"], "producer-output"),
    )
    apply_lx_flip(
        cons,
        LxFlip(edge["consumer_in_idx"], edge["consumer_base"], "consumer-input"),
    )
    datadscs, opfuncs, sched = build_roundtrip_bridge(
        dim_pool=edge["layout"],
        iter_sizes=edge["iter_sizes"],
        stick_size=STICK_SIZE,
        num_cores=edge["num_cores"],
        lx_size=edge["slice_bytes"],
        producer_base=edge["producer_base"],
        scratch_base=edge["scratch_base"],
        consumer_base=edge["consumer_base"],
        layout=edge["layout"],
        stick_dim=edge["stick_dim"],
        split_dim=edge["split_dim"],
    )
    body = _body(cons)
    body["coreIdToDscSchedule"] = sched
    body["datadscs_"] = datadscs
    body["opFuncsUsed_"] = opfuncs
    _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True


def _exact_tile_bytes_for_tiles(slice_bytes: int, num_tiles: int) -> int | None:
    """Return a stick-aligned tile size that yields exactly ``num_tiles``."""
    if slice_bytes <= 0 or num_tiles <= 0:
        return None
    for tile_bytes in range(STICK_BYTES, slice_bytes + STICK_BYTES, STICK_BYTES):
        if num_stream_tiles(slice_bytes, tile_bytes) == num_tiles:
            return tile_bytes
    return None


def _flash_pipeline_row_dim(layout: list[str], split_dim: str, iter_sizes: dict):
    candidates = [dim for dim in layout if dim != split_dim]
    if not candidates:
        return None
    return max(candidates, key=lambda dim: int(iter_sizes.get(dim, 0)))


def build_flash_attention_pipeline_artifact(
    sdscs_json: list[dict],
    *,
    overlap: bool = False,
    name: str = "mixed_flash_pipeline_artifact",
) -> dict | None:
    """Build a non-executed mixed-SDSC proof from real flash-prefill compute SDSCs.

    The artifact combines the generated batchmatmul compute DSCs from one
    flash-prefill bundle with the Stage009 double-buffered STCDPOpLx schedule.
    It is intentionally sidecar-only: no producer/consumer descriptors are
    flipped and bundle.mlir should not execute it yet.
    """
    tile_sdscs = [sdsc for sdsc in sdscs_json if _op_name(sdsc) == "batchmatmul"]
    if not tile_sdscs:
        return None

    first = tile_sdscs[0]
    first_body = _body(first)
    num_cores = int(first_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None
    first_dl = _dl_op(first)
    out_indices = _producer_output_indices(first_dl)
    if len(out_indices) != 1:
        return None
    out_idx = out_indices[0]
    layout = _layout_for_lds(first_dl, out_idx)
    stick_dim = _stick_dim_for_lds(first_dl, out_idx)
    split_dim = _single_split_dim(first_body.get("numWkSlicesPerDim_", {}))
    if layout is None or stick_dim is None or split_dim is None:
        return None
    iter_sizes = _iter_sizes_for_layout(first_dl, layout)
    if iter_sizes is None:
        return None

    row_dim = _flash_pipeline_row_dim(layout, split_dim, iter_sizes)
    if row_dim is None:
        return None
    num_tiles = len(tile_sdscs)
    slice_bytes = per_core_same_stick_slice_bytes(
        iter_sizes,
        split_dim,
        stick_dim,
        STICK_SIZE,
        num_cores,
    )
    tile_bytes = _exact_tile_bytes_for_tiles(slice_bytes, num_tiles)
    if tile_bytes is None:
        return None
    try:
        bases = allocate_flash_attention_pipeline_bases(
            num_lanes=2,
            tile_bytes=tile_bytes,
            scratch_regions=2,
            region0=PRODUCER_LX_BASE,
        )
    except ValueError:
        return None

    datadscs, opfuncs, schedule = build_flash_attention_pipeline_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_bases=bases["source_bases"],
        dst_lane_bases=bases["lane_bases"],
        layout=layout,
        stick_dim=stick_dim,
        split_dim=split_dim,
        row_dim=row_dim,
        lane_names=["k", "v"],
        tile_bytes=tile_bytes,
        overlap=overlap,
    )
    compute_dscs = [copy.deepcopy(_body(sdsc)["dscs_"][0]) for sdsc in tile_sdscs]
    artifact = build_flash_attention_pipeline_mixed_sdsc(
        name,
        datadscs,
        opfuncs,
        schedule,
        compute_dscs,
        num_cores,
    )
    root = artifact[name]
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in first_body:
            root[key] = copy.deepcopy(first_body[key])
    root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-batchmatmul-tiles",
            "row_dim": row_dim,
            "split_dim": split_dim,
            "stick_dim": stick_dim,
            "layout": layout,
            "iter_sizes": iter_sizes,
            "tile_bytes": tile_bytes,
        }
    )
    return artifact


def build_flash_attention_pipeline_tile_artifacts(
    sdscs_json: list[dict],
    *,
    name_prefix: str = "mixed_flash_pipeline_tile",
    overlap_prefix: bool = False,
) -> list[dict]:
    """Build DXP-compatible one-compute mixed sidecars for flash-prefill tiles."""
    artifacts = []
    tile_sdscs = [sdsc for sdsc in sdscs_json if _op_name(sdsc) == "batchmatmul"]
    for tile_index, sdsc in enumerate(tile_sdscs):
        artifact = None
        if overlap_prefix and tile_index + 1 < len(tile_sdscs):
            artifact = build_flash_attention_pipeline_overlap_prefix_tile_artifact(
                tile_sdscs[tile_index: tile_index + 2],
                tile_index,
                name_prefix=name_prefix,
            )
        if artifact is None:
            artifact = build_flash_attention_pipeline_artifact(
                [sdsc],
                overlap=False,
                name=f"{name_prefix}_{tile_index}",
            )
        if artifact is None:
            continue
        root = artifact[next(iter(artifact))]
        root["flashAttentionPipeline_"]["tile_index"] = tile_index
        root["flashAttentionPipeline_"]["replaces_sdsc"] = next(iter(sdsc))
        root["flashAttentionPipeline_"].setdefault("overlap_prefix", False)
        artifacts.append(artifact)
    return artifacts


def build_flash_attention_pipeline_overlap_prefix_tile_artifact(
    tile_sdscs: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_pipeline_tile",
) -> dict | None:
    """Build a one-compute sidecar that overlaps next-tile prefetch with compute.

    This is the executable prefix of the full overlap schedule.  It keeps the
    current Foundation one-DL-DSC contract by copying only the first compute DSC,
    while staging two K/V prefetch tiles and scheduling the first next-tile
    prefetch row together with compute tile 0.
    """
    if len(tile_sdscs) < 2:
        return None

    first = tile_sdscs[0]
    first_body = _body(first)
    num_cores = int(first_body.get("numCoresUsed_", 0))
    if num_cores <= 0:
        return None
    first_dl = _dl_op(first)
    out_indices = _producer_output_indices(first_dl)
    if len(out_indices) != 1:
        return None
    out_idx = out_indices[0]
    layout = _layout_for_lds(first_dl, out_idx)
    stick_dim = _stick_dim_for_lds(first_dl, out_idx)
    split_dim = _single_split_dim(first_body.get("numWkSlicesPerDim_", {}))
    if layout is None or stick_dim is None or split_dim is None:
        return None
    iter_sizes = _iter_sizes_for_layout(first_dl, layout)
    if iter_sizes is None:
        return None

    second = tile_sdscs[1]
    second_body = _body(second)
    if int(second_body.get("numCoresUsed_", 0)) != num_cores:
        return None
    second_dl = _dl_op(second)
    second_out_indices = _producer_output_indices(second_dl)
    if len(second_out_indices) != 1:
        return None
    second_out_idx = second_out_indices[0]
    if (
        _layout_for_lds(second_dl, second_out_idx) != layout
        or _stick_dim_for_lds(second_dl, second_out_idx) != stick_dim
        or _single_split_dim(second_body.get("numWkSlicesPerDim_", {}))
        != split_dim
        or _iter_sizes_for_layout(second_dl, layout) != iter_sizes
    ):
        return None

    row_dim = _flash_pipeline_row_dim(layout, split_dim, iter_sizes)
    if row_dim is None:
        return None
    slice_bytes = per_core_same_stick_slice_bytes(
        iter_sizes,
        split_dim,
        stick_dim,
        STICK_SIZE,
        num_cores,
    )
    tile_bytes = _exact_tile_bytes_for_tiles(slice_bytes, 2)
    if tile_bytes is None:
        return None
    try:
        bases = allocate_flash_attention_pipeline_bases(
            num_lanes=2,
            tile_bytes=tile_bytes,
            scratch_regions=2,
            region0=PRODUCER_LX_BASE,
        )
    except ValueError:
        return None

    datadscs, opfuncs, _schedule = build_flash_attention_pipeline_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_bases=bases["source_bases"],
        dst_lane_bases=bases["lane_bases"],
        layout=layout,
        stick_dim=stick_dim,
        split_dim=split_dim,
        row_dim=row_dim,
        lane_names=["k", "v"],
        tile_bytes=tile_bytes,
        overlap=False,
    )
    if len(datadscs) < 4:
        return None

    name = f"{name_prefix}_{tile_index}"
    compute_dsc = copy.deepcopy(first_body["dscs_"][0])
    if not _input_fetch_neighbor_compute_eligible(compute_dsc):
        return None

    artifact = build_flash_attention_pipeline_mixed_sdsc(
        name,
        datadscs[:4],
        opfuncs[:4],
        flash_pipeline_overlap_prefix_schedule(num_lanes=2, num_cores=num_cores),
        [compute_dsc],
        num_cores,
    )
    root = artifact[name]
    for key in (
        "sdscFoldProps_",
        "sdscFolds_",
        "coreFoldProp_",
        "coreletFoldProp_",
        "coreIdToDsc_",
        "numWkSlicesPerDim_",
        "coreIdToWkSlice_",
    ):
        if key in first_body:
            root[key] = copy.deepcopy(first_body[key])
    root["flashAttentionPipeline_"].update(
        {
            "source": "generated-flash-prefill-overlap-prefix-tile",
            "row_dim": row_dim,
            "split_dim": split_dim,
            "stick_dim": stick_dim,
            "layout": layout,
            "iter_sizes": iter_sizes,
            "tile_bytes": tile_bytes,
            "tile_index": tile_index,
            "replaces_sdsc": next(iter(first)),
            "prefetch_tile_count": 2,
            "compute_tile_count": 1,
            "overlap_prefix": True,
        }
    )
    return artifact


def _latest_producer_of_hbm(
    sdscs_json: list[dict],
    before_index: int,
    hbm_addr: str,
):
    for p in range(before_index - 1, -1, -1):
        prod = sdscs_json[p]
        prod_dl = _dl_op(prod)
        for out_idx in _producer_output_indices(prod_dl):
            if _hbm_base(prod_dl, out_idx) == hbm_addr:
                return p, prod, out_idx
    return None


def _renumber_datadscs(datadscs: list[dict], start: int) -> list[dict]:
    renamed = []
    for offset, datadsc in enumerate(datadscs):
        old_name, body = next(iter(datadsc.items()))
        suffix = old_name.split("_", 1)[1]
        renamed.append({f"{start + offset}_{suffix}": body})
    return renamed


def build_flash_attention_value_flow_tile_artifact(
    sdscs_json: list[dict],
    tile_index: int,
    *,
    name_prefix: str = "mixed_flash_value_flow_tile",
) -> tuple[dict, str] | None:
    """Mutate one flash tile to consume real producer LX values via STCDPOpLx."""
    batch_seen = -1
    for c, cons in enumerate(sdscs_json):
        if _op_name(cons) != "batchmatmul":
            continue
        batch_seen += 1
        if batch_seen != tile_index:
            continue

        cons_dl = _dl_op(cons)
        cons_body = _body(cons)
        num_cores = int(cons_body.get("numCoresUsed_", 0))
        if num_cores <= 0:
            return None
        edges = []
        for in_idx in _consumer_input_indices(cons_dl):
            addr = _hbm_base(cons_dl, in_idx)
            if addr is None:
                continue
            producer = _latest_producer_of_hbm(sdscs_json, c, addr)
            if producer is None:
                continue
            _p, prod, out_idx = producer
            future = _future_consumers(sdscs_json, _p, addr)
            if len(future) != 1 or future[0][0] != c or future[0][2] != in_idx:
                continue
            prod_dl = _dl_op(prod)
            prod_layout = _layout_for_lds(prod_dl, out_idx)
            prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
            cons_layout = _layout_for_lds(cons_dl, in_idx)
            cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
            split_dim = _single_split_dim(cons_body.get("numWkSlicesPerDim_", {}))
            if (
                prod_layout is None
                or prod_stick is None
                or cons_layout is None
                or cons_stick is None
                or split_dim is None
            ):
                continue
            if not _same_physical_stick_layout(
                prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
            ):
                continue
            iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
            if iter_sizes is None:
                continue
            if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
                continue
            slice_bytes = _reserve_bridge_region_bytes(
                per_core_same_stick_slice_bytes(
                    iter_sizes,
                    split_dim,
                    cons_stick,
                    STICK_SIZE,
                    num_cores,
                )
            )
            edges.append(
                {
                    "producer": prod,
                    "producer_idx": out_idx,
                    "consumer_idx": in_idx,
                    "layout": cons_layout,
                    "stick_dim": cons_stick,
                    "split_dim": split_dim,
                    "iter_sizes": iter_sizes,
                    "slice_bytes": slice_bytes,
                }
            )
        if not edges:
            return None

        max_slice = max(edge["slice_bytes"] for edge in edges)
        try:
            bases = allocate_lx_bases(
                len(edges) * 3,
                max_slice,
                region0=PRODUCER_LX_BASE,
            )
        except ValueError:
            return None

        datadscs = []
        opfuncs = []
        edge_meta = []
        for edge_idx, edge in enumerate(edges):
            producer_base, scratch_base, consumer_base = bases[
                edge_idx * 3: edge_idx * 3 + 3
            ]
            apply_lx_flip(
                edge["producer"],
                LxFlip(edge["producer_idx"], producer_base, "producer-output"),
            )
            apply_lx_flip(
                cons,
                LxFlip(edge["consumer_idx"], consumer_base, "consumer-input"),
            )
            bridge_datadscs, bridge_opfuncs, _sched = build_roundtrip_bridge(
                dim_pool=edge["layout"],
                iter_sizes=edge["iter_sizes"],
                stick_size=STICK_SIZE,
                num_cores=num_cores,
                lx_size=edge["slice_bytes"],
                producer_base=producer_base,
                scratch_base=scratch_base,
                consumer_base=consumer_base,
                layout=edge["layout"],
                stick_dim=edge["stick_dim"],
                split_dim=edge["split_dim"],
            )
            datadscs.extend(_renumber_datadscs(bridge_datadscs, len(datadscs)))
            opfuncs.extend(bridge_opfuncs)
            edge_meta.append(
                {
                    "producer": next(iter(edge["producer"])),
                    "producer_idx": edge["producer_idx"],
                    "consumer_idx": edge["consumer_idx"],
                    "layout": edge["layout"],
                    "stick_dim": edge["stick_dim"],
                    "split_dim": edge["split_dim"],
                    "slice_bytes": edge["slice_bytes"],
                    "producer_base": producer_base,
                    "scratch_base": scratch_base,
                    "consumer_base": consumer_base,
                }
            )

        name = f"{name_prefix}_{tile_index}"
        artifact = build_flash_attention_pipeline_mixed_sdsc(
            name,
            datadscs,
            opfuncs,
            mixed_schedule(len(datadscs), num_cores),
            [copy.deepcopy(_body(cons)["dscs_"][0])],
            num_cores,
        )
        root = artifact[name]
        for key in (
            "sdscFoldProps_",
            "sdscFolds_",
            "coreFoldProp_",
            "coreletFoldProp_",
            "coreIdToDsc_",
            "numWkSlicesPerDim_",
            "coreIdToWkSlice_",
        ):
            if key in cons_body:
                root[key] = copy.deepcopy(cons_body[key])
        root["flashAttentionPipeline_"].update(
            {
                "source": "generated-flash-prefill-real-value-flow",
                "tile_index": tile_index,
                "replaces_sdsc": next(iter(cons)),
                "edges": edge_meta,
            }
        )
        return artifact, next(iter(cons))
    return None


def detect_pointwise_handoff(sdscs_json: list[dict]):
    """Find a fully legal same-layout pointwise LX handoff edge."""
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
        cons_shard = cons[next(iter(cons))].get("numWkSlicesPerDim_")
        if cons_shard != prod_shard:
            continue
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_dl = _dl_op(cons)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        if (
            prod_layout is None
            or prod_stick is None
            or cons_layout is None
            or cons_stick is None
            or not _same_physical_stick_layout(
                prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
            )
        ):
            continue

        split_dim = _single_split_dim(cons_shard)
        if split_dim is None:
            continue
        num_cores = int(_body(cons).get("numCoresUsed_", 0))
        split_factor = _split_factor(cons_shard, split_dim)
        if num_cores <= 0 or split_factor != num_cores:
            continue
        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            continue
        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes,
                split_dim,
                cons_stick,
                STICK_SIZE,
                num_cores,
            )
        )
        if slice_bytes > STREAM_THRESHOLD and cons_stick != split_dim:
            continue
        return {
            "producer": prod,
            "consumer": cons,
            "producer_idx": out_idx,
            "consumer_idx": in_idx,
            "iter_sizes": iter_sizes,
            "layout": cons_layout,
            "stick_dim": cons_stick,
            "split_dim": split_dim,
            "num_cores": num_cores,
            "slice_bytes": slice_bytes,
        }
    return None


def detect_flash_score_scale_handoff(sdscs_json: list[dict]):
    """Find a legal flash score ``batchmatmul -> scalar mul`` LX handoff edge."""
    for p in range(len(sdscs_json)):
        prod = sdscs_json[p]
        if _op_name(prod) != "batchmatmul":
            continue
        prod_dl = _dl_op(prod)
        out_indices = _producer_output_indices(prod_dl)
        if len(out_indices) != 1:
            continue
        out_idx = out_indices[0]
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue
        consumers = _future_consumers(sdscs_json, p, prod_addr)
        if len(consumers) != 1:
            continue
        _c, cons, in_idx = consumers[0]
        if _op_name(cons) != "mul":
            continue
        prod_layout = _layout_for_lds(prod_dl, out_idx)
        prod_stick = _stick_dim_for_lds(prod_dl, out_idx)
        cons_dl = _dl_op(cons)
        cons_layout = _layout_for_lds(cons_dl, in_idx)
        cons_stick = _stick_dim_for_lds(cons_dl, in_idx)
        if (
            prod_layout is None
            or prod_stick is None
            or cons_layout is None
            or cons_stick is None
            or not _same_physical_stick_layout(
                prod_dl, prod_layout, prod_stick, cons_dl, cons_layout, cons_stick
            )
        ):
            continue

        prod_shard = _body(prod).get("numWkSlicesPerDim_", {})
        cons_shard = _body(cons).get("numWkSlicesPerDim_", {})
        if not _same_shard_on_layout(prod_shard, cons_shard, cons_layout):
            continue
        split_dim = _single_split_dim(cons_shard)
        if split_dim is None:
            continue
        num_cores = int(_body(cons).get("numCoresUsed_", 0))
        if num_cores <= 0:
            continue
        if (
            _split_factor(cons_shard, split_dim) != num_cores
            or _split_factor(prod_shard, split_dim) != num_cores
        ):
            continue
        iter_sizes = _iter_sizes_for_layout(cons_dl, cons_layout)
        if iter_sizes is None:
            continue
        if int(iter_sizes.get(cons_stick, 0)) > FLASH_SCORE_SCALE_MAX_STICK_ELEMS:
            continue
        if split_dim not in iter_sizes or iter_sizes[split_dim] % num_cores != 0:
            continue
        slice_bytes = _reserve_bridge_region_bytes(
            per_core_same_stick_slice_bytes(
                iter_sizes,
                split_dim,
                cons_stick,
                STICK_SIZE,
                num_cores,
            )
        )
        if slice_bytes > STREAM_THRESHOLD and cons_stick != split_dim:
            continue
        return {
            "producer": prod,
            "consumer": cons,
            "producer_idx": out_idx,
            "consumer_idx": in_idx,
            "iter_sizes": iter_sizes,
            "layout": cons_layout,
            "stick_dim": cons_stick,
            "split_dim": split_dim,
            "num_cores": num_cores,
            "slice_bytes": slice_bytes,
        }
    return None


def detect_onchip_edge(sdscs_json: list[dict]):
    """Find an eligible same-stick same-shard producer->consumer edge.

    The original proof matched add->add only.  Keep that production-shaped narrow
    contract, but allow the same pointwise shape for attention's Inductor-level
    online-softmax graph.  We require a single future consumer to avoid fanout
    values that still need the HBM materialization.
    """
    edge = detect_pointwise_handoff(sdscs_json)
    if edge is None:
        return None
    return (
        edge["producer"],
        edge["consumer"],
        edge["producer_idx"],
        edge["consumer_idx"],
    )


def realize_pointwise_handoff(sdscs_json: list[dict]) -> bool:
    edge = detect_pointwise_handoff(sdscs_json)
    if edge is None:
        return False
    return _realize_handoff_edge(edge)


def _realize_handoff_edge(
    edge: dict,
    *,
    region0: int = PRODUCER_LX_BASE,
    producer_core_state_init: bool = True,
    producer_num_corelets: int | None = None,
) -> bool:
    if edge["slice_bytes"] <= STREAM_THRESHOLD:
        realization = realize_same_layout_handoff(
            iter_sizes=edge["iter_sizes"],
            layout=edge["layout"],
            stick_dim=edge["stick_dim"],
            split_dim=edge["split_dim"],
            stick_size=STICK_SIZE,
            num_cores=edge["num_cores"],
            producer_ldsidx=edge["producer_idx"],
            consumer_ldsidx=edge["consumer_idx"],
            region0=region0,
        )
    else:
        realization = realize_streamed_handoff(
            iter_sizes=edge["iter_sizes"],
            layout=edge["layout"],
            stick_dim=edge["stick_dim"],
            split_dim=edge["split_dim"],
            stick_size=STICK_SIZE,
            num_cores=edge["num_cores"],
            producer_ldsidx=edge["producer_idx"],
            consumer_ldsidx=edge["consumer_idx"],
            region0=region0,
        )
    if realization is None:
        return False
    prod = edge["producer"]
    cons = edge["consumer"]
    apply_lx_flip(
        prod,
        realization.producer_flip,
        core_state_init=producer_core_state_init,
        num_corelets=producer_num_corelets,
    )
    apply_lx_flip(cons, realization.consumer_flip)
    body = _body(cons)
    body["coreIdToDscSchedule"] = realization.schedule
    body["datadscs_"] = realization.datadscs
    body["opFuncsUsed_"] = realization.opfuncs
    _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True


def realize_flash_score_scale_handoff(sdscs_json: list[dict]) -> bool:
    edge = detect_flash_score_scale_handoff(sdscs_json)
    if edge is None:
        return False
    # PT producers use the same allocator-shaped endpoint contract as the
    # value-correct first-principles PT-LX bridge: base 0, explicit corelet
    # count, and no producer-side coreStateInit_ injection.
    return _realize_handoff_edge(
        edge,
        region0=0,
        producer_core_state_init=False,
        producer_num_corelets=1,
    )


def realize_flash_attention_pointwise_handoffs(
    sdscs_json: list[dict],
    *,
    score_scale_handoff: bool = False,
) -> int:
    """Realize every legal same-layout flash handoff in one flash bundle."""
    count = 0
    # One realization mutates the graph by turning an HBM producer output into an
    # LX endpoint, so at most one new edge can disappear per SDSC iteration.
    for _ in range(len(sdscs_json)):
        if score_scale_handoff and realize_flash_score_scale_handoff(sdscs_json):
            count += 1
            continue
        if not realize_pointwise_handoff(sdscs_json):
            break
        count += 1
    return count


def realize_onchip_handoff(
    sdscs_json: list[dict],
    *,
    attention_score_handoff: bool = False,
    static_matmul_handoff: bool = False,
    min_handoff_bytes: int = 1 << 20,
) -> bool:
    """Realize the eligible same-core handoff edge in place; fail-closed.

    When requested, stock SDPA score fanout is handled first because its
    producer feeds both max and sub.  Otherwise, or if that fails closed, detect
    the original pointwise edge, build a same-layout bridge with the same
    size-aware LX allocation as the standalone realization helpers, flip
    producer-output + consumer-input to LX, and fold the bridge into the
    consumer (mixed DL+data-op SuperDSC).
    """
    if attention_score_handoff and realize_attention_score_handoff(
        sdscs_json, min_handoff_bytes
    ):
        return True
    if static_matmul_handoff and realize_static_matmul_handoff(
        sdscs_json, min_handoff_bytes
    ):
        return True

    return realize_pointwise_handoff(sdscs_json)
