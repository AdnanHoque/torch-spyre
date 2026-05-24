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
    allocate_lx_bases,
    build_same_layout_bridge,
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
        bases = allocate_lx_bases(2, slice_bytes, capacity=capacity)
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


def _dl_op(sdsc_json: dict) -> dict:
    """Return the single DL op dict of an SDSC body's first dsc."""
    body = sdsc_json[next(iter(sdsc_json))]
    dsc = body["dscs_"][0]
    return dsc[next(iter(dsc))]


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


def detect_onchip_edge(sdscs_json: list[dict]):
    """Find the eligible same-stick same-shard add->add producer->consumer edge.

    Mirrors splice_2048_stcdp: a producer add SDSC whose single output labeledDs
    HBM base matches a later add SDSC's first input HBM base, both sharding the
    same way. Returns (producer, consumer, prod_out_idx, cons_in_idx) or None.
    """
    for p in range(len(sdscs_json)):
        prod = sdscs_json[p]
        if not next(iter(prod)).endswith("_add"):
            continue
        prod_dl = _dl_op(prod)
        out_labels = prod_dl["computeOp_"][0]["outputLabeledDs"]
        if len(out_labels) != 1:
            continue
        out_idx = int(out_labels[0].rsplit("-idx", 1)[1])
        prod_addr = _hbm_base(prod_dl, out_idx)
        if prod_addr is None:
            continue
        prod_shard = prod[next(iter(prod))].get("numWkSlicesPerDim_")
        for c in range(p + 1, len(sdscs_json)):
            cons = sdscs_json[c]
            if not next(iter(cons)).endswith("_add"):
                continue
            cons_dl = _dl_op(cons)
            in_lbl = cons_dl["computeOp_"][0]["inputLabeledDs"][0]
            in_idx = int(in_lbl.rsplit("-idx", 1)[1])
            if _hbm_base(cons_dl, in_idx) != prod_addr:
                continue
            if cons[next(iter(cons))].get("numWkSlicesPerDim_") != prod_shard:
                continue
            return prod, cons, out_idx, in_idx
    return None


def realize_onchip_handoff(sdscs_json: list[dict]) -> bool:
    """Realize the eligible same-core handoff edge in place; fail-closed.

    Detects the add->add edge, builds a single STCDP same-layout bridge at the
    device-proven LX bases, flips producer-output + consumer-input to LX, and
    folds the bridge into the consumer (mixed DL+data-op SuperDSC). Allocator
    validates the 2 regions fit per-core LX; otherwise returns False. Mirrors
    splice_2048_stcdp.
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
    iter_sizes = {f"{d}_": num_cores * STICK_SIZE for d in shard}
    split_dim = f"{split[0]}_"
    slice_bytes = per_core_slice_bytes(iter_sizes, split_dim, STICK_SIZE, num_cores)
    try:
        allocate_lx_bases(2, slice_bytes)
    except ValueError:
        return False
    datadscs, opfuncs, sched = build_same_layout_bridge(
        dim_pool=layout,
        iter_sizes=iter_sizes,
        stick_size=STICK_SIZE,
        num_cores=num_cores,
        lx_size=DATAOP_LX_SIZE,
        src_base=PRODUCER_LX_BASE,
        dst_base=CONSUMER_LX_BASE,
        layout=layout,
        stick_dim=split_dim,
        src_split_dim=split_dim,
        dst_split_dim=split_dim,
    )
    apply_lx_flip(prod, LxFlip(out_idx, PRODUCER_LX_BASE, "producer-output"))
    apply_lx_flip(cons, LxFlip(in_idx, CONSUMER_LX_BASE, "consumer-input"))
    body = cons[next(iter(cons))]
    body["coreIdToDscSchedule"] = sched
    body["datadscs_"] = datadscs
    body["opFuncsUsed_"] = opfuncs
    _dl_op(cons)["numCoreletsUsed_DSC2_"] = 1
    return True
