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

"""Pure-persistence on-chip pass: keep a same-shard same-core SDSC chain LX-resident.

Realizes a *run* of consecutive same-shard same-core activation edges as pure LX
persistence: no move, no data-op, no mixed SDSC, stock dxp. The classic example
is the softmax chain inside the materialized-scores SDPA decomposition
(max -> sub -> exp -> sum -> realdiv) -- every edge has identical work-division
and per-core HBM bases across all 32 cores, so flipping both endpoints
LX-resident at the same base (producer write @B, consumer read @B) lets the
consumer pick up exactly what the producer left in LX.

This module is self-contained: it inlines the small set of pure-Python dict
surgery helpers (``LxFlip``, ``_dl_op``, ``_core_state_init_entry``,
``apply_lx_flip``) and the ``LX_CAPACITY_BYTES`` constant it needs, so it has no
dependency on any other on-chip realization machinery. The flip primitive is the
safe shared base-pointer flip: it rewrites a labeledDs to LX-resident at a given
per-core base and clears its HBM addressing.

DETECTION (per-edge eligibility, ``detect_same_core_chain``)
-----------------------------------------------------------
An activation edge producer-OUT(p) -> consumer-IN(c) is *same-shard same-core*,
hence LX-persistable with no move, iff:

  1. producer and consumer SDSC share the same ``numWkSlicesPerDim_`` (split),
  2. the producer-output allocate node and the consumer-input allocate node have
     IDENTICAL per-core HBM bases (every core reads exactly the slot it writes),
  3. both endpoints are HBM-backed activations (an intermediate, not a graph
     output / weight).

A *chain* is the maximal run of such edges over consecutive SDSCs. Each chain
intermediate (one producer output buffer) becomes one LX region shared by the
producer and ALL its in-chain consumers (a buffer with N readers contributes one
region read by N consumer inputs). Cross-shard edges (matmul boundaries with
mismatched split / bases) are correctly skipped here.

REALIZATION (``plan_lx_placement`` + ``apply_chain``)
-----------------------------------------------------
For each chain intermediate, in birth order, ``plan_lx_placement`` assigns an LX
base from the usable window with liveness-aware first-fit: a base is reused once
its holder is dead (its last in-chain reader has run). A budget check fails the
buffer closed (it stays HBM, edge untouched) when no base fits. ``apply_chain``
then flips producer-output + every consumer-input to that base via
:func:`apply_lx_flip` on the in-memory SDSC dicts.

USABLE LX WINDOW (device-calibrated)
------------------------------------
Per-core LX is 2 MB (``LX_CAPACITY_BYTES``) but matched-base persistence is
value-correct only in the lower ~1.5 MB; the upper ~512 KB holds each op's
auto-assigned working buffers (a persisted tensor placed at 1572864 collides).
The allocator hands out bases from ``[0, USABLE_LX_BYTES)`` only.

Gated by ``config.onchip_softmax_chain``. Default-off; output byte-identical to
before when off.
"""

from __future__ import annotations

import dataclasses

from .logging_utils import get_inductor_logger

logger = get_inductor_logger("onchip_softmax_chain")

# Per-core LX capacity (2 MB). Inlined here so this pass is self-contained.
LX_CAPACITY_BYTES = 2 << 20  # 2_097_152
# DL-DSC LX size sentinel written into a flipped labeledDs / allocate node.
DL_LX_SENTINEL = 2147483647

# The upper ~512 KB of per-core LX is reserved for each op's auto-assigned LX
# working buffers; matched-base persistence collides above this. Device-
# calibrated: 0/524288/1048576 persist, 1572864 fails.
RESERVED_TOP_BYTES = 512 << 10
USABLE_LX_BYTES = LX_CAPACITY_BYTES - RESERVED_TOP_BYTES  # 1.5 MB lower window
STICK_BYTES = 128
WORD_LENGTH_FP16 = 2


# ---------------------------------------------------------------------------
# Inlined pure-Python SDSC dict surgery (torch-free): the safe shared
# base-pointer LX flip primitive and its helpers.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class LxFlip:
    """Flip one DL labeledDs to LX-resident: ``base`` per core, sentinel size."""

    ldsidx: int
    lx_base: int
    role: str  # "producer-output" | "consumer-input"


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

    Rewrites the labeledDs (memOrg_ -> lx, hbm addr/size cleared, lx size
    sentinel, per-core coreStateInit_) and its scheduleTree allocate node
    (name/component/per-core LX base). In place.
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


# ---------------------------------------------------------------------------
# Chain detection / placement / application.
# ---------------------------------------------------------------------------
def _align_up(n: int, a: int) -> int:
    return ((n + a - 1) // a) * a


def _shard(sdsc_json: dict) -> dict | None:
    """The ``numWkSlicesPerDim_`` shard descriptor, or None if absent."""
    return sdsc_json[next(iter(sdsc_json))].get("numWkSlicesPerDim_")


def _alloc_node(dl: dict, lds_idx: int) -> dict | None:
    for n in dl["scheduleTree_"]:
        if n.get("nodeType_") == "allocate" and n.get("ldsIdx_") == lds_idx:
            return n
    return None


def _percore_bases(dl: dict, lds_idx: int) -> dict[int, int] | None:
    """Per-core HBM base map ``{core: addr}`` for an HBM-backed labeledDs.

    Returns None if the labeledDs is not present, is not HBM-backed (e.g.
    already LX), or otherwise unfit as a same-core persistence endpoint.
    """
    node = _alloc_node(dl, lds_idx)
    if node is None or node.get("component_") != "hbm":
        return None
    out: dict[int, int] = {}
    for k, v in node["startAddressCoreCorelet_"]["data_"].items():
        out[int(k.strip("[]").split(",")[0])] = int(v)
    return out


def _io_indices(dl: dict) -> tuple[list[int], list[int]]:
    cop = dl["computeOp_"][0]
    ins = [int(x.rsplit("-idx", 1)[1]) for x in cop.get("inputLabeledDs", [])]
    outs = [int(x.rsplit("-idx", 1)[1]) for x in cop.get("outputLabeledDs", [])]
    return ins, outs


def _slice_bytes_for_shard(
    dl: dict, lds_idx: int, shard: dict, word_length: int = WORD_LENGTH_FP16
) -> int:
    """Per-core LX bytes for the buffer: total elements / num_cores, stick-aligned.

    The buffer's total extent is the product of every dim's folds; an even
    same-shard split hands each of ``num_cores`` cores an equal piece. Padding
    to the 128-byte stick happens once at the per-core footprint (the device's
    stick granularity), not per split-dim chunk, so a sub-stick split chunk
    that shares a stick with the contiguous (stick) dim is not over-counted.
    """
    num_cores = 1
    for v in shard.values():
        num_cores *= v
    node = _alloc_node(dl, lds_idx)
    if node is None:
        return 0
    coord = node["coordinates_"]["coordInfo"]
    total = 1
    for _, folds in coord.items():
        for a in folds["folds"]["dim_prop_attr"]:
            total *= a["factor_"]
    return _align_up((total // num_cores) * word_length, STICK_BYTES)


@dataclasses.dataclass
class Intermediate:
    """A chain intermediate buffer: one producer output, >=1 consumer reads."""

    prod_ordinal: int  # producer SDSC index in sdscs_json
    prod_out_idx: int
    born: int  # = prod_ordinal (alias for clarity in the planner)
    last_read: int  # last consumer SDSC ordinal (liveness end)
    consumers: list  # [(cons_ordinal, cons_in_idx)]
    slice_bytes: int


@dataclasses.dataclass
class Placement:
    intermediate: Intermediate
    lx_base: int


def detect_same_core_chain(sdscs_json: list[dict]) -> list[Intermediate]:
    """Find all same-shard same-core LX-persistable intermediates in the bundle.

    Walks the SDSCs in order; for each HBM-backed producer output, finds every
    later SDSC whose input has IDENTICAL per-core bases AND the same split.
    Each such producer output is one intermediate (possibly several consumers).
    Cross-shard edges (split mismatch / base mismatch) are skipped -- those are
    matmul boundaries that need a data-op move.

    Returns the intermediates in producer order; an empty list means no chain
    was eligible (and the pass is a no-op).
    """
    intermediates: list[Intermediate] = []
    for p_idx, prod in enumerate(sdscs_json):
        pshard = _shard(prod)
        if pshard is None:
            continue
        pdl = _dl_op(prod)
        _, p_outs = _io_indices(pdl)
        for out_idx in p_outs:
            p_bases = _percore_bases(pdl, out_idx)
            if p_bases is None:
                continue  # already LX, or non-HBM -> not a persistable producer
            consumers: list[tuple[int, int]] = []
            last_read = p_idx
            for c_idx in range(p_idx + 1, len(sdscs_json)):
                cons = sdscs_json[c_idx]
                if _shard(cons) != pshard:
                    continue  # cross-shard: needs a move, not pure persistence
                cdl = _dl_op(cons)
                c_ins, _ = _io_indices(cdl)
                for in_idx in c_ins:
                    c_bases = _percore_bases(cdl, in_idx)
                    if c_bases is None or c_bases != p_bases:
                        continue
                    consumers.append((c_idx, in_idx))
                    last_read = max(last_read, c_idx)
            if consumers:
                slice_bytes = _slice_bytes_for_shard(pdl, out_idx, pshard)
                intermediates.append(
                    Intermediate(
                        prod_ordinal=p_idx,
                        prod_out_idx=out_idx,
                        born=p_idx,
                        last_read=last_read,
                        consumers=consumers,
                        slice_bytes=slice_bytes,
                    )
                )
    return intermediates


def plan_lx_placement(
    intermediates: list[Intermediate],
    usable_bytes: int = USABLE_LX_BYTES,
    prereserved: list[tuple[int, int, int]] | None = None,
) -> tuple[list[Placement], list[Intermediate]]:
    """Assign each intermediate a non-overlapping LX base; budget fail-to-HBM.

    Liveness-aware first-fit over the lower usable window: an intermediate born
    at SDSC b reuses any region whose holder is dead (last_read < b). A buffer
    that cannot fit is dropped (returned in ``skipped``) and stays HBM-backed --
    the edge is left untouched (correct, just not accelerated).

    ``prereserved`` seeds the LX with regions an upstream occupant already pins
    in LX, as ``(base, size, last_read)`` triples carrying their own liveness.
    Returns ``(placements, skipped)``.
    """
    placements: list[Placement] = []
    skipped: list[Intermediate] = []
    # Active allocations: (base, size, dead_after_ordinal).
    active: list[tuple[int, int, int]] = list(prereserved or [])

    for itm in sorted(intermediates, key=lambda x: x.born):
        # Reclaim regions whose holder died strictly before this birth.
        active = [a for a in active if a[2] >= itm.born]
        size = _align_up(itm.slice_bytes, STICK_BYTES)
        # First-fit: candidate bases at 0 and just above each active region.
        candidates = [0] + [_align_up(b + s, STICK_BYTES) for b, s, _ in active]
        placed: int | None = None
        for base in sorted(set(candidates)):
            if base + size > usable_bytes:
                continue
            overlap = any(
                not (base + size <= ab or base >= ab + asz) for ab, asz, _ in active
            )
            if not overlap:
                placed = base
                break
        if placed is None:
            skipped.append(itm)
            continue
        active.append((placed, size, itm.last_read))
        placements.append(Placement(itm, placed))
    return placements, skipped


def apply_chain(sdscs_json: list[dict], placements: list[Placement]) -> int:
    """Flip every placed intermediate (producer-out + all consumer-ins) to LX.

    Returns the number of labeledDs endpoints flipped (1 producer + N consumers
    per placement). In-place mutation of ``sdscs_json``.
    """
    flipped = 0
    for pl in placements:
        itm = pl.intermediate
        apply_lx_flip(
            sdscs_json[itm.prod_ordinal],
            LxFlip(itm.prod_out_idx, pl.lx_base, "producer-output"),
        )
        flipped += 1
        for c_ordinal, c_in_idx in itm.consumers:
            apply_lx_flip(
                sdscs_json[c_ordinal],
                LxFlip(c_in_idx, pl.lx_base, "consumer-input"),
            )
            flipped += 1
    return flipped


def realize_softmax_chain(
    sdscs_json: list[dict],
    prereserved: list[tuple[int, int, int]] | None = None,
) -> int:
    """Detect + place + apply the same-core chain LX-persistence pass.

    In-place mutation of the SDSC list; returns a count of flipped endpoints
    (zero when no chain is eligible). Caller gates on
    ``config.onchip_softmax_chain``.

    ``prereserved`` lets an upstream occupant pin its own LX regions with their
    own liveness so the chain tail packs on top once the upstream occupant dies.
    Pass None when no such upstream occupant exists.
    """
    intermediates = detect_same_core_chain(sdscs_json)
    if not intermediates:
        return 0
    placements, skipped = plan_lx_placement(intermediates, prereserved=prereserved)
    flipped = apply_chain(sdscs_json, placements)
    if logger:
        logger.info(
            "onchip_softmax_chain: %d intermediates detected, %d placed "
            "(%d endpoints flipped), %d over budget -> HBM",
            len(intermediates),
            len(placements),
            flipped,
            len(skipped),
        )
    return flipped


__all__ = [
    "Intermediate",
    "Placement",
    "USABLE_LX_BYTES",
    "apply_chain",
    "apply_lx_flip",
    "detect_same_core_chain",
    "plan_lx_placement",
    "realize_softmax_chain",
]
