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

"""G3 communication cost model for on-chip LX-ring collectives.

This module is DELIBERATELY SEPARATE from the matmul cost model in
``work_division.py`` (``_matmul_split_cost``).  It prices a *different*
resource: on-chip LX-ring *movement* (relayout / broadcast / gather / reduce
edges).  The matmul model prices the OP (compute / HBM / PSUM); G3 prices the
EDGE (the collective the op's split induces on its operands).  The planner
composes the two by ADDITION at the seam (``op_cost + edge_cost``); G3 must
never be multiplied into ``hbm_us``.  Conflating the two resources is exactly
the Bet-2 ``_cohort_penalty`` hack this module exists to replace.

Pure Python, stdlib only.  No ``torch`` import — the core stays unit-testable
anywhere.  The adapter that builds a :class:`CommEdge` from a realized
``LXRelayoutPlan`` lives in ``lx_relayout.py`` so this file has zero backend
dependency; ``work_division.py`` imports THIS module (one-directional dep).

Calibration constants are MEASURED this session (see the seed model
``g1_ring_model.py``); do not invent others.  Every equation here reproduces
the seed's validated anchors:

  * Attention 1 MiB/head, G=8: naive_all2all (1 execute, occ=16) = 22.3 us is
    the FASTEST schedule; ring_carousel (7 executes) = 57.7 us is 2.6x slower;
    recursive_doubling = 28.5 us; bidir_ring = 32.9 us.
  * ring beats naive-1-exec only for operand > 5.2 MiB/head.
  * LX-capacity interception: 5.2 MiB crossover > 2 MiB LX cap, so any operand
    large enough to favor the ring is force-tiled below the crossover and the
    ring NEVER wins for an LX-resident broadcast.
  * LSE reduce over P=32: plain ring (P-1)=31 hops = 226 us (98% fixed cost);
    a log2-depth tree fold = 5 rounds = 36.5 us.
"""

from __future__ import annotations

import dataclasses
import math

# --- Communication-class tags (mirror lx_relayout.py:55-62; duplicated here so
# --- this module keeps zero backend dependency). ---
COMM_CLASS_SCATTER = "scatter"
COMM_CLASS_BROADCAST = "broadcast"
COMM_CLASS_MULTICAST = "multicast"
COMM_CLASS_GATHER = "gather"
COMM_CLASS_ALL_GATHER = "all_gather"
COMM_CLASS_REDUCE = "reduce"
COMM_CLASS_ALL_REDUCE = "all_reduce"
COMM_CLASS_UNSUPPORTED = "unsupported"

_BROADCAST_LIKE = frozenset(
    {COMM_CLASS_BROADCAST, COMM_CLASS_MULTICAST, COMM_CLASS_ALL_GATHER}
)
_REDUCE_LIKE = frozenset({COMM_CLASS_REDUCE, COMM_CLASS_ALL_REDUCE})

# =====================================================================
# CALIBRATION CONSTANTS — MEASURED this session (see g1_ring_model.py).
# Do NOT invent others.
# =====================================================================
F = 7.3e-6  # s, fixed per STCDP execute (LX_TO_LX_SPEED). MEASURED.
# B/s, F-corrected asymptotic per-link raw rate (uniform p->p+1 shift). MEASURED.
# Matches the seed RATE=140e9; this is the rate used in the schedule TIME model.
RAW_RATE = 140e9
LX_CAP = 2 * 1024 ** 2  # B, ~2 MiB/core LX capacity. STRUCTURAL / MEASURED.
HBM_BW = 204.8e9  # B/s, ONE shared HBM pipe (== _HBM_BW_GBS work_division.py:957).

# Effective-bandwidth band plateaus keyed on PER-LINK TRANSFER COUNT (the band
# variable is transfer COUNT, not burst size; burst x50-100 gave 1.0x MEASURED).
# These describe the *effective aggregate* bandwidth of a high-occupancy link
# and drive the C7 slow-band attribution / the ``effective_bw_bps`` resource
# field.  The schedule TIME model itself uses RAW_RATE on the busiest physical
# link (that is what reproduces the seed's 22.3 us naive anchor).
_BAND_UNIFORM_BPS = RAW_RATE  # occ == 1: uniform shift, ~140 GB/s. MEASURED.
_BAND_SCATTER_BPS = 36e9  # occ in 2..9: scatter / all-to-all plateau. MEASURED.


def band(occupancy: int) -> float:
    """Effective aggregate bandwidth (B/s) for a physical link carrying
    ``occupancy`` transfers.  MEASURED band table:

      * occ <= 1: uniform p->p+1 shift, ~140 GB/s (RAW_RATE).
      * occ in 2..9: scatter / all-to-all plateau, ~36 GB/s.
      * occ > 9: strictly worse than 36 GB/s ("16/link is worse than 36"),
        modeled as 36 * (9 / occ).

    This is the *effective-bandwidth* view (for the resource vector / C7
    attribution).  It is NOT the divisor in the schedule time model; see
    :func:`_t_step`.
    """
    if occupancy <= 1:
        return _BAND_UNIFORM_BPS
    if occupancy <= 9:
        return _BAND_SCATTER_BPS
    return _BAND_SCATTER_BPS * (9.0 / occupancy)


# =====================================================================
# INPUT: CommEdge — the descriptor G3 prices.
# =====================================================================
@dataclasses.dataclass(frozen=True)
class CommEdge:
    """A logical on-chip collective edge to be priced.

    Two construction modes: either supply ``(group_count, group_size)`` and let
    G3 synthesize a within-group all-to-all ``coord_map`` over contiguous core
    ids, or supply an explicit ``coord_map`` (src_core -> {dest_core}) read off
    a realized ``LXRelayoutPlan``.

    Fields:
      comm_class:      one of the COMM_CLASS_* tags; selects the schedule set.
      operand_bytes:   total logical operand moved (e.g. 1 MiB/head).
      dtype_bytes:     element size (2 for fp16).
      group_count:     number of disjoint consumer groups (anchor: 4).
      group_size:      cores per group == replicas/group == chunks/group (8).
      coord_map:       explicit src_core -> {dest_core} on the P-ring; when
                       None it is synthesized from (group_count, group_size).
      payload_per_hop: bytes per logical transfer; defaults to the chunk
                       (operand_bytes // group_size).  For reduce/LSE this is
                       the tiny L-independent payload, supplied explicitly.
      is_reduction:    picks the reduce / all_reduce F-dominated chain path.
    """

    comm_class: str
    operand_bytes: int
    dtype_bytes: int = 2
    group_count: int = 1
    group_size: int = 1
    coord_map: dict[int, set[int]] | None = None
    payload_per_hop: int | None = None
    is_reduction: bool = False

    @property
    def chunk_bytes(self) -> int:
        """Default per-transfer payload = operand split across the group."""
        if self.payload_per_hop is not None:
            return int(self.payload_per_hop)
        g = max(1, self.group_size)
        return int(self.operand_bytes // g)


# =====================================================================
# OUTPUT: CommCost — a RESOURCE VECTOR, not a scalar.
# =====================================================================
@dataclasses.dataclass(frozen=True)
class CommCost:
    """Priced cost of a :class:`CommEdge`.

    time_us:               scheduled wall time of the chosen best schedule.
    executes:              STCDP execute count (dominates for small operands).
    schedule:              name of the winning schedule.
    max_link_occupancy:    peak transfers on any one physical link (band var).
    effective_bw_bps:      band(max_link_occupancy) — the slow-band indicator.
    lx_highwater:          peak LX bytes/core the schedule holds (<= LX_CAP or
                           the operand is force-tiled).
    n_tiles:               tile count if operand_bytes > LX_CAP (else 1).
    dram_bytes_saved_vs_spill: HBM round-trip (write+read) avoided by keeping
                           the operand LX-resident instead of spilling.  This
                           is what the 1.053x Granite kernel actually banks
                           (HBM-round-trip elimination, NOT ring efficiency).
    """

    time_us: float
    executes: int
    schedule: str
    max_link_occupancy: int
    effective_bw_bps: float
    lx_highwater: int
    n_tiles: int
    dram_bytes_saved_vs_spill: int


# =====================================================================
# TERM FUNCTIONS
# =====================================================================
def _ring_size(edge: CommEdge) -> int:
    """Number of cores on the physical P-ring the coord_map spans."""
    if edge.coord_map is not None:
        cores = set(edge.coord_map)
        for dsts in edge.coord_map.values():
            cores |= dsts
        return max(1, max(cores) + 1) if cores else 1
    return max(1, edge.group_count * edge.group_size)


def _synthesize_coord_map(group_count: int, group_size: int) -> dict[int, set[int]]:
    """Within-group all-to-all over contiguous core ids: each core sends to
    every core in its group (including itself; the self-transfer is FREE and is
    dropped by :func:`link_occupancy`)."""
    cmap: dict[int, set[int]] = {}
    for g in range(group_count):
        base = g * group_size
        members = set(range(base, base + group_size))
        for src in members:
            cmap[src] = set(members)
    return cmap


def link_occupancy(
    coord_map: dict[int, set[int]], ring_size: int, group_size: int | None = None
):
    """Route each logical transfer on the SHORTEST ARC and count, for each
    physical adjacent-hop link, how many transfers cross it.

    Each disjoint cohort is its own independent ring.  When ``group_size`` is
    given, a transfer between cores in the same contiguous group of
    ``group_size`` cores is routed on that group's local ring (modulus
    ``group_size``): this is the MEASURED reality — four contiguous 8-core
    groups are four independent 8-core rings, so a within-group all-to-all peaks
    at 16/link (the anchor), NOT 32 on a single 32-core ring.  When
    ``group_size`` is None the whole ``ring_size`` ring is used (a single
    cohort).

    SAME-CORE (src == dest) is a FREE local copy: it contributes 0 links and 0
    executes (a self-ring must lower to a BusFence, never a transfer;
    STRUCTURAL).  Returns ``(per_link_counts, max_occupancy, n_cross, n_same)``.

    Anchor check (4 groups of 8, within-group all-to-all, group_size=8): 32
    same-core (free) + 224 cross-core, MAX occupancy 16/link.
    """
    per_link: dict[tuple, int] = {}
    n_cross = 0
    n_same = 0
    for src, dsts in coord_map.items():
        for dst in dsts:
            if src == dst:
                n_same += 1
                continue
            n_cross += 1
            if group_size:
                # Route on the local ring of this transfer's group.
                g = src // group_size
                P = group_size
                s = src % group_size
                d = dst % group_size
                key_prefix = g
            else:
                P = ring_size
                s, d = src, dst
                key_prefix = 0
            fwd = (d - s) % P
            bwd = (s - d) % P
            if fwd <= bwd:
                # forward arc: links s, s+1, ..., d-1
                for step in range(fwd):
                    link = (key_prefix, (s + step) % P)
                    per_link[link] = per_link.get(link, 0) + 1
            else:
                # backward arc: links d, d+1, ..., s-1
                for step in range(bwd):
                    link = (key_prefix, (d + step) % P)
                    per_link[link] = per_link.get(link, 0) + 1
    max_occ = max(per_link.values()) if per_link else 0
    return per_link, max_occ, n_cross, n_same


def _t_step(max_link_bytes: float) -> float:
    """Time (s) of ONE sequential STCDP execute whose busiest physical link
    carries ``max_link_bytes``.  This is the seed time model exactly:

        t_step = F + (max bytes on any one physical link) / RAW_RATE

    (g1_ring_model.py lines 3-5, 22).  RAW_RATE is used here — the seed's
    22.3 us naive anchor is ``F + 16*128KiB/140e9``, so the schedule time model
    divides by RAW_RATE on the busiest link; the 36 GB/s band is the effective
    *aggregate* view reported separately in the resource vector.
    """
    return F + max_link_bytes / RAW_RATE


# =====================================================================
# SCHEDULE COSTS (per-tile; each returns (time_s, executes, max_occ)).
# Reproduces g1_ring_model.py:model_operand exactly.
# =====================================================================
def _sched_naive_all2all(chunk: int, occ: int) -> tuple[float, int, int]:
    """ONE execute; the busiest link carries occ*chunk bytes at RAW_RATE."""
    return (_t_step(occ * chunk), 1, occ)


def _sched_ring_carousel(chunk: int, g: int) -> tuple[float, int, int]:
    """Bandwidth-optimal ring all-gather: G-1 sequential steps, occ 1/link."""
    steps = g - 1
    return (steps * (F + chunk / RAW_RATE), steps, 1)


def _sched_bidir_ring(chunk: int, g: int) -> tuple[float, int, int]:
    """Bidirectional ring: ceil((G-1)/2) steps, occ 1/link."""
    steps = math.ceil((g - 1) / 2)
    return (steps * (F + chunk / RAW_RATE), steps, 1)


def _sched_recursive_doubling(chunk: int, g: int) -> tuple[float, int, int]:
    """Recursive-doubling all-gather: log2(G) rounds; round r moves 2^r chunks.
    Optimistic pipelined occ=1 (the variant the seed ranks; 28.5 us anchor)."""
    steps = max(1, int(math.log2(g))) if g >= 2 else 1
    t = sum(F + (2 ** r) * chunk / RAW_RATE for r in range(steps))
    return (t, steps, 1)


# Schedule sets keyed by comm-class family.
def _broadcast_schedules(chunk: int, g: int) -> dict[str, tuple[float, int, int]]:
    occ_naive = _all2all_max_occ(g)
    return {
        "naive_all2all_1exec": _sched_naive_all2all(chunk, occ_naive),
        "recursive_doubling": _sched_recursive_doubling(chunk, g),
        "bidir_ring": _sched_bidir_ring(chunk, g),
        "ring_carousel": _sched_ring_carousel(chunk, g),
    }


def _all2all_max_occ(g: int) -> int:
    """Peak physical-link occupancy for a within-group all-to-all over g
    contiguous cores (shortest-arc routing).  For g=8 this is 16 (the anchor's
    MEASURED central-link occupancy)."""
    if g <= 1:
        return 0
    cmap = _synthesize_coord_map(1, g)
    _, max_occ, _, _ = link_occupancy(cmap, g, group_size=g)
    return max_occ


# =====================================================================
# REDUCE / LSE-FOLD (F-dominated linear chain vs log2 tree).
# =====================================================================
def _reduce_schedules(payload: int, P: int) -> dict[str, tuple[float, int, int]]:
    """Reduce / all_reduce over P participants with tiny L-independent payload.

    plain_ring: (P-1) sequential hops -> F-dominated (P=32 -> 226 us, 98% F).
    tree_fold:  ceil(log2 P) rounds -> the strictly-cheaper schedule the model
                selects (P=32 -> 5 rounds -> 36.5 us).
    """
    plain_hops = max(0, P - 1)
    tree_hops = max(0, math.ceil(math.log2(P))) if P >= 2 else 0
    return {
        "reduce_plain_ring": (plain_hops * (F + payload / RAW_RATE), plain_hops, 1),
        "reduce_tree_fold": (tree_hops * (F + payload / RAW_RATE), tree_hops, 1),
    }


# =====================================================================
# best_schedule + LX_CAP tiling + the public cost() entry point.
# =====================================================================
def _candidate_schedules(edge: CommEdge, chunk: int):
    """Return {name: (time_s, executes, max_occ)} for this edge's comm class."""
    if edge.is_reduction or edge.comm_class in _REDUCE_LIKE:
        # payload is the (tiny) per-hop payload; P is the participant count.
        payload = edge.chunk_bytes
        P = _ring_size(edge)
        return _reduce_schedules(payload, P)
    if edge.comm_class in _BROADCAST_LIKE:
        return _broadcast_schedules(chunk, max(1, edge.group_size))
    if edge.comm_class in (COMM_CLASS_SCATTER, COMM_CLASS_GATHER):
        # Permutation: occupancy read directly from the coord map.
        cmap = edge.coord_map or _synthesize_coord_map(
            edge.group_count, edge.group_size
        )
        gs = edge.group_size if edge.coord_map is None else None
        _, max_occ, _, _ = link_occupancy(cmap, _ring_size(edge), group_size=gs)
        return {"naive_all2all_1exec": _sched_naive_all2all(chunk, max_occ)}
    # Unsupported / no-communication edge: zero cost.
    return {}


def best_schedule(edge: CommEdge, chunk: int) -> tuple[str, float, int, int]:
    """Pick the cheapest schedule (argmin time) for a SINGLE tile of ``chunk``
    payload.  Returns ``(name, time_s, executes, max_occ)``.  Empty schedule set
    (no-communication edge) -> zero cost."""
    scheds = _candidate_schedules(edge, chunk)
    if not scheds:
        return ("none", 0.0, 0, 0)
    name = min(scheds, key=lambda k: scheds[k][0])
    t, execs, occ = scheds[name]
    return (name, t, execs, occ)


def cost(edge: CommEdge) -> CommCost:
    """Price a :class:`CommEdge` and return the RESOURCE VECTOR.

    LX_CAP TILING (the KEY STRUCTURAL interception): if the operand exceeds the
    ~2 MiB/core LX capacity it is force-tiled to n_tiles = ceil(bytes/LX_CAP)
    tiles, each priced independently and summed.  Because the ring-vs-naive
    crossover (5.2 MiB) exceeds LX_CAP (2 MiB), every LX-resident tile sits
    BELOW the crossover, so naive-1-exec wins each tile and the ring NEVER wins
    for an LX-resident broadcast.
    """
    # No-communication classes (seam-transparent M-split, unsupported): 0 cost.
    if edge.comm_class == COMM_CLASS_UNSUPPORTED and not edge.is_reduction:
        return CommCost(
            time_us=0.0,
            executes=0,
            schedule="none",
            max_link_occupancy=0,
            effective_bw_bps=band(0),
            lx_highwater=0,
            n_tiles=1,
            dram_bytes_saved_vs_spill=0,
        )

    reduce_like = edge.is_reduction or edge.comm_class in _REDUCE_LIKE

    if reduce_like:
        # Reduce payload is tiny and L-independent; not tiled by LX_CAP.
        name, t, execs, occ = best_schedule(edge, edge.chunk_bytes)
        return CommCost(
            time_us=t * 1e6,
            executes=execs,
            schedule=name,
            max_link_occupancy=occ,
            effective_bw_bps=band(occ),
            lx_highwater=edge.chunk_bytes,
            n_tiles=1,
            dram_bytes_saved_vs_spill=0,
        )

    # Broadcast-like: tile the operand to LX_CAP, price one tile, scale by tiles.
    operand = max(0, edge.operand_bytes)
    n_tiles = max(1, math.ceil(operand / LX_CAP)) if operand > 0 else 1
    tile_bytes = operand // n_tiles if n_tiles else operand
    g = max(1, edge.group_size)
    tile_chunk = edge.payload_per_hop if edge.payload_per_hop is not None else (
        tile_bytes // g
    )

    name, t_tile, execs_tile, occ = best_schedule(edge, int(tile_chunk))
    time_us = t_tile * n_tiles * 1e6
    executes = execs_tile * n_tiles

    # A resident collective banks a saved HBM round trip (write + read) of the
    # whole operand vs spilling it to the single shared HBM pipe. This is what
    # the 1.053x Granite kernel actually wins (HBM-round-trip elimination), and
    # it is a DIFFERENT resource from the on-chip edge time above.
    dram_saved = 2 * operand

    return CommCost(
        time_us=time_us,
        executes=executes,
        schedule=name,
        max_link_occupancy=occ,
        effective_bw_bps=band(occ),
        lx_highwater=min(operand, LX_CAP) if operand else 0,
        n_tiles=n_tiles,
        dram_bytes_saved_vs_spill=dram_saved,
    )


def comm_edge_cost_us(edge: CommEdge) -> float:
    """Scalar wall-time (us) of the cheapest schedule for ``edge``.

    This is the value the planner ADDS to ``_matmul_split_cost`` at the seam
    (``op_cost + edge_cost``).  It is NEVER multiplied into ``hbm_us``.
    """
    return cost(edge).time_us


# =====================================================================
# Analysis helpers (used by validation / the interception theorem).
# =====================================================================
def crossover_mib_per_head(group_size: int = 8) -> float:
    """Operand size (MiB) above which ring_carousel beats naive_all2all_1exec.

    Binary search on the two per-operand schedule costs (untiled), reproducing
    the seed crossover of ~5.20 MiB/head at G=8.  This is > LX_CAP (2 MiB): the
    interception theorem.
    """
    g = group_size
    occ = _all2all_max_occ(g)

    def ring_lt_naive(operand: float) -> bool:
        chunk = operand / g
        t_ring = _sched_ring_carousel(chunk, g)[0]
        t_naive = _sched_naive_all2all(chunk, occ)[0]
        return t_ring < t_naive

    lo, hi = 0.1 * 1024 ** 2, 64 * 1024 ** 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if ring_lt_naive(mid):
            hi = mid
        else:
            lo = mid
    return hi / 1024 ** 2
