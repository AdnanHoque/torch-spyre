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

"""Ring-occupancy cost model for the carousel work-division family.

Layers on the byte-level comm cost model (``comm_cost_model.py``), which prices a
single producer->consumer handoff as delivered-bytes / fabric-bandwidth. That
model is right for one edge but blind to the ring itself: a rotation or a fold
uses the SAME links over and over, and two ring-heavy regions in the same
program window fight for them. This module makes the ring a *finite resource*.

The resource is LINK-TIME, not bytes. One transfer of ``b`` bytes across ``h``
links busies each of those links for ``b/rho`` microseconds, so it demands
``h * b/rho`` link-us. ``ring_occupancy(region)`` returns that demand summed over
a region; the planner budgets it against the ring supply of ``P * DUPLEX``
link-us per wall-us and, in v1, serializes regions that overlap.

Three prices are exposed:

  price_carousel(plan)  weight-rotation region. Ring occupancy grows from 0 at
                        naive replication (radius = cores) to ~(P-1)*|W|/rho at
                        the full carousel (each of P tiles crosses the ring once).
                        The compute-vs-ring RATIO per step is radius-invariant.
                        Also reports the compute-bound gate that says whether the
                        rotation hides under the matmul.
  price_fold(variant)   LSE ring-fold for the KV-carousel merge: per-hop payload x
                        hops, for the linear / reduce_scatter / recursive topologies.
  serialize(*priced)    v1 contention: ring-heavy regions cannot share links, so
                        their walls and occupancies add.

Plain arithmetic throughout. Assumptions and known inaccuracies are at the bottom
of the file; read them before trusting a number.
"""

from __future__ import annotations

import dataclasses
import math

# --- Fabric constants -------------------------------------------------------
# rho: RIU BiRing bandwidth PER DIRECTION. The ring is duplex (two independent
# directions), so its aggregate link supply is DUPLEX x this.
RING_GBPS = 166.0
P_CORES = 32
DUPLEX = 2

# lambda: per-hop pipeline latency of one ring move (sync + launch of the
# STCDPOpLx that realizes the reshard). Sets the crossover between few-big-hop
# and many-small-hop fold topologies. HONEST: PLACEHOLDER, needs a device
# measurement; the fold ranking is sensitive to it.
HOP_LATENCY_US = 0.5

# HBM read-only ceiling (weight streaming is a pure read). Used only to size the
# DRAM the carousel removes, for context against the ring it adds.
HBM_READ_GBPS = 143.0

# SFP elementwise rate for the LSE-combine that every fold hop runs on the moved
# partial (the ring only moves; the max/exp/add is a separate SFP op). Rough.
SFP_GBPS = 166.0

# Per-core LX scratchpad. A rotated slab must double-buffer inside it.
LX_BYTES = 2 * 1024 * 1024

# Aggregate fp16 array rate, only for the carousel compute-bound gate. ROUGH,
# needs the datasheet; the ring occupancy numbers do not depend on it.
FP16_TFLOPS_AGG = 150.0

# Ring supply: link-us of occupancy the whole ring can retire per wall-us.
RING_SUPPLY = P_CORES * DUPLEX


def _us(num_bytes: float, gbps: float) -> float:
    """Microseconds to move num_bytes at gbps. Same helper as the parent model."""
    return num_bytes / (gbps * 1e3)


# --- Ring region: the finite resource ---------------------------------------
@dataclasses.dataclass(frozen=True)
class RingWave:
    """One synchronized burst on the ring.

    count:  transfers that fire together (P for a rotation step, 1 for a chain hop).
    nbytes: payload of each transfer.
    hops:   links each transfer crosses (its distance on the ring).
    """

    count: int
    nbytes: float
    hops: int


@dataclasses.dataclass(frozen=True)
class RingRegion:
    """A ring-traffic region = waves that run back-to-back in program order.

    Occupancy (link-us) is the demand the region places on the fabric; it is
    additive and direction-agnostic. Wall time is pattern-specific and computed
    by the pricing functions, because a rotation spreads its bytes over all P
    links while a chain fold piles them onto one.

    directions: ring directions the region can spread across (1 or DUPLEX). Only
    affects the balanced occupancy floor, never the demand.
    """

    name: str
    waves: tuple[RingWave, ...]
    directions: int = 1


def ring_occupancy(region: RingRegion) -> float:
    """Link-us the region demands: sum over waves of count * hops * bytes/rho.

    This is the first-class resource. A planner sums it over the regions live in a
    program window and compares to ``RING_SUPPLY * window_us`` to tell whether the
    ring, not the array or HBM, is the bottleneck (see ``ring_bound``).
    """
    return sum(w.count * w.hops * _us(w.nbytes, RING_GBPS) for w in region.waves)


def occupancy_floor_us(region: RingRegion) -> float:
    """Lower bound on wall time if the demand balanced perfectly over the links it
    may use (P * directions). A rotation reaches this floor; a chain does not."""
    return ring_occupancy(region) / (P_CORES * region.directions)


# --- (a) Weight carousel: rotation region -----------------------------------
@dataclasses.dataclass(frozen=True)
class CarouselPlan:
    """Prefill matmul Y[S,N] = X[S,K] @ W[K,N], M-split across ``cores``.

    radius r in divisors(cores): 1 = full carousel (W loaded once, rotated),
    cores = today's naive replication (W loaded per core, no rotation). The ring
    partitions into r arcs of P' = cores/r; each arc holds one W replica and
    rotates it in P' matmul steps (P'-1 ring hops). r is the single knob spanning
    old and new plans.

    slab_k: rows of K moved per rotation message (K-slab granularity) so the
    destination form double-buffers inside LX. None = whole-K tile (usually too
    big for LX -> that is why slabbing exists).
    """

    name: str
    S: int
    K: int
    N: int
    bytes_per_elem: int = 2
    cores: int = P_CORES
    radius: int = 1
    slab_k: int | None = None


def price_carousel(plan: CarouselPlan) -> dict:
    """Price one carousel matmul's rotation region.

    P' matmul steps per arc; each step every core computes on its current N-tile,
    then P'-1 times pushes that tile one hop to its ring neighbor. Per hop that is
    ``cores`` transfers of ``tile`` bytes over 1 link, so every link carries one
    tile -> transport wall per hop = tile/rho, all links in parallel, one
    direction (rotation is one-way, duplex does not help it). Occupancy is 0 at
    naive replication (P'=1) and (P-1)*|W|/rho at the full carousel (each tile
    crosses the ring once). The compute-vs-ring ratio per step is radius-invariant
    (a bigger radius shrinks the tile and the step count together).
    """
    p, r = plan.cores, plan.radius
    assert p % r == 0, "radius must divide cores"
    steps = p // r  # P' matmul steps per arc (one per N-tile seen)
    rot = steps - 1  # ring hops: each core already holds one tile
    w_bytes = plan.K * plan.N * plan.bytes_per_elem  # |W|
    tile = w_bytes / steps  # bytes each core holds/rotates in one replica

    region = RingRegion(
        name=f"{plan.name}:rotate",
        waves=tuple(RingWave(count=p, nbytes=tile, hops=1) for _ in range(rot)),
        directions=1,
    )
    occ = ring_occupancy(region)  # 0 .. (P-1)*|W|/rho

    # Per-step transport and compute. Compute is the rough context for the gate.
    t_x = _us(tile, RING_GBPS)
    c_core = FP16_TFLOPS_AGG * 1e12 / p  # flop/s per core
    flops_step = 2 * (plan.S / p) * plan.K * (plan.N / steps)
    t_c = flops_step / c_core * 1e6

    # Compute-bound gate: transport hides under the matmul iff rho >= C*b*P/(2S),
    # i.e. t_x <= t_c. Prefill is array-under-filled, so this often fails and the
    # real value of the carousel is seam-transparency + array-fill, not wall-clock.
    gate_rho = c_core * plan.bytes_per_elem * p / (2 * plan.S) / 1e9
    transport_hidden = RING_GBPS >= gate_rho

    lam = rot * HOP_LATENCY_US
    wall_serial = steps * t_c + rot * t_x + lam  # today: P2 barrier, no overlap
    wall_overlap = t_c + rot * (max(t_c, t_x) + HOP_LATENCY_US)  # needs P2 edit

    # DRAM the plan reads vs naive replication, for context (read-only stream).
    naive_read = _us(p * w_bytes, HBM_READ_GBPS)  # W streamed once per core
    carousel_read = _us(r * w_bytes, HBM_READ_GBPS)  # once per arc replica

    # LX double-buffer footprint of the rotated slab per core.
    slab_rows = plan.slab_k if plan.slab_k is not None else plan.K
    lx_dbuf = 2 * slab_rows * (plan.N / steps) * plan.bytes_per_elem

    return {
        "name": plan.name,
        "radius": r,
        "steps": steps,
        "tile_KiB": round(tile / 1024, 1),
        "occupancy_us": round(occ, 2),
        "occupancy_floor_us": round(occupancy_floor_us(region), 2),
        "transport_wall_us": round(rot * t_x + lam, 2),
        "compute_wall_us": round(steps * t_c, 2),
        "wall_serial_us": round(wall_serial, 2),
        "wall_overlap_us": round(wall_overlap, 2),
        "gate_rho_gbps": round(gate_rho, 1),
        "transport_hidden": transport_hidden,
        "dram_saved_x": round(p / r, 1),
        "naive_read_us": round(naive_read, 2),
        "carousel_read_us": round(carousel_read, 2),
        "lx_dbuf_KiB": round(lx_dbuf / 1024, 1),
        "lx_fits": lx_dbuf <= LX_BYTES,
        "note": "overlap needs the Conv->matmul P2 gate; today wall_serial holds",
    }


# --- (b) KV carousel: LSE ring-fold -----------------------------------------
@dataclasses.dataclass(frozen=True)
class FoldPlan:
    """Decode-attention merge of P per-core partials (A[h,d] fp32, l[h], m[h]).

    Every core owns a KV-sequence shard and emits an unnormalized partial over all
    query heads; the fold combines the P partials with the associative LSE-combine
    operator. Payload per core = A (heads*head_dim fp32) + l + m.

    cut_through: does a distance-d ring move cost ~ a distance-1 move in wall time
    (latency flat in distance)? Only then does the recursive topology's log2(P)
    rounds beat the linear P-1. Distance always costs d links of OCCUPANCY either
    way. HONEST: unverified on this fabric; defaults False (store-and-forward).
    """

    name: str
    cores: int = P_CORES
    heads: int = 8
    head_dim: int = 128
    cut_through: bool = False


def _payload_bytes(plan: FoldPlan) -> float:
    a = plan.heads * plan.head_dim * 4  # A accumulator, fp32
    lm = 2 * plan.heads * 4  # l (lse mass) + m (running max), fp32
    return a + lm


def _hop_wall(nbytes: float, hops: int, cut_through: bool, combine: bool) -> float:
    """Critical-path time of one serial fold hop: move + latency + optional SFP
    LSE-combine on the received partial. Distance adds to latency only under
    store-and-forward; under cut-through the wire latency is flat in distance."""
    move = _us(nbytes, RING_GBPS)
    lat = HOP_LATENCY_US if cut_through else HOP_LATENCY_US * hops
    combine_us = _us(nbytes, SFP_GBPS) if combine else 0.0
    return move + lat + combine_us


def price_fold(variant: str, plan: FoldPlan | None = None) -> dict:
    """Price the merge fold for one topology.

    linear         P-1 serial hops, full payload each, one link at a time. Lowest
                   occupancy, most hops on the critical path -> lambda dominates at
                   small payload. Output lands on one core.
    reduce_scatter reduce-scatter-over-heads (P-1 hops) then all-gather (P-1 hops),
                   payload/P per hop across all P links. ~2x the occupancy of
                   linear but each hop is tiny; the endpoint is head-split == the
                   K-split the O-projection wants -> zero follow-on relayout.
    recursive      recursive-halving reduce-scatter + doubling all-gather, log2(P)
                   rounds, distance doubling. Fewest rounds (lambda-cheap) but
                   distance-d hops cost d links of occupancy -> only wins on wall
                   if cut_through makes the latency flat in distance.

    The fold is L-independent and sits on every layer's critical path every decode
    step, so lambda sets the ranking. It is composed as ring-move + a separate SFP
    LSE-combine per hop (no fused single-AIU move-then-reduce primitive exists).
    """
    plan = plan or FoldPlan(name="kv_merge")
    p_cores = plan.cores
    payload = _payload_bytes(plan)

    if variant == "linear":
        hops = p_cores - 1
        region = RingRegion(
            name=f"{plan.name}:linear",
            waves=tuple(RingWave(count=1, nbytes=payload, hops=1) for _ in range(hops)),
        )
        wall = sum(
            _hop_wall(payload, 1, plan.cut_through, combine=True) for _ in range(hops)
        )
        endpoint = "single core (needs a scatter before O-proj)"

    elif variant == "reduce_scatter":
        seg = payload / p_cores
        steps = p_cores - 1
        waves = tuple(
            RingWave(count=p_cores, nbytes=seg, hops=1) for _ in range(2 * steps)
        )
        region = RingRegion(name=f"{plan.name}:reduce_scatter", waves=waves)
        # reduce-scatter phase combines; all-gather phase only moves.
        wall = sum(
            _hop_wall(seg, 1, plan.cut_through, combine=True) for _ in range(steps)
        )
        wall += sum(
            _hop_wall(seg, 1, plan.cut_through, combine=False) for _ in range(steps)
        )
        endpoint = "head-split (== O-proj K-split, zero relayout)"

    elif variant == "recursive":
        log2p = int(math.log2(p_cores))
        assert 2**log2p == p_cores, "recursive fold assumes cores is a power of two"
        # round i: partner at distance 2^i, half the running payload.
        rs_rounds = [(payload / 2 ** (i + 1), 2**i) for i in range(log2p)]
        ag_rounds = [(payload / 2 ** (log2p - i), 2**i) for i in range(log2p)]
        waves = tuple(
            RingWave(count=p_cores, nbytes=nb, hops=h) for nb, h in rs_rounds + ag_rounds
        )
        region = RingRegion(name=f"{plan.name}:recursive", waves=waves)
        wall = sum(
            _hop_wall(nb, h, plan.cut_through, combine=True) for nb, h in rs_rounds
        )
        wall += sum(
            _hop_wall(nb, h, plan.cut_through, combine=False) for nb, h in ag_rounds
        )
        endpoint = "head-split"

    else:
        raise ValueError(f"unknown fold variant: {variant!r}")

    return {
        "name": plan.name,
        "variant": variant,
        "payload_KiB": round(payload / 1024, 2),
        "occupancy_us": round(ring_occupancy(region), 2),
        "wall_us": round(wall, 2),
        "hops": len(region.waves),
        "endpoint": endpoint,
        "note": "L-independent, on every layer's critical path; SFP combine per hop",
    }


# --- (c) Contention ---------------------------------------------------------
def serialize(*priced: dict) -> dict:
    """v1 contention: ring-heavy regions cannot share links, so overlapping ones
    take turns -> walls and occupancies add. Each argument must be a normalized
    ``{name, occupancy_us, wall_us}`` dict (price_fold returns one directly; for a
    carousel pick wall_serial_us or wall_overlap_us as its ``wall_us``). Coarse
    (real duplex/pipelining reclaims some) but never optimistic -- the safe bias
    for a planner."""
    return {
        "name": "+".join(p["name"] for p in priced),
        "occupancy_us": round(sum(p["occupancy_us"] for p in priced), 2),
        "wall_us": round(sum(p["wall_us"] for p in priced), 2),
    }


def ring_bound(occupancy_us: float, window_us: float) -> bool:
    """Is the ring the bottleneck over a program window? True when demand exceeds
    what the ring can retire (RING_SUPPLY link-us per wall-us) in that window."""
    return occupancy_us > RING_SUPPLY * window_us


# --- Demo -------------------------------------------------------------------
def _main() -> None:
    # A Granite-shaped prefill matmul, 512-token block, fp16.
    print("== weight carousel (S=512, K=4096, N=14336) ==")
    for r in (32, 4, 1):
        row = price_carousel(
            CarouselPlan("mlp_up", S=512, K=4096, N=14336, radius=r, slab_k=256)
        )
        print(
            f" r={row['radius']:>2} steps={row['steps']:>2} "
            f"occ={row['occupancy_us']:>8}us floor={row['occupancy_floor_us']:>6}us "
            f"wall_serial={row['wall_serial_us']:>7}us "
            f"wall_overlap={row['wall_overlap_us']:>6}us "
            f"dram_saved={row['dram_saved_x']}x lx_fits={row['lx_fits']}"
        )
    row = price_carousel(CarouselPlan("mlp_up", S=512, K=4096, N=14336, radius=1))
    print(f" gate_rho={row['gate_rho_gbps']}GB/s transport_hidden={row['transport_hidden']}")

    print("\n== LSE ring-fold (P=32, heads=8, d=128) ==")
    plan = FoldPlan("kv_merge", heads=8, head_dim=128)
    priced = [price_fold(v, plan) for v in ("linear", "reduce_scatter", "recursive")]
    for p in priced:
        print(
            f" {p['variant']:<14} occ={p['occupancy_us']:>7}us "
            f"wall={p['wall_us']:>7}us hops={p['hops']:>3}  {p['endpoint']}"
        )
    ct = price_fold("recursive", FoldPlan("kv_merge", heads=8, head_dim=128, cut_through=True))
    print(f" recursive(cut_through) wall={ct['wall_us']}us  (vs store-forward above)")

    print("\n== contention: carousel + fold in one window (serialize) ==")
    car = price_carousel(CarouselPlan("mlp_up", S=512, K=4096, N=14336, radius=1))
    car_n = {"name": car["name"], "occupancy_us": car["occupancy_us"],
             "wall_us": car["wall_serial_us"]}
    both = serialize(car_n, priced[1])
    print(f" {both['name']}: occ={both['occupancy_us']}us wall={both['wall_us']}us")
    print(f" ring_bound over a 800us window? {ring_bound(both['occupancy_us'], 800)}")


if __name__ == "__main__":
    _main()


# --- Assumptions and known inaccuracies -------------------------------------
# Read these before trusting a number. This is a first-iteration estimate.
#
# What it adds over the byte-level model (comm_cost_model.py):
#   - LINK-TIME occupancy: a transfer costs bytes x HOPS, not just bytes, so the
#     ring is a finite resource (RING_SUPPLY link-us per wall-us) the planner can
#     budget, not a bandwidth every move sees in full.
#   - A per-hop latency lambda, which decides fold topology at small payloads.
#   - A duplex factor in the ring supply and the occupancy floor.
#
# Known inaccuracies:
#   - lambda (HOP_LATENCY_US) is a PLACEHOLDER. It sets the linear-vs-recursive
#     fold crossover; the ranking is only as good as this number. NEEDS A DEVICE
#     MEASUREMENT.
#   - Contention is v1 additive: overlapping ring regions serialize completely.
#     Real duplex and compute/transfer pipelining reclaim some of that; the model
#     is pessimistic, never optimistic (the safe bias).
#   - The carousel wall_overlap assumes the rotation hides under the matmul. Today
#     the relayout is a SEQUENTIAL BARRIER for matmul consumers (the input-fetch
#     overlap hook is gated to Conv2D in the backend), so wall_serial is what runs
#     until that gate is widened. That is a BACKEND (deeptools) change, not a
#     frontend one.
#   - The fold's LSE-combine is priced as an SFP pass per hop at SFP_GBPS (rough).
#     No single-AIU fused move-then-reduce primitive exists, so the fold must be
#     composed as ring-move + a separate SFP op per hop; the per-hop combine term
#     reflects that composition.
#   - recursive-fold payloads and distances are the standard halving/doubling
#     approximation; cut_through defaults False (distance costs latency). Whether
#     the fabric is cut-through is UNVERIFIED.
#   - The compute gate uses a ROUGH FP16_TFLOPS_AGG; it only decides
#     transport_hidden and touches no occupancy number. NEEDS THE DATASHEET.
#   - No LX-capacity legality beyond the carousel double-buffer footprint check; a
#     fold partial that overflows LX is not caught here.
#   - Occupancy assumes uniform per-link pricing; a real reshard's STCDPOpLx pieces
#     may land unevenly across links, so occupancy is a mean, not a hot-link max.
