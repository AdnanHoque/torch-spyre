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

"""First-iteration cost model for on-chip communication.

When a producer op leaves a tensor on the cores under one work-division and the
consumer wants it under a different one, there are two ways to bridge the gap:

  spill  - producer writes the tensor to HBM, consumer reads it back
  move   - keep it on-chip and send it core-to-core over the ring

The planner should pick the cheaper one instead of always spilling. This module
prices both, per communication class, so that choice becomes visible. It is a
plain-arithmetic estimate: bytes moved divided by the bandwidth of the fabric
that carries them. It exposes the cross-op seam a matmul-only cost model cannot.
"""

from __future__ import annotations

import dataclasses

# Bandwidths in GB/s. HBM is the measured bus ceiling for a mixed read+write
# stream (a spill is both); the ring is the on-chip fabric per direction.
HBM_RW_GBPS = 113.0
RING_GBPS = 166.0

# A form-changing move (restickify) also reshuffles the stick layout, which the
# pointwise/SFP path does at roughly the ring rate again on top of the move.
RESTICKIFY_FORM_GBPS = 166.0


@dataclasses.dataclass(frozen=True)
class CommEdge:
    """One producer->consumer tensor handoff that needs bridging.

    tensor_bytes:  size of the logical tensor.
    fanout:        how many consumer cores need each producer piece.
                   1 = scatter/gather (each piece has one destination),
                   >1 = broadcast/all-gather (each piece is replicated).
    form_change:   True if the stick layout changes, not just the owner core.
    reduce:        True if consumers combine producer pieces arithmetically.
    """

    name: str
    tensor_bytes: int
    fanout: int = 1
    form_change: bool = False
    reduce: bool = False


def _us(num_bytes: float, gbps: float) -> float:
    """Microseconds to move num_bytes at gbps."""
    return num_bytes / (gbps * 1e3)


def spill_cost_us(edge: CommEdge) -> float:
    """Producer writes the tensor once; every consumer reads what it needs.

    Replication (fanout>1) is paid on the read side: each of the fanout copies
    is a separate HBM read.
    """
    written = edge.tensor_bytes
    read = edge.tensor_bytes * edge.fanout
    return _us(written + read, HBM_RW_GBPS)


def move_cost_us(edge: CommEdge) -> float:
    """Send the tensor over the ring; replicated pieces cross once per copy.

    A form change adds a second on-chip pass to reshuffle the stick layout.
    """
    delivered = edge.tensor_bytes * edge.fanout
    cost = _us(delivered, RING_GBPS)
    if edge.form_change:
        cost += _us(edge.tensor_bytes, RESTICKIFY_FORM_GBPS)
    return cost


def price(edge: CommEdge) -> dict:
    """Compare spill vs on-chip move. Reduce edges are flagged, not priced:
    combining partials needs arithmetic the ring cannot do, so it is not a
    move-vs-spill choice at all."""
    spill = spill_cost_us(edge)
    move = move_cost_us(edge)
    return {
        "name": edge.name,
        "spill_us": round(spill, 2),
        "move_us": round(move, 2),
        "cheaper": "move" if move < spill else "spill",
        "speedup": round(spill / move, 2) if move else float("inf"),
        "note": "reduce: needs arithmetic, not a move/spill choice" if edge.reduce else "",
    }


# Representative edges from the Granite/attention spill taxonomy. Sizes are a
# 512-token prefill block at fp16 (2 B/element).
_MiB = 1024 * 1024
TAXONOMY_EDGES = [
    # 1:1 ownership change: projection output -> pointwise SwiGLU chain.
    CommEdge("granite_scatter_proj_to_pointwise", tensor_bytes=25 * _MiB, fanout=1),
    # value-side @V operand: each co-splitting core needs the whole Lk slice.
    CommEdge("attn_allgather_value_operand", tensor_bytes=4 * _MiB, fanout=8),
    # softmax scores -> @V: Lk flips from stick dim to contraction dim.
    CommEdge("attn_restickify_scores_to_v", tensor_bytes=8 * _MiB, fanout=1, form_change=True),
    # softmax Lk collapse: many partials combined -> not a move/spill choice.
    CommEdge("attn_softmax_lk_reduce", tensor_bytes=1 * _MiB, fanout=1, reduce=True),
]


def _main() -> None:
    header = f"{'edge':<38}{'spill_us':>10}{'move_us':>10}{'cheaper':>9}{'x':>7}"
    print(header)
    print("-" * len(header))
    for edge in TAXONOMY_EDGES:
        r = price(edge)
        line = (
            f"{r['name']:<38}{r['spill_us']:>10}{r['move_us']:>10}"
            f"{r['cheaper']:>9}{r['speedup']:>7}"
        )
        print(line + (f"   {r['note']}" if r["note"] else ""))


if __name__ == "__main__":
    _main()
