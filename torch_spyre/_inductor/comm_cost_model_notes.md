# Communication cost model — notes

First iteration. Prices the two ways to bridge a producer/consumer work-division
mismatch — spill through HBM vs move on-chip over the ring — so the planner can
choose the cheaper one instead of always spilling.

Run: `python3 torch_spyre/_inductor/comm_cost_model.py`

## What it is

Bytes moved divided by the bandwidth of the fabric that carries them.

- **spill** = producer writes the tensor once + every consumer reads what it
  needs. Replication (fanout > 1) is paid on the read side.
- **move** = the same delivered bytes over the ring; a form change adds a second
  on-chip pass to reshuffle the stick layout.

## Why it exists

A matmul-only cost model prices ops in isolation and cannot see the cost of a
producer/consumer ownership mismatch — the cross-op seam. That seam is exactly
where a spill hides. This model makes it a number.

## What the first run shows

| edge | spill | move | cheaper |
|---|---:|---:|---|
| scatter (proj → pointwise) | 464 us | 158 us | move 2.94× |
| all-gather (@V operand, fanout 8) | 334 us | 202 us | move 1.65× |
| restickify (scores → @V) | 148 us | 101 us | move 1.47× |
| reduce (softmax Lk) | — | — | flagged, not a choice |

The scatter case ranks the on-chip move ~2.9× cheaper than its spill, consistent
with removing scatter spills being a win. The all-gather win is smaller because
naive full replication pays the copies on the ring too — which is the argument
for multicast / loop-scoped fetch (deliver fewer bytes) over full materialization.

## Known inaccuracies (first iteration)

- No ring contention or hop distance — a move is priced at peak ring bandwidth
  regardless of how many transfers share a link.
- No LX-capacity gating — if the moved piece does not fit in LX the move is
  illegal, but the model still prices it. Capacity is a separate concern.
- Multicast forwarding is not modeled, so broadcast/all-gather is over-costed:
  the ring can forward one copy along a chain instead of sending fanout copies.
- Realized HBM bandwidth under array contention can be well below the bus ceiling
  used here, which would make spills look cheaper than they are in practice.
- Latency and compute/movement overlap are ignored; this is a throughput estimate.
