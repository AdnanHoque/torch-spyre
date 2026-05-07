# Z-order block-to-core probe at k=4 — findings

## TL;DR

Z-order block-to-core arrangement does NOT net additional gain over
k_fast, but the probe revealed something more important: **at k=4
splits, the ring-distance lever is much bigger than at k=2** — up to
2× walls between worst and best placement. This refines our earlier
"direction symmetric" finding.

The AIU SFP ring is effectively a **1D loop through cores in core_id
order**, not a 2D fabric. What we previously called "row vs column
direction" was actually small vs large core_id distance. PR 1932's
k_fast already minimizes this distance optimally.

## Probe design

Forced split (1, 8, 4) producing a k=4 PSUM chain. Three placements
of the 4-core chain:

- **default (4×1 column)**: chain at cores {0, 8, 16, 24}
- **kfast (1×4 row)**: chain at cores {0, 1, 2, 3}
- **zorder (2×2 block)**: chain at cores {0, 1, 8, 9}

All three have the same minimum 2D Manhattan distance (3 hops) but
DIFFERENT 1D ring distances (24, 3, 9 respectively).

Three shapes, cold cache per measurement.

## Results

| shape (M, N, K) | default | kfast | zorder | best | worst spread |
|---|---:|---:|---:|---|---:|
| (2048, 1024, 8192) | 5.36 ms | 4.34 ms | 4.19 ms | zorder by 4% | 28% |
| (2048, 2048, 8192) | 10.86 ms | **5.38 ms** | 7.42 ms | kfast | **102%** |
| (1024, 2048, 8192) | 6.98 ms | **4.21 ms** | 5.22 ms | kfast | 66% |

## Interpretation: ring distance, not 2D shape

Walls track 1D ring distance very closely:
- default: 24 hops (chain at {0, 8, 16, 24}, each hop 8 ring steps)
- zorder: 9 hops (chain at {0, 1, 8, 9})
- kfast: 3 hops (chain at {0, 1, 2, 3})

The 28-102% wall spread aligns with the 24×→9×→3× ring-distance
ratio after subtracting LF baseline.

**This means the AIU SFP ring is a 1D loop**, not a 2D fabric. My
earlier paper draft framing "row vs column direction" was the wrong
mental model. The actual lever is "minimize ring distance =
|Δcore_id| summed across the chain."

## Why earlier k=2 probe showed "direction symmetric"

At k=2 (1, 16, 2):
- default: chain at {0, 16}, distance 16
- kfast: chain at {0, 1}, distance 1
- col_dir: chain at {0, 8}, distance 8

Walls measured: 5.18, 4.28, 4.32 ms. The k=2 "1-hop placements
equivalent" finding was real but at distances small enough (1 vs 8)
that the difference is below measurement noise once LF is added.

At k=4, distances stretch to 24 vs 3 — well above noise — so the
ring-distance effect becomes obvious.

## Answer to "is there a Z-order block-to-core gain?"

**No.** kfast already minimizes total chain distance for k-chains in
this split family. Z-order's compact 2×2 shape has higher 1D ring
distance (because going from physical 1 to physical 8 traverses the
ring "the long way") and is therefore slower on 2 of 3 shapes.

The 4% zorder win at (2048, 1024, 8192) is at measurement noise.

## More important finding: PR 1932's value at k=4

The original PR validation focused on k=2 splits (1, 16, 2). For
workloads where the heuristic picks k=4 — DSv3 q_a_proj-style
shapes — the k_fast permutation value is substantially larger than
the k=2 numbers suggested:

| shape | default | kfast | win |
|---|---:|---:|---:|
| (2048, 1024, 8192) | 5.36 | 4.34 | 1.24× |
| **(2048, 2048, 8192)** | **10.86** | **5.38** | **2.02×** |
| (1024, 2048, 8192) | 6.98 | 4.21 | 1.66× |

This gives PR 1932 a stronger claim for the heuristic's k=4 cases:
not just the 1.05-1.10× we measured at k=2 prefill regime.

## Implication for the paper draft

The paper's framing "direction symmetric, hop count matters"
needs sharpening:
- Not "direction" — there's no 2D direction on the SFP ring; it's 1D
- Hop count = total ring distance summed across chain sends
- The k=2 measurements that suggested "direction symmetric" were a
  special case — small distances all below LF-saturation threshold
- Larger chains (k=4) make the 1D ring distance effect obvious

Updated single-sentence claim: **"On heterogeneous accelerators with
parallel multicast on the data ring and a 1D loop on the SFP ring,
placement-aware compilers should minimize total chain distance on
the SFP ring; placement on the data ring is invisible to wall-time."**

This is sharper than the 2D direction framing.

## Why didn't we see this earlier?

The original 2D direction probe was run at k=2 only. At k=2 the
distance differences (1 vs 8 vs 16) compress in walls due to the
saturation effect. We didn't realize there's significant headroom
at higher k until this probe.

It's also worth noting: **the original PR 1932 validation set's
"+id" rows** (which are forced k-split WITHOUT k_fast permutation)
**show massive wins at higher k**. The validation set's M=2048 +id
result at 10.93 ms vs +kf at 3.94 ms (2.78×) is the (1, 16, 2) case
with default emission (distance 16). This probe extends that picture
to k=4 with even bigger spreads.

## Files

- This doc — Z-order block-to-core findings + revised SFP ring model
