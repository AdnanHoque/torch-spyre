# Projected On-Chip Speedup — MoE Routing Data Movement

Offline projection (2026-05-24). Uses the **empirical anchor** from
`CoreToCoreDataMovementRecipe.md` (§0, §9). No device run. Every number is labelled
**MEASURED-anchor** (taken from the device-proven benchmark) or **PROJECTED**
(derived from the anchor for MoE shapes). Inferences flagged **[INFER]**.

---

## 1. The empirical anchor (MEASURED-anchor)

From the recipe (§0 / §9, proven on real Spyre hardware):

- **~0.029 ms saved per MB** of eliminated same-stick handoff tensor.
- **minus ~0.005 ms STCDP setup** (fixed per move).
- **net-positive above ~1 MB** of handoff tensor.

So for a handoff that moves `B` MB of activation, the on-chip saving is:

```
saving_ms(B) = 0.029 * B - 0.005        (B in MB; net-positive when B > ~0.17 MB,
                                          comfortably positive at MoE MB-scale)
```

This is the saving on the *handoff itself* (the HBM round-trip eliminated by keeping
the move on the RIU ring / in LX). The recipe's measured end-to-end speedups
(1.13×–1.22× on the `(a+b.t+c.t)@d` graph, §9) are the *blended* result once the
handoff saving is weighted by the rest of the graph's compute — see §4.

---

## 2. Handoff bytes per MoE routing op (from eligibility.md §6)

fp16, `cap_fac = 1.0`. Dispatch buffer = `EC*H*2`; combine output = `T*H*2`. With
top-k=1, `EC ≈ T`; with top-k=2, the dispatch buffer doubles.

| shape (E, T, H, topk) | dispatch MB | combine MB |
|---|---|---|
| (8,  2048, 2048, 1) | 8.0  | 8.0  |
| (8,  2048, 4096, 1) | 16.0 | 16.0 |
| (8,  2048, 2048, 2) | 16.0 | 8.0  |
| (64, 4096, 4096, 1) | 32.0 | 32.0 |
| (64, 4096, 4096, 2) | 64.0 | 32.0 |

(MB = bytes / 2^20; e.g. 2048*2048*2 = 8,388,608 B = 8.0 MB.)

---

## 3. Projected per-op handoff saving (PROJECTED)

Applying `saving_ms(B) = 0.029*B - 0.005` to the bytes above. This is the latency
removed from each routing handoff by replacing its HBM round-trip with the proven
on-chip same-stick move (CONDITIONAL on the dynamic-`memId` capability of
`eligibility.md` §4 existing — these moves are same-stick, so the anchor applies once
addressing is dynamic).

| shape (E,T,H,topk) | dispatch saving ms | combine saving ms |
|---|---|---|
| (8,  2048, 2048, 1) | 0.029*8  - 0.005 = **0.227** | **0.227** |
| (8,  2048, 4096, 1) | 0.029*16 - 0.005 = **0.459** | **0.459** |
| (8,  2048, 2048, 2) | 0.029*16 - 0.005 = **0.459** | **0.227** |
| (64, 4096, 4096, 1) | 0.029*32 - 0.005 = **0.923** | **0.923** |
| (64, 4096, 4096, 2) | 0.029*64 - 0.005 = **1.851** | **0.923** |

All are **>> the ~0.005 ms setup**, so every MoE routing shape is firmly
net-positive — the opposite of the S=512 micro-case where the move was sub-stick and
regressed (recipe §9 i). MoE routing buffers are MB-scale and always above threshold.

---

## 4. Translating saving to speedup — MoE is bandwidth-bound (PROJECTED)

The *relative* speedup depends on what the saving is a fraction of. The recipe's key
teaching (§9 ii, §12.3): the on-chip handoff matters most in **bandwidth-bound**
regimes, because there the eliminated HBM traffic is a large share of total time; in
compute-bound prefill the O(N^3) matmul dwarfs the O(N^2) handoff.

**Why MoE routing is the bandwidth-bound regime where on-chip wins most:** dispatch
and combine are *pure data movement* — they have **no compute to amortize against**
(the matmul-permutation realization is a sparse one-hot GEMM that is itself
bandwidth-bound, not a dense FLOP-heavy matmul). So for the *isolated routing op*, the
HBM round-trip is essentially **100% of the cost**, and eliminating it is close to the
full saving rather than a small blended fraction.

Estimate the isolated-op baseline from HBM bandwidth (166 GB/s shared, recipe §1). A
routing op reads `B` MB from HBM and writes `B` MB back (round-trip = 2B):

```
baseline_ms(B) ≈ 2*B*2^20 / (166e9) * 1e3      [HBM round-trip lower bound]
```

| shape (E,T,H,topk) | op | B MB | HBM baseline ms (PROJECTED) | saving ms (PROJECTED) | implied speedup |
|---|---|---|---|---|---|
| (8, 2048, 2048, 1)  | dispatch | 8  | ~0.101 | 0.227 | move dominated by ring vs HBM — see note |
| (8, 2048, 4096, 1)  | dispatch | 16 | ~0.202 | 0.459 | " |
| (64,4096, 4096, 1)  | dispatch | 32 | ~0.404 | 0.923 | " |

**Note / honest caveat:** the anchor's 0.029 ms/MB is *larger* than the 2-way HBM
round-trip lower bound (0.0126 ms/MB) — i.e. the measured per-MB saving exceeds the
naive HBM-only model. The anchor was measured on the `(a+b.t+c.t)@d` graph where the
handoff also carries restickify/scheduling overhead that the on-chip path removes, so
0.029 ms/MB folds in more than pure HBM bytes. For a *pure* routing data-move op the
true on-chip speedup is bounded by the RIU-ring-vs-HBM bandwidth ratio (~64×, recipe
§1) but realized speedup is gated by setup and the fact that the move still has to
traverse the ring once. **The defensible claim is therefore directional, not a precise
multiple:** routing handoffs are MB-scale, bandwidth-bound, same-stick, and far above
the net-positive threshold — the single most favorable Tier-1 profile of any MoE
piece — and the per-MB saving grows linearly with H, T and top-k. I do **not** assert
a specific X× without a device run; the orchestrator's baseline (the matmul
`BENCH ... median_ms`) supplies the denominator and `saving_ms` the numerator for an
A/B once the dynamic-`memId` move is available.

---

## 5. A/B framing for the orchestrator

The microbench prints, per op, a `BENCH moe_routing op=... median_ms=...` baseline
(the matmul-permutation realization running on device today). The on-chip A/B is:

```
projected_onchip_ms ≈ baseline_median_ms - saving_ms(B)        (B from §2)
projected_speedup   ≈ baseline_median_ms / projected_onchip_ms
```

Plug the orchestrator's measured `median_ms` (MEASURED) into this with the PROJECTED
`saving_ms` from §3. This keeps MEASURED and PROJECTED cleanly separated: the
baseline is real, the saving is the anchor-derived estimate.

---

## 6. Other top pure-data-movement Tier-1 candidates ("whatever benefits most")

Beyond MoE routing, the pure-data-movement ops across real workloads that are top
Tier-1 candidates — same-stick (so they avoid the blocked transpose) and crossing an
HBM SDSC boundary — from `real_edge_analysis.md` and the recipe §12:

1. **Attention score → softmax** (`batchmatmul(QK^T) → max/sub`), stick `['out']` —
   same-stick AND genuinely cross-core (producer shards `{mb:32}`, consumer `{x:32}`),
   AND **statically addressable** (the placement is compile-time, not router-driven).
   Recurs in *every attention layer of every model*. This is the recipe's named
   best-first-demo target (§12.4 / real_edge_analysis §best target) — strictly ahead
   of MoE routing because it has no dynamic-addressing gap.

2. **MLP / linear output → elementwise scale** (granite `batchmatmul → mul`), stick
   `['out']` — same-stick, cross-core, static. The non-attention second data point.

3. **Residual adds** (`add → mul`, `mul → add` chains in the RMSNorm tail, e.g.
   granite `[5]mul→[6]add`, `[6]add→[7]mul`), stick `['out']`, **same-shard** →
   degenerate same-core LX-resident copy: pure HBM-elimination, *no* ring needed.
   These are the simplest possible win (recipe §12.4 lowest-friction), present in every
   transformer block. They are pure data movement (an add fed by an LX-resident
   producer) and need only items 1+2+4 of the productionization list.

4. **Same-stick reshapes / views that stay same-orientation** — `identity` SDSCs from
   `_unsafe_view`/`clone`/`expand`/`unsqueeze` that do not flip the stick dim (e.g.
   SDPA `[0]identity→[1]mul`, attn `[2]id→[3]bmm`). When the stick dim is preserved
   these are pure same-stick re-placements; when they flip it (the reduction-reshape
   `out↔x` in RMSNorm) they are layout-changing and blocked.

**Ranking for "benefits most" today:** (1) attention score→softmax and (3) residual
adds are the top *immediately-addressable* candidates (same-stick + static placement).
MoE dispatch/combine are the **highest-byte** candidates (MB-scale, §2) and the best
*bandwidth-bound* fit, but they trail on addressability (dynamic `memId`, §4). So the
"whatever benefits most" answer is two-tiered: **biggest absolute bytes = MoE
routing; first realizable wins = attention score→softmax + residual adds.**

---

## 7. Bottom line

- **MoE dispatch/combine are same-stick** (the key verdict) — they avoid the
  Compute-CB transpose wall, cross an HBM SDSC boundary, and move MB-scale activation
  far above the ~1 MB net-positive threshold → the most favorable *bandwidth-bound*
  Tier-1 profile of any MoE piece.
- **Projected per-op handoff saving** ranges **~0.23 ms** (8/2048/2048) to **~1.85 ms**
  (64/4096/4096, top-k=2), all PROJECTED from the MEASURED ~0.029 ms/MB anchor.
- The one gap is **dynamic/index-driven addressing** (routing placement is a runtime
  router output, `STCDPOpLx` places statically) — a strictly smaller gap than the
  layout-changing transpose frontier, on the *same* proven data path.
- For *immediate* Tier-1 wins, attention score→softmax and residual adds lead (no
  dynamic-addressing gap); MoE routing leads on *absolute bytes saved*.
