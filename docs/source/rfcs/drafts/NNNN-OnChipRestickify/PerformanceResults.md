# Performance results — on-chip core-to-core data movement

Consolidated results for the on-chip restickify RFC. Every row is labeled
**MEASURED** (run on device), **PROJECTED** (derived from the measured anchor),
or **TODO** (pending a device run the orchestrator will fill). Projected rows are
summarized from the per-workload `reproduction/workloads/*/projection.md`.

## Empirical anchor (MEASURED)

Same-core on-chip handoff saves **~0.029 ms per MB** of eliminated same-stick
handoff tensor, minus **~0.005 ms** STCDP setup; net-positive above ~1 MB
handoff. This anchor is the per-MB rate used by every PROJECTED row below.

## add-mm proof: `(a + b.t() + c.t()) @ d`, fp16, 32 cores, median of 60 iters

| size | baseline HBM ms | same-core on-chip ms (speedup) | cross-core round-trip ms | max_err | label |
|---|---|---|---|---|---|
| 512 | 0.0957 | 0.1006 (0.95x regression) | 0.1067 | 0.004883 | MEASURED |
| 1024 | 0.3202 | 0.2629 (1.22x) | 0.2768 | 0.007812 | MEASURED |
| 2048 | 1.5570 | 1.3127 (1.19x) | 1.3294 | 0.013672 | MEASURED |
| 4096 | 7.9100 | 7.0151 (1.13x) | 7.1389 (broken, fixed bases) | base 0.013672; round-trip 6.15 | MEASURED |

2048 two-rep stability (MEASURED): baseline 1.534/1.541, same-core 1.321/1.306,
round-trip 1.339/1.340. Relative speedup peaks mid-range (1.22x @1024) and tapers
as matmul O(N^3) dwarfs the O(N^2) handoff; 512 regresses (sub-stick per-core
slice).

## Per-size cross-core fix (allocator bases) — VALUE-CORRECT on device (MEASURED)

| size | max_err | label |
|---|---|---|
| 512 | 0.004883 | MEASURED |
| 1024 | 0.007812 | MEASURED |
| 2048 | 0.013672 | MEASURED |
| 4096 | round-trip 3-region NOFIT (3x1 MB > 2 MB/core LX) | MEASURED limit |

The 4096 NOFIT is a proof-construct limit; a production single cross-core move is
2 regions, not 3.

## Compiler-driven E2E (add-mm 2048, the milestone) — MEASURED

| flag | max_err | compiler_emitted_mixed | opFuncsUsed_ |
|---|---|---|---|
| OFF | 0.013672 | False | (baseline) |
| ON (`SPYRE_ONCHIP_HANDOFF_REALIZE=1`) | 0.013672 | True | `['STCDPOpLx']` |

torch.compile emitted the mixed bundle; patched dxp accepted it (gate exercised);
device value-correct; negative control clean. No splice, no redirect.

## Cross-core ring signature (MEASURED)

- add-mm 2048 round-trip: all 32 cores emit `L3_LDU`+`L3_STU` to mirror core
  `31-i`; degenerate same-core emits 0.
- Attention QK^T->softmax edge: 64 `L3_LDU` + 64 `L3_STU`, mirror `31-i`.

## Attention A/B

| config | baseline HBM ms (min) | on-chip ms (min) | max_err | label |
|---|---|---|---|---|
| seq=64, bh=32, hd=128 | 0.1811 (0.1754) | 0.1832 (0.1762) | 0.000214 | MEASURED, neutral/slight regression |
| seq=512, bh=32, hd=128 | 2.5998 (2.5502) | — | 0.000092 | baseline MEASURED; on-chip TODO |

seq=64: score handoff ~256 KB is below the ~1 MB floor and is a 3-region
round-trip construct. Value-correct cross-core on a REAL attention edge (no
Compute-CB). seq=512 on-chip A/B is **TODO(pending device run)**.

## Block baselines

| workload | baseline ms (min) | max_err | label |
|---|---|---|---|
| MoE expert FFN (E=8, H=2048, INTER=8192, T=128) | 125.8477 (120.8593) | 0.044922 (high) | MEASURED, on-chip = PROJECTED |
| transformer block | first run failed to compile (fp32 mean); fp16 fix re-run | — | TODO(pending re-run) |
| MoE full block | — | — | TODO(pending, compiling) |

## Device findings

- Spyre does NOT support fp32 `mean` -> RMSNorm must stay fp16 (granite's does).
- SDPA decomposes on Spyre into separate bmm/softmax SDSCs; the score->softmax
  edge is a real same-stick cross-core handoff.
- Two on-chip frontiers: (A) layout-changing transpose (Compute-CB fault); (B)
  dynamic addressing (MoE routing same-stick but needs runtime memId). Tier-1
  proven = same-stick + static addressing.

## Projected speedups (PROJECTED, from `reproduction/workloads/*/projection.md`)

| workload | regime | projected whole-block speedup | source |
|---|---|---|---|
| transformer block | hidden 2048 sweet spot | ~10-15% (~5-20% bracket) | `transformer_block/projection.md` |
| transformer block | hidden 4096 | ~3-12% | `transformer_block/projection.md` |
| MoE block | mid H 1k-2k, 128-512 tok | 1.11x .. 1.39x | `moe_block/projection.md` |
| MoE block | prefill 4096/14336 | 1.06x .. 1.10x | `moe_block/projection.md` |
| MoE block | decode 1 token | ~0.95x, gate OFF | `moe_block/projection.md` |
| MoE routing | dispatch/combine, MB-scale | +0.23 to +1.85 ms/op saved | `moe_routing/projection.md` |
| mamba2 | prefill in-proj 4.2 MB | +0.117 ms; decode net-neg | `mamba2/projection.md` |
| attention | prefill seq=512 (16 MB) | +0.459 ms/layer | `attention/projection.md` |
| attention | prefill seq=2048 (256 MB) | +7.42 ms/layer | `attention/projection.md` |
| attention | decode seq=1 | sub-MB, gate to HBM | `attention/projection.md` |

~63% of MoE activation bytes ride eligible same-stick edges; ~37% on blocked
layout-changing edges. Projections cap the anchor at the physical HBM round-trip
ceiling (~0.012 ms/MB). All workload projections exclude blocked layout-changing
handoffs.

## TODO(pending device run) — orchestrator fills

- Attention seq=512 on-chip A/B (baseline 2.5998 ms measured).
- Transformer block baseline re-run (fp16 mean fix).
- MoE full block baseline.
- MoE FFN, transformer, MoE-block, mamba2, routing on-chip A/B numbers vs
  projections above.
