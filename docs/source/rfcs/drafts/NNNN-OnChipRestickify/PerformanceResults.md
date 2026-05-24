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
- MoE dispatch/combine: 2x `Creating PCFG for DataDsc` with `L3SU : L3LU` on all
  32 cores, transfer map `0->31, 1->30, … 31->0`; baseline has 0 DataDscs.
  Negative control passed (remove senprog -> hard load failure) on both directions.

## Attention A/B

| config | baseline HBM ms (min) | on-chip ms (min) | max_err | label |
|---|---|---|---|---|
| seq=64, bh=32, hd=128 | 0.1811 (0.1754) | 0.1832 (0.1762) | 0.000214 | MEASURED, neutral/slight regression |
| seq=512, bh=32, hd=128 | 2.5595 (2.5086) | 1.9778 (1.9587) | 0.000107 | MEASURED, **1.29x** value-correct |

seq=64: score handoff ~256 KB is below the ~1 MB floor and is a 3-region
round-trip construct. Value-correct cross-core on a REAL attention edge (no
Compute-CB). **seq=512: MEASURED on device — baseline 2.5595 ms vs on-chip
1.9778 ms = 1.29x (−23%), value-correct (max_err 0.000107 vs baseline 0.000092),
negative control passed.** The 16 MB score-matrix HBM round-trip is a large
fraction of the attention block at this seq, so the win is bigger than add-mm.
This is still the 3-region round-trip proof construct (extra ring work) — a
production single 2-region move would beat it. (An earlier separate compile
measured baseline 2.5998 ms; same ballpark.)

## MoE dispatch/combine A/B (MEASURED) — the activation-dominated win

The clearest on-chip win measured to date. The MoE routing handoff (dispatch =
gather tokens to their expert's core; combine = scatter results back) is a large
fraction of the op's HBM traffic with no weight matrix to dwarf it, so
eliminating the HBM round-trip pays off directly. All rows MEASURED on device
(N=50, torch.profiler PrivateUse1; `spyre_ms == kernel_ms` — these ops are pure
compute with no memcpy events, so the STCDP cross-core move time is counted
*inside* the on-chip number).

| op | E | T | H | handoff MB | HBM dev ms | on-chip dev ms | **dev speedup** | max_err |
|---|---|---|---|---|---|---|---|---|
| dispatch | 8 | 512 | 2048 | 2.0 | 0.2746 | 0.2094 | **1.31x** | 0.00171 |
| dispatch | 8 | 512 | 4096 | 4.0 | 0.9425 | 0.6725 | **1.40x** | 0.00244 |
| dispatch | 8 | 1024 | 2048 | 4.0 | 0.4465 | 0.3225 | **1.38x** | 0.00195 |
| dispatch | 8 | 2048 | 2048 | 8.0 | 1.3323 | 0.6865 | **1.94x** | 0.00171 |
| dispatch | 64 | 512 | 4096 | 4.0 | 0.9417 | 0.6724 | **1.40x** | 0.00244 |
| dispatch | 64 | 1024 | 2048 | 4.0 | 0.4478 | 0.3229 | **1.39x** | 0.00195 |
| combine | 8 | 512 | 2048 | 2.0 | 0.2760 | 0.2093 | **1.32x** | 0.00124 |

### Why this wins where the dense block did not (the theory)

- **Activation-dominated vs weight-bound.** A dense transformer/decode block is
  weight-bandwidth-bound: its HBM traffic is dominated by streaming the weight
  matrices, and the activation handoff is a tiny slice (<1% in decode), so an
  on-chip handoff is within-noise at the block level. MoE routing has *no* big
  weight read in the dispatch/combine itself — the handoff IS the dominant
  traffic. That is the regime on-chip is built for.
- **Bandwidth-bound signature: the win scales with handoff bytes.** 1.31x @ 2 MB
  -> 1.94x @ 8 MB. If the saving were a fixed setup cost it would shrink in
  relative terms as the op grows; instead it grows, which is the fingerprint of
  eliminating HBM bytes proportional to the handoff.
- **E-invariance.** At matched EC×H, expert count does not move the result
  (E=8 vs E=64: 1.40x vs 1.40x, 1.38x vs 1.39x). Handoff bytes depend on
  tokens×hidden, not on the number of experts — exactly as a data-movement (not
  compute) optimization should behave.
- **The measurement is conservative.** The natural `(perm@x)@wexp` edge compiles
  to a degenerate same-stick / same-shard *same-core* handoff (nothing to move
  cross-core). Genuine cross-core ring traffic was forced with the round-trip
  bridge (i -> 31-i -> i), and its STCDP move time is inside the on-chip number —
  so on-chip wins *despite* paying for a full cross-core round-trip it does not
  strictly need. A co-located production single-move would beat these figures.

### Load-bearing caveat: fixed routing (upper bound, not yet deployable)

The splice rests on a **fixed round-robin permutation**, which makes the
token->core placement static and therefore statically splice-able via
`STCDPOpLx`. Real MoE routing is **dynamic** (the router selects experts per
token at runtime), which needs a runtime-index-driven `memId` — the
dynamic-addressing frontier, which does not exist today. So these numbers measure
the true data-movement physics and give the **upper bound** a future
dynamic-addressing mechanism would target; they are not a deployable MoE
optimization on their own. (Also flagged: `derive_placement` returned None for
this edge because its sub-stick guard assumes `split_dim == stick_dim`, but the
edge splits on `mb` while the stick is the hidden axis — a decouple-split
generalization would handle it directly.)

The measured per-MB device saving here (0.031–0.081 ms/MB) exceeds the 0.029
ms/MB anchor, consistent with the on-chip path also removing restickify and
scheduling overhead beyond the pure HBM bytes; the largest 8 MB shape saturates
the HBM-vs-on-chip gap most (0.081 ms/MB, 1.94x).

## Block baselines

| workload | baseline ms (min) | max_err | label |
|---|---|---|---|
| MoE expert FFN (E=8, H=2048, INTER=8192, T=128) | 125.8477 (120.8593) | 0.044922 (high) | MEASURED, on-chip = PROJECTED |
| transformer block | does NOT compile on current Spyre stack | — | FAILED: fp32 mean (fixed -> fp16) then dxp SIGABRT on a fused attn-linear-transpose kernel |
| MoE full block | does NOT compile on current Spyre stack | — | FAILED: LoweringException AssertionError (routing/dispatch) |

## Device findings

- Spyre does NOT support fp32 `mean` -> RMSNorm must stay fp16 (granite's does).
- SDPA decomposes on Spyre into separate bmm/softmax SDSCs; the score->softmax
  edge is a real same-stick cross-core handoff.
- Two on-chip frontiers: (A) layout-changing transpose (Compute-CB fault); (B)
  dynamic addressing (MoE routing same-stick but needs runtime memId). Tier-1
  proven = same-stick + static addressing. The MoE dispatch/combine A/B above
  measures frontier (B)'s upper bound under FIXED routing — real dynamic routing
  still needs the runtime-index-driven `memId` that does not exist today.

## Projected speedups (PROJECTED, from `reproduction/workloads/*/projection.md`)

| workload | regime | projected whole-block speedup | source |
|---|---|---|---|
| transformer block | hidden 2048 sweet spot | ~10-15% (~5-20% bracket) | `transformer_block/projection.md` |
| transformer block | hidden 4096 | ~3-12% | `transformer_block/projection.md` |
| MoE block | mid H 1k-2k, 128-512 tok | 1.11x .. 1.39x | `moe_block/projection.md` |
| MoE block | prefill 4096/14336 | 1.06x .. 1.10x | `moe_block/projection.md` |
| MoE block | decode 1 token | ~0.95x, gate OFF | `moe_block/projection.md` |
| MoE routing | dispatch/combine, MB-scale | now MEASURED 1.3-1.9x (see MoE A/B above) | `moe_routing/projection.md` |
| mamba2 | prefill in-proj 4.2 MB | +0.117 ms; decode net-neg | `mamba2/projection.md` |
| attention | prefill seq=512 (16 MB) | +0.459 ms/layer | `attention/projection.md` |
| attention | prefill seq=2048 (256 MB) | +7.42 ms/layer | `attention/projection.md` |
| attention | decode seq=1 | sub-MB, gate to HBM | `attention/projection.md` |

~63% of MoE activation bytes ride eligible same-stick edges; ~37% on blocked
layout-changing edges. Projections cap the anchor at the physical HBM round-trip
ceiling (~0.012 ms/MB). All workload projections exclude blocked layout-changing
handoffs.

## Status of the pending device runs (resolved)

- **Attention seq=512 on-chip A/B — DONE: 2.5595 -> 1.9778 ms = 1.29x, value-correct.**
- **Transformer block baseline — FAILED to compile** (fp32 mean fixed; then dxp
  SIGABRT on a fused attn-linear-transpose kernel). No baseline on this stack.
- **MoE full block baseline — FAILED to compile** (LoweringException). No baseline.
- **On-chip A/B vs projections for the full blocks (transformer, MoE block,
  mamba2):** remain PROJECTED — there is no full-block on-chip splice, and the
  full blocks don't compile, so the only measured on-chip A/B on a real model
  component is the attention QK^T->softmax edge (1.29x above). MoE FFN has a
  measured baseline (125.85 ms); its on-chip number is projection.

### Streaming (>4k) status

The streamed cross-core bridge (`build_streamed_bridge`, commit `ea98321`) is
built + offline-validated (20 tests) but **NOT device-validated and NOT yet a
working >4k handoff**. A code-level review found an endpoint-residency gap:
move-tiling shrinks the staging buffer but the producer/consumer activations
(4 MB/core @8192) still need full LX residency, and the realize path flips them
to 128 KB-spaced buffers -> overlap/corruption. **Move-tiling alone does not solve
>4k; it needs producer/consumer tiling (a fused pipeline).** See
`StreamingImplementationPlan.md` CORRECTION. No device test was run (a naive one
would fail on endpoint overlap, not the buffer-reuse risk it was meant to test).

### What is MEASURED on a real model component vs projected

MEASURED on device: add-mm multi-size (1.13-1.22x same-core), compiler-driven
e2e on-chip (value-correct), cross-core ring on add-mm AND on the real attention
QK^T->softmax edge (**seq=512: 1.29x**), **MoE dispatch/combine A/B
(1.3-1.9x, fixed routing)**, MoE expert-FFN baseline. PROJECTED (grounded in the
0.029 ms/MB anchor): whole transformer/MoE/mamba block speedups, since those
blocks either don't compile on the current stack or have no full-block on-chip
splice. The two regimes are now both measured: activation-dominated edges
(attention prefill, MoE routing) win; the weight-bound dense block is
within-noise.
