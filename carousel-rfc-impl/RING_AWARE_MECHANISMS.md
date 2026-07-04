# Ring-Aware Mechanisms for Spyre: A First-Principles Plan

## 1. Premise — residency is the floor, not the answer

"Pin the tensors to LX" is the trivial baseline. It says *where* an operand lives; it says nothing about *how* data moves between cores, and on a 1D ring the movement pattern is what sets the achievable bandwidth. Two plans with identical residency and identical byte counts can differ by ~3.6x purely in their transport *shape* — a uniform neighbor shift versus an all-to-all scatter. That delta is invisible to placement and invisible to today's cost model.

This document is about the layer *above* residency: ring-aware algorithms that exploit a specific topology property (uniform-shift speed, multicast forwarding, neighbor reduce-scatter, cross-bundle pipelining). Every mechanism here is disqualified if it collapses to "just pin it." The two shelved carousels (Weight, KV) are not re-proposed; the mechanisms below are new or genuinely reshaped to survive the specific killers the device measurements impose.

The verified source anchors: `test_flash.py:17` (`# FIXME: current limitation disallows coarse tiling in Lk`, `kv_block_size = Lk // 1`); `test_paged.py:70-72` (`scores.transpose(-1, -2).contiguous()  # avoid stick reduction`); `work_division.py:684` (`cohort_penalty = max(1.0, max(m, n) / _COHORT_LIMIT)`); `deeptools/dsm/dsmperf.cpp:3725` (`overlapInpFetchWithCompute`, gated Conv-only).

---

## 2. The ring physics (treat as device-measured law)

> **Per-link transfer count is the distinguishing variable — not payload size.**
> Enlarging bursts 50–100x did nothing; piling many transfers on one link is what floors bandwidth.

| Pattern | Per-link transfers | Effective BW | Verdict |
|---|---|---|---|
| Uniform p→p+1 shift | ~1 | ~130 GB/s (up to ~250 streaming) | FAST |
| Neighbor reduce-scatter | ~1 | ~130 GB/s | FAST |
| Multicast all-gather (one→many, STCDPOpLx) | ~1 forward/link | fast band | FAST, **device-proven** |
| Linear fold-to-one root | P on one link | → 36 GB/s | CONTENTION FLOOR |
| All-to-all scatter | 4–9 per link | ~36 GB/s | CONTENTION FLOOR |

Five structural constraints every mechanism must respect:

1. **Multicast is live.** One-source→many forwarding (STCDPOpLx) is device-proven, EBR-safe, with a real HBM win (−147µs on fused SwiGLU LX→LX, no HBM bounce). This is the one ring primitive we know works today.
2. **No fused move-then-reduce.** A reduction hop is a STCDPOpLx move **plus a separate SFP combine** op. A *linear* fold chain piles full payload on one link (contention); a *neighbor* reduce-scatter does not. The fold topology is a real, costed choice.
3. **HBM is one flat memId=-1 space.** No channel/bank affinity control. (This is what killed the KV-carousel's per-channel story — do not resurrect it.)
4. **The cross-bundle wall.** LX does **not** persist across separate device bundles. On-chip producer→consumer handoff survives *only* if the ops are co-bundled into one program. This is the single biggest structural blocker. The one free breaker: a coarse-tile `CountedLoopSchedulerNode` is one SDSC bundle by construction, so ops inside it co-bundle automatically.
5. **The cost model is transport-blind.** It prices every move at flat `bytes/128` — no contention, hop-distance, or multicast term. It **cannot** distinguish a fast shift from a slow scatter and will silently pick the fold. **Selectability is a first-class requirement, not an afterthought.** Additionally, `overlapInpFetchWithCompute` fires only for Conv2D/SparseConv2D, so matmul/BMM transport is a serialized barrier today.

Matmul is **weight-stationary**: weight `[K,N]` resident, tokens (M) stream past; M is never a stick. Ring dataflows must rotate the *streaming* operand, never the big stationary weight.

---

## 3. Top bets

Two of these are prototype-now and pure-inductor; build them in parallel — they are non-overlapping.

### Bet 1 — Flash-in-a-loop: online-softmax as a coarse-tiled Lk reduction *(the one thing)*

**Goal:** FLASH/PAGED + Granite PREFILL — the single goal with **zero banked coverage**.
**Ownership:** inductor-only. **Verdict:** prototype-now.

**The trick.** Today `kv_block_size = Lk` (`test_flash.py:17`) forces the online-softmax loop to run *once* and materialize the full `[Lq,Lk]` scores — 32MB/head, spilled to HBM (~12ms round-trip). Turn the disallowed Lk coarse-tile into a Stage-1 reduction loop: stream Lk in blocks, keep only the `O(Lq)` running `(m, l, A)` state in a `per_tile_fixed` `lx:0` scratchpad that persists across iterations. Scores never materialize; the spill dies.

**Why it is not residency.** Residency would pin the full scores in LX — impossible at 32MB/head vs 2MB/core. This *never materializes* them; the win is the streaming reduction structure, not where a tensor lives.

**Why selectable today.** The dominant win is deleting real HBM bytes the flat `bytes/128` pricer **already counts**, and flash is hint-driven. No new cost term gates it. It is not itself a ring collective — the core Lk loop is local — but it is the mandatory substrate under the *entire* flash ring family (Bets 3, plus ring-rotation and the layout-dividend). Building it de-risks all of them at once.

**Blocker.** The reduction-tiling pass assumes a single binary combine (add/max). The online-softmax merge is a compound 3-tensor associative state:
`m_new = max(m, m_blk); A = A·exp(m−m_new) + A_blk·exp(m_blk−m_new); l` likewise. It must be emitted as a contiguous pointwise sub-run co-bundled at fixed `lx:0`, with transposed-Lk kept **non-stick** so it stays Stage-1 (not the Stage-2 stick-reduction `RuntimeError`).

**First step (verifiable).** In `flash_spyre`, adopt paged's `scores.transpose(-1,-2)` off-stick pattern and set `kv_block_size < Lk`. Compile `test_flash.py`; confirm via `TORCH_COMPILE_DEBUG` that the full `[Lq,Lk]` scores no longer materialize and the softmax reduction lands off-stick (no Stage-2 `RuntimeError`). That validates the substrate *before* touching the compound combine.

### Bet 2 — Per-link contention cost term (selectability primitive + cohort-penalty fix)

**Goal:** cross-cutting — makes every ring plan cost-selectable. **Ownership:** inductor-only. **Verdict:** prototype-now.

**The trick.** Encode the measured physics as a per-move term keyed on max per-link transfer count: uniform shift and neighbor reduce-scatter price at ~130 GB/s, linear fold and all-to-all scatter at the ~36 GB/s floor, multicast as one source read (not N). Burst size is deliberately **not** a term.

**Why it is not residency.** It prices the transport *topology* residency ignores entirely. Without it, the flat model rates a slow linear fold identical to a fast reduce-scatter and picks whichever the enumerator emits first — so residency-plus-ring is unreachable by the planner.

**The sharpest live slice.** `work_division.py:684` charges a *linear* `cohort_penalty = max(1, max(m,n)/8)` to **every** operand broadcast — pricing the device-proven multicast STCDPOpLx as if it were a 36 GB/s scatter. This actively suppresses the wide-N splits the banked array-fill/min-cores fixes want, and the attention all-gather past cohort 8 that flash needs. Fixing this one term re-ranks a mispriced matmul *today*.

**Blocker.** The classifier must be **structural** — key off whether the moved operand is *identical* across the receiving cohort (using the shared-weight/broadcast analysis the layout pass already computes), not a tuned scalar — or it leaks across regimes and regresses the banked min-cores/shared-weight fixes it shares the pricer with. (Global cost knobs have repeatedly leaked here.) The forwarding constant must be device-measured. Its *new* wins are gated on a runnable ring consumer.

**First step.** Edit `_matmul_split_cost:684` to gate `cohort_penalty` on identical-operand (multicast: ~constant + hops) vs distinct-data (scatter: linear). Behind an ablation flag, prove it re-ranks one known-mispriced wide-N/narrow-N reduction matmul toward the device-faster split **and** leaves the banked min-cores/shared-weight picks unchanged.

### Bet 3 — Flash LSE ring-fold merge (neighbor reduce-scatter of (m,l,O) partials)

**Goal:** FLASH/PAGED — the genuine ring-aware flash win. **Ownership:** hybrid. **Verdict:** needs-primitive-first.

**The trick.** Once Lk is work-divided across *cores*, each core owns an Lk-shard and produces a partial `(m, l, O)`. The flash-combine monoid is associative *and* commutative, so merge the partials with a neighbor reduce-scatter (uniform p→p+1, 1 transfer/link, ~130 GB/s) instead of the linear fold-to-one that piles the O payload on one link (~36 GB/s floor). Co-bundles into one SDSC bundle by construction (CountedLoop), dodging the cross-bundle wall.

**Why it is not residency.** Residency cannot even hold the scores; and pinning `O_i` says nothing about how 8–32 shard partials *combine*. The reduce-scatter topology is the entire added layer.

**Blocker (two-deep).** (1) Cross-**core** Lk work-division must produce per-core partials — the temporal loop of Bet 1 is a prerequisite but not sufficient (needs the *spatial* split). (2) Stage-1 today folds into one sequential HBM accumulator; the cross-core LX `(m,l,O)` reduce-scatter is a genuinely new codegen path. Plus Bet 2's cost term to be selectable. The merge is a two-pass fold (max-allreduce, then rescaled sum), not a single reduce.

**First step.** After Bet 1 lands, extend `_propagate_tiled_reduction_op` so Lk is a cross-core split; emit the `(m,l,O)` combine as a neighbor-ordered cross-core LX fold and diff its device-ms against the linear gather-to-one on the flash prefill shape to size the ring-specific increment. **Honest:** the spill-kill credit belongs to Bet 1; this measures only the fold's marginal win.

### Bet 4 — GQA KV multicast-forward to the query-head cohort

**Goal:** Granite DECODE, on a new axis. **Ownership:** hybrid. **Verdict:** promising-unproven.

**The trick.** Granite GQA has H_kv=8, H_q=32 → each KV head is shared by 4 query heads. In the 1-head-per-core decode split those 4 cores form a natural multicast cohort. Read the head once, forward on the fast ring (STCDPOpLx), and save 3 of 4 redundant HBM KV reads at the decode KV-bandwidth bottleneck. This is a *different* axis from the banked reduction-axis all-gather — it shares along the head-group.

**Why it is not residency.** The 32768-slot paged cache is far too big to pin to 2MB/core LX. The lever is replacing 4 independent flat-HBM reads of the same head with one read + on-ring replication — traffic elimination the residency baseline cannot express.

**Blocker.** The graph currently *materializes* the GQA broadcast to 32 distinct heads pre-BMM via `expand(...).flatten().transpose().contiguous()` restickify (`test_attn_k.py`) — destroying the share before QK^T. That materialization must be removed first. And the ring is load-bearing only under a forced 1-head-per-core placement; if work_div keeps 4 heads/core the cohort co-locates and it collapses to residency. Needs Bet 2's cost term to credit 1-read+multicast below 4 reads.

**First step (cheap, pure-inductor).** Remove the `expand+contiguous` restickify in the GQA lowering and confirm the K/V share survives to QK^T; separately inspect what H-split work_div actually emits for the decode shape (is it 1-head-per-core?). Both are prerequisites, answerable with no deeptools work.

### Bet 5 — Neighbor reduce-scatter for cross-core reduction fold (reusable primitive)

**Goal:** MATMUL/DECODE — the reusable fold under Bet 3. **Ownership:** hybrid. **Verdict:** promising-unproven.

**The trick.** Replace gather-all-K-partials-to-one with a ring nearest-neighbor reduce-scatter for any cross-core reduction (split-K matmul, flash @V sum). Moves the accumulate off the ~36 GB/s scatter floor onto the ~130 GB/s uniform band. Primitives already exist (STCDPOpLx move + per-hop SFP add), so it is **not** needs-primitive-first for correctness.

**Why it is not residency.** Residency keeps one partial in LX but is silent on how P partials from P cores combine; the reduce-scatter ordering is the added algorithmic layer.

**Blocker.** Payoff scales with shard count, so it is *marginal* at split-K decode's banked 2–4-way home (tiny `[1,N]` partials, latency/step-count-bound where a tree beats a ring) and only clearly wins at high-split-K matmul and many-shard flash — and that workload doesn't exist until Bet 1/3 land. Unselectable until Bet 2's cost term is in.

**First step.** On the already-banked 1.7x split-K decode path, measure the fold-tax magnitude (device-ms of the current linear `(k−1)` accumulate vs the extra `2(P−1)` neighbor hops) to confirm whether the crossover ever favors the ring at the shard counts that actually occur. A cheap measurement on a banked path, no new codegen.

---

## 4. Full ROI-ranked table

| # | Mechanism | Class | Intellectual ROI | Production ROI | Verdict | Ownership |
|---|---|---|---|---|---|---|
| 1 | Flash-in-a-loop (temporal Lk online-softmax reduction) | flash-paged | high | **high** | prototype-now | inductor-only |
| 2 | Per-link contention cost term (+ cohort-penalty fix) | cross-cutting | high | medium | prototype-now | inductor-only |
| 3 | Flash LSE ring-fold merge (neighbor reduce-scatter of m,l,O) | flash-paged | high | medium | needs-primitive-first | hybrid |
| 4 | GQA KV multicast-forward to query-head cohort | decode | medium | medium | promising-unproven | hybrid |
| 5 | Neighbor reduce-scatter cross-core fold (ring split-K) | matmul | medium | low | promising-unproven | hybrid |
| 6 | Layout-dividend collective endpoint selection | cross-cutting | high | medium | needs-primitive-first | hybrid |
| 7 | Ring-attention rotation (stationary KV, uniform Q rotation) | flash-paged | high | medium | needs-primitive-first | needs-deeptools |
| 8 | MoE expert token-rotation (circulate tokens, skim expert) | cross-cutting | high | low | needs-primitive-first | needs-deeptools |
| 9 | Extend overlapInpFetchWithCompute to matmul/BMM | matmul-prefill | medium | low | needs-primitive-first | needs-deeptools |
| 10 | Costed cross-bundle ring-handoff | cross-cutting | high | low | needs-primitive-first | needs-deeptools |
| 11 | Ring-occupancy as a costed resource (link-hours budget) | cross-cutting | medium | low | needs-primitive-first | hybrid |

**Device-verified vs modeled:** the ring physics table (§2) and multicast STCDPOpLx (P0 pass, −147µs LX→LX) are **device-measured**. The `test_flash.py`/`test_paged.py`/`work_division.py:684`/`overlapInpFetchWithCompute` anchors are **source-verified**. Everything in the "why it wins" bandwidth arithmetic below the primitive level (3.6x fold speedups, spill-kill ms) is **modeled from the measured bands** and must be device-confirmed at the first-step gates. Bets 6–11 are structurally sound but **speculative on payoff** until their gating primitives land.

---

## 5. What we killed and why

Deduped 19 survivors to 11 distinct mechanisms. The four flash-fold proposals (Rabenseifner LSE fold, neighbor reduce-scatter across Lk-cores, `(m,l,O)` ring-fold, congestion-free static schedule) are **one mechanism** — merged into Bet 3. The three cost-term proposals plus the cohort-penalty fix are **one selectability primitive** — merged into Bet 2, with `work_division.py:684` as its live slice. The two overlap-gate proposals are one deeptools task (#9). The two cross-core-fold proposals merged into Bet 5.

Six mechanisms were killed upstream and not resurrected:

- **Weight-multicast for M-split** — the banked shared-weight unit-BMM fix (29.8%→73% util) *is* the load-once-distribute mechanism it rediscovers. Self-defeating.
- **Split-K partial-sum ring all-reduce on SwiGLU** — device-NEGATIVE (+19% on prefill SwiGLU), and latency-bound at M=1 decode where step count, not bandwidth, caps.
- **Ring-pipelined paged handoff** — paged has no temporal loop; reduces to residency across a seam.
- **Page→core placement (channel affinity)** — HBM is one flat memId=-1 space, no bank control; runtime block fragmentation defeats compile-time arc-locality → real serving floors at 36 GB/s.
- **Multicast-Q GQA** — M=1 Q payload ~1KB, below the win floor.
- **Speculative-decode amortizer** — needs out-of-scope serving infra.

**Device-grounded caveats that shaped the ranking:**

1. Bet 1 is not itself a ring collective — its Lk loop is local — but it is the only prototype-now/production-high on a zero-banked goal *and* the substrate under the ring family, so it ranks #1 despite the "prefer ring-aware" bias. Bet 3 is where the topology physics actually earns credit; its production ROI is kept honest at *medium* because the spill-kill credit belongs to Bet 1.
2. Bet 2 makes nothing faster on its own, and its *new* wins are gated on runnable consumers that mostly don't exist — but the cohort-penalty slice re-ranks a live-mispriced matmul today. It must ship behind an ablation gate proving no regression on the banked min-cores/shared-weight fixes.
3. Every needs-deeptools mechanism (MoE rotation, overlap gate, ring-rotation, cross-bundle handoff) is real and high-novelty but off the near-term inductor-bias path and workload- or primitive-gated. MoE rotation additionally **no-ops on granite-3.3-8b (dense)**.
4. The overlap-gate family (#9) carries an extra squeeze: the banked 73% array-fill cap was device-proven to be a 2MB-LX/chunk-size *layout* geometry cap (29.5% flops/byte), **not** inter-slab transport starvation — so hiding transport may not touch the residual gap, and a 2x-slab double-buffer collides with the same cap.

---

## 6. The one thing to do next

**Build the flash-in-a-loop substrate now, and land the per-link contention cost term in parallel.**

Both are pure-inductor, non-overlapping, and prototype-now. Flash-in-a-loop (Bet 1) wins standalone — it kills the 32MB/head score spill that nothing banked touches and needs no new primitive — and it unblocks the entire flash ring family (Bets 3, 6, 7). The cost term (Bet 2) is the selectability linchpin without which no ring plan is ever chosen, and its cohort-penalty slice fixes a live mis-pricing that fights the banked levers today.

Concrete first move: adopt paged's off-stick transpose in `flash_spyre`, set `kv_block_size < Lk`, and confirm the scores stop materializing under `TORCH_COMPILE_DEBUG`. That one experiment validates the substrate before any compound-combine or ring codegen is written.
