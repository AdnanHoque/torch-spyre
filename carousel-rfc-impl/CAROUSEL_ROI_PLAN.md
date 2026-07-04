# Ring-Aware Matmul & Attention: Consolidated Plan for the Carousel RFCs

## 1. The question

Two RFCs proposed ring-aware "carousel" lowerings — a **Weight Carousel** that rotates the weight operand across cores to make prefill matmul compute-bound, and a **KV Carousel** that sequence-shards the decode KV cache and merges partials with an LSE ring-fold. Both were pitched as the way to close a large under-roofline gap by exploiting on-chip ring bandwidth instead of hauling bytes through HBM.

We ran the measurements. This document reports what they settled, disposes of each carousel on its own stated goal, and ranks what to build next.

**Headline:** Both carousels lose to cheaper, device-verified levers on their own goals. The prefill layout fix and the cost-model fix are banked wins that already cover three of the four production goals (prefill matmul, decode projections, cross-model matmul array-fill). The one ring-aware mechanism worth building is not a rotation — it is the **cross-bundle co-bundling redesign** that both the attention all-gather and any reshaped KV work require, and it is the correct home for the single carousel premise that survived: the 130 GB/s uniform-shift ring transport.

Throughout, **device-verified** means measured on hardware; **modeled** means analytic/roofline; **refuted/confirmed/open** are stated plainly. A refuted idea named clearly is worth more than a hedge.

---

## 2. What the measurements settled

Ten premises across the two RFCs were tested. The transport premise is the strongest survivor; the motivation premises are the biggest casualties.

| # | Premise | Verdict | Basis |
|---|---------|---------|-------|
| W1 | Token-split replicates weights ×32 in DRAM (the "×32" headline) | **Refuted** | Cost model counts each operand once; `cohort_penalty` caps `max(m,n)≤8`, so replication is **≤×8**. Weight-stationary model: M-split duplicates weight traffic by the M-factor (≤8), N-split is free. |
| W2 | Prefill runs ~30× under roofline because LPDDR hauls weights 32× | **Refuted** | The real prefill gap was a **layout artifact** (shared-weight unit-BMM), closed 29.8%→72.7% util with **zero new transport**. The binding constraint was layout, not weight-byte replication. |
| W3 | The cost model prices the replication penalty and prunes token-split | **Refuted** | There is **no replication byte term**; token-split is limited by `cohort_penalty`, not a priced penalty. A proven byte win is therefore **unselectable** until `_matmul_split_cost` gains an m-fold term. |
| **W4** | **Carousel rotation runs at ring speed (uniform p→p+1 shift escapes the scatter floor)** | **Confirmed (device)** | ρ_eff = **54 / 90 / 130 GB/s** at 4.06 / 8.13 / 16.25 MB, R²≥0.9985; streaming aggregate ~244–254 GB/s. **2.5–3.6×** the ~36 GB/s scatter floor. This is the RFC's one genuinely validated premise. |
| W5 | Full async overlap of rotation with compute (perf model term) | **Open** | No async-overlap measurement exists. The deeptools `overlapInpFetchWithCompute` gate fires only for Conv2D/SparseConv2D, **not matmul** — so `overlap=False` (a producer→move→consumer barrier) is the honest wall-clock today. |
| W6 | Compute-bound at full-sequence prefill (gate ρ≥C·b·P/2S) | **Open → NO-GO at production S** | Roofline is **NO-GO at S≤512** (ρ_min 176 > 166 GB/s ring) — exactly the Granite prefill block. GO only at S≥1024, and C is an admitted placeholder. |
| W7 | SHUFFLE accepts a δ=±1 rotation | **Confirmed (device)** | A forced fused-SwiGLU relayout fired 2 `OnChipMoveSTCDPOpLx` SDSCs (δ-1 rotate + multicast, LX→LX, no HBM bounce). The generic reshard path already carries it. |
| W8 | LX→LX transfer with no LPDDR bounce | **Confirmed (device)** | −147 µs on fused SwiGLU by keeping the down-proj LHS on-chip; same op as `stcdp_range` (~1.017×). Real, but only **1.7% of the layer** — the ring move is a minority of the HBM it displaces. |
| K1 | Decode attention at BS=1 is bandwidth-bound | **Confirmed** | HBM is one shared ~170 GB/s pipe across all 32 cores; decode KV streaming is HBM-serialized. |
| K2 | Sequence-sharding to channel-affine LPDDR lifts the ceiling H_kv·β → P·β (≥4×) | **Refuted** | HBM is one **flat `memId=-1`** space with no channel/bank affinity. The clean per-channel model is dead; the gain collapses to an **unmeasured core-ownership fill factor u**. |
| K3 | KV fold transport is de-risked by the fast-ring probe | **Refuted** | The 130 GB/s result is the **weight-carousel's uniform shift**. The KV LSE fold is a **linear chain** piling full 16.5 KiB payload on one link (the contention pattern), where the ~58 µs fixed per-move cost dominates. |
| K4 | No new collective op needed (compose SHUFFLE + local merge) | **Refuted** | No single-AIU move-then-reduce primitive exists. The fold must be **STCDPOpLx move + a separate SFP `lse_combine`** per hop, on the critical path every layer every step, behind the same closed overlap barrier. |
| K5 | Local flash-decode partial pass is "an ordinary local op" | **Refuted** | There is **no flash lowering**: SDPA decomposes to dense QK^T → full `torch.softmax` → attn@V with a dead `torch.empty` logsumexp. The KV carousel must **create** the entire flash substrate first. |
| K6 (both) | The rewrite is numerically inert / bit-exact | **Confirmed (executed)** | `reference.py` 12/12 PASS (tile-exact, schedule-cover, LSE fold vs single-pass softmax incl. ±80-logit overflow, GQA read-once). Retires **numerics only** — not speed, not selectability. |

Two cross-cutting results also landed, independent of either RFC's fate:

- **The ~36 GB/s figure is a contention floor, not a ring cap.** Strided all-to-all scatter puts 4–9 transfers on every link (524 descriptors) and serializes; a uniform shift puts 1 transfer/link (33 descriptors) and hits raw ring. The distinguishing variable is **per-link transfer count, not burst size** — enlarging bursts 50–100× gave 0.958–1.004× (zero improvement).
- **The cost model prices every move at flat bytes/128** with no contention, hop-distance, LX-capacity, or multicast-forwarding term. It cannot tell a uniform shift from a scatter, so it **under-costs scatter ~4×** and is blind to the carousel's entire bandwidth advantage.

---

## 3. Dispositions

### Weight Carousel — **Shelve**

The problem framing is refuted at the root. Prefill weight replication is ≤×8, not ×32; the "30× under roofline" gap was a **layout artifact**, not DRAM weight traffic; and it is already closed to ~73% util by a device-verified pure-inductor fix that is not a carousel. The carousel does not change compute util — only DRAM bytes — and those bytes are **invisible to the planner** (no replication cost term), so even a proven byte win is unselectable without a new term the M0 docs themselves call premature. Its one confirmed premise, the 130 GB/s uniform-shift roofline, is real but worth only ~1.7% in this vehicle, is gated by an unmeasured async overlap that is currently closed for matmul, and at the production sequence block (S≤512) the compute-bound gate is a roofline NO-GO. Correctness is bit-exact and the movement mechanism (δ=±1, LX→LX) is proven — but a proven mechanism with a refuted motivation, an unselectable payoff, and an incumbent that already reached the goal is a shelve, not a reshape. **Harvest the transport finding into the co-bundling/ring-fold work and close the RFC.**

### KV Carousel — **Reshape**

The structural idea — sequence-shard the KV cache, run local flash partials, merge with a deterministic LSE ring fold — is the right shape for long-context decode, and both its **numerics** (12/12) and its **capacity dividend** (max context rises ×P/H_kv at fixed per-region budget, independent of the speed story and of channel affinity) are genuinely de-risked. But every quantitative **speed** premise is refuted or unmeasured: the ≥4× ceiling needs channel affinity the backend cannot express (flat `memId=-1`); the "fast transport" probe validates the *weight* carousel's shift pattern, not the KV fold's contention-heavy linear chain; the fold needs a backend primitive that does not exist, behind an overlap barrier that is currently closed; and the flash substrate it builds on is a `NotImplementedError` stub facing the **same cross-bundle LX-residency wall** that has so far defeated the more-mature attention all-gather. Meanwhile the decode-projection array-fill goal is already won cheaply by shipped split-K plus the one-file cost-model fix, so the carousel's real remaining target shrinks to **attention-over-cache** — which no one has yet shown is H_kv-capped. **Reshape:** keep the sequence-shard + LSE-fold + capacity thesis, replace the channel-affinity story with a measured core-ownership `u`, and converge the fold onto the single cross-bundle redesign the all-gather already requires — fund **one** vehicle into that wall, not two.

---

## 4. ROI-ranked roadmap

Best-first. "Evidence" distinguishes device-verified from modeled. Land the two banked levers before spending intellectual capital on anything downstream — every carousel claim must be measured *on top of them*.

| # | Move | Intel. ROI | Prod. ROI | Targets | Evidence | Cost | Blocker |
|---|------|-----------|-----------|---------|----------|------|---------|
| 1 | **Land shared-weight unit-BMM layout + planner tie-break fix** (prefill) | Med | **High** | prefill, matmul | **Device-verified** (29.8→72.7% util; MLP-proj 3.122→1.024ms; QO 42.6→73.7%) | Low (~4 files + one tie-break edit, pure inductor) | Rebase/re-validate on latest main + latest deeptools (verified on b18cca8; ablation is coupled — both edits load-bearing) |
| 2 | **Land cost-model min-cores fix** | Med | **High** | matmul, prefill, decode | **Device-verified** (+76%→+13% aggregate; 10/12 within 3% of device-best) | Low (one `work_division.py` cost-term edit) | Guard the QK^T-decode tiny-M regression before landing |
| 3 | **Cross-bundle co-bundling redesign** (co-bundle mul(K)/flash-partial into the consumer program) | **High** | Med | flash/paged attention, decode | Device-verified substrate + P0 pass + dxp-on-latest; **E2E value-correctness blocked by the cross-bundle wall** | High (deeptools + inductor SDSC wiring) | LX does not persist across separate device programs — needs the redesign, not another patch. **This is the home for the 130 GB/s uniform-shift finding.** |
| 4 | **Pattern-aware ring cost-model correction** | **High** | Med | matmul, prefill, decode | **Device-verified** (ρ: 36 scatter vs 130 uniform-shift, R²≥0.998) | Low–Med (`comm_cost`/`work_division` Python) | none — key cost on per-link transfer count (1 vs 4–9), not burst; gate on LX capacity so it doesn't over-derate |
| 5 | **Split-K decode — keep and verify it fires** | Low | **High** | decode, matmul | **Device-verified** (~1.73–1.79× on B=1 M=1 narrow-N KV/Q/O) | ~Zero (shipped; heuristic already picks it) | none — regression-guard that it stays gated to memory-bound decode (it hurts prefill SwiGLU +19%) |
| 6 | **KV Carousel — reshape: honest M0 measuring core-ownership `u`** | Med | Low | flash/paged attention, decode | **Modeled** (numerics 12/12; speed refuted/unmeasured) | M0 low; full build high (needs a new flash-decode lowering + a composed move+SFP fold) | No flash lowering exists and `u` is unmeasured — cannot yet show decode attention is even H_kv-capped |
| 7 | **Weight Carousel — shelve** | Med | Low | prefill, matmul | Refuted motivation; one confirmed premise (130 GB/s) worth ~1.7% in this vehicle | High for zero selectable payoff | Unselectable — no cost term rewards a DRAM-byte saving, and the layout fix already reached the goal |

Notes on the risk edges of each move:

- **Move 1** — the layout fix and the planner N-heavy tie-break are a *coupled* ablation (layout-only reaches 47.4%/62.3%; full 72.7%/73.7%). Re-confirm the pairing still fires on latest main + latest deeptools, not just the b18cca8 it was verified on.
- **Move 2** — the fix eliminates the attn@V and Q/O/K/V decode regressions but leaves a structural QK^T-prefill holdout (+62%) and can regress tiny-M QK^T-decode; add an explicit tiny-M guard before landing.
- **Move 3** — five debug layers have already been peeled on the all-gather without value convergence: compile gates pass while the packer places the multicast dests physically wrong because the consumer re-reads K from HBM via an in-bundle ReStickify that shards on `x:32`, not the gather's assumed `out/Lk-band:32`. This is a hard multi-layer SDSC problem, not a patch.
- **Move 4** — scatter is currently under-costed ~4×, biasing the planner *toward* on-chip relayout where it is slowest; a naive derate could over-correct without LX-capacity gating.
- **Move 6** — first step is a measurement, not a build: is decode attention (today dense QK^T + full softmax, not a head-split streaming pass) even bandwidth-capped, and what `u` does core-ownership deliver? Only fund the fold if `u` beats the all-gather vehicle on measured numbers.

---

## 5. The one thing to do next

**Land moves 1 and 2 — the two banked, device-verified inductor levers — before anything else.**

Together they are pure-inductor, branch-ready, and already reach the prefill/matmul array-fill goal the weight carousel was built to chase, plus most of the decode-projection goal the KV carousel targets, at near-zero cost:

- The shared-weight unit-BMM **layout + planner** fix takes prefill MLP-proj from 3.122 to 1.024 ms and array util from 29.8% to ~73%, matching sendnn — with **zero new transport**.
- Stacked on it, the **cost-model min-cores** fix (one `work_division.py` edit) takes the aggregate Granite-matmul gap from +76% to +13%, 10/12 shapes within 3% of device-best, covering both prefill and decode projections.

Shipping these **retires the weight carousel's entire motivation** (the util gap it targeted is already closed, and its byte win is both ≤×8 and invisible to the planner) and **shrinks the KV carousel's real target** to attention-over-cache alone. They are also the baseline any future carousel claim must be measured on top of — measuring against unpatched main would double-count a +76% work-division loss the fix already recovers.

**Only after they land** does the single flagship intellectual bet become the right place to spend the one surviving carousel premise. The genuinely-confirmed **130 GB/s uniform-shift ring transport** belongs in the **cross-bundle co-bundling redesign** (move 3) — the shared prerequisite for the sendnn-proven attention all-gather *and* any reshaped KV LSE-fold — where a producer shard can stay LX-resident into its consumer and the multicast operates on the true shards instead of re-reading HBM. That is the one ring-aware mechanism worth building. It is not a weight rotation whose payoff is unselectable and NO-GO at production S≤512; it is the wall that both remaining attention vehicles have to go through, and the uniform-shift roofline is what makes going through it fast.
