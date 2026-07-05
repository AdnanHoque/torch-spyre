# Ring Collectives: How Codex's `comms-collectives` Work Folds Into Ours

## 0. TL;DR for the team

Codex and we are building **two halves of one ring-communication substrate**, not two competing things.

- **Codex is building the mechanism / lowering half**: make an on-chip collective (LX-to-LX relayout, ring, local copy) *legal and value-correct* by expressing the producer/consumer coordinate mismatch as DLDSC metadata and letting Deeptools synthesize the physical move. He has one production-shaped class landed (scatter, **measured** ~1.065×) and is blocked on the next (all-gather into a matmul KERNEL operand).
- **We are building the selectability and arithmetic-collective halves**: a cost term that makes the *planner pick* the on-chip move (Bet 2), and an LSE ring-fold that enters the *reduce lane codex has deliberately not entered* (Bet 3). Our third bet (flash-in-a-loop, Bet 1) and Codex's flash restickify-on hit the same workload from different levers.

Three seams converge hard; one bet stays independent; one hoped-for connection is **refuted**.

---

## 1. What Codex built

### 1.1 The north star (identical to ours)

Remove **in-scope, non-weight HBM round trips** in Granite by moving activations on-chip instead of spilling to HBM between producer and consumer. The design contract:

> Torch exposes enough **coordinate / layout metadata** for Deeptools to *derive and schedule* the movement — Torch does **not** emit physical ring schedules.

In scope: scatter, broadcast/multicast, gather/all-gather, layout-changing LX restickify, and reduce/all-reduce. Explicitly **out** of scope: weight restickifies/preload, working-set-reduction streaming, and wall-time-only speedup claims. This is byte-for-byte the substrate our ring work targets.

### 1.2 The four approaches (why the current one)

| Approach | What it was | Verdict |
|---|---|---|
| **A** — explicit coordinate-remap data-op | Torch computes exact src/dst cells, emits a mixed SDSC with movement rows | Proved HBM-handoff removal yields real speedups, but deprecated (data-dsc support dying in the SuperDSC direction; pushed too much physical scheduling into Torch) |
| **B** — STCDPOpLx / ranged carrier | Reuse the existing LX-transfer primitive | Kept as a backend building block; a list of range transfers is a *physical carrier*, not a *logical contract*; coarse full-resident form is **capacity-unsafe** at Granite scale |
| **C** — DLDSC backend relayout insertion | Torch emits distribution metadata; Deeptools detects incompatibility and inserts the move | **Current production direction**; PR1 scatter rides this |
| **D** — kernel-neighbor carousel | Loop-scope the all-gather-into-KERNEL-operand class instead of full-resident | **Active thread**; the current blocker |

Note that **STCDPOpLx (Approach B) is the same primitive our attention all-gather substrate rides on.** Codex independently found it capacity-unsafe when it materializes the full RHS operand — which is exactly why our substrate needs loop/tile-scoping rather than full residency.

### 1.3 The measured / structural / value results — tagged honestly

**PR1 scatter — MEASURED win.** The production-shaped first class: disjoint 1:1 same-layout relayout where producer/consumer differ in core *ownership* but the stick/layout form is compatible. It changes **DLDSC metadata on existing ops** (no new Torch graph nodes). Granite S=512:

- kernel: **14.7258 → 13.8213 ms/iter (~1.065×, +6.1%)** — trace-derived, trustable
- wall: 27.6074 → 26.5205 ms (~1.041×) — smaller; guardrails say don't trust wall alone
- Mechanism: removed all explicit *non-weight* `ReStickifyOpHBM` rows; remaining explicit HBM restickifies are weight-shaped (out of scope). Run was at backend LX frac 0.2.

**Flash restickify-on — STRUCTURAL only (not yet value or perf).** Compiling flash attention with relayout OFF vs ON, both 550 SDSCs, both returncode 0:

- OFF: 32 `ReStickifyOpHBM` / 0 LX-restickify / 0 backend plans
- ON: **0 HBM-restickify / 32 `ReStickifyOpLx` / 32 `matmul_operand_broadcast` backend plans**

This **proves** the flash HBM-restickify class is *structurally* convertible to on-chip form. It does **not** prove value-correctness: the unpatched value run mismatched 31.5% of elements — and the relayout-**OFF** baseline itself mismatched 75.1% in that env, so `test_flash.py`'s CPU assert is **not a clean oracle** there. Codex flags this explicitly. State: structurally unblocked, value-correctness is the open gate — mirroring our attention all-gather (structurally P0-PASS, value still open).

**M=4 operand-broadcast probe — the active blocker, split cleanly:**

- Value-**correct** but capacity-**unsafe** (full-resident staged): baseline no-relayout M4 ALLCLOSE True 0/1024; archived carousel `kernel_neighbor_carousel_M4` ALLCLOSE True 0/1024; full-resident M16 staged ALLCLOSE True max-diff 0.002. At Granite scale it fails: *"unable to allocate final matmul operand LX region on core 0."*
- Capacity-**safe** but value-**wrong** (loop-scoped ring-into-KERNEL): current reproduction ALLCLOSE False, **949/1024 mismatch**, max-diff 0.749; skipviews M16 3829/4096; self-ring triggered a **`RAS::PCI::BusFence`**.

So the class is strictly broader than PR1 scatter: it is **`all_gather_replicate` + `layout_conversion` into a `DsTypes::KERNEL` operand**, and neither the value-correct form nor the capacity-safe form is *both* at once yet.

### 1.4 The representation layer is NOT the bottleneck

`dxp_standalone` cardinality probes (scatter / one-to-many / many-to-one / many-to-many) **all pass**, and a Deeptools gtest confirms broadcast/gather/all-gather produce LxRelayout descriptors. DLDSC can *express* all four cardinalities. Hard rule discovered: coordinate maps must be **dense** over relayout dims (split-1 dims must appear as explicit slice 0, else `std::out_of_range map::at`). This **de-risks our selectability side**: the planner can emit these classes once it decides to.

---

## 2. The mapping: Codex's lanes vs. our mechanisms

| Codex lane | Our mechanism | Relationship | Why |
|---|---|---|---|
| **PR1 scatter** (1:1 disjoint same-layout ownership relayout; DLDSC metadata insertion; MEASURED +1.065×) | **Bet 2** — per-link contention cost term (`ah/ring-cost-term`, DONE + unit-tested, not device-measured) | **Complement** | Codex makes the move *legal*; nothing in his work *prices* it so the planner prefers it. He lists "reshard-aware work-division" as future intent. Our cost term is exactly that missing selectability layer. Pick vs. make-legal — non-overlapping halves of one pipeline. |
| **Flash restickify-on** (32 HBM → 32 LX `ReStickifyOpLx`, layout-preserving all-gather) | **Bet 1** — flash-in-a-loop (streaming Lk online-softmax; blocked at `optimize_restickify`) | **Independent (hoped-for link REFUTED)** | His restickify-on is a layout-**form** change on a **dense** operand. Our wall is `optimize_restickify` failing to gather the **sparse** `amax(dim=-1)` real_max/denominator (multi-stick → single-stick) — a **reduction**, which his own taxonomy parks in the uncovered reduce lane. Same-named pass, different subclass. |
| **`matmul_operand_broadcast` / kernel-neighbor carousel** (all-gather activation shards + layout-convert into a KERNEL matmul operand; ACTIVE BLOCKER) | **Attention all-gather substrate** (QK^T `mul(K)`→BMM multicast STCDPOpLx; P0 PASS) | **Same thing** | Same class: all-gather activation shards + layout-convert into a KERNEL matmul operand. Codex hits the AV **value-operand** (Tensor1/buf21); we hit the QK^T **K-operand**. Different operand, same mechanism, same backend region, both with a passing/archived run the production loop-scoped path can't reproduce clean. |
| **Reduce / all-reduce** (explicitly NOT covered by relayout; no positive single-AIU evidence) | **Bet 3** — LSE ring-fold merge (neighbor reduce-scatter of (m,l,O) partials with LSE-combine; math-validated 13/13) | **Ours extends theirs** | Both taxonomies agree reduce is out of the copy/relayout lane and needs op/axes/dtype/identity + arithmetic fan-in scheduling. Our LSE ring-fold is a frontend-composed entry into that lane (STCDPOpLx move + SFP `lse_combine` per hop = a move-then-reduce the backend offers as no single primitive). We pioneer what Codex flags as absent. |
| **Two-stage loop-scoped design + capacity ceiling** (full-resident = correct but alloc-fails; ring-into-KERNEL = safe but wrong) | **All-gather substrate loop/tile-scoping requirement + our RING_SPEED per-link physics** | **Complement** | Codex independently derived the same capacity constraint and the same fix shape: ring-gather into tile-sized loop-local staging, same-core pieces via local LX copy (never self-ring), then local `ReStickifyOpLx` into KERNEL. His self-ring BusFence and source-alias failures refine our per-link cost model. |

---

## 3. The shared blockers

### 3.1 The DCG coordinate-metadata → physical-placement wall (the big one)

Both efforts have a **passing-but-unreproducible** all-gather-into-KERNEL run, and both stall in the same Deeptools region — **but at adjacent stages, with an important correction to our earlier framing.**

**Codex's symptom (metadata absent, upstream):** renaming the artificial transfer `_lx_neighbor` → `_lx_local` dodges a value-corrupting DDC path but exposes `fillDataInfo` throwing `std::out_of_range map::at` on node `transfer_lds1_src:lxlu_dst:ptrow0`, because its coordinate maps are **empty** (`coordDims=0`, `coordCoreMap=0`). The generated transfer is processed as an ordinary tensor transfer carrying no coordinate metadata. Relevant files: `L3DlOpsScheduler.cpp` (`populateMatmulOperandBroadcastStickRingTransfers…`), `dsc2Pcfg.cpp` (`buildLxNeighborRingTransfers`), `ddc_fold.cpp`, `dsc2.cpp` (`fillDataInfo`).

**Our symptom — CORRECTED (was overstated in the framing paragraph):** The original one-paragraph description said our substrate is "blocked by a dxp version/DDC conflict." **That is stale.** The dxp base-drift was fixed on latest Deeptools (flag-OFF SDPA now matches CPU at 1e-3, was 0.02). The *current* blocker is deeper and is **not** the same literal exception:

- A device **value** bug (~0.02) traced through five layers to **cross-bundle source provenance**: `mul(K)` is a separate device program, so the consumer bundle **re-reads K from HBM** via an in-bundle `ReStickify` sharded on `x:32` — not the gather's assumed out/Lk-band model. As architected (operating within the consumer bundle on that ReStickify), the substrate **cannot** be value-correct *and* cannot avoid the HBM read it meant to eliminate.
- Our separate `04` capability doc independently pins a matching region: `runDcgForInputFetchNeighbor` (the loop-scoped operand-fetch generator) is **present-but-dead in-pipeline** and **hard-pinned to `DsTypes::INPUT`** while the operand is `DsTypes::KERNEL`.

**Honest implication.** It is **not** the same literal exception, and our substrate's true fix (co-bundling `mul(K)` into the QK^T program so K stays LX-resident) is a **frontend/bundling redesign**, not only a backend patch. But Codex's carousel and our `04`-doc gap point at the **same backend capability hole**: coordinate-map → physical placement for a replicated/gathered **`DsTypes::KERNEL`** operand under loop-scoped fetch. Generalizing `runDcgForInputFetchNeighbor` off `INPUT`-only and populating/consuming the coordinate maps is the single Deeptools task most likely to move **both** passing-but-unreproducible runs forward — with the caveat that our side *also* needs the co-bundling prerequisite before the placement fix can bite. This is a Deeptools tooling task, not a Torch-frontend gap, for the placement half.

### 3.2 Coordinate-fidelity in the planner → backend handoff

Codex's flash first-failing edge is value-wrong because the **count-based** backend plan groups cores contiguously (0,1,2,…,7) while the real SDSC coordinate map is **strided** (0,4,8,…,28). Count/group metadata alone is too weak — the actual coordinate maps must drive grouping.

This lands directly on Bet 2's doorstep: our edge cost descriptor must carry the **true coordinate stride**, not a cohort count, or the selected plan is mispriced/misrouted the same way. Our own attention geometry bug lineage (sendnn head-grouped vs. tsp Lk-band core assignment) is the same stride-vs-count class. Action: converge the descriptor format with Codex's backend-plan JSON (`consumer_operand_ds_type=KERNEL`, `group_count`, `consumer_replicas_per_group`) so the priced edge and the lowered plan agree.

---

## 4. The convergence picture

**Converge hard on three seams; stay independent on one.**

1. **CONVERGE — Bet 2 is the selectability layer Codex explicitly names as future intent and has not built.** Land the cost term on top of his DLDSC lowering so the planner *selects* the on-chip move his PR1/flash work makes legal. Without it, nothing picks the ring plan — our own Granite ~1.19× win was honestly **LX-residency, not collective selection** (the collective classes emit zero classification rows on the full Granite block; the lowering exists but is unselected).

2. **CONVERGE — our attention all-gather and Codex's `matmul_operand_broadcast` are the same class in the same DCG region.** Pool the debugging on generalizing the `DsTypes::KERNEL` loop-scoped operand fetch. One backend fix likely clears both placement halves; note our side additionally needs the `mul(K)` co-bundling prerequisite.

3. **CONVERGE — adopt Codex's two-stage loop-scoped shape** (ring-gather into tile-sized staging → local `ReStickifyOpLx` → KERNEL) as the capacity-safe form our full-resident substrate needs. His self-ring BusFence and source-alias diagnostics are directly reusable.

4. **STAY INDEPENDENT — Bet 3 (LSE ring-fold) owns the reduce lane Codex has deliberately left empty.** It is the arithmetic-collective *extension* of his data-movement substrate, not a duplicate. Position it as the concrete first entry into his declared-empty reduce lane.

**Do NOT** bet on Codex's flash restickify-on unblocking Bet 1: **REFUTED.** Layout-form relayout and sparse-amax reduction gather are different subclasses; his own taxonomy puts our wall in the uncovered reduce lane.

### Physics substrate all of this rests on (MEASURED)

Per-link *transfer count* (not burst size) is the variable. Device-measured: all-to-all scatter ~34–36 GB/s (4–9 transfers/link); uniform p→p+1 shift 54 GB/s @4MB → 90.5 @8MB → **130 @16MB** (1 transfer/link); ~250 GB/s streaming aggregate; R² ≥ 0.998 on every sweep. Burst enlargement bought a measured null (1.0×). This is the ~130-vs-36 band Bet 2 encodes, the LSE reduce-scatter (Bet 3) rides, and the all-gather multicast exploits. **Same-core replicas are a free local copy** (Codex's self-ring BusFence proves it) — only genuinely cross-core links pay the band.

---

## 5. Revised next step per bet

### Bet 2 — per-link contention cost term (DONE, not device-measured)

Codex supplies the lowering this term selects for, a live Granite S=512 harness with a **measured** 1.065× relayout baseline, and a concrete fidelity requirement (his flash value-wrong root cause = count-based grouping losing the true stride).

- **(a)** Make the edge cost carry the **true coordinate stride**, not a cohort count, so priced-edge == lowered-plan (match his `KERNEL`/`group_count` JSON).
- **(b)** Device-measure on Codex's Granite S=512 setup — confirm the planner now **picks** the on-chip broadcast and quantify beyond his structural 1.065×.
- **(c)** Add the **same-core-is-free-local-copy** refinement (his self-ring BusFence): only cross-core links pay the ~130-vs-36 band.

Promote Bet 2 from independent experiment to *the selectability layer on Codex's DLDSC lane.*

### Bet 1 — flash-in-a-loop (blocked upstream)

Codex's restickify machinery does **not** help (REFUTED). Stop waiting on it.

- Reframe the blocker as a **reduce-lane** problem: `amax(dim=-1)` multi-→single-stick gather is arithmetic fan-in, not layout relayout. Couple it to Bet 3's framing.
- Track the **Stage-1 reduction-dim-tiling** work (the `coarse_tile` design doc already on main) as the path to surfacing **Lk as a tiling level** — Codex's loop-scoped path tiles the matmul *operand*, not the softmax reduction axis, so Lk-tiling is absent from his lane too. That upstream pass, not Codex's lane, is the real dependency.

### Bet 3 — LSE ring-fold merge (math-validated 13/13, not compiler-wired)

The reduce lane is uncontested and ours. But the *move* half of our move-then-reduce hop (STCDPOpLx) needs the same DCG coordinate plumbing Codex is fixing.

- Keep implementation **independent** (we pioneer the reduce lane), but **sequence it after** Codex's `fillDataInfo`/coordinate-metadata fix lands.
- Reuse his STCDPOpLx move + two-stage loop-scoped staging for the reduce-scatter transport; add the SFP `lse_combine` as the arithmetic half he does not offer.
- Position Bet 3 in the handoff as the concrete first entry into Codex's declared-empty reduce lane — an arithmetic-collective extension, not a competing data-movement effort.

---

## 6. One-line honesty ledger

- PR1 scatter 1.065× kernel — **MEASURED** (trace-derived, LX frac 0.2).
- Flash 32-HBM→32-LX — **STRUCTURAL** (both 550 SDSC, rc 0); value **NOT** established (31.5% mismatch on a dirty 75.1% oracle).
- M=4 operand-broadcast correct run — **VALUE-VERIFIED** but capacity-unsafe; capacity-safe form value-wrong.
- Our ~130-vs-36 band, uniform-shift scaling — **MEASURED** (R² ≥ 0.998).
- Bet 2 cost term — **DONE + unit-tested (7/7)**, no-regression structural; **NOT device-measured**.
- Bet 3 LSE fold — **MATH-VALIDATED 13/13**, not compiler-wired.
- Bet 1 flash-in-a-loop — **BLOCKED UPSTREAM** at `optimize_restickify`; guard committed but inert.
- Attention all-gather substrate — **P0 PASS structurally**; the "dxp version conflict" framing is **stale** — the live blocker is **cross-bundle source provenance**, whose fix is co-bundling `mul(K)` into the QK^T program (a redesign, not a patch).
