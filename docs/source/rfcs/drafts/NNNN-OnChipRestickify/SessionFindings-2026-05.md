# On-Chip Restickify — Session Findings (2026-05)

State-of-the-frontier snapshot for the on-chip core-to-core LX data-movement work
as of 2026-05. It consolidates this session's device measurements and design
investigations into one place: what is **measured + working** on silicon, the
**frontier map** of what it takes to scale to real models, the **sendnn 3-way
comparison** status, and the consolidated **deeptools RFC asks**.

This is a synthesis, not a primary record. The underlying detail lives in the
sibling docs cited inline; the two new design docs (`TransposeFaultDeepDive.md`,
`AsymmetricReshardPlan.md`) are this session's outputs copied in alongside.
Inferences are flagged **[INFER]**; device numbers are MEASURED unless labeled
otherwise.

---

## 1. What is measured + working — per-edge on-chip handoff

The proven capability is the **same-stick cross-core handoff** (`STCDPOpLx`): a
producer→consumer activation slice moves core-to-core over the RIU ring, staying
LX-resident with **zero HBM round trip**, value-correct on device. It is proven
across multiple shapes and sharding regimes, with the mandatory remove-the-senprog
negative control passing in every case
(`CoreToCoreDataMovementRecipe.md` §8/§11; `PerformanceResults.md`).

| Edge / workload | shape | on-chip vs HBM | correctness | source |
|---|---|---|---|---|
| Attention prefill, 3-way | `[1,32,512,128]` | **1.28x** (spyre+kernel) | value-correct, neg-control passed | this session |
| Attention prefill, `B*H=1` | q512 / kv4096 (`report.txt:119`) | **1.15x device / 1.20x wall** | value-correct | this session |
| add-mm `(a+b.t()+c.t())@d` | 512–4096 | **1.13–1.28x** across sizes (peaks mid-range) | value-correct to baseline `max_err` | `PerformanceResults.md` |
| Compiler-driven e2e (add-mm 2048) | 2048 | mixed bundle emitted by `torch.compile`, gate exercised | value-correct, neg-control clean | `PerformanceResults.md`; recipe §5 |

Notes:

- The 3-way attention prefill and `report.txt:119` runs are **this session's**
  device measurements. They corroborate and extend the previously committed
  attention A/B (`PerformanceResults.md` records the seq=512 prefill edge at
  **1.29x** — same ballpark; the 1.28x here is the 3-way `[1,32,512,128]`
  configuration). **[INFER]** the small 1.28x/1.29x delta is configuration
  (3-way vs the prior A/B), not a regression.
- "Compiler-driven e2e" means the win is reachable through the real
  `torch.compile` path (`SPYRE_ONCHIP_HANDOFF_REALIZE=1`) emitting the mixed
  SuperDSC and the patched `dxp` accepting it — no manual splice or artifact
  redirect (`PerformanceResults.md`, recipe §5).
- Both attention numbers are still the **3-region round-trip proof construct**
  (extra ring work); a production **2-region** single move would beat them
  (`PerformanceResults.md`).

**Bottom line:** the on-chip same-stick handoff is proven, value-correct, and
net-positive above the ~1 MB handoff floor on real attention edges and on the
add-mm probe, end-to-end through the compiler.

---

## 2. The frontier map — what it takes to scale to real models

Per-edge wins are proven. Scaling them to whole real-model blocks needs four
distinct unlocks, only one of which is inductor-side. The table is the map; the
subsections give the precise ask and citation.

| Frontier | Owner | Effort | Status | Unlocks |
|---|---|---|---|---|
| Asymmetric same-stick reshard | **inductor** | ~250 LOC | **NEXT** | real-model block edges (granite bmm→mul 8→25) |
| Layout-changing transpose | **deeptools RFC** | — | PTLx wall | Q/K/V + pre-matmul stick reorientation without HBM |
| >4k tiled pipeline | **deeptools RFC** | — | endpoint-residency gap | handoffs whose slice > 2 MB/core LX |
| Dynamic addressing | inductor + runtime | — | needs runtime `memId` | MoE token dispatch/combine routing |

### 2.1 Asymmetric same-stick reshard — PURE-INDUCTOR, ~250 LOC, NEXT

This is the next concrete step and **requires no new deeptools support**. The
proven handoff is symmetric (32→32 matching boundaries, or the reversed
`i↔31-i` mirror). Real-model block edges are **asymmetric**: e.g. the granite
block's producer `batchmatmul` sharded `{out:8, in:4}` across 32 cores hands off
to a consumer `mul` sharded `{out:25}` across 25 cores — unequal, non-aligned
piece boundaries on the same stick (`out`).

The STCDP frontend **already** computes the producer-piece × consumer-piece
overlap cells and the src→dst ring moves: `DcgFE::createSubPieces(STCDPOpLx*)`
loops every output piece × every input piece, intersects rectangles
(`doesPiecesOverlap`), and registers one LX→LX sub-move per non-empty overlap
keyed by src `memId` → dst `memId`. So inductor's only job is to **feed it the
producer and consumer pieces at their native (unequal) sizes** and let DCG do the
cells. The only deeptools dependency is the **existing dxp gate** (recipe §5) —
nothing new.

Scope: ~250 LOC pure inductor (two `PieceInfo` builders + a realize path + edge
detection), composing with the existing symmetric path. Same-stick only;
layout-changing stays the separate (blocked) transpose frontier.

See **`AsymmetricReshardPlan.md`** for the full file-by-file plan, the overlap-cell
algorithm worked on the granite 8→25 edge, the correctness gate, and the STCDP
feasibility verdict.

### 2.2 Layout-changing transpose — deeptools RFC (PTLx wall)

A layout-changing cross-core move (the stick changes, e.g. `out_↔mb_`) cannot be
done on-chip today. The Tier-2 bridge faults on device with a Compute-CB hardware
error (`0x7b1b`). This session **disproved** the leading fix theory and **reversed
the verdict from "integration bug" to "deeptools/hardware wall"**:

- Fix #1 (split into a genuine local per-core transpose + a separate cross-core
  STCDP reshard) landed perfectly **in the descriptor**, but the transpose
  **compute senprog is byte-for-byte identical** to the faulting one (same LXLU
  2048 / L0SU 32768 / PT 64 loop bounds). The per-core element count and
  `subOpInfo` are invariant under the reshard, so the fix could not have removed a
  fault that lives in the compute program.
- The only sanctioned `ReStickifyOpWithPTLx` geometry in all of deeptools is a
  4-dim matmul-internal `out_↔j_` transpose with a tiny (128-deep) per-core piece
  — a **codegen** test only, never executed (no SDSC, DDL, senulator, or device
  test runs a PTLx anywhere in the tree). Our op is a flat 2-dim `{mb_,out_}`
  `out_↔mb_` transpose with a 2048-deep per-core piece — off the trodden path.
- deeptools itself routes around it: PTLx is **unconditionally disabled on the
  next arch** (Sen1.5, `dsm.cpp:12875`).

**The RFC ask (precise):** deeptools should either (a) make `ReStickifyOpWithPTLx`
execute correctly for a **flat 2-dim `{mb_,out_}` `out_↔mb_` transpose of
arbitrary per-core depth** on RCUDD1A/DD2 **and add a senulator/device execution
test** (none exists today), or (b) declare that geometry unsupported and provide a
**sanctioned layout-changing cross-core primitive** (the consumer-endpoint adapter
/ gather→transpose→scatter chain) so inductor can emit Q/K/V and pre-matmul
transposes without an HBM round trip.

See **`TransposeFaultDeepDive.md`** for the senprog smoking-gun table, the source
trace of why the compute is invariant, the exhaustive no-execution-test search,
and the one cheap device experiment (shrink per-core depth to the stock 64-deep
envelope) that would harden the RFC repro. The two raw investigation/validation
docs (`/tmp/transpose_fault_investigation.md`, `/tmp/transpose_fix_validation.md`)
are the prior diagnoses; the deepdive corrects their "integration bug" premise and
their conclusions are cited there.

### 2.3 >4k tiled pipeline — deeptools RFC

The proven single 2-region move fits handoffs up to ~4k hidden (1 MB/core × 2 = 2
MB, with usable LX really ~1.6 MB after the 20% backend reserve). Past 4k the
producer-output + consumer-input slices exceed LX. Move-tiling **alone does not
close this** — it tiles only the transfer buffer, while the producer/consumer DL
ops still need full-slice LX residency, and the realize path flips both endpoints
to 128 KB-spaced buffers → overlap/corruption
(`PerformanceResults.md` "Streaming (>4k) status"; `StreamingImplementationPlan.md`
CORRECTION).

Closing >4k requires a **fused tiled producer→move→consumer pipeline** (produce
tile *k*, hand it off, consume tile *k*, double-buffered) so only a tile is ever
LX-resident — collapsing live LX to a flat `4T + W` independent of S. This is
**Ask 3A** in `TiledOnChipPipelineDesign.md` and is a deeptools surface
(multi-op fused tiled scheduling), not inductor-only.

See **`TiledOnChipPipelineDesign.md`** for the full design and
`Research-2MB-Streaming.md` for the LX-budget analysis.

### 2.4 Dynamic addressing — MoE routing

MoE token dispatch/combine is a **same-stick** all-to-all (so not blocked by the
transpose wall), but the destination core is data-dependent: it needs a **runtime
`memId`**, whereas the proven path uses static, compile-time addressing. This is a
distinct unlock (runtime-addressed STCDP) identified in an earlier session and
recorded here for completeness (`PerformanceResults.md` "Device findings"). Not
worked this session.

---

## 3. Three-way comparison status (sendnn)

The goal of a 3-way comparison is on-chip vs HBM-restickify vs the **sendnn /
torch_sendnn production stack** on the same edges. This session got the sendnn leg
**partially unblocked** but it remains doubly-blocked below the torch-spyre
boundary:

- **The 2.10 rebuild** (isolated `/tmp/torch-spyre-210` + `/tmp/sendnn210-venv`)
  cleared the `torch.ops.spyre.overwrite` registration that the 2.11 build had
  broken. So sendnn loads on the 2.10 stack.
- **Blocker A (generate path):** sendnn `generate` hits `auto_functionalized_v2`
  in the **torch_sendnn converter** — a production-stack handler gap, below the
  torch-spyre boundary.
- **Blocker B (prefill-only path):** prefill-only then hits a **deeptools
  `Dsm::stickify` `map::at` crash** — also below the torch-spyre boundary.

Both blockers are in the production stack / deeptools, **out of inductor scope**.

| Workload | sendnn status |
|---|---|
| granite one-block, tsp-HBM | prefill-forward **50.21 ms** / generate **~2.17 s** (the HBM baseline leg, MEASURED) |
| granite via sendnn, generate | blocked — `auto_functionalized_v2` in torch_sendnn converter |
| granite via sendnn, prefill-only | blocked — deeptools `Dsm::stickify` `map::at` crash |
| attention via sendnn | runs, but **memcpy-dominated / compute-not-isolable** — no clean compute A/B |

**What a sendnn block number would take:** resolve the torch_sendnn converter's
`auto_functionalized_v2` handler **and** the deeptools DSM stickify crash — both
production-stack / deeptools work, out of the inductor team's scope (per the
standing scope note). Until then, the granite tsp-HBM numbers above are the
available baseline leg; the sendnn-attention path runs but is memcpy-dominated so
the compute is not isolable for a fair on-chip-vs-sendnn compute comparison.

---

## 4. Deeptools RFC asks (consolidated)

The two asks below are the only deeptools dependencies on the **scaling** path.
The asymmetric same-stick reshard (§2.1) is **not** an RFC ask — it is inductor
work that reuses the existing `createSubPieces` engine and the existing dxp gate.
(The Foundation contract — mixed-bundle import + binding hook — is already covered
by the main RFC and is demonstrated on device; it is not re-listed here.)

### Ask A — layout-changing cross-core primitive (transpose)

`ReStickifyOpWithPTLx` faults Compute-CB on a flat 2-dim `out_↔mb_` transpose and
has **no execution test anywhere** in deeptools; it is disabled on Sen1.5. Either:

- (A1) make PTLx execute correctly for a flat 2-dim `{mb_,out_}` `out_↔mb_`
  transpose of arbitrary per-core depth on RCUDD1A/DD2, **and add a
  senulator/device execution test**; or
- (A2) declare that geometry unsupported and provide a sanctioned
  layout-changing cross-core primitive (consumer-endpoint adapter /
  gather→transpose→scatter).

Repro + evidence: `TransposeFaultDeepDive.md` (senprog smoking-gun §1.2,
no-execution-test search §3, the disabled-on-Sen1.5 gate §3, and the cheap
hardening experiment §5).

### Ask B — fused tiled producer→move→consumer pipeline (>4k)

A multi-op fused tiled schedule that keeps only a tile (plus its double-buffer
twin) LX-resident, so a cross-core handoff whose per-core slice exceeds the ~1.6
MB usable LX still stays on-chip. This is Ask 3A in `TiledOnChipPipelineDesign.md`
(the endpoint-residency gap; move-tiling alone does not close it).

---

## Source docs

| Doc | Role |
|---|---|
| `NNNN-OnChipRestickifyRFC.md` | the why/what (tier model, Foundation/Transform contracts) |
| `CoreToCoreDataMovementRecipe.md` | the how (proven single-move recipe, §5 dxp gate, §8/§11 ring proof) |
| `PerformanceResults.md` | all device measurements (add-mm, attention A/B, ring signature, streaming status) |
| `AsymmetricReshardPlan.md` | **this session** — the pure-inductor asymmetric reshard plan (§2.1) |
| `TransposeFaultDeepDive.md` | **this session** — the PTLx wall verdict + RFC ask (§2.2, Ask A) |
| `TiledOnChipPipelineDesign.md` | the >4k fused tiled pipeline design (§2.3, Ask B) |
| `StreamingImplementationPlan.md` | the move-tiling CORRECTION (why >4k needs endpoint tiling) |
| `Research-2MB-Streaming.md` | the LX-budget analysis |
| `NvidiaGpuEquivalent.md` | CUDA/GPU translation of the primitive |
