# Warp-Specialized FlashAttention on Spyre — A First-Principles Explainer

*A standalone, read-once guide to the `flash-ws` branch: what codex built, why,
how it actually works in the compiler, what is genuinely proven, and whether it
is worth iterating on. Written from a four-agent deep read of the 99 stage notes,
the two 900-line design docs, and the source. Every load-bearing claim cites a
file:line or a `StageNNN.md:line`.*

---

## 0. The one-paragraph truth

Codex's "warp-specialized FlashAttention" is **not** a GPU warp-group kernel. It
is a **compiler-generated, loader-specialized K/V prefetch sidecar** layered on
top of the existing on-chip SDPA flash-prefill graph. One core (core 31) is given
a "loader" role: it issues a direct **HBM→LX** load of the *next* K/V tile while
the other 31 cores compute the *current* attention tile, then fans that tile out
over the ring into every core's scratchpad so the next matmul reads K/V from LX
instead of re-streaming HBM. It is real, value-correct on standard inputs, and
honestly fail-closed — but on the verified shape island it only **ties** the
strong `onchip_master` baseline (geomean **0.99×**); the "1.15×" headline is
against a *weak* baseline codex itself later disowns. The genuine warp-group
overlap (the real concurrency idea) was tried and **failed value-correctness**
(NaN, 99% mismatch) and was abandoned in favor of the conservative serialized
loader. **Verdict: a real, numerically-sound, exploratory probe — not a shippable
performance win, and crucially it does *not* inherit the cross-shard reshard bug
that broke our sibling `attention-overlap` work.**

---

## 1. First principles: why K/V streaming is the bottleneck

FlashAttention's whole trick is to **never materialize the O(L²) score and
probability matrices in HBM**. It tiles the attention so each Q tile streams
through all of K/V, accumulating softmax statistics online. On Spyre the score
matrix for one batch-head at L=1024 is 1024×1024 = ~1M elements — quadratic in L
— and FlashAttention removes it (`Stage094:88-92`).

Once that O(L²) traffic is gone, **the next bottleneck is the K/V stream itself**
(`Stage094:38-41`):

```
K/V streaming traffic  ≈  2 · L · D · bytes · (L / Bq)
                          └── per-tile K/V ──┘ └ #Q tiles ┘
```

For L=1024, D=64, Bq=64 the K/V stream is re-read across **16 Q tiles**
(`Stage094:152-178`). Every one of those reads hits the **one shared HBM pipe**.

### The hardware fact that makes this matter

```
        Spyre AIU memory hierarchy (the asymmetry that defines the problem)

   ┌─────────────────────────────────────────────────────────────┐
   │  HBM   ── ONE shared pipe ──   ~170 GB/s peak (~100 realized) │  ← bottleneck
   └─────────────────────────────────────────────────────────────┘
        ▲                                                   contended by all 32 cores
        │  every Q tile re-reads all K/V through here
   ┌────┴───────────────────────────────────────────────────────┐
   │  LX scratchpad   ~2 MB/core · ~51 MB aggregate · ~4.5 TB/s   │  ← ~45× the HBM rate
   └──────────────────────────────────────────────────────────────┘
        ▲
        │  RIU ring moves data core↔core: ~166 GB/s/dir/core, ~idle today
   ┌────┴─────────────────────────────────────────────────────────┐
   │  32 cores · per-core units run independent programs:          │
   │  L3LU/L3SU (ring DMA) │ PE (matmul) │ SFP (softmax/act)        │
   └───────────────────────────────────────────────────────────────┘
```

The lever is obvious from the asymmetry: **move the K/V handoff off the slow
shared HBM pipe and onto the fast, idle on-chip fabric**, and **overlap the
unavoidable next-tile HBM fetch behind current compute**. Codex states the goal
precisely (`Stage094:180-183`): *"make the remaining K/V stream land where the
future batchmatmul needs it, with less HBM pressure and without corrupting
current compute."*

---

## 2. What "warp specialization" means here (and what it does NOT)

On a GPU, warp specialization = different warp groups in **one kernel** issue to
different units **concurrently** (producer warps fire async copies, consumer
warps do MMA), software-pipelined through a multi-stage buffer. Codex is
repeatedly explicit that the Spyre version is an **analogue**, not that
(`Stage089:331-359`, `Stage094:351-374`):

```
   GPU concept            Spyre realization (this branch)        Honest difference
   ─────────────────────  ─────────────────────────────────────  ──────────────────────
   Producer warp       →  Loader core 31                          a scheduling role, not a
                                                                  warp; core 31 still has
                                                                  its own SERIALIZED slice
   Consumer warps      →  Cores 0–30 (current-tile compute)       —
   Shared memory       →  LX scratchpad regions                   addressed by compiler
                                                                  PieceInfo metadata
   Async copy          →  STCDPOpHBM inside a mixed SDSC          ordered by generated
                                                                  schedule; no async-copy API
   Warp barrier        →  `nop` rendezvous rows + schedule order   no hardware barrier
   Multi-stage buffer  →  ONE look-ahead stage                    double-buffer, not a deep
                                                                  pipeline
   Multicast           →  STCDPOpLx fanout (a tuning knob)        transport choice, not the
                                                                  correctness invariant
```

So "warpspec" here = **a producer/consumer core-role split + explicit on-chip
staging**, gated by strict metadata (`Stage089:349-353`). It is a
*loader-specialized attention schedule*, not a hand-written kernel — and in the
source it isn't even a code path: **the string `warpspec` appears nowhere in
`onchip_realize.py` or `bundle.py`** (Agent B, grep = 0 hits). It is a *config
preset name* (§5).

---

## 3. The mechanism — what the compiled bundle actually is

The unit of scheduling is the **SDSC** (one generated device op). A **mixed
SuperDSC** carries data-ops (`STCDPOpHBM`, `STCDPOpLx`, `nop`) *plus* a DL compute
op under an explicit per-core schedule (`Stage089:119-147`). The standard SDPA
graph is ~11 SDSCs (`mul, mul, ReStickify, bmm(QKᵀ), max, sub, exp, sum, realdiv,
bmm(PV), identity`); warpspec **rewrites the K/V-producer SDSC in place** into a
loader-specialized mixed SuperDSC — it is spliced into `bundle.mlir`, *not* a
separate kernel (Agent B; `bundle.py:1030-1073`).

### The pipeline, in time

```
   time ──────────────────────────────────────────────────────────────────▶

   cores 0–30:   prologue │ COMPUTE current tile        │ barrier │ consume fanned K/V │ future compute
                          │                             │  (nop)  │                    │
                          │      ⟍ overlaps ⟋           │         │                    │
   core 31:      prologue │ STCDPOpHBM: load FUTURE K/V │ own     │ STCDPOpLx fanout   │ future compute
                          │   (HBM→LX, direct fill)     │ slice   │ (ring multicast)   │
                                                          serial

   INVARIANT (Stage078 / Stage089:200-216):
     core 31 must NOT run its own current-tile compute in the SAME window
     as its STCDPOpHBM future load.  This is the correctness rule that
     survived after the "real overlap" variants failed (§6).
```

### The fanout topology — single loader, not all-core fill

```
            HBM[ future K/V tile ]
                   │
                   │  STCDPOpHBM   (onchip_realize.py:2808 — "direct HBM→LX
                   ▼                fill with no LX→LX roundtrip")
            core 31 LX source buffer
                   │
                   │  STCDPOpLx full-tile-piece fanout  (ring multicast)
        ┌──────────┼──────────┬─────── … ───────┬──────────┐
        ▼          ▼          ▼                  ▼          ▼
     core0 LX   core1 LX   core2 LX           core30 LX  core31 LX
        └──────── the FUTURE batchmatmul reads its K/V operand from LX ────────┘
```

Three storage identities live at once (`Stage094:222-236`): (a) the **original
HBM producer output** — *preserved*, because reductions like max/sum may still
read it (`Stage089:188-193`); (b) the **loader LX source buffer**; (c) the
**per-consumer-core LX input regions**.

### How the schedule is built (source, Agent B)

- **Entry:** `_apply_flash_attention_route_policy()` (`decompositions.py:86`,
  called at `:667`) → `apply_flash_attention_route_policy()`
  (`flash_attention_route_policy.py:110`). This *is* the "core selector" of the
  latest commit.
- **Builder:** `build_flash_attention_kv_repack_hbm_prefetch_hoist_tile_artifacts`
  (`onchip_realize.py:4540`). Deep-copies producer/consumer SDSC bodies and
  field-surgeries them with `apply_lx_flip` (`:4953` — which only **redirects an
  LX base pointer**, it does NOT re-partition a shard; this is why it is safe —
  see §8).
- **Schedule rows** `[dataopIdx, dlIdx, waitFlag, signalFlag]` are emitted by
  `_kv_repack_hbm_prefetch_source_fanout_schedule` (`onchip_realize.py:4217`):
  per core, pre-compute rows, then (loader only) serialized HBM-source-load rows,
  a barrier row, then the STCDPOpLx fanout row.
- The data-ops themselves: `_stcdp_op` (`onchip_bridge.py:317`); the mixed
  SuperDSC schema "matches deeptools SuperDsc JSON byte-for-byte"
  (`onchip_bridge.py:15-27`).

---

## 4. The lineage — warpspec is the last step of a chain

Warpspec is a **specialization on top of** the on-chip handoff machinery, not a
separate invention. The progression:

```
  Stage002–004           Stage005–007        (umbrella)         Stage008            Stage079+
  SCORE HANDOFF    →   MoE STATIC ROUTING →  ON-CHIP REALIZE  → PIPELINE PROOF  →  WARPSPEC
  keep QKᵀ score       same realize           "keep a producer    double-buffer      adds the certified
  in LX; softmax       machinery on a          →consumer edge      schedule, but      HBM→LX STCDPOpHBM
  reads it without     different graph         in LX not HBM";     LX→LX ONLY —       future-K/V prefetch
  HBM round-trip       (dispatch/scatter)      bind by LX-base     explicitly does    + loader-core role
                                               coincidence         NOT claim HBM→LX   + ring fanout
                                               (LX doesn't persist  (Stage008:65-75)
                                                across sdsc_execute)
```

The key conceptual jump: **Stage008 proved a double-buffered schedule but only
for LX→LX moves and explicitly deferred HBM→LX prefetch** (`Stage008:65-75`).
Warpspec is the generalization that finally crosses the **HBM→LX** boundary,
scoped to the K/V operand of flash prefill, reusing the handoff binding
(LX bases, mixed SDSC, preserve-original-HBM-output) wholesale
(`Stage089:454-465`).

---

## 5. Layout decoupling — the turning point (Stage088)

There were originally **two coupled mechanisms**: (1) a **layout-transform pair**
that restickifies/repacks K/V into a new on-chip layout, and (2) the
**loader-core K/V prefetch** schedule. Codex found these are *separable
correctness concerns* and that coupling them was actively harmful
(`Stage088:21-25`):

> "Coupling the two **hid valid warpspec rows behind layout-pair numerical
> failures.**"

```
   BEFORE (coupled):                          AFTER (decoupled, Stage088):
   ┌────────────────────────────┐             ┌──────────────────────────┐
   │ layout-transform pair      │  BROKEN     │ loader-core prefetch      │  ✓ value-correct
   │   (restickify/repack K/V)  │  long rows  │   (HBM→LX + fanout)       │    long-row island
   │            +               │  max abs    │   LAYOUT_XFORM = 0         │    RECOVERED
   │ loader-core prefetch       │  0.21–0.67  │   LAYOUT_XFORM_PAIR = -1   │
   └────────────────────────────┘             └──────────────────────────┘
        ↑ the 0.21–0.67 mismatch (Stage086/087) is the SAME failure signature
          as the bug that broke our sibling attention-overlap branch (§8)
```

Disabling the layout pair (`SPYRE_..._LAYOUT_XFORM=0`,
`..._LAYOUT_XFORM_PAIR_TILE=-1`) **recovered** B1/H4/D64 L768/L1024 and the whole
B2/H4/D128 long island (`Stage088:28-31`). The certified default target is now the
**layout-free** loader-core path. Durable lesson (`Stage089:851-857`): keep three
evidence streams separate — layout-transform correctness, loader-prefetch
correctness, and combined correctness.

**This is the single most important engineering decision on the branch**, and it
is the reason the work avoided the trap our sibling branch fell into (§8).

---

## 6. What is actually proven — the honest evidence ledger

This is where the headline and the reality diverge. (Agent C, all cited.)

### The "best" number, and its asterisk

```
   SDPA prefill, 8-row island (B1H4D64, B1H8D64, B2H4D128; L384–1024; block64; non-causal)

   vs  flash_hbm   (WEAK baseline)   ── geomean 1.1518×  ──  looks like a win   (Stage092:38)
   vs  onchip_master (STRONG base)   ── geomean 0.9929×  ──  a NET LOSS         (Stage092:40)
                                          4 of 8 rows win, best 1.0349×
                                          (B1H4D64 L768); worst 0.9559×
```

- `flash_hbm` is **not** the production baseline. Codex itself later relabels it
  as "not equivalent to upstream" (`Stage098:200`).
- The decisive three-way table (upstream-main vs onchip_master vs route-policy)
  **was never produced** — the clean upstream build failed to compile against the
  runtime (`Stage098:234-250`).
- The in-compiler route-policy run (Stage099) measured **0.9892× geomean vs
  master** — i.e. net *slower* in that run (`Stage099:126`).
- **All timings are host wall-time, not AIU kernel-time** (`Stage098:21-28`),
  warmup 1–2 / iters 1–7, medians only — codex's own "diagnostic, not production"
  caveat (`Stage089:698-700`).

```
   Verified performance ceiling:
   ────────────────────────────────────────────────────────────────────
   onchip_master  │██████████████████████████████████│  1.00×  (the bar to beat)
   warpspec       │█████████████████████████████████▉│  0.99×  (ties / slightly under)
   flash_hbm      │█████████████████████████████      │  0.87×  (the weak baseline
                                                                  the 1.15× is measured against)
   ────────────────────────────────────────────────────────────────────
   "Does not yet prove a meaningful AIU kernel-time speedup." — codex, Stage098
```

### What was tried and FAILED (the real concurrency idea)

The genuine warp-group same-row overlap — the actual FlashAttention pipelining —
**broke value-correctness and was abandoned**:

| Stage | What | Result |
|---|---|---|
| **052** | Forced same-row IFN-prefix overlap (the real mechanism) | **99.2% mismatch, NaN** — single-SDSC IFN row has no real producer (`Stage052:90-96`) |
| **053** | Predecessor-backed same-row overlap | **99.1% mismatch, NaN** — read-after-write hazard (`Stage053:113-119`) |
| **035/037** | Independent prefetch sidecar sharing the compute row | 77% mismatch — the row shape itself is unsafe (`Stage037:64-69`) |
| **082** | block128 long rows | max abs 0.254 / 0.120 — numerically broken (`Stage082:52-53`) |
| **086/087** | layout-coupled long rows | max abs 0.21–0.67 — fixed only by *removing* the overlap (§5) |

The arc is honest: **every "values wrong" result is a probe codex recorded as
failed and walked away from.** What survived is the conservative
serialized-loader retreat — value-correct, but with the loader's own compute
slice *not* overlapped, which is the fixed cost that keeps it at break-even.

### Maturity: a probe, not a product

- **Nothing is default-on.** Every gate defaults `"0"`/`"-1"` (`config.py:55,
  308-489`); the SDPA path is inert unless env vars are set.
- **~119 of 217 commits are docs-only** stage notes (Agent D).
- **The route policy is a hardcoded 4-shape lookup table** baked from one sweep
  (`flash_attention_route_policy.py:31-38`), not a cost model. Codex admits "no
  implemented production routing policy" (`Stage094:757`).
- **The causal path is plan-artifact-only** — `causal_mask_dataop.py:456`
  self-reports `requires_deeptools_dataop_parser_extension: True`, written to disk
  but never added to `bundle.mlir`. Causal does not compose with warpspec today.
- The ~7k-line `onchip_realize.py` growth is dominated by ~30 diagnostic A/B knob
  paths — bring-up scaffolding (Agent B).

---

## 7. The route policy, precisely

The promotion gate proves *correctness*; the route policy answers *is warpspec
the fastest route for this shape* (`Stage095:58-73`). It is **two layers**:

```
   Stage095 (offline)                    Stage099 (in-compiler)
   ───────────────────                   ──────────────────────────
   tools/onchip_sdpa_route_policy.py     flash_attention_route_policy.py
   reads perf-compare JSON               at SDPA decomp time, reads the shape
   emits a shape→variant table           (B,H,D,block,causal,L), looks it up
   rule: pick warpspec only if           in the 4-entry frozenset, and if matched
     base_ok ∧ target_ok ∧               MUTATES global config to the decoupled
     base_ms/target_ms ≥ min_speedup     preset, else applies onchip_master
   (min_speedup = 1.0)                    (decompositions.py:667 → :86)
```

The implicit rule is **"long L → warpspec, short/mid → master"** — because the
warpspec fixed overhead (serialized loader slice + extra fanout rows) only pays
off when K/V movement dominates it. But this is a **measured crossover, not a
predicted one** — there is no closed-form roofline anywhere in the 1800 lines of
"first-principles" docs (Agent A). A senior reviewer's question stands: *where is
the bandwidth/overlap arithmetic that predicts the crossover L instead of
memorizing it?*

---

## 8. The question that actually matters for us: does it inherit our bug?

**No — and this is the most important finding.** (Agent D, decisive.)

On our sibling `attention-overlap` branch, the productionised cross-shard handoff
broke with `max_err 0.669` on standard inputs. The bug lived in
`realize_asymmetric_onchip_handoff` — a producer-`{mb:32}` → consumer-`{x:32}`
**reshard** mis-strided in the STCDP overlap-cell engine, masked for a while
because tests used benign `randn*0.1` inputs.

```
   OUR broken path (attention-overlap)         CODEX's certified path (flash-ws)
   ───────────────────────────────────         ─────────────────────────────────
   realize_asymmetric_onchip_handoff           build_..._hbm_prefetch_hoist_tile
     producer {mb:32}  ─reshard─▶               STCDPOpHBM: HBM→LX DIRECT FILL
     consumer {x:32}                              "no LX→LX roundtrip"
     (overlap-cell re-partition)                  (onchip_realize.py:2808)
            ▼                                            ▼
     MIS-STRIDED → max_err 0.669                  same-layout full-tile fanout
            ✗ BROKEN                              apply_lx_flip only redirects an
                                                  LX BASE POINTER, no re-partition
   masked by benign randn*0.1                          ✓ value-correct
                                                  validated on STANDARD randn,
                                                  error gate ≤ 0.01
```

Three concrete confirmations:

1. **`realize_asymmetric_onchip_handoff` does not exist on `flash-ws`** — grep
   returns zero hits (Agent D). The broken function simply isn't there.
2. **The certified path is a same-layout fill + fanout, not a reshard.** It loads
   K/V HBM→LX directly (`onchip_realize.py:2808`) and fans the *whole tile* out;
   no mb→x re-partition occurs. `apply_lx_flip` (`:4953`) only moves a base
   pointer.
3. **Codex independently found and routed around the same failure class.** The
   *layout-coupled* variant failed with the exact `0.21–0.67` signature
   (Stage086/087) — and codex's response was to **disable the layout transform**
   (`LAYOUT_XFORM=0`) and certify the layout-free path (§5). It diagnosed the
   risky reshard and walked away, on standard inputs, with a fail-closed gate.

And the validation is rigorous: standard `torch.randn` (no `*0.1`), CPU
`F.scaled_dot_product_attention` reference, `max_abs_error ≤ 0.01` promotion gate
(`onchip_sdpa_sweep.py:1301-1328`, `onchip_sdpa_promotion_gate.py:520`). Promoted
rows report 0.002–0.006. **This branch avoided the trap that bit us.**

---

## 9. Outlook — should we iterate?

**Iterate for the mechanism, not for the current numbers.** The honest scorecard:

| Dimension | Assessment |
|---|---|
| Mechanism real? | **Yes** — a real mixed SuperDSC, executed on device, spliced into `bundle.mlir` |
| Numerically sound? | **Yes** — standard randn, fail-closed gate, 0.002–0.006 error on certified rows |
| Inherits our cross-shard bug? | **No** — different, simpler, safer mechanism (HBM→LX fill, no reshard) |
| Perf win over the STRONG baseline? | **No** — geomean 0.99× vs `onchip_master`; the 1.15× is vs a weak baseline codex disowns |
| The real overlap idea? | **Failed** (NaN/99% mismatch) and was abandoned for a serialized loader |
| Coverage | **Narrow** — 8 rows, block64, non-causal, H8-long broken, causal plan-only |
| Productized? | **No** — default-off, lookup-table "policy", scaffold-shaped code |

### What would have to become true for iteration to pay off

1. **Beat `onchip_master`, not `flash_hbm`** — on a documented shape island. Today
   the serialized loader's own compute slice is a fixed cost only long rows
   amortize; the win is within noise (`Stage094:711`).
2. **Recover real overlap safely** — the same-row concurrency that failed
   (Stage052/053, read-after-write / no-producer) is *the actual idea*. The
   serialized-loader retreat is a correctness workaround, not the destination.
   This is the high-value, high-risk frontier.
3. **A real route policy** — replace the 4-shape frozenset with a cost model that
   *predicts* the crossover L (the first-principles arithmetic the docs narrate
   but never derive).
4. **Kernel-time measurement** — all current numbers are host wall-time, tiny
   samples; re-measure on AIU kernel-time before trusting any speedup.
5. **Generalize** past the island: batch>1 D128, H8-long, causal, block128 —
   each currently fails or is plan-only.

### The one-line recommendation

The loader-core prefetch is a **legitimately different and safer design** than our
asymmetric handoff — building on it does not inherit our bug, and codex's
rigor (standard inputs, fail-closed gate, honestly-recorded failures) makes it a
trustworthy foundation. **But treat the perf as unproven**: the certified result
ties the strong baseline, the real concurrency idea is still unsolved, and the
highest-value next step is exactly the thing codex could not make value-correct —
genuine same-row producer/consumer overlap without the read-after-write hazard.

---

## Appendix — where to look

| Topic | Location |
|---|---|
| First-principles design | `docs/source/rfcs/drafts/NNNN-OnChipRestickify/Stage089-...md`, `Stage094-...DeepDive.md` |
| Layout decoupling | `Stage088-WarpspecLayoutDecoupling.md` |
| The failures | `Stage052`, `Stage053`, `Stage035`, `Stage037`, `Stage082`, `Stage086`, `Stage087` |
| Perf envelope | `Stage092-DecoupledWarpspec8RowPerf.md`, `Stage098` (timing scope) |
| Route policy | `Stage095`, `Stage099`; `torch_spyre/_inductor/flash_attention_route_policy.py` |
| Entry / dispatch | `decompositions.py:86,667` |
| Builder | `onchip_realize.py:4540`; schedule `:4217`; HBM→LX fill `:2808`; lx-flip `:4953` |
| Bundle wire-in | `bundle.py:114` (`fold_onchip_handoff`), `:1030-1073` (splice) |
| Mixed-SDSC synthesizer | `onchip_bridge.py:15-27,317,351` |
| Causal (plan-only) | `causal_mask_dataop.py:456-463` |
| Config surface | `config.py:45-627` (~80 `SPYRE_FLASH_ATTENTION_*` flags, all default-off) |

*Sources: a four-facet read-only investigation of `flash-ws` (design intent,
source mechanism, evidence ledger, adversarial review), the 99 stage notes, and
the branch source at tip `10ec5a4`.*
