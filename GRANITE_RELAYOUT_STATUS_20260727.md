# Granite relayout — verified status, 2026-07-27

Two things are stacked here. The **ledger below** is an adversarial re-derivation of
every edge from the raw run artifacts: 17 agents harvested each track and a second
pass tried to refute what the first found. It is deliberately hostile to the running
narrative, and it overturns several claims that had been treated as settled.

This preamble records what changed *after* that ledger was written.

## Superseding updates from this session

**P07 now runs.** The ledger says "no run ever reached the device". That was true of
the shared-PVC track (`p07/torch-spyre`, 284 insertions). A second, later P07 lane
existed on the `adnan-cdx` pod — whose home is *not* on the shared PVC and which the
harvest therefore never saw — carrying 564 insertions. It failed with a different
error, an `IndexError` in `parse_op_spec`, which is now root-caused and fixed
(`inductor: extend short dim-label overrides…`). With that fix P07 compiles and emits
the four independent row shuffles it was designed to produce, and generates the
correct one-layer token 44. It is still one-layer and unintegrated; full-40
integration is in progress.

**P09 has a full-40 run.** The ledger's P09 verdict ("no full-40 run exists; 203
appears zero times") again describes the shared-PVC track. The cdx lane contains
`p09_full40_a`, a full 40-layer run generating token **203** with 5 relayout plans.
That materially changes P09's status — though the ledger's substantive warning stands
and should be read first: at one layer, P09's logits differ from their own control on
99.75% of positions, so its *correctness* is not established merely by an argmax.
That run also captured no transport dump, so its LX gate is unverified.

**The 247 ms result reproduces**, independently, from the archived patch alone, on a
different pod: 246.322 ms median, token 203 on 6/6. Treat 247 as the headline; the
delta is pod variation.

## One conflict left open, deliberately

The ledger states that **op 45 is not the P08 edge** — that it is the pre-existing P12
residual-add shuffle, renumbered, and that the real P08 edge is **op 38**. The
original P08 report claims the opposite, and the reproduction's own traffic table is
consistent with the report: op 45 is the only shuffle with a 16-source → 32-destination
topology and 1 MiB local / 3 MiB remote, exactly the figures the report quotes.

I have not resolved this and am not asserting either side. It is settled by one cheap
experiment: run the identical stack with `SPYRE_RELAYOUT_ORACLE_PREFILL_ATTN_PERMUTATION=0`
and see which shuffle disappears. Until then, treat P08's attribution as unverified.

This matters more than a numbering quibble: if the ledger is right, the reproduction's
"P08 topology matched exactly" check validated the wrong operator.

## The single most important number here

The ledger's cross-cutting finding is that the promoted edges are not where the time
went. P12 is worth −7.465 ms and fires 40× per request; P13 and P14 show **no**
whole-request win at all. Meanwhile the **SwiGLU → down-projection 16×2 split is worth
~18.2 ms — roughly 12× P08's contribution** — and it is not one of the numbered relayout
edges. Anyone resuming this should weigh that before spending another day on edges.

---

## Where this stands

The goal was to stop Granite 3.3 8B's prefill from bouncing activations through HBM. At eleven producer→consumer boundaries in the model, we rewrote the layout/ownership map so the data hands off core-to-core over the on-chip LX network instead. Three of those edges — P12 (post-attention residual add), P13 (last-token → LM head) and P14 (fused final norm → last-token slice) — cleared a full promotion gate: token 203 on every request, logits bit-identical to the relayout-off control, and post-PCFG payloads showing `STCDPOpLx` with zero HBM or DMA ops. Together they are worth **−4.65 ms (1.19%)** of device kernel time on a clean T-C-T-C bracket; that is the only whole-request win in the project that survived adversarial re-derivation from raw traces. Four more edges (P03, P05, P06, P08) run correctly and transport cleanly but each has a hole: P03 has no stock baseline and is not bit-exact against its control, P05 measures to zero, P06 has no off-control at all, and P08's exact-parity variant produces the wrong token. P07 has never executed on hardware; P09 was dropped and never had valid correctness evidence at any scope. Everything is uncommitted working-tree diff on `59545440`.

## Edge ledger

| Edge | What it relayouts | Status | Measured device median | Correctness gate | Transport proof | Worktree holding it |
|---|---|---|---|---|---|---|
| **P03** | RMSNorm output `buf52` → gate/up projection BMMs `buf53`/`buf55`. 8×4 producer grid all-gathered ×4 so each matmul core holds the whole 4096-wide hidden vector. | MEASURED-NOT-PROMOTED | **299.65 ms** LX vs **428.95 ms** HBM (−129.3 ms / 30.1%). Median of 2 runs per arm, **1 profiled request each** — the 5-request protocol was not met. Counting device memset/memcpy too: 357.3 vs 466.2 ms (23.4%). | PASS — 203 in all 4 full-40 runs, both requests, argmax re-verified from logits. **Not bit-exact vs control** (max Δ 0.279). | Clean — 2 payloads, `STCDPOpLx`=1 each, 0 HBM / 0 ReStickify / 0 Dma | `p03/torch-spyre` (1075 ins, uncommitted) |
| **P05** | RMSNorm input `buf47` → mean reduction `buf48`. 32 producers → 8 owners, reducing gather ×4. | MEASURED-NOT-PROMOTED | **265.705 ms** (on) vs **265.941 ms** (off), same pod, same tree, 5 requests each → **Δ 0.24 ms, inside a ~1 ms within-run spread**. Isolated pair: 394.454 vs 390.648 → ~4 ms *slower*. | PASS — 203, 6/6 requests | Clean — `STCDPOpLx`=1, 0 HBM / 0 Dma; delivery split measured 1 local / 31 remote | `p05_p06_integration/torch-spyre` (accepted map). `p05_owner_map`, `p05_sparse_owners` hold rejected variants, still in failing state |
| **P06** | Rotary query output `buf14` → QK batched matmul `buf20`. | MEASURED-NOT-PROMOTED | Stack median **267.179 ms**, 5 requests. **P06's own contribution is unverified** — no P06-off control exists anywhere in any track. | PASS — 203, 6/6 requests | Clean — `18_shuffle` `STCDPOpLx`=1, 0 HBM / 0 Dma | `p06_completion/torch-spyre` (P06 is ~20 of its 1558 lines) |
| **P07** | Shared RoPE frequency input → Q rotary `buf12` and K rotary `buf16`. | IN-PROGRESS | unverified — no run ever reached the device; every `trace/` directory is empty | NOT REACHED — zero tokens generated in any of 5 runs | Q-side only: one clean `STCDPOpLx` payload, 0 HBM / 0 Dma. K-side payload never emitted (compiler aborted first). | `p07/torch-spyre` (284 ins) |
| **P08** | Attention output `buf40` → `buf41`, head-major to token-major permutation. | MEASURED-NOT-PROMOTED | **247.407 ms** vs matched same-stack control **248.937 ms** → **−1.530 ms**, 5 requests each | PASS — 203, 6/6. The exact 32→32 SenDNN-source variant **fails**: token 29. | Clean — the edge is **op 38** (not 45), `STCDPOpLx`=1, 0 HBM / 0 Dma, and it is the one payload that differs between the failing and passing configurations | `p06_completion/torch-spyre`. `p05_p08_combined/torch-spyre` carries the same P08 code but was **never executed** |
| **P09** | Normalized activation `buf10` → Q/K/V projection BMMs `buf11`/`buf15`/`buf29`. | REJECTED (dropped) | unverified — all 6 runs are one-layer with a single profiled request; the ~20.3 vs 20.5 ms one-layer sums are not the gate metric and are swamped by the LM-head kernel | NOT ESTABLISHED — no full-40 run exists; "203" appears zero times in every P09 log. Treated logits differ from their own control on **99.75%** of positions (Pearson 0.396). | Clean at one layer — 3 payloads, `STCDPOpLx`=1 each, 0 HBM / 0 Dma | `p09/torch-spyre`. Absent from `final_prefill_integration_20260727` |
| **P10/P11** | RMSNorm scalar chain `buf50` → `buf51`, grouped multicast back to 4-core cohorts. | MEASURED-NOT-PROMOTED | **264.478 ms**, 5 requests. **No matched control** — this is a stack median, not a P10/P11 delta. | PASS — 203, 6/6 requests | Clean — 9 payloads incl. `51_shuffle`, `STCDPOpLx`=1 each, 0 HBM / 0 Dma. Sparse ownership visible in the payload (memIds 0,4,…,28). | `p08_p10_p11_completion/p05_p10p11_integrated` |
| **P12** | Post-attention residual add `buf45` → `buf46`. 16 even-core owners → 8-token × 4-hidden grid. **Fires once per decoder layer, i.e. 40× per request.** | ACCEPTED | **390.363 ms** vs **397.828 ms** → **−7.465 ms / 1.88%**, T-C-T-C, 5 requests per run | PASS — 203, 6/6 in all four runs | Clean — `45_shuffle` `STCDPOpLx`=1, all 176 memory-org entries `lx`, 0 HBM / 0 Dma | `p12/p12_patch_worktree` (authoritative; this is what the clean runs loaded, **not** `p12/torch-spyre`) |
| **P13** | Last-token hidden vector `buf0` → 28-owner LM head `buf1`. 32-source all-gather onto 28 destinations, K-reduction left unsplit. | ACCEPTED | Whole request: **371.325 vs 370.510 ms** → **+0.8 ms, i.e. slightly slower, inside noise**. Final-head kernel only: **3.42 vs 4.00 ms → −0.57 ms / −14%**. | PASS — 203, 6/6 in all four clean runs | **Partial** — no post-PCFG payload exists in *any* full-40 P13 run. The clean `STCDPOpLx`=1 / 0-HBM payload comes from a 4-kernel isolated compile that generated token 44. | `p13_clean_20260726/torch-spyre` (198 ins, genuinely edge-isolated) |
| **P14** | Fused final RMSNorm output `buf5` → last-token slice `buf6`. Cores 28–31 (which physically hold token 511) → all 32, destination is 1/512 of source. | ACCEPTED | Whole request: **no win** (T 378.9/379.3 vs C 379.0/379.7; a second bracket shows T *slower*). Final-stage kernel: **−148.6 µs / −3.5%**, reproduced to 0.2 µs in two independent brackets. | PASS — 203 on 24/24 requests across four runs; logits **byte-identical** to control (`64e9304d…`) | Clean — post-PCFG `7_shuffle` `STCDPOpLx`=1, 0 HBM / 0 Dma, literal "HBM" absent from the file. Decoded init packet: 32 transfers × 256 B, 31 remote + 1 local. | `p14/p14_promotion_clean_20260726/torch-spyre`; hunk-minimal set in `p14_minimal_20260726` |

**Combined P12+P13+P14:** treatment **387.657 ms** vs control **392.309 ms** → **−4.65 ms / 1.19%**, T-C-T-C, 5 requests per run, within-run scatter under 0.4 ms, arms never overlapping, all three edges proven on LX in the same artifact that generated token 203. This is the strongest result in the project.

Notes on the table:
- Every number is a **device kernel-time median** (sum of `cat=="kernel"` durations per request), independently re-derived from the Kineto traces. Host "First-token latency" figures (~1.1–2.6 s) appear in the same logs and are never the metric.
- The P06/P08 lane (247–273 ms) and the promoted non-attention lane (~387 ms) run different stacks and different fusions (42 vs 43–44 kernels/request). **Do not compare absolute medians across lanes** — only within a matched pair.
- Two pods were used (`adnan-spyre-current-pf`, `adnan-spyre-dev-pf`). Identical code measures ~0.9 ms apart across them. Do not compare across pods either.

## Corrections to the running narrative

**P03**
- The ~1075-line diff overstates what the runs exercise. The gather *planning* and the 4× destination allocation already exist at base commit `59545440`; every added line in `lx_relayout.py`, `work_division.py` and `superdsc.py` sits behind oracles the P03 run script sets to 0. The measured runs differ from base only by the `spyre_kernel.py` codegen fixes.
- "Only the consumer descriptor changed" between the failing and passing probe is wrong — the **producer** descriptor changed too, with the same `coreIdToWkSlice_` permutation. The fix rewrote ownership at both endpoints.
- "30.1%" counts `cat=="kernel"` only. The LX arm also adds 22.5 ms of device `gpu_memset` (2.2× the control). On the broader reading the win is 23.4%.
- The transport and control logits are **not** bit-identical (max Δ 0.279 on the [1,49280] fp16 logits). Token 203 wins with margin, but bit-exactness against the HBM path is not established — this is why P03 sits outside the promoted stack.
- The "5 measured requests" gate was never met: Kineto clears events each cycle, so each trace holds exactly one profiled forward pass.

**P05**
- The position that a "sparse owners" variant passed token 203 is **false**. That run has no `run.log` at all; its two logit dumps carry `next_val=[0]` and all 49280 entries are `+inf`. Both routes to an 8-local/24-remote map (mb-fast producer, sparse-owner consumer) fail identically.
- The 270.842 ms headline run wrote its outputs into `p05_owner_map/runs` but actually executed `p14/torch-spyre` (confirmed from the log's own warning paths). p14 carries the identical accepted P05 code, so the number stands — but the provenance label was wrong.
- The isolated "+3.8 ms slower" rests on a 3-sample control spanning 9.87 ms. Sign is robust (min-vs-min gives +4.34 ms); the precise magnitude is not.
- Only the mb-fast route's 8-local/24-remote split was *measured*; for the sparse variant it is inferred from plan geometry, never observed.

**P06**
- The claimed −6.213 ms has no baseline. No P06-off run exists anywhere under `device_parity_tracks_20260726`, and the "273.392 ms prior median" appears in no run directory.
- The flattened-`buf14` fix is **inert**. Every `relayout_plans.jsonl` records `buf14` at the rank-5 `[1,8,4,512,128]` shape that base already accepts, and `plans.jsonl` reads the same `producer.get_size()` expression the gate tests — so the branch never fired in any run cited as its acceptance evidence. No unit test covers it.
- `p06_attention`'s unit test contains **zero assertions** and never calls the function under test; it would pass vacuously. The assertions exist only in a sidecar `.patch` whose hunk header (84 lines) contradicts its body (92 lines).
- "The run.log contains no HBM markers" is vacuous evidence — the log prints no op-type names at all (`STCDPOpLx` count is also zero). Cite the payload dumps only.
- The 16×2 down-projection is a *trade*, not a strict residency improvement: the slower 32×1 control actually has more LX-resident MLP buffers.

**P07**
- The emitted shuffle shards along the **64-wide rotary-frequency axis**, not the 512-token axis. The allocator explicitly asks for a 16-way split on the token axis and does not get it. Since the abort is "query fold dimension with higher fold factor" and the 64-wide axis *is* the stick dimension, this is plausibly the direct cause — "fix the fold abort" and "fix the transfer geometry" are one task, not two.
- The `p12/deeptools` path in the `DtException` is a compile-time source path baked into the p05 build (69 matching strings in `libdxp.so`). There is no build mismatch. Dead lead.
- The pytest cache shows 18 collected tests, not 19, and only 3 are new. No saved pytest output exists.

**P08**
- **Op 45 is not the P08 edge.** It is the pre-existing P12 residual-add shuffle (`buf45`→`buf46`), renumbered because P08 inserts an earlier op; it appears content-identical in the P08-*off* control. The real P08 edge is **op 38**, whose descriptor geometry matches `[1,8,4,512,128]` and which exists in no P08-off run. Consequence: the earlier finding that "the P08 transport artifact is byte-identical in the token-29 and token-203 runs and therefore proves nothing" **collapses in P08's favour** — op 38 is exactly the one file that differs, and it differs precisely as the planner change predicts.
- The md5 pair quoted for op 38 matches no digest of those files under md5/sha1/sha256.
- The 248.937 ms "accepted-stack reference" is a fair matched control for P08, but it is **not** the accepted stack. The accepted stack without the down-projection change is 267.179 ms; the down-projection alone is worth ~18.2 ms, roughly 12× what P08 contributes.

**P09**
- Token 44 is a **degenerate gate**. Four mutually inconsistent logit distributions all argmax to 44 — the designated one-layer reference and the P09 control correlate at Pearson 0.303. A gate satisfied equally by such different distributions is not a gate.
- P09's treated logits differ from their **own matched control** on 99.75% of positions, with the top logit tripling (10.69 → 33.13) and the top-5 collapsing to a low-contiguous cluster [44, 45, 499, 30, 33]. "P09 does not corrupt the math at one layer" is not supported. The honest statement is that P09 has no valid correctness evidence at any scope.
- The "evenfix" (owner `i` → core `2i`) is a **no-op with respect to every measurement taken**: pre-fix and post-fix runs produce bit-identical logits and identical transport payloads. The only difference anywhere is the key set in `relayout_plans.jsonl`.
- The new `ALL_GATHER` classifier branch in `lx_relayout.py` is **not** behind the P09 env flag; it can re-classify other edges with the flag unset.

**P12 / P13 / P14**
- "Every treatment request is faster than every control request" is **false**: control request 5 (388.844 ms) beats all ten treatment requests — because it dropped a ~9.6 ms decoder-layer kernel from the profile. With the artifact excluded the separation does hold (control min 397.402 vs treatment max 391.192).
- P12 is a **per-decoder-layer** edge invoked 40× per request, not "the last prefill residual add". That is the mechanism behind the 7.5 ms: 40 × ~190 µs, not one large tail saving.
- P13's marker proof and token proof live in **different runs**. No full-40 P13 run carries a post-PCFG payload.
- P13's `decoded_relayout_only/` is a re-compiled stub produced ~13 minutes after the timed run, not a decode of the timed artifact.
- The "accepted stack" figures 387.619 / 391.371 ms are **means**, not medians. Medians give 387.657 / 392.309 → −4.65 ms, not −3.75 ms.
- **The 248.937 ms figure is misattributed by ~140 ms.** It belongs to `p06_completion/runs/swiglu_downproj_16x2_timing_5x_20260727_b` (P06 lane, 2026-07-27, 42 kernels/request), not to P12+P13+P14 (43–44 kernels/request, ~387 ms).
- The snapshot's 8.222 ms / 2.084% (control 394.507, treatment 386.286) matches **no run** in these tracks.
- The p14 hunk-minimal patch set has **never run on device**; the minimal DeepTools file was `-fsyntax-only` compiled and never linked. Every device run used `p14/deeptools-build` → symlink to `p13/deeptools-build`, whose binaries are dated 15:55, hours *before* the promotion-clean overlay sources.
- `transfer_compute.cpp` in the 4-file overlay is **not** dump-only — it contains an env-gated transformation (`STCDP_FORCE_UNICAST_SPLIT`) that rewrites the delivery table. Verified it never fired in any run, so results are unaffected, but the 4-file → 1-file minimization is not purely instrumentation removal.
- `final_prefill_integration_20260727` was **never executed** — no runs, no logits, no trace. Its pytest cache shows 24 failing tests at 00:38 and source edits at 01:57 with no evidence of a re-run.
- Profiler **zero-duration kernel events** (~9.4 ms each) contaminate individual request sums in the p14, p12 and p13 brackets, and are exactly the source of the ~370-vs-379 ms bimodality. Medians must be taken after filtering these.

**Cross-cutting**
- No `contract.txt` exists in any run under the p03, p07, p09, p12 or p13 tracks. Which `SPYRE_*` flags were set is only inferable from the presence and contents of `relayout_plans.jsonl`. Several drivers were not saved.

## What is unfinished, in priority order

1. **The promoted stack has never run from its own minimal patches.** Blocked on a DeepTools rebuild. Next: rebuild `dxp` from `p14_deeptools_hunk_minimal.patch` (23 insertions, 1 file), separately re-enable the STCDP dump instrumentation that the minimization deliberately dropped, then run one full-40 / 5-request T-C-T-C bracket. Resume from `p14_minimal_20260726/{torch-spyre,foundation-model-stack,deeptools}`.
2. **P03 has no stock baseline.** One run closes it: full-40, 5 requests, `SPYRE_RELAYOUT_ORACLE_PREFILL_MLP_INPUTS=0` and planner off. Until it exists the honest claim is "LX beats HBM by 30% at fixed layout", not "P03 saves 30% versus shipping Torch-Spyre". The same run should dump logits so the 0.279 non-bit-exactness can be adjudicated. Resume from `p03/torch-spyre`.
3. **P13's transport proof is in the wrong artifact.** Re-run one full-40 5× treatment with the post-PCFG payload dump enabled so `STCDPOpLx`=1 / zero-HBM lands in the same run that generates token 203. Helper is `work/run_p13_full40.sh`; it intentionally fails if the run dir exists, so use a fresh run id. Resume from `p13_clean_20260726/torch-spyre`.
4. **P06 has no off-control.** Run the identical stack with `SPYRE_RELAYOUT_ORACLE_PREFILL_QK_QUERY=0`, 5 requests, and recompute the device median. Until then P06's contribution is unquantified. Resume from `p06_completion/torch-spyre`.
5. **P07 is blocked on one bug.** The shuffle shards the 64-wide frequency axis instead of the 512-token axis the allocator requests, and the fold manager aborts on it. Fix the split in `_materialize_p07_rope_input_source` / the emitted layout, re-run the one-layer compile, and confirm the K-side (`17_shuffle`) payload emits. Resume from `p07/torch-spyre`.
6. **P03's second edge does strictly more work for nothing.** Gate-only and up-only each capture the entire win (7250.8 / 7250.0 vs 7265.4 µs/layer with both); the allocator canonicalizes one destination per source, so enabling both performs the same gather twice. Either teach codegen to reuse one materialized gather across both consumers, or ship the single-edge form. Resume from `p03/torch-spyre`.
7. **P05: decide keep-or-drop.** There is no measurement showing it pays for itself. If keeping, run a one-layer value probe comparing `buf48`'s per-cohort mean against CPU to explain why *both* 8-local/24-remote routes produce all-`+inf` logits. Also restore `p05_owner_map/torch-spyre` and `p05_sparse_owners/torch-spyre`, which are still sitting in states that generate token 0, and add the missing `| tee "$RUN/run.log"` to `run_p05_sparse_owners_e2e.sh` so its crashes leave a traceback. Resume from `p05_p06_integration/torch-spyre` (accepted map).
8. **P08 exact source parity is open, and the P05+P08 merge has never run.** The 32→32 form gives token 29; only native-source/SenDNN-destination is correct. Separately, `p05_p08_combined/torch-spyre` carries byte-identical P08 code plus the sparse-owner P05 layer but has no runs directory and its sources postdate its last test run — run its unit tests, then 1-iteration correctness, then a 5× timing bracket. Resume from `p05_p08_combined/torch-spyre`.
9. **P09 needs a ground truth or a formal drop.** The one-layer token-44 gate cannot adjudicate correctness. Either produce a CPU reference for the one-layer configuration, or record P09 as dropped. It is already absent from the integration tree. Resume from `p09/torch-spyre`.
10. **`final_prefill_integration_20260727` has never touched a device.** Run the full `tests/inductor/test_lx_relayout_dldsc.py` in that worktree (24 tests were left red at 00:38 before the 01:57 edits), then one full-40 correctness run. Its P14 code is byte-identical to `p14/torch-spyre`; the only additions are the MLP down-projection MB/OUT knobs and flattened-`buf14` acceptance.
11. **Measurement hygiene, before anything is cited elsewhere.** Filter zero-duration Kineto kernel events before taking any median. Write a `contract.txt` recording the full `SPYRE_*` environment into every run directory. Save the driver scripts (the `p14_promotion_clean` driver is gone; the p03 environment had to be reconstructed from `run_p03_granite_ab.sh`).
12. **Archive the decode-safe FMS variant.** `prefill_edges_unmodified_decode_20260726/foundation-model-stack` is 97 insertions of uncommitted state in a *different repository* and exists as no patch file. It has no logits dump, no control arm, no post-PCFG proof, and only 1 measured request. Extract it as a standalone patch and give it one control run with `ANTONI_LOGIT_DUMP_DIR` set.

## How to resume

Everything lives on the dev pods (`adnan-spyre-current-pf` and `adnan-spyre-dev-pf`; identical code measures ~0.9 ms apart between them, so never compare a run on one against a run on the other). The track root is:

```
/home/adnan/codex-isolated/device_parity_tracks_20260726/<track>/
    torch-spyre/      # git worktree, detached at 59545440, work uncommitted
    runs/<run-id>/    # run.log, relayout_plans.jsonl, allocations.jsonl,
                      # origsdsc_debug_*.json, stcdp_after_pcfg_*.json,
                      # logits/, trace/, export/
```

Every `torch-spyre` worktree is a **detached checkout at `59545440f0e7091ff1b2f90df63580da1842f3fe`**, which is exactly the current tip of the remote branch `ah/granite-relayout`. None of the work was ever committed — it is all working-tree diff, which is why no result here can be cherry-picked or bisected as it stands. The diffs are now archived as patches under `experiments/granite_relayout/tracks_20260727/`.

Two companion repositories are pinned separately and **cannot ride this branch**: `foundation-model-stack` at `61bc991b` and `deeptools` at `406142af`. Their patches must be archived and applied independently. Note that `p14/deeptools-build` is a symlink to `p13/deeptools-build`, and those binaries (dated 2026-07-26 15:55) predate the promotion-clean DeepTools overlay — no run in this project used a `dxp` built from either the 4-file overlay's final state or the 1-file minimal patch.

Run environment for the acceptance gate: Granite 3.3 8B instruct, `batch_size=1`, `fixed_prompt_length=512`, `default_dtype=fp16`, `max_new_tokens=1`, `iters=5`, full 40 decoder layers, `DXP_LX_FRAC_AVAIL=0.2`, chicken-soup prompt, expected token id **203**. The metric is the median over the 5 measured requests of the summed `cat=="kernel"` durations in the Kineto trace — never the host "First-token latency" lines, which are 4–7× larger.