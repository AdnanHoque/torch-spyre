# Ring-Aware Restickify — Research Notes (2026-05)

Internal working notes capturing the empirical work done on the
ring-aware restickify project. Not a draft for any external audience —
this is a "where we are" document so we don't lose the threads.

## Context

The ring-aware restickify project hypothesises that the AIU's RIU
bidirectional ring (10.6 TB/s aggregate vs 166 GB/s HBM) can replace
today's HBM round-trip for cross-core relayout, yielding ~16–28× per-op
speedup on FUNDAMENTAL restickifies. The session's goal was to **ground
that hypothesis empirically**: nail down today's restickify cost,
verify the cost model, and characterise the lever space.

## Empirical work catalog

| Probe | Pattern | Knob varied | Headline result |
|---|---|---|---|
| **v1** (`diag_fundamental_restickify_cost.py`) | `(X @ W) + Y.t()` pointwise | LX_PLANNING=1 | Global optimizer **absorbs** the FUNDAMENTAL via output-STL choice; no separate restickify kernel emitted. |
| **v2** (`diag_fundamental_restickify_cost_v2.py`) | `matmul(X.t(), Y)` matmul-forced | M ∈ {128,512,2048,8192} | Matmul is `FixedInOutNode`; optimizer **cannot** absorb. Restickify kernel emitted. `Δ_measured / Δ_pred = 0.85–0.89×` at M≥512. |
| **v2b** | same as v2, sweep to M=128k | M ∈ {…,131072} | Same 0.85–0.89× ratio; HBM round-trip model validated within 15%. |
| **v3** (`diag_restickify_memory_path.py`) | matmul-forced, allocation + opfunc + BW fingerprint | LX_PLANNING=1 | Triangulates fabric: allocation says HBM, opfunc is `ReStickifyOpHBM`, BW fingerprint 121–140 GB/s ≈ HBM. **Discovered:** classifier misses a codegen-emission path at `spyre_kernel.store` that fires without an FX-inserted restickify node. |
| **v4** (`diag_restickify_pattern_sweep.py`) | 6 patterns × S ∈ {64,256,2048} | LX_PLANNING ∈ {0,1}, DXP_LX_FRAC_AVAIL ∈ {0, 0.2, 0.5, 1} | Three-path classification (HBM / LX-LX all-to-all / LX-LX optimal). At LX_PLANNING=0, S=2048: **6/6 patterns classify HBM**, r_HBM cluster 0.87–1.24 (mean 1.06). |
| **k_fast PSUM probe** (`diag_kfast_psum_probe.py`) | `matmul((128,8192), (8192,8192))` | `SPYRE_CORE_ID_K_FAST_EMISSION` ∈ {0,1} | k_fast OFF: 3632 µs, ON: 1288 µs. **2.82× speedup.** SDSC diff shows split changes from (32,1,1) pure-M to (1,16,2) K-split with K-cohort cores at adjacent ring positions. `init.txt` instruction stream differs (11776 → 12032 bytes). |

All probe sources are checked in under `tests/diag_*.py`; the
v2/v2b/v3/v4 series were committed earlier this session on
`AdnanHoque/rfc-ring-aware-restickify`. The k_fast probe is on the
current branch as an untracked working file.

## Findings

### 1. Today's restickify is HBM round-trip — Path 1, definitively

Three independent signals converge:

- **Allocation:** at SENCORES=32, every restickify buffer is forced to
  HBM by the scratchpad planner's `core_div_mismatch` rule
  (`scratchpad.py`'s "buf users have diff core-splits → cross-core
  LX read/write" disqualification).
- **Op-func emission:** `ReStickifyOpHBM` is the kernel name in every
  bundle we generated, both with and without LX_PLANNING.
- **Bandwidth fingerprint:** measured `2|B|/dt` is **85–116 GB/s** at
  S=2048 across 6 patterns under LX_PLANNING=0; clusters at the HBM-
  effective ceiling. The two on-chip ring models would predict
  ~1328 GB/s or ~4480 GB/s — orders of magnitude off.

The three-path classifier from probe v4 confirms: r_HBM 0.87–1.24
(mean 1.06), r_a2a 21–31, r_opt 73–105. HBM wins unambiguously.

### 2. HBM-effective bandwidth: 100 ± 12 GB/s (measured)

Better-grounded than the 107 GB/s single-shot number we'd been using.
Range across 6 patterns at S=2048:

```
at_plus_x          89.3 GB/s
matmul_xt_y       109.7 GB/s
matmul_x_yt       115.3 GB/s
(a@b)+c.t()        98.9 GB/s
(a.t()+b).t()+c    85.7 GB/s
(a@b).t()@c       109.8 GB/s
                  ─────────
mean              101.4 GB/s,  σ ~12 GB/s
```

### 3. Speedup bracket: 10–28×, central ~20×

The 24.8× I'd been quoting was the optimistic corner of a 2×2 spec/
effective grid:

|              | HBM-spec (166) | HBM-eff (107) |
|---|---:|---:|
| Ring-spec (1328)         | 16.0× | 24.8× |
| Ring-eff @ 64% (850)     | 10.2× | 15.9× |

Central estimate ~20× with a 10–28× bracket, depending on whether one
trusts spec or empirically-derated numbers for the ring fabric. The
single clean paired-graph data point — `(a@b)+c.t()` at S=2048, dt =
0.167 ms for 8.39 MB — sits at the high end of the bracket (17–28×).

### 4. LX_PLANNING regime changes the *relative* cost, not the absolute

- LX_PLANNING=0: clean restickify-cost measurements. Matmul reads
  HBM-bound, so kernel-plan variance is dwarfed by HBM bw.
- LX_PLANNING=1: matmul reads LX-resident, so kernel-plan variance
  dominates and breaks paired-baseline isolation. T_A drops 30–45% on
  matmul-heavy patterns, **restickify becomes a larger share** of
  total wall-clock — *more* important to optimize in production.

The data path itself doesn't change with LX_PLANNING; restickify is
HBM-bound in both regimes (the planner's `core_div_mismatch` rule
forces it).

### 5. DXP_LX_FRAC_AVAIL stability cliff

At sc=32 across 6 test patterns:

| frac | Planner allowance | Stability |
|---:|---|---|
| 1.0 | 0 MB (starved) | All EAR-feasible patterns succeed |
| 0.5 | 1 MB | Some patterns crash with `LX_MODLRFIMM` immediate-out-of-bounds |
| 0.2 (default) | 1.6 MB | Same crash pattern as 0.5 |
| 0.0 | 2 MB (max LX) | **100% crash** across every (pattern, S) — LX planner unusable |

Inverse correlation: more LX availability → more deeptools crashes.
Default 0.2 is the empirically-tuned minimum-viable point.
Corroborates known issue #2062 (LX planning suite: 234 failures /
2258 passed / 126 xfailed).

### 6. k_fast empirically uses the bidirectional ring — but only inferentially

The k_fast PSUM probe at (128, 8192, 8192) gives a clean **2.82×
speedup** with `SPYRE_CORE_ID_K_FAST_EMISSION=1` vs `=0`.

SDSC diff reveals two distinct effects:

- **Layer 1 (work-division):** split changes from (mb=32, out=1,
  in=1) pure-M to (mb=1, out=16, in=2) — better PT utilization.
- **Layer 2 (core-id permutation):** `coreIdToWkSlice_` map has K-
  cohort members at adjacent core IDs (cores 0+1 share output tile
  0 with `in=0` and `in=1`; cores 2+3 share output tile 1; …).

The `init.txt` instruction stream differs between OFF and ON (11776
vs 12032 bytes, hex content differs from line 1 onward). Bundle text
files contain *no* references to `psum`/`ring`/`sfp`/`chain` — that's
opcode-level encoding.

### 7. The codegen-emission gap (from probe v3)

`ReStickifyOpHBM` gets emitted from **two distinct code paths** in
`spyre_kernel.store`:

- **(a) FX-inserted restickify nodes** — what the classifier sees via
  `op.origins is torch.ops.spyre.restickify.default`.
- **(b) Codegen-time stick-mismatch detection** ([spyre_kernel.py:516](torch_spyre/_inductor/spyre_kernel.py#L516)) — fires when a `TensorAccess` is stored with a stick orientation different from its load, *without* an FX restickify node.

Probe v3's `matmul(X.t(), Y)` at HD=4096 emits `ReStickifyOpHBM` via
path (b). The classifier doesn't see it. Any future Phase 1.5–style
gate at `spyre_kernel.py:516` needs to catch both paths.

### 8. EAR hardware limit at S=32k

`per_core_tensor_span 512 MB exceeds hardware limit of 256 MB` —
hits at S=32768 for the square-matrix patterns. The 256 MB per-core
EAR (effective address range) is a hard ceiling. For probe coverage,
the working operating range is roughly **S ∈ [512, 8192]**: below 512
the wall-clock noise floor (~50 µs) swamps the predicted restickify
time; above 8192 we hit address-span overflow.

## Evidence quality summary

For each major claim, what we have:

| Claim | Documentation | Code-level | Empirical (indirect) | Empirical (direct) |
|---|:-:|:-:|:-:|:-:|
| Restickify today is HBM round-trip | ✓ | ✓ (`ReStickifyOpHBM` const, scratchpad rule) | ✓ (probe v4: 6/6 r_HBM ≈ 1) | – |
| HBM-effective bandwidth ~100 GB/s | – | – | ✓ (probe v4 cluster) | – |
| Per-op ring speedup 10–28× | – | – | model + measured baseline | – |
| LX_PLANNING=1 makes restickify a larger share | – | ✓ (planner forces HBM for restickify) | ✓ (probe v4 LX comparison) | – |
| LX planner unstable at high `frac` | ~ (issue #2062) | ✓ (deeptools bug under aggressive LX) | ✓ (probe v4 frac sweep, 100% crash at frac=0) | – |
| k_fast uses the bidirectional ring for PSUM | ✓ (KB `aiu.md` table; `psumRing="dataring"` default) | ✓ (DDL templates emit ring-using primitives) | ✓ (2.82× speedup matches expected adjacent-K-cohort 1-hop ring savings) | – |
| Codegen-emission path bypasses FX-classifier | – | ✓ (`spyre_kernel.py:516`) | ✓ (probe v3 sees `ReStickifyOpHBM` without FX-node classifier hit) | – |
| FUNDAMENTAL restickify is Path 1, not Path 2 or 3 | – | – | ✓ (probe v4 r_HBM ≈ 1, r_a2a ≈ 25×, r_opt ≈ 80×) | – |

**No claim has been validated at the hardware-counter / disassembly /
simulator level in this session.** The strongest empirical evidence is
the wall-clock fingerprint + SDSC bundle inspection. The PSUM-via-ring
claim in particular rests on KB documentation + code-level DDL evidence
+ indirect timing — *not* a direct trace.

## Open questions

1. **Hardware-counter validation of PSUM-via-ring.** Would close the
   inferential chain on whether `dataring` mode's `unichain` /
   `singleshot` PSUM algorithm actually traverses the bidirectional
   ring at the silicon level. Available tools: `libaiupti` (HPM via
   `AIUPTI_ACTIVITY_METRIC_ID_HPM`) or `senulator` (full emulator with
   `psum_destCoreIds` modeling). Both are ~hour-plus integration
   work, not quick checks.

2. **Codex's Stage 3B — three open blockers.** Reviewed and identified:
   (a) correctness unverified (`--skip-correctness` in his command line);
   (b) the v1 RFC's −48.7% Granite regression contradicts Codex's +25%
   adds_then_matmul win unless the mechanisms differ — needs Granite-
   layer test to disambiguate; (c) SDSC-diff in both modes to confirm
   what changes (his RIU bound was off by ~4× — he used single-link
   333 GB/s instead of aggregate 1328 GB/s; with the right bound, his
   observed 53.6 µs savings matches predicted 48 µs, *strong* match).

3. **Phase 1 (joint coordination) untested at scale.** The v2 RFC's
   strategic direction needs an off-line evaluator running on a real
   Granite layer to determine whether the joint optimum beats greedy
   by ≥5% (the kill threshold the RFC sets).

4. **Composability of Phase 1.5 (STCDPOpLx swap) with k_fast.** Both
   touch the same fabric. The k_fast existence proof says deeptools
   already emits ring-using code for PSUM; extending to restickify is
   "same primitive, different op", which lowers the implementation
   risk. But whether the two optimizations could *interfere* at the
   bundle-compile stage hasn't been investigated.

## Implications for the RFC

- **Cost model is solid.** 10–28× per-op bracket with central ~20× is
  defensible based on the LX_PLANNING=0 sweep + the validated HBM
  ceiling.
- **The "compiler does not use the ring" claim in
  `docs/source/compiler/scratchpad_planning.md` §5.4 is outdated** —
  the bidirectional ring is used for PSUM today by default
  (`psumRing="dataring"`, `unichain` or `singleshot` algorithm). The
  doc needs a correction.
- **The deeptools work for STCDPOpLx is lower-risk than originally
  framed.** The compiler already emits ring-using code for one
  operation class (PSUM matmul reductions); extending to restickify
  is a feature extension of an existing capability, not a new
  architectural primitive.
- **The "Stage 3B" direction (Codex's PR) and the "Phase 1.5"
  direction (STCDPOpLx swap) address disjoint cost sources** —
  in-graph computed restickifies (Stage 3B) vs graph-input/weight
  restickifies + matmul-output transposes (Phase 1.5). They compose;
  neither subsumes the other.

## Session artifacts

Probes committed under `AdnanHoque/rfc-ring-aware-restickify`:

- `tests/diag_fundamental_restickify_cost.py` — probe v1
- `tests/diag_fundamental_restickify_cost_v2.py` — probe v2/v2b
- `tests/diag_restickify_memory_path.py` — probe v3
- `tests/diag_restickify_pattern_sweep.py` — probe v4 (with the
  three-path classification + extended-S + LX_PLANNING + DXP_LX_FRAC
  knobs all implemented)

Untracked on current branch (`AdnanHoque/1813-layernorm-non-contig`):

- `tests/diag_kfast_psum_probe.py` — k_fast PSUM probe
- `tests/research_notes_ring_aware_restickify_findings.md` — this file

The Phase 1 classifier, Phase 2 RFC augmentation, and Phase 3 gate
work was committed earlier on `AdnanHoque/rfc-ring-aware-restickify`
but appears reverted on the current branch. The committed history of
that branch is the canonical record.
