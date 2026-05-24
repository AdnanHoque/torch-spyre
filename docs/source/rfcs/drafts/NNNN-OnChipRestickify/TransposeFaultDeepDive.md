# Transpose Compute-CB Deep-Dive — Why Fix #1 Still Faults

Offline source + senprog re-dump, 2026-05-24. No device touched, no code edited,
nothing committed. Goal: determine whether Fix #1 (local per-core transpose + separate
cross-core STCDP reshard) landed at the senprog level, why the device still faults
Compute-CB (`0x7b1b`), whether any PT-restickify runs clean on this device, and the
fixable-vs-RFC verdict.

Inferences flagged `[INFER]`. Binary-senprog reads flagged `[SENPROG]`. RAS/CB-trigger
linkage flagged `[INFER-RAS]`.

---

## 0. TL;DR — verdict reversal

1. **Did Fix #1 land?** **Half.** The *descriptor* fix landed perfectly (the output
   piece is now a true same-region local transpose). But the *transpose compute senprog
   is byte-for-byte identical to the faulting one* — same loop bounds
   (LXLU 2048, L0SU 32768, PT 64). **The fix did NOT change the compute program**, so it
   could never have removed a Compute-CB fault that lives in that program.
2. **Why?** The earlier root-cause theory was wrong. The PT/L0/SFP loop bounds are a
   **product** over the output piece dims × `subOpInfo`, and both descriptors have the
   **same per-core element count** (2048×64) and the **same `subOpInfo` ((64,8),
   numBlocks 8)**. Resharding which dim is the 2048-band vs the 64-band only redistributes
   the product — the total compute (and hence the senprog) is invariant. There was never
   a "2048-out-row the core can't fill" in the compute program; that was a
   descriptor-coherence issue, not a compute issue.
3. **Remaining Compute-CB cause:** the **local PT transpose compute program itself**, run
   at our geometry. Our op is an `out_↔mb_` transpose of a **flat 2-dim** `{mb_,out_}`
   LX tensor with a **2048-deep** per-core piece. The ONLY sanctioned PTLx geometry in
   all of deeptools (the `datadsc_gen.cpp` codegen test) is an `out_↔j_` transpose inside
   a **4-dim** `{j_,i_,out_,mb_}` matmul-output layout with a tiny (128-row) per-core
   piece. Ours is off the trodden path.
4. **Does any PTLx run clean on this device?** **No evidence it runs anywhere.** No SDSC
   regression test, no DDL lit test, and no senulator/device test ever *executes* a PTLx.
   The single stock usage is a DCG-frontend **codegen** test (senprog generation only, no
   execution). The op is *listed executable* in the DD2 sysconfig and *enabled* by the
   DSM on DD2, but I found **zero** proof it has ever executed cleanly.
5. **Verdict:** **RFC handoff.** Fix #1's premise (the descriptor incoherence caused the
   fault) is disproven by the identical senprog. The fault is in the PT-transpose
   *execution* of an op that (a) appears never to have been device-validated and (b) is
   actively disabled by deeptools' own optimizer on the next arch (Sen1.5). This is a
   deeptools/hardware wall, not an inductor/splice misconfiguration. One concrete
   device experiment is still worth running first to harden the claim (§5).

---

## 1. Senprog re-dump verdict — descriptor landed, compute did NOT (FIRST FORK)

Re-ran the patched dxp on a fresh copy of the FIXED bundle (CPU only):

```
cp -r /tmp/transpose_fix/spliced-transpose-fixed /tmp/transpose_deepdive_scratch
rm -rf /tmp/transpose_deepdive_scratch/debug
DXP_VERBOSE=1 build/deeptools-onchip/dxp/dxp_standalone --bundle \
    -d /tmp/transpose_deepdive_scratch -b senulator
```

Aborts in DCC with the known unrelated `std::out_of_range: map::at` (same as the prior
validation); senprogs are dumped before the abort. Transpose senprog:
`/tmp/transpose_deepdive_scratch/debug/sdsc_3p_MixedReStickifyOpWithPTLxConsumer/senprog.txt`.

### 1.1 Descriptor fix DID land (firm, from the JSON)

`sdsc_3p_MixedReStickifyOpWithPTLxConsumer.json`, `0_ReStickifyOpWithPTLx_dataop`,
per-core piece (core 0), FAULTING vs FIXED:

| | FAULTING (`/tmp/spliced-2048`) | FIXED (`/tmp/transpose_fix/...`) |
|---|---|---|
| dataIN piece0 size | `{mb_:2048, out_:64}` | `{mb_:2048, out_:64}` |
| dataOUT piece0 size | `{out_:2048, mb_:64}` | **`{out_:64, mb_:2048}`** |
| out region == in region (same-core transpose)? | **False** (out_ 0..64 read, mb_ 0..64 written) | **True** (out_ 0..64, all mb_) |

Verified for cores 0, 1, 31: FIXED is a coherent same-region per-core transpose in every
core; FAULTING is not. So the splice's intended Fix #1 is present in the descriptor.

### 1.2 The transpose COMPUTE senprog is UNCHANGED — fix did NOT land here (firm) `[SENPROG]`

Per-unit `MVLOOPCNT` for the transpose op, Core 0 / Corelet 0, **fresh re-dumps of both
bundles** (FAULTING from `/tmp/transpose_fix_scratch`, same dxp):

| unit | FAULTING | FIXED |
|---|---|---|
| lxlu | `2048 16 512 16 3 512 16` | `2048 16 512 16 3 512 16` |
| lxsu | `32 8 8 16 3` | `32 8 8 16 3` |
| l0lu | `32 8 8` | `32 8 8` |
| **l0su** | `32 64 32768` | `32 64 32768` |
| sfp | `256 4096` | `256 4096` |
| **pt_row0** | `64 0 64 16 32` | `64 0 64 16 32` |

The compute portion (lxlu→pt_row7, Core 0) is **identical line-count (1471 each)**; the
only diffs are **register-slot / DSTMASK indices** (`(4<<6)` vs `(3<<6)`,
`(3<<10)` vs `(2<<10)` — buffer-allocation noise), **not** loop counts or data extents.
`diff` of the Core-0 compute block = 410 lines, all register-index churn.

So the FIRST-FORK answer: the PT/L0 transpose loop bound is **64** in BOTH bundles (it
always was — the prior FAULTING dump also showed PT_ROW 64). The load-bearing extents
LXLU **2048** and L0SU **32768** are **identical** between fixed and faulting. **The fix
did not change the transpose compute program at all.**

> Note: the whole-file `diff` is ~19k lines, but that is the **L3 ring (gather) portion**
> at the top of each core's program changing (the STCDP-style cross-core gather wiring
> the fix added) plus register churn — NOT the LX/L0/PT/SFP compute. The Compute-CB error
> comes from the compute units, which are unchanged.

### 1.3 Why the compute is invariant (source, firm)

`restickifyOp.cpp` `transformToPcfgReStickifySpecialSFPPTL0`:
- Loop bound = `firstPieceOut.dimToSize_.at(loopDimName)` (lines 1450-1451, "use
  output"), divided by stick size if that dim is a stick dim (1454-1463).
- Loop order = `getInnerLoopOrder` = output stick dims then output layout dims
  (`apeOp.cpp:765-769`).
- Inner-most PT/L0 loops come from `subOpInfo` (numBlocks, transposeType — lines
  1561-1598), which `determineSubOp` derives purely from word length + stick sizes
  (1849, 1898-1922).

For our two descriptors the loop is a PRODUCT over the output dims:
- FAULTING out `{out_:2048(not stick), mb_:64(stick)/64=1}` → product 2048.
- FIXED out `{out_:64(not stick), mb_:2048(stick)/64=32}` → product 64×32 = **2048**.

Same product (the output piece total is 131072 elements in both); same `subOpInfo`
(`elemPerStick=64` → transposeType (64,8), numBlocks 8 in both). Hence identical senprog.
**The descriptor reshard the fix performed is invisible to the per-core compute program.**

---

## 2. The remaining Compute-CB cause (given the fix landed in the descriptor)

Because the compute senprog is the *same one that faulted*, the Compute-CB cause is in
**(a) the local PT transpose compute program itself** — NOT (b) the reshard STCDP gather
and NOT (c) a sequencing interaction. Reasoning:

- The fix MOVED the cross-core gather out of the transpose op into a separate STCDP, and
  the negative control already proved the bundle loads/runs (pure STCDP rings, same-core
  and cross-core, are device-proven value-correct, `CORE_TO_CORE_AIU_RECIPE.md` §8/§11).
  So the gather (b) is not the new suspect.
- The transpose compute program is byte-identical to the one that faulted before the
  STCDP existed, so the fault reproduces with the **transpose op alone** — it is not an
  emergent (c) interaction.
- Therefore the fault is the PT/L0/SFP transpose program executing at **our geometry**:
  a `(64,8)`-tile transpose looped 32× over a **2048-deep `mb_`** per-core piece, on a
  **flat 2-dim** `{mb_, out_}` LX tensor. `[INFER-RAS]` for the exact CB instruction
  (the RAS table that decodes `0x7b1b` was not readable); firm that it is the compute
  program, not the data move.

### 2.1 How our geometry differs from the only sanctioned PTLx geometry

The single PTLx constructor in all of deeptools — `datadsc_gen.cpp`
`populateDataDSCwithReStickifyWithPTLX` (lines 5302-5450) — builds:

| | Stock PTLx (datadsc_gen.cpp:5318-5404) | Our splice (fixed) |
|---|---|---|
| layout dims | **`{j_, i_, out_, mb_}` (4-dim)** | **`{mb_, out_}` (2-dim)** |
| IN stickDim | `out_` (size 64) | `out_` (size 64) |
| OUT stickDim | **`j_`** | `mb_` |
| transpose | **`out_ ↔ j_`** (matmul-internal) | **`out_ ↔ mb_`** (flat activation) |
| layout sizes | `j_=128, i_=1, out_=64, mb_=cores` | `mb_=2048, out_=2048` |
| per-core piece | `{j_:128,i_:1,out_:64,mb_:1}` (128 rows) | `{out_:64,mb_:2048}` (2048 rows) |

This is exactly the `CORE_TO_CORE_AIU_RECIPE.md` §11a note made concrete: PTLx "emits a
native `j_,i_,out_,mb_` descriptor ≠ the consumer's 2D LX descriptor." The op is built
and golden-tested only for the **post-matmul `out_↔j_` stick reorientation inside a 4-dim
matmul output**, with a *tiny* per-core depth. Our use — a stand-alone `out_↔mb_`
transpose of a flat 2-dim activation with a 2048-deep per-core piece — is a geometry the
op's authors never constructed a test for. Both pass `determineSubOp` (same `subOpInfo`),
so it compiles; whether the PT array drives correctly at 2048-deep on this flat 2-dim
descriptor is unproven and is the live fault. `[INFER]`

### 2.2 useARF — ruled out (consistent with prior validation)

`determineSubOp` (1878-1896): `useARF=1` selects ARF only when arch ≥ `RCUDD1A_ISA`;
ARF actually fires (in `transformToPcfgReStickifySpecialSFPPTL0`, line 1548-1551) only if
`numMaxRegiterPerRow < numBlocks(8)`. The senprog shows PT on **XRF** (`PTOP_XRFACCESS`),
zero ARF, with `useARF:1`. So useARF is a no-op here; not the cause. An A/B is low value.

---

## 3. Does ANY ReStickifyOpWithPTLx run clean on this device? — No evidence anywhere

Exhaustive search of `deeptools-onchip`:

- **SDSC regression tests** (`dcc/unittests/SDSC/*.json`, incl. `convos1-fp16`,
  `bmmxrfch-fp16`, all bmm/conv): **0** reference `ReStickifyOpWithPTLx` OR `PTHBM`.
  These are single-op input descriptors; the restickify is inserted downstream.
- **DDL lit tests:** only `ddc/ddl_templates/restickify_sen1p5.ddl` binds the op, and it
  is `required=false` inside a `min_num_valid=1, max_num_valid=1` constraint — i.e. the
  selector picks ONE of {HBM, Lx, PTHBM, PTLx}; PTLx is merely *available*, not exercised.
- **Only constructor / "test":** `datadsc_gen.cpp:5302` via `dcg_test_init.cpp:185-193`
  (`ReStickfySpecialLX*` dataOps). This is a **DCG-frontend codegen test** — it builds
  the descriptor and emits the senprog; it does **not** run a senulator or device. No
  golden execution, no value check, no RAS path.

So: **no stock PTLx is ever executed** (senulator or device) anywhere in the tree —
not just ours. The op is *declared* executable in `sentient_dd2_sysconfig.json`
(`supportedOps`, line 257-269) and the DSM *will* promote PTHBM→PTLx on DD2 when
`canExecuteOpFunc` is true (`lxopt.cpp:3804-3806`, `dsm.cpp:12963-12966`), but I found
**zero** evidence any such promotion has ever been device-validated.

**Strong corroborating signal — the DSM disables PTLx on the next arch:**
`dsm.cpp:12875-12877`:

```cpp
if (dscGlobal->sysDef.coreArch >= IsaCoreGen::SEN1P5_ISA) {
  useRestickifySpecial = false;   // never emit PTLx on Sen1.5+
}
```

`IsaCoreGen` order (`isa.hpp:24-31`): `MPW2 < MPW3 < MPW4 < RCUDD1A(DD2) < SEN1P5`. So
PTLx is *not* gated off on DD2 by this check (DD2 < Sen1.5), but it **is** unconditionally
turned off on Sen1.5 — deeptools chose to route layout-changing restickifies back to HBM
on the newest arch rather than trust the PT transpose. That is the behavior of a path the
vendor does not consider production-solid. `[INFER]` on intent; firm on the code.

---

## 4. Verdict — RFC handoff (deeptools/hardware wall), not an inductor/splice fix

**Evidence the layout-changing transpose is a deeptools/hardware wall:**
1. **Fix #1's premise is disproven.** The descriptor incoherence it fixed does not drive
   the transpose compute program; the senprog is byte-identical, so the Compute-CB cause
   was never the thing the fix changed. We have now exhausted the descriptor-level levers
   for the transpose op (in/out are a coherent same-region same-core transpose; subOpInfo
   is the textbook (64,8); useARF is a no-op). There is no remaining splice-side knob that
   alters the faulting compute program without changing the per-core element count.
2. **The op is never executed anywhere** — no SDSC, DDL, senulator, or device test runs a
   PTLx. Ours faulting is consistent with an op that has not been hardened for execution.
3. **deeptools itself routes around it** — disabled on Sen1.5 (`dsm.cpp:12875`), and the
   only PTLx geometry the codebase tests (`datadsc_gen.cpp`) is the narrow post-matmul
   `out_↔j_` 4-dim case, not a flat `out_↔mb_` 2-dim activation transpose.
4. This matches the standing frontier note (`CORE_TO_CORE_AIU_RECIPE.md` §11a; memory
   `project_ring_aware_restickify.md`): the layout-changing cross-core move is the
   "genuinely-missing composable primitive," likely the same wall as `sfpring` being
   psum-only.

**Why not still "fixable from our side":** the only thing left that would change the
transpose senprog is changing the *per-core geometry* (e.g. shrinking the per-core depth,
or wrapping the activation in a 4-dim `{j_,i_,out_,mb_}` layout to mimic the stock case).
That is no longer a splice-descriptor reshard — it is reverse-engineering the op's
validated envelope by trial, against an op with no execution test to validate against.
That work belongs in deeptools (who own the PT-transpose codegen + its hardware
contract), i.e. an RFC. The user is on the inductor team; per the standing scope note
(`user_torch_inductor_team.md`), deeptools internals are a handoff, not a patch.

---

## 5. The single most-likely next action

**One last device experiment to harden the RFC (cheap, high-information), THEN the RFC:**

**Device experiment — shrink the per-core transpose depth to the stock envelope.**
Build a splice whose per-core transpose piece matches the only geometry the op is tested
at: a **single tile per core** (per-core `out_:64 × mb_:64`, i.e. depth 64 not 2048),
either by using a tiny M (mb_=64) or by inserting an extra outer loop dim so each PT
invocation transposes one 64×64 tile. If a **64-deep** per-core PTLx runs clean but the
**2048-deep** faults → the wall is the deep-loop PT-transpose codegen (a concrete,
reportable deeptools bug with a minimal repro). If even the 64-deep faults → the op is
non-functional on this device for ANY flat `out_↔mb_` geometry (the stronger wall). Run
solo (single shared accelerator) with the mandatory remove-the-senprog negative control.

**The RFC ask (precise):** deeptools should either (a) make `ReStickifyOpWithPTLx`
execute correctly for a **flat 2-dim `{mb_,out_}` `out_↔mb_` transpose with arbitrary
per-core depth** on RCUDD1A/DD2 (and add a senulator/device test — none exists today), or
(b) declare it unsupported for this geometry and provide the sanctioned layout-changing
cross-core primitive (the "consumer-endpoint adapter" / gather→transpose→scatter chain)
so inductor can emit Q/K/V and pre-matmul transposes without an HBM round trip. Attach
the minimal repro from the experiment above + the senprog
(`/tmp/transpose_deepdive_scratch/debug/sdsc_3p_.../senprog.txt`) and the smoking-gun
table in §1.2.

---

## Appendix — what was run / read

| Artifact | Used for |
|---|---|
| `/tmp/transpose_deepdive_scratch/` (fresh dxp re-dump of FIXED bundle) | §1.2 fixed senprog |
| `/tmp/transpose_fix_scratch/debug/.../senprog.txt` (FAULTING re-dump) | §1.2 comparison |
| `/tmp/transpose_fix/spliced-transpose-fixed/sdsc_3p_*.json` | §1.1 fixed descriptor |
| `/tmp/spliced-2048/sdsc_3p_*.json` | §1.1 faulting descriptor |
| `deeptools-onchip/dcg/dcg_fe/pcfg_gen/restickifyOp.cpp` (1340-1463, 1548-1685, 1819-1938) | loop-bound source, determineSubOp, why compute is invariant |
| `deeptools-onchip/dcg/dcg_fe/pcfg_gen/apeOp.cpp` (761-772) | getInnerLoopOrder (output-dim loop order) |
| `deeptools-onchip/dcg/unit_tests/datadsc_gen.cpp` (5302-5450) | the ONLY sanctioned PTLx geometry (4-dim `out_↔j_`) |
| `deeptools-onchip/dcg/unit_tests/dcg_test_init.cpp` (185-193) | PTLx test is codegen-only, no execution |
| `deeptools-onchip/dsm/dsm.cpp` (12831-12877, 12959-12989) | useRestickifySpecial gate; PTLx disabled on Sen1.5 |
| `deeptools-onchip/dsm/workOptimizer/baseOptimizer/lxopt.cpp` (3804-3806) | PTHBM→PTLx promotion gated on canExecuteOpFunc |
| `deeptools-onchip/dsc/isa.hpp` (24-34) | IsaCoreGen order (DD2=RCUDD1A < SEN1P5) |
| `deeptools-onchip/dsc/HardwareArchMapping/.../sentient_dd2_sysconfig.json` (257-269) | PTLx listed executable on DD2 |
| `deeptools-onchip/ddc/ddl_templates/restickify_sen1p5.ddl` (34-40) | PTLx is `required=false` selector option, not exercised |
| `dcc/unittests/SDSC/*.json` | 0 PTLx/PTHBM references (no regression test) |
| `/tmp/transpose_fix/splice_transpose_fixed.py` | confirms fix = local transpose + separate STCDP, 2-dim `{mb_,out_}` |
| `/tmp/CORE_TO_CORE_AIU_RECIPE.md` §8/§11a, `/tmp/transpose_fault_investigation.md`, `/tmp/transpose_fix_validation.md` | prior diagnoses (one premise corrected here) |
