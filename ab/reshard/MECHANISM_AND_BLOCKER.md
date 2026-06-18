# On-chip handoff mechanisms, and why the 2-D SwiGLU reshard needs deeptools

A distilled, self-contained analysis of *why* the SwiGLU `matmul → neg`
cross-division reshard is blocked where the on-chip repo's proven moves were not.
Companion to `PATH_A_PROGRESS.md` (the running log) and `../coassign/README.md`
(the value-correct inductor-only alternative). Grounds: the on-chip repo
(`github.ibm.com/Adnan-Hoque1/spyre-onchip-core-to-core`, docs 02 + 06), Agent A's
CPU root-cause, and the per-band probe (2026-06-18).

## 1. Four ways to keep a producer→consumer handoff on-chip

A torch-spyre bundle hands every cross-SDSC activation off through HBM by default
(the planner evicts LX at each `sdsc_execute`). There are four ways to keep it
on-chip; they differ entirely in **whether data physically moves between cores**:

| Mechanism | Data moves? | data-op / mixed SDSC | dxp gate | DCG EBR packer | Ships? |
|---|---|---|---|---|---|
| HBM round-trip (default) | via HBM | no | no | no | yes, but slow (shared 170 GB/s) |
| **softmax-chain LX flip** (`onchip_softmax_chain`) | **no** | no | no | no | yes (1.88× SDPA) |
| **co-assignment** (`ab/coassign/`) | **no** | no | no | no | yes (~7%, value-correct) |
| **reshard** (this dir) | yes, core→core ring | **yes** | yes | **yes — the blocker** | no |

The first three need none of the data-op machinery. Only the reshard does a genuine
cross-core **move**, and that is the entire source of its cost.

## 2. Why the reshard needs a data-op (the mixed SuperDSC)

**LX is per-core private.** The base-pointer flip (`apply_lx_flip`, the proven 1.88×
softmax-chain mechanism) keeps a handoff on-chip with no data-op *only* when the
edge is **same-shard same-core**: each consumer core's input already sits in *its
own* core's LX, so you just rename the planner's HBM address to an LX address.
Nothing moves. The eligibility condition (repo 06 §2) is literally "every core's
local view of the producer's slot IS the consumer's slot."

The SwiGLU `matmul → neg` edge fails that. It is **cross-division**: consumer core
`c` (neg, `{mb:32}`) needs rows `[16c, +16)` across all 12800 columns, but those
were produced by **eight different** producer cores (`{mb:4, out:8}`, one per
`out`-band). The bytes core `c` needs live in eight *other* cores' private LX. A
base-pointer flip cannot help — core `c` physically cannot read core `c'`'s LX. The
data must be **transported** (LX → RIU ring → LX).

Transport means an `STCDPOpLx` data-op, and a data-op can only run inside an SDSC.
So the ring move must be packaged either folded into the consumer SDSC (the **mixed**
DL + data-op SuperDSC) or as a standalone data-op SDSC. Stock dxp rejects both at
the import gate (`SdscTree.cpp:152` "Datadsc not allowed") — hence the §5-patched
dxp. The mixed form is the cleaner one: the data-op runs, then the consumer's DL op
reads its now-local input in the same launch.

So **mixed dsc → dxp gate → EBR packer are all the price of moving data on-chip.**
Co-assignment avoids the column entirely by realigning the divisions so the
consumer reads what its own core produced — which is why it ships and the reshard
does not.

## 3. Why the on-chip repo never needed the EBR fix — but SwiGLU does

Every cross-core move the repo proved on device is a **1-D `out:N` split**:

- The proven round-trip (`i → 31-i → i`, repo 02 §6c) has
  `numWkSlicesPerDim_ = {mb:1, out:32}`: one split dim, 32 cores, **each owning
  exactly one column band** (`chunk = 2048/32 = 64`). Core `i` owns column band `i`.
- Its attention splice is `mb:32 → x:32` — again 1-D each side.

The DCG EBR packer computes the L3SU **dest store column** as `EBR = core_index ×
stride` (`dcgbeCodegen.cpp:2720` ← `getDestStAddr` ← per-core `ebrInit_`). For any
1-D `out:N` split that is **correct by construction**: core index *is* the
column-band index, so core `i` → column `i × stride` is exactly right. The
reversed-ownership trick only permutes *which* core holds a band; the
`core == column` identity is preserved and the data lands in the consumer's native
layout (no consumer descriptor surgery). The packer has a baked-in assumption:
**core index == column-band index.**

**SwiGLU is the first 2-D `{mb:4, out:8}` producer.** Now four cores share each
column band — core `p` owns column band `p // 4`, not `p`. The packer's
`EBR = core × stride` is wrong by exactly the `mb_split = 4` factor: it emits
`p × 1600` where the answer is `(p // 4) × 1600`. So cores 8–31 store **out of
bounds** of the 12800-col gate → ~zero output. (The repo's own *productionised*
asymmetric pass broke the same way — max_err 0.669 — the moment it left the clean
1-D case.)

We didn't need the fix before because the matmul's fast split is 2-D and nothing the
repo touched was. The two "stay 1-D" escapes are both unsatisfying: make the
producer `out:8` (8 cores, `core == column` again — but that drops the `mb:4`
M-split that fills the array, the slow split steering already lost on), or
co-assign (no move at all).

## 4. The frontend is exhausted (per-band probe, 2026-06-18)

The strongest remaining frontend idea was to decompose the single 2-D-scatter STCDP
into **8 per-band STCDPs** (one per `out`-band, each a fixed `[*, b*1600 : +1600)`
column band → `src_col == dst_col`, a pure row redistribution at a constant column —
no intra-row placement handed to the packer). Built in
`pieces.build_swiglu_unfused_perband_edges` +
`substrate.build_perband_reshard_bridge` + `splice_swiglu.splice_bundle_perband`.

CPU result (patched dxp, no device): **compiles, exit 0, 248 `L3_LDU`/`L3_STU`,
correct ROW scatter** — but `smc.txt` `@regInit:EBR:R0` is **still `3200*core`**
(`0 3200 … 99200`), byte-identical to the single-STCDP bug. Two independent frontend
structures → the **same** core-linearised EBR. The column comes from the core index
*in the packer*, regardless of what the pieces say. **No frontend structure can fix
it.**

## 5. The fix (deeptools) and the one cheap probe left

`setPlacementInfoSubPiece` (`stcdpOp.cpp:2676`) already computes the correct
per-subpiece **LX** (LBR) address from the piece's logical coordinates. The bug is
that the L3SU **dest column** (EBR) is taken from `cidx`, discarding the subpiece's
`out_` coordinate. **The fix is to generalise the packer** from "1-D column split"
to "2-D co-split": derive the dest column from the subpiece `out_` coordinate (i.e.
`core // mb_split` instead of `core`). It is a principled generalisation, not a
SwiGLU special-case — it makes the packer correct for *any* co-split producer.

Then: rebuild `dxp` (deeptools build) → re-splice → device-validate `max_err`.

**One cheap frontend probe remains untried** (CPU, no build): explicitly stamp the
`dataOUT` per-core `ebrInit_ = (core // 4) * 1600` on the data-op `labeledDs` and
see if DCG honours an input value instead of recomputing from `cidx`. If it does →
inductor-only fix after all; if it ignores it (likely) → the deeptools generalisation
is required.

## 6. Alternatives to the mixed SDSC, and the least-invasive analysis

"On-chip reshard" is the *outcome*; the mixed SDSC is one *mechanism*. Only one
branch of the design tree actually moves data:

| Mechanism | Moves data? | Deeptools cost | Hits the EBR packer? | Note |
|---|---|---|---|---|
| Co-assignment (`ab/coassign/`) | **no** | none | no | least invasive; covers the whole element-wise tail |
| softmax-chain LX flip | no | none | no | same-shard persistence only |
| **Mixed SDSC + STCDP** (this dir) | yes | §5 gate (built) + EBR fix | yes | most-proven move path |
| Standalone data-op SDSC | yes | gate + consumer-binding (§11c) | yes | **more** invasive than mixed |
| Gather (`InputFetchNeighbor`, l3lu) | yes | gate + ? | **shares `computeMulticastOptMetadata`** | cheap probe, not a clean escape |
| LX-planner extension | yes | large planner rework | n/a | cleanest long-term, **most** invasive |
| Epilogue fusion | n/a (no handoff) | needs deeptools fusion; silu×mul-into-mm **refuted** | no | limited |
| Steer producer→consumer layout | via HBM/local | none | no | loses the matmul M-split (~1.6×) |

Verified structural fact: **both the scatter (`STCDPOpLx`) and the gather
(`InputFetchNeighbor`) route through the same `computeMulticastOptMetadata`
packer** (`inputNeighFetchOp.cpp:553`), and the EBR `initValue` is computed for
both `L3LU` and `L3SU` (`dcgbeCodegen.cpp:2720`). So the EBR packer is the **single
shared chokepoint for every same-stick cross-core data-op** — you cannot dodge it
by switching primitive, and fixing it once **generalises every move primitive at
once**. That is the argument for the targeted packer fix over any heavier
alternative.

**Is the mixed SDSC the least-invasive way?** For SwiGLU's element-wise tail, **no**
— co-assignment is (no move at all), and it ships. For a *genuine* cross-core move
(non-co-assignable consumer), **yes among proven options**: the mixed fold is less
invasive than the standalone (it makes the consumer's own input the bridge output,
sidestepping the §11c binding gap) and far less than the planner extension; the gate
is one file and already built; the only open cost is the localized, general EBR fix.
**There is no non-deeptools path to a true cross-core move** — any ring data-op
needs the gate, and 2-D co-split needs the EBR fix. Every non-deeptools option works
by *not moving*.

## 7. Is the LX-planner extension an inductor-only change? No.

Split "LX planner" into its two halves:

1. **Plan LX residency + coordinate addresses across SDSC boundaries** — pure
   inductor (`onchip_softmax_chain` / `apply_lx_flip` / co-assignment do exactly
   this, stock dxp).
2. **Realize a cross-core move** — **not** inductor-only, and cannot be made so.
   LX is per-core private; the *only* thing that emits ring microcode
   (`L3_LDU`/`L3_STU`) is DCG codegen of a data-op. Inductor emits SDSC JSON; the
   senprog is "produced **only** by dxp's post-DCC orchestration" (recipe §5).
   Inductor literally cannot synthesize a ring transfer.

So an inductor pass can **author** a move (decide it, emit the mixed SDSC — that is
`realize_onchip_handoff`) but cannot **execute** it without the deeptools data-op
(gate + EBR packer). Extending the LX planner to do the reshard therefore does
**not** escape deeptools — it relocates the inductor-side work from a per-edge splice
into a general planner, i.e. **more** inductor surface on the **same** deeptools
floor. The only fully inductor-only "LX planner" is the **no-move** variant
(same-shard persistence + co-assignment) — inductor-only precisely because it never
crosses a core boundary. This is itself an argument *for* the targeted EBR-packer
fix: same deeptools requirement, far less inductor surface, one shared chokepoint.

## 8. Bottom line

- **Co-assignment ships the value-correct inductor-only MLP win** (~7%, no data-op,
  no dxp gate, no EBR packer) — it realigns the divisions so no data moves.
- **The reshard (a genuine 2-D cross-core move) is blocked by one deeptools bug:**
  the DCG EBR packer assumes `core == column-band`, true for the repo's 1-D `out:N`
  moves, false for SwiGLU's 2-D `mb×out`. The frontend is exhausted; landing it
  needs the packer generalisation + a dxp rebuild + device validation. No
  mechanism — mixed, standalone, gather, or planner — escapes that one fix, because
  they all share the `computeMulticastOptMetadata` chokepoint.
