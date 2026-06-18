# Core-to-Core On-Chip Reshard for the Granite SwiGLU MLP

**Branch:** `core-to-core` · **Status:** co-assignment ships now; reshard inductor-side complete, gated on one deeptools fix · **Date:** 2026-06-18

---

## 1. Goal & the one-line technique

**Goal.** Keep the Granite SwiGLU matmuls at their ideal array-filling `{mb:4, out:8}` (8×4 = 32-core) split, let the element-wise SwiGLU tail (neg/exp/add/realdiv/mul) take whatever split is cheapest, and route the matmul→consumer hand-off **on-chip** (core-to-core over the RIU BiRing, LX→LX) instead of through an HBM round-trip — *without* steering the matmul to a slow split.

**One-line technique.** Split the hand-off problem into two disjoint edge classes and serve each with the minimal mechanism:

> **Co-assign the split-agnostic element-wise edges (no move, pure Inductor — SHIP NOW); reshard only the one genuinely non-co-assignable edge (the down-proj K-reduction) with an LX→RIU→LX mixed-SDSC move, gated on a single shared-chokepoint EBR-packer fix in deeptools; then warp-specialize that one move (PT ∥ SFP ∥ L3 overlap) on top.**

The literal goal (matmul `{mb:4,out:8}` AND element-wise *pinned* pure-M `{mb:32}` AND on-chip) requires the genuine reshard plus the deeptools fix. [[co-assignment]] meets the *spirit* — matmul fast split kept, hand-off on-chip — by making the element-wise tail `{mb:4,out:8}` instead of `{mb:32}`, with **zero** deeptools dependency. This document ships the spirit now and lays out the literal path honestly.

---

## 2. End-to-end flow: PyTorch → work-division → on-chip hand-off → consumer

### 2.1 What the PyTorch layer actually lowers to (empirically verified)

`down_proj(silu(gate_proj(x)) * up_proj(x))`, shape `1×512×4096`, hidden `12800`.

- **`silu` is not a Spyre primitive.** Dynamo/AOTAutograd decomposes `silu(x)=x/(1+exp(-x))` into **four SFP ops**: `neg → exp → add(+1) → realdiv`. The gate-mul is a separate `mul`.
- **There is no matmul epilogue and no `silu×mul` fusion.** The Inductor SchedulerNode label `sdsc_fused_linear_mul_silu_0` is a *kernel-partition label* over a Python **list of 10 independent single-op `OpSpec`s** (`async_compile.sdsc('sdsc_fused_linear_mul_silu_0', [...])`). Every emitted SDSC JSON has `len(dscs_)==1` and `len(computeOp_)==1` — verified across `/tmp/c2c-unfused-final/sdsc_0..9.json` and the coassign inductor-logs. **The authoritative fusion test is structural** (count `computeOp_` entries), never the kernel name.
- **Engine map** (`computeOp_[0].exUnit`): the 2 `batchmatmul`s → **PT/PE array** (`pt`); `neg/exp/add/realdiv/mul` + 3 `ReStickifyOpHBM` → **SFP** (`sfp`). No op is an L3-ring compute primitive.

### 2.2 The work-division plan (today, unchanged)

In `passes.py`, `_distribute_work` runs `cost_model_matmul_division` then `work_distribution(graph, preassigned_ops)` (verified: `cost_model_matmul_division` returns the ops it claims; `work_distribution` skips them).

- `cost_model_matmul_division` (`work_division.py:982`) picks **`{mb:4,out:8,in:1}`** for every matmul → per-core `128×1600` tile (`512/4 × 12800/8`), 32 cores.
- `work_distribution` defaults the pointwise consumers to **pure-M `{mb:32,out:1}`** → per-core `16×12800` band, 32 cores.

These two divisions are chosen **independently**, which is the *entire* reason a cross-division hand-off exists.

### 2.3 The edge taxonomy

| Class | Edge | Consumer | Co-assignable? | Mechanism |
|---|---|---|---|---|
| **A** | gate-mm → neg/exp/add/realdiv; up-mm → mul (the SwiGLU tail) | split-agnostic `Pointwise` | **Yes** | **Co-assignment** (no move, no deeptools) |
| **B** | `mul`-output → **down-proj** matmul | reduces over `K=12800` = the producer's split dim | **No** | **Reshard** (LX→RIU→LX mixed SDSC) |

Class A is **~5 of the ~6 edges**, eliminated move-free. Class B is the **one** edge where a real cross-core move is unavoidable.

### 2.4 Class A — co-assignment (Inductor-only, SHIPS NOW)

`ab/coassign/coassign.py::apply_coassign` monkeypatches `passes.cost_model_matmul_division`: it calls the original (matmuls get `{mb:4,out:8}`), then BFS-walks the `Pointwise` consumer chain on producer-write/consumer-read buffer overlap (`_bufs(op,"reads") & pbufs`), maps the producer split onto each consumer iter-space by matching dim extents (`_map_split_by_extent`), commits it via `apply_splits` (`apply_splits_from_index_coeff` recovers it from `op_it_space_splits`), and returns the consumers as **preassigned** so `work_distribution` honors them. Result: the matmul→pointwise edge becomes **same-division same-core** — each consumer core reads exactly the tile its own core produced. The baseline `STCDPOpLx` `datadscs_` LX re-layout op (present on the `neg` SDSC in `/tmp/c2c-unfused-final/sdsc_2.json`, 32 cores, no `computeOp_`) **disappears** after co-assignment (verified in the coassign inductor-logs).

**Productionisation:** promote `apply_coassign` from a monkeypatch into a config-gated pass (`config.swiglu_coassign`, default-on once landed) inside the `_distribute_work` slot. No other change.

### 2.5 Class B — the reshard (Inductor authors, deeptools executes)

The down-proj is `[512,12800]@[12800,4096]` reducing over `K=12800`, which **is** the `mul`'s `out`-split dim. The cost model gives down-proj `{mb:4,out:8,in:1}` (K **not** split). You cannot co-assign: inheriting the producer's `out:8` column split would force a K-split (`in:8`) → a PSUM partial-sum ring, a *different and more expensive* mechanism the cost model already declined (and steering the matmul to pure-M lost 1.4–1.6×, so keeping `{mb:4,out:8}` is load-bearing). So the reshard is scoped to **this one edge**: each down-proj core `q` gathers its full-K M-band on-chip from the 8 `mul` cores that hold its column bands, over the RIU BiRing (LX→LX), instead of round-tripping the `[512,12800]=12.8 MB` activation through HBM.

**Inductor passes + insertion points (all in the `_distribute_work` → `_maybe_scratchpad_planning` region of `passes.py`):**

1. **Reshard planner** — extend `onchip_handoff.plan_onchip_handoffs` with an `is_reduction_input_edge` predicate (producer split dim == consumer reduced dim). Pure observer, fail-closed, **never mutates splits**. Runs in the work-division slot.
2. **Realize pass** at `codegen/bundle.py::generate_bundle` (today a monkeypatch at bundle.py ~323-339; productionise into a real pass keyed off the planner's plans):
   - `apply_lx_flip` (`substrate.py:309`) on `mul`-out + down-proj-in (memOrg→lx, HBM addr/size cleared, `lxSize` sentinel, per-core `coreStateInit_`, scheduleTree allocate-node rewrite; the substrate variant also clears `backGapCore_` at `:336`).
   - `build_asymmetric_reshard_bridge` (`substrate.py:153`, 2-D `Piece` lists; `pieces.py:199` renders row-band × col-band `Piece` → deeptools `PieceInfo`).
   - `splice_reshard` mixed-fold (`substrate.py:349`): attach `datadscs_` + `coreIdToDscSchedule` + `opFuncsUsed_` to the consumer body, set `numCoreletsUsed_DSC2_=1`.
   - **Bind by LX-base coincidence** (resolves recipe §11c without a graph API; the mixed fold makes the consumer's *own* input the bridge output, so there is no cross-SDSC binding at all).
   - `allocate_lx_bases` (`substrate.py:256`, `LX_CAPACITY_BYTES=2MB`) is the fail-closed liveness gate (raises if two regions exceed 2 MB/core).

The full scatter authoring stack **exists and is CPU-proven**: compiles (exit 0), 248 `L3_LDU`/`L3_STU`, correct ROW scatter, correct DCG transfer permutation `p → consumers [8·(p%4), +8)`, `assert_partition` 7/7. *(Verified: `substrate.py` implements only the L3SU scatter — there is no gather/`InputFetchNeighbor` authoring path; see §6.)*

---

## 3. The EBR / deeptools piece

### 3.1 The symptom (device-proven, undisputed)

The authored reshard runs on the §5-patched dxp and emits real cross-core ring traffic, but produces **≈0 output** (device `mean|reshard|≈0`, max_abs_diff up to 0.567). Decode of `debug/sdsc_2/smc.txt @regInit:EBR:R0`: the per-core L3SU **dest store column** is `EBR = 3200·core` (`0, 3200, …, 99200`). The **correct** value is `3200·(core//4)` = `out_band·1600`. Cores 0–3 (all out-band 0) should land at col 0 but get cols 0/1600/3200/4800; cores ≥8 write **past the 12800-col gate** → `silu(≈0)=0` → output ≈ 0.

**Two independent frontend framings** (single 2-D STCDP; 8 per-band STCDPs with `src_col==dst_col`) produce **byte-identical** broken EBR. The frontend is exhausted — it is the packer, not the authoring code. `setPlacementInfoSubPiece` (`stcdpOp.cpp:2676`) already computes the correct per-subpiece **LX/LBR** address from the piece coordinates; only the **HBM/dest EBR initValue** discards the `out_` coordinate and uses the raw core index.

### 3.2 Why this is the shared-chokepoint, minimal, general fix

The repo's proven cross-core moves are all **1-D `out:N`** splits, where `core == column-band` by construction, so `EBR = core·stride` is correct (`core//1 == core`). SwiGLU is the **first 2-D `{mb:4,out:8}` producer**: four cores share each column band (`core p` owns band `p//4`), so `EBR = core·stride` is wrong by exactly the `mb_split=4` factor.

**Both** the scatter (`STCDPOpLx`) and the gather (`InputFetchNeighbor`) route through the **same** `computeMulticastOptMetadata` (`inputNeighFetchOp.cpp:553`), and the EBR `initValue` is computed for both `L3LU` and `L3SU`. So this is the **single shared chokepoint for every same-stick cross-core data-op**: you cannot dodge it by switching primitive, and fixing it once **generalises every move primitive at once**. The fix is a principled generalisation from "1-D column split" to "2-D co-split", not a SwiGLU special-case. Blast radius is contained: 1-D `out:N` is unchanged (`core//1==core`).

### 3.3 The exact fix — and an unresolved location dispute (marked)

**The fix, stated mechanically (high confidence):** derive the L3SU dest column from the subpiece `out_` coordinate (col-band = `core // mb_split`) instead of the core index, mirroring what `setPlacementInfoSubPiece` already does for LX/LBR. One function, contained blast radius.

**The fix location is contested** (this is the single hardest open risk — §8). Three candidate carriers have been advanced by different analyses; the *symptom* is device-proven but the *carrier* for the LX-flipped path is **not settled**:

| Carrier hypothesis | Location | Status |
|---|---|---|
| **(GT) DSM HBM-stride** | `dsm.cpp:6761-6770` (`ebrInit_` from raw `coreId` via `startAddressCoreCorelet_`), fix `perfDscToSdsc.cpp:~2099` core_fold stride-0 | high-confidence ground-truth claim, but… |
| **(corrected) HBM-gated, mis-located for LX move** | the DSM `ebrInit_` fill is inside `if memOrg_.count(HBM)` (`dsm.cpp:6756`); the reshard **LX-flips** producer-out (`substrate.py:293` stamps `ebrInit_=-1`) → the **LX branch** (`dsm.cpp:6779`) runs and fills only `lbrInit_`. The `3200·core` is then **recomputed in the DCG packer** (`stcdpOp.cpp:5033` sets dest=-1, then the per-piece fill at `stcdpOp.cpp:2227-2238` / `computeMulticastOptMetadata` derives placement). | source-traced; **overturns the GT headline for the LX path** |
| **(alt) dead HBM store** | the per-core EBR is on a dead HBM store, not the live ring node; real bug is the ring LX-landing addresses | medium confidence |

**Resolution gate (mandatory, cheap, CPU-only, no build):** before any deeptools build, run the **EBR attribution probe** — zero-compute decode of the existing reshard debug to attribute the `3200·core` carrier to the live ring node vs an HBM store, plus stamp `dataOUT` per-core `ebrInit_ = (core//4)·1600` (via `substrate._core_state_init_entry` / the dataOUT `PieceInfo`) and recompile with the patched dxp to see whether DCG **honours an input value** or **recomputes from `cidx`**. If honoured → EBR drops to `0 0 0 0 3200 …` and the 2-D reshard becomes **inductor-only** (unlikely). If ignored (likely) → the deeptools DCG-packer generalisation is confirmed required, *and* the probe tells us whether it is one function or two call sites.

### 3.4 Inductor vs deeptools split

- **Inductor (authors, no device):** (1) co-assignment as-is, promoted to a config-gated pass — **ships independently of everything below**. (2) the reduction-input-edge planner predicate. (3) the realize pass (LX-flip + bridge + splice, bound by LX-base coincidence). All CPU-proven; all fail-closed; none can produce value-correct *reshard* output until the fix lands.
- **Deeptools (executes; only for the reshard — Class A needs ZERO):** (i) the dxp import-gate patch (admits the mixed DL+data-op SuperDSC past `SdscTree.cpp:152`) — **already built** at `/home/adnan/dt-inductor/build/deeptools-onchip/dxp`, CPU-proven exit 0, 248 L3. (ii) the **one** EBR/dest-column fix (§3.2/§3.3). Hand off as a deeptools Transform/Foundation contract.

---

## 4. Value-correctness argument

**Co-assignment (the shipping bulk) — correct by construction.** `neg/exp/add/realdiv/mul` are pointwise/split-agnostic: each output cell computes the same value regardless of how rows/cols are partitioned. Propagating `{mb:4,out:8}` changes **which core owns which tile**, not the arithmetic. There is **no move** → no EBR involved. **Device-validated:** max_abs_diff 0.0059, mean_abs_diff 0.00081, `allclose(1e-2,1e-2)=True` vs CPU eager (fp16-noise level; device −0.00341/0.1064 vs eager −0.00339/0.1055).

**Reshard (the one gated edge) — correctness reduces entirely to dest-column placement.** The PieceInfo, the DCG transfer permutation (`p → consumers [8·(p%4), +8)`), and the consumer read base (409600, threaded as `l3lu LBR=3200` sticks) are all **CPU-proven correct** (`assert_partition` 7/7; hand-replaying `setPlacementInfoSubPiece` yields the correct `0,0,0,0,3200,…`). The **only** defect is the L3SU per-core dest column. After the fix derives the column from `out_` (`core//mb_split`), each down-proj core's full-K band lands where the reduction expects → **bit-exact w.r.t. the HBM-round-trip baseline** (same fp16 arithmetic; only transport changes).

**Critical acceptance gate (load-bearing lesson).** The offline `assert_partition` cell-coverage (7/7) and senprog ring-traffic presence are **necessary but INSUFFICIENT** — *both passed while the output was ≈0*. Two prior reshard bugs (the fused `backGapCore_` sub-slice and the 2-D EBR) sailed through these gates. **Acceptance MUST be an on-device `max_err` vs CPU eager with a negative control** (remove the emitted senprog → the run must fail). Abstract cell coverage is never sufficient.

---

## 5. The warp-specialization layer (PT ∥ SFP ∥ L3 overlap)

### 5.1 The empirical silu-epilogue answer (settles the premise)

The schedule today is **strictly serial PT-then-SFP**: while the 2 `batchmatmul`s run on PT, SFP idles; while the 8 SFP ops run, PT idles. There is **no** matmul epilogue and **no** `silu×mul` fusion (§2.1 — `len(computeOp_)==1` everywhere). The cross-division hand-off is the serializing boundary: in the baseline it is a `datadscs_ STCDPOpLx` LX re-layout on the `neg` SDSC (not a ring `computeOp`), which co-assignment removes. So "fuse silu into the matmul" is a non-starter — the lever is **overlap of the three engine classes** (PT / SFP / L3 ring), not body fusion.

### 5.2 The technique — K-chunked, software-pipelined mixed SuperDSC on the **one** reshard edge

The Spyre analog of CUTLASS Hopper warp specialization: overlap PT (matmul) ∥ SFP (silu/mul) ∥ L3 RIU ring (the STCDP gather) by interleaving per-core schedule steps **inside one SDSC body** via `coreIdToDscSchedule`, instead of `gather → barrier → compute`.

Today the down-proj is one un-tiled DL step `[[-1,0,0,0]]`; the reshard's mixed schedule is fully serial `[[0,-1,0,1],[-1,0,1,0]]` (`mixed_schedule` in `substrate.py:85`; row schema `[datadsc_idx, dldsc_idx, after_sync, before_sync]`, recipe 02 §3.1). The `before_sync=1` between gather and compute is exactly the no-overlap barrier.

**Four mechanical edits (no new deeptools surface beyond what the reshard already needs):**
1. **Tile the consumer DL op over K** into G chunks (e.g. G=4, each 3200 of 12800) → G partial-matmul-accumulate steps. *This also caps the per-core bridge footprint — solving the 3.2 MB > 2 MB LX sizing risk (§8) for free: 3.2 MB / 4 = 0.8 MB.*
2. **Split the full-K STCDP gather** into G per-chunk gathers (extend `build_asymmetric_reshard_bridge` into G per-K-chunk STCDPs).
3. **Rewrite `coreIdToDscSchedule`** into a depth-1 double-buffered (ping/pong) pipeline that moves `before_sync` **off** the gather and **onto** the compute step, and gives each gather an `after_sync` that waits only the *prior* transfer:
   ```
   PROLOGUE:  [0, -1, 0, 1]              # gather chunk 0 into buf A
   STEADY g=1..G-1:
              [g,  -1, 1, 0]              # issue gather chunk g into the other buf (async on the ring)
              [-1, g-1, 0, 1]            # compute partial-matmul on chunk g-1 (other buf) CONCURRENT with the gather above
   EPILOGUE:  [-1, G-1, 1, 0]            # compute the last chunk after its gather lands
   ```
   This is the AOT spelling of a CUTLASS producer(TMA)/consumer(MMA) mainloop. The down-proj's K-accumulate is **associative**, so partial-sum-per-chunk is value-identical to the monolithic reduction (no new EBR surface — each chunk reuses the same per-subpiece dest column the fix supplies, sliced over K). Steady-state critical path = `max(ring_time(K/G), pt_time(K/G))`.
4. Co-assignment stays independent (Class A is same-core, no data-op, nothing to overlap).

### 5.3 How it stacks

Pure additive rewrite on the reshard's **one** mixed-SDSC edge. It changes **nothing** in: co-assignment (same-core); the dxp import-gate patch (mixed dispatch fires on `has_dsc_schedule` regardless of step count); the PieceInfo/permutation; or the EBR fix (each K-chunk's dest column is the same `core//mb_split` derivation, sliced over K — fix is necessary-and-sufficient for the chunked form too). Implementation point: extend `substrate.mixed_schedule` to emit the pipelined rows and `build_asymmetric_reshard_bridge` to emit G per-chunk STCDPs; splice and dxp path unchanged.

### 5.4 When it helps / when it hurts (inverse of the GPU intuition)

- **Helps (fat consumer, move hideable under compute):** the **PREFILL down-proj** here is the good case — `512×4096` out, `K=12800` is a fat GEMM, so `pt_time(chunk) ≫ ring_time(chunk)` and the gather goes nearly free. Strong in the reshard's true homes: MoE router→expert→combine all-to-all, attention probs→PV.
- **Does NOT help / can hurt:** **DECODE / skinny matmuls** (PT-util ~0.2%, array idle) — the move IS the critical path and there is nothing to overlap with; there you want the reshard's *raw bandwidth* win (RIU vs HBM), not engine overlap. **Sub-stick chunks** (G too large → per-core slice < 64 fp16) regress the ring move (recipe 06: 0.95× at S=512) and add per-chunk `L3_SYNC` overhead — gate G so each chunk's per-core slice ≥ 1 stick *and* `pt_time(chunk) ≥ ring_time(chunk)`.

### 5.5 Expected win (honest framing, no invented numbers)

This is a **second-order win ON the reshard**, bounded by Amdahl on the one down-proj edge — **not a throughput multiplier**. The matmul stays `{mb:4,out:8}` either way.
- **Co-assignment** (ships now, the floor): device-proven **~7%** on the unfused SwiGLU (12.9 vs 13.9 ms), move-free, stock dxp.
- **Reshard** (gated on the fix): replaces a `12.8 MB`-fp16 HBM round-trip (≈166 GB/s shared) with a RIU-ring LX→LX move (166 GB/s/dir, LX 4.5 TB/s aggregate). Path A measured **~12%** when wired (17.4 vs 19.8 ms fused) but that run was **value-BROKEN** — treat it as an upper anchor, *not* a result. Realistic combined target: low-double-digit % over 13.9 ms.
- **Warp-spec overlap**: recovers the part of the reshard's own move latency that STRICT_ORDERING would re-serialize — *unverified*, gated on whether dxp/DCG actually issues the gather on ASYNC_DMAI concurrently with the matmul on COMPUTE (§8 risk #1).

For **prefill** specifically, the matmul O(N³) dwarfs the O(N²) hand-off and co-assignment already captures most of the move-free win — the reshard's marginal prefill value is small. The reshard *earns* the deeptools fix in **movement-bound** regimes (decode, MoE all-to-all, attention), which motivate the eventual planner generalization, not this landing.

---

## 6. Alternatives considered and why rejected

**Gather (`InputFetchNeighbor`, L3LU).** The structural insight is real and source-confirmed: L3LU fills `DestStartCondAndVal` per-piece from `outSP_` (consumer-local, **core-correct by construction**), so the gather moves the 2-D placement off the broken scalar dest leg — the more robust primitive for a general move library. **Rejected for this landing because:** (a) **not implemented** — `substrate.py` has only the L3SU scatter; adopting it now is net-new unproven inductor surface. (b) it still routes through the **same** `computeMulticastOptMetadata` chokepoint and concedes it needs the same `col-band = core//mb_split` fix on its source leg — new cost, **no** reduction in the deeptools floor. **Adopt its insight** (it points to where the DCG fix lives — the col-band mapping, not a DSM HBM stride) but **build on the proven scatter**.

**Standalone data-op SDSC (Option b).** Functionally equivalent but **more invasive** than the mixed fold: it carries the gate **plus** a consumer-binding concern (recipe §11c). The mixed fold makes the consumer's own input the bridge output → no cross-SDSC binding at all. Rejected in favor of the mixed fold.

**LX-planner extension.** The cleanest long-term framework, but **not inductor-only for a move**: planning is inductor, but the move bottoms out on the deeptools data-op (gate + EBR packer). LX is per-core private; only DCG emits ring microcode (`L3_LDU`/`L3_STU`). An inductor pass can **author** a move but cannot **execute** it. So the planner relocates inductor work into a general planner on the **same** deeptools floor — **more** inductor surface for the same blocker, and the productionised cross-shard generalization **already broke once** (commit `0b994bb`, `_partition_pieces` split_dim bug, max_err 0.669). Defer to a follow-up once the EBR fix lands and a second move-home (MoE/decode) is in scope.

**Explicit-`ebrInit_` stamp (inductor-only escape).** Stamp `dataOUT` per-core `ebrInit_ = (core//4)·1600` and hope DCG honours it. Kept as the **cheap CPU probe** (§3.3) but **not assumed to work** — the alt analysis notes the stamp lands only on the flipped DS and the per-core store may be dead; likely DCG recomputes from `cidx`. Run it first to settle DCG-vs-DSM, but plan for the deeptools fix.

**Epilogue fusion (silu×mul into the matmul).** Refuted: there is no fused SDSC body today and the `silu×mul`-into-matmul fusion does not exist on either deeptools path. Not a hand-off mechanism.

**Steer producer→consumer layout (matmul → pure-M).** Loses the matmul M-split → 1.4–1.6× regression. This is the lever co-assignment specifically *avoids* by steering the cheap consumer instead.

---

## 7. Phased implementation plan

Inductor-first; every step CPU/offline-gated before any device run; **device validation serialized (single shared accelerator — run all device probes SOLO)**. The orchestrator runs every device/build step; this document only recommends them.

**Phase 0 — Ship co-assignment (no deeptools, independent).**
1. `[inductor]` Promote `apply_coassign` → config-gated pass (`config.swiglu_coassign`) in the `_distribute_work` slot. **Verify:** offline — `[COASSIGN]` flips all 5 element-wise ops to `{mb:4,out:8}`; the `neg` `datadscs_ STCDPOpLx` is absent in emitted SDSC.
2. `[device]` Re-confirm max_abs_diff ≤ ~0.006 vs CPU eager and ~7% kernel-time win (already device-proven; re-run only if the pass refactor changes emitted SDSC). **Verify:** `allclose(1e-2,1e-2)=True` + negative control.

**Phase 1 — Settle the EBR carrier (CPU-only, no build).**
3. `[inductor]` Run the EBR attribution probe: zero-compute decode of the existing reshard debug to attribute `3200·core` to the live ring node vs HBM store. **Verify:** carrier identified.
4. `[inductor]` Run the `ebrInit_` stamp probe: stamp `dataOUT` per-core `ebrInit_=(core//4)·1600`, recompile with the patched dxp, grep `debug/sdsc_2/smc.txt @regInit:EBR:R0`. **Verify:** EBR → `0 0 0 0 3200 …` (inductor-only) or unchanged (deeptools fix required — likely).

**Phase 2 — The deeptools fix (only if Phase 1 says so).**
5. `[deeptools]` In the DCG packer (`stcdpOp.cpp` per-piece fill / `computeMulticastOptMetadata`, the carrier confirmed in Phase 1), derive the L3SU dest column from the subpiece `out_` coordinate (`core//mb_split`) instead of `cidx`. **Verify:** CPU — recompile the spliced bundle, `EBR == 3200·(core//4)`; 1-D `out:N` regression check (`core//1==core`, unchanged).

**Phase 3 — Productionise the reshard authoring (inductor, on top of Phase 0).**
6. `[inductor]` Add the `is_reduction_input_edge` planner predicate (`onchip_handoff.plan_onchip_handoffs`); fail-closed observer. **Verify:** offline — fires only on the `mul`→down-proj edge, no-op elsewhere.
7. `[inductor]` Productionise the realize pass at `generate_bundle` (replace the monkeypatch): `apply_lx_flip` + `build_asymmetric_reshard_bridge` + `splice_reshard` mixed-fold, bound by LX-base coincidence; `allocate_lx_bases` fail-closed. **Verify:** CPU — exit 0 on patched dxp, `assert_partition` 7/7, correct permutation; **note the down-proj band 128×12800 = 3.2 MB > 2 MB → REQUIRES K-chunking (Phase 5) or `build_streamed_bridge` before it can fit.**

**Phase 4 — Device-validate the reshard (serialized, SOLO).**
8. `[device]` With the fixed dxp on PATH: `max_err` vs CPU eager + negative control + kernel time vs baseline. **Verify:** `max_err` at fp16-noise level (NOT just `assert_partition`/senprog presence — those are insufficient).

**Phase 5 — Warp-spec overlap (inductor, on the value-correct reshard).**
9. `[inductor]` Extend `mixed_schedule` to emit the pipelined prologue/steady/epilogue rows; split the bridge into G per-K-chunk STCDPs; reserve two non-overlapping ping/pong LX banks (`allocate_lx_bases`, assert non-overlap). **Verify:** offline — schedule rows match §5.2; both banks ≤ 2 MB.
10. `[device]` (gated on Phase 4 value-correct) Confirm overlap actually happens — grep the senprog for concurrent ASYNC_DMAI gather ∥ COMPUTE matmul; measure delta vs the serial reshard. **Verify:** value-correct (re-run max_err) AND a measurable overlap delta; if the senprog serializes regardless of sync flags, the overlap is a no-op (risk #1) — do not claim the win.

---

## 8. Open risks / what could still break

1. **(Highest) EBR fix LOCATION is contested.** The symptom (`3200·core` vs `3200·(core//4)`) is device-proven; the *carrier for the LX-flipped path* is not. The GT headline (DSM `perfDscToSdsc.cpp:2099` stride-0) sits in the **HBM-gated** branch (`dsm.cpp:6756`), but the reshard LX-flips producer-out (`ebrInit_=-1`), so the LX branch runs and `3200·core` is recomputed in the DCG packer. The alt analysis adds a "dead HBM store" caveat. **We do not yet know with certainty it is one function.** *Mitigation:* Phase 1 attribution probe gates any build.

2. **(Highest, warp-spec) Concurrency is UNVERIFIED.** It is unproven that dxp/DCG issues the STCDP gather on ASYNC_DMAI **concurrently** with the matmul on COMPUTE when `before_sync=0` within one SDSC. The cross-SDSC pipelines (COMPUTE/ASYNC_DMAI/ASYNC_DMAO + OP_ORDERING) are documented "unplumbed". If the senprog serializes all steps regardless of sync flags, the overlap is a no-op + extra `L3_SYNC` overhead. The only on-disk schedules are fully serial. *Mitigation:* Phase 5 device check with a senprog concurrency grep; settle before claiming any overlap win.

3. **K-tiling a mixed-SDSC DL op into accumulate steps is new DCG surface.** Today the down-proj is one un-tiled DL step. G partial-accumulate steps sharing a PSUM accumulator may need machinery DCG does not expose to a hand-authored mixed schedule. *Unverified.*

4. **Down-proj band sizing (3.2 MB > 2 MB/core LX).** The full-K consumer band exceeds LX; the existing CPU-proven EBR evidence (1.6 MB silu edge) does not directly transfer to this larger edge. K-chunking (G=4 → 0.8 MB; double-buffered 1.6 MB) solves it, but double-buffering must reserve two non-overlapping banks (`allocate_lx_bases`) — the recipe's overlap-corruption bug (02 §778-780, reversed scratch overlapped at 1 MB) is the exact failure mode if mis-sized.

5. **Barrier-flag semantics inferred from doc, not a working overlapped senprog.** Getting `after_sync`/`before_sync` wrong silently corrupts (compute reads a half-gathered buffer) and `assert_partition` will NOT catch it. *Mitigation:* on-device max_err + negative control is mandatory.

6. **Stacks on a value-broken base.** The reshard is correct only after the contested fix; warp-spec is meaningless until the reshard produces correct output. Do **not** pursue overlap before the EBR carrier is attributed and the move is device-validated.

---

## Appendix — verified file map

| Purpose | Path |
|---|---|
| Co-assignment pass | `/tmp/core-to-core-wt/ab/coassign/coassign.py`, `README.md` |
| Reshard substrate (scatter only) | `/tmp/core-to-core-wt/ab/reshard/{substrate.py,pieces.py,splice_swiglu.py,cells.py}` |
| Reshard analysis | `/tmp/core-to-core-wt/ab/reshard/{PATH_A_PROGRESS.md,MECHANISM_AND_BLOCKER.md}`, `ab/STATUS.md` |
| Work-division slots | `/tmp/core-to-core-wt/torch_spyre/_inductor/passes.py` (`_distribute_work`, `_maybe_scratchpad_planning`) |
| Cost model | `/tmp/core-to-core-wt/torch_spyre/_inductor/work_division.py:982` (`cost_model_matmul_division`), `:486` (`apply_splits`) |
| Split helpers | `/tmp/core-to-core-wt/torch_spyre/_inductor/pass_utils.py` (`apply_splits_from_index_coeff`) |
| Patched dxp (built) | `/home/adnan/dt-inductor/build/deeptools-onchip/dxp` |
| Deeptools source | `/home/adnan/dt-inductor/deeptools-onchip/dcg/` (`dcgbeCodegen.cpp:2720`, `dcg_fe/pcfg_gen/stcdpOp.cpp:2676/5033`, `inputNeighFetchOp.cpp:553`) |
| Captured baseline SDSC | `/tmp/c2c-unfused-final/` (`sdsc_0..9.json`, `bundle.mlir`) |
| Coassign inductor-logs | `/tmp/core-to-core-wt/ab/results/coassign_unfused_1x512x4096/inductor-logs/` |
