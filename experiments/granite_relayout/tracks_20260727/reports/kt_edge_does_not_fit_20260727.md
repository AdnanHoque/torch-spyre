# K^T All-Gather Edge â Implementation Spec

Target: SenDNN P01 / QC_3 Â· QC_12 Â· QC_21 â the post-RoPE, scaled, transposed K becoming the weight operand of the QÂ·K^T score matmul.

---

## Can this edge fit at all?

**No. A literal 32-way replication of K^T does not fit at `DXP_LX_FRAC_AVAIL=0.2`, and it misses by 176,896 B (10.9%) â a shortfall no address assignment can recover.** The cheapest form that *does* fit is a **4-core broadcast cohort per KV head**, which requires the score BMM's work division to change from `{query_pos: 32}` to `{kv_head: 8, query_group: 4}`. That lever already exists in the tree, switched off.

First, correct two premises in the brief.

**The brief's "32 KiB/core" is off by 32Ã.** SenDNN's destination is one piece covering the whole tensor, placed on all 32 cores: `labeledDs_[1].lxSize_ = 1048576`, `PieceInfo` length 1 with `dimToSize_ {in:128, out:512, x:8}`, `PlacementInfo[0].memId = [0..31]`, `startAddr = 32768` on every core. Cross-checked three ways: `128Ã512Ã8Ã2 B = 1,048,576`; `totElements 524288 Ã wordLength 2`; and `dst_top 1,081,344 â startAddr 32,768`. 32,768 B is the *source* piece size, and 32768 in the dump is an **address**, not a size. A whole-tensor replica costs **1,048,576 B per core**.

**The brief's "7434 KiB/core already resident" is not residency.** It is a sum of every placed buffer's size over the entire graph (measured: 7100.0 KiB in `merged_p07on_full40_5x`, 6916.0 KiB in `roles_probe_1x`). Simultaneous live high-water is **1280.0 KiB**. There *is* headroom â just not 1 MiB of it at the score-BMM instant.

**The budget** (`torch-spyre/torch_spyre/_inductor/scratchpad/allocator.py:110-115`, `:1242-1259`):

```
_LX_PHYSICAL_CAPACITY_BYTES            2<<20  = 2,097,152
_LX_PROGRAM_DEBUG_RESERVATION_BYTES    64<<10 =    65,536
tracker                                       = 2,031,616   (1984.00 KiB)
_lx_planning_size() = round_up_128(int(2,031,616 Ã 0.8))
                                              = 1,625,344   (1587.25 KiB)  <- our frontend share
DXP remainder                                 =   406,272   ( 396.75 KiB)
```

**Live at the score BMM** (`runs/roles_probe_1x/allocations.jsonl`, alloc t=23; identical set at t=25 in `merged_p07on_full40_5x`). All sizes are per-core â verified two ways: `allocator.py` writes `size_per_core`, and buf20's 524,288 B reproduces exactly from its own SDSC extents (512 keys Ã 16 queries/core Ã 4 mb Ã 8 x Ã 2 B).

| buffer | role | B/core |
|---|---|---|
| buf20 | scores (output) | 524,288 |
| buf14 | Q, already LX (P06) | 131,072 |
| buf4 | carried across the BMM | 65,536 |
| **live total** | | **720,896 (704.00 KiB)** |

A relayout **source stays sliced and stays co-live with its destination** â this is the fact that breaks the arithmetic. Proof from our own artifact: `buf52` (131,072 B, addr 524288, uses [56,57,60]) co-exists with `__spyre_lx_relayout_destination__:buf52` (524,288 B, addr 0, uses [57,58,60,61]). So a 32-way gather costs source **plus** destination:

```
704.00 KiB  live
1024.00 KiB  replicated K^T destination
  32.00 KiB  K^T source slice (does NOT go away)
-----------
1760.00 KiB  required   =  1,802,240 B
1587.25 KiB  budget     =  1,625,344 B
             OVER BY      176,896 B  (172.75 KiB, 10.9%)
```

This is a sum over simultaneously-live buffers, so it is a necessary condition on *any* legal placement. No fragmentation trick rescues it. The eviction ladder: dropping buf4 leaves it over by 111,360 B; you only fit by *also* evicting Q â which kills P06, our existing query-side win â and then with 19,712 B (1.2%) of slack and zero tolerance for anything moving. Fitting as-is would need `DXP_LX_FRAC_AVAIL â¤ 0.112`; production is 0.20 and fixed.

SenDNN affords it because it owns the whole 1984 KiB tracker, not 80% of it. Independent confirmation that its plan is simply not expressible under our contract: SenDNN's own LxRelayout high-water is **1,766,400 B = 1725.0 KiB** (`Stcdp_QC_38` / `Exx2_QC_{2,4,5,6}` lds[0] at 1,635,328 + 131,072) â above our entire frontend share. And SenDNN was itself budget-bound: it spent LX on K and left V in HBM (`bmm_1-BMM_1.json` `labeledDs_[1]` KERNEL, `hbm isPresent=1`).

### What fits

Our score BMM today is split 32 ways on query position â `origsdsc_debug_18_batchmatmul.json`, `numWkSlicesPerDim_ {in:1, out:1, mb:1, x:1, y:32}`, `coreIdToWkSlice_[c] = {y:c}` â which is *semantically identical* to SenDNN's `mb:32` (`bmm-BMM_1.json`, `coreIdToWkSlice_[c] = {mb:c}`). That is precisely why replication is forced: with 16 query positions of *all* 32 query heads on every core, every core needs every key of every KV head.

Change the division to `{kv_head: 8, query_group: 4}` and each core owns one KV head:

| buffer | B/core, head-owned | vs today |
|---|---|---|
| buf20 scores | 1Ã1Ã512Ã512Ã2 = 524,288 | unchanged |
| buf14 Q | 1Ã1Ã512Ã128Ã2 = 131,072 | unchanged |
| buf4 | 65,536 | unchanged |
| K^T destination (**1 KV head**) | 1Ã512Ã128Ã2 = **131,072** | **8Ã smaller** |
| K^T source slice (buf66) | 32,768 | â |
| **total** | **884,736 (864.00 KiB)** | vs 1587.25 KiB budget |

Headroom **740,608 B (723.25 KiB)**. The plan-wide peak (1,310,720 B at op30) is untouched, because the +163,840 B lands only at the BMM instant where live is 720,896.

The relayout becomes a **broadcast to a 4-core cohort (fanin 4)**, not a 32-way all-gather. The lever is already written, with the design intent stated verbatim in a comment at `work_division.py:1146-1163`: *"Keep those four cores contiguous so the K consumer view is a four-core broadcast cohort and the Q consumer view owns one full query head per core."*

---

## The change

**Zero correctness guards are removed. `_restickify_barrier` is not touched. `SPYRE_LX_ALLOW_RESTICKIFY_READ` stays 0.** See the next section for why.

Line numbers below are from `allocator.py` md5 `6b989e29b2ee771c264bafe5f372bf77` (2557 lines). **The tree moved three times mid-investigation** (`827f3165` â `547d720b` â `6b989e29`). Re-anchor by symbol name before applying anything.

### 1. Stop suppressing the edge (`scripts/run_integ.sh:60` and `:62`)

This is the whole reason no `buf66 â buf20` plan is ever collected. Both branches of the disabled-sources list end in `,buf66`:

```
lx_relayout.py:417-421   disabled_sources = <- config.lx_relayout_disabled_sources
lx_relayout.py:460       if dep.name in disabled_sources: continue
```

The skip happens *before* any view machinery runs. So the `"op not allowed"` reject on buf66 (`allocator.py:579-580`) is self-inflicted by an env var, not by the `OP_OUTPUT_GOOD_FOR_LX_REUSE` allowlist.

```diff
-SPYRE_LX_RELAYOUT_DISABLED_SOURCES="...,buf66"
+SPYRE_LX_RELAYOUT_DISABLED_SOURCES="..."
```

No allowlist edit is needed. `_op_output_good_for_lx_reuse` (`allocator.py:497-506`) admits a buffer if the op name is in the allowlist **or** the buffer is a planned relayout source. Once a plan with `source_name == "buf66"` is collected, `"op not allowed"` dissolves on its own.

### 2. Make the edge affordable (`scripts/run_integ.sh:90`)

```diff
-SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=0
+SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=1
```

Existing scaffolding, all verified present: `config.py:210-212` (flag), `work_division.py:805-806` (gate on buf20), `:931-932` (returns `{"kv_head": 8, "query_group": 4}` for shape `["1","8","4","512","512"]`), `:1146-1163` (extent binding on `[8,4,512,8,2]`, raises `Unsupported` on mismatch rather than mis-slicing, sets `_spyre_oracle_gather_dim_symbol = query_group_sym`).

### 3. Leave COMPACT_GQA exactly as production has it

Keep `SPYRE_RELAYOUT_ORACLE_COMPACT_GQA_BUFFERS=buf18,buf29` (`run_integ.sh:28`). It supplies the compact 5-D planner shape `[1,8,1,512,128]` for buf18/buf66 (`decompositions.py:565-569`) and the buf29 division â both still wanted.

**Do not add buf66.** `config.py:171-172` defaults to `buf18,buf29,buf66`, which is *wrong for this plan*: it would pin the restickify to `{kv_head:4, token:8}` (`work_division.py:909-911`), giving each core 2 KV heads Ã 64 tokens. The solver's default `{mb:4, x:8}` â verified emitted in `origsdsc_debug_17_ReStickifyOpHBM.json`, `coreIdToWkSlice_[c] = {mb: c mod 4, x: c div 4}` â puts 4 cores on each KV head, which is exactly the source cohort a head-owned gather wants. **Align `config.py:172` down to `buf18,buf29`** so the code default cannot silently destroy the cohort.

### 4. Emission-order pin â conditional, decided by gate G3, not by guesswork

There is a known class of bug here where the planner's core map and the emitted SDSC's `coreIdToWkSlice_` disagree about which axis varies fastest. It is confirmed on buf18 (plan: `core = 4Â·token + kv`; emitted `origsdsc_debug_16_mul.json`: `core = token + 8Â·kv`) and â decisively â it already happened on buf29 and was hand-patched: `spyre_kernel.py:147-168` rewrites `producer_map` to `{core%8, core//8}`, which is precisely the emitted order. The mechanism is `superdsc.py:1131-1145` (`contiguous_dim` follows `gather_dim` only if the symbol survives `symbol_mapping`); four accepted oracles have hard-coded `contiguous_dim = 1` pins at `superdsc.py:1160 / 1181 / 1194 / 1206`. There is none for head-owned.

**Do not add a pin blind.** Run G3, read both orders, and add a pin beside the existing four only if they disagree â not another `producer_map` hand-patch.

---

## Why this cannot silently corrupt

### The barrier relaxation: DO-NOT-APPLY

The brief's key observation is *logically* sound. If a core physically holds the entire tensor, then read-set â whole-tensor by set inclusion alone, for any output frame, any stick permutation, any padding. There is no geometry left to vary. It fails for two reasons that have nothing to do with the logic.

**Reason 1 â "replicated" is a property of a different buffer.** SenDNN's op order is `restickify â mul_6 â all-gather â score BMM`. Ours is `mul â buf18 â restickify â buf66 â score BMM`. SenDNN gathers *downstream* of its restickify. The positional analogue of SenDNN's `mul_6_out` is our **buf66**, not buf18. And buf66 is not barred by `_restickify_barrier` at all â its only restickify use is its own producer, excluded by the `graph.operations[u].name != name` test. Meanwhile SenDNN keeps its *own* restickify's input in HBM: `bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify.json`, `labeledDs_[0]` (INPUT) `memOrg_.hbm.isPresent = 1`, `labeledDs_[1]` (OUTPUT) lx-only. **The reference implementation accepts exactly the round-trip our barrier enforces.** There is no parity argument for lifting it.

**Reason 2 â the counterexample.** Suppose we relax the barrier whenever the buffer is the source of an ALL_GATHER/BROADCAST plan ("it gets replicated, so it's safe"). A relayout **source is stored sliced**; the replica is a separate destination buffer (proved above with buf52). So buf18 gets an LX address at 32,768 B/core, sliced. Then, from our own artifacts:

- Producer `origsdsc_debug_16_mul.json`: `{out:1, mb:8, x:4}`, `core = (c mod 8) tokens + (c div 8) kv`. **Core 0 holds tokens 0â63 of KV heads 0â1.**
- Consumer `origsdsc_debug_17_ReStickifyOpHBM.json`: `{out:1, mb:4, x:8}`, `core = (c mod 4) tokens + (c div 4) kv`. **Core 0 must write tokens 0â127 of KV head 0.**
- Tokens 64â127 of KV head 0 were written on **core 1**. Core 0 reads bytes it does not hold. Wrong K, valid-looking logits.

The guard a skeptic would expect to catch this â the core-div/PerCoreView mismatch check â **is deliberately blinded for exactly this case**: `scratchpad/utils.py:468-475`, `if buf_name in planned_relayout_sources and writer_cores is not None: num_cores = writer_cores`. It discards the mismatch and sizes the buffer `dev_size // writer_cores` anyway. `allocator.py:945-956` passes `planned_sources = set(self._lx_relayout_plans_by_source)` into both `get_ncores_for_buffers` and `mem_usage_by_buf`. So the moment buf18 becomes a relayout source â the stated goal â the last check stops firing.

This is not hypothetical. `runs/kt_restickify_lx_1x` (`contract.txt: restickify_lx=1`) already ran it and died in codegen:

```
ValueError: allocation uses unmapped device dimension 4; slot=1;
  device_dim_to_sdsc_dim={'2': x, '1': out, '0': mb}; dim_order=[x, out, mb];
  per_device_dim={'0': 0, '4': 1}          superdsc.py:394
```

`dim_order=[x,out,mb]` is the restickify's **output** frame; the split dims `{0,4}` come from buf18's **input** frame. Device dim 4 has no SDSC dim in the output frame â that *is* the cross-frame hazard. It raised only because `superdsc.py:390-398` raises when `int(slot) != 0`; **at slot 0 it silently `continue`s and drops the dimension.** The loud failure was luck of the slot, not design. And the plan it produced was a 32,768 B all-to-all permute, not an all-gather â so even on success it would not have been this edge.

`config.lx_allow_restickify_read` (`config.py:51`) currently short-circuits `allocator.py:790` **with no predicate whatsoever**. Leave it at 0.

### Why the proposed change is safe

It touches no guard. `_restickify_barrier` still bars buf18; buf18 still round-trips HBM, as SenDNN's does. The only new correctness surface is the plan/emission core-order agreement, and it has two falsifiable conditions, both checkable from compile artifacts before any device run:

1. **No un-redirected consumer of buf66.** Removing buf66 from the disabled list re-arms the `utils.py:468-475` blinding for buf66. That is only dangerous if some consumer reads buf66 *directly* under a division different from the writer's. Check: `buf66.uses` must have exactly two entries â its producing restickify and the score BMM â and the BMM must be the relayout consumer (rewritten to the destination). Today `runs/p09on_lx02_1x` shows `buf66 uses [22,24]`. If a third use ever appears, stop.
2. **Plan core map == emitted core map** for buf66's source. That is gate G3, mandatory, and it must pass before any device number is believed.

---

## Gates

Cheapest first. Every correctness check is paired with a transport check **from the same run**. Token 203 alone proves nothing: a planned-then-dropped edge still yields 203.

**G0 â free, and it decides whether to spend the week.** Determine whether the score BMM's KERNEL operand fetch from HBM is duplicated per core or shared. Observable: the post-PCFG descriptor for `origsdsc_debug_18_batchmatmul.json` `labeledDs_[1]` â either `ldsShareInfo_` is non-empty (shared fetch) or there are 32 distinct HBM read descriptors. The origsdsc dumps are pre-DXP (`datadscs_` empty, every `lxSize_` is the uint64 sentinel), so this needs a perfdsc-stage dump on our side. **If the fetch is already shared, the ceiling collapses to ~0.65 ms and this project should stop.** See the payoff section.

**G1 â compile only, no device, no code change.** `SPYRE_RELAYOUT_ORACLE_PREFILL_QK_HEAD_OWNED=1` on the accepted stack, everything else unchanged. Answers *"is the geometry even reachable"* for zero device time.
- Correctness: it compiles; `work_division.py:1150-1157` does not raise `Unsupported`; buf20's emitted `numWkSlicesPerDim_` is head-owned rather than `{y:32}`.
- Transport/cost: `allocations.jsonl` â no new `no room on scratchpad` rejections; the softmax chain (roles_probe ops 19â27, currently `{y:32}`/`{mb:32}`/`{out:32}`) has not sprouted new core-div mismatches or new relayouts; high-water still â¤ 1,310,720 B.

**G2 â compile only.** G1 + buf66 removed from `SPYRE_LX_RELAYOUT_DISABLED_SOURCES`. This is the whole edge, planned.
- Correctness: `allocations.jsonl` â `buf66` `reject_reason` is `null`, `size 32768`, real address; `buf18` `reject_reason` is **still** the restickify-barrier string (it must not have moved); `buf66.uses` still has exactly 2 entries.
- Transport: `relayout_plans.jsonl` contains a `buf66 â buf20` plan, kind `broadcast` or `all_gather`, **ratio 4**, plus a `__spyre_lx_relayout_destination__:buf66` of 131,072 B in `allocations.jsonl`. If the plan is absent, nothing downstream matters.

**G3 â compile only, MANDATORY, same artifacts as G2.** Compare the plan's `source_core_id_to_device_slice` for buf66 against `coreIdToWkSlice_` in `origsdsc_debug_17_ReStickifyOpHBM.json`. They must agree on which axis varies fastest across cores. Precedent: buf18's disagree, and buf29's disagreed and needed `spyre_kernel.py:147-168`. If they disagree, add a `contiguous_dim` pin beside `superdsc.py:1160/1181/1194/1206` and re-run G3. **Do not proceed on a disagreement â that is the silent-wrong-K failure mode, and no timing gate will catch it.**

**G4 â device, 1Ã.** Token 203, full 40 layers.
- Correctness: token 203.
- Transport, same run: the score BMM's `labeledDs_[1]` shows `lx.allocateNode_` non-empty and `hbm.isPresent = 0`; post-PCFG HBM/DMA byte count for that operand is **zero**; `STCDP_FINAL_END` present with the expected payload.

**G5 â device, 5Ã.** Median device kernel ms over 5 requests, against the accepted stack's median from the same session. Report the delta alongside the Â±1.2 ms floor. Claim a win only if the delta exceeds 1.2 ms **and** G4's transport proof held in each of the 5.

---

## Expected payoff, honestly

**Whether this edge is worth doing at all is currently unknown, and the unknown is one measurement (G0) away.**

What the change removes: buf66's HBM write (1.00 MiB/layer) and the score BMM's HBM read of K^T. What it does **not** remove: buf18's write and the restickify's read â those stay in HBM by design, exactly as SenDNN keeps them.

K^T whole tensor = 8 KV heads Ã 512 keys Ã 128 head_dim Ã 2 B = 1,048,576 B = 1.00 MiB (matches SenDNN's stated payload). At 129.7 Ã 10â¹ B/s effective:

| assumption about the BMM's HBM read | saved/layer | saved Ã40 | ms | clears Â±1.2 ms? |
|---|---|---|---|---|
| 32 independent per-core fetches | 33.00 MiB | 1,384,120,320 B | **10.67** | yes, 8.9Ã the floor |
| shared per corelet pair (fold factor 2) | 17.00 MiB | 713,031,680 B | **5.50** | yes |
| fully multicast, one fetch | 2.00 MiB | 83,886,080 B | **0.65** | **no â below the floor** |

**Break-even: the saving exceeds 1.2 ms only if the HBM weight fetch is duplicated more than ~2.7Ã across cores.** (1.2 ms Ã 129.7e9 = 155.6 MB total = 3.71 MiB/layer; minus the 1.00 MiB write leaves 2.71 MiB of read.)

Context for the ceiling: the gap to SenDNN is 246 â 190.4 = **55.6 ms**. Even the optimistic 10.67 ms is **19% of the gap** â real, but not the whole story, and the audit's claim that this edge is 73% of the *missing LX bytes* is a statement about bytes on the ring, not about milliseconds off the clock.

In the multicast world the edge would still be *correct* and would still land â you simply would not be able to tell from timing, only from the G4 transport proof. That is a legitimate outcome to report, but it is not a win, and it is not worth the engineering risk described below. **Run G0 first.**

Two honest costs against the ceiling:
- **The softmax chain.** ops 19â27 all run token-major today. Re-dividing buf20 may force new relayouts or new core-div rejections there. The reduction axis (keys) stays core-local under both divisions, so the reduction structure itself is safe â but a cascade of new relayouts could eat the win outright. G1 measures this for zero device time.
- **Parity with SenDNN's full 12-edge plan is not attainable at `DXP_LX_FRAC_AVAIL=0.2`.** SenDNN's LxRelayout high-water (1,766,400 B) exceeds our entire frontend share (1,625,344 B). Some of its edges are structurally out of reach without renegotiating that contract, which is out of scope.