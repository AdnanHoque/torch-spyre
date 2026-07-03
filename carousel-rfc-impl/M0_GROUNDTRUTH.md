# M0 groundtruth (live-code verified)

## torch-spyre

GROUNDTRUTH G1 — verdicts (torch-spyre @ cf67411, read-only).

**Q1 — Prefill wide-N matmul work-division & regime.**

The only cost model is `_matmul_split_cost` (work_division.py:654–704), driven by `_cost_model_matmul_planner` (707–824) as Pass 2 (`cost_model_matmul_division`, 982–995). It is UNCONDITIONAL for any `batchmatmul` reduction op (726, 928). It picks the lowest-cost `(b,m,n,k)` core-split of the iteration space — **there is NO relayout / weight-replication plan in the model**; it only assigns iteration-space slices to cores. So "relayout plan" = NO by construction.

For a wide-N prefill matmul (B=1, large M=S, wide N, K) it does **not** pick pure-N and **not** pure-M — it picks an **M×N co-split with max(m,n) ≤ 8** (e.g. m4×n8 / m8×n4). Two terms force this: `cohort_penalty = max(1, max(m,n)/_COHORT_LIMIT)` with `_COHORT_LIMIT=8` (684) makes any single-axis split >8 inflate the HBM term, and `target_m_us` (695–698) penalizes M-splits whose per-core M under-fills the PT pipeline (`pt_passes < _TARGET_PT_PASSES=8`), biasing away from pure-M toward N. This matches the device-confirmed behavior in prior work (m8,n4→m4,n8 flips).

Crucially the HBM term counts each operand **once** — `bytes_total = (M*K + K*N + M*N)*2` (683) — i.e. the model **already assumes the weight is read once and broadcast is amortized up to cohort 8**. It never models m-fold DRAM weight replication. Consequence for the weight carousel: (a) the emitted co-split replicates W only **m-fold** (the M-split count, ≤8), not ×P=32; (b) that replication is invisible to the cost model, so a carousel that saves DRAM weight bytes **cannot be rewarded by the current cost function** — it would need a new HBM term. Flag as load-bearing.

Regime: the model adds `compute_us + hbm_us` and they come out comparable, but the emitted plan is **compute-underfilled, not cleanly DRAM-bound** — the `pt_eff = sqrt(pt_passes/8)` derate (675–678) is precisely the array-under-fill knob, and device measurement puts wide-N prefill at ~29.5% PT-util. So the carousel's ×P wall-clock DRAM claim is **moot for prefill**; its real value is array-fill + seam-transparency (consistent with the RFC's own caveat).

**Q2 — Decode attention KV cache / layout / RoPE.**

There is **no KV-cache, paged-attention, or rotary/RoPE op anywhere in torch-spyre** (grep clean; the only `index_put` hit is an autocast promotion rule, spyre_autocast.cpp:136). SDPA is **decomposed**, not fused: `spyre__sdpa_overrideable` (decompositions.py:498–575) emits scale → GQA expand+**materialize** (`key.unsqueeze(2).expand(...).flatten(1,2)`, 534) → QK^T `matmul` (538) → causal mask → `torch.softmax` (549) → attn@V `matmul` (563). `logsumexp` is a dead `torch.empty` (556): **no flash/online-softmax, no LSE ring-fold — the KV-carousel's merge operator does not exist in the lowering.**

KV cache layout is therefore **not decided in torch-spyre** — K/V arrive as SDPA inputs, assembled upstream (model graph). The attention compute becomes two 4D `spyre.batched_matmul` ops (temp_passes.py:287–360) whose **batch dims are (B,H)**; the same `_cost_model_matmul_planner` engages (batch_dims=[B,H], m_dim=Sq). So "head-split" is not hard-wired KV layout — it emerges per-op as a **batch-split candidate**, and is actively **penalized** by `batch_penalty = b**1.4` (702, `_BATCH_SPLIT_EXPONENT`), competing against N-splitting the KV-sequence dim (Skv) of QK^T. GQA is materialized to 32 heads (534), so the "only H_kv=8 channels active" underfill is a placement/DMA fact upstream, not visible here.

RoPE-before-cache (A-RoPE): **not verifiable from torch-spyre** — no RoPE op exists; RoPE runs upstream as generic pointwise ops. A-RoPE is the standard FMS/Granite pattern but is a model-authoring fact, not enforced or observable in this backend. Flag as model-level, needs the FMS graph to confirm.

## deeptools

All probes verified against live compile-pipeline code. Here are the verdicts.

---

**PHASE-0 SHUFFLE CAPABILITY PROBES — deeptools @ codex/ah-comms-collectives**

All findings are on the LIVE lowering path: `Dxp::insertRelayoutSdsc` is invoked from `dxp/dxp.cpp:615` (`runDsmRelayout`), a DXP pass, not a standalone tool.

**P1 — rotation δ=±1 / arbitrary coordinate reshard: LIVE.**
`dxp/SdscRelayoutInsertion.cpp:119 Dxp::insertRelayoutSdsc`. A relayout fires on ANY mismatch between producer and consumer coordinate maps: `allocCoords.coreIdToWkSlice_ != sdsc->coreIdToWkSlice_` (lines 135–137). It then "builds STCDPOpLx pieces from producer tensor coordinates to the consumer compute distribution, including replicated output pieces for grouped all-gather" (comment 144–147). A ±1 ring rotation is a strict subset of an arbitrary `coreIdToWkSlice_` permutation → accepted. Multicast (1:many) is live: `op->reqMulticast=true` when a piece has >1 dest memId (`dcg/.../stcdpOp.cpp:180`), handled at L3LU (`stcdpOp.cpp:1160`). Strong corroboration: the just-landed flash-allgather feature (`SdscRelayoutInsertion.cpp:58–66`, `synthesizeLayoutAllgatherRestickifyMovementPlan`) is *classified* as an all-gather but *physically realized through this same generic STCDPOpLx reshard path* (explicit comment 101–103) — i.e. a real all-gather already lowers through the generic reshard.

**P2 — async/overlap: GATED (barrier for matmul today).**
The relayout is emitted as its own `SuperDsc` (`SdscRelayoutInsertion.cpp:160,449`) inserted at its own program step (`memTrackers->insertPsBefore(ps)`, line 178) → default schedule is a sequential barrier producer→move→consumer. The overlap hook is real and already fuses a relayout move into consumer input-fetch: `dsm/dsmperf.cpp:3725 overlapInpFetchWithCompute` detects `parRelayoutStcdp = (parDdsc->opName==STCDPOpLx)` (line 3762) and overlaps it — BUT it is hard-gated to `Conv2D`/`SparseConv2D` consumers only (3733–3736). Matmul/BMM are PriOps (`dsm/dsmds.cpp:27 isPriOp`) but not Conv → hook does not fire. Eligibility gate is carousel-compatible: `dsm/graphOptimizer.cpp:18491 assignCanOverlapInpFetch` only disables overlap when the source STCDP *changes layout* (`isSrclayoutChangeStcdp`); a seam-transparent core-reshard/rotation keeps `layoutDimOrder` identical → flag stays true. Verdict: usable hook base, but overlapping a SHUFFLE with a matmul needs a backend edit (extend the Conv-only consumer gate to matmul). This is the open perf gate, as expected.

**P3 — LX→LX no HBM bounce: LIVE (conditional on LX capacity).**
Primary path emits STCDPOpLx with LX-pinned input and LX output, no HBM staging (`SdscRelayoutInsertion.cpp:209–260`, "Lx space found, inserting stcdpLx"). HBM bounce (STCDPOpHBM, line 494+) is a *fallback* taken only when `lx_space_found==false` (line 203) — i.e. no contiguous LX block for the post-relayout form. Confirmed: LX→LX is default; carousel must size K-slabs so the destination form fits LX (double-buffer budget).

**Fold — single-AIU move-then-reduce: BLOCKED / absent.**
`STCDPOpLx` (`dsc/dataOpDsc.h:479 : baseSTCDPOp`) is pure movement — no reduce/accumulate member (grep of reduce/accum/sum in stcdpOp.cpp hits only address/padding). The only recv→op→send (`COMPUTE_TREE`) primitives live in the multi-device collective layer `dsm/coll/` (allreduce.cpp/allgather.cpp; `getCommSize/next_rank`). Their sentient mapping (`dsm/dsm.cpp:3781`) requires `_WORLD_SIZE/_RANK/_DEV_LIST` + `doCollectiveOnDevice` → multi-AIU/cross-chip, not an intra-AIU 32-core ring. No single-AIU fused move-then-reduce exists; the KV LSE ring-fold must be composed as STCDPOpLx (move) + a separate SFP LSE-combine op per hop.

**A3 — channel-affine LPDDR placement: BLOCKED / not exposed.**
Placement model: `FoldManager<std::vector<int>> memId; // in HBM Id=-1, in Lx id=coreid` (`dsc/dataOpDsc.h:184`). HBM is one flat `memId=-1` space; the only spatial affinity is per-core LX. `hbmTrack` is a linear address tracker (`dsm/dsmds.h` `hbmStartAddress`), no channel/bank field; allocator grep for channel|bank is empty. `XRF_CH` (`dsm/sharedFuncs.cpp:667`) is weight-staging interleave for matmul, NOT LPDDR channel affinity. Verdict: the compiler cannot pin a persistent HBM region to a channel; LPDDR channel is a fixed HW function of address. The KV-carousel "all 32 channels stream" premise is only realizable indirectly via which CORE (LX memId) owns/streams a shard — not via HBM channel placement.

**RFC gating:** Weight carousel — P1 (rotation+multicast) and P3 (LX-local) LIVE; wall-clock win caveated by the P2 matmul barrier (needs the Conv→matmul overlap-gate extension). KV carousel — P1 reshard LIVE, but the Fold primitive is absent (compose move+SFP) and channel affinity (A3) is not expressible, so the ×4 ceiling argument must rest on core-ownership streaming, not HBM channel pinning.