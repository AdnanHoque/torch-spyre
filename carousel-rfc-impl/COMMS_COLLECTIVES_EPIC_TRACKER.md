# Granite Comms-Collectives Epic — Master First-Principles Progress Log

**Epic #3049 · Remove avoidable Granite HBM activation spills with on-chip LX communication collectives**

This is the prose-complete, self-contained companion to the HTML tracker. Every quantitative claim is tagged by evidence kind:

- **(MEASURED)** — device or trace-derived.
- **(STRUCTURAL)** — a compile-artifact delta (SDSC/plan counts); the *value* is not verified.
- **(VALIDATED)** — device-free reference or unit test; correct by construction, not yet on device.
- **(MODELED)** — analytic result from the cost model.

No speedup is ever claimed from wall time alone — that is an epic guardrail. Wall time is reported flat-to-noise; only kernel-time from archived traces counts. Ownership is split two ways: **codex** owns the mechanism/lowering half, **claude** owns the selectability + arithmetic-collective half.

---

## 1. North star, why HBM round-trips are the enemy, and the ring physics

### The north star

When an intermediate Granite activation is already resident in a core's on-chip **LX scratchpad**, spilling it to HBM and reloading it *only* to satisfy a different consumer's layout or work-division is pure waste — a full write plus a full read of bytes that never had to leave the chip. The epic replaces that round-trip with a direct **core-to-core move over the ring (an LX→LX relayout)**.

The move is expressed as **DLDSC coordinate metadata**, and the pipeline is a clean contract chain: Inductor records the residency/compute contract → the LX planner classifies (and eventually *costs*) the communication class → Deeptools checks legality and synthesizes the physical move → later work overlaps that move with compute. Weight restickifies are explicitly **out of scope** — they are handled by offline preload/prelayout, not by this substrate.

### Why HBM round-trips are the enemy

A Spyre accelerator is up to 32 cores, each with a small (~2 MiB) on-chip LX scratchpad, wired into a ring fabric. HBM is one flat off-chip space (~166 GB/s, no channel affinity). The matmul array is weight-stationary: activations stream past resident weights. When a producer computes an activation that already lives in LX, and the next consumer needs the *same values* under a different per-core ownership or layout, the compiler today spills the whole tensor to HBM and reloads it. Those are bytes that were already on-chip; the round-trip is the avoidable cost the epic attacks.

### The ring physics

The ring is **not one flat "bandwidth number."** Device measurement, using an additive-differential slope method with R²≥0.998 (MEASURED), shows the variable that sets effective bandwidth is the **peak per-link transfer count**, not the burst size and not a hidden hardware cap.

- A **uniform p→p+1 shift** places exactly one transfer on each link (occupancy = 1) and reaches the fast band: **54 GB/s at 4 MB, 90 GB/s at 8 MB, 130 GB/s at 16 MB** (MEASURED), with a streaming asymptote of **~244–254 GB/s** (MEASURED).
- An **all-to-all scatter or a linear fold-to-one** piles many transfers (4–9) onto the busiest link and collapses to a contention floor of **~34–36 GB/s** (MEASURED, 524 descriptors).
- At byte-matched volume, the uniform shift beats scatter by **2.5× (8 MB) to 3.6× (16 MB)** (MEASURED).

Two hard consequences are banked from this physics:

1. **Burst is not the lever.** Enlarging per-descriptor burst 50–100× at fixed bytes moved effective bandwidth by a **0.958× median / 1.004× min ratio** (MEASURED) — zero improvement — versus a naive "BurstEfficiency" model that predicted **2.9×**. The perf model assigns raw 128 to every link, and only per-link contention pulls the effective number down.
2. **The bandwidth-optimal ring is unreachable for LX-resident operands.** The ring-carousel-vs-naive crossover sits at **~5.2 MiB/head** (MODELED), which is *above* the ~2 MiB/core LX cap. Any operand is therefore force-tiled below the crossover, and moves become **F-dominated** — ~**7.3 µs** fixed per STCDP execute (MEASURED). The only real win banked to date is thus **HBM round-trip elimination**, not ring transport efficiency. Every number below is tagged to keep that distinction honest.

---

## 2. The convergence: two halves of one pipeline

The work is one ring-communication substrate, not two competing efforts. Both halves meet at the **DLDSC coordinate-metadata contract**.

- **codex builds the mechanism/lowering half** — making the on-chip collective *legal* and *value-correct*: emitting the DLDSC coordinate metadata, classifying topology, and letting Deeptools synthesize the physical LX→LX move (scatter, flash all-gather, matmul-operand broadcast).
- **claude builds the selectability + arithmetic-collective half** — the cost model that makes the planner actually *pick* the on-chip move, plus the reduce lane codex explicitly parks.

They are provably the same substrate at three points:

1. **Same edge, two owners (all-gather).** codex's `matmul_operand_broadcast` and claude's attention QKᵀ all-gather are the *same class in the same DCG region* — codex's is the AV value-operand (buf21/Tensor1), claude's is the QKᵀ K-operand. Both have an archived/passing run the production loop-scoped path cannot yet reproduce clean. They converge on one shared KERNEL-operand debugging effort.
2. **Mechanism ↔ selectability (scatter).** codex's productionized scatter is the *complement* of claude's Bet-2 per-link contention cost term: codex makes the move legal; the cost term makes the planner prefer it. codex explicitly names "reshard-aware work-division" as future intent — that intent *is* the Bet-2 term, landed on top of his DLDSC lowering.
3. **Ours-extends-theirs (reduce).** codex parks reduce/all-reduce as out of copy-relayout scope (value-combining collectives need reduction axis/op/dtype/identity, not a coordinate copy). claude owns that uncovered arithmetic-collective lane via the LSE ring-fold.

One hoped link is **honestly refuted**: codex's flash restickify-on is a *dense* layout-form change, whereas claude's Bet-1 flash-in-a-loop wall is the *sparse* `amax(dim=-1)` reduction gather — the same-named pass, a different subclass. They do not unblock each other.

---

## 3. The six phases

Each phase below carries a status, its logged contributions (with owner, status, first-principles detail, load-bearing evidence, and honest gap), and the phase-level gaps.

### P1 — Productionize scatter/permutation · **LANDED**

**Goal.** The minimal reviewable first slice: DLDSC coordinate metadata on *existing* ops flips a disjoint 1:1 producer/consumer handoff from an HBM restickify to an LX relayout, with a profiled model-shaped speedup.

**Contributions.**

- **Scatter — same-layout disjoint 1:1 LX relayout** · codex · **(MEASURED).** Producer and consumer own different core slices but the physical stick/layout form is already compatible, so Deeptools derives and inserts the LX move from DLDSC tensor-distribution-vs-compute metadata — adding **no new GraphLowering nodes**, only coordinate metadata on existing ops. *Why it matters:* it proves the tensor-vs-compute mismatch contract without taking on fanout/fanin/layout conversion; it is the only class treated as production-ready, on a lean branch of its own. *Evidence:* Granite S512 kernel **14.7258 → 13.8213 ms/iter (1.065×)** (MEASURED); wall **27.6074 → 26.5205 (1.039×, reported flat-to-noise, not claimed as the win)**; a dense-coordinate variant holds the win at **13.8503 ms** (MEASURED). Path: `lx_relayout.py` `_classify_coordinate_topology → ('scatter','one_to_one')`. *Gap:* kept intentionally lean; the artifact branch and dirty deeptools fork must never merge in, and scatter alone does not remove the remaining attention spills.

- **DLDSC coordinate-topology classifier** · codex · **(VALIDATED).** One helper names the logical overlap class (scatter/broadcast/multicast/gather/all-gather) from producer/consumer core-slice maps plus work-slice dims, emitting `communication_class` / `pattern` / `transfer_count` / `max_fanout` / `max_fanin`. This is the contract the whole taxonomy rides on. *Evidence:* `lx_relayout.py:137–192`; `test_lx_relayout_dldsc.py` **15 passing (18 after the clone-source patch)** (VALIDATED). Hard rule: coordinate maps must be **dense over relayout dims** (split-1 dims made explicit as slice 0), or Deeptools aborts with `map::at`. *Gap:* the classifier only *names* topology; it does not yet drive selection/cost — that is claude's Bet-2 half.

- **Resident-LX scatter sizing fix (deeptools)** · codex · **(STRUCTURAL).** Corrects byte sizing of the resident-LX destination so the inserted SDSC allocates the right per-core LX footprint. *Evidence:* `SdscRelayoutInsertion.cpp` / `ddc_fold.cpp`. *Gap:* compile-artifact only; no in-branch device re-measurement.

- **LX-class metadata on SuperDsc (plumbing)** · codex · **(STRUCTURAL).** Threads the class metadata (scatter vs flash-allgather vs matmul-operand-broadcast) through SuperDsc so downstream dxp/ddc passes dispatch by class. *Evidence:* `superdsc.cpp`; `lxRelayoutClassifications_` consumed in `SdscRelayoutInsertion.cpp:71,107`. *Gap:* none structurally — it is substrate.

- **Clone-source + backend incremental numbers** · codex · **(MEASURED).** Each step converts one more attention-activation HBM handoff to `ReStickifyOpLx` plus a realized matmul-operand-broadcast plan. Progression: scatter **13.8213** → clone-source 1 plan **12.1038** → +2 plans **11.9182 ms** (MEASURED). *Evidence:* enabled **11.9182 vs same-branch disabled 12.548 = 1.053× kernel** (MEASURED), wall **30.838 → 30.447 (1.013×, flat-to-noise)**, 2 realized backend plans. *Gap:* baselines differ between run epochs (not stackable); the guardrail forbids wall-time speedup claims.

**Phase gaps.** The scatter branch is kept lean and must not absorb the dirty fork; scatter does not remove the remaining attention spills (the matmul-operand all-gather); clone-source kernel deltas are small and baselines drift per run.

### P2 — Granite spill audit · **IN PROGRESS**

**Goal.** A complete Granite block inventory — before/after SDSC tables classifying *every* non-weight HBM activation spill by comms class, each marked covered/blocked/future — plus a reproduction runbook.

**Contributions.**

- **Residual in-scope class identification** · codex · **(STRUCTURAL).** Of **5** baseline `ReStickifyOpHBM` rows, **4** are weight/prelayout (out of scope) and **1** attention-activation row flips to `ReStickifyOpLx`. This establishes the **attention value matmul-operand all-gather** (buf21: producer out-sharded 32 cores, consumer mb-sharded 32 cores) as *the* remaining non-flash Granite class. *Gap:* it separates in-scope from out-of-scope cleanly, but the full covered/blocked/future table is not the deliverable here.

- **Current-state audit summary (partial)** · codex · **(IN PROGRESS).** The enumeration scoping every later phase: same-core LX persistence handled; scatter converts the HBM restickify → LX; flash's **32** rows are replaceable; **2** realized all-gather activation handoffs; the weight-format remainder excluded. *Gap:* the full classified inventory plus runbook is not yet produced.

**Phase gaps.** The full inventory and runbook are outstanding. The FFN/SwiGLU activation boundary is a *separate* fused-region / pool-boundary residency problem (likely a WSR/streaming concern), **not** the explicit-restickify class already solved — and is not yet addressed.

### P3 — Broadcast & multicast · **IN PROGRESS**

**Goal.** One-to-many LX movement: synthetic broadcast/multicast correctness, allocator safety for replicated destination residency (no clobber of live source ranges), and one real Granite/attention edge.

**Contributions.**

- **DCG input-neighbor fetch extended to KERNEL operands** · codex · **(STRUCTURAL).** Widens input-neighbor fetch so a matmul weight/KERNEL operand — not just an `INPUT` activation — can be fetched from a neighbor core over LX. This is the enabling change letting broadcast/all-gather land its result in a `DsTypes::KERNEL` operand (the attention QKᵀ target). *Evidence:* `inputNeighFetchOp.cpp`, `dcg_frontend.h`, `pcfg_gen.cpp`. *Gap:* it enables the path; the correctness of the KERNEL landing is gated by the staged-conversion contract.

- **Matmul-operand-broadcast planning: synthesizer + insertion hooks** · codex · **(IN PROGRESS).** Synthesizes the grouped all-gather replicating activation/weight shards into a matmul KERNEL operand (the sendnn-proven on-chip all-gather for attention QKᵀ, mul(K)→BMM), then hooks it into relayout insertion. *Evidence:* `synthesizeMatmulOperandBroadcastMovementPlan`; hooks at `SdscRelayoutInsertion.cpp:407,431`. *Gap:* env-gated; full E2E blocked by the DDC fold + `fillDataInfo` path.

- **Broadcast-group derivation from coordinates** · codex · **(VALIDATED).** Parses producer/consumer core→device-slice coordinate maps, intersects common dims, and partitions cores into broadcast groups purely from coordinate metadata. *Evidence:* `deriveExactGroupedCoreMaps` / `parseCoreCoordinateMap`, grouped cardinality, fanout-by-overlap filter. *Gap:* group derivation is unit-validated; physical execution shares the blocked DDC path.

- **Overflow-safe transfer expansion + gated diagnostics** · codex · **(STRUCTURAL).** Hardens group→core-pair transfer expansion against integer overflow, caps it (`kMaxArtifactLogicalTransfers = 1e6`), and records an expansion status/reason so an oversized fanout skips expansion instead of exploding. *Gap:* safety/diagnostic scaffolding, flag-gated — not a shippable default.

- **Staged layout-conversion contract (fail-closed)** · codex · **(BLOCKED).** The key invariant: the LX source-shard layout differs from the final matmul KERNEL-operand layout, so moving bytes core-to-core is *not sufficient* — a local `ReStickifyOpLx` must run *after* the ring gather. The code fails closed on the direct bypass (`dsc2.h StagedLayoutConversionInfo`, DT_CHECK "direct LX-neighbor into final KERNEL operand is value-unsafe"; accepts `gather_then_restickify` + unit tests). *Gap:* contracted and unit-accepted, but not yet executing value-correct end-to-end.

- **DDC fold core-map preservation / `map::at` localization** · codex · **(BLOCKED).** The loop-scoped KERNEL-neighbor operand needs its per-core `coreIdToWkSlice_` maps to survive the DDC fold; the default fold clobbers them, producing a value-wrong offset and a `map::at` throw. **This is the active integration frontier.** *Evidence:* `ddc_fold.cpp` (preserve relayout core maps, seed `loopDistributionParamInfo`, guard corelet-split fold); `ddcv1.cpp` `fillLoopOffsetsAndAddresses` debugPhase. *Gap:* being localized (allocation_lookup → loop_offsets → coordinate_offsets), not yet resolved.

- **Cardinality unit coverage + dense-coordinate rule** · codex · **(VALIDATED).** Device-free proof that the coordinate-metadata contract expresses all four cardinalities (scatter, one-to-many, many-to-one, many-to-many) at the descriptor level. *Evidence:* `DxpTestFixture.CoreWorkDivIncomptLxRelayout*` (2 pass); cardinality bundle probes pass through `dxp_standalone -b ddc`. *Gap:* unit-level only — a generic all-gather descriptor is *not* enough for a matmul RHS KERNEL operand, because the final view/layout binding differs from the source activation layout.

**Phase gaps.** No isolated pure-broadcast/gather Granite spill exists — they appear only *folded* inside the matmul-operand all-gather. Physical execution of the derived groups shares the blocked DDC fold / `fillDataInfo` path. Allocator live-range safety for replicated destinations is not yet demonstrated on a real edge.

### P4 — Gather & all-gather · **IN PROGRESS (the active frontier)**

**Goal.** Many-to-one and many-to-many assembly: gather + all-gather correctness on patterned tensors, a capacity-aware lowering (the LX-explosion guard), and an attention value-side / matmul-operand case with no HBM fallback.

**Contributions.**

- **`matmul_operand_broadcast` frontend contract** · codex · **(STRUCTURAL).** Producer operand sharded on an operand dim, consumer batchmatmul sharded on mb — a naive resident relayout would materialize a full operand per core. Torch names the logical handoff and class; Deeptools derives the loop-scoped movement. **This is the north-star DLDSC contract shape.** *Evidence:* `lx_relayout.py:195–203`; backend plan (buf21): **32 shards, 32 replicas, 1024 logical transfers** (STRUCTURAL). *Gap:* the backend fails closed for this class.

- **Flash restickify 32 HBM → 32 LX transform** · codex · **(STRUCTURAL).** The 32 HBM restickifies in flash attention become 32 on-chip `ReStickifyOpLx` + 32 all-gather layout-restickify plans. *Evidence:* off = **550 SDSC, 32 ReStickifyOpHBM, 0 LX, 0 plans**; on = **0 HBM, 32 LX, 32 plans** (all_gather, transfer_count=256, fanout/fanin=8); compile-probe wall **435s → 243s** (STRUCTURAL). *Gap:* compile-artifact delta only — **not** a speedup, **not** value-verified.

- **Flash layout-allgather: checker + synth + artifact** · codex · **(STRUCTURAL).** Validates a flash layout-allgather contract, synthesizes the 32 HBM → 32 LX all-gather plan, and emits a deterministic backend-plan JSON. *Gap:* plan synthesis + artifact only; no device trace of it executing.

- **LX restickify DDL realization (op-bind + templates)** · codex · **(STRUCTURAL).** Adds the DDL op-bind + templates so a local layout conversion targets LX not HBM, and teaches `finalizeScheduleTree` to ignore DDL-only staging dims. *Evidence:* `restickify.ddl %rst_lx_fp16_op`, `restickify_lx.ddl`, `restickify_lx_sen1p5.ddl`. *Gap:* scaffolding present; value-correct wiring is the blocked frontier.

- **DLDSC collective realization prototype (ProgIR)** · codex · **(IN PROGRESS).** The end-to-end scaffold turning a DLDSC classification into a physically lowered on-chip move spanning dxp → dcg → ProgIR. *Evidence:* `SdscRelayoutInsertion`, `ConstructProgIRHelper`, `inputNeighFetchOp`, `stcdpOp.cpp`, `L3DlOpsScheduler`. *Gap:* prototype-level; DDC fold + `fillDataInfo` not yet value-stable for the KERNEL-operand case.

- **Physical lowering via kernel-neighbor carousel** · codex · **(BLOCKED).** The backend *can* emit L3 ring send/recv PCFG nodes, but the gathered bytes are not in the physical PT KERNEL layout the matmul reads. Direct fused ring writes into the final KERNEL layout are value-wrong; a ring for same-core pieces caused a PCI bus fence. *Evidence:* synthetic M4 **archived pass 0/1024**, current repro **949/1024 mismatch** (MEASURED) at ROWMAP_OUT0 `[0,0,0,0.75]`; `_lx_local → distributeElemArrToTemporalLoops` assert then `fillDataInfo map::at`. *Gap:* needs the true two-stage realization — ring-gather to source-layout staging, local copy for same-core, then a local `ReStickifyOpLx` to the final KERNEL.

- **Staged `gather_then_restickify` matmul-operand strategy** · codex · **(IN PROGRESS).** The production-leaning value-correct alternative to the value-wrong direct KERNEL-neighbor path: separate the all-gather from the local layout conversion into an explicit restickify stage. Stages: `source_operand_shards → grouped_all_gather_replicate → local_layout_conversion → gather_then_restickify → bind_matmul_kernel_operand`. *Gap:* still contract-level; the backend fails closed until the staged path is realized and verified.

- **Clone-source LX eligibility** · codex · **(VALIDATED).** A surgical eligibility fix (`_clone_output_good_for_lx_relayout`) targeting the attention value-side handoff whose `all_gather_replicate` source clone was still HBM-pinned; narrows eligibility to internal computed-activation clones only. *Evidence:* `allocator.py` (active under `SPYRE_LX_PLANNER_RELAYOUT_COLLECTIVES=1`); `test_lx_relayout_dldsc.py` **18 passed** (VALIDATED). *Gap:* hardware verification pending — needs a Granite rerun to confirm Tensor1 flips to LX-backed.

- **LX cardinality + matmul-operand-broadcast unit suite (26 gtests)** · codex · **(VALIDATED).** The reference backstop for the mechanism half: correct group/core-pair expansion; rejection of scatter/resident-replication/incomplete-rename metadata; coordinate-derived grouping; the layout-conversion contract. *Evidence:* `LayoutAllgatherRestickify_unit_test.cpp` (`synthesizesFlash*`, `synthesizesGroupedMatmulOperandBroadcast*`, `AcceptsGatherThenRestickifyRealization`, `RejectsResidentReplication`) — **26 gtests** (VALIDATED). *Gap:* unit-level only; does not prove on-device value correctness of the blocked loop-scoped path.

- **Attention QKᵀ all-gather substrate (multicast 8 K-shards → full K to the Lq cohort)** · claude · **(BLOCKED).** `broadcast_reshard.py` is the structural DUAL of `reduction_reshard` — the *same class* as codex's `matmul_operand_broadcast` in the same DCG region (his AV value-operand vs claude's QKᵀ K-operand). P0 passed: the patched dxp accepts the multicast `STCDPOpLx`, builds the exact cohort replicate map, and leaves EBR inert. *Evidence:* branch `swiglu-ws-v2`, flag `SPYRE_ONCHIP_ATTN_ALLGATHER`; on latest deeptools, flag-OFF SDPA matches CPU eager at **1e-3** (was **0.02** on the old base) (MEASURED); flag-ON still mis-gathers at **~0.02** (MEASURED). *Gap — root cause:* mul(K) is a *separate device program*, so the consumer bundle re-reads K from HBM via an in-bundle `ReStickify` sharded `x:32`, not the gather's assumed `out/Lk-band:32`. As architected the substrate cannot be value-correct; the real fix is co-bundling mul(K) into the QKᵀ program — a redesign, not a patch.

- **Measured grounding — realized broadcast banks its win via HBM-round-trip elimination, not ring efficiency** · claude · **(MEASURED).** The cost model attributes the entire win to `dram_bytes_saved_vs_spill` (HBM write+read elimination), a resource distinct from ring-transport time. This establishes codex's `matmul_operand_broadcast` == the attention all-gather the seam prices. *Evidence:* **12.55 → 11.92 = 1.053×** (MEASURED); the realized move is **256 transfers, max link occupancy 16/link** (32 same-core free + 224 cross-core) — *worse* than the 36 GB/s scatter floor; `comm_cost.py:443–457` `dram_saved = 2×operand`. *Gap:* the mechanism half is BLOCKED — value-correct XOR capacity-safe (**232 ALLCLOSE-False / 26 True** on CDX M4, MEASURED).

**Phase gaps.** All-gather → KERNEL operand is blocked upstream: value-correct and capacity-safe are currently mutually exclusive. Flash all-gather value-correctness is not a clean signal — the flash baseline is independently value-wrong (a zero-stride broadcast bug, 31.5% mismatch). The capacity-aware two-stage loop-scoped lowering (the LX-explosion guard) is required, not optional, because full-resident gather does not fit Granite attention in LX. The single most-leveraged shared task is to generalize `runDcgForInputFetchNeighbor` off `DsTypes::INPUT`-only for a replicated KERNEL operand under loop-scoped fetch (needs mul(K) co-bundling).

### P5 — Reduce & all-reduce · **IN PROGRESS (the arithmetic-collective lane)**

**Goal.** Arithmetic collectives: a reduce primitive/lowering, split-K correctness, gather-vs-reduce separation, reduce-before-relayout ordering, and all-reduce *only after* reduce and broadcast are independently correct. Home of the LSE ring-fold.

**Contributions.**

- **Reduce/all-reduce lane parked as out of copy scope** · codex · **(DESIGN).** codex's copy-based DLDSC relayout deliberately stops at byte-movement classes; reduce/all-reduce combine values arithmetically and need a reduction axis/op/dtype/identity, not a coordinate copy. This is the lane codex parks and claude owns. *Gap:* no frontend contract, no backend, no Granite evidence on codex's side — entirely the claude convergence lane.

- **LSE ring-fold value oracle** · claude · **(VALIDATED).** Merges two flash partials `(m, l, A)` over *disjoint* key sets: `m = max`; `aᵢ = exp(mᵢ − m)`; `A = Σ aᵢAᵢ`; `l = Σ aᵢlᵢ`; `O = A/l` once at fold end — with fp32 carry and a static ring order for bit-determinism. The folded payload is the **L-independent** `(m,l,A)` triple, so the reduce is F-dominated regardless of key-tile count. *Evidence:* `lse_fold_ref.py:158–223`; `test_lse_fold.py` **16 tests pass** (VALIDATED) — associativity, commutativity, bit-determinism, equivalence to single-pass softmax, tree==linear, L-independent payload. *Gap:* device-free/unit-validated only; the frontend owns the operator, the backend owns realization.

- **Bet 3 LSE ring-fold merge (neighbor reduce-scatter)** · claude · **(VALIDATED).** A genuinely novel collective: merge Lk-shard softmax partials with a neighbor reduce-scatter (1 transfer/link, ~130 band) instead of a linear fold-to-one that piles O on one link at the ~36 floor. It lands the output **head-split** — exactly the layout the O-projection wants — so the merge is relayout-free. *Evidence:* branch `ah/flash-ring`; `carousel/fold.py` + `reference.py` **13/13 device-free checks pass** (VALIDATED), including fold == single-pass softmax at rtol **1e-4**. *Gap:* a MODELED collective — not compiler-wired, not device-measured.

- **Reduce-lane cost wiring (LSE fold priced F-dominated)** · claude · **(VALIDATED).** Selects `reduce_tree_fold` (⌈log₂P⌉ hops) over `reduce_plain_ring` (P−1 hops); **≥98%** of plain-ring cost is fixed F, not payload. The reduce is *not* tiled by LX_CAP (payload tiny / L-independent). This corrects an earlier assumption that the fold rides the 130 GB/s uniform shift. *Evidence:* `comm_cost.py:335–347`; `test_reduce_prices_tree_fold_at_P32` = **37.10 µs tree vs 230.04 µs plain-ring** (VALIDATED, ~6.2×). Here **P is the split factor along Lk, not a full 32-core ring**. *Gap:* realization gated behind the reduce-lane flag + the missing SFP primitive.

- **`lx_relayout` reduce-lane reroute (gated)** · claude · **(IN PROGRESS).** When gated ON (`SPYRE_LX_PLANNER_RELAYOUT_REDUCE`) and the class is reduce-like, realizes the edge as an `lse_ring_fold` plan priced as a reduction CommEdge — closing the loop from the existing reduce classifier to the tree-fold cost without touching the value path. *Evidence:* `lx_relayout.py:612–692`, `271–291`; `config.py:79–81`. *Gap:* **default OFF**; a pricing/routing wire — the backend reduction-collective lowering is absent.

- **`make_lse_ring_fold_contract` (DLDSC reduce contract builder)** · claude · **(DESIGN).** Encodes the **layout dividend**: reduce-over-Lk-within-head + scatter-result-over-heads lands the output head-split == the k_fast out-projection input layout ⇒ **zero relayout** at attention→out-proj. Carries class, LSE-combine operator identity, `(m,l,A)` payload, participant_count, tree_fold_hops. *Evidence:* `layout_allgather_restickify.py:79–170`; contract tests. *Gap:* authored but not emitted into a real compile path nor consumed by a backend lowering — design-only.

- **Bet 1 flash-in-a-loop (online-softmax as coarse-tiled Lk reduction)** · claude · **(BLOCKED).** An off-stick transpose proves the mechanism is real — scores never materialize — but the resident-scratch guard is inert, blocked upstream of `coarse_tile`. A **byte-identical base-vs-fix stamp ablation** proved the "one-step fix" does not fire. *Evidence:* branch `ah/flash-ring`; `scores.transpose(-1,-2)` collapses `[1,32,4096,4096] → [1,4,1024,4096]`; **146 tests green** but ablation-inert; committed nothing. *Gap — real blockers:* `optimize_restickify_locations` cannot gather the sparse `amax(dim=-1)`; the reachable e2e test never surfaces Lk as a tiling level; an in-code limit disallows Lk coarse tiling. Reframed as a reduce-lane problem — codex's flash restickify-on does not help (refuted).

**Phase gaps.** The backend SFP `lse_combine` device primitive that must value-match the oracle is **not lowered** — so there is no device value-correctness. Bet 1 needs Stage-1 reduction-dim tiling (design doc on main) to lift the Lk coarse-tile limit, plus the sparse amax gather and an Lk hint. Bet 3 depends on cross-CORE Lk work-division (a spatial split) + Bet-2 selectability + codex's `fillDataInfo`/coordinate fix. All-reduce is explicitly deferred until reduce and broadcast are independently correct.

### P6 — Costing & scheduling · **IN PROGRESS (the selectability half)**

**Goal.** Make the planner relayout-*aware*, not merely relayout-*capable*: a comms cost model by class, work-division that accounts for relayout cost (reshard-aware selection), and schedule/overlap experiments judged by kernel-time from archived traces (never wall time).

**Contributions.**

- **Standalone communication cost model (`comm_cost.py`)** · claude · **(VALIDATED).** Prices the on-chip LX-ring edge as a resource *separate* from the matmul OP model. On-chip moves are F-dominated — **F = 7.3 µs per STCDP execute** (MEASURED) — not bandwidth-bound; `t_step = F + max_link_bytes / RAW_RATE(140e9)` on the busiest link. It is the successor to the old Bet-2 `_cohort_penalty` multiply-into-`hbm_us` hack, returning a CommCost resource vector (time_us, executes, schedule, max_link_occupancy, effective_bw_bps, lx_highwater, n_tiles, dram_bytes_saved). *Evidence:* `comm_cost.py:15–467`; `test_comm_cost.py` **20 tests pass (36/36 with lse_fold)** (VALIDATED); pure-Python, zero backend dep. *Gap:* none for the standalone model; its planner consumption is flag-gated.

- **Per-link contention pricing (`band()` + `link_occupancy()`)** · claude · **(VALIDATED).** Maps peak per-link transfer *count* to effective aggregate bandwidth: occ ≤ 1 → 140 GB/s uniform shift; occ 2..9 → 36 GB/s plateau; occ > 9 → 36×(9/occ). It routes each transfer on the shortest ring arc, treats same-core as free, and models disjoint groups as independent rings. *Evidence:* `comm_cost.py:84–105`, `209–264`; anchor (4 groups of 8) = 32 same-core free + 224 cross-core, **max occ 16/link**; grounds the MEASURED 54/90/130 vs 36 GB/s ladder (R²≥0.998). *Gap:* the realized attention broadcast lands occ = 16/link → slower than the 36 GB/s floor, so the model attributes its win to HBM-round-trip elimination, not ring efficiency.

- **LX-capacity interception theorem** · claude · **(VALIDATED/MODELED).** Because the ring-carousel-vs-naive crossover (**~5.20 MiB/head**, MODELED) sits *above* the ~2 MiB/core LX cap, `cost()` force-tiles any operand > LX_CAP into ⌈bytes/LX_CAP⌉ tiles, each priced below the crossover — so the bandwidth-optimal ring *never* wins for an LX-resident broadcast. This is why the two carousel RFCs are shelved. *Evidence:* `comm_cost.py:386–458` + `crossover_mib_per_head()`; a 16 MiB operand → 8 tiles, carousel never argmin. *Gap:* consistent with the MEASURED 1.053× being pure HBM-round-trip elimination.

- **Ring-speed physics (device-measured bandwidth ladder)** · claude · **(MEASURED).** Per-link transfer count — not burst size, not a hidden HW cap — sets effective ring bandwidth. *Evidence:* additive-differential slope (R²≥0.998); scatter **34–36 GB/s**; uniform shift **54@4.06MB / 90.5@8.13MB / 130.3@16.25MB**; streaming asymptote **~244–254**. *Gap:* the fold/all-gather bandwidth arithmetic derived from it remains MODELED until device-confirmed at each bet's first-step gate.

- **Finding 2 — burst-is-not-the-lever** · claude · **(MEASURED).** Enlarging per-descriptor burst 50–100× at fixed bytes gave **ρ_contig/ρ_strided = 0.958 median / 1.004 min** (vs a predicted 2.9×). The model predicts raw 128 for every pattern; only per-link contention pulls it down. *Evidence:* `perfmodel.cpp` assigns raw `ringBw=128` to every link; `BurstEfficiency.def` is consumed only by `L3DlOpsScheduler` as a chunk-selection objective. *Gap:* directly motivates a pattern-aware ring cost term (~36 all-to-all, ~130+ uniform, +~7 µs fixed per STCDP execute).

- **Bet 2 per-link contention cost term** · claude · **(VALIDATED).** Replaced the lumped `cohort_penalty = max(1, max(m,n)/8)` with a structural identical-operand gate: LHS multicast over the n-cohort, RHS over the m-cohort (both capped at the ~130 band), output distinct at peak/36. It prices the transport topology the flat bytes/128 model ignores, so a ring-aware wide-cohort plan becomes selectable. *Evidence:* branch `ah/ring-cost-term`; **7/7 new unit + 153 inductor tests pass**, ruff clean; decisive test — wide-N `(m2,n16)` now costs *less* than `(m8,n4)`, the mispricing inversion is gone. *Gap:* no-regression is STRUCTURAL not empirical (`_cohort_penalty` returns 1.0 for cohort ≤ 8); not device-measured.

- **Flag-gated planner seam (`SPYRE_COMM_COST_SEAM`)** · claude · **(IN PROGRESS).** When ON, `hbm_us` stays a pure HBM term and the split search *adds* the broadcast edge cost as a separate additive term (op_cost + edge_cost), never multiplied into `hbm_us`. It prices RHS[K,N] all-gather to the M-split cohort and LHS[M,K] to the N-split cohort as ALL_GATHER CommEdges — the reshard-aware selectability half. *Evidence:* `work_division.py:969` env gate, `1021–1030` pure-hbm branch, `1123–1165` `_matmul_broadcast_edge_cost_us`, `1307–1315` additive call; flag OFF by default, zero-change no-op. *Gap:* default OFF and not device-validated to reproduce the device-verified Granite selections — pending a device gate.

- **`comm_edge_from_plan` adapter** · claude · **(STRUCTURAL).** Builds a CommEdge from a realized `LXRelayoutPlan` so plan-time and realize-time costs use the *same* model, enforcing a one-directional dependency (`work_division` imports `comm_cost`; the adapter lives in `lx_relayout` so `comm_cost` keeps zero backend dep). *Evidence:* `lx_relayout.py:294–331`. *Gap:* group derivation is *approximate* (group_size/count from producer-slice cardinality); a TODO remains to derive exact `(group_count, replicas, producer_chunks)` from the fan-in/out overlap map.

- **Ring-aware mechanism catalog + carousel ROI dispositions** · claude · **(DESIGN).** The organizing analysis separating residency (a trivial floor) from the ring-aware algorithm layer, gating every mechanism on "disqualified if it collapses to just pin it." It killed 6 upstream with reasons and concludes: fund ONE vehicle — the cross-bundle co-bundling redesign. *Evidence:* the mechanism catalog (11 mechanisms, 5 bets, 6 killed) and the carousel ROI plan (10 premises); the weight carousel is shelved (util **29.8% → 72.7%** with zero transport), the KV carousel reshaped. *Gap:* the remaining bets are structurally sound but speculative-on-payoff and need deeptools.

- **Granite S512 e2e blocker isolation (batchmatmul does not fit in LX)** · codex · **(BLOCKED).** The `matmul_operand_broadcast` plan generates, but DXP cannot fit the batchmatmul in LX (a capacity/schedule realization limit, not a metadata one). This confirms full-resident gather is too large for Granite attention and the loop-scoped staging is required. *Evidence:* `8_batchmatmul` plan (all_gather_replicate, logical_transfer_count=**512**, group_count=**2**) then `DtException` "initial chunk must fit in LX" at `L3DlOpsScheduler.cpp:1701`, **rc=134** (STRUCTURAL). *Gap:* the scheduling frontier the P6 reshard-aware cost term is meant to steer — but that cost/selection term is not present in codex's branches.

**Phase gaps.** The seam and Bet-2 term are validated at unit level but not device-validated to reproduce the device-verified Granite selections (all AIU devices were degraded at authoring time). Productionizing Bet 2 onto codex's lowering needs the edge cost to carry the true coordinate *stride* (not a cohort count) plus a same-core-is-free refinement. Schedule/overlap experiments have not started, and the kernel-time-from-archived-traces guardrail must hold. The cross-bundle co-bundling redesign is high-cost deeptools+inductor SDSC wiring, blocked by the cross-bundle wall (LX is not persistent across programs).

---

## 4. The 7-class taxonomy coverage

The organizing spine of the epic. Each class is a distinct movement pattern.

1. **Scatter / permutation** (N→N 1:1, pure relabel) — **DONE** · codex. The productionized, MEASURED **~1.065× kernel** on Granite S512 — the proven contract direction. Classifier + sizing fix + SuperDsc plumbing landed. All other classes build on it.

2. **Broadcast** (1→all, full-chip) — **PROTOTYPE (~55%)** · codex. Backend synthesizer + coordinate-driven group derivation (validated unit) + overflow-safe expansion exist, but only *folded* inside the matmul-operand all-gather; no isolated pure-broadcast spill; physical execution blocked on staged-conversion + DDC fold.

3. **Multicast** (1→subset, cohort) — **PROTOTYPE (~55%)** · shared. The live ring primitive. Cohort/group derivation validated device-free; claude's cost model prices multicast-vs-scatter per-link contention (validated). Planner cohort representation partial; physical execution shares the blocked DDC path; a cost-gated over-replication guard is not built.

4. **Gather** (many→1 copy, no arithmetic) — **PARTIAL (~38%)** · codex. The gather-vs-reduce split exists in the classifier; gather appears only folded inside the matmul-operand all-gather. No pure-gather Granite spill and no standalone assembly-correctness suite yet.

5. **All-gather** (N→N, every consumer holds all pieces) — **PARTIAL (~45%)** · shared. Flash restickify 32 HBM → 32 LX is STRUCTURAL; the `matmul_operand_broadcast` frontend contract + backend synth exist but are BLOCKED (value-correct XOR capacity-safe); clone-source eligibility validated; incremental **1.053×** MEASURED on CLC. The capacity-aware two-stage lowering is the required unbuilt piece. claude's QKᵀ all-gather is the same class, blocked on cross-bundle source provenance.

6. **Reduce** (many→1 *with* arithmetic) — **PROTOTYPE (~50%)** · claude. codex parks this lane; claude owns it. The LSE ring-fold oracle + tree-fold cost are VALIDATED device-free (**16 + 13 tests**); the frontend reroute + DLDSC contract are wired (in-progress/design). The backend SFP `lse_combine` primitive is NOT lowered — no device value-correctness.

7. **All-reduce** (many→many reduce + replicate) — **NOT STARTED (~8%)** · claude. Explicitly sequenced last, gated behind reduce AND broadcast landing independently. The contract builder can emit an `all_reduce` class, but no acceptance criteria are enumerated, no backend exists, and there is no Granite evidence.

---

## 5. Results panel — every concrete number, tagged

| Result | Value | Kind | Source |
|---|---|---|---|
| Scatter, Granite S512 kernel | 14.7258 → 13.8213 ms (1.065×); dense-coord 13.8503 preserves | **MEASURED** | current_state (LX frac 0.2) |
| Scatter, Granite S512 wall | 27.6074 → 26.5205 (1.039×, flat-to-noise) | **MEASURED** | current_state |
| Clone-source + backend, kernel, 2 plans | 12.548 → 11.9182 ms (1.053× vs same-branch disabled); wall 30.838 → 30.447 | **MEASURED** | backend2162 status |
| Clone-source, 1 plan kernel | 12.1038 ms | **MEASURED** | backend2162 status |
| Flash relayout SDSC delta | 32 HBM → 32 LX (550 SDSC, 0→32 plans); compile-probe wall 435s → 243s | **STRUCTURAL** | remaining_hbm_spill_classes |
| Flash value-correct run (patch unset) | 31.5% mismatch (5285717/16777216) — partly a pre-existing zero-stride baseline bug | **MEASURED** | value_correctness_boundary |
| Synthetic M4 kernel-neighbor carousel | 949/1024 mismatch (BLOCKED; archived pass was 0/1024) | **MEASURED** | comms handoff §5–6 |
| CDX M4 operand-broadcast ALLCLOSE | 232 False / 26 True (BLOCKED: value-correct XOR capacity-safe) | **MEASURED** | orchestration plan |
| Granite e2e all-gather blocker | rc=134 (transfer_count=512, group_count=2; "chunk must fit in LX" L3DlOpsScheduler.cpp:1701) | **STRUCTURAL** | relayout_clc |
| Ring physics — scatter | 34–36 GB/s (524 descriptors, 4–9 xfers/link); R²≥0.998 | **MEASURED** | ring_speed |
| Ring physics — uniform shift @4/8/16 MB | 54 / 90 / 130 GB/s (1 xfer/link); streaming asymptote ~244–254 | **MEASURED** | ring_speed |
| Burst-is-not-the-lever null | 0.958× median ρ_contig/ρ_strided (vs predicted 2.9×) — zero improvement | **MEASURED** | ring_speed Finding 2 |
| Realized attention broadcast link occupancy | 16/link (256 transfers, 32 same-core free + 224 cross-core) — slower than 36 GB/s floor | **MEASURED** | orchestration plan §9b |
| Reduce tree-fold vs plain-ring @P=32 | 37.1 vs 230.0 µs (~6.2×); ≥98% of plain-ring cost is fixed F | **VALIDATED** | comm_cost + lse_fold C6 |
| Ring-carousel crossover | ~5.20 MiB/head (above ~2 MiB/core LX cap; 16 MiB → 8 tiles, carousel never argmin) | **MODELED** | comm_cost:386–458, C2/C3 |
| F fixed cost per STCDP execute | 7.3 µs (F in t_step = F + max_link_bytes/RAW_RATE(140e9)) | **MEASURED** | comm_cost:15–467 |
| LSE fold reference | 13/13 + 16/16 pass; fold == single-pass softmax at rtol 1e-4 (±80-logit) | **VALIDATED** | ah/flash-ring; test_lse_fold |
| comm_cost model tests | 20/20 (36/36 with lse_fold); pure-Python, zero backend dep | **VALIDATED** | test_comm_cost |
| Bet 2 ring-cost-term tests | 7/7 unit + 153 inductor; wide-N (m2,n16) < (m8,n4) | **VALIDATED** | ah/ring-cost-term |
| Deeptools cardinality / matmul-operand unit suite | 26 gtests | **VALIDATED** | dt-comms |
| Torch DLDSC classifier unit | 15 (18 after clone-source patch) | **VALIDATED** | test_lx_relayout_dldsc |
| Attn QKᵀ all-gather substrate | ~0.02 mis-gather (BLOCKED: flag-OFF matches CPU at 1e-3; flag-ON mis-gathers) | **MEASURED** | swiglu-ws-v2 |
| W2 array-fill layout fix, PT-Util | 29.8% → 72.7% with ZERO transport (refutes the under-roofline carousel premise) | **MEASURED** | carousel ROI plan |

---

## 6. Remaining work and the shared blocker

The top gaps, in dependency order.

### ⚠ The shared upstream blocker: all-gather → `DsTypes::KERNEL` operand

One deeptools DCG plumbing problem gates the *entire* attention matmul-operand comms class — the last remaining non-flash Granite spill — and blocks both halves at once. The all-gather is **value-correct XOR capacity-safe**: the direct KERNEL-neighbor path is value-wrong (949/1024, fail-closed asserted); the staged `gather_then_restickify` path is contract-level only; and full-resident gather does not fit Granite attention in LX. The single most-leveraged shared task is to **generalize `runDcgForInputFetchNeighbor` off `DsTypes::INPUT`-only** and populate/consume coordinate maps for a replicated KERNEL operand under loop-scoped fetch — with the descriptor carrying the true coordinate **stride**, not a cohort count (codex's flash value-bug root was count-based grouping). claude's prerequisite: **mul(K) must be co-bundled into the QKᵀ program** so the consumer does not re-read K from HBM.

**G1 — All-gather → KERNEL operand is value-correct XOR capacity-safe.** *Blocks:* P4 completion, P3 physical execution, and any device-validated speedup for the entire attention matmul-operand class. *Next:* **codex** realizes the two-stage loop-scoped `gather_then_restickify` (ring-gather to source-layout staging + local `ReStickifyOpLx` to final KERNEL) and lands it value-correct end-to-end.

**G2 — DDC fold propagation clobbers per-core `coreIdToWkSlice_` coordinate maps.** *Blocks:* the active integration frontier — both the Granite `matmul_operand_broadcast` and the synthetic M4 carousel throw `map::at` / produce wrong offsets in `fillLoopOffsetsAndAddresses`. *Next:* **codex** resolves the localized (allocation_lookup → loop_offsets → coordinate_offsets) fold-preservation crash under debugPhase instrumentation.

**G3 — Generalize `runDcgForInputFetchNeighbor` for a replicated KERNEL operand (true coordinate stride).** *Blocks:* the single most-leveraged shared task — likely clears *both* codex's and claude's passing-but-unreproducible runs, and is the productionization anchor for claude's Bet-2 edge cost. *Next:* **shared** — codex's mechanism plus claude's co-bundling prerequisite (mul(K) into the QKᵀ program).

**G4 — Planner seam + Bet-2 term not device-validated to reproduce Granite selections.** *Blocks:* P6 relayout-aware planning shipping ON by default — closing the loop so the planner *picks* the on-chip move on cost grounds. *Next:* **claude** runs the KV-proj [4096,1024] wide-N + banked shared-weight device sweep on a stable pod once devices recover.

**G5 — Backend SFP `lse_combine` primitive not lowered.** *Blocks:* P5 reduce/all-reduce (the arithmetic-collective lane) and Bet 3 device measurement. *Next:* **shared** — deeptools lowers the SFP reduce primitive (written as a deeptools ask); claude's oracle is the value reference; also needs cross-CORE Lk work-division (a spatial split).

**G6 — Bet-1 flash-in-a-loop blocked upstream of `coarse_tile`.** *Blocks:* killing the 32 MB/head score spill and making the LSE fold reachable in a real compile (the temporal prerequisite for Bet 3). *Next:* **claude/upstream** — the in-progress Stage-1 reduction-dim-tiling work (design doc on main) must lift the Lk coarse-tile limit; then surface an Lk hint and handle the sparse amax gather.

**G7 — Granite spill audit only partial.** *Blocks:* P2's definition-of-done (the full class-by-class covered/blocked/future inventory) and the scoping of every later phase; the FFN/SwiGLU fused-region residency boundary is a separate WSR/streaming problem. *Next:* **codex** produces the complete before/after SDSC classification table + reproduction runbook.

---

## 7. Honesty ledger

Every claim above is tagged by evidence kind. No speedup is claimed from wall time alone (an epic guardrail); wall time is reported flat-to-noise, and only kernel-time from archived traces counts.

- **MEASURED (device/trace).** The scatter **1.065×** and the +backend **1.053×** kernel deltas; the ring bandwidth ladder (**36 vs 130 GB/s**); the **F = 7.3 µs** execute cost; the burst-is-not-the-lever null (**0.958×**); and the BLOCKED value-mismatch counts (**949/1024, 232/26, 31.5%, ~0.02**).

- **STRUCTURAL (compile-artifact delta, value not verified).** Flash **32 HBM → 32 LX**; the sizing/plumbing/DDL scaffolding; the frontend contracts and backend-plan artifacts. A representable move, not a proven-correct one.

- **VALIDATED (device-free reference/unit).** The DLDSC classifier (**15/18**), the deeptools suite (**26 gtests**), the comm_cost model (**20/36**), the LSE fold oracle (**13/13 + 16/16**), the tree-fold pricing (**37.1 vs 230.0 µs**), and the Bet-2 term (**7 + 153**). Correct by construction, not yet on-device.

- **MODELED / BLOCKED (analytic or upstream-blocked).** The **~5.2 MiB/head** crossover, the F-dominated reduce, and the fold bandwidth arithmetic are analytic. The KERNEL-operand all-gather, the DDC fold crash, the LX-capacity e2e blocker (**rc=134**), Bet-1's inert guard, and the missing SFP primitive are blocked. The mechanism half is blocked upstream; the cost model prices the move correctly but cannot yet be device-validated end-to-end.

**Two hoped links, honestly refuted.** codex's flash-restickify-on does **not** unblock claude's Bet-1 flash-in-a-loop (a dense layout-form change versus a sparse amax reduction gather), and flash value-correctness is not a clean signal for either lane (an independent zero-stride baseline bug drives its 31.5% mismatch). The one banked win — the realized attention broadcast — runs at **occ = 16/link** (worse than the 36 GB/s floor), so its **1.053×** is attributed entirely to **HBM-round-trip elimination, not ring transport efficiency**.

One ring substrate, two halves, converging at the DLDSC coordinate-metadata contract.
