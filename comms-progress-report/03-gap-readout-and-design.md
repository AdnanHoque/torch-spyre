# Comms-Collectives: Gap Readout + Implementation Design

## Where the effort stands

Production covers **(1:1, no-form-change) resident scatter**: producer/consumer core-ownership mismatch on an LX-resident non-weight tensor is recorded as DLDSC coordinate metadata; the backend's `insertRelayoutSdsc` synthesizes `STCDPOpLx` (verified live, `SdscRelayoutInsertion.cpp:220`). Measured ~14.70 → ~12.34 ms/iter (~1.19×). Everything below is a **spill class that scatter does not cover**. Two prototype branches (stcdp-agent = explicit collective planning; dldsc-agent = restickify-lx) attack the two adjacent corners but are **classified-only / not measured**. Priority order for a buildable-next design is (1) restickify, (2) all-gather, then (3) gather, (4) reduce (state-only).

---

## (1) layout-restickify-activation — *buildable next*

**(a) Gap.** One computed-activation edge in Granite still emits `ReStickifyOpHBM` and spills: softmax `mul_6`/scores → `@V` bmm (`buf46→buf14`), where `Lk` flips from an output/stick dim in QK^T to the contraction dim of `@V`, so the physical stick form changes. Same class recurs in flash (`exp_scores → exp_scores@V`, per-Lq-tile under WSR). The 4 remaining `ReStickifyOpHBM` rows are weight restickifies — out of scope (offline prelayout owns them).

**(b) Communication.** Cardinality many:many; **form change = yes** (axis-2). This is the pure layout-transform lane.

**(c) Design.**
- *Frontend (torch-spyre fork).* Scatter's `LXRelayoutPlan` carries only ownership coords (`producer_core_id_to_device_slice`, `producer/consumer_work_slice_dims`, `read_index`, `kind`); it has **no pre/post stick-layout field**, so it cannot express a form change. The dldsc-agent prototype dodges this by leaving the plan `realized=False` and doing a **string swap** in `superdsc.py` (`ReStickifyOpHBM`→`ReStickifyOpLx`) gated on env flag + `ComputedBuffer` source + all SDSC args LX-resident. That is rename-and-hope: correct only because the restickify SDSC's own operand descriptors already carry the source/dest layout implicitly. The real fix is a **DLDSC interface expansion**: add `source_layout` / `dest_layout` (stick dim order + stick size) and `operand_identity` fields to `LXRelayoutPlan`, populated for the `layout_restickify_activation` class, and flip it to `realized=True` so it emits coordinate metadata instead of relying on the opfunc rename.
- *Backend (deeptools fork — mostly NOT needed).* The op is **LIVE**: `ReStickifyOpLx` is a first-class `OpFuncs` enum (`dscdefn.h:270`), SDSC-ingested (`dataOpDsc.cpp:446`), DDC-lowered, pcfg-gen'd, perf-modeled. Critically the DSM **already self-promotes** `ReStickifyOpHBM → ReStickifyOpLx` (`lxopt.cpp:3798-3801`) gated only on `canExecuteOpFunc(ReStickifyOpLx, seid)` + both operands LX-opted — verified live. So the Torch swap is **front-running a promotion the backend attempts anyway**. What is genuinely absent backend-side: no DXP pass *synthesizes* a layout-restickify from a coordinate mismatch — `insertRelayoutSdsc` only emits `STCDPOpLx`, which assumes same-layout input/output LDS.

**Buildable-next recommendation.** The **cheapest load-bearing experiment is an A/B that costs almost no code**: run the Granite block with the dldsc opfunc-swap ON vs OFF and check whether the DSM's own `lxopt.cpp:3798` promotion already fires on this exact computed-activation restickify. If it does, the Torch swap is **redundant** and the whole lane reduces to "confirm the backend already handles it" — the real work becomes the interface-expansion only if you want Torch to *cost/plan* it. If the promotion does **not** fire (operands not both LX-opted at that point), the Torch swap is load-bearing and should be promoted from string-substitution to the `realized=True` metadata path above. **This A/B is the single missing measurement and gates everything else in this lane.** Land: frontend fields + realize in torch-spyre fork; no new deeptools op required.

---

## (2) all-gather / broadcast operand movement — *buildable next, backend wiring required*

**(a) Gap.** The `@V` bmm non-primary operand (`buf21→buf22`, `read_index not in (0,None)`): producer shards `values[B,H,Lk,D]` across cores, but each consumer core co-splitting the bmm needs the **full Lk slice replicated**. Classifier tags it `matmul_operand_broadcast` / `all_gather_replicate`. Flash multiplies this by the `{H:4,Lq:8,Lk:8}` co-split cardinality. The one `realized=True` probe **fails**: `Unexpected corelet cardinality mismatch … allocate-Tensor1_lx`; forcing the resident path materializes ~4 MiB/core → LX-capacity fail or wrong HBM fallback.

**(b) Communication.** Cardinality 1:many (broadcast-replicate) → many:many (all-gather); **form change = no** (pure ownership + replication).

**(c) Design.**
- *Frontend (torch-spyre fork).* The DLDSC coordinate contract is **the right interface and stays coordinate-metadata** — the wall is backend, not "Torch must emit a physical plan." The one addition needed: carry `read_index` (operand ordinal) and the operand's **ds-type** from classification into lowering, so the backend can fetch the *classified* operand rather than a hard-coded input.
- *Backend (deeptools fork — REQUIRED, this is the real gap).* Two sub-findings from the caps study, both verified:
  1. Multicast replication itself is **LIVE** — `STCDPOpLx` lowering builds `prodConsList` share-groups with `multicastDegree` / mode-3 promotion (`stcdpOp.cpp:3193`, `transfer_compute.cpp:92,98`), reachable from the same path as scatter. So 1:many replication is not the blocker.
  2. The correct loop-scoped operand entrypoint, `runDcgForInputFetchNeighbor → generatePcfgIRForDataOpInpFetch → fillDataDSCForInputFetchNeighbor`, is **present-but-dead for this lane**: its only caller is `dcg/tools/dcg_inpfetch_standalone.cpp:89` (verified — no DXP/DSM/DDC caller), and `inputNeighFetchOp.cpp:16,73` hard-`DT_CHECK`s `DsTypes::INPUT`, whereas the Granite `@V` operand is `DsTypes::KERNEL`. **This is the confirmed blocking gap.** `OpFuncs::AllGather` exists but is **multi-AIU / cross-chip** (`dsm/coll/*`, `multiAIUOptimizer`, `comm_size`/`Rank_t`) — **not reachable** as a single-AIU on-chip primitive; do not wire to it.

**Buildable-next recommendation.** The fix is **DXP pipeline wiring + a `DsTypes` generalization**, not a new ring primitive: (i) wire `runDcgForInputFetchNeighbor` into the compile path so it's reachable from an SDSC (currently only the standalone tool reaches it); (ii) generalize `inputNeighFetchOp.cpp` off the hard-coded `DsTypes::INPUT` to the operand ds-type selected by the classifier's `read_index`. Prefer routing replication through the already-live `STCDPOpLx` multicast (`prodConsList`) rather than materializing a resident per-core view (which is what fails today). Land: `read_index`/ds-type plumb in torch-spyre fork; **InputFetchNeighbor generalization + DXP wiring in deeptools fork.**

---

## (3) gather (many:1) — *classified, not prioritized*

**(a) Gap.** The inverse of broadcast — many producer shards consolidated to one consumer owner. Appears as the mirror of the `@V` operand movement and in any op whose consumer div collapses a dim the producer split. Not independently surfaced as a distinct failing Granite edge in the study (subsumed by broadcast + reduce).

**(b) Communication.** Cardinality many:1; form change = no.

**(c) Design.** *Frontend:* same coordinate-metadata contract, `communication_pattern="gather"`. *Backend:* `GatherOpHBM` is **live but HBM-routed** (`dscdefn.cpp:411`, `dataOpDsc.cpp`) — wrong lane, it spills. The on-chip many:1 consolidation would ride the **same `STCDPOpLx` path in reverse** (share-group with one destination). No dedicated on-chip gather op exists. **Recommendation:** do not build standalone; falls out of the InputFetchNeighbor generalization in (2). Land: deferred.

---

## (4) reduce / all-reduce — *STATE, do not design*

**(a) Gap.** Softmax `amax(Lk)`/`sum(Lk)` collapse Lk: Lk-shard owners are producers, the reduced `running_max`/`denominator` lives on a different owner set → many:1 gather **+ arithmetic**. Flash concentrates this per-tile.

**(b) Communication.** many:1 + reduce.

**(c) Design — none.** Per the caps study (verified): `OpFuncs::SUM/MAX/MEAN` are **in-array / PSUM-ring DL compute** ops (`dlOps.cpp:55-71`, `forceZeroInit`+`psumRing`), **not a movement collective** — there is no single-AIU "move-then-reduce across cores" LX relayout primitive. `AllReduce` (`dscdefn.cpp:337`) is **multi-AIU / spyreccl inter-device only** (`dsm/coll/comm.cpp`, world-size), out of lane. So the softmax reduction is a **compute-path concern (PSUM ring), not a relayout spill** — classify it, hand it to the reduce compute path, and do **not** attempt a relayout design. State-only, consistent with the out-of-scope framing.

---

## Summary table

| Class | Spill edge | Cardinality / form | Frontend (torch-spyre fork) | Backend (deeptools fork) | Live? |
|---|---|---|---|---|---|
| **restickify-activation** | `buf46→buf14` softmax→@V | many:many / **form** | add source/dest-layout + operand fields, realize plan | none new — `ReStickifyOpLx` LIVE + DSM self-promotes (`lxopt.cpp:3798`) | **op LIVE, auto-insertion absent** |
| **all-gather operand** | `buf21→buf22` @V operand | 1:many→many:many / none | plumb `read_index` + ds-type | wire `InputFetchNeighbor` into pipeline + generalize off `DsTypes::INPUT`→KERNEL | **multicast LIVE; InputFetchNeighbor present-but-dead** |
| **gather** | mirror of above | many:1 / none | `communication_pattern="gather"` | falls out of (2); `GatherOpHBM` is HBM-lane (wrong) | reverse-STCDPOpLx, not standalone |
| **reduce/all-reduce** | softmax Lk collapse | many:1 + arith | classify only | PSUM-ring compute; `AllReduce` = multi-AIU only | **compute LIVE; no LX movement collective — do not design** |

**Bottom line.** Restickify is the cheapest next step (A/B the existing DSM self-promotion before writing any interface expansion). All-gather is the higher-value structural fix but needs real deeptools work (generalize + wire `InputFetchNeighbor`, route replication through live `STCDPOpLx` multicast). Neither needs a new ring primitive. Gather is deferred; reduce/all-reduce is state-only (no single-AIU LX collective exists).

Key read-only refs: torch-spyre `_inductor/lx_relayout.py`, `codegen/superdsc.py`, `scratchpad/allocator.py` (branches `origin/ah/comms-collectives-{stcdp,dldsc}-agent`); deeptools `dxp/SdscRelayoutInsertion.cpp`, `dcg/.../inputNeighFetchOp.cpp`, `dcg/dcg_manager/dcg_manager.cpp:533`, `dsm/workOptimizer/baseOptimizer/lxopt.cpp:3798`, `dcg/.../stcdpOp.cpp`, `dsm/coll/comm.cpp`.
