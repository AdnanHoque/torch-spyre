# The NVIDIA GPU Equivalent of On-Chip Core-to-Core Data Movement on the IBM Spyre AIU

A translation layer for CUDA/GPU engineers. It maps the work described in
`CoreToCoreDataMovementRecipe.md` (the Spyre AIU core-to-core LX↔LX data-movement
primitive proven on silicon) onto its closest NVIDIA GPU analogs, and is honest about
where the analogy breaks down.

Hardware numbers for the AIU are grounded in the recipe (§1) and
`reference_aiu_architecture.md`. GPU numbers are from public NVIDIA documentation and
are flagged where I'm not certain. This doc **extends** the high-level GPU-vs-Spyre
table already in `docs/source/architecture/dataflow_architecture.md` ("Comparison with
GPU and Other Accelerators", which stops at *thread blocks ↔ core tiles*); it does not
duplicate it. The new content here is the **distributed-shared-memory / cluster** mapping
that the existing table omits, which is the truest analog of what we built.

---

## 1. TL;DR — if you know CUDA, here's what this is

We took a producer→consumer activation handoff that was going **out to HBM and back
between two kernels** and kept it **on-chip**, moving the data **directly from one core's
local scratchpad to another core's local scratchpad over an on-die ring**, with **zero
global-memory traffic**.

In CUDA terms: this is **FlashAttention's "keep the intermediate in SRAM instead of
materializing it to HBM" idea, generalized to an arbitrary producer→consumer edge**, plus
**Hopper Thread Block Cluster Distributed Shared Memory (DSM)** — where SMs in a cluster
read and write each other's shared memory through the SM-to-SM network *without* a
global-memory round trip. We fused a producer op and a consumer op into one persistent
kernel-like unit (the Spyre "mixed SuperDSC") and inserted explicit SM-to-SM-style copies
(the data-ops) between them, so the activation never leaves on-chip SRAM. We proved on
real hardware that all 32 cores actually exchange data over the ring (the GPU equivalent
of confirming, at the SASS level, that `cp.async` / DSM `st`/`ld` instructions target a
peer SM's shared memory rather than touching L2/HBM), and that the result is bit-faithful
to the HBM baseline.

The one-line slogan: **we turned a global-memory round trip into a cluster DSM
exchange.**

---

## 2. Hardware mapping table

| Spyre AIU concept | NVIDIA GPU concept | Notes / where the analogy is tight or loose |
|---|---|---|
| **Core** (32 on one AIU, on a RIU ring) | **SM** (Streaming Multiprocessor) | Both are the unit of parallel execution that owns a private fast scratchpad. 32 cores is *tiny* vs a modern GPU (H100 ≈ 132 SMs). |
| **Corelet** (2 per core: CW / CCW, each `8×8 PE • PT • PE • SFP`) | roughly a **processing block / SM sub-partition** (an SM has 4) | Loose. Spyre corelets are systolic PE tiles, not SIMT warp schedulers. |
| **PE array (8×8 systolic) + PT** | **Tensor Cores** | Both are the dense matmul engines. Spyre's are explicitly dataflow/systolic; Tensor Cores are warp-issued MMA units. |
| **LX scratchpad (2 MB/core, ~140 GB/s/core, ~4.5 TB/s aggregate)** | **SM shared memory / L1 (SMEM)** | The key on-chip SRAM. **Size differs sharply: 2 MB/core on AIU vs ~228 KB/SM max on Hopper** (configurable SMEM up to 227 KB; ~256 KB combined L1+SMEM). Spyre's per-core SRAM is ~8–9× larger. |
| **RIU BiRing (166 GB/s/dir, 333 GB/s/link, 33 nodes), cross-core LX↔LX moves via `l3lu`/`l3su` (`L3_LDU`/`L3_STU`)** | **Hopper Thread Block Cluster + Distributed Shared Memory over the SM-to-SM network** | **The headline analog.** Both let one compute tile read/write a peer tile's private SRAM *without* going to global memory. (See §3.) Inter-GPU analog is NVLink/NVSwitch, but DSM is the truer intra-device match. |
| **Shared HBM bus (LPDDR5, 166 GB/s, shared across all 32 cores)** | **Global memory (HBM) + L2** | The off-chip pool. On Spyre it's a *single shared 166 GB/s pipe* — the binding bottleneck. GPU HBM is far higher BW (H100 HBM3 ≈ 3.35 TB/s) and there's a large L2 (~50 MB) the AIU has no equivalent of. |
| **SFP UniRing (35.2 GB/s, intra-corelet)** | on-SM datapath between sub-partitions | **NOT** an inter-core fabric — easy to confuse with the RIU ring. Don't map this to NVLink. |
| **The HBM round trip we eliminate** (producer writes activation to HBM, consumer reads it back, in a separate SDSC) | **Writing an activation to global memory in kernel *k* and reading it back in kernel *k+1*** | The exact thing kernel fusion / FlashAttention exists to avoid. |
| **SDSC** (the runtime launch unit; LX *does* persist across an `sdsc_execute` boundary in PF / single-user VF — measured; the default HBM round-trip is the planner evicting to HBM, not a hardware wipe) | **a kernel launch** (shared memory genuinely does not persist across kernel boundaries) | Looser than it looks. On the GPU the SMEM wipe is a hardware fact; on Spyre the default HBM round-trip is a *scheduling choice* (the planner evicts at SDSC boundaries), so the handoff can be kept on-chip via a mixed SuperDSC *or* an LX-planner change (same-shard) without fusing. |
| **Mixed SuperDSC** (`dscs_` consumer DL op + `datadscs_` data-ops + `coreIdToDscSchedule`) | **a fused / persistent kernel with a producer→consumer pipeline** | The data-ops are the explicit SM-to-SM copies inserted before the consumer op; `coreIdToDscSchedule` is the per-SM step schedule with barriers (like cluster-wide `barrier.cluster.arrive/wait`). |
| **`STCDPOpLx`** (same-stick LX→LX move; rides the ring) | **a `cp.async` / TMA bulk copy whose source is a peer SM's DSM address** | The actual data-movement instruction. Same-stick ⇒ same layout ⇒ a plain async copy, no swizzle. |
| **`L3_MVLOOPCNT` over `L3_LDU`/`L3_STU` (hardware streaming loop)** | **looped `cp.async` / `cp.async.bulk` (TMA) tiling loop** | The microcode streaming loop that pumps sticks across the ring ≈ the async-copy pipeline loop in a CUTLASS mainloop. |
| **`memId` field on each per-core `PieceInfo`** (which physical core owns logical slice *i*) | **the destination SM's `%cluster_ctarank` / mapped DSM address** | Same `memId` src/dst = same-SM copy (no network traffic); different `memId` = real peer-SM DSM transfer. This is the one field that decides local vs cross-core. |
| **`ReStickifyOpWithPTLx`** (layout-CHANGING transpose; **faults Compute-CB on device**) | **a transpose in shared memory** (the bank-conflict / swizzle problem) | The blocked frontier. A same-layout move is cheap; a layout change is the hard, separately-engineered case. (See §4, §7.) |
| **Static `senprog`** (the only loadable program file, produced AOT by dxp) | **the compiled cubin/SASS** (AOT) | Spyre is *purely* AOT with a compile-time core schedule; CUDA mixes AOT cubins with a runtime hardware warp scheduler. |
| **Ring distance `min(\|i-j\|, 32-\|i-j\|)`, `byte_hops` cost metric** | **cluster locality / NVLink hop topology cost** | On a ring, distance matters and wraps at 32; on NVSwitch it's a near-uniform crossbar (any-to-any roughly equal cost). |

---

## 3. The technique mapping (the heart of it)

Two distinct GPU techniques together describe what we built. Keep them separate.

### 3a. Eliminating the HBM round trip ≡ kernel fusion / SMEM-resident producer→consumer

On Spyre, a *bundle* is a sequential list of SDSCs. **LX persists across an
`sdsc_execute` boundary in PF / single-user VF (the de-facto mode) — measured**; the
earlier "LX is wiped" was the *planner* conservatively evicting to HBM and resetting its
LX tracking at SDSC boundaries, not a hardware wipe. So a producer in SDSC *k* and a
consumer in SDSC *k+1* communicate through HBM **by default** (the stock pipeline inserts
a `ReStickifyOpHBM` SDSC between them) — but keeping it on-chip is a scheduling choice.
The GPU analogy is *looser* than a hardware equivalence: on CUDA **shared memory genuinely
does not persist across kernel launches**, so to keep the intermediate in SMEM you must
**fuse the two kernels into one** (or run a **persistent kernel**). On Spyre the same
on-chip handoff is realized today via the mixed SuperDSC, or — for a same-shard handoff —
via an LX-planner change (don't-evict + coordinate LX addresses across consecutive
OpSpecs, measured to work on stock dxp), which needs no fusion.

The canonical CUDA instance is **FlashAttention** \[Dao 2022\]: instead of materializing
the N×N attention-score / softmax matrix to HBM between the QKᵀ matmul and the softmax·V
matmul, it keeps the scores tiled **in SRAM** and fuses the whole attention into one
kernel. The speedup is purely from **not paying the HBM round trip** — not from doing less
math. Our Spyre work is the same move: the producer→consumer activation stays in LX, the
explicit `ReStickifyOpHBM` is deleted, and we save `2 × tensor_bytes` of off-chip traffic
on a 166 GB/s shared pipe per edge.

Difference from FlashAttention: FlashAttention keeps the intermediate in **one** SM's SRAM
(it's a within-tile/within-block fusion). Our handoff is **across cores** — slice *i*
produced on core *i* may be consumed on a different core — which is why §3b is also needed.

### 3b. The cross-core ring move ≡ Hopper cluster Distributed Shared Memory (DSM)

This is the part a CUDA engineer should latch onto. Before Hopper, the only way one SM
could see data another SM produced in *its* shared memory was to round-trip through global
memory (or L2). **Hopper Thread Block Clusters** changed that: thread blocks in a cluster
are co-scheduled on SMs of the same GPC, and the **SM-to-SM network** lets a block
**directly load/store another block's shared memory** ("Distributed Shared Memory"),
addressed via the cluster's mapped SMEM. No global-memory trip.

That is **precisely** what the Spyre RIU ring + `STCDPOpLx` does: core *i* reads/writes
core *j*'s LX scratchpad over the on-die ring (`L3_LDU`/`L3_STU`), no HBM. The `memId`
field is the moral equivalent of selecting a peer SM within the cluster (`%cluster_ctarank`
/ DSM-mapped address). The reversed-ownership proof (core *i* ↔ core *31-i*) is literally
**every SM in the "cluster" exchanging shared-memory contents with a peer SM, verified at
the instruction level** — analogous to confirming in SASS that a DSM load targets a peer
SM's SMEM rather than emitting a global load.

Why DSM (not NVLink) is the *truer* analog: both DSM and the RIU ring are **intra-device,
SRAM-to-SRAM, sub-microsecond** fabrics that bypass the global-memory hierarchy. NVLink /
NVSwitch is the **inter-GPU** analog (peer-GPU HBM access across a multi-GPU box); it's a
fair mapping for "core-to-core *between accelerators*," but here everything is on **one**
32-core die, so DSM is the right mental model. (Caveat: I'm confident DSM is SMEM↔SMEM
without a global round trip; I'm less certain of exact Hopper DSM bandwidth numbers, so I
don't quote one.)

### 3c. The mixed DL + data-op SuperDSC ≡ a warp-specialized producer-consumer pipeline (CUTLASS/CuTe)

The mixed SuperDSC packs, in one schedulable unit: the consumer compute op (`dscs_`), a
list of explicit data-movement ops that run *first* (`datadscs_` = the `STCDPOpLx`
moves), and a per-core step schedule with barriers (`coreIdToDscSchedule`, rows of
`[datadsc_idx, dldsc_idx, after_sync, before_sync]`).

The closest CUDA construct is a **warp-specialized, software-pipelined kernel** as written
in **CUTLASS/CuTe** for Hopper: dedicated **producer (DMA/TMA) warps** issue async copies
to fill SMEM stages while **consumer (MMA) warps** compute on filled stages, coordinated
by named barriers (`mbarrier` / `barrier.cluster`). Our `datadscs_`-then-`dscs_` schedule
with `after_sync`/`before_sync` flags is the AOT, per-core spelled-out version of exactly
that producer→barrier→consumer pipeline. A **megakernel** (one persistent kernel that
runs many fused stages keeping data resident in SRAM) is the same spirit at larger grain.

The difference: CUTLASS expresses this with **warp specialization decided at runtime by
the scheduler**; Spyre spells out **a fixed per-core schedule at compile time** in the
senprog (see §4).

**On overlap.** Cross-SDSC execution is **serial by default** (`STRICT_ORDERING`;
measured `time(M+C)=time(M)+time(C)`); the runtime has 3 pipelines
(`COMPUTE`/`ASYNC_DMAI`/`ASYNC_DMAO`) + an `OP_ORDERING` mode, but it is unplumbed.
Producer→move→consumer **overlap** therefore has two routes: a co-scheduled mixed
SuperDSC (compile-time; the warp-specialization analog above), or plumbing `OP_ORDERING`
for separate SDSCs (runtime; the CUDA-streams analog). The mixed SuperDSC is *sufficient*
but not the *only* overlap path.

### 3d. The hardware streaming loop ≡ TMA / `cp.async`, looped

`STCDPOpLx` lowers to a microcode loop: `L3_MVLOOPCNT` sets the iteration count and the
body issues `L3_LDU` / `L3_STU` (with `L3_SYNC` barriers) to stream sticks across the
ring. That is the AIU spelling of a **bulk asynchronous copy loop** — `cp.async` on
Ampere, or **TMA (`cp.async.bulk` / the Tensor Memory Accelerator)** on Hopper — iterated
over tiles in a mainloop. A "stick" (128 B / 64 fp16 elements) is the transfer
granularity, analogous to a TMA tile / cache-line-aligned async-copy chunk.

---

## 4. What's genuinely DIFFERENT (where the analogy breaks down)

Be honest about these — a GPU engineer who assumes 1:1 will get burned.

1. **Ring topology vs crossbar.** The RIU is a **bidirectional ring** of 33 nodes; cost
   scales with `ring_distance(i,j) = min(|i-j|, 32-|i-j|)` and wraps. NVSwitch (inter-GPU)
   and the Hopper SM-to-SM cluster network behave much more like a **near-uniform
   crossbar** within their domain — peer locality matters far less. On Spyre, *which* core
   you talk to is a first-class cost term (`byte_hops`); on a GPU cluster it mostly isn't.

2. **Static senprog vs dynamic launch + warp scheduler.** Spyre is **purely
   ahead-of-time**: the only loadable program is the `senprog`, and the per-core op
   schedule (which core runs which data-op when, with which barriers) is **frozen at
   compile time** by dxp. There is **no runtime hardware scheduler, no occupancy, no
   dynamic block dispatch, no register-pressure-driven occupancy tradeoff**. CUDA launches
   a grid and a hardware **warp scheduler** hides latency at runtime by swapping warps.
   This is the deepest structural difference: we had to *synthesize the schedule
   ourselves* (`coreIdToDscSchedule`), which a CUDA programmer never does explicitly.

3. **Explicit per-core data-ops vs warp specialization.** On Spyre the cross-core copies
   are **first-class compiler-emitted ops** (`STCDPOpLx` entries in `datadscs_`) with
   per-core placement (`memId`). In CUTLASS the equivalent moves are **implicit in warp
   roles** and issued by producer warps at runtime. We schedule data movement the way a
   compiler schedules instructions; CUDA schedules it the way a runtime fills a pipeline.

4. **SRAM sizes are inverted from intuition.** Spyre LX is **2 MB per core**; Hopper SMEM
   is **~228 KB per SM max**. So each Spyre core has ~8–9× the on-chip SRAM of an H100 SM —
   but there are only **32 cores vs ~132 SMs**, and **no L2** (GPUs have a large ~50 MB L2
   that absorbs a lot of would-be-global traffic with no programmer effort). Spyre has *no*
   hardware cache at all — *every* byte of data movement is explicit (this is the existing
   dataflow doc's "explicit DMA via compiler" vs "implicit caching" row). A GPU engineer's
   instinct that "SMEM is tiny, spill to L2/HBM" does not transfer: on Spyre there's
   nothing to spill to except the one slow shared HBM pipe.

5. **No cache coherence, no implicit reuse.** GPUs give you L1/L2 coherence and implicit
   caching of reloaded data. Spyre gives you neither — which is *why* keeping the handoff
   in LX matters so much: there is no cache to silently save you from the HBM round trip.

6. **The transpose wall (Tier-2).** A **same-stick** cross-core move (`STCDPOpLx`) runs
   clean on device. A **layout-changing** move (`ReStickifyOpWithPTLx`, a stick transpose)
   **faults with a Compute-CB hardware error** and is the open frontier. On a GPU, a
   transpose-during-SM-to-SM-exchange is "just" a **shared-memory transpose with swizzling
   to avoid bank conflicts** — annoying but a solved, well-trodden technique (padding /
   XOR swizzle). On Spyre it's currently a hardware/deeptools blocker, not a swizzle you
   can write. So **Tier-1 (same-layout) ≈ a same-layout DSM copy** (easy, done), and
   **Tier-2 (layout-changing) ≈ a swizzled transpose-in-SMEM** (on GPU: routine; on Spyre:
   blocked).

---

## 5. What a GPU engineer would call our results

Plain-English translations of the recipe's claims (§8, §9) into GPU vocabulary:

- "Eliminated the HBM round trip for a producer→consumer activation handoff" →
  **"fused two kernels so the intermediate stays in SMEM instead of round-tripping
  global memory"** (FlashAttention-style, but for an arbitrary producer→consumer edge,
  not just attention).
- "Genuine cross-core ring STCDP, all 32 cores `L3_LDU`/`L3_STU` to mirror core *31-i*" →
  **"a cluster-wide distributed-shared-memory exchange where every SM reads a peer SM's
  shared memory over the SM-to-SM network, verified in SASS."**
- "max_err 0.0137 = baseline, no Compute-CB fault, negative control fails as required" →
  **"bit-faithful to the global-memory baseline; and we proved the device actually ran our
  fused/DSM kernel (delete the cubin → it must fail), not a cached fallback."** The
  `g_artifact_cache` gotcha is the CUDA "stale cubin / JIT cache served the old kernel"
  trap, and the negative control is the fix.
- "Speedup peaks mid-range (1.22× @1024), tapers as matmul O(N³) dwarfs handoff O(N²)" →
  **"the fusion win is bandwidth-bound: it's largest when the handoff is a big fraction of
  runtime, and shrinks (relatively) once the GEMM dominates — classic
  arithmetic-intensity / roofline behavior."**
- "Same-core STCDP rings are dead-code-eliminated (zero `L3_LDU`/`L3_STU`)" →
  **"if producer and consumer tiles are co-located on the same SM, there's no SM-to-SM
  traffic at all — it degenerates to an in-SMEM copy."** (The proof had to *force*
  cross-SM traffic via reversed ownership, exactly like writing a microbenchmark that
  guarantees the DSM path instead of letting the compiler keep it local.)

In one sentence: **it's FlashAttention-style on-chip residency generalized to arbitrary
producer→consumer edges, with the cross-tile hop realized as a Hopper-cluster DSM exchange
instead of a global-memory round trip — and proven at the instruction level on silicon.**

---

## 6. Where the analogy helps prioritize

The same reasoning that tells a CUDA engineer *when fusion and DSM pay off* tells us where
these Spyre wins land — and it matches the recipe's measured results (§9, §12).

- **Bandwidth-bound regimes win most.** Fusion/DSM help when you're **memory-bound**, not
  compute-bound. The recipe's measured sweet spot (relative speedup peaks at mid hidden
  dim ~1k–2k, tapers as the GEMM grows) is the textbook **roofline** story: removing a
  global round trip matters most on the bandwidth-bound side of the ridge.
- **Decode / autoregressive generation.** Skinny matmuls (batch≈1, seq=1) are
  **memory-bandwidth-bound** — activations dominate traffic. This is the GPU world's prime
  fusion target (decode kernels live and die by HBM traffic), and it's the recipe's top
  pick too.
- **MoE.** Router → expert-FFN → combine shuffles a lot of activation **between tiles** —
  high cross-core/cross-SM data volume relative to compute. On GPUs this is exactly where
  cluster DSM / NVLink all-to-all matter; on Spyre it's the highest ring-vs-HBM leverage
  case (and the real `bmm` case per the project notes).
- **De-prioritize compute-bound prefill with large GEMMs.** Just as a CUDA engineer
  wouldn't bother fusing away a tiny epilogue copy in front of a huge GEMM, the recipe
  shows the *relative* win shrinks there (the O(N²) handoff is dwarfed by O(N³) matmul) —
  even though absolute bytes saved grow.
- **Small / awkward shapes belong off-chip.** At S=512 the per-core slice is a *sub-stick*
  (16 < 64 fp16) and the move regresses (0.95×) — the analog of a transfer too small to
  amortize `cp.async`/TMA setup; leave it in "global memory." Gate the on-chip path on a
  minimum size, exactly as you'd gate a fusion on a minimum tile.

---

## 7. Quick reference: the two tiers in GPU terms

| Spyre tier | What it is | GPU analog | Status |
|---|---|---|---|
| **Tier-1: same-stick cross-core move** (`STCDPOpLx`) | move a slice to a peer core's LX, same layout | a **same-layout DSM copy** between SMs in a cluster (looped `cp.async`/TMA, peer-SM destination) | **proven on device**, value-correct, HBM-free |
| **Tier-2: layout-changing move** (`ReStickifyOpWithPTLx`) | move + transpose the stick orientation | a **swizzled transpose-in-shared-memory** during the exchange (bank-conflict territory) | **blocked** (Compute-CB hardware fault); on GPU this would be routine |

---

## Sources and provenance

- Spyre work mapped here: `CoreToCoreDataMovementRecipe.md` (the implementation recipe —
  hardware §1, mechanism §3–4, proof §8, perf §9, applicability §12).
- AIU hardware numbers: `reference_aiu_architecture.md` (32 cores, RIU BiRing 166 GB/s/dir,
  HBM 166 GB/s, LX 2 MB/core ~4.5 TB/s aggregate) and recipe §1.
- Existing (and complementary) high-level table this doc extends:
  `docs/source/architecture/dataflow_architecture.md` → "Comparison with GPU and Other
  Accelerators" (covers scheduling/memory-model/parallelism; this doc adds the
  DSM/cluster/FlashAttention/CUTLASS mapping it lacks).
- GPU facts (FlashAttention IO-awareness, Hopper Thread Block Clusters + Distributed
  Shared Memory, TMA/`cp.async`, CUTLASS warp-specialized pipelines, NVLink/NVSwitch) are
  from public NVIDIA/CUDA documentation and literature; specific Hopper figures (≈228 KB
  SMEM/SM, ≈132 SMs, ≈50 MB L2, ≈3.35 TB/s HBM3) are commonly-cited H100 numbers — treat
  as approximate. I deliberately do **not** quote a Hopper DSM bandwidth figure because I
  am not certain of it; the load-bearing claim (DSM is SMEM↔SMEM without a global round
  trip) is well-established.
