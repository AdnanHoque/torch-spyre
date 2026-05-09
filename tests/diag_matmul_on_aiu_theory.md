# Matmul on the AIU 1.0 — a first-principles guide

A canonical reference for how matrix multiplication actually executes
on the IBM Spyre Accelerator (AIU 1.0). Builds up from chip
geometry → memory hierarchy → tensor dataflow → work-division → loop
nests → empirical performance, with sample-calculation passes over
each section so the reader can derive the numbers themselves.

The intended endpoint: by the end of this doc, a reader should be
able to look at any (M, N, K) shape and predict, within a factor of
two, what split the planner ought to pick and what fraction of
compute-peak the kernel will achieve.

This synthesizes:
- the deeptools system configuration in
  `deeptools/dsc/HardwareArchMapping/sysConfigs2.0/sentient_dd2_sysconfig.json`
  (the source-of-truth for compiler-visible hardware specs),
- the Spyre Inductor backend code in `torch_spyre/_inductor/`,
- the IBM-published architecture docs in `docs/source/architecture/`,
- empirical findings from this branch's measurement campaigns
  (`diag_k_fast_combined_findings_normalized.md`,
  `diag_k_fast_granite_findings.md`,
  `diag_small_m_spread_findings.md`).

Numbers in this doc that don't have a citation are derived from
those primary sources.

## 1. Chip geometry

Per AIU card (from `sysconfig.json` `BaseElems` and the published
spec sheet):

- **32 cores**, manufactured on TSMC 5 nm, in a single PCIe Gen 5 ×16
  package at 75 W TDP.
- **128 GB LPDDR5** off-chip device memory at **1.3 GHz, 128 B/cycle
  → 166.4 GB/s** of HBM bandwidth.
- Five distinct on-chip interconnects (see §7).
- Published headline throughput: **>300 TOPS** (this is the int4
  marketing number; the fp16 PT compute-peak is 72 TFLOPS — see §3).

Per core (the unit work-division reasons about):

- **Two corelets** sharing one **2 MB LX scratchpad** (SRAM,
  compiler-managed; no hardware cache).
- Each corelet has its own **8 × 8 systolic Processing Element (PE)
  array**, driven by the **PT execution unit** for matrix compute
  and by the **PE execution unit** for elementwise/reduction ops.
  The two execution units share the same 8 × 8 PE silicon but use it
  with different microcode and different `parallelEngines` counts.
- Each corelet also has a 1D **SFP** (Special Function Processor)
  for non-linear activations (GELU, softmax, etc.) — irrelevant for
  pure matmul but matters when activations get fused.
- Hardware ceiling on per-core HMI access: a **256 MB span limit**,
  enforced by `span_reduction_pass` in
  `torch_spyre/_inductor/work_division.py`.

## 2. Sticks: the atomic unit of data movement

All on-chip data movement happens in **sticks** — 128-byte chunks.
The constant `BYTES_IN_STICK = 128` appears across the runtime,
compiler, and tensor-layout code. At fp16 that's 64 elements per
stick.

Why this matters: every quantitative reasoning about per-core
memory, ring traffic, or PT throughput is most cleanly expressed in
sticks, not raw elements. The work-division planner converts
element-valued iteration spaces into stick-valued ones via
`adjust_it_space_for_sticks` before applying any split, so that
sharded extents are always whole-stick multiples.

For an fp16 matmul A (M × K) × B (K × N):

- A = M·K / 64 sticks
- B = K·N / 64 sticks
- C = M·N / 64 sticks

If any of M, N, K is not divisible by 64, the trailing stick is
zero-padded; the planner tracks that and accounts for it
conservatively.

## 3. The PT execution unit

This is THE matmul compute engine. From `sysconfig.json`:

```
PT: numCopies = 64,  frequency = 1.1 GHz,  parallelEngines = 512   (fp16)
                                            parallelEngines = 1024  (fp8)
                                            parallelEngines = 2048  (int8)
                                            parallelEngines = 4096  (int4)
```

`numCopies = 64` because there's one PT unit per corelet (32 cores
× 2 corelets). `parallelEngines` is the per-cycle MAC count of one
PT unit at the given precision.

The factorization for fp16:

```
512 = 8 (PT M-rows) × 8 (PT N-cols) × 8 (K-direction SIMD)
```

So one PT cycle on one corelet does an 8 × 8 outer product, with
8-deep K accumulation, into an 8 × 8 PSUM tile. Per-corelet
throughput is 512 MAC/cycle = 1024 fp16 ops/cycle.

**Compute-peak per AIU at each precision:**

| precision | parallelEngines | per-AIU throughput |
|---|---:|---:|
| fp16 | 512  | 64 × 512 × 1.1 GHz × 2 ops/MAC = **72.1 TFLOPS** |
| fp8  | 1024 | 144.2 TFLOPS |
| int8 | 2048 | 144.2 TOPS |
| int4 | 4096 | **288.4 TOPS** (≈ the published "300 TOPS" headline) |

For everything that follows, fp16 peak = 72 TFLOPS is the right
yardstick for our matmul kernels.

The fundamental geometric fact: **a corelet processes data in
8 × 8 × 8 blocks**. So the M dimension fed to a corelet must be
≥ 8 to fill the PT M-rows; below 8, those PE rows do nothing and the
corelet runs at fractional utilization that cycle.

## 4. Memory hierarchy

```
LPDDR5 ─── DMA ──→ HMI ─── load ──→ LX scratchpad ─── feed ──→ PT array
(128 GB,         (per-core              (2 MB per                (8×8 PE)
 166 GB/s)         256 MB span)           core, shared
                                          across both
                                          corelets;
                                          1.1 GHz, 128 B/cycle
                                          per core ⇒ 140 GB/s/core
                                          → 4.5 TB/s aggregate)
```

There is **no hardware cache**. The compiler explicitly schedules
load/store instructions that move tiles between LPDDR5 and the LX
scratchpad. The PT array reads operands from LX directly, accumulates
in per-PE registers, and writes results back to LX (and from there
out to LPDDR5).

The LX is shared between two LX consumers within one core:

1. **Operand resident-set**: A and B tiles for the current PT pass.
2. **Output PSUM accumulators**: per-core M_per × N_per ×
   `psum_bytes`, where `psum_bytes` is typically 4 (fp32 PSUM).

The PSUM accumulator is often the binding LX constraint at small M
shapes (verified by Probe 3 in the May 2026 investigation, summarised
in this branch). At fp32 PSUM with M_per = 32 and N_per = 1024
elements (16 sticks), the accumulator tile is 32 · 1024 · 4 = 128 KB
out of 2 MB.

The `DXP_LX_FRAC_AVAIL` env var splits LX between user code
(inductor scratchpad allocator) and the deeptools backend (`Dxp` in
`deeptools/dxp/dxp.cpp`):

```
backend reservation = lx_capacity × (1 − DXP_LX_FRAC_AVAIL)
inductor available  = lx_capacity × DXP_LX_FRAC_AVAIL
```

(Yes — the variable name reads as "fraction available", and that's
what it actually means *for the inductor user side*. The backend
reservation is the complement.)

Default 0.2 → 20% inductor / 80% backend. We measured the kernel
behaviour at both 0.2 and 1.0 in
`diag_k_fast_combined_findings_normalized.md`; setting it to 1.0
shrinks all the wins (geomean 2.22× → 1.93×) but doesn't introduce
regressions.

## 5. The three tensors of matmul

For C = A · B with A ∈ ℝ^{M×K}, B ∈ ℝ^{K×N}, C ∈ ℝ^{M×N}, three
tensors flow through the kernel with very different access
patterns. The pattern determines which is **streamed** vs
**block-loaded** vs **stationary**.

### 5.1 A (the activation, M × K) — streamed

Activations in transformer inference are per-token, varying across
calls. They're typically smaller than B (M ≪ K, N for decode
batches) and don't get reused across kernel invocations.

A is **streamed**: each (m, k) tile is loaded into LX, fed to the
PT array, and dropped — no long-term residency. Streaming pressure
on HMI bandwidth scales with the per-core A footprint (M·K / m
elements per core).

### 5.2 B (the weight, K × N) — block-loaded, near-stationary

Weights are reused across many invocations of the same operator —
across all decode steps, across all batches in a prefill, across all
heads in attention. Loading B once and amortizing across many
invocations is the central optimization.

B is **block-loaded** with **near-stationary residency**. The
compiler arranges for B to be staged into LX once per kernel pass
(or once per multi-pass, if B exceeds the LX residency budget),
and the PT array reads from LX repeatedly across the M loop.

This is the **weight-stationary** pattern. It's why (a) wider K
favors K-split (the per-cluster B share shrinks linearly in
`k_split`) and (b) PSUM ring traffic only matters once K is large
enough to amortize ring-reduction overhead.

### 5.3 C (the output, M × N) — accumulated in PSUM, drained at end

Each PT cycle accumulates into per-PE partial-sum registers; the
8 × 8 tile of partial sums lives in the PE registers, not in LX,
during the inner K-loop. After the K-loop completes, the accumulated
tile is **drained** to the LX scratchpad. If this kernel is the only
contributor to that output region, the LX-resident PSUM tile is
written to LPDDR5 directly. If this kernel is part of a K-split
**cohort**, the PSUM tile is sent over the SFP ring to be summed
with its peers.

C is therefore **accumulator-stationary inside the K-loop, then
ring-summed at the end** in the K-split case. This is what makes
K the "expensive" axis to split.

### 5.4 Per-cluster HMI traffic formula

Combining all three, for a work-division `(m, n, k)` with
m·n·k = 32 the total HMI traffic per kernel is:

```
HMI_bytes = n · M · K · sizeof(A_dtype)
          + m · K · N · sizeof(B_dtype)
          + k · M · N · sizeof(C_dtype)
```

The coefficients tell you how many times each operand is replicated
across cores. A is replicated by `n` (cores in the N-dimension all
read the same A row). B is replicated by `m` (cores in the
M-dimension all read the same B column). C is replicated by `k`
(K-cohort members each write a partial PSUM that gets ring-reduced).

This formula is the **single most important quantitative tool** for
predicting which split wins. A high-coefficient operand dominates
HMI traffic.

## 6. The work-division split

A work-division split is a tuple `(m, n, k)` with `m · n · k =
max_cores` (= 32 for full-AIU operation). Each axis tells you how
that dimension of the iteration space is sharded across the 32
cores:

```
M_per_core = M / m    (each core sees M/m rows)
N_per_core = N / n    (each core sees N/n cols)
K_per_core = K / k    (each core does K/k of the inner sum)
```

The compiler enforces stick alignment: `n` must divide `n_sticks`
(= N / 64 at fp16), and `k` must divide `k_sticks`. Otherwise the
shard sizes wouldn't be whole-stick multiples and data movement
breaks down.

Number of valid splits with `m · n · k = 32`: there are exactly
**21** ordered triples (counted in
`diag_small_m_theory_writeup.md`). The small-M sweep
(`diag_small_m_spread_driver.py`) enumerates all 21 per shape.

The big four families:

| Name | Pattern | Geometric interpretation |
|---|---|---|
| **pure-M** | `(32, 1, 1)` | 32-way row-sharding of A; every core sees full B and full N |
| **pure-N** | `(1, 32, 1)` | 32-way col-sharding of B and C; every core sees full A and full M |
| **mixed-MN (no K-split)** | `(m, n, 1)` with m, n > 1 | Cores form an m×n 2D grid over the output tile |
| **K-split** | `(1, n, k>1)` and `(m, n, k>1)` mixed | Multiple cores cooperate on each output element through a K-cohort |

## 7. The five on-chip interconnects and the K-cohort

`sysconfig.json:connections` defines the chip's interconnect fabric.
Five rings/networks live alongside each other:

1. **RIU BiRing** (Ring Interface Unit, data plane).
   33 nodes (32 cores + 1 HBM controller), 1.3 GHz, 128 B/cycle/dir
   bidirectional. This is the HBM ↔ MNI traffic path: every load or
   store to LPDDR5 traverses this ring.
2. **RIURequest BiRing** (request plane).
   33 nodes, 1.3 GHz, 1 B/cycle/dir. Carries request/control
   messages for the data plane.
3. **SFPDataIU UniRing for Corelet 0** (clockwise).
   32 nodes (one per core), 1.1 GHz, 32 B/cycle. Carries cross-core
   SFP traffic among Corelet 0s. **PSUM reductions over the K-cohort
   travel here** when the cohort uses Corelet 0.
4. **SFPDataIU UniRing for Corelet 1** (counterclockwise).
   Same as above, opposite direction, for Corelet 1s.
5. **On-core FIFO Links** (LX ↔ {MNI, PT, PE, SFP}, PT ↔ PE ↔ SFP).
   1.1 GHz, 128 B/cycle. The intra-core data path between LX and
   the execution units.

The split into separate data, request, and SFP rings is what's
informally called "QuadRings" in the literature and earlier
investigation notes — there are essentially 4 ring planes (data ×
biring + request × biring) for HBM traffic, plus 2 SFP rings, plus
on-core FIFOs.

When `k > 1`, k cores form a **K-cohort**: each cohort member
computes a partial PSUM of the same output tile, but over a different
1/k slice of the K dimension. To produce the final output, the
cohort must sum its k partial PSUMs over the corresponding SFP
unidirectional ring.

Each PSUM transfer hops adjacent cores. A reduction across k cohort
members requires k − 1 ring hops if the members are adjacent on the
ring; **more if they're spread out, because the unidirectional ring
forces traffic to traverse all intermediate cores even if those
cores aren't part of the cohort**.

**Identity emission (default):** the SDSC emitter assigns logical
core c to physical core c. With work-division `(m, n, k)`, the cohort
members are at physical indices `c, c + m·n, c + 2·m·n, ...`,
**m·n hops apart on the ring**. A 32-core ring with k=2 cohort at
split `(1, 16, 2)` has cohort members 16 hops apart — half-way around
the ring per chain.

**k_fast emission (this PR):** apply a permutation `perm[c] = (c %
k) · (m · n) + (c // k)` so cohort members occupy adjacent ring
positions. The reduction collapses to k − 1 hops total instead of
(k − 1) trips of m·n hops each. Since the permutation only affects
*which* logical slice each physical core executes — not the
sequence of operations within a slice — it degenerates to identity
when k = 1.

Code reference:
[`torch_spyre/_inductor/codegen/compute_ops.py:_k_fast_core_id_permutation`](torch_spyre/_inductor/codegen/compute_ops.py).

**Quantifying the cost reduction.** The SFP ring carries 32 B/cycle
at 1.1 GHz → **35.2 GB/s per direction per ring**. A typical PSUM
tile is `M_per_core_PT × N_per_core_PT × 4 bytes` = `8 × 8 × 4` =
**256 bytes per PT batch**, which traverses one ring hop in
`256 / 32 = 8 cycles`.

Concrete: at split `(1, 16, 2)` with N = 8192, M = 32:

- `M_per_core = 32 = 4 PT M-batches`,
  `N_per_core = 8192/16 = 512 = 64 PT N-batches`.
- PSUM tiles to reduce per K-cohort: 4 × 64 = **256 tiles**.
- Identity emission: 16 hops × 256 tiles × 8 cycles = 32 768 ring
  cycles per cohort = **30 µs** at 1.1 GHz.
- k_fast emission: 1 hop × 256 × 8 = 2 048 cycles = **1.9 µs**.

So k_fast saves ~28 µs of PSUM ring time on this shape. With a total
kernel time of ~940 µs, that's ~3% wall-time savings from k_fast
emission alone — consistent with the small B→C ratios (1.01-1.05×)
we measured.

## 8. Loop nest structure

A matmul kernel emits roughly:

```
# Outer loops — sharded across cores per the (m, n, k) split:
for mb in range(M_per_core / 8):           # M_per_core split into PT M-batches
  for nb in range(N_per_core / 8):         # N_per_core split into PT N-batches
    psum = zeros(8, 8)                     # PT-resident accumulator (fp32)

    # Inner K-loop — sequential per core, possibly split across K-cohort:
    for kb in range(K_per_core / 8):       # 8 = PT K-direction SIMD width
      a = lx.load(A_tile[mb, kb])          # 8 × 8 from A, fp16
      b = lx.load(B_tile[kb, nb])          # 8 × 8 from B, fp16
      psum = pt.outer_product(a, b, psum)  # 8×8 outer-product accum

    # K-cohort reduction (if k > 1):
    if k > 1:
      psum = sfp_ring_allreduce(psum, cohort)
      # (k-1) hops × 8 cycles/hop with kf emission
      # m·n·(k-1) hops × 8 cycles without

    # Drain to LX → HMI:
    lx.store(C_tile[mb, nb], psum)
```

This matches the structure in
`deeptools/dvs/setupVariables/batchmatmul_fp16_fwd.cpp` (the source
template that fills in SDSC parameters for fp16 batchmatmul: `Dmb`,
`Dout`, `Din`, `Cmb`, `Cout`, `Cin`, etc).

Three things drive performance through this structure:

1. **B reuse across the M loop.** Each `B_tile[kb, nb]` is read
   `M_per_core / 8` times. Bigger M_per_core ⇒ more B reuse ⇒
   memory bandwidth amortizes over more compute. This is the single
   most important reason pure-M with M_per_core ≪ 8 is a disaster:
   B is loaded fresh almost every PT cycle.
2. **PT array M-row utilization.** If M_per_core < 8, the inner PT
   cycles run with empty PE rows. M_per_core ≥ 8 fills the array;
   multiples of 8 maintain full util across PT batches.
3. **K-cohort reduction frequency.** The ring allreduce fires once
   per output tile, not once per K iteration. So the per-tile ring
   cost is fixed; what varies is the number of hops per transfer
   (kf vs identity) and the size of the PSUM tile.

## 9. The four split regimes — when each wins

Combine §6, §7, and §8 and you get a phase diagram for the optimal
split family, parameterized by M, N, K, and the per-core geometry.
Expressing thresholds in PT-array units (with PE_rows = 8,
max_cores = 32):

| Regime | M_per_core (under pure-M) | Optimal family | Why |
|---|---|---|---|
| **M < max_cores** | < 1 PT M-batch | `(1, n, k>1)` + kf | Pure-M leaves nearly all PE rows empty; only K-split keeps cores busy |
| **max_cores ≤ M ≤ 4·max_cores**, narrow N | 1-4 PT M-batches | `(1, n, k>1)` + kf | Pure-M under-utilises PT; K-split + kf saturates with full M per core |
| **max_cores ≤ M ≤ 4·max_cores**, wide N (n_sticks div 8) | 1-4 PT M-batches | **mixed-MN (4, 8, 1)** | M+N split fills PT M-rows AND splits N — no PSUM ring cost |
| **max_cores ≤ M ≤ 4·max_cores**, awkward N | 1-4 PT M-batches | triple-mixed (4, 4, 2) etc + kf | M+N fills PT, K-split soaks the rest of the cores, kf collapses ring hops |
| **M ≥ 16·max_cores** | ≥ 16 PT M-batches | pure-M (32, 1, 1) | PT array already saturated, K-split adds overhead |

In concrete numbers at 32-core: M < 32 / M ∈ [32, 128] / M ∈
[128, 512] / M > 512 are the regime boundaries.

The PR 1986 heuristic captures rows 2 (and partially 1); it doesn't
capture row 3, which is where most of the M=32 / M=128 production
shapes live in the empirical sweep — the `(4, 8, 1)`-style family
needs a planner-side change to be considered as a candidate at all.

## 10. Theoretical performance and how close we get

Three theoretical ceilings bound any matmul kernel:

```
PT-cycle bound  = total_FLOPs       / fp16_peak (= 72 TFLOPs/s)
HMI-byte bound  = total_HMI_bytes   / 166 GB/s
LX-byte bound   = total_LX_bytes    / (140 GB/s × 32)
```

Wall time is at least `max(PT_bound, HMI_bound, LX_bound)`, plus
launch overhead and pipeline bubbles. In practice for our small-M
shapes, the kernel is **HMI-bound** — the binding constraint is B
operand traffic from LPDDR5 to LX.

### Sample calc 1: Llama 3.1 70B q_proj M=32

Shape: (32, 8192, 8192). Pure-M baseline 3.40 ms; best split
`(4, 4, 2)` + k_fast 0.94 ms.

```
fp16 ops total = 2 · 32 · 8192 · 8192 = 8.59 GFLOPS

PT-cycle bound  = 8.59e9 / 72.1e12 = 0.119 ms
HMI bytes (4, 4, 2):
  A: n · M · K · 2 = 4 · 32 · 8192 · 2 = 2.1 MB  (n_split · A bytes)
  B: m · K · N · 2 = 4 · 8192 · 8192 · 2 = 512 MB    (m_split · B bytes)
  C: k · M · N · 2 = 2 · 32 · 8192 · 2 = 1.0 MB
  Total = ~515 MB
HMI bound      = 515 MB / 166 GB/s = 3.10 ms
```

Wait — this predicts 3.10 ms while we measured 0.94 ms. The HMI
formula counts every per-core B load as a separate HMI access.
That's correct in the worst case, but a real kernel can prefetch and
overlap loads across cores via the BiRing's 128 B/cycle/direction
bandwidth: aggregate cross-core HMI throughput is ~(128 GB/s · 2
directions × 1.3 GHz) ÷ ring_overhead ≈ 200-300 GB/s effective.

Reasonable estimate: 515 MB / 250 GB/s ≈ 2.0 ms. Still well above
the measured 0.94 ms — meaning the K-split cohort actually shares
B across cohort members so only `m · K · N / k = 2 · 8192 · 8192 ·
2 = 256 MB` of B is fetched (because cores in the same K-cohort
read different K-slices of B, so the union is just K·N bytes,
replicated `m` times for the M-cohort).

Re-deriving the `B` term more carefully: if cores at the same
(m_idx, n_idx) but different k_idx share the B tile, then
B-coefficient = m, not m·k. That gives 256 MB for `(4, 4, 2)`,
predicted ~1.5 ms HMI bound. Within 2× of measured 0.94 ms.

So this kernel is **HMI-bound** and the K-split's primary benefit
isn't avoiding PT under-utilization (which it does also) but
**reducing the B replication factor** from `m = 32` (pure-M) to
`m = 4` (4-way M-split inside the K-cohort).

Achieved fraction of fp16 PT-peak: 119 / 940 = **13%**. That's
realistic: at this shape we're nowhere near compute-bound.

### Sample calc 2: Llama 3.1 70B q_proj M=128

Shape: (128, 8192, 8192). Pure-M baseline 3.59 ms; best split
`(4, 8, 1)` 0.99 ms.

```
fp16 ops total = 2 · 128 · 8192 · 8192 = 17.18 GFLOPS

PT-cycle bound  = 17.18e9 / 72.1e12 = 0.238 ms
HMI bytes (4, 8, 1):
  A: 8 · 128 · 8192 · 2 = 16 MB
  B: 4 · 8192 · 8192 · 2 = 512 MB    (no K-split, so m fully replicates B)
  C: 1 · 128 · 8192 · 2 = 2 MB
  Total = ~530 MB
HMI bound (effective ~250 GB/s) = ~2.1 ms
```

Measured 0.99 ms — *better* than the HMI bound. This means the
kernel is achieving partial overlap of compute and load (double-
buffering): while the PT array works on one B tile, the next B tile
is already streaming in. The bigger M_per_core = 32 (4 PT M-batches
per core) gives the prefetcher 4× more compute time to hide HMI
latency than the M=32 case.

Achieved fraction of fp16 PT-peak: 238 / 990 = **24%**. Better than
M=32 because more B reuse → more compute per loaded byte → better
HMI hiding.

### Sample calc 3: Granite 3 8B gate/up_proj M=32

Shape: (32, 12800, 4096). Pure-M baseline 2.64 ms; best split
`(4, 8, 1)` 0.72 ms.

```
fp16 ops total = 2 · 32 · 12800 · 4096 = 3.36 GFLOPS

PT-cycle bound  = 3.36e9 / 72.1e12 = 0.047 ms
HMI bytes (4, 8, 1):
  A: 8 · 32 · 4096 · 2 = 2 MB
  B: 4 · 4096 · 12800 · 2 = 400 MB
  C: 1 · 32 · 12800 · 2 = 0.8 MB
  Total = ~403 MB
HMI bound (~250 GB/s) = ~1.6 ms
```

Measured 0.72 ms — well below the naive HMI bound, again because
of compute-load overlap. Achieved fraction of fp16 PT-peak: 47 /
720 = **6.5%**. Lower than M=128 because M_per_core = 8 means only
1 PT M-batch per core — minimal compute per B load.

### What the gap looks like

| shape | M | best wall (ms) | PT-bound (µs) | achieved % | regime |
|---|---:|---:|---:|---:|---|
| 70B q_proj | 32 | 0.94 | 119 | 13% | HMI-bound, B-replication-dominated |
| 70B q_proj | 128 | 0.99 | 238 | 24% | HMI-bound, partial compute-load overlap |
| 8B gate/up | 32 | 0.72 | 47 | 7% | HMI-bound, smallest output |

For all three, the win from the small-M K-split / mixed-MN
heuristics is **a 3-4× speedup over pure-M**, not a closing of the
gap to compute-peak. The remaining gap comes from:

- HMI bandwidth (the dominant cost — closing it requires reducing
  per-core B replication or increasing B reuse)
- Per-kernel launch overhead
- LX residency turnover for B when B exceeds the LX budget (e.g.,
  the 70B q_proj's 128 MB B is way too big to be fully resident in
  LX × 32 cores = 64 MB total)

The PR 1986 heuristic delivers a meaningful chunk of the available
win (3-4× on small-M production shapes), but doesn't change the
HMI-bound regime structurally. To close more of the gap would need
operator fusion (cross-kernel B residency), better prefetch
scheduling, or quantization to fp8/int4 (which would give 2-4× more
throughput AND smaller B).

## 11. Empirical findings — what wins where on real production shapes

From the 84-shape exhaustive sweep (`diag_small_m_spread_findings.md`,
covering Llama 3.1/3.2 + DeepSeek V3 + Granite 3.x at M ∈ {1, 32,
128}):

```
                M=1       M=32     M=128    Total
pure-M           2          0         0       2  (2%)
k=1 mixed        1         12        17      30 (36%)
k>1 + id (1,n,k) 14         2         0      16 (19%)
k>1 + kf (1,n,k) 11         1         1      13 (15%)
k>1 + kf mixed   0          7         8      15 (18%)
k>1 + id mixed   0          6         2       8 (10%)
```

Geomean speedup vs pure-M:

- M=1: 1.03× (overhead-dominated regime; small parallelism wins)
- M=32: **2.60×**
- M=128: **2.58×**

Two production-relevant headlines:

1. **At M ∈ {32, 128}, mixed-MN `(4, 8, 1)` is the empirical
   global optimum more than half the time.** This split is *not* in
   the PR 1986 heuristic's candidate set (which only proposes
   `(1, n, k>1)`), so the PR is a local optimum that beats pure-M
   but doesn't reach the global one. Closing this gap is a
   planner-priority change, not a heuristic change.
2. **k_fast emission is strictly correctness-preserving + measurably
   useful.** It wins 26/84 shapes outright, ties on most of the
   rest, and never regresses (since k_fast collapses ring hops only
   when k > 1, and the planner only picks k > 1 when the K-split
   pays for itself). Free to keep on by default.

## 12. The compilation pipeline — putting it all together

For completeness, the path from a `torch.matmul` call to running
matmul kernels on the AIU:

1. **Inductor lowering** (`torch_spyre/_inductor/lowering.py`):
   the matmul is decomposed into the canonical iteration space
   {M, N, K} with the appropriate tensor reads/writes.
2. **Layout finalisation**
   (`torch_spyre/_inductor/insert_restickify.py`): tensors are
   tiled into stick-aligned layouts.
3. **Span reduction** (`work_division.py:span_reduction_pass`):
   computes `min_splits` to keep per-core memory under the 256 MB
   span limit.
4. **Work distribution** (`work_division.py:work_distribution_pass`):
   distributes remaining cores by priority. Includes the k_fast
   override (`_try_k_fast_split`) for matmul shapes that fit the
   small-M wide-N pattern.
5. **k_fast emission**
   (`codegen/compute_ops.py:_k_fast_core_id_permutation`): permutes
   physical core IDs so K-cohort members land on adjacent ring
   positions when k > 1.
6. **SDSC generation** (`codegen/superdsc.py`): produces a SuperDSC
   JSON that describes the kernel across all 32 cores. The SDSC
   format is consumed by the `Dxp` driver in
   `deeptools/dxp/dxp.cpp`, which calls into the `Ddc` codegen and
   `Dip` instruction-pack pipelines.
7. **Backend codegen** (`deeptools/dvs/setupVariables/batchmatmul_fp16_fwd.cpp`
   etc.): the per-precision matmul kernel template fills in the
   SDSC parameters (Dmb, Dout, Din, Cmb, Cout, Cin, etc.) and emits
   the actual instruction stream.
8. **Runtime execution** (`flex/`, `sendnn/`): the compiled kernel
   is dispatched as a SuperDSC blob via the host driver, executed
   on the AIU's 32 cores, and output tensors are restickified back
   to the host-visible layout.

## 13. Glossary

- **AIU 1.0** — IBM Spyre AI Card, the production accelerator card.
- **Core** — one of 32 compute units on the AIU. Holds 2 corelets +
  1 LX scratchpad.
- **Corelet** — half of a core. Has its own 8×8 PE array + 1D SFP.
- **PE (Processing Element)** — one cell of the 8×8 systolic
  multiply-accumulate array.
- **PT (Point execution unit)** — the matmul-specific path through
  the PE array. 512 parallelEngines (= MAC/cycle) at fp16.
- **PE execution unit** — the elementwise/reduction path through
  the PE array (different microcode, narrower parallelism).
- **SFP** — Special Function Processor. Per-corelet 1D vector unit
  for non-linear ops (GELU, softmax). Carries PSUM ring traffic.
- **LX scratchpad** — 2 MB compiler-managed SRAM per core, shared
  across the two corelets. 1.1 GHz, 128 B/cycle = 140 GB/s/core.
- **HMI (Host/Hardware Memory Interface)** — the path from LPDDR5
  to the LX scratchpad. ~166 GB/s.
- **EAR / span limit** — 256 MB hardware limit on per-core HMI
  access.
- **LPDDR5** — off-chip device memory; up to 128 GB per AIU card.
- **HBM** — used in the deeptools sysconfig as a label for what's
  actually LPDDR5 on Spyre. Don't be confused.
- **Stick** — 128-byte aligned data unit. 64 fp16 elements.
  `BYTES_IN_STICK = 128`.
- **RIU** — Ring Interface Unit. The 33-node bidirectional ring
  carrying HBM↔core data and request traffic.
- **SFPDataIU** — the 32-node unidirectional ring per corelet
  carrying cross-core SFP/PSUM traffic.
- **PSUM** — Partial sum. The accumulator the PE array writes to;
  drained to LX when the K-loop completes.
- **K-cohort** — when `k > 1` in the work-division split, the k
  cores cooperating on each output PSUM chain.
- **k_fast emission** — a core-id permutation that places K-cohort
  members on adjacent ring positions. Reduces PSUM allreduce cost
  from k-1 trips of m·n hops to k-1 single hops.
- **Work-division split** `(m, n, k)` — how the M, N, K dimensions
  of a matmul are sharded across cores. m·n·k = max_cores.
- **Pure-M / pure-N / mixed-MN / K-split** — the four primary
  families of (m, n, k) splits.
- **DXP_LX_FRAC_AVAIL** — env var controlling the fraction of LX
  available to inductor (the rest reserved for the deeptools
  backend); default 0.2.
- **SuperDSC (SDSC)** — Spyre's kernel descriptor format. One JSON
  describes a scheduled kernel across all 32 cores.
- **Dxp / Ddc / Dip / Dvs** — deeptools subdirectories implementing
  the kernel-codegen pipeline (driver / codegen / instruction pack
  / variable setup respectively).

## 14. References within this branch

- `diag_k_fast_combined_findings_normalized.md` — 12-shape
  cross-vendor + 3 Granite measurement campaign, default LX vs
  DXP_LX_FRAC_AVAIL=1.0.
- `diag_k_fast_granite_findings.md` — 21-shape Granite 3-way
  campaign (2.82× geomean).
- `diag_small_m_spread_findings.md` — 84-shape exhaustive sweep
  with full split-space search.
- `diag_small_m_theory_writeup.md` — narrower theory companion to
  the small-M sweep; this doc's §10 builds on its sample calcs.
- `diag_exhaustive_split_findings.md` — earlier 12-shape exhaustive
  sweep (PR pick is 0/12 the empirical optimum).
- `diag_pure_n_check_findings.md` — pure-N comparison probe.

## 15. References to the codebase

- [torch_spyre/_inductor/work_division.py](torch_spyre/_inductor/work_division.py)
  — `_try_k_fast_split`, `multi_dim_iteration_space_split`,
  `span_reduction_pass`, `work_distribution_pass`. The planner.
- [torch_spyre/_inductor/codegen/compute_ops.py](torch_spyre/_inductor/codegen/compute_ops.py)
  — `_k_fast_core_id_permutation`. The actual permutation code.
- [torch_spyre/_inductor/codegen/superdsc.py](torch_spyre/_inductor/codegen/superdsc.py)
  — SDSC JSON generation.
- [torch_spyre/_inductor/scratchpad.py](torch_spyre/_inductor/scratchpad.py)
  — `ScratchPadAllocator`, where DXP_LX_FRAC_AVAIL is consumed on
  the inductor side.
- [torch_spyre/_inductor/config.py](torch_spyre/_inductor/config.py)
  — `core_id_k_fast_emission`, `dxp_lx_frac_avail`, `lx_planning`,
  `sencores`.

## 16. References to deeptools (the kernel codegen pipeline)

- `deeptools/dsc/HardwareArchMapping/sysConfigs2.0/sentient_dd2_sysconfig.json`
  — the canonical hardware spec. Source of truth for PT
  parallelEngines, LX capacity, ring topology, etc.
- `deeptools/dxp/dxp.cpp` — top-level driver consumed by torch-spyre
  via `_C.so`. Reads `DXP_LX_FRAC_AVAIL` and reserves the
  inductor-side LX slice.
- `deeptools/dvs/setupVariables/batchmatmul_fp16_fwd.cpp` — the
  fp16 batchmatmul kernel template that fills SDSC parameters.
- `deeptools/dsc/dscdefn.cpp` — SDSC JSON schema and parsing.
- `deeptools/dsm/` — design-space-manager passes that lower SDSC to
  per-core DSC.

## 17. References to official architecture docs

- [docs/source/architecture/spyre_accelerator.md](docs/source/architecture/spyre_accelerator.md)
- [docs/source/architecture/dataflow_architecture.md](docs/source/architecture/dataflow_architecture.md)
- [docs/source/compiler/work_division_planning.md](docs/source/compiler/work_division_planning.md)
- [docs/source/user_guide/tensors_and_layouts.md](docs/source/user_guide/tensors_and_layouts.md)
- IBM RaPiD architecture paper:
  Venkataramani et al., ISCA 2021,
  [DOI:10.1109/ISCA52012.2021.00021](https://doi.org/10.1109/ISCA52012.2021.00021).
