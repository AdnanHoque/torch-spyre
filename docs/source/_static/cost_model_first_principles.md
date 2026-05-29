# The Spyre matmul work-division cost model, from first principles

This is a single, self-contained explanation of the cost model that drives
work-division on the IBM Spyre AI Accelerator (AIU 1.0) inside the torch-spyre
Inductor backend. It is written so that a reader — human or model — who has
never seen the code can understand, on one pass, *what* the model does, *why*
each term exists, *how* each term is computed, and *how* each coefficient was
calibrated. Every formula here is traced to its source in
[`torch_spyre/_inductor/work_division.py`](../../../torch_spyre/_inductor/work_division.py)
and [`torch_spyre/_inductor/config.py`](../../../torch_spyre/_inductor/config.py).
Where a number is quoted, it is the number actually in the code, not an
approximation.

---

## 1. Overview

When the Spyre backend compiles an operation, it must decide **how to spread
that operation's work across the 32 cores of the accelerator**. This is the
*work-division* problem. For a matmul `[B, M, K] @ [B, K, N]`, the question is:
how many cores should split the batch dimension (`b`), how many should split
the output rows `M` (`m`), how many should split the output columns `N` (`n`),
and how many should split the reduction dimension `K` (`k`)? The product
`b·m·n·k` must not exceed 32, and each factor must evenly divide its dimension.

The cost model answers this question by **enumeration plus scoring**. It lists
every feasible `(b, m, n, k)` split, assigns each a predicted kernel time in
microseconds via a closed-form cost function, and picks the split with the
lowest score (`argmin`). That is the entire control flow: enumerate, score,
pick the minimum. There is no search heuristic, no learned model at inference
time — just a small analytic function evaluated over a divisor grid.

The model is **unified across three op classes**. A single dispatcher,
`_classify_op` → `_score_split`, routes each operation to one of three scorers:

- `_matmul_split_cost` for batched matmul / linear / bmm,
- `_pointwise_split_cost` for elementwise ops (add, silu, mul, ...),
- `_reduction_split_cost` for simple reductions (sum, mean, max, amax, ...).

Each scorer is gated by its own config flag, all **off by default**
(`SPYRE_COST_MODEL_MATMUL_PLANNER`, `..._POINTWISE_PLANNER`,
`..._REDUCTION_PLANNER`). When a flag is off, that op class uses the existing
priority-based heuristic distributor.

The headline result, stated honestly:

- **Matmul is a real win.** On the validated scorecard the cost-model split
  matches the empirical best split on ~70% of measured shapes, with clear
  prefill wins (e.g. QO bs=1 at 0.87× and MLP bs=4 at 0.85× against the
  production backend) and several other matmul shapes at parity or below.
  Single-token decode gains a K-split path that, by filling otherwise-idle
  cores, is up to ~1.7× faster than the same matmul with no K-split — which
  shows up as a 0.78–0.93× ratio against the production backend (§9.2). The
  single largest gain is **MLP bs=1 at N=12800 — the worst shape in the suite —
  improving from 1.51 to 1.20** against the production backend purely by
  enabling the planner (the heuristic's pick was far worse). It still loses to
  the backend at 1.20× (a kernel-level gap below ~1.2, not work-division), and
  the model's `(8,4)` is itself ~10% off the empirical optimum `(4,8)` — the
  wide MLP prefill band at `N ∈ [12800, 15872]` is a documented split-choice
  miss where the `(m,n)` winner flips non-monotonically and is not landable
  without regressing the adjacent priority shape N=12288 (§10).
- **Pointwise and reduction are parity, not wins.** The two sibling scorers are
  validated as non-regressive against the heuristic, but they produce no
  measurable speedup, because the heuristic already makes good work-division
  choices for those op classes (largest-dim split for pointwise; reduction-dim
  fallback for reductions). They exist so the *one* cost model covers the whole
  graph uniformly, not because they beat the heuristic.

The rest of this document derives every piece of that machine.

---

## 2. The hardware, from first principles

Every term in the cost model is a consequence of one of these hardware facts.
We list the facts first, then in later sections each cost term points back here.

**Cores and corelets.** The AIU 1.0 has **32 cores**. Each core has **2
corelets**. Work-division decides how to split an op across the 32 cores; the
2-way corelet split underneath is decided by the downstream scheduler
(deeptools), not by this model — but it is why some splits the model treats as
equivalent are not equivalent in practice (see "known limits").

**The PT array (the matmul engine).** Each corelet contains an **8×8 systolic
array** ("PT" = processing-tile array). A systolic array is a pipeline: it
takes several cycles to fill before it produces results at full rate, and the
fill cost is amortized only if you feed it enough rows. The array consumes
**8 rows per pass** (`_PT_ROWS = 8`). If a core is given fewer than ~8×8 = 64
rows of `M`, the pipeline never fully fills and the core runs below peak. This
is the entire physical basis for the `compute_us` derate and the `target_m_us`
tie-break.

**Peak compute.** Dense fp16 ("DL16") peak is **98.304 TFLOPS**. This is
`32 cores × 2 corelets × 8 rows × 8 cols × 8 SIMD × 1.5 GHz × 2 FLOPs/MAC`.
Note this is the fp16 figure, **not** the public "300+ TOPS" number (that is
INT8). Dividing peak by 2 (FLOPs→MACs) and by 32 (cores) gives the per-core
MAC rate the cost model uses:

```
_COST_PEAK_MACS_US_CORE = (98.304e12 / 2 / 32) / 1e6 = 1.536e6  MACs/us/core
```

**XRF (weight register file): 64 KB/corelet.** That is **64 fp16 weight-sticks**
(a stick = 64 fp16 elements = 128 bytes). XRF caps how much weight can be held
resident at the matmul array's input. This is the upstream reason weights have
a residency budget at all.

**LX scratchpad: 2 MB/core.** The on-chip scratchpad holds the three working
sets of a matmul simultaneously: activations, weights, and the output
accumulator. The asymmetry between these three (below) is what makes some
splits cheaper than others — and is also where the model's single biggest
historical mistake lived (the removed `lx_pressure_us` term, §4.7).

**HBM bandwidth: 204.8 GB/s** aggregate (LPDDR5). Crucially this is *shared*
across all cores. Per-core effective bandwidth therefore *falls* as more cores
read the same data — this is the basis of the "cohort penalty". The cost model
charges HBM traffic against this single aggregate number:

```
_COST_HBM_BW_GBS = 204.8        # bytes/us = 204.8 * 1000 = 204800
```

**One bidirectional ring** connects the 32 cores. When a K-split spreads the
reduction across cores, each core computes a partial sum and the partial sums
must be combined by passing them around this ring — one "hop" per step. This
ring traffic is the basis of the `psum_us` term and the entire rationale for
the `k_fast` core-id remapping (§11).

**The LX 3-way asymmetry (the key insight).** During a matmul's inner K-loop:

| tile | access pattern | residency |
|---|---|---|
| **activations** | each row touched **once**, then discarded | none — small rolling buffer; per-core slice can be many MB and it doesn't matter |
| **weights** | each column touched **K times** (once per reduction step) | **must stay resident** or the kernel re-streams from HBM |
| **output** | partial sums **accumulate** across the whole K loop | **must fit** or the kernel chunks the output and re-loads weights per chunk |

This asymmetry — weights reused, activations streamed once — is why
broadcasting *weights* to many cores is cheap (they sit in LX) while
broadcasting *activations* to many cores is expensive (each row is streamed
fresh). The cost model captures part of this and deliberately ignores part of
it; §4.2 and §10 make the boundary explicit.

**Default dtype is fp16**, 2 bytes (`_COST_DTYPE_BYTES = 2`).

---

## 3. The split space

For a matmul, the iteration space has up to four kinds of dimension. We use a
consistent `(b, m, n, k)` convention throughout (the number of *cores* assigned
to each):

| split factor | splits dimension | meaning |
|---|---|---|
| `b` | batch `B` | one core (group) per batch slice |
| `m` | output rows `M` | one core (group) per row tile |
| `n` | output cols `N` | one core (group) per column tile |
| `k` | reduction `K` | cores cooperate on a partial-sum reduction |

A split is **feasible** iff `b·m·n·k ≤ 32` and each factor divides its
dimension evenly (the enumeration draws each factor from `sympy.divisors` of
the dim size — see `_enumerate_splits`). The product is the number of cores
actually used; using fewer than 32 is allowed but the planner will not pick a
split that uses *fewer* cores than the caller's default (`_matmul_cost_planner`
guards `math.prod(new_splits) < math.prod(splits)`).

**How cores map to tiles.** Cores are assigned to a Cartesian grid of work
slices. With `(m=8, n=4)` the 32 cores form an 8×4 grid: 8 row-tiles × 4
column-tiles, each core owning one `(M/8)×(N/4)` output block. The
*assignment* of physical core IDs to grid positions is normally row-major; the
`k_fast` path (§11) changes that assignment for K-splits so K-collaborators are
adjacent on the ring.

**Sticks.** Memory is tiled into 128-byte sticks (64 fp16 elements). The N and
K iteration dimensions are *measured in sticks* inside the iteration space, so
the planner's divisor lists for N and K are divisors of the *stick count*, not
the element count. The cost model converts back to elements (`N_e`, `K_e`) so
that byte counts and FLOPs are physical. This stick-granularity is also why
splitting the innermost ("stick") dimension is penalized for pointwise ops
(§5): a split that doesn't land on a stick boundary fragments the access.

**SDSC dim labels.** Inside the SDSC kernel descriptor the same dimensions wear
different names. You will see these in planner logs and split dumps:

| SDSC label | math dim | `(b,m,n,k)` factor |
|---|---|---|
| `mb` | M (rows) | `m` |
| `out` | N (cols) | `n` |
| `in` | K (reduction) | `k` |
| `x` | B (batch) | `b` |

So a planner pick logged as `(mb=8, out=4, in=1)` is exactly `(m=8, n=4, k=1)`:
8 cores split rows, 4 split columns, no K-split.

---

## 4. The matmul cost function, term by term

The whole matmul score (`_matmul_split_cost`) is:

```
total_us = (compute_us + hbm_us + psum_us + target_m_us) * batch_penalty
           + redistribution_us
```

The four terms inside the parentheses are the per-core kernel costs that scale
together with batch; `batch_penalty` multiplies them; `redistribution_us` is a
flat additive fusion tie-break added outside. We take each in turn: the
**formula** (verbatim), the **mechanism** (why it exists), the **computation**
(how it is evaluated), and the **calibration** (how the coefficient was set).

### 4.1 `compute_us` — per-core MAC work, derated for PT-pipeline fill

**Formula** (from source):

```python
m_t            = M // m if m else 1
pt_passes      = max(1.0, m_t / _PT_ROWS)              # _PT_ROWS = 8
pt_eff         = min(1.0, (pt_passes / _TARGET_PT_PASSES) ** 0.5)   # _TARGET_PT_PASSES = 8
effective_peak = _COST_PEAK_MACS_US_CORE * pt_eff      # 1.536e6 * pt_eff
compute_us     = (B * M * N * K / cores_used) / effective_peak
```

**Mechanism.** Each core does `B·M·N·K / (b·m·n·k)` MACs. At full pipeline fill
it runs at the 1.536e6 MACs/us/core peak. But the 8×8 systolic array needs rows
to amortize its fill latency. The number of "passes" through the array is the
per-core row count `M/m` divided by the 8 rows the array eats per pass. The
pipeline reaches full efficiency at about `_TARGET_PT_PASSES = 8` passes (i.e.
~64 rows per core). Below that, the array runs below peak and we *derate*.

**Why the derate is a square root, not linear.** This is the single most
important calibration story in the compute term. A naïve linear ramp
`pt_eff = pt_passes / 8` predicts 50% efficiency at 4 passes. Device
measurement showed the real loss at 4 passes was only **10–30%**, i.e.
efficiency ~0.7–0.9, not 0.5. The sqrt form `(pt_passes / 8)**0.5` gives
exactly 0.71 at 4 passes and 0.35 at 1 pass — matching the measured shoulder.
The linear form was too pessimistic and would have wrongly pushed the planner
toward over-splitting M to chase passes it didn't need.

**Computation.** `cores_used = b·m·n·k`; if it is 0 or `> max_cores` the score
is `+inf` (infeasible). Otherwise the formula above is a handful of arithmetic
ops. `compute_us` dominates the score for compute-bound shapes (large
`K·M / cores`).

### 4.2 `hbm_us` — bytes over bandwidth, with a broadcast cohort penalty

**Formula:**

```python
bytes_total    = (B*M*K + B*K*N + B*M*N) * _COST_DTYPE_BYTES   # activations + weights + output, fp16
cohort         = max(m, n)
cohort_penalty = max(1.0, cohort / _COST_COHORT_LIMIT)         # _COST_COHORT_LIMIT = 8
hbm_us         = bytes_total / (_COST_HBM_BW_GBS * 1000) * cohort_penalty
```

**Mechanism.** The total bytes touched are the three matrices (read both inputs,
write the output) at 2 bytes each. The HBM bus is shared, so the wall-clock to
move those bytes is `bytes / 204800` µs at best. The **cohort penalty** captures
contention: when a row or column has to be *broadcast* to many cores, those
cores contend for the shared bus. A "cohort" is the number of cores a given
operand is broadcast to. Below a knee of 8 cores the bus absorbs it; above 8,
the extra fan-out scales the cost linearly.

**Computation.** `cohort = max(m, n)` and `cohort_penalty = max(1, cohort/8)`.
For `(8,4)` and `(4,8)` the cohort is 8 in both cases → penalty 1.0; for
`(16,2)` it is 16 → penalty 2.0.

**Calibration / the symmetry limitation.** The knee at 8 is the cohort
threshold above which contention is observable. But note the deliberate
limitation: `cohort = max(m, n)` is **symmetric in `(m, n)`**, while the real
cost is **asymmetric**. From §2's LX 3-way asymmetry: broadcasting *weights* to
many cores (wide `n`) is cheap because weights stay resident in LX; broadcasting
*activations* to many cores (wide `m`) is expensive because each activation row
is streamed once. At identical nominal cohort, weight-broadcast splits can run
2–3× faster than activation-broadcast splits. The model does not distinguish
them. We tried an asymmetric `(cohort_lhs=m, cohort_rhs=n)` pair with separate
weights, and several other cohort decompositions; under leave-one-out
cross-validation **every cohort-aware variant either matched or regressed the
symmetric form** on the available data — the calibration was degenerate. So the
symmetric form was kept, and the asymmetry is documented as a known limit (§10).

### 4.3 `psum_us` — K-split reduction hops across the ring

**Formula:**

```python
psum_us = max(0, k - 1) * (B * M * N) * _COST_PSUM_PER_ELEM_US   # 1.4e-4
```

**Mechanism.** When `k > 1`, the reduction over `K` is spread across `k` cores,
each producing a partial sum of the full `B·M·N` output. Those partials must be
combined over the ring: `k - 1` reduction steps, each touching every output
element. The cost is the number of output elements times the number of extra
ring hops times a per-element-per-hop coefficient. For `k = 1` the term is
exactly 0 — no K-split, no ring reduction.

**Computation.** Trivial once `k` is known. The term grows with `B·M·N`, so it
is large when the output is large and the K-split is deep — which is exactly the
regime where K-split should be discouraged.

**Calibration.** `_COST_PSUM_PER_ELEM_US = 1.4e-4` was fit from a 7-shape
K-split sweep (Llama-7B QO/KV/Down, Granite MLP, Mistral MLP, Llama-70B QO,
plus a wide-N synthetic). The implied per-element-per-hop coefficients across
those shapes clustered at 1.1–1.4e-4; the high end was chosen so the term is not
under-counted. This term is what correctly keeps extreme-K shapes
(e.g. `K=32768, M=128, N=512`) on `k=1` even though their raw per-core FLOP
counts make a K-split look appealing — the PSUM cost dominates the FLOP saving.

### 4.4 `target_m_us` — the PT sweet-spot tie-break

**Formula:**

```python
target_m    = max(_M_MIN, min(max_cores // 2, max(1, M // (_TARGET_PT_PASSES * _PT_ROWS))))
              # _M_MIN = 4, _TARGET_PT_PASSES*_PT_ROWS = 64
m_dist      = abs(math.log2(max(1, m) / target_m))
target_m_us = m_dist * _COST_TARGET_M_PENALTY_US                # 50.0 us per log2 step
```

**Mechanism.** Two splits can have nearly identical compute and HBM cost (e.g.
`(8,4)` vs `(16,2)` at M=512), and the model needs a principled tie-break. The
PT pipeline runs most efficiently when each core gets about `8 passes × 8 rows`
= 64 rows of M. `target_m` is the number of M-cores that lands per-core M near
that 64-row sweet spot, clamped to `[_M_MIN=4, max_cores/2]`. The penalty grows
with the log2-distance of the candidate's `m` from that target.

**Computation.** `target_m = clamp(4, 16, M/64)`. For M=512 that is
`clamp(4,16,8) = 8`, so `m=8` is the sweet spot (zero penalty), `m=16` and
`m=4` each cost one log2 step = 50 µs.

**Calibration.** 50 µs/log2-step fits the big-M regime well (~48 µs/log2
measured at M≥256). It over-counts the small-M regime by ~4× (real ~12 µs/log2
there). A variant scaled by `compute_us` was tried to fix the small-M bias but
it flipped the QO shape (a real win) into a miss, so it was deferred. Because
this term is only a tie-break — it is small relative to compute and HBM — the
small-M over-count rarely changes a pick.

### 4.5 `batch_penalty` — the b^1.4 power law

**Formula:**

```python
batch_penalty = b ** _COST_BATCH_SPLIT_EXPONENT      # 1.4
```

**Mechanism.** Splitting the batch across cores is *more* expensive per core
than tiling the batch sequentially, because each batch item is independent work
with its own kernel-launch / HBM-banking overhead; a batch-split pays that
overhead `b` times concurrently and contends. For `b=1` (batch iterated
sequentially) the penalty is 1.0 and disappears.

**Calibration.** From a `bmm[8,512,4096,512]` batch sweep, the measured
slowdown `T(b)/T(1)` was 2.56× at b=2, 7.57× at b=4, 19.0× at b=8. Fitting a
power law `b^x` to those points gives x ≈ 1.36, 1.46, 1.42 — hence the rounded
**1.4**. The previously used linear form `1 + 0.6·(b-1)` under-predicted by 3–4×
at b=8 and let the planner pick ruinous batch-splits. The 1.4 power law
correctly suppresses batch-split for MoE expert FFNs (the real bmm case),
keeping them on `b=1` with an `m×n` co-split.

### 4.6 `redistribution_us` — the fusion-bundle tie-break

**Formula:**

```python
redistribution_us = B * M * N * _COST_DTYPE_BYTES * _COST_REDISTRIBUTION_US_PER_BYTE   # 1e-6 us/byte
# charged only when the matmul is in a fusion bundle with a non-matmul op
# AND the candidate split diverges from the bundle's default layout
```

**Mechanism.** When a matmul shares a fusion bundle with a non-matmul partner
(e.g. `silu(linear(x))`), a split that differs from the partner's layout would,
in principle, force the matmul output to be reshuffled across cores before the
partner consumes it. The model charges this so it only rewrites a bundled
matmul's split when the kernel savings beat the bundle penalty. The
`_matmuls_fused_with_nonmatmul` helper identifies which matmul outputs are in
such bundles; `_score_split` adds the penalty only when `fused` and the
candidate `diverges_from_default`.

**Calibration.** Device measurement of fused `silu(linear)` bundles found the
*actual* reshuffle cost is essentially 0 — the original coefficient `1e-4` was
~100× too large and was *blocking* otherwise-beneficial bundled matmul
rewrites. It was lowered to `1e-6`, which keeps it as a gentle tie-break rather
than a hard gate.

### 4.7 The removed `lx_pressure_us` term — why it was dropped

An earlier model had a sixth term:

```
per_core_weights = K * (N / n) * 2
lx_pressure_us   = max(0, per_core_weights - 2MB) * 5e-6
```

It was introduced to capture a measured ~120 µs win for `(m=4, n=8)` over
`(m=8, n=4)` on the Granite MLP shape (`M=512, K=4096, N=12800`). It was
**removed**, and understanding why is important because it is the model's
clearest lesson about overfitting.

The story (from the corner-stress sweep): the term was named for *weight*
overflow but the real cost it tracked was *output* overflow. The LX 3-way
asymmetry (§2) means weight overflow is cheap (~9 µs/MB — weights re-stream from
HBM, amortized over the K loop) while output overflow is expensive (~750 µs/MB —
the kernel must *chunk* the output and **re-load the per-core weights for every
chunk**). The two correlate at the calibration shape because both per-core
weights `K·(N/n)` and per-core output `(M/m)·(N/n)` grow with `(N/n)`, so
penalizing weight overflow accidentally tracked the true output cost there. But
the proxy is a kludge that breaks away from the calibration regime:

- A clean K-sweep up to 16 MB per-core weights showed **zero detectable
  per-byte weight cost** — the term was fitting kernel-template artifacts, not
  physics.
- The coefficient is **wrong at adjacent N**: at N=16384 and N=20480 the term
  predicts `(4,8)` wins, but `(8,4)` empirically wins by ~40% there.
- It was tuned to capture exactly **one** N value (12800), where the kernel
  template happens to favor `(4,8)`.

Removing it costs ~120 µs (~10%) on that single Granite-MLP kernel — well under
1% of end-to-end Granite latency — in exchange for theoretical cleanness and
not mis-ranking the much larger neighborhood of N values. A physically correct
*output*-pressure term was considered, but per-core output stays small (64–400
KB) at every shape we actually compile, so the term would never fire on a real
workload. It was not added. (See §10 for the residual N-flip band this leaves.)

---

## 5. The pointwise cost function

`_pointwise_split_cost` scores elementwise ops (add, mul, silu, ...). These have
no reduction dimension; every iteration dim is an output dim.

**Formula:**

```python
bytes_total = (sum(s * f for s, f in zip(input_sizes, input_fanouts)) + out_size) * 2
hbm_us      = bytes_total / (_COST_HBM_BW_GBS * 1000)
per_core_elements = out_size / cores_used
compute_us  = per_core_elements / _COST_PEAK_ELEMENTS_US_CORE     # 1.76e3 elem/us/core
stick_penalty_us = (_COST_STICK_FRAG_US_PER_BYTE * (stick_split - 1)
                    * out_size * 2)                               # 4.5e-7 us/byte; 0 if stick_split==1
batch_penalty = batch_split ** _COST_BATCH_SPLIT_EXPONENT         # b ** 1.4
cost = (max(compute_us, hbm_us) + stick_penalty_us) * batch_penalty + redistribution_us
```

**Roofline (max, not sum).** A pointwise kernel spends its time either moving
bytes or doing SFP (elementwise) math, whichever is larger — never both
serially. So compute and HBM combine as `max(compute_us, hbm_us)`, a roofline.
Once the per-core slice is large the op is HBM-bound; once it shrinks past the
SFP roofline it becomes compute-bound. `compute_us` is a *floor*: it stops the
planner from believing that splitting a tiny slice across all 32 cores is free.

**Broadcast-aware HBM (the fanout).** This is the key difference from a naïve
byte count. Each input is charged `numel × fanout` bytes, where `fanout` is the
product of splits over the dims that input *lacks*. A *partitioned* input (one
that carries the split dim) has `fanout = 1` — each core reads its own
`1/cores` slice and the slices sum to `numel`. A *broadcast* input (one that is
missing the split dim) has `fanout > 1` — every core in the fan-out must read
the whole thing, and the `(fanout − 1)` extra reads are the cohort tax. The
output is always partitioned, charged once. This is the same broadcast-cohort
idea as the matmul term, but expressed per-input rather than via `max(m,n)`.

**Stick-fragmentation penalty.** The innermost ("stick") dim should not be
split, because a split that doesn't land on a 64-element stick boundary causes
partial-stick reads and HBM bank conflicts. Without this term the roofline is
*flat* for non-broadcast inputs (`max(compute, hbm)` is identical for all
splits that move the same bytes), so the planner would arbitrarily pick the
last-enumerated candidate — often a stick-dim split — and regress 100–1000 µs.
The penalty scales `(stick_split − 1) × out_bytes`; `4.5e-7 us/byte` was tuned
to make the planner reliably avoid stick-dim splits.

**Batch penalty.** Same `b^1.4` form as matmul, applied to "batch-like" dims —
output dims that are neither the stick dim nor the largest non-stick dim. The
largest non-stick dim is the "main work" axis (analogous to M); splitting it
parallelizes cleanly. Splitting a *different* outer dim instead carries the same
per-batch-unit overhead as the matmul batch-split. Without this term the
roofline ties on shapes like `add([4,2048,4096], [4,1,4096])` — where a B-split
shrinks the broadcast input's fanout and looks free — and regresses ~1.6 ms.

**Why it ends at parity.** Notice what these terms collectively *do*: they steer
the planner toward splitting the largest non-stick dim, away from the stick dim,
and away from batch-like dims. That is exactly what the existing heuristic
already does (it splits the largest dim by priority and naturally avoids the
stick dim). So the cost model and the heuristic converge on the same pick for
ordinary pointwise shapes. The scorer's value is that it *also* gets the
broadcast-fanout cases right via the same machine, and it is validated
non-regressive — but it produces no measurable speedup over the heuristic. It is
parity by construction.

---

## 6. The reduction cost function

`_reduction_split_cost` scores simple reductions (the `_SIMPLE_REDUCE_TYPES`:
`sum, mean, max, amax, min, amin, exx2`). Topk, welford, prod, argmax fall back
to the default planner.

**Formula:**

```python
cores       = prod(d_splits) * prod(r_splits)
bytes_total = (elems_in + elems_out) * 2
compute_us  = (elems_in / cores) / _COST_REDUCE_ELEM_PER_US_CORE   # 1.2e4 elem/us/core
hbm_us      = bytes_total / (_COST_HBM_BW_GBS * 1000)
r_prod      = prod(r_splits)
psum_us     = max(0, r_prod - 1) * elems_out * _COST_PSUM_PER_ELEM_US   # 1.4e-4
cost        = max(compute_us, hbm_us) + psum_us + redistribution_us
```

Here `d_splits` are the core counts on output (kept) dimensions and `r_splits`
are core counts on reduced dimensions.

**Roofline.** Same as pointwise: a reduction kernel is either compute-bound or
HBM-bound, so compute and HBM combine as `max`. `psum_us` and
`redistribution_us` add *on top* because they happen after the local compute.

**No cohort term — and why this differs from matmul.** This is the most
important design point of the reduction scorer. A pure reduction has **no
broadcast**: each core reads its `1/cores` slice of the input and writes its
slice of the output. Nothing is broadcast to a cohort of cores, so there is no
contention to charge. The cohort/broadcast term in the matmul cost model exists
specifically because matmul *broadcasts weights (and activations) across the
cohort dimension*; a reduction has no such operand. An earlier draft wrongly
inherited a cohort term from the matmul model; it was removed because it modeled
a contention that physically does not occur for reductions.

**PSUM for reduction-dim splits.** When `r_splits` introduce a cross-core split
of a *reduced* dimension, each core produces a partial reduction and the
partials must be combined over the ring — exactly the same ring-hop mechanism
as the matmul K-split, reusing the same `1.4e-4` coefficient. The term scales
with `elems_out` (the number of partial results to combine) and `(r_prod − 1)`
(the extra hops).

**Calibration of the reduce rate.** `_COST_REDUCE_ELEM_PER_US_CORE = 1.2e4` was
calibrated 2026-05-28 from device timings of forced-split `sum` reductions on
`[1,32,512,4096]` and `[1,512,4096]` across `cores ∈ {1,2,4}`. The closed-form
fit `compute_us = elems_in / cores / K` yielded `K = 1.20e4 elem/us/core` with
R² = 0.97. (Softmax fits a much lower ~1.4e3 because its compound
amax+exp+sum+div lowering does roughly 8× more per-element work; the simple-
reduce constant tracks the simple-reduction target, not softmax.)

**Why it ends at parity.** Like pointwise, the reduction scorer steers toward
the same answer the heuristic already gives: split the output (kept) dims first,
fall back to the most-splittable reduction dim for leftover cores. The roofline
plus the PSUM-on-reduction-split penalty reproduce the heuristic's reduction-dim
fallback ordering. Validated non-regressive; no measurable win.

---

## 7. The unified architecture

There is **one** cost model with three op-class scorers, not three independent
models. The flow:

```
work_distribution_pass(op)
  → op_class = _classify_op(op)          # MATMUL | POINTWISE | REDUCTION | None
  → if op_class enabled by its flag:
        _cost_model_planner(op, ..., op_class, ...)
            → dispatches to _matmul/_pointwise/_reduction_cost_planner
                 → _identify_*_dims(op)            # extract M/N/K/batch or dims/fanouts
                 → _enumerate_splits(dims, divisor_lists, max_cores)
                 → for each candidate: _score_split(op_class, candidate, ctx)
                 → pick argmin
```

`_classify_op` reads the IR node: a `Reduction` with `reduction_type ==
BATCH_MATMUL_OP` is MATMUL; a `Reduction` whose type is in
`_SIMPLE_REDUCE_TYPES` is REDUCTION; a `Pointwise` is POINTWISE; everything else
returns `None` and uses the heuristic.

`_score_split` is the single dispatch point: given an `OpClass` and a candidate
split dict, it unpacks the class-specific context (`ctx`) and calls the matching
`_*_split_cost`. Each class enumerates its own divisor grid:

- **MATMUL** enumerates `(b..., m, n, k)` over divisors of `(B, M, N-sticks,
  K-sticks)`, scores with `_matmul_split_cost`, and adds `redistribution_us`
  only for fused-bundle candidates that diverge from the default.
- **POINTWISE** enumerates per-output-dim splits over divisors of each dim,
  computes per-input fanout, and breaks ties by `(cost, -cores_used)` — on a
  cost tie it prefers *more* cores (more parallelism, more downstream-fusion
  room). It also skips data-motion (ReStickify) pointwise ops, where the SFP-
  rate model does not apply.
- **REDUCTION** enumerates `(d_splits..., r_splits...)` and scores with the
  no-cohort roofline form.

Each scorer is independently flag-gated (`cost_model_matmul_planner`,
`cost_model_pointwise_planner`, `cost_model_reduction_planner`), all default
off. You can turn matmul on while leaving the siblings off — which is the
recommended configuration, given matmul is the only win.

**Why one model, not three?** Because the three op classes share the *same
physics and the same constants*: the same HBM bandwidth (204.8 GB/s), the same
PSUM ring-hop coefficient (1.4e-4), the same `b^1.4` batch penalty, the same
roofline reasoning. Factoring them into one model with three scorers means a
calibration improvement to a shared constant improves all three, and the graph
is covered uniformly by one mechanism instead of three drifting heuristics.

---

## 8. Methodology — how every term was calibrated

Every coefficient in §4–§6 was set by device measurement, not guessed. The
methodology:

**The force-split device harness.** A `force_split_timing.py` harness
monkey-patches the planner to inject a *chosen* `(b, m, n, k)` split at the IR
level for a named op, then runs the rest of the pipeline (parse, layout,
clSplit, schedule, emit) unmodified and collects device-side `kernel_ms` from
one warm run plus N timed runs. This is the crucial property: it measures **what
the scheduler actually emits** for a candidate, not what we *think* it emits, so
the fitted coefficients absorb the real downstream behavior. (For decode-style
forced splits use `force_split_dbg.py`; the older `force_split_timing_2d.py`
no-ops on the current pipeline.)

**Sweeping `(m, n, k)` per shape.** For each shape the harness probes: the two
co-split candidates around full occupancy `(8,4)` and `(4,8)`; pure splits along
the larger dim `(16,2)`, `(2,16)`, `(32,1)`, `(1,32)`; K-splits when K is large
(`(8,2,2)`, `(4,4,2)`, `(4,2,4)`, `(4,1,8)`); and odd-cohort candidates when N
has non-power-of-two factors (e.g. `(2,15)` for N=2880). The empirical winner is
the lowest measured `kernel_ms`, with a 2% margin treated as a tie.

**The ~53-shape validation set.** The shapes come from the 1H-2026 priority
models (GPT-OSS 20B, Granite-4 dense and MoE, Mistral-Small 22B, Qwen-2.5-7B,
Llama-3.1-8B, Ministral-8B, Mistral-Nemo 12B) plus M-sweep, K-split, and
N-landscape synthetics, deduplicated to 53 distinct `(B,M,K,N)` shapes, each
with a dict `{(b,m,n,k): kernel_ms}` of measured times.

**Leave-one-out cross-validation as the overfitting guard.** Every candidate
term and coefficient was scored not just on in-sample fit but under LOO-CV:
re-fit the coefficient on 52 shapes, predict the 53rd, repeat. A term only ships
if it does not regress LOO. This is what killed the non-linear variants (§10)
and the `lx_pressure_us` term (§4.7): they improved in-sample fit but did not
generalize.

**Same-session A/B against the production backend.** The validation scorecard
(§9) was run in a single clean serial session at one commit, with no device
contention and a fresh Inductor cache, comparing torch-spyre (`tsp`) against the
production `sendnn` backend op-for-op. (The shared accelerator must not be
contended during timing, so timing runs are serial, not parallel.)

**LX config held constant.** All sweeps were run at a fixed LX fraction
(`DXP_LX_FRAC_AVAIL`), so LX-allocation differences never confounded the
work-division comparison. The cost model is calibrated for that fixed LX budget;
changing it would shift the absolute kernel times (though not, in general, the
relative split ranking).

---

## 9. Validation results

Measured at commit `c8c49f3`, all three planners on, single clean serial
session, fresh cache, no contention. `kernel_ms` is the sum of non-Memset /
Memcpy kernel-bundle mean time over the profiled runs (the production-backend
convention). `ratio = tsp / sendnn`; **lower is better**, ≤ 1.0 is a win.

### 9.1 Matmul prefill / batched

| shape | split `(m,n,k)` | tsp ms | sendnn ms | ratio | note |
|---|---|---|---|---|---|
| QO bs1 `[1,512,4096]×[4096,4096]` | 8,4,1 | 0.324 | 0.372 | **0.871** | win |
| QO bs4 `[4,1,4096]×[4096,4096]` | 4,8,1 | 0.240 | 0.249 | **0.964** | slight win (bs4 flattens to M=4) |
| KV bs1 `[1,512,4096]×[4096,1024]` | 8,4,1 | 0.115 | 0.108 | 1.065 | near parity (narrow-N prefill) |
| KV bs4 `[4,1,4096]×[4096,1024]` | 4,8,1 | 0.062 | 0.061 | 1.016 | parity (was 1.05) |
| MLP bs1 `[1,512,4096]×[4096,12800]` | 8,4,1 | 1.149 | 0.960 | 1.197 | **1.51→1.20 vs heuristic** (the headline gain); still > 1.0 (kernel-level) and `(8,4)` is ~10% off the optimal `(4,8)`; §10 |
| MLP bs4 `[4,1,4096]×[4096,12800]` | 4,8,1 | 0.761 | 0.900 | **0.846** | clear win (was 0.97) |

The clearest prefill wins are **QO bs=1** (`(mb=8, out=4)` co-split, 0.87×) and
**MLP bs=4** (`(mb=4, out=8)`, 0.85×). Note that
`torch.nn.functional.linear` flattens the bs4 `[4,1,4096]` input to `M=4`, so
the planner sees `B=1, M=4` and does an `m×n` co-split, not a batch split.

**MLP bs=1 at N=12800 is simultaneously the headline gain and a documented
miss — both are true and worth stating precisely:**

- *Versus the heuristic (the prior production behaviour):* a clear win. The
  cost model takes this shape — the worst ratio in the suite — from **1.51 to
  1.20** (tsp `kernel_ms` 1.45 → 1.15), a ~22% reduction, by choosing the
  `(mb=8, out=4)` co-split instead of the heuristic's pick. This is the single
  largest improvement the cost model delivers.
- *Versus the production backend:* still a loss at 1.20×. The residual below
  ~1.2 is **kernel-level** (the backend's N-tiling / PT-column packing), not a
  split the cost model can choose — see §10.
- *Versus the empirical-optimal split:* a work-division miss. At this N the
  optimum is `(mb=4, out=8)`, ~10% faster than the model's `(mb=8, out=4)`.
  N=12800 sits at the low edge of the `N ∈ [12800, 15872]` band where the
  `(m,n)` winner flips non-monotonically — a discrete kernel-template effect a
  smooth closed-form cost cannot capture (§10). Closing it would regress the
  adjacent priority shape N=12288, which wants `(8,4)` by +57%, so it is not
  landable. The miss costs ~10% on one kernel, well under 1% end-to-end.

### 9.2 Single-token decode (the K-split home)

| model | shape | split | tsp ms | sendnn ms | ratio | K-split fired |
|---|---|---|---|---|---|---|
| Llama-3.1-8B / Ministral-8B / Granite-4-H KV | `[1,1,4096]×[4096,1024]` | n=16, k=2 | 0.062 | 0.067 | **0.925** | yes |
| Qwen2.5-7B KV | `[1,1,3584]×[3584,512]` | n=8, k=4 | 0.028 | 0.036 | **0.778** | yes |
| QO decode control | `[1,1,4096]×[4096,4096]` | n=32, k=1 | 0.247 | 0.246 | 1.004 | no |

Decode K-split fires **exactly as the heuristic predicts** (§11): on narrow-N
single-token decode, K-split fills idle cores and beats the production backend
(0.93× and 0.78×). On the QO control, N=4096 alone fills all 32 cores via the
N-split, so no K-split is taken and the result is parity — validating that
K-split is only chosen when N is too narrow to fill the array.

### 9.3 Pointwise / reduction — parity, not wins

The reduction planner fires on softmax and rms_norm (split `mb=32, out=1`) and
the pointwise planner fires on elementwise bundles, both producing valid splits.
They are validated non-regressive but show **no measurable speedup** over the
heuristic, because the heuristic already makes good largest-dim choices for
these ops. (Separately, on decomposed softmax/rms_norm the `kernel_ms` metric
*understates* tsp because tsp fuses each into a single SDSC bundle while the
production backend decomposes rms_norm into two kernels; on full-device
`spyre_ms` tsp is actually 2.3–2.5× faster on those two ops. That is a
fusion/kernel effect, not a work-division effect, and is out of scope for this
cost model. The one genuine kernel-level loss is fused SDPA attention, ~1.23×
on `spyre_ms` — again a single-kernel efficiency issue, not work-division.)

---

## 10. Known limits

The matmul cost model lands the empirical best split on **37 of 53 measured
shapes (~70%)**. The misses are not random — they share one fingerprint and were
characterized across four independent investigations.

**The closed-form ceiling (~70%).** Every miss falls in a single failure mode:
at `M=512, K=4096`, the `(4,8)` split wins at a *scattered* set of N values
(notably a "danger band" roughly `N ∈ [12800, 15872]`, plus a tight pocket at
`N=1536`) while `(8,4)` wins at all the *other* N values. The model picks
`(8,4)` across this whole neighborhood and is right most of the time — which is
why the hit rate is 70%, not coin-flip.

**The non-monotone (m,n) band.** The winner is *non-monotone* in N: `(8,4)` wins
at N=2560, 4096, 10752, 11264, then `(4,8)` wins through 12800–15872, then
`(8,4)` wins again at 16384+. A smooth closed-form cost function in `(B,M,K,N)`
cannot reproduce a non-monotone winner flip, because the flip is not smooth
physics — it is discrete *kernel-template selection* downstream of
work-division.

**The N=1536 pocket is mechanistically understood** (the one exception): at
`(8,4)` the per-core weight slice is `4096·384·2 = 3 MB`, over the 2 MB LX cap;
at `(4,8)` it is `1.5 MB`, under the cap. A K-scaling experiment confirmed the
flip appears only at K=4096, exactly where the LX cap separates the two
candidates (at K=2048 both fit and tie; at K=8192 neither fits and they tie).
This is a real mechanism but fires only in a narrow `K·(N/n)` band, and a
one-shape kludge for it (the removed `lx_pressure_us`) broke adjacent-N picks —
so it is accepted as a documented miss.

**The wider danger band is diffuse.** Four investigations ruled out the obvious
mechanisms for `N ∈ [12800, 15872]`: it is *not* uniform LX overflow (both
candidates overflow heavily, yet `(8,4)` wins at N=16384, 20480); *not* HMI
channel alignment (the flipping N values share no alignment signature); *not*
pure clSplit=1 (counter-examples at adjacent N); *not* PT tile alignment (no
N-dependent flip); *not* XRF/ring bandwidth (magnitude too small by
back-of-envelope). A useful negative result: the production backend's own
internal cost function (`getCoreEqPerf`) is *also* cohort-blind and HBM-blind —
it gets these shapes right only because its downstream *scheduler* encodes the
missing physics implicitly through kernel-template choice, which is not visible
to a work-division-level cost model.

**Non-power-of-2 N is handled** by the divisor enumeration: odd-cohort
candidates like `(2,15)` for N=2880 are enumerated and the cohort penalty ranks
them correctly (validated on GPT-OSS odd-N shapes).

**Several misses are empirical ties.** A number of the 16 misses are within the
~1–2% tie band — the "miss" is a label artifact of picking a single winner from
two statistically indistinguishable candidates.

**Non-linear variants were tried and rejected.** Ten smooth-term variants
(sigmoid LX-fit, saturating cohort, asymmetric cohort, roofline re-stream,
log-cohort, Gaussian band, ...) plus a decision-tree baseline were fit and
evaluated under LOO-CV. **No variant beat baseline:** smooth coefficients
collapsed to zero (the optimizer disabled the new term to avoid regressing
neighbors), and the decision tree hit 40/53 in-sample but collapsed to 25/53 LOO
— catastrophic overfitting on 53 points. The honest conclusion is a closed-form
ceiling of ~70% on this dataset; the remaining 30% lives in discrete scheduler
decisions, costs ~5–15% per affected kernel and well under 1% end-to-end, and
would require either a downstream `predict_schedule`-style simulator or far more
measurement data to close.

**Work-division vs kernel-level.** Finally, keep the boundary clear: this cost
model only chooses the *work split*. Fusion decisions, kernel-template choice,
clSplit, and single-kernel SDPA efficiency are *downstream* and not in scope.
The softmax/rms_norm `kernel_ms` "losses" in §9.3 are fusion/measurement
artifacts (tsp actually wins on device time), and the SDPA loss is a
single-kernel efficiency gap — none of these are work-division and none are
addressable by this model.

---

## 11. Split-K

K-split (`k > 1`) is the most misunderstood part of the model, so it gets its
own section. The summary: **K-split is never picked for prefill, and its real
home is narrow-N single-token decode, where it is the only way to fill the
array.**

**Why K-split is never picked for prefill.** Consider any prefill matmul with N
large enough to occupy the cores. An N-split and a K-split *do the same total
compute* — both spread the same `M·N·K` MACs over the same number of cores. But
the K-split *adds* the `psum_us` ring-reduction cost (§4.3) that the N-split does
not pay. So for the *same* core count, an N-split strictly dominates a K-split
whenever N is wide enough to absorb the cores. The cost model encodes exactly
this: `psum_us = (k−1)·B·M·N·1.4e-4` is a pure penalty with no offsetting
benefit when N can fill the array. This is why the QO decode control in §9.2
(N=4096) takes `n=32, k=1` — N alone fills the 32 cores, so K-split only adds
cost.

**The real home: narrow-N single-token decode.** Now consider decode: `B=1,
M=1`, and a *narrow* N (e.g. 1024 or 512). With M=1 there is no M-split, and N
is too narrow to occupy 32 cores by itself (`N/sticks < 32`). The cores that an
N-split cannot fill would sit **idle**. A K-split puts those idle cores to work
on the reduction. The arithmetic-intensity argument: at M=1 the matmul is deeply
*memory-bound* (one activation row, huge weight matrix), the array is starved,
and the only lever to use more cores is K. The trade is favorable precisely
because the `psum_us` term is *tiny* here — it scales with `B·M·N`, and with
`M=1` and small N that product is small, so the ring cost of the extra reduction
is negligible against the win of filling otherwise-idle cores. Measured (§9.2):
KV decode `[4096,1024]` → `n=16, k=2` at 0.93×; Qwen `[3584,512]` → `n=8, k=4`
at 0.78× — the narrower N forces a deeper K-split and yields the biggest decode
win, up to ~1.7× over a no-K-split baseline. The heuristic already picks these.

**The `k_fast` emission layer.** When the planner does pick a K-split, the
physical core-id assignment matters for the ring reduction. The default
(row-major) assignment scatters the K-collaborators (cores that share a fixed
`(m,n)` tile but differ in `k`) across the ring, so combining their partial sums
takes up to `m·n` hops. The `k_fast` mapping
(`_k_fast_core_to_slice_mapping` / `_should_use_k_fast_mapping` in
`torch_spyre/_inductor/codegen/superdsc.py`) treats the K dim as the
*fastest-varying* axis along `core_id`,
so K-collaborators land on **adjacent** ring positions and the PSUM reduction
traverses **1 hop per output tile instead of m·n**. It fires only when all three
hold: the op is a matmul, the `SPYRE_CORE_ID_K_FAST_EMISSION` flag is on, and the
planner chose `k > 1` (at `k=1` the mapping is identical to the default).

**Why the k_fast layer has a structural no-home beyond decode.** This is the
honest scoping of split-K. The emission layer minimizes ring hops — but ring
hops only matter when the K-split actually fires, and the K-split only fires in
the narrow-N decode regime above. In that regime the kernel is *memory-bound*
(M=1): wall-clock is set by HBM weight traffic, not by the PSUM ring. So
minimizing ring hops, while correct, optimizes a term that is not on the
critical path in the only regime where K-split is chosen. K-split implies
memory-bound implies ring-hop minimization is second-order. The `k_fast` layer
is retained because it is cheap and correct and shaves the small PSUM term, but
it does not unlock a new performance regime on its own — the win in §9.2 comes
from *filling idle cores* via K, not from the hop reduction.

---

## Appendix: constant reference

| constant | value | meaning |
|---|---|---|
| `_PT_ROWS` | 8 | PT array rows consumed per pass |
| `_TARGET_PT_PASSES` | 8 | passes to fully fill the PT pipeline (~64 rows/core) |
| `_M_MIN` | 4 | smallest useful m-split (half a PT pass), floor for `target_m` |
| `_COST_PEAK_MACS_US_CORE` | 1.536e6 | per-core peak MACs/us = 98.304e12/2/32/1e6 |
| `_COST_PEAK_ELEMENTS_US_CORE` | 1.76e3 | per-core SFP elementwise rate (silu sweep asymptote) |
| `_COST_REDUCE_ELEM_PER_US_CORE` | 1.2e4 | per-core simple-reduce rate (sum sweep, R²=0.97) |
| `_COST_HBM_BW_GBS` | 204.8 | HBM aggregate bandwidth (bytes/us = ×1000) |
| `_COST_DTYPE_BYTES` | 2 | fp16 |
| `_COST_PSUM_PER_ELEM_US` | 1.4e-4 | per output element per K/reduction ring hop |
| `_COST_COHORT_LIMIT` | 8 | broadcast-cohort knee (contention above this) |
| `_COST_BATCH_SPLIT_EXPONENT` | 1.4 | batch-split power law `b^1.4` |
| `_COST_TARGET_M_PENALTY_US` | 50.0 | tie-break, per log2 step from target m |
| `_COST_REDISTRIBUTION_US_PER_BYTE` | 1e-6 | fused-bundle reshuffle tie-break |
| `_COST_STICK_FRAG_US_PER_BYTE` | 4.5e-7 | pointwise stick-dim split tax |
| `MAX_SPAN_BYTES` | 256 MB | hardware per-core memory-span limit (span-reduction pass) |
| max cores | 32 | `SENCORES` default; AIU 1.0 core count |
