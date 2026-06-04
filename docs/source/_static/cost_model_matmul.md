# The Spyre matmul work-division cost model, from first principles

This is a single, self-contained explanation of the **matmul** cost model that
drives work-division on the IBM Spyre AI Accelerator (AIU 1.0) inside the
torch-spyre Inductor backend. It is written so that a reader — human or model —
who has never seen the code can understand, on one pass, *what* the model does,
*why* each term exists, *how* each term is computed, and *how* each coefficient
was calibrated.

Every formula here is traced to its source in
[`torch_spyre/_inductor/work_division.py`](../../../torch_spyre/_inductor/work_division.py).
Where a number is quoted, it is the number actually in the code, not an
approximation; line numbers refer to the merged PR #2407 revision of that file.

> **Scope.** This document covers the **matmul / bmm** cost model only
> (`_matmul_split_cost` and `_cost_model_matmul_planner`). Sibling cost models
> for pointwise and reduction ops exist separately and are out of scope here;
> the matmul model is the only one wired into the merged pass pipeline.

---

## 1. Overview and where it runs

When the Spyre backend compiles a matmul, it must decide **how to spread that
matmul's work across the 32 cores of the accelerator**. This is the
*work-division* problem. For a matmul `[B, M, K] @ [B, K, N]`, the question is:
how many cores should split the batch dimension (`b`), how many should split
the output rows `M` (`m`), how many should split the output columns `N` (`n`),
and how many should split the reduction dimension `K` (`k`)? The product
`b·m·n·k` must not exceed 32, and each factor must evenly divide its dimension.

The cost model answers this by **enumeration plus scoring**. It lists every
feasible `(b, m, n, k)` split, assigns each a predicted kernel time in
microseconds via a closed-form cost function (`_matmul_split_cost`), and picks
the split with the lowest score (`argmin`). That is the entire control flow:
enumerate, score, pick the minimum. There is no search heuristic and no learned
model at inference time — just a small analytic function evaluated over a
divisor grid.

### 1.1 Its own pass, always on, no flag

The matmul cost model is registered as its **own Inductor pre-scheduling pass**,
`cost_model_matmul_division`, and it runs **unconditionally**. There is no
config flag and no `SPYRE_COST_MODEL_MATMUL_PLANNER` environment variable — the
pass always executes. (Earlier drafts of this design were gated behind such a
flag, off by default; that is no longer the case. The merged PR #2407 body text
still describes the old flag-gated behaviour — it is stale on this point; the
shipped code is always-on.) When the cost model declines to re-split an op
(§6.4), that op simply keeps the split the default distributor would have given
it, so "always on" does not mean "always changes the answer."

The three work-division passes run in a fixed order in
[`passes.py`](../../../torch_spyre/_inductor/passes.py):

```text
span_reduction(operations)                              # pass 1
cost_model_ops = cost_model_matmul_division(operations) # pass 2  (this model)
work_distribution(operations, cost_model_ops)           # pass 3
```

- **Pass 1 — `span_reduction`.** Mandatory. Computes the minimum splits needed
  so no tensor's per-core memory span exceeds the 256 MB hardware limit. These
  commitments are written to `op.op_it_space_splits`.
- **Pass 2 — `cost_model_matmul_division`.** This model. For every matmul/bmm
  it computes the split pass 3 *would* pick, hands it to the cost model, and
  commits the cost model's choice when it differs. It returns the list of ops it
  re-split.
- **Pass 3 — `work_distribution`.** The default priority distributor. It skips
  the ops pass 2 already claimed (`preassigned_ops`), so **every op is divided
  by exactly one pass**.

This ordering matters: because pass 2 runs *before* pass 3, when the cost model
inspects `op.op_it_space_splits` it sees only pass 1's span commitments, never
pass 3's distribution. The model recomputes pass 3's default itself (via the
shared `_default_split` helper) so it can compare its pick against the default
and decline a trade-down (§6.4).

The headline result, stated honestly: on the validated scorecard the
cost-model split matches the empirical best split on **~70% of measured
shapes**, with clear prefill wins (QO/KV/MLP at 2.59×/1.76×/2.07× over the
previous heuristic main, §8) and a decode K-split path that fills otherwise-idle
cores. The misses are documented in §10. The rest of this document derives
every piece of the machine.

---

## 2. The hardware, from first principles

Every term in the cost model is a consequence of one of these hardware facts.
We list the facts first; each later cost term points back here.

**Cores and corelets.** The AIU 1.0 has **32 cores**. Each core has **2
corelets**. Work-division decides how to split an op across the 32 cores; the
2-way corelet split underneath is decided by the downstream scheduler
(deeptools), not by this model — but it is why some splits the model treats as
equivalent are not equivalent in practice (see "known limits", §10).

**The PT array (the matmul engine).** Each corelet contains an **8×8 systolic
array** ("PT" = processing-tile array). A systolic array is a pipeline: it takes
several cycles to fill before it produces results at full rate, and the fill
cost is amortized only if you feed it enough rows. The array consumes **8 rows
per pass** (`_PT_ROWS = 8`). If a core is given fewer than ~8×8 = 64 rows of
`M`, the pipeline never fully fills and the core runs below peak. This is the
entire physical basis for the `compute_us` derate and the `target_m_us`
tie-break.

**Peak compute.** Dense fp16 ("DL16") peak is **98.304 TFLOPS** =
`32 cores × 2 corelets × 8 rows × 8 cols × 8 SIMD × 1.5 GHz × 2 FLOPs/MAC`. Note
this is the fp16 figure, **not** the public "300+ TOPS" number (that is INT8).
Dividing peak by 2 (FLOPs→MACs) and by 32 (cores) gives the per-core MAC rate
the cost model uses:

```python
_PEAK_MACS_US_CORE = (98.304e12 / 2 / 32) / 1e6 = 1.536e6  # MACs/us/core
```

**HBM bandwidth: 204.8 GB/s** aggregate (LPDDR5). Crucially this is *shared*
across all cores. Per-core effective bandwidth therefore *falls* as more cores
read the same data — this is the basis of the "cohort penalty." The cost model
charges HBM traffic against this single aggregate number
(`_HBM_BW_GBS = 204.8`; bytes/µs = `204.8 * 1000 = 204800`).

**One bidirectional ring** connects the 32 cores. When a K-split spreads the
reduction across cores, each core computes a partial sum and the partial sums
must be combined by passing them around this ring — one "hop" per step. This
ring traffic is the basis of the `psum_us` term (§4.3).

**The LX 3-way asymmetry (the key insight).** Each core has a 2 MB on-chip LX
scratchpad. During a matmul's inner K-loop it must hold three working sets, which
behave very differently:

| tile | access pattern | residency |
|---|---|---|
| **activations** | each row touched **once**, then discarded | none — small rolling buffer; per-core slice can be many MB and it doesn't matter |
| **weights** | each column touched **K times** (once per reduction step) | **must stay resident** or the kernel re-streams from HBM (~cheap, ~9 µs/MB) |
| **output** | partial sums **accumulate** across the whole K loop | **must fit** or the kernel chunks the output and re-loads weights per chunk (~expensive, ~750 µs/MB) |

This asymmetry — weights reused, activations streamed once — is why
broadcasting *weights* to many cores is cheap (they sit in LX) while
broadcasting *activations* to many cores is expensive (each row is streamed
fresh). The cost model captures part of this and deliberately ignores part of
it; §4.2 and §10 make the boundary explicit. The deep-dive on the LX budget and
the empirical per-MB pressure slopes lives in
[`lx_residency_and_output_pressure.md`](lx_residency_and_output_pressure.md).

**Default dtype is fp16**, 2 bytes (`_DTYPE_BYTES = 2`).

---

## 3. The split space and the (b, m, n, k) convention

For a matmul the iteration space has up to four kinds of dimension. We use a
consistent `(b, m, n, k)` convention throughout — the number of *cores* assigned
to each:

| split factor | splits dimension | meaning |
|---|---|---|
| `b` | batch `B` | one core (group) per batch slice |
| `m` | output rows `M` | one core (group) per row tile |
| `n` | output cols `N` | one core (group) per column tile |
| `k` | reduction `K` | cores cooperate on a partial-sum reduction |

A split is **feasible** iff `b·m·n·k ≤ max_cores` (default 32) and each factor
divides its dimension evenly. The enumeration draws each factor from
`sympy.divisors` of the dim size. The product is the number of cores actually
used; using fewer than 32 is allowed, but the planner will **not** pick a split
that uses *fewer* cores than the default distributor already found — the guard
`math.prod(new_splits.values()) < math.prod(splits.values())` declines such a
trade-down (work_division.py:813).

**How cores map to tiles.** Cores are assigned to a Cartesian grid of work
slices. With `(m=8, n=4)` the 32 cores form an 8×4 grid: 8 row-tiles × 4
column-tiles, each core owning one `(M/8)×(N/4)` output block. The *assignment*
of physical core IDs to grid positions is normally row-major; the
`core_id_k_fast_emission` path (§6.5) changes that assignment for K-splits so
K-collaborators are adjacent on the ring — but it is a pure emission-time
reorder, downstream of and independent from the split this cost model chooses.

**Sticks.** Memory is tiled into 128-byte sticks (64 fp16 elements). The N and K
iteration dimensions are *measured in sticks* inside the iteration space, so the
planner's divisor lists for `n` and `k` are divisors of the *stick count*, not
the element count. The cost model converts back to elements (`N_e = n_sticks ×
64`, `K_e = k_sticks × 64`) so that byte counts and MAC counts are physical
(work_division.py:766-771).

---

## 4. The matmul cost function, term by term

The whole matmul score (`_matmul_split_cost`, work_division.py:653-701) is:

```python
return (compute_us + hbm_us + psum_us + target_m_us) * batch_penalty
```

**Four** terms inside the parentheses are the per-core kernel costs that scale
together with batch; `batch_penalty` multiplies them. There is no additive term
outside the product. (Earlier drafts had a fifth `redistribution_us` additive
term and a sixth `lx_pressure_us` term; both were removed before merge — see the
history in §11.)

Each axis is passed as a `(size, split)` pair, e.g. `m_axis = (M, m)`, so a
dim's size can never be accidentally paired with another dim's split. If
`cores_used = b·m·n·k` is 0 or exceeds `max_cores`, the score is `+inf`
(infeasible). We take each term in turn: the **formula** (verbatim), the
**mechanism** (why it exists), the **computation** (how it is evaluated), and
the **calibration** (how the coefficient was set).

### 4.1 `compute_us` — per-core MAC work, derated for PT-pipeline fill

**Formula** (work_division.py:674-677):

```python
m_t        = M // m if m else 1
pt_passes  = max(1.0, m_t / _PT_ROWS)                  # _PT_ROWS = 8
pt_eff     = min(1.0, (pt_passes / _TARGET_PT_PASSES) ** 0.5)   # _TARGET_PT_PASSES = 8
compute_us = (B * M * N * K / cores_used) / (_PEAK_MACS_US_CORE * pt_eff)
```

**Mechanism.** Each core does `B·M·N·K / (b·m·n·k)` MACs. At full pipeline fill
it runs at the 1.536e6 MACs/µs/core peak. But the 8×8 systolic array needs rows
to amortize its fill latency. The number of "passes" through the array is the
per-core row count `M/m` divided by the 8 rows the array eats per pass. The
pipeline reaches full efficiency at about `_TARGET_PT_PASSES = 8` passes (~64
rows per core). Below that, the array runs below peak and we *derate*.

**Why the derate is a square root, not linear.** This is the single most
important calibration story in the compute term. A naïve linear ramp
`pt_eff = pt_passes / 8` predicts 50% efficiency at 4 passes. Device measurement
showed the real loss at 4 passes was only **10–30%**, i.e. efficiency ~0.7–0.9,
not 0.5. The sqrt form `(pt_passes / 8)**0.5` gives exactly 0.71 at 4 passes and
0.35 at 1 pass — matching the measured shoulder. The linear form was too
pessimistic and would have wrongly pushed the planner toward over-splitting M to
chase passes it didn't need.

**Computation.** A handful of arithmetic ops once `cores_used` is known.
`compute_us` dominates the score for compute-bound shapes (large `K·M / cores`).

### 4.2 `hbm_us` — bytes over bandwidth, with a broadcast cohort penalty

**Formula** (work_division.py:682-684):

```python
bytes_total    = (B*M*K + B*K*N + B*M*N) * _DTYPE_BYTES   # activations + weights + output, fp16
cohort_penalty = max(1.0, max(m, n) / _COHORT_LIMIT)      # _COHORT_LIMIT = 8
hbm_us         = bytes_total / (_HBM_BW_GBS * 1000) * cohort_penalty
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

**Calibration / the symmetry limitation.** The knee at 8 is the cohort threshold
above which contention is observable. But note the deliberate limitation:
`cohort = max(m, n)` is **symmetric in `(m, n)`**, while the real cost is
**asymmetric**. From §2's LX 3-way asymmetry: broadcasting *weights* to many
cores (wide `n`) is cheap because weights stay resident in LX; broadcasting
*activations* to many cores (wide `m`) is expensive because each activation row
is streamed once. At identical nominal cohort, weight-broadcast splits can run
2–3× faster than activation-broadcast splits. The model does not distinguish
them. We tried an asymmetric `(cohort_lhs=m, cohort_rhs=n)` pair with separate
weights, and several other cohort decompositions; under leave-one-out
cross-validation **every cohort-aware variant either matched or regressed the
symmetric form** on the available data — the calibration was degenerate. So the
symmetric form was kept, and the asymmetry is documented as a known limit (§10).

> **A note on shared-weight bmm.** The `B*K*N` weight term over-counts when a bmm
> broadcasts a single 2D weight across the batch (the same weight is charged once
> per batch slice). The planner *declines* that case entirely (§6.4), so the
> over-count never affects a shipped split; fixing it needs weight-rank
> awareness and is tracked as a follow-up.

### 4.3 `psum_us` — K-split reduction hops across the ring

**Formula** (work_division.py:688):

```python
psum_us = max(0, k - 1) * (B * M * N) * _PSUM_PER_ELEM_US   # 1.4e-4
```

**Mechanism.** When `k > 1`, the reduction over `K` is spread across `k` cores,
each producing a partial sum of the full `B·M·N` output. Those partials must be
combined over the ring: `k - 1` reduction steps, each touching every output
element. The cost is the number of output elements times the number of extra
ring hops times a per-element-per-hop coefficient. For `k = 1` the term is
exactly 0 — no K-split, no ring reduction.

**Computation.** Trivial once `k` is known. The term grows with `B·M·N`, so it
is large when the output is large and the K-split is deep — exactly the regime
where K-split should be discouraged.

**Calibration.** `_PSUM_PER_ELEM_US = 1.4e-4` was fit from a 7-shape K-split
sweep (Llama-7B QO/KV/Down, Granite MLP, Mistral MLP, Llama-70B QO, plus a
wide-N synthetic). The implied per-element-per-hop coefficients clustered at
1.1–1.4e-4; the high end was chosen so the term is not under-counted. This term
is what correctly keeps extreme-K shapes (e.g. `K=32768, M=128, N=512`) on
`k=1` even though their raw per-core FLOP counts make a K-split look appealing —
the PSUM cost dominates the FLOP saving.

### 4.4 `target_m_us` — the PT sweet-spot tie-break

**Formula** (work_division.py:692-695):

```python
target_m    = max(_M_MIN, min(max_cores // 2, max(1, M // (_TARGET_PT_PASSES * _PT_ROWS))))
              # _M_MIN = 4, _TARGET_PT_PASSES * _PT_ROWS = 64
target_m_us = abs(math.log2(max(1, m) / target_m)) * _TARGET_M_PENALTY_US   # 50.0 us/log2 step
```

**Mechanism.** Two splits can have nearly identical compute and HBM cost (e.g.
`(8,4)` vs `(4,8)` at M=512, where the cohort is 8 in both cases), and the model
needs a principled tie-break. The PT pipeline runs most efficiently when each
core gets about `8 passes × 8 rows = 64` rows of M. `target_m` is the number of
M-cores that lands per-core M near that 64-row sweet spot, clamped to
`[_M_MIN=4, max_cores/2]`. The penalty grows with the log2-distance of the
candidate's `m` from that target.

**Computation.** `target_m = clamp(4, 16, M/64)`. For M=512 that is
`clamp(4, 16, 8) = 8`, so `m=8` is the sweet spot (zero penalty), and `m=16` and
`m=4` each cost one log2 step = 50 µs.

**Calibration.** 50 µs/log2-step fits the big-M regime well (~48 µs/log2
measured at M≥256). It over-counts the small-M regime by ~4× (real ~12 µs/log2
there). A variant scaled by `compute_us` was tried to fix the small-M bias but
it flipped the QO shape (a real win) into a miss, so it was deferred. Because
this term is only a tie-break — small relative to compute and HBM — the small-M
over-count rarely changes a pick.

### 4.5 `batch_penalty` — the b^1.4 power law

**Formula** (work_division.py:699):

```python
batch_penalty = b ** _BATCH_SPLIT_EXPONENT      # 1.4
```

**Mechanism.** Splitting the batch across cores is *more* expensive per core
than tiling the batch sequentially, because each batch item is independent work
with its own kernel-launch / HBM-banking overhead; a batch-split pays that
overhead `b` times concurrently and contends. For `b=1` (batch iterated
sequentially) the penalty is 1.0 and disappears.

**Calibration.** From a `bmm[8,512,4096,512]` batch sweep, the measured slowdown
`T(b)/T(1)` was 2.56× at b=2, 7.57× at b=4, 19.0× at b=8. Fitting a power law
`b^x` to those points gives x ≈ 1.36, 1.46, 1.42 — hence the rounded **1.4**.
The previously used linear form `1 + 0.6·(b-1)` under-predicted by 3–4× at b=8
and let the planner pick ruinous batch-splits. The 1.4 power law correctly
suppresses batch-split for MoE expert FFNs (the real bmm case), keeping them on
`b=1` with an `m×n` co-split.

---

## 5. Constants reference

All matmul-cost constants live in a single block at work_division.py:638-650.
They carry **bare names** (no `_COST_` prefix). Each is either an AIU hardware
limit or a coefficient fit to measured device kernel times.

| constant | value | line | derivation / meaning |
|---|---|---|---|
| `_PT_ROWS` | `8` | 638 | PT array rows consumed per pass |
| `_TARGET_PT_PASSES` | `8` | 642 | passes to fully fill the PT pipeline (= `8 × _PT_ROWS` = 64 rows/core) |
| `_M_MIN` | `4` | 643 | `_PT_ROWS // 2`; smallest useful m-split (half a PT pass), floor for `target_m` |
| `_PEAK_MACS_US_CORE` | `1.536e6` | 644 | `(98.304e12 / 2 / 32) / 1e6` MACs/µs/core (DL16 peak ÷ FLOPs-per-MAC ÷ 32 cores) |
| `_HBM_BW_GBS` | `204.8` | 645 | LPDDR5 aggregate peak bandwidth; bytes/µs = `× 1000` |
| `_DTYPE_BYTES` | `2` | 646 | fp16 |
| `_PSUM_PER_ELEM_US` | `1.4e-4` | 647 | per output element, per K-split ring-reduction hop |
| `_COHORT_LIMIT` | `8` | 648 | cores sharing a broadcast before it contends for bandwidth (cohort knee) |
| `_BATCH_SPLIT_EXPONENT` | `1.4` | 649 | batch-split cost grows ~ `b ** this` (fit to bmm sweeps) |
| `_TARGET_M_PENALTY_US` | `50.0` | 650 | tie-break weight, per log2 step off the target m-split |
| `MAX_SPAN_BYTES` | `256 MB` | 57 | hardware per-core memory-span limit (used by `span_reduction`, pass 1) |
| max cores | `32` | — | `SENCORES` default; AIU 1.0 core count (`config.sencores`, validated 1–32) |

---

## 6. The planner

`_cost_model_matmul_planner` (work_division.py:704-821) is the function that
turns the scorer into a decision. Its job: override the default split for a
matmul / bmm with the lowest-cost feasible `(b, m, n, k)` per
`_matmul_split_cost`, or return the default unchanged when this planner does not
model the op. The driver `_cost_model_divide_op` (work_division.py:912-974)
wires it into the pass.

### 6.1 Dimension classification

The planner inspects the op's iteration space and output layout to label each
dimension (work_division.py:731-763):

- **N (output columns).** The *stickified* output coordinate dim — exactly one
  is expected (`n_dims`, the dims in `stick_vars`).
- **Row dims.** The remaining output coordinate dims (not stickified).
- **M (output rows).** The row dim that appears in **exactly one** input (the
  LHS / activations), via the `_appears_in_one_input` helper.
- **Batch dims.** Row dims that appear in **both** inputs.
- **K (reduction).** The lone iteration dim that is *not* an output coordinate
  dim. Exactly one is expected.

### 6.2 Enumeration — the divisor cross-product

The planner converts N and K from sticks to elements
(`N_e = n_sticks × elems_per_stick`, etc.), then enumerates the **Cartesian
product** of divisor lists (work_division.py:776-799):

- `b` over the divisors of each batch dim's size,
- `m` over the divisors of `M_e`,
- `n` over the divisors of `n_sticks`,
- `k` over the divisors of `k_sticks`,

keeping only combinations with `b·m·n·k ≤ max_cores`. Each surviving candidate
is scored by `_matmul_split_cost`, and the planner takes the **argmin**.

### 6.3 Commit

The winning `(b, m, n, k)` is written back onto a copy of the default split
dict, mapping each classified dim to its chosen factor (work_division.py:804-810),
and `_cost_model_divide_op` commits it with `apply_splits` only if it differs
from the default; the op is then returned so pass 3 skips it.

### 6.4 Decline conditions — when the planner leaves the default

The planner **declines** (returns `splits` unchanged, so the op keeps the
default distributor's split) in any of these cases:

| condition | where | reason |
|---|---|---|
| op is not a `Reduction` with `reduction_type == BATCH_MATMUL_OP` | 721-723 | not a matmul/bmm — this planner only models matmul |
| a span-committed split is already in place (`committed_splits`) | 725-726 | pass 1 already constrained this op for the 256 MB span limit; don't fight it |
| not exactly one stickified N dim, or no row dims | 736 | can't classify the output the model expects |
| **shared-2D-weight bmm** (`len(m_candidates) != 1`) | 753 | a bmm with a shared 2D weight makes the batch dim "appear in one input" like M, giving two M-candidates; the model isn't weight-rank-aware, so it defers to the default distributor |
| multi-K (`len(reduction) != 1`) | 759-762 | the model prices exactly one reduction dim |
| chosen split would use **fewer cores** than the default | 813-814 | never trade down parallelism the default distributor already found |

The shared-2D-weight decline is the same case that makes the `B*K*N` HBM term
over-count (§4.2): both are downstream of the missing weight-rank awareness, and
both are punted to the follow-up that adds it.

### 6.5 k_fast is *not* part of split selection

Earlier revisions had a `k_fast` planning path inside `work_division.py` that
chose K-splits and a special core mapping together. That planning code is
**fully removed**. What remains is `core_id_k_fast_emission` in
[`superdsc.py`](../../../torch_spyre/_inductor/codegen/superdsc.py) — an
unrelated **emission-time** core-ID reorder, on by default
(`SPYRE_CORE_ID_K_FAST_EMISSION=1`). It fires only when the op is a matmul, the
flag is on, and the planner *already chose* `k > 1`; at `k = 1` the mapping is
identical to the default. It reorders physical cores so K-collaborators are
adjacent on the ring (1 hop per output tile instead of `m·n`), but it **does not
gate or change the split** — that decision belongs entirely to the cost model
above. The code comment in `config.py` states it directly: "The split itself is
chosen by the cost-model planner; this only reorders cores at SDSC emission."

---

## 7. Worked example

Take the **QO prefill** shape from the validation suite:
`[1, 512, 4096] @ [4096, 4096]`, i.e. `B=1, M=512, K=4096, N=4096`. The output
is `512 × 4096`. N and K are 4096 elements = 64 sticks each.

`target_m = clamp(4, 16, M/64) = clamp(4, 16, 8) = 8`, so the tie-break sweet
spot is `m=8`.

The planner enumerates all `(m, n, k)` over `divisors(512) × divisors(64) ×
divisors(64)` with product ≤ 32 and scores each. Representative candidates (µs):

| split `(m,n,k)` | compute | hbm | psum | target_m | **total** |
|---|---:|---:|---:|---:|---:|
| `(32,1,1)` — default distributor's pick | 349.5 | 819.2 | 0.0 | 100.0 | **1268.7** |
| `(16,2,1)` | 247.2 | 409.6 | 0.0 | 50.0 | **706.8** |
| `(2,16,1)` | 174.8 | 409.6 | 0.0 | 100.0 | **684.4** |
| `(8,2,2)` — a K-split | 174.8 | 204.8 | 293.6 | 0.0 | **673.2** |
| `(4,8,1)` | 174.8 | 204.8 | 0.0 | 50.0 | **429.6** |
| **`(8,4,1)` — winner** | **174.8** | **204.8** | **0.0** | **0.0** | **379.6** |

The argmin is **`(8,4,1)`** at 379.6 µs — exactly the split PR #2407 reports for
this shape. Reading the table tells the whole story of the model:

- The `(32,1,1)` default is **3.3× worse**: a pure N-split forces `cohort =
  max(1,32)/8 = 4×` HBM penalty *and* an `m=1` that is 3 log2 steps from
  `target_m=8` (the clamp caps the penalty at 100 µs), and its `m=1` derates
  compute hardest.
- `(8,4,1)` and `(4,8,1)` are **compute- and HBM-identical** (cohort is 8 either
  way). The *only* thing separating them is `target_m_us`: `m=8` is the sweet
  spot (0 µs) while `m=4` is one log2 step away (50 µs). The tie-break alone
  picks the winner — exactly what §4.4 is for.
- The K-split `(8,2,2)` ties on compute and HBM but eats a **293.6 µs PSUM
  penalty** (`(k-1)·B·M·N·1.4e-4 = 1·512·4096·1.4e-4`), so it loses decisively.
  This is why K-split is never chosen when N is wide enough to fill the cores
  (§9).

---

## 8. Validation (PR #2407)

The canonical scorecard is the PR #2407 shape table, collected with
`DXP_LX_FRAC_AVAIL=1`. "improvement" is the speedup of the cost-model split over
the previous heuristic "main" split; the last column is the ratio against the
production `sendnn` backend (lower is better, ≤ 1.0 is a win).

| shape | role | main `(m,n,k)` | cost-model `(m,n,k)` | improvement | vs sendnn |
|---|---|---|---|---:|---:|
| `[1,512,4096]×[4096,4096]` | QO prefill | `(32,1,1)` | `(8,4,1)` | **2.59×** | 0.88 |
| `[1,512,4096]×[4096,1024]` | KV prefill | `(32,1,1)` | `(8,4,1)` | **1.76×** | 1.07 |
| `[1,512,4096]×[4096,12800]` | MLP prefill | `(32,1,1)` | `(8,4,1)` | **2.07×** | 1.20 |
| `[4,1,4096]×[4096,4096]` | QO decode bs4 | `(1,32,1)` | `(4,8,1)` | **1.02×** | 0.96 |
| `[4,1,4096]×[4096,1024]` | KV decode bs4 | `(2,16,1)` | `(4,8,1)` | **1.06×** | 1.00 |
| `[4,1,4096]×[4096,12800]` | MLP decode bs4 | `(1,25,1)` | `(4,8,1)` | **1.20×** | 0.82 |
| `[1,1,4096]×[4096,1024]` | KV decode B=1 | `(1,16,2)` | `(1,16,2)` | **1.00×** | 0.94 |

Reading the table:

- **Prefill is the clear win.** QO/KV/MLP prefill all improve **2.59×/1.76×/2.07×**
  over the heuristic main by replacing its degenerate pure-N `(32,1,1)` split
  with the `(8,4,1)` co-split. Against the production backend, QO prefill is a
  win (0.88), KV is near parity (1.07), and MLP is 1.20 (the residual is a
  kernel-level gap, not a split the model can choose — §10).
- **bs4 decode** flattens through `torch.nn.functional.linear` to `M=4`, so the
  planner sees `B=1, M=4` and does an `m×n` co-split `(4,8,1)`, not a batch
  split. MLP decode bs4 reaches **0.82** vs sendnn.
- **B=1 single-token KV decode** keeps the `(1,16,2)` K-split (the heuristic and
  the cost model agree here) and lands **0.94** vs sendnn — the K-split fills
  cores N alone cannot (§9).

---

## 9. K-split / small-M behavior

K-split (`k > 1`) is the most misunderstood part of the model. The summary:
**K-split is never picked for prefill, and its real home is narrow-N
single-token decode, where it is the only way to fill the array.**

**Why K-split is never picked for prefill.** Consider any prefill matmul with N
wide enough to occupy the cores. An N-split and a K-split *do the same total
compute* — both spread the same `M·N·K` MACs over the same cores. But the
K-split *adds* the `psum_us` ring-reduction cost (§4.3) that the N-split does
not pay. So for the *same* core count, an N-split strictly dominates a K-split
whenever N is wide enough to absorb the cores. The QO worked example (§7) shows
this numerically: `(8,2,2)` ties `(8,4,1)` on compute and HBM but loses on the
293.6 µs PSUM term.

**Small M is HBM-bound, so K-split barely moves the needle there either.** At
`M=1` (Q-proj `[1,4096,4096]`) the roofline says `t_compute ≈ 0.34 µs` against
`t_hbm ≈ 164 µs` — three orders of magnitude below the roofline knee. K-split
divides the already-trivial compute term across more cores while leaving the HBM
floor untouched (every K-shard still streams its slice of LHS+RHS, and the
output is unchanged). The win on a 164 µs kernel is ≤ 2%, and the top ~12 splits
cluster within 7 µs of each other — exactly the ±2% noise band the empirical
sweeps saw.

**The real home: narrow-N single-token decode.** Now consider decode: `B=1,
M=1`, and a *narrow* N (e.g. 1024 or 512). With M=1 there is no M-split, and N
is too narrow to occupy 32 cores by itself (`N/sticks < 32`). The cores an
N-split cannot fill would sit **idle**. A K-split puts those idle cores to work
on the reduction. The trade is favorable precisely because `psum_us` is *tiny*
here — it scales with `B·M·N`, and with `M=1` and small N that product is small,
so the ring cost of the extra reduction is negligible against the win of filling
otherwise-idle cores. The KV decode B=1 row in §8 (`[1,1,4096]×[4096,1024]` →
`(1,16,2)` at 0.94 vs sendnn) is exactly this case.

**A K-split sweet spot the model cannot see.** On `[8,14336,4096]`, an
empirical `(4,1,7)` at 28 cores beat `(4,1,8)` at 32 cores by ~25%, because
`14336/7 = 2048 = 2^11` is a native PT inner tile while `14336/8 = 1792` forces
tail handling on every pass. This is per-corelet tile alignment, not
parallelism, and it is **not generalizable** — it fires only when one specific
`k` yields a power-of-two per-core K. The closed-form cost model has no
kernel-template simulator and deliberately does not chase it; a generic
"prefer power-of-two per-core K" bias would mis-fire everywhere else. The PSUM
term is doing its job: small enough not to forbid K-split where it might help,
large enough to break ties against it where it doesn't.

---

## 10. Known limits

The matmul cost model lands the empirical best split on **37 of 53 measured
shapes (~70%)**. The misses are not random — they share one fingerprint,
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
This is a real mechanism but fires only in a narrow `K·(N/n)` band, and the
one-shape kludge for it (the removed `lx_pressure_us`, §11) broke adjacent-N
picks — so it is accepted as a documented miss.

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
evaluated under leave-one-out cross-validation (LOO-CV). **No variant beat
baseline:** smooth coefficients collapsed to zero (the optimizer disabled the
new term to avoid regressing neighbors), and the decision tree hit 40/53
in-sample but collapsed to 25/53 LOO — catastrophic overfitting on 53 points.
The honest conclusion is a closed-form ceiling of ~70% on this dataset; the
remaining 30% lives in discrete scheduler decisions, costs ~5–15% per affected
kernel and well under 1% end-to-end, and would require either a downstream
`predict_schedule`-style simulator or far more measurement data to close.

**Work-division vs kernel-level.** Keep the boundary clear: this cost model only
chooses the *work split*. Fusion decisions, kernel-template choice, clSplit, and
single-kernel SDPA efficiency are *downstream* and not in scope.

---

## 11. Removed terms (history)

Two terms that earlier drafts carried were **removed before the PR #2407 merge**.
They are gone from the finalized code; this section records why, so the history
is not relitigated.

### 11.1 `redistribution_us` — the removed fusion-bundle tie-break

Earlier the score had an additive term *outside* the batch-penalty product:

```python
# REMOVED — not in the merged code
redistribution_us = B * M * N * _DTYPE_BYTES * <per-byte coef>
# total_us = (...) * batch_penalty + redistribution_us
```

It was meant to charge a matmul a small penalty when it shared a fusion bundle
with a non-matmul partner (e.g. `silu(linear(x))`) and the candidate split
diverged from the bundle's default layout, so the planner would only rewrite a
bundled matmul's split when the kernel savings beat the bundle penalty. Device
measurement of fused `silu(linear)` bundles found the *actual* reshuffle cost is
essentially 0; the original `1e-4` coefficient was ~100× too large and was
*blocking* otherwise-beneficial bundled rewrites. It was lowered to `1e-6`, and
then dropped entirely — at `1e-6` it was a no-op tie-break that added a degree
of freedom and a fusion-bundle dependency for no measurable benefit. The
finalized score has no additive term.

### 11.2 `lx_pressure_us` — the removed weight-overflow term

An earlier model had a sixth term:

```python
# REMOVED — not in the merged code
per_core_weights = K * (N / n) * 2
lx_pressure_us   = max(0, per_core_weights - 2MB) * <per-byte coef>
```

It was introduced to capture a measured ~120 µs win for `(m=4, n=8)` over
`(m=8, n=4)` on the Granite MLP shape (`M=512, K=4096, N=12800`). It was
**removed**, and the reasoning is the model's clearest lesson about overfitting:

- The term was named for *weight* overflow but the real cost it tracked was
  *output* overflow. The LX 3-way asymmetry (§2) means weight overflow is cheap
  (~9 µs/MB — weights re-stream from HBM) while output overflow is expensive
  (~750 µs/MB — the kernel chunks the output and re-loads weights per chunk).
  The two correlate only at the calibration shape because both per-core weights
  and per-core output grow with `(N/n)`.
- A clean K-sweep up to 16 MB per-core weights showed **zero detectable**
  per-byte weight cost — the term was fitting kernel-template artifacts, not
  physics.
- The coefficient is **wrong at adjacent N**: at N=16384 and N=20480 the term
  predicts `(4,8)` wins, but `(8,4)` empirically wins by ~40%.

Removing it costs ~120 µs (~10%) on that one Granite-MLP kernel — well under 1%
of end-to-end Granite latency — in exchange for not mis-ranking the much larger
neighborhood of N values. A physically correct *output*-pressure term was
considered, but per-core output stays small (64–400 KB) at every shape we
actually compile, so it would never fire on a real workload. It was not added.

The full LX-residency mechanism — the 3-way budget, why output overflow is ~80×
more expensive than weight overflow, and the corner-stress sweep that derived
the per-MB slopes — is documented in
[`lx_residency_and_output_pressure.md`](lx_residency_and_output_pressure.md).
That residual mechanism is the source of the N=1536 and danger-band misses in
§10.
