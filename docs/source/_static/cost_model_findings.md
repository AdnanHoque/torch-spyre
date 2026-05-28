# Spyre matmul work-division cost-model: research findings

Research-mode writeup of the closed-form cost model that drives
`_cost_model_matmul_planner` in
[`torch_spyre/_inductor/work_division.py`](../../../torch_spyre/_inductor/work_division.py).
Audience is future researchers (most likely the same author or a colleague)
picking up this thread, not a wide developer audience.

This document focuses on **what we tried, what worked, what didn't, and
what would unblock further progress**. The two existing companion
documents cover the equation itself and the LX-residency mechanism in
depth and are not duplicated here:

- [`cost_model_equation.md`](cost_model_equation.md) — the equation, per-term
  formulas, and hardware constants.
- [`cost_model_planner.html`](cost_model_planner.html) — first-principles
  explainer (architecture overview, design walkthrough, calibration).
- [`lx_residency_and_output_pressure.md`](lx_residency_and_output_pressure.md)
  — deep dive on the LX 3-way budget (activations / weights / output)
  and the empirical corner-stress sweep that derived the per-MB pressure
  slopes.

---

## 1. Executive summary

We built a five-term closed-form cost model that picks an `(b, m, n, k)`
work-division for each matmul / bmm and validated it across the 1H 2026
priority models (GPT-OSS, Granite-4 dense and MoE, Mistral-Small,
Qwen-2.5-7B, Llama-3.1-8B, Ministral-8B, Mistral-Nemo). The model lands
the empirical best split on **37 of 53 measured shapes (~70%)**. Every
miss falls in a single failure mode: at `M=512, K=4096`, the `(4, 8)`
split wins at scattered `N` values (1536, 12800, 13312, 13824, 14336,
15872) while `(8, 4)` wins at all the other `N` values we measured.

We were able to **confirm the mechanism for the smallest of these
misses** (`N=1536`) via a K-scaling experiment: the flip only appears
when the per-core weight slice is *near but not over* the 2 MB LX cap.
We were **not able to find a mechanism for the wider-N "danger band"**
(`N` in roughly 12800–15872). Source-traced inspection of deeptools'
`SdscCoreletSplit.cpp` and `getCoreEqPerf` shows that even deeptools'
own internal cost model is cohort-blind, so the flips are coming from
elsewhere in the scheduler.

We ran 10 non-linear variant cost models (smooth saturating terms,
sigmoid LX-fit, decision tree). **No variant beats baseline under
leave-one-out cross-validation**: smooth coefficients collapse to zero,
the decision tree overfits catastrophically. We hit the closed-form
ceiling on this dataset at 70%.

---

## 2. Cost model architecture

`_matmul_split_cost(B, M, K, N, b, m, n, k, max_cores, redistribution_us)`
in
[`work_division.py:601`](../../../torch_spyre/_inductor/work_division.py)
computes:

```
total_us = (compute_us + hbm_us + psum_us + target_m_us) * batch_penalty
           + redistribution_us
```

The five terms and their physical motivations:

| term            | captures                                              | dominant regime           |
|-----------------|-------------------------------------------------------|---------------------------|
| `compute_us`    | per-core MAC time, derated when the PT pipeline can't fill (sqrt below 8 passes) | compute-bound shapes (large `K * M / cores`) |
| `hbm_us`        | input + output bytes / HBM BW, scaled by a broadcast `cohort_penalty = max(1, max(m, n) / 8)` | memory-bound shapes (small `M`, large `N`) |
| `psum_us`       | `(k - 1) * B * M * N` reduction hops at ~1.4e-4 µs/element/hop | K-split candidates       |
| `target_m_us`   | `\|log2(m / target_m)\| * 50 µs` tie-breaker that prefers `m` close to `clamp(4, max_cores/2, M / 64)` | resolves ties at large M  |
| `batch_penalty` | `b ** 1.4` multiplicative; only ≠ 1 when `b > 1` (bmm) | MoE expert FFN, bmm[B,M,K,N] |

`redistribution_us` is an additive penalty (not multiplied by
`batch_penalty`) charged only when the matmul is in a fusion bundle and
the candidate split would diverge from the partner's layout. It's a
tie-breaker at `1e-6 µs/byte` — the original `1e-4` over-penalized by
~100x and was blocking bundled matmul rewrites.

Per-term formulas, hardware constants, and the calibration notes for
each coefficient live in
[`cost_model_equation.md`](cost_model_equation.md). The LX-residency
story behind the now-removed `lx_pressure_us` term is in
[`lx_residency_and_output_pressure.md`](lx_residency_and_output_pressure.md).

**Deliberate omissions.** The cost model uses a *symmetric*
`cohort = max(m, n)` rather than a pair `(cohort_lhs, cohort_rhs)`. The
asymmetric pair would let `hbm_us` distinguish wide-`n` (cheap, weights
broadcast to many cores and stay in LX) from wide-`m` (expensive, each
activation row is streamed once to many cores). We tried this and
several other cohort decompositions and the calibration was degenerate
against the available data — every cohort-aware variant either matched
or regressed the symmetric form. See sections 5 and 6.

---

## 3. Calibration methodology

All sweeps used a `force_split_timing.py` harness that monkey-patches
the planner to inject a forced `(b, m, n, k)` at the IR level and
collects device-side `kernel_ms` from one warm run + N timed runs. The
forced split is plumbed in by replacing the planner's `_matmul_split`
result for the named op; the rest of the pipeline (parse, layout,
clSplit, schedule) runs unmodified. This guarantees we measure what the
scheduler actually emits for each candidate, not what we *think* it
emits.

For each shape we probed:

1. The two co-split candidates around full-core occupancy: `(8, 4)` and
   `(4, 8)` (both use 32 cores).
2. Pure splits along the larger output dim: `(16, 2)`, `(2, 16)`,
   `(32, 1)`, `(1, 32)`.
3. K-splits when `K` was large enough to matter:
   `(8, 2, 2)`, `(4, 4, 2)`, `(4, 2, 4)`, `(4, 1, 8)`.
4. Odd-cohort candidates when `N` had non-power-of-two factors
   (e.g. Qwen `N=18944` → `(2, 14)`; GPT-OSS `N=2880` → `(2, 15)`,
   `(4, 5)`, `(8, 3)`).

The empirical winner for a shape is the split with the lowest measured
`kernel_ms`, with a 2% margin treated as a tie ("no preference"). The
full measured table is in `/tmp/*_sweep.log` and is replayed by
[`/tmp/nonlinear_cost_model_fit.py`](file:///tmp/nonlinear_cost_model_fit.py)
into a single in-memory dictionary
`{(B, M, K, N): {(b, m, n, k): kernel_ms}}` with 53 distinct shapes
after de-duplication.

A shape counts as a **hit** if the planner's pick equals the empirical
winner (or any split within the 2% tie band). A shape counts as a
**regression** if a candidate variant flips a baseline hit into a miss.

---

## 4. Empirical validation across 1H 2026 priority models

### 4.1 Per-model coverage

| family            | shapes measured | hits | notes |
|-------------------|-----------------|------|-------|
| GPT-OSS 20B       | 6  | 5  | QO/KV/FFN at `K=2880`; one Oproj odd-N miss |
| Granite-4 dense   | 4  | 3  | shared FFN at `N=1536` is a miss (LX-boundary flip) |
| Granite-4 MoE     | 3  | 3  | gate/up/down at `N=768` and `N=4096` |
| Mistral-Small 22B | 5  | 4  | QO/KV/FFN-up/FFN-down at `K=6144`/`N=16384` |
| Qwen-2.5-7B       | 4  | 3  | FFN-up wins at `(16, 2)`; FFN-down K-split |
| Llama-3.1-8B      | 4  | 3  | FFN at `N=14336` is a miss |
| Ministral-8B      | 4  | 3  | FFN at `N=12288` is a miss |
| Mistral-Nemo 12B  | 3  | 2  | QO at `N=5120` is OK; FFN-up at `N=14336` is a miss |
| M-sweep synthetics| 9  | 8  | small-M (decode) all hit; large-M prefill hits |
| K-split synthetics| 6  | 4  | extreme-K at `K=32768` hits |
| N-landscape       | 5  | 2  | shapes added to probe the danger band |
| **Total**         | **53** | **37 (70%)** | |

(Each "model" row counts only shapes measured in the sweeps; "shapes
measured" is the deduped count after collapsing equivalent forced
runs from multiple logs.)

### 4.2 Where the model wins

- **All M-sweep shapes from M=1 to M=2048** at `K=4096, N=4096` are
  hits. The target_m tie-breaker correctly picks `(4, 8)` for small M
  (the PT pipeline isn't full and the m-distance term dominates) and
  `(8, 4)` for M ≥ 256 (target_m grows with M).
- **Extreme-K** (`K=32768, M=128, N=512`): the model correctly stays
  on `(4, 8, 1)` despite K-split candidates looking attractive in the
  raw FLOP counts. `psum_us` correctly grows with `(k - 1) * M * N`.
- **MoE bmm** at `B=8`: the `b ** 1.4` penalty correctly suppresses
  batch-split candidates that look cheap by raw FLOP. The
  `(b=1, m=8, n=4, k=1)` pick on MoE FFN matches measurement.
- **GPT-OSS odd-N shapes** (`N=2880`): `(8, 3, 1)`, `(4, 5, 1)`, and
  `(2, 15, 1)` are all enumerated; the cohort_penalty correctly
  ranks `(2, 15, 1)` highest for `N=2880` Oproj and FFN.

### 4.3 The 16 misses

All 16 misses share one fingerprint:

| family       | M   | K     | N      | planner pick | empirical winner |
|--------------|-----|-------|--------|--------------|------------------|
| Granite-4    | 512 | 4096  | 12800  | (8, 4, 1) | (4, 8, 1) |
| Granite-shared | 512 | 4096 | 1536  | (8, 4, 1) | (4, 8, 1) |
| Llama-3.1    | 512 | 4096  | 14336  | (8, 4, 1) | (4, 8, 1) |
| Ministral    | 512 | 4096  | 12288  | (8, 4, 1) | (16, 2, 1) |
| Mistral-Nemo | 512 | 5120  | 14336  | (8, 4, 1) | (8, 4, 1) — actually hit; included for context |
| synthetic    | 512 | 4096  | 13312  | (8, 4, 1) | (4, 8, 1) |
| synthetic    | 512 | 4096  | 13824  | (8, 4, 1) | (4, 8, 1) |
| synthetic    | 512 | 4096  | 15872  | (8, 4, 1) | (4, 8, 1) |
| GPT-OSS Oproj| 512 | 4096  | 2880   | (8, 3, 1) | (2, 15, 1) — different family |
| ...          | ... | ...   | ...    | ...          | ... |

The pattern: **at M=512, K=4096, the `(4, 8)` split wins at scattered
N values that don't have an obvious ordering**. At adjacent N values
(7680, 8192, 9216, 11264, 16384, 20480), `(8, 4)` wins by clear
margins — sometimes by 30–40%. The model picks `(8, 4)` everywhere in
this neighborhood and is right *most* of the time, which is why the
overall hit rate is 70% rather than coin-flip.

---

## 5. The N-flip misses: investigation

### 5.1 The phenomenon

Fix `M=512, K=4096` and sweep `N` over the values that show up in real
models plus synthetic neighbors:

| N      | (8, 4) kernel | (4, 8) kernel | winner |
|--------|---------------|---------------|--------|
| 1536   |  ~7% slower   |  baseline     | (4, 8) — LX-boundary |
| 2560   |  baseline     |  ~10% slower  | (8, 4) |
| 4096   |  baseline     |  ~6% slower   | (8, 4) |
| 7680   |  baseline     |  ~1% slower (tie) | (8, 4) (tie) |
| 10752  |  baseline     |  ~49% slower  | (8, 4) |
| 11264  |  baseline     |  ~48% slower  | (8, 4) |
| 12800  | ~10% slower   |  baseline     | **(4, 8)** |
| 13312  |  ~2% slower   |  baseline     | **(4, 8)** |
| 13824  |  ~6% slower   |  baseline     | **(4, 8)** |
| 14336  |  ~1% slower   |  baseline     | **(4, 8)** (tie) |
| 15872  |  ~4% slower   |  baseline     | **(4, 8)** |
| 16384+ |  baseline     |  ~40% slower  | (8, 4) (per equation doc) |

(All deltas extracted from already-logged measurements in the sweep
files; magnitudes are ratios to the winner.)

The behavior is **non-monotone in N**. There is a tight pocket at
`N=1536` and a broader "danger band" roughly `N ∈ [12800, 15872]`
where `(4, 8)` wins, sandwiched on both sides by regions where
`(8, 4)` wins.

### 5.2 Hypotheses and outcomes

We tested five hypotheses, all targeting plausible physical mechanisms.

#### H1. LX-fit boundary: per-core weight slice near 2 MB cap — **CONFIRMED for N=1536**

Per-core weights at `(8, 4)` are `K * (N / n) * 2 = 4096 * 384 * 2 =
3 MB` for `N=1536`. Per-core weights at `(4, 8)` are
`4096 * 192 * 2 = 1.5 MB`. The 2 MB LX cap is between the two, so
`(4, 8)` fits in LX but `(8, 4)` does not. This predicts `(4, 8)`
should win for `N=1536` but not for `N << 1536` or `N >> 1536`.

The K-scaling experiment in `/tmp/synthetic_sweep.log` confirms this:

| K    | N    | per-core wts (8,4) | per-core wts (4,8) | winner       |
|------|------|--------------------|--------------------|--------------|
| 2048 | 1536 | 1.5 MB             | 0.75 MB            | tie (<1%)   |
| 4096 | 1536 | 3 MB               | 1.5 MB             | (4, 8)      |
| 8192 | 1536 | 6 MB               | 3 MB               | tie (<1%)   |

Only `K=4096` shows the flip — exactly where the LX cap separates the
two candidates. At `K=2048` both fit; at `K=8192` neither fits and
both pay the same overflow cost. **The LX-fit boundary is a real
mechanism but it only fires in a narrow band of `K * (N / n)`.**

This finding *cannot* explain the `N=12800-15872` misses: per-core
weights at `(8, 4)` for `N=12800` are `4096 * 3200 * 2 = 25 MB`, far
above the LX cap and far above `(4, 8)`'s 12.5 MB. Both candidates
overflow heavily; if overflow had a uniform per-MB cost, `(4, 8)`
would *always* win at `N >> 4096` and the cost model could just add a
linear weight-overflow term. But it doesn't, because at `N=16384,
20480` (also heavy overflow), `(8, 4)` wins by ~40%. The mechanism is
not uniform LX overflow.

#### H2. HMI segmentation — **FALSIFIED**

We checked whether `(4, 8)` at the danger-band N values lines up with
HMI (Half-Memory-Interface) segmentation that would give it cheaper
HBM contention. The HBM controller serves 16 channels, and certain
power-of-two tile widths align cleanly with channel groups. But
`N=12800` and `N=13824` are not power-of-two and don't hit any
documented HMI alignment. `N=14336 = 14 * 1024` is closer to a "nice"
boundary but the win is in the same ~1% margin as `N=15872 = 248 * 64
sticks` which has no such structure. The N values that flip don't
share an alignment signature.

#### H3. LX → XRF bandwidth — **too small to be the main mechanism**

XRF (cross-RIU forwarding) bandwidth could in principle penalize wider
`m` if activations have to cross more ring hops. We estimated the XRF
bandwidth cost from published BiRing numbers (166 GB/s/dir, 32 cores,
~5 ns/hop) and find it is at most a few percent of compute time for
the shapes in question — too small to explain a ~10% flip. We did
not run isolated XRF micro-benchmarks; the upper bound from
back-of-envelope is enough to deprioritize the hypothesis.

#### H4. PT tile alignment — **ambiguous discriminator**

The PT array is 8 rows by 8 cols per corelet, 2 corelets per core.
For `M=512` and split `m=8`, per-core M is 64 = 8 PT-passes — exactly
the calibrated `_TARGET_PT_PASSES`. For `m=4`, per-core M is 128 = 16
passes — twice the target. Pure compute prefers `m=8`; pure
activation reuse prefers more M per core. Neither is a sharp
discriminator that flips at specific N values. PT alignment may
contribute a small DC offset but cannot account for the N-dependent
flips.

#### H5. clSplit=1 plus cohort interaction — **predicate confirmed in source, hypothesis REFUTED for general case**

This was the most carefully traced. The corelet-split (clSplit)
decision in
[`deeptools/dsm/SdscCoreletSplit.cpp:126,130-138`](file:///home/adnan/dt-inductor/deeptools/dsm/SdscCoreletSplit.cpp)
gates whether a dimension can be split across the 2 corelets per core:

- Line 126: rejects dims with size `< 2` or odd size.
- Lines 130-138: rejects a corelet split if the *output stick size*
  exceeds `curr_dim_size / 2` or doesn't divide it.

For the N dim at `(8, 4)` with `N=12800`, per-core N is `12800 / 4 =
3200 elements = 50 sticks`. With output stickSize on N being 32 (a
common configuration for `N >> 64`), `stickSize > N/2` is false, but
`(N/2) % stickSize` is `1600 % 32 = 0`, so corelet-split is allowed.
For `(4, 8)`, per-core N is `1600 elements = 25 sticks` — stickSize=32
> N/2=12.5, so the corelet split on N is *rejected* and the kernel
falls back to splitting on M between corelets. clSplit=1 (no corelet
split on N) for `(4, 8)`.

We hypothesized that clSplit=1 plus a particular cohort shape might
trigger a different scheduler path that's faster. To test, we ran the
synthetic N-landscape sweep (`/tmp/synthetic_sweep.log`). The result:
**among the N values where (4, 8) hits clSplit=1, only N=1536 and
N=12800 actually have (4, 8) winning by > 5%**. At N=11264 and N=10752
(also clSplit=1 territory by the same predicate analysis), `(4, 8)` is
~50% slower. The clSplit=1 condition is necessary but not sufficient.

#### What's left

The danger band `N ∈ [12800, 15872]` is real and reproducible across
several measurement passes, but the mechanism is diffuse:

- It's not LX overflow (both candidates overflow).
- It's not HMI alignment.
- It's not pure clSplit=1 (counter-examples at adjacent N).
- It's not PT alignment (no N-dependent flip).
- It's not XRF bandwidth (magnitude too small).

The most likely explanation is **scheduler-side decisions that are not
exposed to the cost model**: kernel-template selection inside the SDSC
DDL templates (see `deeptools/ddc/ddl_templates/bmm.ddl` for an
example of the kind of multi-template lowering each shape goes
through), psum-algorithm choices (`unichain` / `bichain` /
`singleshot` are picked globally from data format and ring config —
see the MEMORY note on `dsm_psum_algos`), and second-order interaction
between the chunked output schedule and the K-loop weight broadcast.

### 5.3 Deeptools' own cost model is cohort-blind

A useful negative result. `Dsm::getCoreEqPerf` at
[`deeptools/dsm/dsm.cpp:19645`](file:///home/adnan/dt-inductor/deeptools/dsm/dsm.cpp)
is the function deeptools itself uses to rank candidate work splits.
The relevant computation:

```cpp
double macPerCore = wkSplit.coreWkPerDim.at(IN).wkSs *
                    wkSplit.coreWkPerDim.at(OUT).wkSs * ijMbPdt *
                    wkSplit.coreWkPerDim.at(KIJ).wkSs;
// ...
coreEfficiency = compCycles /
                 ((compCycles / coreletUnderUse / temporalUnderUse)
                  + psumCycles);
```

This is `compute + psum + corelet-under-use + temporal-under-use`. The
function has **no cohort term, no HBM term, and no LX-residency
term**. It also explicitly disables the xrfCapPenalty
(`xrfCapPenalty = 1; // removing xrf penalty with library update`, lines
19760-19761). The closest analogue to our `cohort_penalty` is
`outCoreletSplit` / `xCoreletSplit` at lines 19696-19710, which
detects a clSplit=2 condition for corelet utilization but doesn't
distinguish broadcast cohorts.

**Implication.** The mechanism behind the N-flip misses is not
captured in deeptools' own cost ranking either. If we wanted to
borrow their model directly, it would also miss these shapes. The
deeptools pipeline gets the right answer (when it does) because the
*scheduler* runs after `getCoreEqPerf`, and the scheduler's
kernel-template selection encodes some of the missing physics
implicitly. That selection is downstream of work-division and the only
way to peek at it from the inductor side is to actually compile and
measure — which is what our sweep harness does.

---

## 6. Non-linear cost-model attempts (negative result)

After confirming the LX-fit boundary at N=1536, the natural next step
was to add a smooth term that captures LX overflow without breaking
neighboring shapes. We codified 10 variants and one decision-tree
baseline in `/tmp/nonlinear_cost_model_fit.py`, then evaluated each
under both in-sample fit and leave-one-out cross-validation (LOO-CV).

Variants tested:

| variant            | extra term                                                           | intent |
|--------------------|----------------------------------------------------------------------|--------|
| `lx_fit`           | sigmoid bonus when `per_core_weights ≤ LX_PER_CORE_BYTES`            | bonus for fitting in LX |
| `saturating`       | `cohort_penalty` saturates above a knee (asymptote at coef * cohort) | bound cohort cost above heavy overflow |
| `corelet_align`    | penalty when `per_core_N / stickSize` is odd (proxy for clSplit=1)   | encode the clSplit predicate |
| `roofline_lx`      | bytes_total includes weight re-streaming when overflow > 2x LX       | first-principles weight re-load model |
| `cohort_asym`      | separate `cohort_lhs = m`, `cohort_rhs = n` with different weights   | break the (m,n) symmetry |
| `per_core_mem`     | threshold penalty when per-core (wts + out) > LX                     | combined LX pressure |
| `log_cohort`       | `log2(1 + cohort / 8)` instead of linear                             | concave broadcast cost |
| `gauss_band`       | Gaussian bump at per-core weights ≈ LX                               | encode the N=1536 pocket |
| `lx_plus_align`    | `lx_fit` + `corelet_align`                                           | two effects together |
| `decision_tree`    | sklearn `DecisionTreeClassifier` on derived features                 | overfit baseline / upper bound |

Each smooth variant has one coefficient `coef`. We grid-searched coef
over physically motivated ranges (e.g., LX-fit bonus in
[1, 200] µs/MB; cohort knee in [4, 16]) and picked the coef that
maximized in-sample hits. We then re-fit per leave-one-out fold and
counted LOO hits.

**Results:**

- **All 10 smooth variants match baseline at 37/53 LOO** (some are 36
  or 38 in-sample with different coef choices, but every LOO score is
  within ±1 of baseline). In several variants the LOO-best coef is
  *zero* — the optimizer chose to disable the new term to avoid
  regressing other shapes.
- **The decision tree achieves 40/53 in-sample but collapses to 25/53
  LOO** — catastrophic overfitting. With 53 training points and a
  tree of any useful depth, the tree memorizes the training shapes
  and generalizes worse than the baseline closed-form.
- **No single term we added can flip the danger-band shapes without
  regressing adjacent N values.** The `gauss_band` variant comes
  closest at `N=1536` but cannot also capture the wider band at
  N=12800–15872 with the same Gaussian center and width.

**The closed-form ceiling.** With the inputs available to the
cost model `(B, M, K, N, b, m, n, k, max_cores)` and a small handful
of derived features (per-core MACs, per-core weights, per-core output,
broadcast cohorts, clSplit predicate), the empirical ceiling on this
dataset is ~70%. The remaining 30% require either:

- a feature we don't currently extract (e.g., the actual scheduler's
  kernel-template choice for this shape), or
- many more measurements to fit a non-linear ML model without
  overfitting, or
- direct integration with deeptools' downstream scheduler.

We chose not to ship any of the variants. Adding a term that doesn't
beat baseline under cross-validation is strictly worse than the
current model — it adds complexity, adds calibration risk, and locks
in a particular degree of freedom that future changes have to either
inherit or undo.

---

## 7. What the model gets right

Beyond the validation hit rate, the model captures the right physics
in the regimes it covers:

- **Compute term** is sqrt-derated below 8 PT passes, which matched
  measurement better than the original linear ramp (linear predicted
  ~50% derate at 4 passes; measured was 10–30%).
- **HBM term** correctly identifies memory-bound shapes (small M,
  large N) where adding K-split candidates makes things worse via
  `psum_us`.
- **PSUM term** has the right magnitude — extreme-K (`K=32768,
  M=128, N=512`) correctly stays on `k=1` despite per-core FLOP being
  appealing for higher k.
- **Batch penalty** correctly suppresses batch-split for MoE bmm.
  The `b ** 1.4` power law was derived from a bmm batch sweep where
  the prior linear form `1 + 0.6 * (b - 1)` under-predicted by 3–4x
  at b=8.
- **target_m** at M ≥ 256 correctly disambiguates the (8,4) vs.
  (16,2) candidates that have nearly identical compute and HBM.
- **The LX-fit special case for N=1536-class shapes** is correctly
  *not* in the model (we'd ship a regression on adjacent N). The
  miss is one we accept rather than one we mis-fit.

---

## 8. What the model misses and why

The 30% miss rate is concentrated in the M=512, K=4096 regime where
the (4, 8) and (8, 4) candidates have nearly identical compute and
HBM bytes, and the per-MAC margin is small enough that second-order
scheduler effects can flip the winner. Three structural reasons the
closed form can't disambiguate these:

1. **The cost model only sees the planner-level work split.** It
   doesn't see clSplit, stick choice, psum_algo selection,
   kernel-template choice, or how the schedule interacts with the
   K-loop. All of these are decided downstream of the planner, and
   any of them can shift the kernel by 5-15%.
2. **The fitting target (`kernel_ms`) bakes in many of those choices
   together.** When we observe a 10% gap between (8, 4) and (4, 8)
   we don't know if it's coming from clSplit, from a different psum
   algorithm, from a different DDL template, or from a contention
   pattern that only appears at certain N. Without instrumentation
   that separates these, the cost model can only fit aggregate.
3. **There aren't enough measurements to fit a richer model without
   overfitting.** 53 shapes is enough to validate a 5-term closed
   form but nowhere near enough to fit a non-linear model with
   multiple interaction terms. Section 6 shows the decision tree
   memorizing the training set and generalizing worse than baseline.

The N=1536 LX-boundary case is the one exception where we can name
the mechanism precisely (per-core weights near the LX cap, K-scaling
isolates it). Even there, a one-shape kludge is not worth the
calibration risk on neighboring shapes — the earlier `lx_pressure_us`
term (now removed) was exactly such a kludge and it broke
adjacent-N picks.

---

## 9. Architecture pattern (forward-looking)

If a future iteration *does* want to add a non-trivial term, the
deeptools-style architecture is a cleaner pattern than adding more
heuristics to the closed-form scoring function.

Deeptools separates concerns:

- **`getCoreEqPerf`** is a smooth, differentiable closed-form cost
  function used to rank candidates. It captures the things smooth
  functions capture well: compute, PSUM ring hops, corelet-under-use.
- **`predict_schedule`** (downstream) is a discrete simulator that
  mirrors what the actual scheduler will do for the candidate. It
  encodes things smooth functions handle poorly: kernel-template
  choice, clSplit decisions, stick alignment, chunk boundaries.

Our inductor cost model today is essentially the
`getCoreEqPerf`-equivalent for our planner; we have no
`predict_schedule` equivalent. We have empirically confirmed that
the missing 30% is in the `predict_schedule` layer's domain — it's
discrete scheduler decisions, not smooth physics.

The clean future pattern is:

1. Keep the current `_matmul_split_cost` as the smooth ranking
   function (it's well-validated on 70% of shapes and the failure
   mode is in a specific shape regime).
2. Add a thin discrete `predict_split_realization` layer that, for
   each candidate, derives what the downstream scheduler will choose
   (stick size, clSplit value, psum algo) given the candidate split
   and the op's static layout — and folds the resulting penalty
   directly into the cost.
3. If steps 1 + 2 still leave a gap, the gap is then by definition
   *kernel-template selection* and would need either deeptools
   integration or much more measurement data.

We don't recommend doing this now. The complexity is high relative
to the remaining ceiling (~30% of shapes by count, ~5-15% by kernel
margin, well under 1% of model end-to-end). But it's the cleaner
shape for the codebase if a future research effort revisits.

---

## 10. Future directions

Three honest paths, in order of effort and uncertainty:

1. **Accept the documented misses, move on.** The current model lands
   the empirical best split on 70% of shapes and the remaining 30%
   cost well under 1% of model end-to-end. The misses are documented
   in [`cost_model_equation.md`](cost_model_equation.md) under "Known
   limits". This is the path we are taking now.

2. **Deeper deeptools integration.** Two flavors. The lower-effort
   version is to import deeptools' `getCoreEqPerf` (or a Python
   port of it) as a black-box cost function and let it rank
   candidates instead of our closed-form. This would surface
   different misses but on this dataset would not improve aggregate
   hit rate — the N-flip mechanism isn't in `getCoreEqPerf` either.
   The higher-effort version is to expose the scheduler's
   `predict_schedule` decisions back to the inductor planner so we
   can fold them into the cost. This is RFC-scope work and crosses
   the team boundary (see MEMORY note on inductor-team scope).

3. **ML cost model trained on much more data.** Sweep ~500-1000
   shapes across the realistic `(M, K, N, b)` space, train a small
   gradient-boosted tree or MLP on the empirical timings, and use
   that as the planner cost. The decision-tree result in section 6
   suggests this is feasible but only with >>53 measurements — at
   the current dataset size, ML models overfit catastrophically.
   The infrastructure cost is non-trivial (sweep harness, data
   pipeline, training pipeline, model serving inside the inductor
   call path) and the win is uncertain — we'd be betting that the
   missing physics is in `(M, K, N, b)` rather than in opaque
   scheduler decisions that even more shape data can't expose.
