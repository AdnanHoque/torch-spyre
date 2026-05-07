# LX residency planner — Phase 0 findings

Companion to `lx_fit.py` (predicate) and `diag_lx_overflow_phase0.py`
(diagnostic driver). Phase 0 was scoped as a **diagnostic only** — no
torch_spyre code changes — to measure whether the planner / k_fast
heuristic ever picks splits that overflow the 2 MB LX scratchpad, and
whether the Phase 0 cost model's residuals are explained by LX
overflow as Project B Phase 0 hypothesised.

## TL;DR

LX overflow under the planner's natural pure-M (32, 1, 1) is rare — it
only happens at high M=2048 on three down_proj shapes (Llama 70B,
Llama 405B, DSv3). Under PR 1933's k_fast heuristic, overflow happens
on 2/15 fired ops (Llama 70B and 405B kv_proj at M=512).

But — and this is the consequential finding — **LX overflow is not
the dominant residual in the cost model**. The biggest cost-model
errors occur on rows that *fit* LX. The Phase 0 hypothesis ("LX
overflow explains the model's k-split miss") is partially wrong: it
matters at extreme overage (≥10× LX) but is dwarfed elsewhere by an
unrelated residual in the PSUM / k_fast term.

This redirects the LX planner work: the within-op LX-fit gate is
still worth shipping — it cleanly catches the catastrophic-overflow
case (DSv3 o_proj M=2048 (1,16,2)+id measured 116 ms vs 44 ms
predicted) — but it isn't going to lift the 30-row validation set's
mean error meaningfully on its own.

## Predicate

The LX-fit predicate gates on the **stationary operand A** only:

    A_per_core = (M / m) * (K / k) * dtype_bytes
    fits ⇔ A_per_core ≤ LX_BYTES_PER_CORE  (2 MB)

A diagnostic conservative form `lx_fits_conservative()` adds B but
is not the headline. Project B Phase 0 documented A as the stationary
operand under the AIU matmul kernel template (B streams via the data
ring chunk-by-chunk and does not stay resident).

A first attempt at the predicate counted A+B together and reported
that 100% of pure-M matmuls overflow LX, which contradicts the
production fact that pure-M is the planner default and works. That
was the predicate bug; the A-only form matches the empirical
boundary that Project B's `hmi_cost_model_phase0_findings.md`
established.

## Section A — production sweep

120 matmul instances scanned (5 models × 4 M values × 6 matmul
ops/block).

| | overflow count | rate |
|---|---:|---:|
| pure-M (32, 1, 1) | 3 | 2% |
| PR 1933 heuristic firing instances | 2 / 15 | 13% |

Pure-M overflows only on three down_proj shapes at M=2048:

- Llama 70B `down_proj (2048, 8192, 28672)`: A_per=3.5 MB
- Llama 405B `down_proj (2048, 16384, 53248)`: A_per=6.5 MB
- DSv3 `down_proj (2048, 7168, 18432)`: A_per=2.25 MB

These are prefill-regime ops; production isn't running pure-M on them
in any case.

The heuristic-overflow cases — Llama 70B and Llama 405B kv_proj at
M=512 under (1, 16, 2) — were noted in
`hmi_cost_model_strategic_findings.md` as a separate side-finding
(`n_sticks` gate too narrow). The LX gate would correctly reject the
heuristic's pick and fall back to pure-M, which fits.

## Section B — residual analysis on the 30-row validation set

|  | n | mean \|err\| | max \|err\| |
|---|---:|---:|---:|
| LX-fitting rows | 23 | 16.3% | 94.8% |
| LX-overflow rows | 7 | 39.7% | 99.4% |

Overflow rows do have higher mean error (40% vs 16%) but the worst
single-row error of the entire set is on a *fitting* row:

- **DSv3 o_proj M=128 (1, 16, 2) + kf**: A_per = 2.00 MB (fits exactly).
  Predicted 9.14 ms, measured 4.69 ms → +94.8% (cost model
  *over*-predicts by ~2×).
- **DSv3 o_proj M=32 natural**: A_per = 32 KB (fits trivially).
  Predicted 8.91 ms, measured 4.84 ms → +84.1% over-predict.

These are k_fast-emission and small-M cases where the cost model
over-estimates wall by a factor of 2. The mechanism is independent
of LX overflow — likely a PSUM-term over-count or an HMI-launch-
floor interaction the model isn't capturing.

The clearest LX-overflow signature is the canonical Project B case:

- **DSv3 o_proj M=2048 (1, 16, 2) + id**: A_per = 32 MB (16× LX).
  Predicted 44.39 ms, measured 116.12 ms → -61.8% (cost model
  *under*-predicts by ~2.6×).

Here the re-fetch penalty is real and severe; the cost model misses
it entirely.

In between, modest overflow (4–8 MB, 2–4× LX) shows mixed signals:

- L3-70B kv_proj M=512 (1, 16, 2) + kf: 4 MB, +17.2% (mild over-pred)
- Mixtral kv_proj M=2048 (1, 16, 2) + id: 8 MB, +14.2% (mild over-pred)
- DSv3 down_proj M=2048 (1, 16, 2) + id: 4 MB, +99.4% (massive over-pred)

The DSv3 down_proj over-prediction at modest LX overage is a tell:
this row's residual is *not* explained by LX overflow at all. The
cost model is mis-modelling identity-emission (1, 16, 2) walls on
narrow-K shapes regardless of LX.

## What this means for the LX planner project

**The LX-fit gate is still worth shipping**, scoped narrowly:

1. As a **safety filter** in any candidate-enumerating planner (the
   Roller-on-AIU enumerator for example), it cleanly rejects splits
   like (1, 16, 2) on DSv3 o_proj M=2048 where the kernel would
   genuinely re-fetch A 16× per call.
2. As a **k_fast heuristic guard**: PR 1933's heuristic fires on
   2 shapes (L3-70B, L3-405B kv_proj at M=512) where the chosen
   split overflows. Those are exactly the cases where the heuristic's
   predicted win is least trustworthy.

But the gate does NOT lift cost-model accuracy across the validation
set, because:

- The dominant residuals (the rows with >50% error) are split
  between under-predict on extreme overflow (Project B's case) and
  over-predict on fitting k_fast/k-split rows.
- On the over-predict side, no LX-aware change can help — the cost
  model needs to be calibrated *down* on those, not up.

## Re-scoping

Original Phase 1 candidate: "add LX-overflow penalty to
hmi_cost_model.predict() and lift validation top-1 accuracy 23% →
60%+." This is no longer the right framing. A penalty term that
fires only on overflow rows would push 7/30 rows further wrong (the
4 modest-overflow over-predicts where the model is already too
high), while only really helping the 2–3 catastrophic underflows.

Revised Phase 1 candidate: **two-track**:

- **Track 1 (LX gate, no penalty):** ship `lx_fit.py` as a *gate*
  only. Wire it into the candidate enumerator and the k_fast
  heuristic as a fallback trigger. Don't touch cost-model wall
  predictions. Solo torch_spyre, days-scale.
- **Track 2 (PSUM/k_fast term recalibration):** separately,
  investigate why the cost model over-predicts on small-M k_fast
  rows by 2×. Different residual, different root cause. Probably
  bigger lever than the LX gate but needs fresh measurements.

The current branch is set up for Track 1. Track 2 is a separate
investigation that should either run after Track 1 ships or in
parallel under a different branch.

## Files

- `tests/lx_fit.py` — predicate + breakdown helper
- `tests/diag_lx_overflow_phase0.py` — diagnostic driver
- This doc — Phase 0 findings

## Next step

Move to Track 1 if the LX gate scope is still attractive given the
narrower-than-expected impact. Or pivot back to Roller-on-AIU
(branch `AdnanHoque/roller-aiu-phase0`) where the enumerator is
already partly built and the LX gate from this branch slots in as
one of its constraints.
