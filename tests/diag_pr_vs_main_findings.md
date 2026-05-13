# PR 1986 end-to-end vs main — findings

End-to-end measurement of the combined k_fast PR (split override +
SDSC core-id mapping) against current main, with **no monkey-patching**:
each shape compiles a single `torch.matmul` with `torch.compile`, lets
the planner pick whatever split it picks, and runs through the full
pipeline.

This is the answer to "does this PR help end-to-end vs main?". It is
different from earlier `diag_k_fast_*_findings.md` reports, which used
a forced pure-M baseline as `A` (an isolation experiment that sized
upside, not delta-vs-main).

## Methodology

| Column | What it measures |
|---|---|
| `main` | main branch (commit `5d33571`), `torch.compile(matmul)`, planner default |
| `PR` | PR branch (commit `d40ec57`), same shape, planner picks split + k_fast SDSC emission |
| `Speedup` | `main median / PR median` |

Run config: WARMUP=5, ITERS=20 timed iterations per shape, single subprocess
per shape, dtype=fp16, SENCORES=32. Median wall time reported. Each
subprocess re-compiles from scratch (`fx_graph_cache=False`,
`torch._dynamo.reset()`).

Driver: `/tmp/granite_sweep.sh` → `/tmp/kfast_clean_measure.py` (no
force-split monkey-patch, no SDSC flag toggling — pure planner-driven).

## Per-shape table — Granite 3.3 8B (sorted highest speedup → lowest)

| Layer | (M, N, K) | main (ms) | PR (ms) | Speedup |
|---|---|---:|---:|---:|
| down_proj M=128 | (128, 4096, 12800) | 2.720 | 0.987 | **2.76×** |
| o_proj M=128 | (128, 4096, 4096) | 0.910 | 0.331 | **2.75×** |
| q_proj M=128 | (128, 4096, 4096) | 0.919 | 0.336 | **2.73×** |
| kv_proj M=128 | (128, 2048, 4096) | 0.477 | 0.186 | **2.56×** |
| kv_proj M=32 | (32, 2048, 4096) | 0.450 | 0.177 | **2.54×** |
| o_proj M=32 | (32, 4096, 4096) | 0.318 | 0.274 | 1.16× |
| gate_proj M=32 | (32, 12800, 4096) | 0.921 | 0.792 | 1.16× |
| q_proj M=32 | (32, 4096, 4096) | 0.310 | 0.274 | 1.13× |
| gate_proj M=128 | (128, 12800, 4096) | 1.028 | 0.985 | 1.04× |
| down_proj M=32 | (32, 4096, 12800) | 0.840 | 0.844 | 1.00× |

## Per-shape table — L3-70B, Mixtral, DSv3 (sorted highest speedup → lowest)

| Shape | (M, N, K) | main (ms) | PR (ms) | Speedup |
|---|---|---:|---:|---:|
| L3-70B q_proj M=128 | (128, 8192, 8192) | 3.576 | 1.282 | **2.79×** |
| L3-70B kv_proj M=128 | (128, 1024, 8192) | 0.476 | 0.187 | **2.55×** |
| L3-70B kv_proj M=32 | (32, 1024, 8192) | 0.453 | 0.180 | **2.51×** |
| Mixtral kv_proj M=128 | (128, 1024, 4096) | 0.257 | 0.110 | **2.33×** |
| DSv3 kv_proj M=128 | (128, 1536, 7168) | 0.604 | 0.290 | **2.08×** |
| DSv3 q_a_proj M=128 | (128, 1536, 7168) | 0.602 | 0.296 | **2.03×** |
| DSv3 down_proj M=128 | (128, 7168, 18432) | 6.784 | 3.827 | **1.77×** |
| L3-70B kv_proj M=512 | (512, 1024, 8192) | 0.474 | 0.368 | **1.29×** |
| DSv3 gate_proj M=32 | (32, 18432, 7168) | 3.541 | 3.683 | **0.96×** ⚠ |
| L3-70B q_proj M=32 | (32, 8192, 8192) | 0.962 | 1.039 | **0.93×** ⚠ |

## Aggregate

| | Granite 3.3 8B | L3 / Mixtral / DSv3 | Combined |
|---|---:|---:|---:|
| Shapes measured | 10 | 10 | 20 |
| Wins (≥ 1.00×) | 10 | 8 | 18 |
| Regressions (< 1.00×) | 0 | 2 | 2 |
| Big wins (≥ 2.0×) | 5 | 6 | 11 |
| Modest wins (1.1–1.99×) | 3 | 2 | 5 |
| Flat (0.95–1.05×) | 2 | 2 | 4 |
| Geomean speedup | **1.74×** | **1.79×** | **1.77×** |

## Observations

- **Trimodal outcome**: 11 of 20 shapes hit 2.0–2.8× wins (big),
  5 land 1.1–1.8× (modest), 4 land in the flat band (0.93×–1.16×). Two
  of those four are small regressions (5–7% slower).

- **The shape of the distribution is driven by what main picks, not
  by k_fast variability.** Probed with a tap on `apply_splits`:
  - **Big wins** (q/o/kv/down M=128 across all model families): main
    picks pure-M `{M: 32}` — 4 rows per core, half-fills the 8-row PT
    block. The PR override replaces that with a `(1, n, k>1)` split
    that keeps PT fully fed.
  - **Flat / regression band**: main already picks a balanced N-split
    using all 32 cores (e.g. `{N: 32}` for L3-70B q_proj M=32, or
    `{N: 25}` for Granite gate_proj M=128 where `core_split(200, 32)`
    returns the divisor 25). PR's `(1, n, k>1)` override only adds
    bichain-PSUM overhead — it can't add core utilization that was
    already saturated. When the PSUM cost > 0 and there's no
    counter-balancing utilization gain, the net is slightly negative.

- **Two regressions, both small and explainable**:
  - **L3-70B q_proj M=32 (32, 8192, 8192) — 0.93×**: main picks `{N: 32}`
    (pure-N, all cores active, no reduce). PR picks `(1, 16, 2)` — 32
    cores active + 2-way K-reduce. Reduce cost dominates the (zero)
    utilization gain.
  - **DSv3 gate_proj M=32 (32, 18432, 7168) — 0.96×**: same pattern.
    `core_split(288_sticks, 32) = 32` for main; PR overrides to `(1, 16, 2)`.

- **Why this number is smaller than 2.82× geomean from
  `diag_k_fast_granite_findings.md`.** The earlier report used
  `A = forced pure-M`, an isolation experiment that measured upside
  if main were doing pure-M. End-to-end vs main, the win is bounded
  by the fraction of shapes where main was *actually* on pure-M
  (~55% in this suite). On those shapes, the historical 2.7–2.9× and
  the measured 2.5–2.8× match closely. On the rest, main's default
  already finds a non-pure-M split and the PR's incremental gain is
  small or slightly negative.

- **Conclusion.** PR 1986 delivers a real, measurable end-to-end
  speedup: **1.77× geomean across 20 shapes**, **2.0–2.8× on the
  ~half of shapes where main is stuck on pure-M**, with **2 small
  regressions (0.93×, 0.96×)** on shapes where main already picks
  a balanced pure-N split. The headline framing for the PR
  description should be the end-to-end number, not the
  isolation-experiment number, and should acknowledge the
  regressions.

## Caveats & follow-ups

- **Cost-model gap.** The flat-band and regression shapes share a
  signature: main picks a balanced pure-N split that already saturates
  all 32 cores. The PR's override fires anyway and adds reduce
  overhead. A future cost-model heuristic could close this by *not*
  firing the override when the current splits already use all
  `max_cores` on a single non-reduction dim with no PT underfeeding.
  Cheap predicate: `splits_use_all_cores AND rows_per_core >= _PT_ROWS`.
- The bmm / multi-output-dim path is still excluded with an explicit
  guard (`work_division.py:_try_k_fast_split`). Generalizing that is
  the larger follow-up.
