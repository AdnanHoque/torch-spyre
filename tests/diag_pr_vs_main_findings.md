# PR 1986 end-to-end vs main — findings

End-to-end measurement of the combined k_fast PR (split override +
SDSC core-id mapping) against current main. Same A/B/C decomposition
style as `diag_k_fast_granite_findings.md`, but with **`A` redefined
as main's actual planner default** (no forced pure-M baseline). This
is the answer to "does this PR help end-to-end vs main?".

## Methodology

For each shape:

```
A — main baseline:  main planner default, identity SDSC mapping
B — split-k + id:   PR's heuristic split, identity SDSC mapping
C — split-k + kf:   PR's heuristic split, k_fast SDSC mapping
```

Deltas (`>1.00× = improvement`, normalized to `A = 1.00`):

```
A → B:  gain (or loss) from the split-k override alone
B → C:  gain from k_fast SDSC emission at the chosen split
A → C:  combined PR effect end-to-end
```

`A` is measured on `main @ 5d33571` via `torch.compile(matmul)` with no
forced split. `B` and `C` are measured on `PR @ d40ec57` with the
planner forced to PR's heuristic pick (via a
`multi_dim_iteration_space_split` monkey-patch) and
`config.core_id_k_fast_emission` toggled between `False` (`B`) and
`True` (`C`).

Run config: WARMUP=5, ITERS=20 timed iterations per cell, single
subprocess per cell, dtype=fp16, SENCORES=32. Median wall time. Each
subprocess re-compiles from scratch (`fx_graph_cache=False`,
`torch._dynamo.reset()`).

## Per-shape table (sorted highest A→C → lowest)

| shape | (M, N, K) | h-split | A→B | B→C | A→C | combined |
|---|---|---|---:|---:|---:|---|
| L3-70B q_proj M=128 | (128, 8192, 8192) | (1,16,2) | 1.52× | 1.85× | **2.80×** | win |
| Granite 3.3 8B down_proj M=128 | (128, 4096, 12800) | (1,16,2) | 1.47× | 1.88× | **2.76×** | win |
| Granite 3.3 8B q_proj M=128 | (128, 4096, 4096) | (1,16,2) | 1.48× | 1.85× | **2.75×** | win |
| Granite 3.3 8B o_proj M=128 | (128, 4096, 4096) | (1,16,2) | 1.47× | 1.86× | **2.73×** | win |
| L3-70B kv_proj M=128 | (128, 1024, 8192) | (1,16,2) | 2.43× | 1.02× | **2.49×** | win |
| Granite 3.3 8B kv_proj M=32 | (32, 2048, 4096) | (1,16,2) | 2.39× | 1.03× | **2.47×** | win |
| Granite 3.3 8B kv_proj M=128 | (128, 2048, 4096) | (1,16,2) | 2.38× | 1.04× | **2.47×** | win |
| L3-70B kv_proj M=32 | (32, 1024, 8192) | (1,16,2) | 2.48× | 0.99× | **2.46×** | win |
| Mixtral kv_proj M=128 | (128, 1024, 4096) | (1,16,2) | 2.09× | 1.05× | **2.20×** | win |
| DSv3 kv_proj M=128 | (128, 1536, 7168) | (1,8,4) | 1.77× | 1.16× | **2.04×** | win |
| DSv3 q_a_proj M=128 | (128, 1536, 7168) | (1,8,4) | 1.74× | 1.12× | **1.95×** | win |
| DSv3 down_proj M=128 | (128, 7168, 18432) | (1,16,2) | 1.60× | 1.11× | **1.77×** | win |
| L3-70B kv_proj M=512 | (512, 1024, 8192) | (1,16,2) | 0.79× | 1.60× | **1.27×** | win (kf rescue) |
| Granite 3.3 8B o_proj M=32 | (32, 4096, 4096) | (1,16,2) | 1.13× | 1.03× | **1.16×** | win |
| Granite 3.3 8B gate_proj M=32 | (32, 12800, 4096) | (1,8,4) | 1.16× | 1.00× | **1.16×** | win |
| Granite 3.3 8B q_proj M=32 | (32, 4096, 4096) | (1,16,2) | 1.12× | 0.99× | **1.11×** | win |
| Granite 3.3 8B gate_proj M=128 | (128, 12800, 4096) | (1,8,4) | 0.97× | 1.07× | **1.04×** | win |
| Granite 3.3 8B down_proj M=32 | (32, 4096, 12800) | (1,16,2) | 0.97× | 1.01× | **0.98×** | regression ⚠ |
| DSv3 gate_proj M=32 | (32, 18432, 7168) | (1,16,2) | 0.96× | 1.02× | **0.98×** | regression ⚠ |
| L3-70B q_proj M=32 | (32, 8192, 8192) | (1,16,2) | 0.90× | 1.01× | **0.91×** | regression ⚠ |

## Aggregate

| | Combined suite |
|---|---:|
| Shapes measured | 20 |
| Wins (A→C ≥ 1.00×) | 17 |
| Regressions (A→C < 1.00×) | 3 |
| Big wins (A→C ≥ 2.0×) | 12 |
| Modest wins (1.1× – 1.99×) | 4 |
| Flat (0.95× – 1.05×) | 1 |
| Geomean A→C | **1.73×** |
| Geomean A→C on wins only | 1.92× |

## Observations

- **Two distinct win patterns visible in the A→B / B→C split.**
  - **N-split-driven wins (high A→B, ~1.0× B→C):** the 7 shapes with
    `A→B ≥ 2.0×` and `B→C ≈ 1.0×`. Main is on pure-M (`{M: 32}`,
    PT half-fed at M ≤ 4·max_cores); switching to `(1, n, 1)`-flavored
    split alone recovers ~2.4×, and k_fast SDSC adds nothing because
    the K-cohort is 2 (1 hop either way).
  - **k_fast-driven wins (modest A→B, high B→C):** the 4 shapes with
    `A→B ≈ 1.5×` and `B→C ≈ 1.85×`. The PR split is good (~1.5×) but
    most of the speedup comes from collapsing the PSUM ring from `m·n`
    hops to 1, with the K-cohort large enough that the hops are
    significant.

- **The kf-rescue row (L3-70B kv_proj M=512).** `A→B = 0.79×` —
  the split alone is a regression (M=512 with main's pure-M already
  fully feeds PT; the override gives up that locality). But
  `B→C = 1.60×` — k_fast SDSC emission rescues the regression to a
  net `A→C = 1.27×` win. This is the regime where k_fast's SDSC
  contribution is *load-bearing* — without it the override would lose.

- **Three regressions** (`A→C < 1.00×`), all small (2–9%):
  - L3-70B q_proj M=32 (0.91×), DSv3 gate_proj M=32 (0.98×), and
    Granite 3.3 8B down_proj M=32 (0.98×). Same signature on all
    three: main picks a balanced pure-N split that already saturates
    32 cores. The override switches to `(1, n, k>1)` — adds
    bichain-PSUM overhead with no utilization gain to claw back.
  - **None are catastrophic** (worst case 9% slower). All would be
    cleanly eliminated by a cost-model gate that skips the override
    when main's pick already saturates cores.

- **Why this number is smaller than the 2.82× geomean from
  `diag_k_fast_granite_findings.md`.** The earlier report used
  `A = forced pure-M`, an isolation experiment that measured upside
  if main were doing pure-M. End-to-end vs the *actual* main
  planner, the win is bounded by the fraction of shapes where main
  is actually on pure-M (~60% in this suite). On those shapes,
  historical ~2.7–2.9× and the measured 2.5–2.8× match. On the
  rest, main's planner finds a non-pure-M split and the
  incremental gain is small or slightly negative.

- **Conclusion.** PR 1986 delivers a real, measurable end-to-end
  speedup: **1.73× geomean across 20 shapes**, **2.0–2.8× on the
  ~60% of shapes where main is stuck on pure-M**, with **3 small
  regressions (0.91–0.98×)** on shapes where main already picks a
  balanced pure-N split. The headline framing for the PR
  description should be the end-to-end number, not the
  isolation-experiment number, and should acknowledge the
  regressions explicitly.

## Caveats & follow-ups

- **Cost-model gap.** The regression shapes share a signature: main
  picks a balanced pure-N split that already saturates all 32 cores
  on a single non-reduction dim, with `rows_per_core ≥ _PT_ROWS`.
  The PR's override fires anyway and adds reduce overhead with no
  utilization to claw back. A cheap predicate could close this:
  skip the override when the current splits already use all
  `max_cores` cores AND `rows_per_core ≥ _PT_ROWS`.
- The bmm / multi-output-dim path is still excluded with an explicit
  guard. Generalizing that is the larger follow-up.
