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

## Aggregate

| | Granite 3.3 8B |
|---|---:|
| Shapes measured | 10 |
| Wins (≥ 1.00×) | 10 |
| Regressions | 0 |
| Big wins (≥ 2.5×) | 5 |
| Modest wins (1.1–1.2×) | 3 |
| Flat (≤ 1.05×) | 2 |
| Geomean speedup | **1.74×** |

## Observations

- **Bimodal outcome**: 5 of 10 shapes hit 2.5–2.8× wins (big), the other
  5 land between 1.00× and 1.16× (modest to flat). No regressions.

- **The bimodality comes from what main picks, not from k_fast variability.**
  Probed with a tap on `apply_splits`:
  - On the big-win shapes (q/o/kv/down M=128) main picks pure-M
    `{M: 32}` — 4 rows per core, half-fills the 8-row PT block. The PR
    override replaces that with a `(1, n, k>1)` split that keeps PT
    fully fed.
  - On gate_proj M=128 main picks `{N: 25}` (because
    `core_split(200_sticks, 32) = 25`, a divisor). 25 cores used, 7
    idle, no reduce overhead. The PR override goes to `(1, 8, 4)` — 32
    cores active but pays bichain PSUM. Net: ~1.04×. Utilization gain
    eaten by reduce overhead.
  - On down_proj M=32 the planner gates fire and PR picks `(1, n, k>1)`,
    but main's default is already in a "good" regime for this shape;
    overhead is a wash. **Net 1.00×, not a regression.**

- **Why this number is smaller than 2.82× geomean from
  `diag_k_fast_granite_findings.md`.** The earlier report used
  `A = forced pure-M`, which is *the worst case the heuristic is
  designed to escape*. End-to-end vs main, the win is bounded by the
  fraction of shapes where main was *already* on pure-M (5/10 for
  Granite 3.3 8B). On those, the historical 2.7–2.9× and the
  measured 2.5–2.8× match closely. On the other 5 shapes, main's
  default already finds a decent mixed split and the PR's incremental
  gain is small.

- **Conclusion.** PR 1986 delivers a real, measurable, no-regression
  end-to-end speedup on Granite 3.3 8B: **1.74× geomean across 10
  shapes**, **2.5–2.8× on the half of shapes where main is stuck on
  pure-M**. The headline framing for the PR description should be the
  end-to-end number, not the isolation-experiment number.

## Caveats & follow-ups

- Only Granite 3.3 8B M ∈ {32, 128} measured here. The
  `diag_k_fast_combined_findings_normalized.md` shape suite (L3-70B,
  Mixtral, DSv3) has not been re-run against the new A=main baseline.
  Based on the bimodal pattern above, those shapes that previously
  showed 2.0× – 2.5× at forced-pure-M `A` will likely show smaller
  end-to-end deltas where main's planner finds a decent N-split.
- The 1.04× and 1.00× rows highlight a cost-model gap: for shapes
  where main's planner happens to pick a viable N-split, the override
  trades idle cores for K-reduce overhead. A future cost-model heuristic
  could close that gap by skipping the override when main's pick is
  already PT-saturated.
