# Combined k_fast PR — normalized perf table (no absolute latencies)

Privacy-preserving copy of the per-shape data table from
`diag_k_fast_combined_findings.md` with all wall times normalized
to A = 1.00. The original findings doc retains absolute ms numbers
for internal reference; this file is the version safe for external
sharing (PR descriptions, design docs, etc.).

For each shape:

A — main baseline: pure-M (32, 1, 1), identity core id mapping
B — split-k + id: heuristic-picked split, identity core id mapping
C — split-k + kf: heuristic-picked split, k_fast core id mapping

Deltas:

A → B: gain from picking split-k alone (better PT util, per-cluster bytes)
B → C: gain from k_fast emission (PSUM hops m·n → 1)
A → C: combined PR effect

This separates the split-k contribution from the core id re-mapping
contribution.

A row where A→B is large and B→C is small means the split itself
does the work; a row where A→B is negative and B→C is positive
means the core id re-mapping is rescuing a regression.

Speedup ratios only (>1× = improvement vs the named baseline).

## Default LX (DXP_LX_FRAC_AVAIL = 0.2, the production default)

Numbers below are from the refactored heuristic (post-review changes:
hardware-derived thresholds + bmm/reshape support + divisors-based
candidate enumeration). 12-shape cross-vendor cohort plus 3
representative Granite shapes.

| shape | (M, N, K) | h-split | A→B | B→C | A→C | combined |
|---|---|---|---:|---:|---:|---|
| L3-70B kv_proj M=32      | (32, 1024, 8192)   | (1,16,2) | 2.44× | 1.03× | 2.52× | win |
| L3-70B kv_proj M=128     | (128, 1024, 8192)  | (1,16,2) | 2.47× | 0.99× | 2.45× | win |
| L3-70B kv_proj M=512     | (512, 1024, 8192)  | (1,16,2) | 0.79× | 1.60× | 1.26× | win (kf rescue) |
| Mixtral kv_proj M=128    | (128, 1024, 4096)  | (1,16,2) | 2.07× | 1.06× | 2.18× | win |
| DSv3 kv_proj M=128       | (128, 1536, 7168)  | (1,8,4)  | 1.75× | 1.17× | 2.04× | win |
| DSv3 q_a_proj M=128      | (128, 1536, 7168)  | (1,8,4)  | 1.75× | 1.16× | 2.04× | win |
| L3-70B q_proj M=32       | (32, 8192, 8192)   | (1,16,2) | 3.22× | 1.01× | 3.25× | win |
| DSv3 gate_proj M=32      | (32, 18432, 7168)  | (1,16,2) | 1.81× | 1.01× | 1.83× | win |
| L3-70B q_proj M=128      | (128, 8192, 8192)  | (1,16,2) | 1.52× | 1.84× | 2.80× | win |
| L3-70B q_proj M=512      | (512, 8192, 8192)  | —        | —     | —     | —     | (skipped, correct) |
| DSv3 down_proj M=128     | (128, 7168, 18432) | (1,16,2) | 1.61× | 1.11× | 1.79× | win |
| L3-70B kv_proj M=2048    | (2048, 1024, 8192) | —        | —     | —     | —     | (skipped, correct) |
| Granite 8B q_proj M=128  | (128, 4096, 4096)  | (1,16,2) | 1.46× | 1.83× | 2.68× | win |
| Granite 8B gate_proj M=32| (32, 12800, 4096)  | (1,8,4)  | 3.32× | 1.00× | 3.32× | win |
| Granite 8B down_proj M=128| (128, 4096, 12800)| (1,16,2) | 1.48× | 1.92× | 2.83× | win |

## Maximum-reserve LX (DXP_LX_FRAC_AVAIL = 1.0)

Same shape suite, run with `DXP_LX_FRAC_AVAIL=1.0`. Inductor's
`scratchpad_planning` pass is gated by `LX_PLANNING=1` (default off),
so this run only exercises the backend's reading of the env var
(`deeptools/dxp/dxp.cpp`).

| shape | (M, N, K) | h-split | A→B | B→C | A→C | combined |
|---|---|---|---:|---:|---:|---|
| L3-70B kv_proj M=32      | (32, 1024, 8192)   | (1,16,2) | 2.30× | 0.99× | 2.28× | win |
| L3-70B kv_proj M=128     | (128, 1024, 8192)  | (1,16,2) | 1.65× | 0.97× | 1.60× | win |
| L3-70B kv_proj M=512     | (512, 1024, 8192)  | (1,16,2) | 1.27× | 1.02× | 1.29× | win |
| Mixtral kv_proj M=128    | (128, 1024, 4096)  | (1,16,2) | 1.37× | 0.99× | 1.35× | win |
| DSv3 kv_proj M=128       | (128, 1536, 7168)  | (1,8,4)  | 1.34× | 1.05× | 1.41× | win |
| DSv3 q_a_proj M=128      | (128, 1536, 7168)  | (1,8,4)  | 1.39× | 1.04× | 1.45× | win |
| L3-70B q_proj M=32       | (32, 8192, 8192)   | (1,16,2) | 2.73× | 1.02× | 2.77× | win |
| DSv3 gate_proj M=32      | (32, 18432, 7168)  | (1,16,2) | 1.73× | 1.03× | 1.78× | win |
| L3-70B q_proj M=128      | (128, 8192, 8192)  | (1,16,2) | 2.17× | 0.95× | 2.05× | win |
| L3-70B q_proj M=512      | (512, 8192, 8192)  | —        | —     | —     | —     | (skipped, correct) |
| DSv3 down_proj M=128     | (128, 7168, 18432) | (1,16,2) | 1.43× | 1.07× | 1.52× | win |
| L3-70B kv_proj M=2048    | (2048, 1024, 8192) | —        | —     | —     | —     | (skipped, correct) |
| Granite 8B q_proj M=128  | (128, 4096, 4096)  | (1,16,2) | 1.76× | 0.97× | 1.71× | win |
| Granite 8B gate_proj M=32| (32, 12800, 4096)  | (1,8,4)  | 3.23× | 1.00× | 3.24× | win |
| Granite 8B down_proj M=128| (128, 4096, 12800)| (1,16,2) | 2.08× | 0.98× | 2.04× | win |

## Aggregate

| | Default LX | DXP_LX_FRAC_AVAIL=1.0 |
|---|---:|---:|
| Shapes in suite | 15 | 15 |
| Heuristic fires on | 13 / 15 | 13 / 15 |
| Wins | 13 | 13 |
| Regressions | 0 | 0 |
| Geomean speedup (A→C) | **2.22×** | **1.93×** |

## Side-by-side delta

A→C ratio in each LX condition (all 13 firing shapes):

| shape | default | LX=1.0 | Δ |
|---|---:|---:|---:|
| L3-70B kv_proj M=32        | 2.52× | 2.28× | -0.24 |
| L3-70B kv_proj M=128       | 2.45× | 1.60× | **-0.85** |
| L3-70B kv_proj M=512       | 1.26× | 1.29× | +0.03 |
| Mixtral kv_proj M=128      | 2.18× | 1.35× | **-0.83** |
| DSv3 kv_proj M=128         | 2.04× | 1.41× | -0.63 |
| DSv3 q_a_proj M=128        | 2.04× | 1.45× | -0.59 |
| L3-70B q_proj M=32         | 3.25× | 2.77× | -0.48 |
| DSv3 gate_proj M=32        | 1.83× | 1.78× | -0.05 |
| L3-70B q_proj M=128        | 2.80× | 2.05× | -0.75 |
| DSv3 down_proj M=128       | 1.79× | 1.52× | -0.27 |
| Granite 8B q_proj M=128    | 2.68× | 1.71× | **-0.97** |
| Granite 8B gate_proj M=32  | 3.32× | 3.24× | -0.08 |
| Granite 8B down_proj M=128 | 2.83× | 2.04× | -0.79 |

Headline: **all 13 shapes still win under DXP_LX_FRAC_AVAIL=1.0**,
no regressions, but the wins shrink. M=128 shapes take the biggest
hit (gap closes by ~0.7-1.0×); M=32 shapes are nearly unaffected.

## Source measurements

- `tests/diag_k_fast_dxp_avail_probe.py` — probe used for both runs
- `tests/diag_k_fast_dxp_avail_results.txt` — raw output, both
  conditions concatenated
- `tests/diag_k_fast_combined_3way_v3_postrebase_results.txt` —
  prior raw v3 run on the pre-refactor heuristic
- `tests/diag_k_fast_combined_findings.md` — full findings with
  absolute numbers and analysis sections

This file is the external-share-friendly copy of the per-shape
table only.

Replaces drafts #1932 and #1933.
