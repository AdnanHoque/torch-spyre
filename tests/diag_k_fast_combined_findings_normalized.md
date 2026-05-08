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

All wall times normalized to A (A = 1.00 by construction). B and C
columns show B/A and C/A — values < 1.00 are faster than baseline,
values > 1.00 are slower. Speedup ratios A→B, B→C, A→C are
unchanged from the original (>1× = improvement).

| shape | (M, N, K) | h-split | A | B | C | A→B | B→C | A→C | combined |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| L3-70B kv_proj M=32 | (32, 1024, 8192) | (1,16,2) | 1.00 | 0.39 | 0.39 | 2.49× | 1.03× | 2.57× | win |
| L3-70B kv_proj M=128 | (128, 1024, 8192) | (1,16,2) | 1.00 | 0.40 | 0.42 | 2.48× | 0.98× | 2.43× | win |
| L3-70B kv_proj M=512 | (512, 1024, 8192) | (1,16,2) | 1.00 | 1.28 | 0.78 | 0.79× | 1.61× | 1.28× | win (kf rescue) |
| Mixtral kv_proj M=128 | (128, 1024, 4096) | (1,16,2) | 1.00 | 0.44 | 0.44 | 2.26× | 1.04× | 2.35× | win |
| DSv3 kv_proj M=128 | (128, 1536, 7168) | (1,8,4) | 1.00 | 0.56 | 0.48 | 1.77× | 1.17× | 2.07× | win |
| DSv3 q_a_proj M=128 | (128, 1536, 7168) | (1,8,4) | 1.00 | 0.56 | 0.48 | 1.77× | 1.17× | 2.08× | win |
| L3-70B q_proj M=32 | (32, 8192, 8192) | (1,16,2) | 1.00 | 0.31 | 0.31 | 3.22× | 1.02× | 3.28× | win |
| DSv3 gate_proj M=32 | (32, 18432, 7168) | (1,16,2) | 1.00 | 0.56 | 0.56 | 1.77× | 1.01× | 1.79× | win |
| L3-70B q_proj M=128 | (128, 8192, 8192) | (1,16,2) | 1.00 | 0.66 | 0.36 | 1.53× | 1.85× | 2.82× | win |
| L3-70B q_proj M=512 | (512, 8192, 8192) | — | 1.00 | — | — | — | — | — | (skipped, correct) |
| DSv3 down_proj M=128 | (128, 7168, 18432) | (1,16,2) | 1.00 | 0.62 | 0.57 | 1.61× | 1.10× | 1.77× | win |
| L3-70B kv_proj M=2048 | (2048, 1024, 8192) | — | 1.00 | — | — | — | — | — | (skipped, correct) |

Replaces drafts #1932 and #1933.

## Aggregate (normalized form)

| | PR 1933 as-shipped | Combined + extension |
|---|---:|---:|
| Heuristic fires on | 6/12 | 10/12 |
| Wins | 5 | 10 |
| Regressions | 0 | 0 |
| Geomean speedup (A→C) | 1.07× | 2.06× |

## Source measurements

Absolute ms measurements (internal reference only) live in:

- `tests/diag_k_fast_combined_3way_v3_postrebase_results.txt` — raw v3 run
- `tests/diag_k_fast_combined_findings.md` — full findings with
  absolute numbers and analysis sections

Both files remain unchanged; this file is the
external-share-friendly copy of the per-shape table only.
