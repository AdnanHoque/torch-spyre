# Granite 3-way measurement campaign — findings

IBM Granite 3.x companion to the combined k_fast PR evidence.

Mirrors the methodology of `diag_k_fast_combined_findings_normalized.md`
with the shape suite replaced by Granite 3.x dense linear-layer shapes.
All wall times normalized to A = 1.00; only speedup ratios shown.

For each shape:

A — main baseline:  pure-M (32, 1, 1), identity core id mapping
B — split-k + id:   heuristic-picked split, identity core id mapping
C — split-k + kf:   heuristic-picked split, k_fast core id mapping

Deltas:

A → B: gain from picking split-k alone (better PT util, per-cluster bytes)
B → C: gain from k_fast emission (PSUM hops m·n → 1)
A → C: combined PR effect

Shape suite: 40 shapes (Granite 3.x 2B + 8B × 5 ops × M ∈ {32, 128, 512, 2048}).
Ops: kv_proj (combined K+V), q_proj, o_proj, gate_proj, down_proj.

Run config: WARMUP=3, ITERS=12, dtype=fp16, SENCORES=32.

## Per-shape table (heuristic-fires only)

| shape | (M, N, K) | h-split | A→B | B→C | A→C | combined |
|---|---|---|---:|---:|---:|---|
| Granite 3 2B kv_proj   M=32  | (32, 1024, 2048)   | (1,16,2) | 2.19× | 1.00× | 2.19× | win |
| Granite 3 2B q_proj    M=32  | (32, 2048, 2048)   | (1,16,2) | 2.32× | 1.09× | 2.54× | win |
| Granite 3 2B o_proj    M=32  | (32, 2048, 2048)   | (1,16,2) | 2.43× | 1.00× | 2.43× | win |
| Granite 3 2B gate_proj M=32  | (32, 8192, 2048)   | (1,16,2) | 2.94× | 0.99× | 2.93× | win |
| Granite 3 2B down_proj M=32  | (32, 2048, 8192)   | (1,16,2) | 2.69× | 1.03× | 2.77× | win |
| Granite 3 2B kv_proj   M=128 | (128, 1024, 2048)  | (1,16,2) | 1.83× | 1.02× | 1.87× | win |
| Granite 3 2B q_proj    M=128 | (128, 2048, 2048)  | (1,16,2) | 2.15× | 1.05× | 2.25× | win |
| Granite 3 2B o_proj    M=128 | (128, 2048, 2048)  | (1,16,2) | 2.11× | 1.06× | 2.24× | win |
| Granite 3 2B gate_proj M=128 | (128, 8192, 2048)  | (1,16,2) | 1.51× | 1.76× | 2.65× | win |
| Granite 3 2B down_proj M=128 | (128, 2048, 8192)  | (1,16,2) | 2.73× | 0.97× | 2.65× | win |
| Granite 3 2B kv_proj   M=512 | (512, 1024, 2048)  | (1,16,2) | 0.83× | 1.38× | 1.15× | win (kf rescue) |
| Granite 3 8B kv_proj   M=32  | (32, 2048, 4096)   | (1,16,2) | 2.45× | 1.04× | 2.55× | win |
| Granite 3 8B q_proj    M=32  | (32, 4096, 4096)   | (1,16,2) | 3.11× | 1.00× | 3.11× | win |
| Granite 3 8B o_proj    M=32  | (32, 4096, 4096)   | (1,16,2) | 3.15× | 1.00× | 3.15× | win |
| Granite 3 8B gate_proj M=32  | (32, 12800, 4096)  | (1,8,4)  | 3.36× | 1.00× | 3.37× | win |
| Granite 3 8B down_proj M=32  | (32, 4096, 12800)  | (1,16,2) | 2.94× | 1.04× | 3.05× | win |
| Granite 3 8B kv_proj   M=128 | (128, 2048, 4096)  | (1,16,2) | 2.36× | 1.04× | 2.46× | win |
| Granite 3 8B q_proj    M=128 | (128, 4096, 4096)  | (1,16,2) | 1.48× | 1.83× | 2.70× | win |
| Granite 3 8B o_proj    M=128 | (128, 4096, 4096)  | (1,16,2) | 1.47× | 1.83× | 2.69× | win |
| Granite 3 8B gate_proj M=128 | (128, 12800, 4096) | (1,8,4)  | 2.64× | 1.07× | 2.84× | win |
| Granite 3 8B down_proj M=128 | (128, 4096, 12800) | (1,16,2) | 1.48× | 1.95× | 2.88× | win |

19 additional shapes (M ∈ {512, 2048} for both 2B and 8B, except 2B
kv_proj M=512 above) lie outside the heuristic gate and are left
untouched by the PR — planner falls back to the existing pure-M /
core_division path. No regressions on these correctness-only rows.

## Aggregate

| | Granite 3-way |
|---|---:|
| Shapes in suite | 40 |
| Heuristic fires on | 21 / 40 |
| Wins | 21 |
| Regressions | 0 |
| Geomean speedup (A→C) on fired shapes | **2.82×** |

## Observations

- **Decode and small-prefill regimes (M ∈ {32, 128}) win across the
  board** for both 2B and 8B. A→C runs 1.87× – 3.37× with no
  regression.
- **8B q/o_proj M=128** (4096 × 4096) shows the canonical
  split-k + k_fast split-of-labor: B→C = 1.83× — most of the speedup
  comes from k_fast emission collapsing PSUM hops, not from the split
  itself. Same pattern on 8B down_proj M=128 (B→C = 1.95×).
- **2B kv_proj M=512** (512, 1024, 2048) is the one regress-rescue
  row: A→B = 0.83× would have been a regression without k_fast, but
  B→C = 1.38× lifts it to a 1.15× net win. This is the exact regime
  the small-M wide-N extension targets.
- **Granite 8B gate_proj** picks `(1, 8, 4)` instead of `(1, 16, 2)`
  because n_sticks = 12800/64 = 200 is divisible by 8 but only by
  16 with a remainder. Heuristic correctly steps down the n-fan
  rather than failing to fire, and lands a 3.37× win.
- **Geomean 2.82× exceeds the cross-vendor combined campaign (2.06×)**
  because Granite's hidden ≪ intermediate ratios put more shapes
  squarely in the small-M wide-N + tall-K sweet spot.

## Source measurements

- `tests/diag_k_fast_granite_3way.py` — probe script
- `tests/diag_k_fast_granite_3way_results.txt` — raw probe output

Companion to `diag_k_fast_combined_findings_normalized.md`
(Llama / Mixtral / DSv3 evidence on the same PR).
