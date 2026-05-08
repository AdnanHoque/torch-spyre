# Combined k_fast PR (1932 + 1933 + small-M extension) — findings

Branch: `AdnanHoque/feat-k-fast-combined`, cut from `main`.

Combines:
- **PR 1932**: k_fast core-id permutation (packs K-collaborators
  adjacent on the SFP ring)
- **PR 1933**: planner override picking `(1, n, k>1)` for narrow-N
  small-M matmul shapes
- **Extension**: drop the `n_sticks ≥ 32` gate when M ≤ 128, capturing
  small-M wide-N wins the original heuristic skipped

## TL;DR

The combined PR with the small-M extension delivers **10/12 production
wins, 0 regressions** on the 3-way measurement campaign. Total wall
saved across the suite is **12.0 ms**. Geomean speedup is **2.06×**
on shapes where the heuristic fires.

The critical empirical finding: **PR 1932 (k_fast emission) is
load-bearing for at least one row**, not a polish. L3-70B kv_proj
M=512 measures A→B = 0.79× (K-split alone regresses) and B→C = 1.61×
(k_fast rescues to a 1.28× win). Without PR 1932, the heuristic would
ship a measured regression on this row. PR 1932 also contributes
incrementally on most other fired rows (B→C ratios of 1.0-1.85×).
The two PRs must land together.

> **Note on measurement.** Numbers below are from a fresh build
> (May 8, 2026) on the rebased PR branch (against current upstream
> main with the work_division refactor). An earlier "v2" run of the
> same probe under an older build showed smaller speedups (1.33×
> geomean) because a ~3 ms host-side launch floor was diluting
> sub-millisecond kernel improvements; the underlying decisions are
> the same in both. See `diag_k_fast_combined_v3_postrebase_findings.md`
> for the v2-vs-v3 delta.

## 3-way measurement design

For each shape:

  - **A — main baseline**: pure-M (32, 1, 1), identity emission
  - **B — K-split + id**: heuristic-picked split, identity emission
  - **C — K-split + kf**: heuristic-picked split, k_fast emission

Deltas:

  - **A → B**: gain from picking K-split alone (better PT util, per-cluster bytes)
  - **B → C**: gain from k_fast emission (PSUM hops m·n → 1)
  - **A → C**: combined PR effect

This separates the planner-heuristic contribution from the emission
contribution. A row where A→B is large and B→C is small means the
split itself does the work; a row where A→B is negative and B→C is
positive means k_fast is rescuing a regression.

## Per-shape data — extended heuristic

All times in ms. Bold cells highlight the row where K-split alone
regresses (A→B < 1) and k_fast emission rescues to a real win.

| shape | (M, N, K) | h-split | A | B | C | A→B | B→C | A→C | combined |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| L3-70B kv_proj M=32 | (32, 1024, 8192) | (1,16,2) | 0.46 | 0.18 | 0.18 | 2.49× | 1.03× | **2.57×** | win |
| L3-70B kv_proj M=128 | (128, 1024, 8192) | (1,16,2) | 0.48 | 0.19 | 0.20 | 2.48× | 0.98× | **2.43×** | win |
| L3-70B kv_proj M=512 | (512, 1024, 8192) | (1,16,2) | 0.47 | **0.60** | 0.37 | **0.79×** | **1.61×** | **1.28×** | win (kf rescue) |
| Mixtral kv_proj M=128 | (128, 1024, 4096) | (1,16,2) | 0.25 | 0.11 | 0.11 | 2.26× | 1.04× | **2.35×** | win |
| DSv3 kv_proj M=128 | (128, 1536, 7168) | (1,8,4) | 0.61 | 0.34 | 0.29 | 1.77× | 1.17× | **2.07×** | win |
| DSv3 q_a_proj M=128 | (128, 1536, 7168) | (1,8,4) | 0.62 | 0.35 | 0.30 | 1.77× | 1.17× | **2.08×** | win |
| L3-70B q_proj M=32 | (32, 8192, 8192) | (1,16,2) | 3.40 | 1.06 | 1.04 | 3.22× | 1.02× | **3.28×** | new win |
| DSv3 gate_proj M=32 | (32, 18432, 7168) | (1,16,2) | 6.59 | 3.71 | 3.67 | 1.77× | 1.01× | **1.79×** | new win |
| L3-70B q_proj M=128 | (128, 8192, 8192) | (1,16,2) | 3.60 | 2.36 | 1.28 | 1.53× | 1.85× | **2.82×** | new win |
| L3-70B q_proj M=512 | (512, 8192, 8192) | — | 3.46 | — | — | — | — | — | (skipped, correct) |
| DSv3 down_proj M=128 | (128, 7168, 18432) | (1,16,2) | 6.83 | 4.25 | 3.86 | 1.61× | 1.10× | **1.77×** | new win |
| L3-70B kv_proj M=2048 | (2048, 1024, 8192) | — | 1.21 | — | — | — | — | — | (skipped, correct) |

## What the extension changes

The change to `_try_k_fast_split` is one line in spirit, ~4 in code:

```python
- if n_sticks >= 32:    # pure-N (1, max_cores, 1) already valid
+ if M > 128 and n_sticks >= 32:
      return None
```

Plus an updated comment explaining the regime structure.

The motivation: at small M, pure-M's `M_per ≤ 4` under-utilises the
PT array (M_per < 8 PT rows). K-split keeps full M per core
(`M_per = M`), giving full PT util. The `n_sticks` gate was guarding
against the medium-M regression regime (M=512 measured 0.59× under
K-split), which is preserved at M > 128.

## Why PR 1932 is load-bearing

L3-70B kv_proj M=512 is the smoking gun — the row where K-split alone
(B) measures slower than pure-M (A):

| row | A | B (k-split + id) | C (k-split + kf) | rescue magnitude |
|---|---:|---:|---:|---:|
| L3-70B kv_proj M=512 | 0.47 | **0.60** (0.79×) | 0.37 | kf gives 1.61× over id |

On this row, the planner picks K-split because it has good PT util
and HMI-byte properties — but the SFP-ring chain cost under identity
emission (m·n hops per send) eats the gain. The k_fast permutation
collapses chain hops to 1, recovering the wall and turning a
measured regression into a measured 1.28× win.

This empirically confirms the design rationale of PR 1932:

> "The companion override is harmful *without* this PR — forced K-split
>  shapes would traverse m·n ring hops per PSUM chain instead of 1,
>  running slower than the planner's pure-M default."

PR 1932's contribution doesn't end there: on every other fired row
its B→C ratio is 1.0-1.85×, contributing incremental speedup on top
of the K-split's primary gain. But the strict requirement comes from
this single regression-rescue case — without PR 1932 in flight,
PR 1933 would land a measured regression on L3-70B kv_proj M=512.

(Note: under the older v2 build, two additional rows — L3-70B q_proj
M=128 and DSv3 down_proj M=128 — also showed K-split-alone
regressions that k_fast rescued. Under the cleaner v3 build, those
two rows show K-split alone winning outright; k_fast still
contributes incremental speedup but isn't strictly necessary on
those specific rows. The L3-70B kv_proj M=512 case is reproducible
across both builds and is the load-bearing example.)

## Where the gains come from per shape

Decomposing each row's A→C gain into the K-split component (A→B)
and the k_fast emission component (B→C):

| primary contributor | rows |
|---|---|
| K-split dominates (A→B ≥ 1.4×, B→C ≤ 1.2×) | L3-70B kv_proj M=32, M=128; Mixtral kv_proj M=128; DSv3 kv_proj M=128; DSv3 q_a_proj M=128; L3-70B q_proj M=32; DSv3 gate_proj M=32; DSv3 down_proj M=128 |
| Roughly balanced (both layers contribute meaningfully) | L3-70B q_proj M=128 (A→B 1.53×, B→C 1.85×) |
| k_fast emission essential (A→B < 1, B→C > 1.4×) | L3-70B kv_proj M=512 |

On most rows, the K-split decision itself is doing the heavy lifting
(better PT utilization at small M_per, fewer per-cluster HMI bytes
under K-split). PR 1932 contributes 0-85% additional speedup on top
of the K-split benefit — useful but not required for those rows.

The L3-70B kv_proj M=512 row is the load-bearing case where PR 1932
is strictly required to avoid a regression.

## What the heuristic still skips (correctly)

| shape | reason | measured outcome if forced (v2 build) |
|---|---|---|
| L3-70B q_proj M=512 (8192) | M > 128, n_sticks ≥ 32 | (1,16,2)+kf is 10.92 vs pure-M 6.41 — 0.59× regression |
| L3-70B kv_proj M=2048 | M > 512 | (1,16,2)+kf is 3.95 vs pure-M 3.65 — 0.92× regression |

The extended heuristic's guards correctly hold these out. The
campaign's "n/a (heuristic skip)" rows on these are *successful*
rejection: the heuristic correctly identifies that K-split would
hurt here.

(The forced-K-split numbers in the right column are from v2
measurements; we didn't reproduce them under the v3 build since the
heuristic correctly skips these shapes and the campaign reports
pure-M only. The v3 pure-M walls are 3.46 ms and 1.21 ms
respectively — same shape, same skip decision, just lower base
overhead in the new build.)

## DSv3 gate_proj M=32 — a sub-optimal but still-winning pick

The extension probe (v2 build) found `(1, 4, 8)+kf` runs at 5.40 ms
on this shape versus `(1, 16, 2)+kf` at 6.41 ms (the heuristic's
pick). Both beat pure-M (9.41 ms) at 1.74× and 1.47× respectively.
Under the v3 build, the heuristic's `(1, 16, 2)+kf` pick measures
3.67 ms vs pure-M at 6.59 ms — a **1.79×** win.

The heuristic picks `(1, 16, 2)+kf` because the loop iterates `n` in
`(16, 8, 4, 2)` and returns the first match. For DSv3 gate_proj
M=32 N=18432 K=7168, n=16 divides cleanly so it's selected.

Switching to `(1, 4, 8)+kf` would yield a small additional speedup
on this specific shape, but on the other small-M wide-N rows
(L3-70B q_proj M=32, M=128) `(1, 16, 2)+kf` is already the right
answer. A more sophisticated pick logic that distinguishes these
cases is left for future work — the simple `n_sticks` gate
relaxation captures the bulk of the wins (1.79-3.28× on the four
new shapes the extension enables).

## Aggregate

| | PR 1933 as-shipped | Combined + extension |
|---|---:|---:|
| Heuristic fires on | 6/12 | 10/12 |
| Wins | 5 | 10 |
| Regressions | 0 | 0 |
| Total wall saved | 1.4 ms (v2 build) | 12.0 ms (v3 build) |
| Geomean speedup (A→C) | 1.07× | **2.06×** |

The extension is a substantial lift in measured production benefit
with no new regressions. The combined branch is hardware-validated
against the same shapes that exercise the original PRs plus four
new shapes that the original heuristic skipped.

The "PR 1933 as-shipped" column above is from v2 measurements (the
original PR's evidence base); the "Combined + extension" column is
from v3 (post-rebase, fresh build). Direct apples-to-apples
comparison between the two columns isn't strictly possible because
the build environments differ — but the structural conclusion holds
under either build: PR 1933 alone fires on 6/12 shapes; the
extension adds 4 more correctly-firing shapes; geomean speedup
roughly doubles.

## Files on the branch

- `torch_spyre/_inductor/core_division.py` — heuristic with
  small-M extension
- `torch_spyre/_inductor/codegen/compute_ops.py` — k_fast permutation
  (from PR 1932)
- `torch_spyre/_inductor/config.py` — combined `core_id_k_fast_emission`
  flag with merged docstring
- `tests/inductor/test_k_fast_planner.py` — updated unit tests
  (24/24 pass), 2 tests changed for new firing behaviour, 2 new tests
  added for the small-M extension
- `tests/inductor/test_k_fast_emission.py` — unchanged from PR 1932
- `tests/diag_k_fast_combined_3way.py` — 3-way measurement probe
- `tests/diag_k_fast_combined_3way_results.txt` — initial run
  (PR 1933 as-shipped)
- `tests/diag_k_fast_combined_3way_v2_results.txt` — extended-heuristic run, v2 build
- `tests/diag_k_fast_combined_3way_v3_postrebase_results.txt` —
  extended-heuristic run on rebased PR + fresh v3 build (numbers in
  the per-shape table above)
- `tests/diag_k_fast_combined_v3_postrebase_findings.md` — v2-vs-v3
  delta analysis
- `tests/diag_k_fast_extension_candidates.py` — Phase 3 probe that
  identified which split to pick
- `tests/diag_k_fast_extension_candidates_results.txt` — Phase 3 raw output

## Path to merge

The branch has all three pieces hardware-validated:

- Combined PR 1932 + PR 1933: 24/24 unit tests pass; 5/6 in-band
  shapes win, 0 regress
- Extension: the n_sticks gate relaxation captures 4 additional
  wins on small-M wide-N shapes; 0 regressions

For a clean merge:
1. Keep the cherry-picks of `ebb0557` (PR 1932) and `b21e108`
   (PR 1933 + extension as a single commit, with the new tests)
2. Squash or keep the diag commits per repo convention
3. Open as a single PR with the campaign findings doc as the PR
   description's evidence base

The 0.59× (L3-70B q_proj M=512) and 0.92× (L3-70B kv_proj M=2048)
regression rows from forced K-split runs (v2 build) are not in any
tier the heuristic fires on, so the merge can ship without further
mitigation. They're useful as documented "known regression regimes
the heuristic correctly avoids." On the v3 build, the heuristic
correctly skips these shapes (pure-M is the chosen and measured
output).
