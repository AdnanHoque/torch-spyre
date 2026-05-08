# v3 measurement campaign — post-rebase, fresh runtime build

Companion to `diag_k_fast_combined_3way_v3_postrebase_results.txt`.
Re-runs the 3-way campaign on the rebased PR branch
(`AdnanHoque/pr-k-fast` against current upstream main, which includes
the `core_division → work_division` refactor) after a fresh full
runtime stack rebuild (Flex/SenDNN/SenBFCC + torch_spyre).

## TL;DR

The combined PR's wins are **larger and cleaner** than v2 measured.
Geomean A→C speedup went from 1.33× (v2) to **2.06× (v3)** with the
same heuristic fire pattern (10/12 shapes) and zero regressions.

The shift is explained by reduced host-side launch overhead in the
fresh build — v2 measurements carried a ~3 ms launch floor on every
sample, which masked sub-millisecond kernel improvements. Under the
new build, wall-time measurements are much closer to kernel-only
latency, so the structural wins emerge undiluted.

## v2 vs v3 side-by-side

| metric | v2 (pre-rebase, May 5 build) | v3 (rebased, May 8 fresh build) |
|---|---:|---:|
| Heuristic fires | 10/12 | 10/12 |
| Wins (A→C ≥ 1.05×) | 9 | **10** |
| Regressions | 0 | 0 |
| Total wall saved (PR-fired shapes) | 13.1 ms | 12.0 ms |
| **Geomean A→C** | **1.33×** | **2.06×** |
| Largest single-shape A→C | 2.00× (DSv3 down_proj M=128) | **3.28× (L3-70B q_proj M=32)** |

Total wall saved is roughly the same in absolute terms; the
**ratio** went up because per-row absolute walls dropped by 5-10×
under the new build. This is exactly the wall-vs-kernel-latency
methodology point we discussed earlier — the launch floor that
diluted v2 ratios is mostly gone in the fresh build.

## Per-shape v2 → v3 deltas

| shape | v2 A→C | v3 A→C | direction |
|---|---:|---:|---|
| L3-70B kv_proj M=32 | 1.09× | **2.57×** | win bigger |
| L3-70B kv_proj M=128 | 1.09× | **2.43×** | win bigger |
| L3-70B kv_proj M=512 | 1.06× | **1.28×** | win bigger (kf rescue still real) |
| Mixtral kv_proj M=128 | 1.05× (neutral) | **2.35×** | promoted to win |
| DSv3 kv_proj M=128 | 1.09× | **2.07×** | win bigger |
| DSv3 q_a_proj M=128 | 1.09× | **2.08×** | win bigger |
| L3-70B q_proj M=32 | 1.58× | **3.28×** | extension win, bigger |
| DSv3 gate_proj M=32 | 1.43× | **1.79×** | extension win, bigger |
| L3-70B q_proj M=128 | 1.32× | **2.82×** | extension + kf rescue, bigger |
| DSv3 down_proj M=128 | 2.00× | **1.77×** | extension win, slightly smaller |
| L3-70B q_proj M=512 | skipped | skipped | correct skip preserved |
| L3-70B kv_proj M=2048 | skipped | skipped | correct skip preserved |

Every PR-fired shape is faster in v3 than v2 except DSv3 down_proj
M=128 (2.00× → 1.77×). On that row the K-split benefit is largely
HMI savings that don't shrink with launch overhead, so the ratio
moved less. Still a clear 1.77× win.

## "Two PRs must land together" — confirmed under cleaner measurement

The load-bearing case is L3-70B kv_proj M=512:

| metric | v2 | v3 |
|---|---:|---:|
| A (pure-M baseline) | 3.38 ms | 0.47 ms |
| B (K-split + id) | 4.91 ms | 0.60 ms |
| C (K-split + kf) | 3.20 ms | 0.37 ms |
| A→B (split alone) | 0.69× regression | **0.79× regression** |
| B→C (kf rescue) | 1.54× | **1.61× rescue** |
| A→C (combined) | 1.06× win | **1.28× win** |

The K-split-alone regression is real in both measurement regimes.
The k_fast emission rescue is real in both. The **combined** PR
ships a clean 1.28× win on this row, but only because both layers
are present. Without PR 1932, the heuristic would land a 0.79×
regression.

## A new finding the cleaner measurement reveals

L3-70B q_proj M=128 shifted from "kf-dominated rescue" (v2: A→B
0.93×, B→C 1.42×) to "split-dominated win with kf bonus" (v3: A→B
1.53×, B→C 1.85×). Combined 2.82×, a notable jump from v2's 1.32×.

This row was previously borderline (small absolute wall wins in v2
mixed with launch overhead). Under cleaner measurement it shows up
as one of the strongest wins in the suite — the heuristic
extension's small-M wide-N regime is paying out more than v2
suggested.

Same direction either way; the v3 numbers just make the case
stronger.

## What this changes for the PR

Nothing structural — same 24/24 unit tests, same heuristic logic,
same scope. But the PR description's evidence base can now point to
the v3 run as the headline measurement (geomean 2.06×) with v2 as
prior-art context.

The wall-vs-kernel-latency caveat we discussed earlier becomes less
necessary to call out explicitly. The v3 build is close enough to
kernel-only that the wins are visible directly. We can still note
that wall measurements remain a conservative lower bound on kernel
latency, but the ratio has shrunk enough that the case stands on
its own.

## Files

- `tests/diag_k_fast_combined_3way.py` — probe (unchanged from v2)
- `tests/diag_k_fast_combined_3way_results.txt` — v1 (pre-extension)
- `tests/diag_k_fast_combined_3way_v2_results.txt` — v2 (post-extension, pre-rebase)
- `tests/diag_k_fast_combined_3way_v3_postrebase_results.txt` — v3 (this run)
- `tests/diag_k_fast_combined_findings.md` — overall combined-PR findings (v2-based)
- This doc — v3 update notes

## Reproducibility

Run on `AdnanHoque/pr-k-fast` (commit a288290 against upstream main 14dd4b4)
after fresh build of:
1. Flex/SenDNN/SenBFCC runtime stack
2. torch_spyre C++ extension

Diag script imports were retargeted from `core_division` to
`work_division` (one-line `sed` change) to match the rebased branch's
module naming.
