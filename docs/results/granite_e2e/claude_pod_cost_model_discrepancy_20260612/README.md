# Claude-pod mirror of the cost-model discrepancy experiment

Mirror of `codex_pod_cost_model_discrepancy_20260612`, run on the Claude pod
(harvested SDK on clc `ba:00.0`). Same disputed forced splits, same extraction:
each split is forced via the cost-function patch, compiled, and the emitted
SDSC + a 20-rep device timing is captured. Purpose: answer Codex's diagnostic —
*"same split string, same SDSC? If yes the discrepancy is runtime/backend/device;
if no, it is lowering/layout/environment."*

## Headline answer: it is runtime / backend / device

**12 of 12 forceable splits produce a byte-identical SDSC across the two pods**
(full sha256 match), yet the measured device time differs by **1.0×–3.6×**. Same
compiler output, different execution cost → the discrepancy is in the
backend/runtime/device, **not** lowering or layout.

| split | SDSC (mine vs codex) | Claude µs | Codex µs | Claude/Codex |
|---|---|--:|--:|--:|
| prefill QK^T `1_2_8_2` | identical | 1009 | 747 | 1.35× |
| prefill QK^T `1_4_8_1` | identical | 1635 | 735 | 2.23× |
| prefill QK^T `4_1_8_1` | identical | 2258 | 731 | 3.09× |
| prefill attn@V `1_16_2_1` | identical | 501 | 198 | 2.53× |
| prefill attn@V `1_32_1_1` | identical | 797 | 327 | 2.43× |
| prefill attn@V `2_8_2_1` | identical | 404 | 204 | 1.98× |
| prefill attn@V `4_4_2_1` | identical | 395 | 198 | 2.00× |
| decode QK^T `32_1_1_1` | identical | 258 | 203 | 1.27× |
| decode QK^T `4_4_1_2` | identical | 159 | 104 | 1.53× |
| decode QK^T `8_2_1_2` | identical | 204 | 90 | 2.27× |
| decode attn@V `1_32_1_1` | identical | 343 | 94 | 3.63× |
| decode attn@V `2_8_2_1` | identical | 60 | 60 | 0.99× |
| decode QK^T `1_4_3_2` | n/a — planner can't emit | — | 105 | — |
| decode attn@V `1_4_2_3` | n/a — planner can't emit | — | 55 | — |

## The mechanism: Codex's backend rescues bad splits; mine doesn't

The slowdown is **split-dependent**, and that is the whole story:

- On a *good* split the pods agree exactly — decode attn@V `2_8_2_1` is 60 µs on
  both (1.0×).
- On a *bad* split they diverge hard — decode attn@V `1_32_1_1` (pure-M) is 94 µs
  on Codex's pod but **343 µs** on mine (3.6×). Same SDSC.
- Codex's QK^T-prefill timing is nearly **split-invariant** (731 / 735 / 747 µs
  for `4_1_8_1` / `1_4_8_1` / `1_2_8_2`); mine spreads 1009 → 2258 µs for the
  same three.

So Codex's backend applies a program-level optimization (most likely
weight-preload / double-buffering — the `libbaseOptimizer` path that the
standalone `dxp_standalone --bundle` entry can bypass) that **compresses the
difference between splits**, lifting inefficient splits toward the efficient
ones. My harvest stack applies it weakly or not at all, so inefficient splits
stay slow.

This explains the four-way result directly: on Codex's pod every cost model
lands near device-best because the backend rescues bad picks; on my pod the
split choice is load-bearing, so the cost-model differences are large.

## Caveats / unreproducible rows

- `decode_QKT 1_4_3_2` and `decode_attnatV 1_4_2_3` are 24-core / non-power-of-2
  splits that **cost-model-tuned** emits but the latest-main planner does not
  enumerate. Forcing them on my pod fell back to the default (`32_1_1_1` /
  `1_32_1_1`), so I cannot diff their SDSC or time them natively. (This also
  means the earlier four-way "measured" those two rows as the fallback split —
  corrected here.)
- My timing is a single-session 20-rep **mean** (<1% noise on this pod), not the
  20-sample median Codex reports. Central values are directly comparable.
- This pod does **not** emit `ideal_cycles.json` (Codex's does) — another
  backend/deeptools difference, recorded but not yet attributed.

## What this means

Neither standalone-probe oracle is "canonical." My pod **overstates** the
work-division lever (no backend rescue → split choice dominates); Codex's pod
**understates** it (backend rescue → split choice barely matters). Which one is
representative depends on the backend the **e2e Granite deployment** actually
runs — that, not either standalone oracle, is the reference to settle on.

## Next step to attribute the backend difference

Same SDSC → the difference is in deeptools / flex / runtime / firmware. The
metadata blocks do not yet pin Codex's versions. To localize it, diff across
pods, for one disputed split (e.g. prefill QK^T `1_4_8_1`):
deeptools version, flex version, runtime/firmware, and the generated **program**
(`execute/` PAGI / DSG), not just the SDSC. `ideal_cycles.json` present on one
pod and absent on the other is the first concrete lead.

## Files

- `artifacts/<label>/` — `sdsc_0.json`, `bundle_0.mlir`, `metadata.json` (20-rep timing) per split
- `cross_pod_comparison.json` / `claude_pod_disputed_split_metadata.json` — the table above, machine-readable
- `stack_versions.txt` — Claude-pod stack + env
