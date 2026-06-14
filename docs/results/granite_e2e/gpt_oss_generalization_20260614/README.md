# GPT-OSS generalization test of `cost-model-physics` (b473bff)

Out-of-sample test: does the physics cost model — calibrated only on Granite
(all power-of-2 dims) — pick device-best on GPT-OSS-20b shapes it never saw?

## Method
- Model: `cost-model-physics` @ b473bff (the lean version: general terms only,
  no `small_output_bmm` gate).
- Picks: real SDSC `numWkSlicesPerDim_` from a compile, via pure-Python overlay
  of b473bff `work_division.py` (no rebuild).
- Device-best: 382-split sweep, **profiled** `self_device_time_total`, at
  `DXP_LX_FRAC_AVAIL=1`, `LX_PLANNING=0`. (Profiler gave real device times for
  every shape on this pod, incl. attention — no 0.0us issue here.)
- GPT-OSS-20b dims: hidden 2880 (**45 sticks, non-pow2**), n_heads 64,
  **head_dim 64 (1 stick → QK^T K / attn@V N cannot split)**, n_kv 8,
  intermediate 2880, 32 experts top-4. Prefill M=512, decode M=64.

## Result: generalizes well; one real, *general* gap (non-pow2 split preference)

| op | shape B·M·N·K | pick | dev-best | gap |
|---|---|---|---|--:|
| Q proj pre | 1·512·4096·2880 | `1_4_8_1` | `1_4_8_1` | +0% |
| Q proj dec | 1·64·4096·2880 | `1_4_8_1` | `1_8_4_1` | +8% |
| K/V proj pre | 1·512·512·2880 | `1_8_4_1` | `1_4_8_1` | +2% |
| K/V proj dec | 1·64·512·2880 | `1_4_8_1` | `1_8_4_1` | +9% |
| **O proj pre** | 1·512·2880·4096 | `1_8_3_1` | `1_2_1_16` | **+38%** |
| **O proj dec** | 1·64·2880·4096 | `1_2_15_1` | `1_8_1_4` | **+155%** |
| MoE up pre | 1·64·2880·2880 | `1_2_5_3` | `1_4_3_1` | +0% |
| MoE down pre | 1·64·2880·2880 | `1_2_5_3` | `1_2_1_5` | +1% |
| QK^T pre | 64·512·512·64 | `1_16_2_1` | `1_16_2_1` | +0% |
| QK^T dec | 64·64·576·64 | `8_4_1_1` | `4_8_1_1` | +11% |
| attn@V pre | 64·512·64·512 | `1_32_1_1` | `4_1_1_4` | +1% |
| attn@V dec | 64·64·64·576 | `32_1_1_1` | `1_8_1_1` | +0% |

**9/12 within 10%, median +2%.** Aggregate +17% is entirely O proj.

## Findings
1. **Not Granite-overfit.** The shapes most likely to break don't: attention
   (head_dim=64, so K or N is a single un-splittable stick) is +0–11%, and MoE
   (both N and K non-pow2, 45 sticks each) is +0–1%. If Granite calibration had
   leaked, these would have broken.
2. **One real gap — O proj — and it is a *general* blind spot, not a special
   case.** When one dim is cleanly power-of-2 and the other is awkward
   non-pow2, the model splits the *awkward* one. O proj: N=2880 (45 sticks,
   awkward), K=4096 (64 sticks, clean). Device K-splits the clean dim
   (`k16`/`k4`, even 4-stick tiles) and leaves N whole; the model N-splits 45
   into ugly `n15` (3-stick) tiles it has no term to penalize. The gap appears
   *only* when a clean alternative exists — MoE (both dims awkward) is fine.
3. **Cross-validation with the Codex-pod GPT-OSS run** (different shapes: M=11
   prefill, ctx-specific attention; wall-clock timing): O proj independently
   shows the same direction — model `1_1_15_2` (N-split) vs measured best
   `1_1_1_32` (K-split). Codex's small-M (M=11) prefill stress additionally
   shows the model over-prefers N/batch splitting and under-uses K-split / M-
   locality when M cannot fill the array. (His magnitudes are wall-clock on
   tiny kernels — directional, not precise; his profiler reported 0.0us so he
   couldn't use device time. This pod's profiler worked, so the table above is
   profiled.)

## Implication
The fix this points to is **general and physics-shaped, not a special-case**:
a *split-evenness / tile-granularity* term that prefers splitting cleanly-
divisible dims and penalizes uneven non-pow2 tiles (and, per Codex, better
small-M-prefill handling — prefer K-split / M-locality over N/batch when M can't
fill PT). This would help any model with awkward dims. It is exactly the right
kind of term for a physics model — and it's the one thing to add next, validated
out-of-sample, rather than chasing the last Granite percent.

## Files
- `gpt_oss_gap.csv` — table above, machine-readable
- `gpt_oss_picks.txt` — b473bff picks
- `gpt_oss_sweep.txt` — 382-split profiled device times (the oracle)
