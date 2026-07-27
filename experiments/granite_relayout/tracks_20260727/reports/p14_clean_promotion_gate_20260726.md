# P14 clean promotion gate

Status: **device-valid and performance-positive; source patch still needs hunk minimization before promotion.**

## Pinned source/runtime

- Torch-Spyre: `59545440f0e7091ff1b2f90df63580da1842f3fe`
- FMS: `61bc991b175103e80cb8202b24a66ba7dbe79d1b`
- DeepTools: `406142afb9f080b9271e7c565a757ab8d8b5ed8f`
- Pod: `a6-quantization/adnan-spyre-dev-pf`
- Clean detached worktrees: `/home/adnan/codex-isolated/device_parity_tracks_20260726/p14/p14_promotion_clean_20260726`
- `DXP_LX_FRAC_AVAIL=0.2`
- No commit, push, PR, or merge was performed.

The clean DeepTools worktree contains the four recorded compatibility-overlay files and is byte-identical to the source subset used by the validated existing `p14/deeptools-build`. The binary was not rebuilt from the detached path.

## Focused gates

- Torch P13/P14 compiler tests: `6 passed, 25 deselected`
- FMS last-token policy tests: `3 passed, 9 deselected`
- The FMS patch includes the mandatory `materialize_decoupled_head_for_spyre(..., padded_vocab=50176)` serialization support.

## Fresh full-model T-C-T-C

Runs are below the clean worktree's `runs/` directory:

| Run | Planner | Final-stage mean (us) | Zero final-stage events |
| --- | ---: | ---: | ---: |
| `p14_clean_t1_5x_20260726_aa` | 1 | 4161.4654 | 0 |
| `p14_clean_c1_5x_20260726_ab` | 0 | 4184.7732 | 0 |
| `p14_clean_t2_5x_20260726_ac` | 1 | 4018.7328 | 0 |
| `p14_clean_c2_5x_20260726_ad` | 0 | 4292.6540 | 0 |

Across the two brackets, treatment averaged `4090.0991 us`, control averaged `4238.7136 us`, and the targeted saving was `148.6145 us` (`3.506%`). Block/whole-request traces contain profiler-zero events and unrelated variance, so this supports only the targeted final-stage claim.

All four runs saved six FP16 logit tensors. Using T1 as the reference, C1, T2, and C2 each compare bit-exactly:

- `0 / 295,680` unequal elements
- maximum absolute error `0`
- no shape or metadata mismatches
- every saved `next_val` is token `203`

## Emitted transport proof

Both treatments emitted byte-identical `7_shuffle` artifacts:

- relayout debug SHA-256: `74a995c62ec4f87996948e30b53b4db13d48585e9a51b64fe640017dacc8c542`
- original shuffle SDSC SHA-256: `69cb1b6b0c20008d6c30431f3b408c0ce0fa58581f25784d18f24924c706205f`
- isolated SDSC SHA-256: `fb7e34353e14f6d812dd740505a5e629a4d2bfae720253ab3bbab825ca3b1cbf`
- emitted init binary SHA-256: `7f72a9347843374567f7fe51986a1b73e47709974e99c47cb2c95fe4badf8e1d`

The post-DXP payload is `STCDPOpLx`. Its two labeled operands have LX memory organizations only. The source is four 1x1024 FP16 pieces on cores 28-31; the destination is 32 1x128 pieces, one on each core. DXP records 32 transfers of 256 bytes: 31 remote and one local, for exactly 7,936 remote bytes. The decoded init packet contains LX load/store transfer instructions.

The standalone decoded wrapper also contains one L3LU `LDU`, so the proof is specifically that the claimed P14 edge has no HBM operand/end point; it is not a claim that the entire standalone wrapper has zero L3 instructions.

## Patch artifacts and caveat

- `p14_torch_clean_selected_files.patch`: `6763909bb3f59d7eb979bb6ef9a7727acd5fca5f64a217998e3353b9a3149e6c`
- `p14_fms_clean.patch`: `befc6c2ec58be5073e7c5fc97e5204c6437dbd6fc7fa3ae2d1d946f8d91404a5`
- `p14_deeptools_compatible_clean.patch`: `95f3fe7249184ee91a49d66718dfbaf24643fda0667a7d6d5c7bec02053e21f5`

All three artifacts pass reverse `git apply --check` against their corresponding validated detached worktree.

The Torch selected-file patch is reproducible from the pinned clean head, but it is not yet hunk-minimal: earlier disabled P03/P05/P12 experiment infrastructure is interleaved in the seven required files. The DeepTools compatible patch likewise retains the complete four-file overlay paired with the validated binary. Before promotion, reduce the Torch patch to the P13/P14 and generic sparse-owner hunks and rerun at least the focused tests plus one fresh treatment/control bracket.
