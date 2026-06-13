# CORRECTION: the cross-pod gap is DXP_LX_FRAC_AVAIL, NOT a DeepTools regression

Supersedes the deeptools-version conclusion in `DEEPTOOLS_VERSION_PIN.md`,
`DEEPTOOLS_REGRESSION_NOTE.md`, and the "DeepTools codegen build" framing in
`CROSS_RUN_RESULT.md`. **There is no DeepTools version regression. Do not file one.**

## What it actually is
The codegen reads `DXP_LX_FRAC_AVAIL` (LX scratchpad fraction it may use) at the
SDSC→program step — *after* the SDSC (which is why SDSCs are byte-identical but
programs differ). My 240-split oracle ran at the torch-spyre **default 0.2**;
Codex's codegen ran at ~**0.8**. That single knob is the whole 2.2x.

## Proof (my harvest +932 deeptools, same stack as my oracle)
| split | @ LX=0.2 | @ LX=0.8 | Codex pod |
|---|--:|--:|--:|
| QK^T prefill `1_4_8_1` | 1634 us | 743 us | 735 us |
| QK^T decode `8_2_1_2` | 202 us | 90 us | 90 us |
| attn@V decode `1_32_1_1` | 343 us | 94 us | 94 us |
| control `2_8_2_1` | 60 us | 60 us | 60 us |

- Both deeptools builds are fast at 0.8: +932@0.8 = 743 us == +549@0.8 = 743 us
  == Codex +549 = 735 us. Version-independent.
- The control is unchanged at both LX values — LX-frac only rescues the
  HBM-rebound "bad" splits (the residency lever), not already-resident ones.
- Earlier "+489/+549 slow on my pod" was simply LX=0.2; "Codex +549 fast" was LX~0.8.

## Consequences
1. The Claude 240-split oracle (run at LX=0.2) understates LX residency -> the
   "bad split" penalties are inflated. Re-run at the representative LX frac.
2. This is the same residency lever the project kept attributing the Granite gap
   to (array-fill / LX). It's an env knob, not a code change: torch-spyre
   `config.dxp_lx_frac_avail` defaults to 0.2 (`DXP_LX_FRAC_AVAIL`).
3. Open question for the *representative* oracle: what LX frac does e2e Granite
   actually run at? 0.8 is safe for a standalone matmul but LX is shared across
   the graph in e2e, so the per-op budget may be lower. Pin the e2e value, then
   the work-division cost model should be calibrated at THAT LX frac.
