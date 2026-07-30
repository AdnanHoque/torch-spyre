# Failure evidence summary

## Packaged stock QFP8CH artifacts

Both files describe M=512, K=N=4096 Torch-Spyre SuperDSCs before the QFP8MB
PoC:

| File | Work division | Result |
|---|---|---|
| `failures/qfp8ch_auto_m512_sdsc_8.json` | compiler-selected M:32 x N:1 | DeepTools rejected the plan for insufficient legal per-core elements |
| `failures/qfp8ch_8x4_m512_sdsc_8.json` | forced M:8 x N:4 | DeepTools rejected the QFP8CH layout/coordinate plan |

They establish that merely forcing the desired outer work division on the
existing channel-packed activation layout is insufficient.

## QFP8MB compiler progression

The pre-corelet-PoC bundle and final input SuperDSC are in `qfp8mb_v3/`. The
post-corelet-PoC diagnostic log is `logs/dxp_v3_corelet_preload.log`. The
progression observed in those runs was:

1. A naive explicit-rank physical representation corrupted the logical loop
   space.
2. Keeping canonical logical ranks and serializing the compound physical
   factors allowed the M:8 x N:4 outer plan to pass initial propagation.
3. Direct DeepTools corelet selection chose N=512+512 and failed at
   `dsc2.cpp:6379: TO DO: Loop split needed?`.
4. The narrow DeepTools PoC selected M=8+8 for QFP8MB and M=32+32 for the
   matmul. The line-6379 failure disappeared.
5. Compilation then failed later at `dsc2.cpp:5862` during compound-coordinate
   distribution.

The log records the final `M=8+8` QFP8MB split, the final `M=32+32` matmul
split, and the exact line-5862 exception. The last step is the current
boundary. It is structural progress, not a successful compile or timing
result.
