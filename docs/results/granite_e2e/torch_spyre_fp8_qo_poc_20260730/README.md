# Torch-Spyre FP8 Q/O matmul PoC

This package archives the Torch-Spyre continuation of the standalone SenDNN
FP8 investigation.

**Status:** work in progress. The DD2 FP8 layout and work-division changes
reach DeepTools, and the wrong within-core output split has been replaced with
the SenDNN-like M split. Compilation now fails later while distributing the
compound FP8 coordinates. There is no accepted optimized Torch-Spyre timing
sweep or Granite end-to-end result yet.

Start with [`HANDOFF.md`](HANDOFF.md). The exact source changes are archived as
patches because this results branch and the experimental PR #2286 checkout do
not share a suitable source base.

## Contents

- [`HANDOFF.md`](HANDOFF.md): findings, non-claims, blocker, and continuation.
- [`PROVENANCE.md`](PROVENANCE.md): exact source and stack snapshots.
- [`patches/torch_spyre_qfp8mb_poc.patch`](patches/torch_spyre_qfp8mb_poc.patch):
  16-file experimental patch against Torch-Spyre PR #2286 head.
- [`patches/deeptools_dd2_fp8mb_corelet_poc.patch`](patches/deeptools_dd2_fp8mb_corelet_poc.patch):
  narrow DD2 corelet-selection experiment.
- [`evidence/FAILURE_SUMMARY.md`](evidence/FAILURE_SUMMARY.md): packaged and
  observed evidence.
- [`evidence/failures/`](evidence/failures/): two stock QFP8CH M=512 SuperDSCs
  that failed during compilation.
- [`evidence/results/`](evidence/results/): the FP16 reference and restricted
  forced-1x32 raw FP8 result JSON.
- [`evidence/qfp8mb_v3/`](evidence/qfp8mb_v3/): the pre-corelet-PoC QFP8MB
  bundle and final input SuperDSC.
- [`evidence/logs/dxp_v3_corelet_preload.log`](evidence/logs/dxp_v3_corelet_preload.log):
  post-corelet-PoC compiler log showing the M splits and later line-5862
  failure.
- [`evidence/source/deeptools_corelet_audit_preload.cpp`](evidence/source/deeptools_corelet_audit_preload.cpp):
  diagnostic interposer source used to inspect final corelet choices.
- [`SHA256SUMS`](SHA256SUMS): checksums for the immutable evidence and patches.

The benchmark and serialized runner live at
[`../../../../benchmarks/torch_spyre_fp8_matmul/`](../../../../benchmarks/torch_spyre_fp8_matmul/).
