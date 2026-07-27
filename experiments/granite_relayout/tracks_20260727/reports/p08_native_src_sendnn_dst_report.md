# P08 native-source to SenDNN-destination bridge

Date: 2026-07-27

## Decision

The correctness-valid bridge is a useful working-stack optimization, but it is
not exact SenDNN P08 source parity.

The route keeps Torch/Deeptools' emitted `buf40` placement and applies the
SenDNN 4 x 8 token-major destination to `buf41`.  Trying to force both planner
endpoints to 32 unique pieces was semantically wrong: Deeptools canonicalizes
the rank-5 `buf40` producer to 16 token pieces on even cores, while the planner
view still describes 32 logical source slices.

## Correctness and timing

Granite 3.3 8B, B1/S512 full 40-layer prefill, one generated token,
`DXP_LX_FRAC_AVAIL=0.2`:

- Chicken-soup warmup plus all five measured requests generated token `203`.
- Device kernel/program sums (ms):
  `247.334, 247.407, 247.618, 247.565, 247.377`.
- Median: `247.407 ms`.
- Mean: `247.460 ms`.
- Range: `247.334-247.618 ms`.
- Zero-duration kernel events: `0`.
- Accepted-stack reference median: `248.937 ms`.
- Delta: `-1.530 ms` (`+0.618%`, `1.00618x`).

Only device kernel/program durations from the Kineto trace are reported.  The
approximately 1.86-2.03 second lines in `run.log` are host end-to-end latency
and are intentionally excluded.

## Emitted topology and transport proof

Planner edge: `buf40 -> buf41`, classified as `all_to_all`.

- Planner source view: 32 logical token slices (`{1: 32, 4: 1}`).
- Planner destination: 8 token cohorts x 4 head/output cohorts
  (`{1: 8, 4: 4}`).
- Emitted input: 16 pieces of `[32 tokens, 4096 hidden]` on even cores
  `0,2,...,30`.
- Emitted output: 32 pieces of `[64 tokens, 1024 hidden]`, one per core.
- Logical tensor: `4 MiB` FP16.
- Overlap accounting: `1 MiB` local and `3 MiB` remote.
- Exact post-PCFG marker counts: one `STCDPOpLx`, zero `STCDPOpHBM`, zero
  `ReStickifyOpHBM`, and zero `DmaOp`.

The five-sample post-PCFG payload is byte-identical to the earlier
correctness-run payload.

## Remaining exact-P08 dependency

Exact 32-source to 32-destination SenDNN parity needs `buf40` canonicalized to
a flattened head/token representation before work division, or an upstream
attention-output route that physically emits the required 4 x 8 source grid.
P07/P09 currently target RoPE-frequency and normalized-QKV grouped gathers;
they do not directly feed `buf40`.  Their integration should only be claimed as
an exact-P08 dependency if their full fused-block output changes the emitted
`buf40` source placement.

## Artifacts

- `p08_native_src_sendnn_dst_5x_trace.json`: five-request Kineto trace.
- `p08_native_src_sendnn_dst_5x_plans.jsonl`: planner evidence.
- `p08_native_src_sendnn_dst_5x_allocations.jsonl`: LX allocation evidence.
- `p08_native_src_sendnn_dst_5x_stcdp.json`: exact post-PCFG payload.

Pod run root:
`/home/adnan/codex-isolated/device_parity_tracks_20260726/p06_completion/runs/p08_native_src_sendnn_dst_timing5_20260727_f`

No commit, push, merge, or PR was created.
