# P06 combined-stack completion

Date: 2026-07-27

## Result

P06 now compiles and executes in the current prefill stack
`P01/P02/P03/P04/P06/P12/P13/P14` at `DXP_LX_FRAC_AVAIL=0.2`.

The compile failure came from a graph-shape drift: the P06 oracle recognized
the older `buf14` shape `[1,8,4,512,128]`, while the current fused graph exposes
the same rotary output as `[1,32,512,128]`. The hint therefore dropped at
`buf14`, inserted an unintended `buf13 -> buf14` relayout, and failed on the
flattened planner axis. The patch recognizes both representations and binds the
flattened 32-head axis to four query-head cohorts.

## Acceptance evidence

- Chicken-soup prompt: warmup plus all five measured requests generated token
  `203`.
- Five-request device-kernel sums (ms):
  `267.640, 266.997, 260.336, 267.179, 267.206`.
- Median: `267.179 ms`.
- Prior current-stack median without P06: `273.392 ms`.
- Median delta: `-6.213 ms` (`-2.27%`, `1.0233x`).
- One profiler event in request 3 has zero duration; that request is the
  `260.336 ms` outlier. The other four requests average `267.255 ms`, and the
  five-request median is unaffected.
- P06 planner edge: `buf14 -> buf20`, source `8 x 4`, destination `32 x 1`.
- Emitted post-PCFG P06 payload (`18_shuffle-Relayout`): one `STCDPOpLx`, zero
  `STCDPOpHBM`, zero `ReStickifyOpHBM`, and zero `DmaOp` markers.

## Artifacts

- `p06_flattened_buf14.patch`: integration patch; `git apply --check` passes
  against the Granite relayout checkout.
- `p06_all_working_5x_trace.json`: five-request Kineto trace.
- `relayout_plans.jsonl`: combined-stack relayout plans.
- `origsdsc_debug_18_shuffle.json`: emitted P06 shuffle SDSC.
- `stcdp_after_pcfg_18_shuffle-Relayout.json`: post-PCFG LX payload.
- `run_p06_all_working_e2e.sh`: exact full-model runner.

Pod run root:
`/home/adnan/codex-isolated/device_parity_tracks_20260726/p06_completion/runs/p06_all_working_timing_5x_20260726_b`

No commit, push, merge, or PR was created.
