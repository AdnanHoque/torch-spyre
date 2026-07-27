# P05 in the P06-complete prefill stack

Date: 2026-07-27

## Result

P05 (`buf47 -> buf48`, residual to RMSNorm reduction) executes in the current
`P01/P02/P03/P04/P06/P12/P13/P14` stack at `DXP_LX_FRAC_AVAIL=0.2`.

The chicken-soup warmup and all five measured requests generated token `203`.
The P05 shuffle emitted one `STCDPOpLx` and no `STCDPOpHBM`,
`ReStickifyOpHBM`, or `DmaOp` markers.  P10/P11 (`buf50`) were disabled.

## Same-pod device timing

All values are sums of Kineto `kernel` event durations for the 42 prefill
kernels in one B1/S512 request.

| Arm | Five requests (ms) | Median (ms) |
| --- | --- | ---: |
| P05 off | 265.426, 266.322, 266.422, 265.530, 265.941 | 265.941 |
| P05 on | 266.045, 265.526, 265.153, 266.079, 265.705 | 265.705 |

Median delta: `-0.235 ms` (`-0.089%`, `1.0009x`).  This is directionally
positive but too small relative to request variation to claim a meaningful
performance win.  It does establish that adding P05 does not regress the
current stack under the user-selected token correctness gate.

Both arms ran on `a6-quantization/adnan-spyre-current-pf` with zero
zero-duration kernel events.

## Pod artifacts

- P05 on:
  `/home/adnan/codex-isolated/device_parity_tracks_20260726/p05_p06_integration/runs/p05_p06_all_working_5x_20260727_a`
- P05 off:
  `/home/adnan/codex-isolated/device_parity_tracks_20260726/p05_p06_integration/runs/p05_samepod_control_5x_20260727_a`

Historical caveat: P05 previously failed the stricter aligned full-logit gate.
The user subsequently defined the chicken-soup generated token as the only
correctness test for this study.
