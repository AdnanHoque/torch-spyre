# Prefill SwiGLU -> down-projection 16x2 completion

Date: 2026-07-27

## Decision

Promote the `16 token shards x 2 output cohorts` experimental topology into
the working relayout stack.  It passes the user-selected chicken-soup token
gate, is materially faster than the P06-complete stack, fits the 20% LX
budget, and emits LX-only transport.

Do not promote the earlier `32x1` topology: although correct and shuffle-free,
its one-request device time was `291.472 ms` because the down projection lost
output parallelism.

## Correctness and timing

Granite 3.3 8B, B1/S512, full 40-layer prefill, one generated token,
`DXP_LX_FRAC_AVAIL=0.2`:

- Chicken-soup warmup plus all five measured requests generated token `203`.
- Device kernel/program sums (ms):
  `249.087, 248.823, 248.937, 249.096, 248.897`.
- Median: `248.937 ms`.
- Mean: `248.968 ms`.
- Range: `248.823-249.096 ms`.
- Zero-duration events: `0`.
- P06-complete reference median: `267.179 ms`.
- Delta: `-18.242 ms` (`-6.83%`, `1.0733x`).
- The 40 transformer blocks fall from a reference median of `263.373 ms` to
  `245.185 ms`; the improvement is in the repeated block, not the final head.

The reference trace has one zero-duration event in its third request, but its
five-request median is unaffected.  The candidate has no such event.

## Topology and emitted proof

Planner edge: `buf56 -> buf57`, classified as `all_gather`.

- Source: 32 unique token slices (`source_device_dim_splits={0: 32}`).
- Destination: 16 unique token slices replicated across two output cohorts
  (`destination_device_dim_splits={0: 16}`, `destination_size_ratio=2`).
- Source allocation: `409600 B/core` at LX address `819200`.
- Destination allocation: `819200 B/core` at LX address `0`.
- Combined live footprint: `1228800 B/core` (`1.172 MiB/core`), within the
  configured 20% LX budget.
- Post-PCFG payload: one `STCDPOpLx`; zero `STCDPOpHBM`,
  `ReStickifyOpHBM`, and `DmaOp` markers.

## Validation

- Targeted oracle test: `1 passed, 30 deselected`.
- Python and runner syntax checks passed.
- `git diff --check` passed for the touched repo files.
- The isolated patch was reverse-checked against the live result and
  forward-apply-checked against a reconstructed pre-16x2 tree.

## Artifacts

- `swiglu_downproj_16x2.patch`: isolated source/test patch.
- `run_p06_all_working_e2e.sh`: exact runner with configurable MLP split.
- `swiglu_downproj_16x2_5x_trace.json`: five-request Kineto trace.
- `swiglu_downproj_16x2_plans.jsonl`: relayout planner evidence.
- `swiglu_downproj_16x2_allocations.jsonl`: LX allocation evidence.
- `swiglu_downproj_16x2_stcdp.json`: post-PCFG emitted transport.

Pod run root:
`/home/adnan/codex-isolated/device_parity_tracks_20260726/p06_completion/runs/swiglu_downproj_16x2_timing_5x_20260727_b`

No commit, push, merge, or PR was created.
