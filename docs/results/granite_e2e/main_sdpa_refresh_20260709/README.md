# Current-main SDPA refresh

Date: 2026-07-09

This refresh reruns the FMS four-head SDPA microbenchmark after updating to
upstream Torch-Spyre main.

## Source and workload

- Torch-Spyre: `e3a79c56efa8543978be3c92ec6a0d09559a53d5`
- Workload: `fms_granite_micro.mha_4h_workdiv_h4_lq8`
- Q: `[1, 4, 512, 128]`
- K/V: `[1, 4, 4096, 128]`
- Runs: 5
- Metric source: Kineto-Spyre profiler
- Runtime/toolchain: current SDK copied from `torch-aiu-runtime-dev:latest`

The aggressive variant changes only
`allow_all_ops_in_lx_planning: False -> True`. It is a research control, not a
proposed production default.

## Results

| Variant | Kernel ms | Spyre ms | Transfer ms | Median wall ms |
|---|---:|---:|---:|---:|
| Current main | 0.521 | 1.508 | 0.987 | 522.522 |
| Current main + aggressive LX eligibility | 0.508 | 1.445 | 0.938 | 518.862 |

Aggressive eligibility improves kernel time by `1.026x` and Spyre time by
`1.044x`. It does not eliminate the remaining layout handoff.

## Structural result

Both variants emit 22 SDSCs and retain one `ReStickifyOpHBM`:

```text
mul (LX output)
  -> ReStickifyOpHBM (LX -> HBM)
  -> score batchmatmul (K operand from HBM)
```

The aggressive variant additionally pins early identity/max intermediates in
LX. It does not realize the all-gather plus stick-layout conversion required by
the remaining matmul operand. That remains PR2 scope.

## Main-head behavior change

The only source delta from the prior `ac3c7395` run is the hint-counter reset in
`e3a79c56`. It prevents each benchmark iteration from receiving different hint
IDs and recompiling the graph. The previous five-run control compiled on every
iteration (median wall `5692.350 ms`); current main compiles once (median wall
`522.522 ms`). Kernel execution itself is unchanged within noise: `0.517 ms`
before versus `0.521 ms` now.

## Artifacts

- `default/`: current-main report, profiler tables, command, SDK identity, and
  Jamie-style SDSC summary.
- `aggressive/`: matching artifacts for the isolated aggressive eligibility
  control, including `source.patch`.
