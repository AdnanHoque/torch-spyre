# Reshard A/B — results

Kernel time = `self_device_time_total` (PrivateUse1 profiler), profiler-build +
harvest stack (see `../profenv.sh`). Device-side, `<1%` noise; runs are slow
(60 s/H2D flex stall) but the kernel metric is unaffected.

## A0 (baseline) vs A1 (steer)

`steer` = `_cost_model_divide_op → False`, confirmed to flip the matmul to
**pure-M** (`{mb:32}`, verified in the SDSC) so the matmul→pointwise edge becomes
**same-division same-core** (neg reads the same per-core HBM base it was written
to) — the cross-division hand-off is genuinely removed.

| shape | op | arm | matmul split | edge | kernel_ms | PT-util |
|---|---|---|---|---|---|---|
| prefill 1×512×4096 | fused | A0 baseline | (m4,n8) | cross-div HBM | **19.8** | 16.9% |
| prefill 1×512×4096 | fused | A1 steer | pure-M | same-div | **27.8** | 12.1% |
| prefill 1×512×4096 | unfused | A0 baseline | (m4,n8) | cross-div HBM | **13.9** | 20.1% |
| prefill 1×512×4096 | unfused | A1 steer | pure-M | same-div | **22.8** | 11.1% |

Steer loses in **both** fused (1.40×) and unfused (1.64×) — consistent. (Decode
A1: not run; decode matmul is tiny / 0.2% util, so steering can't help there and
the movement-bound reshard is the only lever regardless.)

## Verdict

**Steering to pure-M is a net LOSS (27.8 vs 19.8 ms, 1.40×; util 16.9%→12.1%).**
For this wide-N matmul (`N=25600`) the cost model's `(m4,n8)` feeds the PT array
far better than pure-M; that matmul speed outweighs the cross-division hand-off
it creates. So **eliminating the edge by giving up m×n is the wrong trade**.

➡️ **The A2 on-chip reshard is the justified lever**: keep the fast `(m4,n8)`
matmul and eliminate the cross-division HBM hand-off via an on-chip LX↔LX
core-to-core move (the Phase-0 owner map: producer `core = mb + 4·out`, consumer
`c`, `in:1` ⇒ no rep-core ambiguity). A2's win ceiling = the hand-off portion of
A0's 19.8 ms; its floor = A0's m×n matmul compute (A1's pure-M time is *not* a
floor — A2 keeps m×n).

This is exactly why steering had to be measured first: it rules out the cheap
fix and scopes A2's target.
