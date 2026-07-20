# LX ring measurement methodology and current answers

This document records what was measured, how it was measured from first
principles, and which conclusions are justified without hardware performance
counters. The companion [cost-model draft](communication_cost_model.md) turns
the measurements into a predictor. The [joint-placement design](joint_ring_placement.md)
describes the first compiler optimization suggested by the results.

:::{admonition} Revision boundary
:class: important

The promoted measurements were made with these immutable revisions:

- Torch base: `2a20cf3b7ac8aadf629314e40e5059ad82471911`
- Torch measured tree: `24adc85c04da91d61b13b295d6092438cf2029b4`
- Deeptools: `19280fd7c6bbd91000c63c2a6719a0253e513f4a`
- LLVM: `22.1.3`

This documentation branch starts from the later rewritten Torch feature head
`8e8324febe7bb6b266652b9aeda3c778e3b22935`. The measurements characterize the
pinned measured tree above; they are not a claim that the later rewrite has
already been remeasured.
:::

## Answers at a glance

| Question | Current answer | Confidence boundary |
|---|---|---|
| LX-to-LX bandwidth | **131.351 GB/s** for a finite 512 KiB one-way handoff; **136.457 GB/s** fitted sustained one-way service; **255.439 GB/s** aggregate for balanced simultaneous CW and CCW handoffs | Whole-precompiled-bundle service, not ring-active counter cycles |
| Utilization at 1.1 GHz | **93.289%** at 512 KiB, **96.916%** for the fitted slope, and **90.710%** balanced-duplex aggregate | Conditional on `128 B/cycle/direction` using a historical same-PF 1.1 GHz RNG readback |
| Balanced-duplex concurrency | Duplex retains **97.235%** of twice the one-way rate | Confirmed for one balanced 512 KiB-per-direction case |
| Large-transfer distance sensitivity | One hop to eight hops adds **0.0320 us**, or about **0.80%**, at 512 KiB | Same-pod CLC comparison; only two distances, so not yet a calibrated hop curve |
| Flash LX versus HBM | **1.210x** by ratio of condition means; LX removes **44.702 us** | Exact tested Flash shape only |
| Flash LX versus oracle | LX is at **85.915%** of oracle inverse-time performance; **29.961 us** remains; maximum further speedup is **1.164x** | Graph-level no-SHUFFLE oracle, not a raw link-bandwidth oracle |
| Cost-model completeness | Enough for shadow ranking and the allowlisted measured shape, not a generic enabled policy | Matched HBM, multicast, contention, bank/alignment, overlap, and root timing remain uncalibrated |

Raw GB/s is the canonical hardware result. The utilization percentages add a
clock-domain assumption and are therefore reported as conditional engineering
estimates.

## What “bandwidth” means

A bandwidth number is meaningful only when its byte numerator and time
denominator have the same scope. The direct probes use:

```text
effective service bandwidth = injected payload bytes / whole bundle time
```

For one transfer, injected bytes and delivered bytes are equal. For balanced
duplex, the numerator is the sum of the independently injected CW and CCW
payloads. Hop-carried bytes are deliberately not used as this numerator: a
512 KiB packet traversing eight pipelined links is still a 512 KiB injection,
not an 8x-faster 4 MiB transfer.

The time denominator is the launch-correlated AIU device-kernel complete-event
duration. Host wall time is excluded because host/runtime overhead dominates it
and does not describe ring service.

## Direct LX procedure

The procedure was designed to rule out the most common false bandwidth claims.

1. **Freeze the compiler and runtime closure.** Record the Torch, Deeptools,
   LLVM, device extension, and precompiled bundle identities. Reuse the frozen
   bundles during timing so compilation is outside the measured region.
2. **Prove the requested transport.** Inspect the generated bundle and
   descriptors to confirm an LX-only `STCDPOpLx` transfer with explicit source,
   destination, direction, and payload. Reject materialization to memory or a
   different route.
3. **Define physical byte scopes before timing.** Record injected bytes,
   delivered bytes, hottest-directed-link bytes, and total hop-carried bytes as
   different quantities.
4. **Gate correctness over the entire transfer window.** Check all destination
   bytes, require the exact destination-change mask, and prove the source is
   unchanged. The headline 512 KiB and distance probes run full gates around
   timing. The 128/256 KiB fit extensions use seed correctness before timing and
   a full value/change-mask/source check afterward; this distinction is retained
   in their result JSONs.
5. **Warm the exact frozen operation.** Execute 10 untimed warmups.
6. **Time device work only.** Collect exactly 100 launch-correlated device
   kernel events and use the median duration. Require one expected device event
   per operation and reject ambiguous ancillary events.
7. **Calculate bandwidth directly.** Divide the preregistered byte numerator by
   the median device duration. Do not infer bandwidth from the tensor shape
   after the fact.
8. **Replicate before expanding the matrix.** Repeat the 512 KiB one-way and
   duplex points on other available pods/nodes. The replicated values differed
   by less than 0.03%, so more repetitions of the same point were deprioritized.
9. **Separate startup from slope.** Measure 128, 256, and 512 KiB with the same
   one-way route and fit `T = F + S/B`.
10. **Probe topology independently.** Keep payload and direction fixed while
    changing one hop to eight hops.

The local evidence package contains the result JSONs and their integrity hashes.
It intentionally excludes traces, binaries, caches, and raw device logs.

## Direct results

| Case | Payload numerator | Median device time | Service bandwidth |
|---|---:|---:|---:|
| One-way, 1 hop | 128 KiB | 1.1100 us | 118.083 GB/s |
| One-way, 1 hop | 256 KiB | 2.0700 us | 126.640 GB/s |
| One-way, 1 hop | 512 KiB | 3.9915 us | 131.351 GB/s |
| Balanced duplex, 1 hop each way | 1 MiB aggregate | 4.1050 us | 255.439 GB/s aggregate |
| One-way, 8 hops | 512 KiB | 4.0230 us | 130.323 GB/s |

The three one-way sizes fit:

```text
T_one_way_us(S) = 0.14925 + S_bytes / 136,457.185
```

The maximum absolute residual of the three-point fit is below `0.00033 us`.
Adjacent-pair slopes independently give 136.53 and 136.43 GB/s. This makes
`F = 0.14925 us` and `B = 136.457 GB/s` useful first-order coefficients, while
still leaving a broader size campaign as a future robustness check.

Balanced duplex takes only 2.84% longer than one direction while moving twice
the aggregate payload:

```text
duplex retention
  = 255.439 / (2 * 131.351)
  = 97.235%
```

That is strong counter-free evidence that CW and CCW do not suffer a large
shared serialization bottleneck in this case.

## The 1.1 GHz efficiency calculation

A sanitized historical readback from the same CDX PF reported:

```text
RPD Clock = 1100 MHz
RNG Clock = 1100 MHz
SOC Clock = 1000 MHz
DDR Clock = 6400 MHz
```

The raw log is not included because it contains device identifiers. Its SHA-256
is preserved in the artifact provenance file. The readback was not taken inside
the promoted timing process, and the available hardware description does not
formally prove that an SPad-ring “cycle” is the RNG clock. We therefore use
1.1 GHz as the primary engineering frequency but keep the clock-domain binding
explicit.

If the ring moves 128 B/cycle in each direction at 1.1 GHz:

```text
C_direction = 128 B/cycle * 1.1e9 cycle/s = 140.8 GB/s
C_pair      = 2 * C_direction              = 281.6 GB/s
```

| Scope | Peak at 1.1 GHz | Measured | Conditional efficiency |
|---|---:|---:|---:|
| 512 KiB one-way | 140.8 GB/s | 131.351 GB/s | **93.289%** |
| Fitted one-way slope | 140.8 GB/s | 136.457 GB/s | **96.916%** |
| Balanced duplex aggregate | 281.6 GB/s | 255.439 GB/s | **90.710%** |
| Eight-hop 512 KiB | 140.8 GB/s | 130.323 GB/s | **92.559%** |

For the 512 KiB one-way case, peak payload time is `3.7236 us`; observed time is
`3.9915 us`, leaving `0.2679 us`. The size fit assigns about `0.1493 us` to
fixed bundle/carrier cost and about `0.1186 us` to the measured service slope
being below 140.8 GB/s. Without counters, the latter cannot be causally divided
among endpoint backpressure, bank conflicts, bubbles, or ring stalls.

## Quick mental model for avoiding a memory roundtrip

For `S` bytes that would otherwise be written to device memory and later read
back, use achieved bandwidths for the exact layouts:

```text
T_memory(S) = F_memory + S/B_write + S/B_read
T_LX(S)     = F_LX     + S/B_LX

movement-only speedup
  = T_memory / T_LX
  ~= B_LX * (1/B_write + 1/B_read)       # for large S
```

If achieved memory read and write rates are both approximately `B_memory`, the
mental shortcut is:

```text
movement-only speedup ~= 2 * B_LX / B_memory
```

Use `B_LX = 136.457 GB/s` for large one-way handoffs, or the measured finite-size
point when the payload is close to 512 KiB. Use duplex aggregate bandwidth only
when the proposed algorithm truly creates two independent, balanced directional
streams and both alternatives use the same aggregate byte accounting.

Convert the movement-only result to whole-kernel speedup with Amdahl's law. If
`p` is the fraction of baseline time that the handoff can replace and `r` is its
movement speedup:

```text
whole-kernel speedup ~= 1 / ((1 - p) + p/r)
```

We do not yet have a matched, same-stream, LLVM-22.1.3 memory write/read
denominator. The included all-32-core memory roundtrip is provisional because it
uses a different helper scope and LLVM 20; it must not be combined with the
direct LX number as if it were a controlled ratio.

## Flash oracle methodology and answer

The tested full-attention shape was `B=1, H=4, Lq=512, Lk=4096, D=128`, group
size 8, with 128 KiB of source data per core. Five fresh-process,
counterbalanced HBM/LX/oracle triples were run with 30 accepted device events
per condition. The analysis used paired deltas across the triples and preserved
correctness, materialization, route, event-inventory, and wrong-preseed gates.

The oracle is a test-only graph counterfactual: a prefix prepares the correct
final consumer-visible S2 allocation outside the timed region, then the timed
graph runs without SHUFFLE. It is not infinite hardware bandwidth and it is not
a directly deployable algorithm. It answers “what would this graph cost if the
handoff and its dependent overhead disappeared?”

| Condition | Mean of process medians |
|---|---:|
| HBM | 257.422 us |
| Current LX | 212.720 us |
| Preseeded no-SHUFFLE oracle | 182.758 us |

Therefore:

- current LX is `257.422 / 212.720 = 1.210x` faster than HBM;
- LX recovers **59.872%** of the HBM-to-oracle opportunity;
- **29.961 us**, or **40.128%** of that opportunity, remains;
- current LX delivers **85.915%** of oracle inverse-time performance; and
- closing the full residual would be at most `212.720 / 182.758 = 1.164x`
  faster than current LX for this exact graph.

A separate three-triple replay on another node left a similar `30.307 us`
absolute residual. It is descriptive replication and is not pooled with the
primary campaign.

This graph-level oracle is the defensible Flash “peak” today. A physical ring
utilization number cannot be extracted from the fused Flash event because it
also contains endpoint, synchronization, layout, and compute effects.

## Joint-placement methodology

The emitted SHUFFLE allocation maps were converted into source-to-destination
relations on one global 32-core bidirectional ring. Each relation was routed by
shortest path, with the tie direction fixed in advance. The analyzer counted:

- total hop units;
- mean remote distance;
- load on every directed link; and
- load on each combined physical segment.

Structural acceptance also required exact shape and source size, the expected
root set, exactly one materialized SHUFFLE, semantic equality to the default,
preserved allocation components, trace quality, and a wrong-route negative.

A five-block fresh-process A/B then compared default placement with the minimal
closed-region `joint_all` candidate:

| Metric | Default | Joint placement |
|---|---:|---:|
| Total hop units | 2,048 | 672 |
| Mean remote distance | 9.143 | 3.000 |
| Maximum directed-link units | 40 | 16 |
| Maximum combined-segment units | 64 | 32 |
| Whole fused LX time | 213.527 us | 203.140 us |

The paired improvement was `10.3867 us` with a descriptive t95 interval of
`[10.0617, 10.7117]`, or **4.864% / 1.051x**. All five blocks favored joint
placement. This is a whole-region effect, not an isolated ring-timing result;
placement can also change compute scheduling and locality.

## The next Flash experiment

The clean next experiment is a paired `2 x 2` factorial:

```text
placement = {default, joint_all}
handoff   = {LX SHUFFLE, preseeded no-SHUFFLE oracle}
```

Run all four arms in the same five counterbalanced fresh-process blocks, with
the existing exact shape, allocations, 30-event rule, and correctness gates.
Keep default HBM as context, not as an additive placement control. The earlier
joint HBM arm changed unrelated behavior and is not an identifiable correction.

Report:

```text
placement_gain       = LX_default - LX_joint
default_residual     = LX_default - oracle_default
joint_residual       = LX_joint   - oracle_joint
residual_interaction = default_residual - joint_residual
fraction_closed      = residual_interaction / default_residual
```

Add root-scoped timestamps or compiler markers around SHUFFLE, synchronization,
and consumer work in the same campaign if available. This experiment tells us
whether joint placement actually closes the oracle residual rather than merely
speeding another part of the fused graph.

As a non-causal projection only, transplanting the measured `10.3867 us` gain
onto the primary campaign predicts about `202.333 us`, **90.33%** of oracle
inverse-time performance, and roughly `19.575 us` still remaining.

## What is still missing

The initial cost model is actionable, but a complete hardware/software model
still needs:

1. a matched same-stream memory write/read control on the measured toolchain;
2. multicast/fanout and expanded-unicast A/Bs with realized-route evidence;
3. a 1/2/4/8/16-segment contention and concurrency curve;
4. payload size and alignment/bank sweeps;
5. transfer/compute overlap measurements;
6. root-scoped Flash timing; and
7. a timing-bound clock readback, followed later by hardware counters.

These measurements should be added only when they identify a missing model
coefficient. Repeating the already stable 512 KiB point is no longer useful.
