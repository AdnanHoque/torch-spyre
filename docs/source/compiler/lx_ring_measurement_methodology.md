# LX ring measurement methodology and current answers

This document records what was measured, how it was measured from first
principles, and which conclusions are justified without hardware performance
counters. The companion [cost-model draft](communication_cost_model.md) turns
the measurements into a predictor. The [joint-placement design](joint_ring_placement.md)
describes the first compiler optimization suggested by the results.

:::{admonition} Revision boundary
:class: important

The direct-LX and original HBM/LX/oracle measurements were made with these
immutable revisions:

- Torch base: `2a20cf3b7ac8aadf629314e40e5059ad82471911`
- Torch measured tree: `24adc85c04da91d61b13b295d6092438cf2029b4`
- Deeptools: `19280fd7c6bbd91000c63c2a6719a0253e513f4a`
- LLVM: `22.1.3`

This documentation branch starts from the later rewritten Torch feature head
`8e8324febe7bb6b266652b9aeda3c778e3b22935`. The measurements characterize the
pinned measured tree above; they are not a claim that the later rewrite has
already been remeasured.

The corrected placement-by-handoff factorial used a clean test-only combined
Torch head `9b3449e732717c834f78ef0f8897a729c8da8f65` (tree
`2fb5afbbe59bfa4fc95ed029710945cae3c8208f`), the same exact Deeptools head,
LLVM 22.1.3, and perf-suite head
`7ec6df0825e3a07614b82ddae5efae45eac43463`. Its instrumentation is evidence,
not production placement code.
:::

## Answers at a glance

| Question | Current answer | Confidence boundary |
|---|---|---|
| LX-to-LX bandwidth | **131.351 GB/s** for a finite 512 KiB one-way CCW handoff; **136.457 GB/s** fitted one-way service slope; **255.439 GB/s** aggregate for balanced simultaneous CW and CCW handoffs | Whole-precompiled-bundle service, not ring-active counter cycles; standalone CW curve remains unmeasured |
| Utilization at 1.1 GHz | **93.289%** at 512 KiB, **96.916%** for the fitted slope, and **90.710%** balanced-duplex aggregate | Conditional on `128 B/cycle/direction` using a historical same-PF 1.1 GHz RNG readback |
| Balanced-duplex concurrency | Duplex retains **97.235%** of twice the one-way rate | Confirmed for one balanced 512 KiB-per-direction case |
| Large-transfer distance sensitivity | One hop to eight hops adds **0.0320 us**, or about **0.80%**, at 512 KiB | Same-pod CLC comparison; only two distances, so not yet a calibrated hop curve |
| Flash LX versus HBM | **1.210x** by ratio of condition means; LX removes **44.702 us** | Exact tested Flash shape only |
| Baseline Flash LX versus oracle | LX is at **85.915%** of oracle inverse-time performance; **29.961 us** remains; maximum further speedup is **1.164x** | Original HBM/LX/oracle campaign; graph-level no-SHUFFLE oracle, not device peak |
| Coherent Flash placement | **11.697 us** less LX time, **5.550%** lower latency, and **1.059x** rate versus default placement | Corrected value-safe five-block factorial at the exact tested shape |
| Coherent Flash versus paired oracle | **17.168 us** remains; **91.375%** of oracle inverse-time performance; at most **1.094x** further speedup | The paired graph oracle removes SHUFFLE and dependent overhead; it is not ring peak |
| Shuffle-residual fraction closed | **42.618%**, t95 **[41.117%, 44.118%]** | Mean of five within-block residual fractions; the ratio-of-means sensitivity is 42.637% |
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
| One-way CCW, 1 hop | 128 KiB | 1.1100 us | 118.083 GB/s |
| One-way CCW, 1 hop | 256 KiB | 2.0700 us | 126.640 GB/s |
| One-way CCW, 1 hop | 512 KiB | 3.9915 us | 131.351 GB/s |
| Balanced duplex, 1 hop each way | 1 MiB aggregate | 4.1050 us | 255.439 GB/s aggregate |
| One-way CCW, 8 hops | 512 KiB | 4.0230 us | 130.323 GB/s |

The distance delta uses the matched same-CLC-pod one-hop replication
(`3.9910 us`) against the CLC eight-hop result (`4.0230 us`), hence
`0.0320 us`. The table retains the primary CDX one-hop value (`3.9915 us`) as
the headline finite-size measurement.

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

This graph-level oracle is the current measurable Flash graph ceiling, not a
device or ring peak. A physical ring utilization number cannot be extracted
from the fused Flash event because it also contains endpoint, synchronization,
layout, and compute effects.

## Coherent-placement methodology

The emitted SHUFFLE allocation maps were converted into source-to-destination
relations on one global 32-core bidirectional ring. Each relation was routed by
shortest path, with the tie direction fixed in advance. The analyzer counted:

- total hop units;
- mean remote distance;
- load on every directed link; and
- load on each combined physical segment.

The first `joint_all` prototype passed weak low-amplitude checks but was later
invalidated by a high-contrast device-versus-CPU test: 241,384 of 262,144
elements mismatched at `atol=rtol=1e-2`, with maximum absolute error `0.943985`.
No timing from that prototype is valid performance evidence. Its failure was
specific and reproducible: the actual scaled-K producer retained the default
interleaved mapping while a synthetic local-LX source view was labeled
head-contiguous. No transfer separated them, so the same bytes were interpreted
under two mappings before SHUFFLE.

The resulting legality invariant is:

```text
Every unshuffled local-LX producer-to-consumer edge must use the same realized
physical mapping. A mapping change requires an explicit materialized bridge.
```

The corrected experiment first isolated the boundary:

- changing only the synthetic source reproduced the device error;
- changing the actual scaled-K producer and its source view together passed;
- closing the query/score/consumer side independently passed; and
- the fully coherent six-point placement passed high-contrast checks in both
  compile orders and a second adversarial test whose V values encode head,
  token, and channel.

Only after those gates passed was timing allowed. The emitted allocation maps
produce the following **software shortest-path route proxies**:

| Proxy metric | Default | Coherent placement | Change |
|---|---:|---:|---:|
| Remote relations | 224 | 224 | 0 |
| Total hop units | 2,048 | 672 | -67.2% |
| Maximum directed-link units | 40 | 16 | -60.0% |

These proxies assume equal-payload relations, shortest paths, and a fixed tie
direction. They are not measured link traffic, realized multicast behavior, or
device-peak utilization.

## Placement-by-handoff factorial result

The corrected paired `2 x 2` factorial used:

```text
placement = {default, coherent}
handoff   = {LX SHUFFLE, preseeded no-SHUFFLE oracle}
```

Five counterbalanced fresh-process blocks used seeds 31--35 and 30 accepted
device events per cell. Oracle cells generated 30 additional prefix events,
which strict trace-role classification excluded. Each block used its process
median as the analysis unit. All 20 cells passed correctness, materialization,
event-inventory, graph-identity, deterministic-root, and trace gates; the frozen
LLVM/DXP closure matched before and after.

All t95 intervals below are descriptive single-device process-block intervals.
The five serial blocks are not five independent hardware samples.

| Condition | Mean of five process medians |
|---|---:|
| LX default | 210.7515 us |
| Oracle default | 180.8223 us |
| LX coherent | 199.0549 us |
| Oracle coherent | 181.8865 us |

The user-visible coherent LX gain is `11.6966 us`, t95
`[10.7843, 12.6089]`: **5.550%** lower latency and **1.05877x** the default
rate. This remains a whole-fused-kernel placement effect, not an isolated ring
root time.

The preregistered decomposition was:

```text
default_residual     = LX_default  - oracle_default
coherent_residual    = LX_coherent - oracle_coherent
residual_interaction = default_residual - coherent_residual
```

The default residual was `29.9292 us`; coherent placement reduced it to
`17.1684 us`, t95 `[16.6635, 17.6733]`. The interaction was `12.7608 us`, t95
`[11.9086, 13.6130]`, and every block was positive. The mean of the five
within-block fractions closed is **42.6177%**, t95
`[41.1175%, 44.1178%]`; dividing the two aggregate means instead gives 42.6366%.

The coherent graph reaches **91.3749%** of its paired oracle's inverse-time
performance and leaves at most **1.09440x** further speedup for this graph and
shape. The oracle-only placement contrast is `-1.0642 us`, t95
`[-2.2580, 0.1296]`, so a compute-only placement effect is unresolved. This
supports—but, without root timers or counters, does not physically prove—a
handoff-specific explanation for the positive interaction.

## Next Flash steps

The next work is no longer another whole-kernel rerun of the same point:

1. turn the coherent six-point test adapter into a typed, closed-region
   placement contract that includes the actual producer, aliases/views, all
   local allocation users, and explicit bridges;
2. add root-scoped compiler timestamps or markers around SHUFFLE,
   synchronization, and the first consumer to decompose the remaining
   `17.1684 us`;
3. obtain realized direction/path/multicast feedback from the backend so the
   static `672/16` proxy can be compared with actual transport;
4. repeat the corrected factorial across key sequence lengths, head counts,
   payloads, and more than one device before enabling automatic policy; and
5. retain the graph oracle as a regression ceiling while adding microbenchmarks
   for multicast, contention, banks/alignment, and overlap.

## What is still missing

The initial cost model is actionable, but a complete hardware/software model
still needs:

1. a matched same-stream memory write/read control on the measured toolchain;
2. an independent clockwise one-way size curve, not only the balanced-duplex
   clockwise stream;
3. multicast/fanout and expanded-unicast A/Bs with realized-route evidence;
4. a 1/2/4/8/16-segment contention and concurrency curve;
5. payload size and alignment/bank sweeps;
6. transfer/compute overlap measurements;
7. root-scoped Flash timing; and
8. a timing-bound clock readback, followed later by hardware counters.

These measurements should be added only when they identify a missing model
coefficient. Repeating the already stable 512 KiB point is no longer useful.
