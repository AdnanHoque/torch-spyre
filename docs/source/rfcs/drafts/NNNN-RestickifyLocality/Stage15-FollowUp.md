# Stage 15 Follow-Up: Evidence Before More Profiler Work

## Summary

Before improving profiler UI or adding new trace processing, the next step is to
stabilize the Stage 15 evidence. The marker-separated path is working, but the
current counters are aggregate device-memory counters. They show that the full
fused workload still performs read/write memory traffic after Stage 3B removes
modeled byte-hops for the eligible in-graph edge.

The right immediate question is:

```text
Is the Stage 3B runtime delta stable, and can we amplify the eligible
restickify edge enough for aggregate counters to show a clearer delta?
```

## What Stage 15 Already Proved

For `adds_then_matmul_x` at size `2048`:

- baseline modeled byte-hops: `67,108,864`
- Stage 3B modeled byte-hops: `0`
- restickify count unchanged: `2`
- bytes moved unchanged: `16,777,216`
- median runtime improved directionally: `1.7673 ms -> 1.7284 ms`
- `aiu-smi` read/write traffic remained high in both modes

Interpretation:

Stage 3B is a compiler locality win for the eligible in-graph restickify edge.
It is not proof that the whole fused workload stopped touching device memory.
That is expected because the fused graph still includes a graph-input/weight
restickify, matmul, pointwise work, outputs, and runtime/device-memory traffic.

## Immediate Measurement Plan

### 1. Repeat The Same Stage 15 Run

Run five independent repeats for `2048`, alternating mode order by repeat if
possible.

```sh
for rep in 0 1 2 3 4; do
  python3.12 -u tools/restickify_aiusmi_marker_probe.py \
    --case adds_then_matmul_x \
    --size 2048 \
    --mode baseline \
    --mode stage3b \
    --warmup 10 \
    --iters 5000 \
    --sample-interval 0.1 \
    --output-dir /tmp/restickify-aiusmi-marker-repeat-${rep} \
    --fail-on-error
done
```

Acceptance:

- every repeat has `traffic_samples > 0`;
- baseline byte-hops remain `67,108,864`;
- Stage 3B byte-hops remain `0`;
- restickify count and bytes moved are unchanged;
- runtime speedup is reported as a distribution, not a single number.

### 2. Repeat The Small Case As A Negative Control

Run `512` as a control:

```sh
python3.12 -u tools/restickify_aiusmi_marker_probe.py \
  --case adds_then_matmul_x \
  --size 512 \
  --mode baseline \
  --mode stage3b \
  --warmup 10 \
  --iters 5000 \
  --sample-interval 0.1 \
  --output-dir /tmp/restickify-aiusmi-marker-control-512 \
  --fail-on-error
```

Expected:

- byte-hops reduce partially;
- runtime is roughly unchanged;
- counters are nonzero in both modes.

This helps separate "Stage 3B is enabled" from "Stage 3B creates a measurable
runtime win."

### 3. Amplify The Eligible Edge

The aggregate `aiu-smi` counters are too broad to isolate one eligible edge in a
mixed fused graph. The next useful probe should make the eligible in-graph
restickify dominate more of the runtime, for example:

- repeat the known producer-to-restickify pattern multiple times inside one
  compiled graph;
- run larger sizes such as `3072` if the pod stays within time/memory limits;
- create a synthetic chain where graph-input/weight restickifies are minimized
  and the in-graph producer-restickify edge is repeated.

Acceptance:

- Stage 3B byte-hop reduction remains near `100%` for eligible rows;
- runtime delta grows or becomes more stable;
- aggregate `aiu-smi` read/write averages move directionally, or we conclude
  aggregate counters are too coarse for this question.

## Hardware Information Gaps

The current hardware information is enough for a first-order interpretation:

- 32 cores on a bidirectional RIU data ring;
- RIU data ring bandwidth of roughly `166 GB/s` per direction and `333 GB/s`
  aggregate;
- per-core LX scratchpad;
- off-chip device memory and cross-core LX-LX traffic both use ring-facing data
  paths;
- `aiu-smi` exposes aggregate read/write memory bandwidth and request rates.

The missing granularity is what prevents a stronger physical claim:

1. **Counter semantics.** We need a precise definition of `rdmem`/`wrmem`: does
   it count only off-chip/HBM traffic, all device-memory-controller traffic, or
   any ring-facing memory movement?

2. **Fabric split.** We need counters that separate HBM/off-chip traffic from
   cross-core LX-LX RIU traffic. Aggregate read/write bandwidth is not enough to
   say which path a fused restickify edge used.

3. **Per-op attribution.** We need per-SDSC or per-op counter windows so that
   `ReStickifyOpHBM` traffic can be separated from matmul, pointwise, input,
   output, allocation, and runtime overhead.

4. **Direction and topology.** For ring-aware optimization, per-direction or
   per-hop RIU occupancy would be ideal. Without it, compiler `byte_hops` stays
   a locality model rather than a calibrated hardware counter.

5. **Naming semantics.** We should confirm whether `ReStickifyOpHBM` always
   implies an HBM/off-chip materialization path, or whether the name is a
   historical opfunc label that may still be used for device-memory-backed
   layouts.

## Tooling Gaps

Useful now:

- `aiu-smi` / `aiu-monitor` for aggregate device read/write counters;
- marker-separated counter probe for compile/warmup isolation;
- generated SDSC/opfunc inspection for names such as `ReStickifyOpHBM`;
- torch profiler PrivateUse1 timing for SDSC bundle timing;
- the interactive HTML report for correlating compiler telemetry, counters,
  and source attribution.

Still needed for stronger claims:

1. AIUPTI event/metric records in the Chrome trace, not only aggregate
   `aiu-smi` samples.

2. `aiu-trace-analyzer` compatibility with the current torch profiler trace
   metadata, or a trace normalizer that supplies the required AIU
   `deviceProperties` shape.

3. A fabric-specific counter source for RIU versus HBM/off-chip traffic.

4. A stable mapping from generated SDSC/opfunc names to profiler events and
   counter windows.

5. A repeat-run summarizer. This is not a new profiler feature; it is just a
   measurement hygiene tool so we can report distributions instead of one-off
   medians.

## Recommendation

Do not invest in more profiler UI yet. First:

1. have Claude reproduce the profiler environment in the second pod;
2. run five Stage 15 repeats at `2048`;
3. run the `512` negative control;
4. add or run one amplified eligible-restickify probe;
5. only then decide whether the next blocker is hardware counter granularity or
   compiler-side experiment design.

