# SwiGLU Warp-Specialization Research Note

Date: 2026-06-21

This note records the current reasoning about whether a warp-specialized
SwiGLU/MLP kernel is worth pursuing after the LX coordinate-remap work.

The short version: decode does not look promising for this specific
optimization.  Prefill remains theoretically possible, but the case is not yet
strong enough to justify production work before upstream fused-SiLU lowering
lands and we can remeasure the post-fusion shape.

## Question

Can we overlap PT-heavy matmul work with SFP-heavy SwiGLU pointwise work?

SwiGLU has the structure:

```text
gate, up = x @ W_gate_up
hidden = up * silu(gate)
out = hidden @ W_down
```

Today, the fused FMS SwiGLU lowering still exposes a middle pointwise phase.
Upstream work in torch-spyre issue #2763 is expected to fuse the decomposed
SiLU chain into a single SiLU OpFunc, so the remaining research question is not
"can we fuse neg/exp/add/realdiv?"  It is whether the remaining
`up * silu(gate)` work is large enough and schedulable enough to overlap with
matmul work.

## Relationship To Coordinate Remap

LX coordinate remap and warp specialization are related, but they solve
different problems.

Coordinate remap is an inter-op data movement feature:

```text
producer op writes LX in PerCoreView A
consumer op wants LX in PerCoreView B
move exact cells on chip instead of spilling through HBM
```

Warp specialization is an intra-kernel scheduling/fusion feature:

```text
PT computes matmul tile N+1
SFP computes silu/mul for tile N
optional PT down-projection consumes tile N-1
```

So coordinate remap is not strictly required for a warp-specialized kernel.
A fused MLP kernel could avoid the inter-op boundary entirely by choosing the
right tile layout from the beginning.

Coordinate remap is still useful prior art:

- it established LX-resident producer/consumer reasoning;
- it added artifact and profiler workflows for HBM/LX residency claims;
- it proved explicit on-chip movement can be scheduled and validated;
- it provides a fallback strategy when a fully fused kernel is not available.

The production code for warp specialization would not be a small tweak to the
coordinate-remap planner.  It would most likely be a new fused SwiGLU/MLP kernel
path or a deeper compiler scheduling feature.

## Why There Is Still An Overlap Opportunity Despite Data Dependencies

There is a real dependency inside each tile:

```text
gate/up tile N = matmul(x, W_gate_up) tile N
silu/mul tile N depends on gate/up tile N
```

That dependency means SiLU/mul for tile `N` cannot run before matmul tile `N`.

The only overlap opportunity is pipelining across different tiles:

```text
time 0: PT computes gate/up tile 0
time 1: PT computes gate/up tile 1     SFP computes silu/mul tile 0
time 2: PT computes gate/up tile 2     SFP computes silu/mul tile 1
time 3: PT computes down tile 0        SFP computes silu/mul tile 2
```

That requires:

- tile-level scheduling, not whole-SDSC-row scheduling;
- LX double buffering or careful address reuse;
- paired ownership of matching `gate_j` and `up_j`;
- enough SFP work to hide under PT work;
- enough PT work to keep the matmul side busy while SFP runs;
- no HBM spill between the phases being overlapped.

If the compiler only serializes whole operations:

```text
all matmul -> all SiLU/mul -> all down matmul
```

then there is no meaningful overlap, even if the hardware has idle SFP or PT
resources.

## Implementation Size

### Minimal Research Prototype

A useful prototype could be shape-specific and hardcoded:

- one fused SwiGLU/MLP kernel;
- one or two prefill shapes, such as `B=1, S=512, E=4096, H=12800`;
- fixed tiling;
- fixed gate/up packing assumption;
- no generic planner integration;
- correctness checked against the current FMS SwiGLU baseline;
- profiler proof that PT and SFP scheduling changed.

This is medium-sized research work.  It is not just a pass toggle.

### Production Feature

A production-quality feature is larger:

- generic layout-aware gate/up pairing;
- tile selection across prefill and decode regimes;
- LX capacity and double-buffer planning;
- schedule representation for concurrent PT/SFP phases;
- interaction with existing LX planner and coordinate remap;
- correctness tests for multiple hidden sizes and sequence lengths;
- fallbacks for layouts that cannot colocate gate/up pairs;
- Deeptools/runtime support if current schedule representation cannot express
  the overlap directly;
- profiler acceptance criteria proving resource overlap and kernel-time win.

This should be considered a separate feature from PR-1 coordinate remap.

## Probe Setup

Probe infrastructure was added to:

```text
https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench
```

Relevant commit:

```text
e9d25fd Add SwiGLU warp-specialization phase probes
```

Important files:

```text
tools/perf_suite_warpspec_probe_op.py
scripts/run_warpspec_phase_probe.sh
tools/summarize_warpspec_phase_probe.py
docs/warp_specialization_probes.md
docs/warp_specialization_probe_results_2026_06_21.md
```

The probes isolate:

- full SwiGLU;
- SiLU only;
- SiLU plus multiply;
- reduced pointwise variants that consume the full intermediate but return a
  small tensor, avoiding standalone large-output artifacts.

Primary metric is Kineto trace-derived `kernel_ms_per_iter`.

## Probe Results

| Case | Shape | kernel_ms/iter | Spyre ms | memory_ms/iter | PT util % |
|---|---|---:|---:|---:|---:|
| full SwiGLU | `B1 S512 E4096 H12800` | `5.318985` | `11.566` | `6.246770` | `52.505` |
| full SwiGLU | `B1 S512 E128 H512` | `0.044651` | `0.098` | `0.052988` | `6.255` |
| SiLU reduce | `B1 S512 E128 H512` | `0.027203` | `0.071` | `0.043433` | `0.000` |
| SiLU+mul reduce | `B1 S512 E128 H512` | `0.030057` | `0.095` | `0.065305` | `0.000` |
| full SwiGLU decode | `B1 S1 E4096 H12800` | `3.786675` | `9.396` | `5.609741` | `0.173` |
| SiLU+mul reduce decode | `B1 S1 E4096 H12800` | `0.004011` | `0.029` | `0.025267` | `0.000` |
| SiLU reduce decode | `B1 S1 E4096 H12800` | `0.003618` | `0.024` | `0.020113` | `0.000` |

Artifact roots on the pod:

```text
/tmp/warpspec_phase_probe_repo_20260621_040743/prefill_ws_full_swiglu
/tmp/warpspec_phase_small_repo_20260621_042216/prefill_ws_full_swiglu
/tmp/warpspec_single_small_builtin_reduce_20260621_043022
/tmp/warpspec_single_small_silu_mul_reduce_20260621_043220
/tmp/warpspec_single_decode_ws_full_swiglu_20260621_043414
/tmp/warpspec_single_decode_ws_silu_mul_reduce_20260621_043527
/tmp/warpspec_single_decode_ws_silu_builtin_reduce_20260621_043639
```

## Theoretical Speedup Model

If matmul and pointwise are serial:

```text
serial_time = PT + SFP
```

If they overlap perfectly:

```text
overlapped_time = max(PT, SFP)
speedup = (PT + SFP) / max(PT, SFP)
```

Equivalently, if pointwise is fraction `f` of the current full kernel and all
of it can be hidden:

```text
max_speedup = 1 / (1 - f)
```

This is optimistic.  It ignores double-buffer overhead, scheduling overhead,
LX pressure, and any pointwise work that cannot be fully hidden.

### Decode

Real decode probe:

```text
full SwiGLU:        3.786675 ms
SiLU+mul reduce:    0.004011 ms
fraction:           0.00106
```

The optimistic speedup is:

```text
1 / (1 - 0.00106) = 1.001x
```

That is too small to justify warp specialization for decode SwiGLU pointwise
overlap.

### Prefill

The real prefill full-SwiGLU probe completed:

```text
full SwiGLU: 5.318985 ms
PT util:     52.505%
```

The real-shape standalone pointwise probes were not reliable.  Raw standalone
pointwise and reduced real-shape `silu+mul` hit runtime completion behavior
before producing usable traces.  The small controlled prefill shape did
complete:

```text
small full SwiGLU:           0.044651 ms
small SiLU+mul reduce:       0.030057 ms
small pointwise/full ratio:  0.673
```

That small-shape ratio is not representative of the FMS prefill target because
the small case is not matmul-dominated.  It only proves that the pointwise path
is SFP-heavy and overlappable in principle.

For real FMS prefill, the useful sensitivity table is:

| Pointwise fraction of full prefill | Optimistic max speedup |
|---:|---:|
| `5%` | `1.053x` |
| `10%` | `1.111x` |
| `20%` | `1.250x` |
| `33%` | `1.493x` |

We do not yet have evidence that post-fused-SiLU real prefill pointwise remains
large enough to land in the attractive part of this table.

## Current Assessment

Decode:

- not promising for this specific warp-specialization target;
- pointwise work is approximately `0.1%` of full decode SwiGLU kernel time;
- even perfect overlap would be around `1.001x`.

Prefill:

- possible in principle;
- pointwise probes show pure SFP work (`PT util = 0`);
- full real prefill SwiGLU has material PT work (`PT util = 52.5%`);
- direct real-shape standalone pointwise measurement is currently unreliable;
- payoff should be reassessed after fused-SiLU lowering lands upstream.

Implementation priority:

1. Keep coordinate remap as the production-track PR-1 feature.
2. Wait for upstream fused-SiLU lowering.
3. Rerun this probe suite against the post-fused-SiLU lowering.
4. Only start a warp-specialized fused MLP prototype if real prefill pointwise
   remains at least a visible fraction of full kernel time, ideally above
   `10-20%`.

## Open Questions

- Can the compiler or Deeptools expose finer-grained PT/SFP scheduling within a
  fused op, or would this require a new hand-authored kernel pattern?
- Can the first projection produce gate/up pairs in a layout that avoids a
  pairing remap before SiLU/mul?
- Does post-issue-2763 fused-SiLU materially reduce pointwise runtime?
- Is a fused streaming MLP kernel more valuable than a general warp-specialized
  scheduling feature?
- Are real-shape standalone pointwise runtime completion issues a profiler
  artifact, a runtime issue, or a property of these artificial reduced probes?
