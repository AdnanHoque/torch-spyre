# Granite 40-layer SenDNN vs Torch-Spyre gap analysis

## Bottom line

SenDNN wins for two different reasons:

1. **Prefill device time:** both compilers report essentially the same PT
   compute floor, but SenDNN realizes it with a whole-graph schedule and much
   greater LX/LDS retention. Torch-Spyre realizes only 39.49% of its block PT
   cycle proxy, versus 60.71% for the complete SenDNN phase.
2. **Decode device time:** Torch-Spyre's SwiGLU is already close to the assumed
   140 GB/s effective memory ceiling. Almost all of the remaining opportunity
   is attention/projection/KV service: 60.84 useful GB/s today, with roughly
   116-126 GB/s required to match SenDNN under conservative assumptions.
3. **End-to-end wall time:** this is a separate, much larger gap. Torch-Spyre
   falls back to CPU for embedding and launches one compiled region per layer;
   SenDNN includes the whole phase in one device program.

The measured contract is Granite 3.3 8B Instruct, 40 decoder layers, B1/S512,
FP16, unfused weights, SDPA, four generated token phases, five profiled runs.
The repository branch is `adnan/sendnn-granite-antoni-repro-20260725` at
`bec053fa59d4ca81dc0f081acbe074939a9d2abd`.

## Definition of parity

Parity should be an automated two-part gate on this exact contract:

| Gate | SenDNN reference | Torch-Spyre pass condition |
|---|---:|---:|
| Prefill device program | 190.406 ms | <= 192.310 ms (+1%) |
| Decode device program | 123.961 ms | <= 125.200 ms (+1%) |
| Prefill profiled wall | 195.535 ms | <= 197.490 ms (+1%) |
| Decode profiled wall | 132.884 ms | <= 134.213 ms (+1%) |

The device gate prevents a faster host path from hiding a compiler regression;
the wall gate prevents a fast kernel sum from hiding CPU fallbacks and launch
overhead. Both must pass with identical model configuration and output tokens.
For compiler changes, retain a CPU-reference numerical check for isolated
regions before accepting the full-model token check.

## The architectural prerequisite

The central path to parity is **an optimizer-visible phase plan with multiple
internal bundles**, not merely a larger individual Torch-Spyre bundle.
SenDNN exposes one prepared phase to the runtime while letting its compiler
choose 169 internal bundles for prefill and 180 for decode. That combination
provides whole-phase lifetimes, prefetch, relayout, and LX decisions without
forcing every operation into one monolithic placement.

Torch-Spyre currently compiles and launches each fused scheduler node
independently:

- `_inductor/fusion.py` caps an individual bundle at six tensors.
- `_inductor/scheduler.py` says intermediate buffers cannot be removed through
  fusion, disables vertical/horizontal fusion, and emits `call_kernel` once per
  fused scheduler node.
- `_inductor/choices.py` independently returns false for all fusion choices.
- `_inductor/codegen/bundle.py` writes and compiles one `bundle.mlir` at a time.
- `execution/async_compile.py` invokes `dxp_standalone` once per bundle.
- `execution/kernel_runner.py` launches one compiled graph per runner call.

This boundary explains both symptoms: Torch has 200 block launches in prefill
and 240 in steady decode, and the compiler cannot carry a value or schedule a
weight prefetch across those boundaries. Runtime batching of the existing
independently compiled graphs would reduce host gaps but would **not** by
itself close the 111/29 ms device gaps. The phase DAG must be visible to the
compiler's placement/scheduling stage.

The implementation target is therefore:

1. capture one prefill FX graph and one dynamic decode FX graph;
2. lower each into an ordered bundle DAG with explicit dependencies and tensor
   lifetimes;
3. submit that DAG once so the compiler can retain internal bundle boundaries
   while planning LX, relayout, HBM prefetch, and overlap globally;
4. prepare and launch the resulting phase plan once per token phase.

Do not use “one giant SuperDSC bundle” as the acceptance criterion. A previous
production-shaped giant decoder-block bundle was slower because its placement
was worse. The criterion is one optimizer-visible phase plan, with internal
segmentation chosen on cost.

## Latency views

The kernel-sum comparison is the cleanest device-program comparison. The wall
numbers include host/framework behavior and expose a second optimization
problem.

| Phase | SenDNN kernel sum | Torch kernel sum | Torch first-to-last kernel span | SenDNN Python wall | Torch Python wall |
|---|---:|---:|---:|---:|---:|
| Prefill | 190.406 ms | 301.442 ms | 373.028 ms | 195.535 ms | 1293.896 ms |
| Decode average | 123.961 ms | 153.592 ms | 254.860 ms | 132.884 ms | 1105.272 ms |

Device-program gaps are 111.036 ms for prefill and 29.631 ms for decode.
Wall-time gaps are 1098.361 ms and 972.388 ms, respectively.

### Wall-gap budget

| Contribution | Prefill | Decode average |
|---|---:|---:|
| Device-program difference | 111.036 ms | 29.631 ms |
| Torch CPU embedding fallback | 846.040 ms | 836.520 ms |
| Torch per-layer compiled-region time above block kernels | 63.799 ms | 92.249 ms |
| Remaining net host/framework difference | 77.486 ms | 13.989 ms |
| **Total wall gap** | **1098.361 ms** | **972.388 ms** |

Thus only 10.1% of the prefill wall gap and 3.0% of the decode wall gap is the
device-program difference. The run log explicitly reports
`aten.embedding.default is falling back to cpu`. The SenDNN trace has one
externally visible kernel named `embedding`, but that name labels the complete
compiled graph: compiler markers prove that it contains all 40 attention and
SwiGLU layers.

Torch's accelerator timeline has additional gaps that kernel sums omit:

| Phase | Kernel sum | First-to-last span | Recorded memory work outside kernels | Uncovered/launch-runtime gap |
|---|---:|---:|---:|---:|
| Prefill | 301.442 ms | 373.028 ms | 5.032 ms | 66.553 ms |
| Decode average | 153.592 ms | 254.860 ms | 6.770 ms | 94.498 ms |

SenDNN has one kernel per phase, so it has no intra-phase external-kernel gap.
Its DtoH event covers 99.998% of the fused kernel interval and must not be
added to kernel duration. Across all 20 calls, HtoD events total only 0.289 ms
and memsets total 5.135 ms.

## Prefill: compute realization and locality

All 32 fresh full-model Torch compiler bundles match the prior one-layer
bundles byte-for-byte when hashed over `bundle.mlir` and every `sdsc_*.json`.
There are 27 unique content signatures because a few shapes are duplicated.
This validates transferring the prior compiler ideal-cycle reports to this
fresh full-model trace.

| Torch block family | Actual/layer | PT ideal/layer | PT proxy | 40-layer gap to proxy | SDSCs | HBM alloc nodes | LX alloc nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q projection + norm | 0.532906 ms | 0.238313 ms | 44.72% | 11.784 ms | 9 | 11 | 13 |
| K projection + QK/softmax | 1.434337 ms | 0.089367 ms | 6.23% | 53.799 ms | 14 | 20 | 16 |
| V/AV/output projection + norm | 1.056483 ms | 0.327680 ms | 31.02% | 29.152 ms | 9 | 13 | 11 |
| SwiGLU | 4.224541 ms | 2.234182 ms | 52.89% | 79.614 ms | 9 | 15 | 10 |
| Residual | 0.069006 ms | 0 | N/A | 2.760 ms | 2 | 3 | 3 |
| **Block** | **7.317274 ms** | **2.889542 ms** | **39.49%** | **177.109 ms** | **43** | **62** | **53** |

The low percentages are compiler-cycle proxies, not physical PT occupancy
counters. In particular, the K/QK bundle's denominator contains softmax,
reductions, communication, and HBM work whose ideal PT cycle report is zero.
Its 6.23% still identifies the most severe schedule/locality problem, but it
does not mean the PT is literally idle for 93.77% of the interval.

The important cross-compiler fact is the floor:

- Torch 40-layer block PT ideal: 115.582 ms.
- SenDNN complete-prefill PT ideal: 115.587 ms.
- SenDNN actual prefill: 190.406 ms, or 60.71% of that proxy.
- Torch block actual: 292.691 ms, or 39.49% of its proxy.

The compilers therefore agree on required PT work. SenDNN is not winning by
doing materially fewer matrix-multiply cycles; it is winning in realization.
Keeping Torch's 8.752 ms one-off work unchanged, matching 190.406 ms requires
the Torch block to fall to 181.655 ms, or 63.63% aggregate block PT proxy. The
97.6% value in the earlier screenshot applied to a 130 ms target, not the
SenDNN-matching target.

### Structural explanation

SenDNN compiles each phase as one full graph (`numSubGraphs=0`):

| SenDNN phase | Internal bundles | LX-optimized bundles | LDS optimized | Relayouts | External kernels/call |
|---|---:|---:|---:|---:|---:|
| Prefill | 169 | 132 (78.1%) | 1,928 / 2,696 (71.5%) | 55 | 1 |
| Decode | 180 | 139 (77.2%) | 2,048 / 2,738 (74.8%) | 75 | 1 |

Torch emits five block kernels per layer in prefill. Its K/QK bundle contains
an explicit `ReStickifyOpHBM`; all four major families declare many HBM
allocations. These declarations are structural evidence, not measured HBM
bytes, but they show why producer-consumer values cannot remain globally
scheduled across the full block. SenDNN can choose relayout, LX retention,
double buffering, and operation overlap with visibility across the phase.

The largest Torch opportunity pools are SwiGLU (79.614 ms) and K/QK
(53.799 ms). Together they contain more than the entire 111.036 ms measured
gap. A reasonable matching plan does not require perfect PT efficiency: for
example, making the attention/projection families ideal would leave only
16.30 ms to recover from SwiGLU, while making SwiGLU ideal would leave
31.42 ms to recover from attention/projection.

## Decode: attention/KV memory service

The steady-token model counts compulsory FP16 weights once per layer, K/V
reads at an average context of 514.5, and the new K/V write. It assumes dynamic
activations stay on chip. The 140 GB/s value is an effective-ceiling model, not
a hardware counter.

Fresh full-model steady-token measurements are:

| Group | Useful bytes/layer | Time/layer | Useful GB/s | Versus 140 GB/s | 40-layer gap to modeled floor |
|---|---:|---:|---:|---:|---:|
| Attention/projections/KV | 85.998 MB | 1.413518 ms | 60.84 | 43.46% | 31.970 ms |
| SwiGLU | 314.573 MB | 2.305420 ms | 136.45 | 97.46% | 2.339 ms |
| Residual | small | 0.005849 ms | N/A | N/A | held unchanged |
| **Decoder block** | **400.570 MB** | **3.724787 ms** | **107.54** | **76.82%** | **34.309 ms including phase one-offs** |

Torch steady phase time is 153.114 ms. Holding the measured 4.123 ms one-off
and residual time unchanged gives a 140 GB/s modeled floor of 118.806 ms.
SenDNN steady time is 123.981 ms, only 5.175 ms above that floor. The observed
29.133 ms steady-token advantage is consequently consistent with SenDNN
removing nearly all of Torch's attention/KV deficit while retaining a small
amount of residual overhead.

The Torch steady attention path has six external kernels per layer. The KV
projection/update bundle has seven HBM allocation nodes and no LX allocation;
the QK/softmax bundle has 20 HBM and 16 LX allocation nodes plus an explicit
`ReStickifyOpHBM`. In contrast, decode SwiGLU is already within 2.54% of the
assumed bandwidth ceiling. Optimizing decode MLP first cannot close a 29 ms
gap.

To match SenDNN while retaining Torch's current one-off work:

- if SwiGLU stays at 2.305 ms/layer, attention must reach 125.5 useful GB/s;
- if SwiGLU reaches the 140 GB/s modeled floor, attention must reach
  115.6 useful GB/s.

The concrete attention target is therefore about 0.685-0.744 ms/layer versus
1.414 ms today, or roughly 1.9-2.1x higher useful bandwidth.

## Prioritized Torch-Spyre levers

1. **Put embedding on device and inside the compiled graph.** This is the
   highest-impact end-to-end change by an order of magnitude. It should remove
   roughly 846 ms prefill and 837 ms/token decode from this historical-stack
   run without changing the device kernel-sum comparison.
2. **Reduce external execution boundaries.** Execute a full decoder block or
   a multi-layer persistent plan rather than one compiled region per layer and
   five-to-seven device kernels per layer. The immediate wall target is the
   measured 64 ms prefill / 92 ms decode per-layer-region overhead; the device
   benefit comes only if the larger scope also enables LX retention and HBM
   overlap.
3. **Decode: attack the attention/projection/KV group, not SwiGLU.** Fuse or
   globally schedule Q/K/V projection, KV write, QK, softmax, AV, and output
   projection; double-buffer weights; overlap HBM service with PT; eliminate
   the `ReStickifyOpHBM` and the HBM-only KV-update handoff. Gate success on
   0.685-0.744 ms/layer or 116-126 useful GB/s.
4. **Prefill: integrate tiled/Flash causal attention with on-chip Q/K/V
   handoff.** The K/QK family has the worst PT proxy and the explicit HBM
   restickify. Prior isolated exact-shape causal SDPA reached about 0.706 ms,
   but it must be proven in the production FMS block with projections and
   bundle-level locality evidence before using that number as an E2E
   projection.
5. **Prefill: improve the true-BMM SwiGLU schedule.** It is the largest absolute
   pool. Prior isolated rank-two MM relayout experiments improved, but the
   production rank-three true-BMM FMS relayout/aligned policies regressed by
   roughly 13-14%. Do not repeat pointwise-only work-division tuning; optimize
   true-BMM tiling, PT work balance, double buffering, and cross-op lifetime as
   a production-shaped block.
6. **Treat output-head/one-off work as secondary.** Torch spends 8.752 ms in
   prefill and about 4.123 ms in steady decode outside decoder blocks. Fusing
   these can help, but it does not explain the core device gap.

## Execution roadmap to parity

### M0: make the parity gate permanent

Check the full 40-layer B1/S512 trace into the performance workflow and emit,
for every candidate stack:

- kernel sum, first-to-last accelerator span, and Python wall time per phase;
- external compiled-region and device-launch counts;
- the five prefill family times and six steady-decode family times;
- compiler ideal cycles, HBM/LX allocation nodes, and restickify/relayout
  inventory;
- output token equality and isolated numerical tolerances.

Stop a candidate immediately if it improves wall time but regresses either
device phase, or if it moves traffic without bundle-level realization proof.

### M1: remove graph and execution boundaries

**M1a — device embedding.** `torch_spyre/ops/fallbacks.py` explicitly registers
`aten.embedding.default` as a CPU fallback because indirect indexing is not
supported by the current pointwise framework. Add a real Gather/Embedding
lowering, including `padding_idx`, bounds, FP16 output, and dynamic token-index
tests. Run capture with `fullgraph=True`; any remaining graph break is a bug to
inventory rather than silently accept.

Acceptance:

- no `FallbackWarning` for embedding;
- no large CPU `aten::embedding` event;
- identical tokens and embedding-vs-CPU tensor correctness;
- one captured prefill graph and one decode graph.

**M1b — optimizer-visible phase DAG.** Extend the scheduler/codegen/async
compile boundary so the ordered bundle DAG is compiled as one phase plan.
Preserve internal bundles and expose cross-bundle tensor identities and
lifetimes to the compiler. Cache the prepared plan and launch it once.

Acceptance:

- one framework compiled region and one prepared-plan launch per phase;
- zero HBM materialization for an LX-eligible producer/consumer probe across
  an internal bundle boundary;
- lower first-to-last span, with no kernel-sum regression;
- full-model correctness.

M1 is the enabling work for parity. It attacks approximately 846/837 ms of CPU
embedding time and 64/92 ms of per-layer execution-region overhead, and gives
the device optimizer the scope needed by M2 and M3.

### M2: close decode first

Decode has one dominant device lever, so it is the cleanest proof that the new
phase planner works. Start with only the attention/projection/KV subgraph:

1. keep the KV update in-place with a persistent cache address;
2. stream KV tiles directly into QK/AV rather than materializing a restickified
   copy in HBM;
3. keep Q/K/V intermediates in LX across projection, softmax, and AV where
   lifetime/capacity permits;
4. double-buffer projection/output weights and overlap their HBM service with
   PT work;
5. let the compiler choose compatible divisions or an explicit relayout based
   on measured cost.

The hard per-layer acceptance envelope is:

| Decode budget | Current | Parity target |
|---|---:|---:|
| Attention/projections/KV | 1.4135 ms, 60.84 useful GB/s | <= 0.685 ms if MLP is unchanged; <= 0.744 ms if MLP reaches 140 GB/s floor |
| SwiGLU | 2.3054 ms, 136.45 useful GB/s | do not regress; floor is 2.2469 ms |
| Residual | 0.00585 ms | hold |
| Phase one-offs | 4.123 ms | hold or improve |
| **Steady phase** | **153.114 ms** | **<= 123.981 ms** |

Do not spend the first decode iteration on MLP scheduling: perfecting MLP at
the modeled ceiling saves only 2.34 ms across 40 layers. The attention target
requires roughly 116-126 useful GB/s, so counters should distinguish better
useful service from merely issuing more physical traffic.

### M3: close prefill with a joint attention and MLP budget

Holding Torch's measured 8.752 ms one-off time fixed leaves 181.655 ms for the
40 decoder blocks, or 4.541 ms/layer. The following is a concrete working
budget that reaches parity; it is not the only valid allocation:

| Prefill family | Current/layer | Working target/layer | 40-layer saving | Target PT proxy |
|---|---:|---:|---:|---:|
| Q projection + norm | 0.5329 ms | 0.400 ms | 5.316 ms | 59.6% |
| K projection + QK/softmax | 1.4343 ms | 0.450 ms | 39.373 ms | 19.9% |
| V/AV/output projection + norm | 1.0565 ms | 0.720 ms | 13.459 ms | 45.5% |
| SwiGLU | 4.2245 ms | 2.900 ms | 52.982 ms | 77.0% |
| Residual | 0.0690 ms | 0.069 ms | 0 | N/A |
| **Block** | **7.3173 ms** | **4.539 ms** | **111.130 ms** | **63.7% aggregate** |

This produces about 190.31 ms including unchanged one-offs. It makes the
dependency explicit: neither attention nor MLP alone can close the gap.

**M3a — attention.** Integrate the already demonstrated tiled causal-SDPA
schedule into the production FMS block, then plan the Q/K/V producer-consumer
lifetimes with it. The code seams are the fused-attention decomposition,
`stickify.py`/`insert_restickify.py`, `core_division.py`, and `scratchpad.py`.
The first proof is removal of the K/QK `ReStickifyOpHBM`; the performance gate
is <= 1.570 ms/layer for the combined Q + K/QK + V/AV/output families.

**M3b — true-BMM SwiGLU.** Target <= 2.900 ms/layer. Work on rank-three
true-BMM PT division, double buffering, and gate/up/down lifetime planning.
The existing isolated rank-two all-to-all result is not the target shape, and
the prior production true-BMM relayout/aligned variants regressed. Any new
policy must first beat 4.2245 ms in the exact FMS region and then survive the
full phase plan.

### M4: close residual wall and one-off gaps

Once both device gates pass, fold final norm, LM head, sampling, and token
feedback into the prepared phase where legal. Remove remaining HtoD/memset
setup from the critical path, reuse allocations, and ensure plan preparation
does not occur per token. This stage owns the 8.752/4.123 ms device one-offs and
whatever remains between prepared-plan duration and Python wall time.

### Order of work

The shortest dependency chain is:

`M0 -> M1a -> M1b -> M2 -> M3a + M3b -> M4`

M3a and M3b can proceed independently after the phase-DAG contract exists,
but both must land for prefill parity. M2 should be the first device proof
because it has a single measurable target and does not require improving
decode SwiGLU.

## Minimum experiments to validate the levers

1. Run embedding entirely on Spyre and confirm the fallback disappears; report
   wall time and device kernel sum separately.
2. Add a full-block/persistent execution A/B. Require fewer external launches,
   lower first-to-last span, and unchanged numerics; do not claim device gain
   from launch reduction alone.
3. Build a production FMS prefill-attention A/B. Require removal of
   `ReStickifyOpHBM`, bundle proof of LX producer-consumer handoff, and a lower
   K/QK plus V/AV aggregate time.
4. Build a steady-decode attention-only measurement with Q/K/V weights and KV
   cache. Gate on 0.744 ms/layer first, then 0.685 ms/layer; obtain physical HBM
   counters if available to distinguish useful-payload improvement from
   traffic-amplification reduction.
5. Test true-BMM SwiGLU scheduling independently for prefill and decode. The
   prefill acceptance target is schedule/PT efficiency; the decode acceptance
   target is no regression from the current 136.45 useful GB/s.

## Evidence and caveats

- The SenDNN kernel duration and Torch kernel sum are measured device events.
- PT utilization is compiler ideal cycles divided by measured kernel time, not
  a physical PT occupancy counter.
- Decode GB/s values are semantic useful-byte models at an assumed 140 GB/s
  effective ceiling, not measured bus bandwidth.
- HBM/LX allocation-node counts and `ReStickifyOpHBM` are compiler-structure
  evidence, not dynamic byte counters.
- The single externally visible SenDNN kernel prevents an exact per-family
  timing split from the Kineto trace. The exact observable conclusion is that
  SenDNN realizes the same prefill PT floor much more efficiently and places
  the full decode phase close to the modeled memory floor. Per-family SenDNN
  attribution needs internal bundle counters or controlled graph ablations.

Primary artifacts inspected:

- `results/2026-07-25/full_model_comparison/metrics.json`
- `results/2026-07-25/full_model_comparison/traces/*.pt.trace.json.gz`
- fresh Torch compiler cache under `work/gap-analysis/full_torch_cache`
- SenDNN compiler/export data under `work/gap-analysis/sendnn_export`
- prior exact PT/bandwidth run
  `ptutil_decodebw_140gbs_20260725_020235`
