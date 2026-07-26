# SenDNN vs Torch-Spyre generated-SMC study

## Bottom line

The generated SMCs expose a concrete architectural difference that is directly
actionable for Torch-Spyre performance parity:

1. **Torch-Spyre's repeated decoder-block programs contain no L3 core-to-core
   data-transfer instructions.** Their L3 data motion is expressed through
   memory load/store opcodes. SenDNN emits substantial `LD/LDU/LDG/LDGU` and
   `ST/STU/STG/STGU` peer traffic, while using far fewer memory operations in
   decode.
2. **SenDNN emits a phase-wide, heavily patch-specialized program.** Torch
   emits five external bundle jobs per layer in prefill and six per layer in
   steady decode, and recomputes/transfers a program-correction payload for
   every job.
3. **SenDNN's static synchronization density is lower and its PT program is
   more compact/looped.** This is especially pronounced in decode.
4. **The advantage is not simply “fewer headers” or “fewer PT operations.”**
   SenDNN has more marginal initialization headers, and the prior PT-cycle
   reports show essentially identical prefill arithmetic floors. The useful
   differences are transfer topology, cross-operation scheduling scope,
   patch-based core specialization, synchronization, and PT loop/tiling form.
5. **The post-LXOpt SDSCs identify the exact peer-LX consumers.** Every one of
   the 55 folded prefill and 75 folded decode `LxRelayout` SDSCs is an
   `STCDPOpLx` whose source/destination placement requires data on another
   core. The dominant consumers are BMM inputs in both phases; decode also
   routes attention restickify, softmax, and KV-scatter edges through peer LX.

The highest-priority Torch-Spyre change is therefore a compiler-visible phase
plan that preserves LX ownership across today's bundle boundaries and emits
direct producer-to-consumer L3 transfers. Runtime batching without that
compiler visibility will reduce launch gaps but will not reproduce SenDNN's
device program.

## Scope and performance contract

The comparison uses Granite 3.3 8B Instruct, 40 decoder layers, B1/S512, FP16,
unfused weights, SDPA, and the same five-run/four-phase benchmark used by the
full-model study.

Repository state:

- branch: `adnan/sendnn-granite-antoni-repro-20260725`
- commit: `bec053fa59d4ca81dc0f081acbe074939a9d2abd`

Measured device-program references:

| Phase | SenDNN | Torch-Spyre | Torch gap |
|---|---:|---:|---:|
| Prefill | 190.406 ms | 301.442 ms | 111.036 ms |
| Decode average | 123.961 ms | 153.592 ms | 29.631 ms |

Parity remains the two device gates from the main gap study: prefill at or
below 192.310 ms and decode at or below 125.200 ms, with identical output
tokens and no wall-time optimization hiding a device regression.

## Method

The attached **AIU 1.0 Rapid Core ISA Specification**, revision 2026-01-21,
was used for the packet framing and opcode interpretation. The relevant parts
are sections 2.11.1-2.11.4 for unit initialization, IBUFF layout, parallel
slice arrangement, and patch records; section 3.5 defines the L3 load/store
opcodes.

Both compilers expose the same 128-byte initialization-flit format:

- Torch-Spyre: `spyreCodeDir/init_binary.bin`
- SenDNN: `export_deeprt/.../embedding/init.txt`, one 128-byte flit per row

The Torch binaries were split into the same textual flit form. DeepTools
decoded the packet records, and the IBUFF words were classified using the ISA
opcode tables. All 32 Torch bundle payloads and all four SenDNN phase/control
payloads decoded without a framing failure.

The SenDNN full-model result is normalized with its one-layer control:

```text
marginal decoder layer = (40-layer program - 1-layer program) / 39
```

This removes the common embedding/head/phase setup. It is the average static
contribution of layers 2-40, not a claim that each layer has identical dynamic
execution.

### Important measurement boundary

These are **static delivered-program counts**, not dynamic executed-instruction
counts. IBUFF payloads can contain padding, instructions after an exiting
`RETURN`, and subroutine bodies. A single instruction can execute repeatedly
under loop control and can encode a burst of up to 32 sticks. Patch records can
modify an already delivered core program, but the opcode histogram describes
the decoded base IBUFF slots.

Consequently:

- an opcode count is not a byte count, cycle count, or hardware utilization
  counter;
- `NOP` or `RETURN` slots do not prove runtime idleness;
- the transfer *kind* is reliable, while its dynamic volume requires a trace,
  simulator, or hardware counters;
- unknown slots are below 1.1% in each normalized phase, and the decisive L3
  load/store opcodes are fully decoded.

## 1. Program construction differs at phase scale

SenDNN presents one external job per phase while retaining many internal
bundles: 169 in prefill and 180 in decode. Torch presents every repeated bundle
as a separate external job.

| Static/runtime-plan property | SenDNN prefill | Torch prefill | SenDNN decode | Torch steady decode |
|---|---:|---:|---:|---:|
| External device jobs per call | 1 | 204 | 1 | 244 |
| Repeated block structure | internal phase plan | 5 x 40 + 4 one-offs | internal phase plan | 6 x 40 + 4 one-offs |
| Logical initialization payload | 15.616 MB | 25.329 MB | 14.116 MB | 28.881 MB |
| Regular initialization flits | 50,099 | 143,022 | 40,932 | 161,856 |
| Patch flits | 71,897 | 54,656 | 69,346 | 63,533 |
| Patch share of stream | 58.9% | 27.6% | 62.9% | 28.2% |
| Base IBUFF flits | 102,666 | 102,212 | 62,933 | 103,773 |
| Patched IBUFF references | 35,537 | 9,102 | 24,748 | 11,302 |
| Initialization headers | 17,413 | 13,866 | 17,660 | 14,346 |
| Declared per-execution correction payload | no analogous per-bundle plan exposed | 5.441 MB | no analogous per-bundle plan exposed | 5.947 MB |

The Torch logical payload is the sum of the programs referenced by all jobs in
the phase; it is 1.62x SenDNN in prefill and 2.05x in decode. It is not a claim
that every byte is physically retransferred on every invocation.

Every one of the 32 Torch `spyrecode.json` files has the same plan shape:

```text
JobPreparationPlan: Allocate -> InitTransfer
JobExecPlan:        ComputeOnHost -> DataTransfer -> ComputeOnDevice
```

The `ComputeOnHost` and `DataTransfer` sizes match. The repeated block alone
declares 135,168 correction bytes/layer in prefill and 147,840 bytes/layer in
decode. Four one-offs add 33,792 bytes. SenDNN's export instead contains the
core-specific changes as static patch records inside its single phase program;
the export does not expose an equivalent 204/244-job correction sequence.

The ISA describes a patch record as a common multicore base program plus
selected per-core slice overwrites. SenDNN's much higher patch share and its
35.5K/24.7K patched IBUFF references show that it leans heavily on this form of
core specialization. Torch already uses patch records within each bundle, but
then retains a second dynamic address-correction boundary for every bundle.

### What this rules out

SenDNN is not faster because it has fewer initialization headers. It has more.
The prefill base IBUFF volume is also essentially equal. The meaningful
distinction is that SenDNN's records belong to one coordinated phase plan,
whereas Torch's records belong to hundreds of independently corrected and
submitted jobs.

## 2. The decisive ISA difference is L3 transfer topology

The ISA distinguishes two data paths:

- `LDM/LDMU/LDGM/LDGMU` load from memory into LX; the `GM` forms are memory
  multicast.
- `LD/LDU/LDG/LDGU` load from another Rapid Core's scratchpad; the `G` forms
  are peer multicast.
- `STM/STMU/STIM/STIMU` store from LX to memory.
- `ST/STU/STG/STGU` participate in core-to-core producer/consumer transfers.

The normalized decoder-block programs are:

| Phase and L3 path | Torch per layer | SenDNN marginal layer |
|---|---:|---:|
| Prefill: memory -> LX load sites | 204 | 140.0 |
| Prefill: core -> core load sites | **0** | **83.8** |
| Prefill: LX -> memory store sites | 28 | 39.0 |
| Prefill: core -> core store sites | **0** | **12.1** |
| Decode: memory -> LX load sites | 170 | 59.0 |
| Decode: core -> core load sites | **0** | **217.2** |
| Decode: LX -> memory store sites | 32 | 9.0 |
| Decode: core -> core store sites | **0** | **29.2** |

The paired Torch steady-decode variants have exactly the same aggregate packet
statistics, so this is not an artifact of selecting one token specialization.

This is the strongest SMC result:

- In prefill, SenDNN has 31% fewer static memory-load sites and introduces
  direct peer handoffs.
- In decode, SenDNN has 65% fewer memory-load sites and 72% fewer memory-store
  sites, while peer transfers dominate the generated L3 program.
- Torch emits no peer-transfer sites in either repeated block.

The counts cannot establish exact HBM bytes because burst widths and loop
iterations are dynamic. They do establish that the compilers chose different
physical data paths. The decode result aligns directly with the trace study:
Torch's attention/projection/KV group realizes only 60.84 useful GB/s, while
roughly 116-126 GB/s is required for parity. SenDNN is not merely issuing the
same HBM schedule faster; its SMC routes substantial data between cores
without going back through memory.

## 2A. Post-LXOpt SDSCs identify the peer-transfer operations

The full model was recompiled on `adnan-clc-spyre-dev-pf` with:

```bash
SENDNN_PERFDSC_DEBUG=sdsc,lxopt,lxAnalysis,dsg,isg,bo \
SENDNN_RUN_ROOT=/home/adnan/codex-isolated/sendnn_granite_antoni_20260725/runs/full_40_layer_sdsc_attribution_runtime_20260725_1620 \
scripts/run_sendnn_full_model.sh
```

This retained every final SDSC after LX optimization. `execute_itr0` is
prefill (`mb=512`); `execute_itr256` is decode (`mb=1`). An LX relayout SDSC
contains all of the following:

- the consumer family and input LDS in its filename;
- the exact following consumer in the compiler's final execution order;
- the producer tensor name;
- source and destination `PieceInfo` rectangles;
- LX `memId` core placements for every piece;
- the folded time factor covering repeated decoder layers.

Every retained relayout uses `STCDPOpLx`, and every placement map requires at
least one destination core to obtain data from another core. This is stronger
than merely observing that both tensors have an LX allocation.

| Consumer family | Prefill folded SDSCs | Prefill expanded instances | Decode folded SDSCs | Decode expanded instances |
| --- | ---: | ---: | ---: | ---: |
| `BatchMatMulV2` | 22 | 244 | 23 | 245 |
| `Mul` | 12 | 160 | 15 | 200 |
| `LayerNormNorm` | 12 | 160 | 14 | 162 |
| `Exx2` | 4 | 41 | 7 | 81 |
| `Restickify` | 3 | 40 | 3 | 40 |
| `Max` / `Sub` / `Sum` | 0 | 0 | 9 | 120 |
| `Scatter` | 0 | 0 | 3 | 40 |
| `Add` | 1 | 1 | 0 | 0 |
| `Stcdp` | 1 | 1 | 1 | 1 |
| **Total** | **55** | **647** | **75** | **889** |

The expanded counts reproduce the compiler-log family totals exactly. For
example, BMM has 489 instances across the two phases, `Mul` has 360,
`LayerNormNorm` has 322, `Exx2` has 122, and `Restickify` has 80. The factor-38
records are the middle 38 decoder layers; first/last boundary records remain
separate.

One concrete decode record is:

```text
relayout: BatchMatMulV2_QC_3_inpLds_0_...-LxRelayout
consumer: bmm-BMM_1
tensor:   bmm-actAttnHeadBreak-VirtualReshape_out
source:   32 pieces, each resident on one core
target:   8 pieces, each resident on a four-core group
op:       STCDPOpLx
```

For the first target piece, the destination group is cores
`[0, 8, 16, 24]`, while the corresponding source piece resides only on core
0. Cores 8, 16, and 24 therefore require peer delivery. This is the SDSC-side
representation of the peer multicast observed as `LDG/LDGU` and `STG/STGU` in
the final SMC.

The placement-derived volume emphasizes where to implement parity first:

| Phase | Expanded peer destination-byte demand | BMM share | Attention restickify share |
| --- | ---: | ---: | ---: |
| Prefill | 4,540,039,936 | 94.15% | 0.87% |
| Decode | 242,810,112 | 72.77% | 24.25% |

These are logical destination bytes implied by piece placement, including
replicated destinations and folded layers. They are not measured fabric bytes,
dynamic opcode counts, or cycle allocations. The exact rows are in
`sdsc/sendnn_sdsc_lx_attribution.csv`.

The operation-level conclusion is now concrete:

- **Prefill:** projection/MLP/attention BMM inputs dominate peer-LX volume;
  rotary `Mul`, normalization, and attention restickify are secondary users.
- **Decode:** BMM inputs plus attention-head restickify account for 97.02% of
  placement-derived peer demand. Softmax `Max -> Sub -> Sum -> Mul` and KV
  `Scatter` also use peer LX on every decoder layer, but move much less data.

The dump is before DXP program-address assignment, so it does not retain an
`SDSC -> final IBUFF range` table. Consequently the study can prove the exact
producer tensor, consumer operation, input index, core topology, and aggregate
peer-opcode presence, but cannot assign an individual final `LDGU/STGU` slot
number to a particular SDSC. Adding that range table to the DXP export is the
remaining instrumentation improvement; it does not change the operation-level
attribution above.

## 3. Static schedule density and synchronization

The following table compares one Torch repeated block with the SenDNN marginal
layer. Fractional SenDNN values are expected from the 39-layer normalization.

| Static program metric | Torch prefill | SenDNN prefill | Torch decode | SenDNN decode |
|---|---:|---:|---:|---:|
| Regular flits | 3,520 | 1,245.8 | 3,991 | 1,015.2 |
| Patch flits | 1,355 | 1,791.1 | 1,577 | 1,720.6 |
| Total delivered flits | 4,875 | 3,036.9 | 5,568 | 2,735.8 |
| Base IBUFF slots | 9,606 | 9,499.9 | 9,794 | 5,756.5 |
| `SYNC` slots | 267 | 200.0 | 256 | 150.0 |
| `RETURN` slots | 1,734 | 1,742.1 | 1,541 | 1,246.3 |
| PT IBUFF flits | 1,096 | 832.0 | 1,304 | 512.0 |
| PT `FMA` slots | 2,960 | 1,932.0 | 3,828 | 716.0 |
| PT `MVLOOPCNT` slots | 177 | 263.0 | 177 | 261.0 |
| Initialization headers | 345 | 433.0 | 357 | 439.1 |

SenDNN uses 25% fewer static `SYNC` slots in prefill and 41% fewer in decode.
The largest reductions occur in the L3 store and LX store programs, exactly
where an independent-bundle implementation tends to require producer/consumer
barriers.

The PT encoding is also materially different. SenDNN has fewer PT IBUFF/FMA
slots and more loop-control setup. In prefill, the prior compiler reports show
virtually identical ideal PT cycles for both stacks, so the smaller FMA count
cannot represent less arithmetic. It is evidence of a more compact looped/tiled
microprogram. In decode the difference is larger, but static PT slots still do
not prove dynamic PT cycles. SenDNN's decode program also contains many padded
PT `NOP` slots, which is another reason not to equate static slots with active
cycles.

The return/header data supplies another useful negative result: prefill return
counts are effectively identical and SenDNN has more headers. Eliminating
setup instructions alone is not the path to the 111 ms prefill gain.

## 4. Why this explains the measured gaps

### Prefill

Both compilers report essentially the same PT arithmetic floor, yet SenDNN
realizes 60.71% of that proxy while Torch realizes 39.49%. The SMC difference
is consistent with a realization problem rather than an arithmetic-count
problem:

- SenDNN substitutes direct peer handoffs for some memory loads.
- It has a 25% lower static synchronization count.
- Its phase scope allows prefetch, ownership, relayout, and LX lifetime choices
  to cross the five Torch bundle boundaries.
- Its PT program expresses the same work with more loop reuse and fewer static
  FMA slots.

This does not allocate the 111.036 ms gap to individual mechanisms. Dynamic
unit timelines or controlled codegen experiments are still required for that.
It does identify the missing compiler capabilities that can plausibly raise PT
realization without changing the model math.

### Decode

The decode interpretation is stronger because the independent roofline study
already identifies attention/projection/KV as memory-bound, while SwiGLU is at
136.45 useful GB/s and has little headroom.

SenDNN's decode SMC has:

- 65% fewer static memory-load sites;
- 72% fewer static memory-store sites;
- 217 peer-load and 29 peer-store sites per marginal layer, versus zero in
  Torch;
- 41% fewer `SYNC` slots;
- about 41% fewer base IBUFF slots overall.

This is exactly the direction needed to remove the attention/KV deficit. The
decode work should therefore focus on projection/KV/attention ownership and
on-chip transfer, not further tuning of the already bandwidth-efficient MLP.

### Device gap versus wall gap

One phase job and static patch specialization also explain why SenDNN avoids
Torch's 204/244 external launch-and-correction sequence. That is a high-
confidence explanation for Torch's large first-to-last accelerator gaps and
wall overhead. It cannot by itself explain the kernel-sum gap: the device win
requires the phase compiler to use the larger scope to change placement,
transfer topology, and overlap.

## 5. Concrete Torch-Spyre parity levers

### Lever 1: introduce an optimizer-visible phase DAG

Capture prefill and decode as ordered bundle DAGs with explicit dependencies,
tensor lifetimes, and placement ownership. Preserve internal bundle boundaries
where they are profitable, but let the placement/scheduling stage see across
them.

Relevant seams from the source audit are:

- `_inductor/scheduler.py`: retain a multi-bundle phase instead of emitting a
  `call_kernel` for every fused scheduler node;
- `_inductor/codegen/bundle.py`: add a phase program emitter around internal
  bundles;
- `execution/async_compile.py`: compile the phase artifact once;
- `execution/kernel_runner.py`: prepare and submit one phase job.

Simply raising the individual fusion cap or concatenating everything into one
giant bundle is not the target; prior production-shaped giant-bundle results
regressed when placement became worse.

### Lever 2: make LX ownership and peer handoff first-class

For each producer-consumer edge, the planner should choose among:

1. retain locally in the same core's LX;
2. emit a matched peer producer/consumer route using
   `ST/STU/STG/STGU` + `LD/LDU/LDG/LDGU`;
3. spill/reload through HBM only when lifetime, fan-out, or capacity requires
   it.

The generated SMC must prove that this path is real. A planner log saying an
edge is on-chip is insufficient if the emitted L3 program still contains an
HBM store/load round trip.

Decode should be the first implementation target: projection outputs, K/V
updates, QK/softmax, AV, and output projection have both the largest measured
deficit and the clearest SenDNN peer-transfer signature.

### Lever 3: bind addresses once and specialize cores with patches

Move correction from every external bundle execution into a phase preparation
step. Emit a common base program plus per-core patch records for addresses,
group IDs, tile bounds, and loop counts. Dynamic decode state should update a
small phase-level parameter block rather than regenerate 240 independent
correction payloads.

The first target is to remove the current scaling:

- prefill: 135,168 correction bytes/layer plus 33,792 one-off bytes;
- decode: 147,840 correction bytes/layer plus 33,792 one-off bytes.

### Lever 4: replace bundle barriers with dependency-local synchronization

Derive producer/consumer masks and readiness from the phase DAG, double-buffer
weights/activations, and overlap L3 service with PT work. The static SenDNN
profiles provide directional targets of roughly 200 sync slots/marginal layer
for prefill and 150 for decode, but acceptance must use executed stalls and
latency rather than a raw opcode quota.

### Lever 5: reproduce the PT loop/tiling form without changing the math

SenDNN's lower static PT FMA count plus higher `MVLOOPCNT` use suggests more
reuse of a compact tiled body. Compare tile shapes, XRF pointer setup,
double-buffering, and loop nesting for the production true-BMM SwiGLU and
attention projections. Require no regression in compiler ideal cycles or
numerical behavior.

### Lever 6: keep phase-specific priorities

- **Decode:** implement peer/LX handoff for attention/projection/KV first.
  The target remains 0.685-0.744 ms/layer for that group, or 116-126 useful
  GB/s. MLP tuning cannot close the full gap.
- **Prefill:** integrate tiled causal attention with Q/K/V handoff, then retune
  the true-BMM SwiGLU schedule under the phase planner. These are the two
  largest opportunity pools.
- **Wall time:** put embedding on device and inside the phase graph. This is
  required for wall parity but does not change the SMC explanation for the
  device gap.

## 6. Recommended experiment sequence

1. **One steady-decode block, six internal bundles, one compiler-visible
   plan.** Preserve output correctness. Demonstrate at least one real peer
   producer-consumer edge and removal of its HBM store/load pair in the
   emitted SMC.
2. **Measure the block.** Record device latency, first-to-last span, HBM bytes,
   L3/LX stalls, PT utilization, static opcode mix, and correction payload.
   Do not accept a static-SMC improvement without a device-latency win.
3. **Extend to the 40-layer decode phase.** Submit once, bind addresses once,
   and validate the 125.200 ms parity gate.
4. **Apply the same mechanism to prefill attention.** Prove Q/K/V and
   attention intermediates remain on chip across former bundle boundaries.
5. **Retune prefill SwiGLU under the phase plan** and validate the 192.310 ms
   prefill gate.

The first experiment is deliberately decode-first: it has the strongest causal
chain from trace evidence to ISA topology and therefore gives the fastest
falsifiable test of the proposed architecture.

## 7. Acceptance gates

| Gate | Required evidence |
|---|---|
| Phase visibility | One optimizer-visible dependency/lifetime plan, not runtime-only batching |
| External execution | At most one submitted job per prefill or decode phase |
| On-chip realization | Emitted matched peer L3 opcodes and no HBM round trip on selected edges |
| Program correction | Correction work/payload no longer scales as bundles x 40 layers |
| Synchronization | Lower executed wait/stall time; static `SYNC` count is supporting evidence only |
| Decode locality | Attention/projection/KV reaches 116-126 useful GB/s or equivalent measured HBM reduction |
| Prefill realization | PT ideal work unchanged and actual phase reaches <=192.310 ms |
| Decode parity | Actual phase reaches <=125.200 ms |
| Correctness | Isolated numerical checks plus exact full-model token equality |

No candidate should pass solely because its raw SMC is smaller or because it
contains fewer opcode slots. Placement, bundle evidence, dynamic traces, and
the full-model parity gates must agree.

## Artifact inventory

The analysis was produced from:

- `work/gap-analysis/full_torch_cache/inductor-spyre`: 32 Torch generated
  bundles, `spyrecode.json`, and `init_binary.bin` files;
- `work/gap-analysis/sendnn_export/export/r0_1/export_deeprt`: full-model
  SenDNN prefill/decode exports;
- `work/pod-artifacts/one_layer_b1_s512_20x4_20260725_0905`: SenDNN one-layer
  controls;
- `work/smc-study/*/initpacketstat.json`: packet and patch statistics;
- `work/smc-study/*/isa_summary.json`: decoded base-IBUFF opcode summaries;
- `work/smc-study/aggregate_smc_study.py`: normalization and aggregation;
- `outputs/sendnn_vs_torch_spyre_gap_analysis.md`: trace, PT-utilization, and
  latency context.
- `results/2026-07-25/full_model_comparison/sdsc/sendnn_full_model_post_lxopt_sdsc_20260725.tar.gz`:
  post-LXOpt SDSCs, final PerfDSCs, graphs, phase init streams, and compiler log;
- `results/2026-07-25/full_model_comparison/sdsc/sendnn_sdsc_lx_attribution.json`:
  complete machine-readable topology analysis;
- `results/2026-07-25/full_model_comparison/sdsc/sendnn_sdsc_lx_attribution.csv`:
  one row per folded relayout SDSC;
- `results/2026-07-25/full_model_comparison/smc/generated_smc_study_inputs.tar.gz`:
  Torch generated bundles, init binaries, packet summaries, ISA summaries,
  and the original aggregation helpers;
- `results/2026-07-25/full_model_comparison/smc/generated_smc_study_summary.json`:
  the complete machine-readable SMC comparison;
- `tools/analyze_sendnn_sdsc_lx.py`: dependency-free reproducer.
