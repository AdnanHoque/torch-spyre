# Matrix Multiplication on the IBM AIU — A First-Principles Reference

> A standalone, self-contained guide to how matrix multiplication
> actually executes on the IBM AIU (Spyre) accelerator. Built from
> the architecture specification, the kernel-template literature, the
> torch_spyre Inductor backend, and direct measurements across
> production prefill matmul shapes.

This document assumes you understand matrix multiplication
mathematically and are familiar with how matmul runs on modern GPUs
(at the level of tensor cores, shared memory, async copy, and
warpgroup MMA). It does not assume any prior AIU-specific knowledge.

The goal is to be the one place a practitioner needs to read to:

1. Understand the AIU's compute, memory, and interconnect hierarchy
2. Reason about how a logical matmul becomes hardware actions
3. Predict where performance comes from and where it goes
4. Map AIU concepts onto familiar Hopper / Blackwell concepts
5. Avoid the recurring performance pitfalls

It does not cover model-level concerns (KV cache, attention variants,
quantization schemes), nor does it cover the full software stack
(graph capture, op fusion, scheduling). The focus is the matmul
inner loop and the immediate concentric layers of hardware and
software around it.

## Table of contents

1. [Why the AIU is shaped the way it is](#why-the-aiu-is-shaped-the-way-it-is)
2. [The hardware landscape](#the-hardware-landscape)
3. [The "stick" — the atomic memory unit](#the-stick--the-atomic-memory-unit)
4. [The two reference dataflows](#the-two-reference-dataflows)
5. [Building the kernel: spatial mapping and temporal sequencing](#building-the-kernel-spatial-mapping-and-temporal-sequencing)
6. [Cross-core data movement](#cross-core-data-movement)
7. [Where performance lives — a decomposition](#where-performance-lives--a-decomposition)
8. [The work-division planner](#the-work-division-planner)
9. [Performance pitfalls observed in practice](#performance-pitfalls-observed-in-practice)
10. [Comparison with NVIDIA Hopper and Blackwell](#comparison-with-nvidia-hopper-and-blackwell)
11. [A mental model for reasoning about new shapes](#a-mental-model-for-reasoning-about-new-shapes)
12. [Glossary and references](#glossary-and-references)

## Why the AIU is shaped the way it is

The IBM AIU is a **dataflow accelerator**. This phrase is used
loosely in literature, but on the AIU it means something specific:
the hardware exposes its spatial structure (compute units arranged in
a grid, memory units placed alongside, interconnect links between
them), and software is expected to map operations onto this structure
explicitly. There are no hardware caches transparently absorbing
locality. There is no out-of-order issue. The compiler decides which
core processes which slice of work, which scratchpad holds which
operand, which direction data flows, and how the inner loop pipelines.

This is a deliberate trade. By making the dataflow explicit, the
hardware can be densely packed with compute (the entire chip area is
spent on multiply-accumulate units, scratchpad SRAM, and wires
between them — no caches, no branch predictors, no L1/L2). The
trade-off is that suboptimal mapping shows up as direct performance
loss with no hidden mechanism to absorb it.

Concretely: the AIU has a peak around 150 TFLOPs/s aggregate at fp16.
A NVIDIA H100 has ~990 TFLOPs/s at fp16 with sparsity. On a per-area
basis the AIU is more compute-dense for the matmul-heavy portion of
its die area, but the cost is that maintaining high utilization
requires the software to feed the compute structure correctly. This
document is about how that feeding works.

## The hardware landscape

### The chip — RCU

The reconfigurable compute unit (RCU) is the AIU chip. It contains:

- **32 RaPiD cores**, arranged as a 16×2 grid
- Two **counter-rotating data rings** (CW + CCW), each 128 B wide,
  wrapping the perimeter of the core grid
- A separate **32 B SFP ring** for cross-core partial-sum reduction
- The **HMI** (host memory interface), which connects to the
  off-chip DRAM (LPDDR5)
- The **QGI** (quad-global interface) for chip-to-chip connections
  in multi-AIU systems

Both HMI and QGI are nodes on the data ring. This is a fact that
will come up repeatedly: cross-core operand sharing and DRAM
streaming compete for the same ring bandwidth.

### Inside a RaPiD core — corelets

Each of the 32 cores contains:

- **Two corelets** (CL0 and CL1) sharing a single 2 MB **LX
  scratchpad**
- Two **L3 units** (`L3-LU` for load, `L3-SU` for store) that move
  data between LX and the rest of the chip via the data rings
- A **Ring Interface Unit** (RIU) that mediates ring access

The two corelets are independent compute units. CL0 connects
clockwise on the SFP ring; CL1 connects counter-clockwise.

Note that today's torch_spyre planner addresses cores at a
core-id-of-32 granularity and uses one corelet per core — see the
[per-corelet findings](../../tests/per_corelet_findings.md) for
why and what it would take to use both.

### Inside a corelet — compute and storage

Each corelet contains:

- An 8×8 **PT array** (the "primary tensor" array — these are the
  systolic-array-like multiply-accumulate engines), each PT unit
  with 8-way SIMD inside it. **8 × 8 × 8 = 512 MAC units per
  corelet.**
- A 1D **PE array** (primitive element units, 8-wide) for
  post-PT-array operations like sub-chunk accumulation
- A 1D **SFP array** (special function units, 8-wide) for activation
  functions, reductions, and partial-sum routing
- An **L0 scratchpad** divided into 8 slices of 1 KB each (one per
  PT row)

Inside each PT unit:

- A 64-entry **XRF** (extended register file) — the largest
  register file, used for "weight-stationary" storage of kernel
  values
- A smaller **LRF** (local register file) for "output-stationary"
  storage of partial outputs

### The compute hierarchy in numbers

| level | structure | per-corelet MAC count |
|---|---|---|
| PT array | 8 rows × 8 cols × 8 SIMD | 512 |
| PT row | 1 row × 8 cols × 8 SIMD | 64 |
| PT unit | 1 row × 1 col × 8 SIMD | 8 |
| sub-SIMD lane (INT8 only) | 4 INT8 multipliers per PT unit | 32 (INT8) |

At fp16, the chip's peak compute is `32 cores × 1 corelet × 512
MACs × 2 ops/MAC = ~32,768 ops/cycle`. At ~1 GHz that's ~33 TOPS
of fp16 throughput — but with 2 corelets engaged it'd be 65 TOPS
(today's stack uses one). At INT8 peak is ~130 TOPS via the
double-pumped 4×INT8 sub-SIMD.

These peak numbers are **rarely achieved** in practice for matmul.
The reason is what the rest of this document is about.

### The memory hierarchy

| level | size | who reads | who writes | typical use |
|---|---|---|---|---|
| **DDR (off-chip)** | tens of GB | HMI | HMI | model weights, activations |
| **LX scratchpad** | 2 MB per core | L3-LU, LX-LU | L3-SU, LX-SU | working set for one or more ops |
| **L0 scratchpad** | 8 × 1 KB per corelet | L0-LU | L0-SU | input streaming buffer for PT |
| **XRF** | 64 entries per PT unit | PT | PT (block-load) | weight-stationary storage |
| **LRF** | small per PT unit | PT | PT | output-stationary storage |
| **PE/SFP RFs** | 16 entries each | PE/SFP | PE/SFP | post-PT accumulation, scalars |

Movement between levels is performed by **named programmable
units**: `LX-LU`, `LX-SU`, `L0-LU`, `L0-SU`, `L3-LU`, `L3-SU`. Each
is a small DMA engine with its own program. Synchronization between
producers and consumers of a scratchpad is explicit.

This is markedly different from GPU memory hierarchies. There is no
hardware-managed cache anywhere. Every transfer is scheduled by the
compiler and emitted as instructions in one of the named units'
programs.

### The interconnect

Three logical networks share the chip:

1. **Two data rings (CW + CCW), 128 B per cycle each.** Used for
   moving operand data between LX scratchpads of different cores,
   and between LX and HMI / QGI. The two rings can carry independent
   traffic in parallel; aggregate bandwidth is ~256 B/cycle.
2. **The SFP ring, 32 B per cycle.** Dedicated to partial-sum
   reduction across cores during K-split matmul. Carries no
   regular operand traffic.
3. **HMI as a ring node.** When a core fetches weights from DDR, the
   transfer traverses HMI → ring → that core's LX. When two cores
   share an operand, the transfer traverses one core's LX → ring →
   the other core's LX. **Both kinds of traffic share the data
   ring.** This is critical for understanding bottlenecks.

We measured pure ring-share bandwidth at ~88 GB/s per direction, with
HMI contention adding ~24% overhead when DRAM streaming runs
concurrently. See
[broadcast_topology_findings.md](../../tests/broadcast_topology_findings.md).

## The "stick" — the atomic memory unit

A **stick** is a 128-byte chunk of contiguous tensor data, aligned
to 128-byte boundaries in LX. It is the atomic transfer unit for
LX↔ring↔HMI. **Every memory access decision in the kernel template,
every layout decision, and every work-division check eventually reduces
to "how many sticks does this need."**

For an fp16 tensor, a stick contains 64 elements. For int8, 128
elements. The stick is the AIU analog of a GPU cache line, but it's
explicit at every level of the software stack.

Practical consequences:

- **Tensor inner dimensions must be multiples of stick size.** A
  4096-element row at fp16 = 64 sticks. A 14336-element row = 224
  sticks. An odd-sized row gets padded to the next stick boundary.
- **Work-division splits along stick dimensions must produce
  whole-stick slices per core.** Splitting a 64-stick dim across
  32 cores gives 2 sticks per core. Splitting into 33 cores doesn't
  work — the compiler will reject the split.
- **Layout describes which logical dimensions live "in the stick"
  and which live "outside the stick."** For matmul output `C[M, N]`
  at fp16, the canonical layout is `[M, N/64, 64]` — outer dims
  are M and the count of N-sticks; the innermost dimension is the
  64 elements within each stick.
- **Non-power-of-2 stick counts (especially with factor of 7) are a
  recurring pain point** in the AIU stack. We've observed this in
  L3-70B MLP down (K=28672 = 7 × 4096) and the LX-budget regression
  on L3-8B MLP gate/up (N=14336 → 7 sticks per core under one
  particular split). It's not a bug we can fix from torch_spyre but
  it's a known degraded regime.

The stick is not just a packaging detail; it's a shaping constraint
that affects which splits are valid, which reorderings are legal,
and which performance regimes are accessible.

## The two reference dataflows

Matmul on the AIU runs as one of two **dataflow templates**: weight
stationary (WS) or output stationary (OS). The template choice is
made at compile time based on the shape characteristics, and it
determines almost everything else about how the kernel runs.

A **dataflow template** is more than a kernel implementation. It's a
coordinated specification of:

- Which operand stays in which register file (the "stationary" one)
- Which operands stream past it (and in which direction through the
  PT array)
- How the L0 scratchpad is used as a streaming buffer
- How partial outputs accumulate before writing back to LX
- How chunk-level loops drive the outer iteration

Both templates spatially map output dimensions across the PT array's
columns/SIMD (so different output channels are computed in parallel),
and one reduction dimension across the PT array's rows. The
difference is in what's stationary inside the PT and what flows in.

### Weight stationary (WS) — the workhorse

Used for most matmul where the input channel dimension is "wide
enough" to fill the PT row dimension (8 for fp16, 32 for INT8).
This is the case for almost all transformer prefill matmul.

**What's stationary:** the kernel (weight) tensor sits in the XRF.
Each PT unit holds the weight values it needs to multiply against
incoming inputs. Block-loading the kernel into XRF is a deliberate,
PT-array-idle phase that happens periodically.

**What flows in:** the input tensor streams from LX → L0 → PT-west
edge, then propagates west-to-east across each PT row. All PT
columns in a row see the same input element, multiplying against
their unique weight value.

**What flows out:** the output tensor accumulates north-to-south
across PT rows. The top row (row 0) starts from zero, each
subsequent row adds its product to the partial sum, and row 7 sends
the result south to the PE for further accumulation, eventually to
LX.

```
            input flows W → E
        ┌──┬──┬──┬──┬──┬──┬──┬──┐
input → │00│01│02│03│04│05│06│07│   PT row 0 (8 input channels at once)
        ├──┼──┼──┼──┼──┼──┼──┼──┤
input → │10│11│12│13│14│15│16│17│   PT row 1
        ├──┼──┼──┼──┼──┼──┼──┼──┤
        │  │  │  │  │  │  │  │  │
        ├──┼──┼──┼──┼──┼──┼──┼──┤
input → │70│71│72│73│74│75│76│77│   PT row 7
        └──┴──┴──┴──┴──┴──┴──┴──┘
            ↓  ↓  ↓ ... output flows N → S, accumulating
            output to PE
```

The chip's natural mapping for fp16 is:

- **Output channels (Nout)** → spread across columns (8) and SIMD
  lanes within each column (8) → 64 output channels computed in
  parallel
- **Input channels (Nin)** → spread across PT rows (8) → 8 input
  channels computed per cycle, with row 0 handling channels 0-7,
  row 1 handling 8-15, etc.
- **Per-PT-cycle compute:** `64 output channels × 8 input channels
  × FMA = 512 ops`

### Output stationary (OS) — the corner case

Used when the input channel dimension is too small to fill the PT
rows usefully. The first layer of a CNN with 3 input channels is
the canonical example. Padding the input channels would waste >20×
of the PT compute, so an entirely different mapping is used.

**What's stationary:** the partial output sits in the LRF inside
each PT unit. Each PT unit accumulates one specific output value.

**What flows in:** the input streams W→E (same as WS).

**What flows out:** at the end, the LRF contents are block-stored
to LX (the analog of WS's block-load, but in the other direction).

The chip's mapping for OS-FP16:

- **Output channels (Nout)** → cols/SIMD (same as WS, 64 per cycle)
- **Output spatial (j pixels)** → PT rows (8 pixels per cycle)
- **Per-cycle compute:** still 512 ops, but allocated differently

For matmul, OS is used in special cases (small input channels,
unusual reduction patterns) but the bulk of LLM matmul runs WS. The
rest of this document focuses on WS.

## Building the kernel: spatial mapping and temporal sequencing

A working WS-FP16 matmul kernel is built up from the spatial mapping
above plus several layers of temporal sequencing that together
ensure the PT array doesn't stall and that data is reused
maximally. The dataflow architecture documentation walks through
this in detail; what follows is the condensed version focused on
the why of each layer.

### Layer 1: per-row accumulations and PE sub-chunk reduction

Each PT row produces a partial sum that flows south. To minimize
the latency before all 8 rows are productive, each row sends its
output south after **only 2 multiply-accumulate cycles** (along 2
successive input channels). The remaining accumulations for that
output happen in the PE's register file — the "PE sub-chunk
accumulation."

Why 2 cycles? Because if a PT row produced an output every cycle,
the PE wouldn't be able to consume them fast enough — back-pressure
would propagate into the array and stall it.

### Layer 2: PT interleaving for MAC latency

A PT MAC takes 4 cycles to complete (fp16). Two MAC operations
that accumulate to the *same* output have to be spaced ≥ 4 cycles
apart, or the PT pipeline stalls.

The kernel template handles this by **interleaving 4 different
outputs** through each PT unit. Each output gets one MAC every 4
cycles. Each output's partial sum sits in a separate LRF entry
between updates. Practical result: each output gets exactly the
throughput it would in a non-pipelined design, with the PT array
fully utilized.

For matmul, the interleaving dimension is `j` (the output spatial
dim) for CONV or `mb` (minibatch) for BMM — anything that is "wide
enough" to provide ≥ 4 distinct outputs per PT unit.

For INT8, the MAC latency is 2 cycles, so 2-way interleaving
suffices. The chip uses fewer LRFs accordingly.

### Layer 3: kernel block-loading

Loading kernel values into XRF uses the same data path as input
streaming, so during a block-load the PT array is **idle**. To
minimize this idle time as a fraction of total time, the kernel
block-load is amortized over many computations on the loaded weights
— a "chunk loop" surrounds the inner accumulation loops, reusing
the loaded weights across many input/output pairs.

The utilization upper bound from this is:

```
Util = useful_compute_cycles / (useful_compute_cycles + 8 * num_blocks_loaded)
```

The "8" is the number of PT rows that need to be sequentially
loaded (block-load is row-by-row). To keep utilization high, the
chunk loop must be sized so that `useful_compute_cycles >> 8`. In
practice this means chunk loops of dozens to hundreds of iterations.

### Layer 4: output circulation

The "stationary" weight in WS isn't perfectly stationary — for
shapes where the kernel doesn't fit entirely in XRF, it has to be
re-loaded periodically, and partial outputs have to leave the PE
register file and be written back to LX between block-load
iterations. They then have to be re-read on the next iteration to
continue accumulating.

This LX↔PE round-trip for output is called **output circulation**.
The chunk loop is structured so that output circulation happens at
the slowest possible cadence — typically once per kernel-block-load
iteration.

### Layer 5: chunked input fetch

The input tensor doesn't sit in LX at the start of the kernel
either (in the common case). It's fetched in chunks from neighbor
cores' LX (via the data ring) **concurrently with PT compute**. The
L3-LU asynchronously pulls the next chunk while the current chunk
is being consumed by the PT array. A soft-sync mechanism lets the
LX-LU read input data the moment it's available, without blocking
the L3-LU from queueing the next request.

This overlap is critical. Without it, the cross-core transfer cost
would stall every kernel call. With it, ring traffic largely hides
under compute.

### Layer 6: chunk-level (D/B) loops

The outermost layer is a "double-buffering" loop that walks across
the entire work assigned to this core. Each chunk loop iteration
loads a new kernel block, fetches the corresponding input chunks,
and runs the inner loops to completion. Chunks are double-buffered
when fetching from DDR (Scenario 2 in the architecture doc) so
DDR latency overlaps with compute on the previous chunk.

The full pseudo-code is something like:

```
For chunk_dimensions:               # D/B loops
  Load scalar constants for fused ops
  Load kernel block from LX → PT XRF
  For input_dimensions:              # data-staging loops
    Fetch input chunk from neighbor LX → my LX
    For inner spatial dims:          # batched
      LX → L0 input transfer + sync
      For Nin/Pin, Tki, Tkj:         # per-chunk accumulations
        For Pin,row/2:                # 2 accumulations per row before send
          For LoopPtw:                # 2 accums per output (matched to 4-cycle latency)
            For Tj=4:                 # 4-way PT interleaving
              Compute MAC (send south)
            PE sub-chunk accumulate
        PE writes output to LX
        SFP psum across cores (if K-split)
        SFP fused op (if last block) and write to LX
```

This is the full stack of nested loops that constitute one matmul
kernel call. Every loop has a reason — covering MAC latency, hiding
block-load overhead, amortizing input fetch, double-buffering DDR.

The torch_spyre Inductor backend doesn't generate this loop nest
from scratch on every compile. Instead, it picks a **dataflow
template** (WS-FP16, OS-INT8, etc.) which specifies all of this
structure, and only the per-shape parameters (Tj, Pin, By, Bmb,
chunk sizes) are filled in.

## Cross-core data movement

So far we've described what happens within one core. The next layer
is how work and data are distributed across the 32 cores.

### The work-division decision

For an op with iteration variables (M, N, K), the compiler decides
**how to split each variable across cores**: a triple `(m, n, k)`
where `m·n·k = num_cores`. This is the most consequential
work-division decision for matmul performance. We've shipped one
heuristic that touches it (`output_element_priority`) and
investigated several others.

Each split has a different operand-traffic pattern:

| split | per-core unique A | per-core unique B | per-core unique C |
|---|---|---|---|
| `(m, n, k)` | (M/m × K/k) | (K/k × N/n) | (M/m × N/n) |

If we ignore cross-core sharing for a moment, the **total bytes
each core must pull from DDR** (or from another core's LX) is the
sum of A, B, and C slices for that core. The peak chip throughput
is bounded by min(per-core compute time, per-core data-transit
time), and the data-transit time is bounded by the slowest
operand.

For an LLM prefill matmul with M small (token count), N moderate
(hidden), K large (intermediate or hidden):

- B (the weight matrix, K×N) is by far the largest tensor
- A (the activations, M×K) is small
- C (the output, M×N) is medium

The "worst" split is one that puts the per-core unique B at its
maximum — `(m, 1, 1)` (pure-M split) means every core needs the
**full B**. The "best" split is one that keeps per-core unique B
small — `(1, n, 1)` (pure-N split) means each core needs only
`B / n` worth of B.

### Operand sharing across cores

Cross-core sharing partially absorbs the redundancy in the naive
"every core fetches independently" picture. When the same operand
is needed by multiple cores, the runtime can fetch it once from
DDR (or from one core's LX) and broadcast it across the ring to
all consumers. This shows up as "effective DDR bandwidth" exceeding
the LPDDR5 chip rating in benchmark measurements.

Whether sharing actually fires depends on the operand size relative
to the LX scratchpad. Small operands (~few MB) can stage in LX and
broadcast cheaply. Large operands (hundreds of MB of weights) must
be streamed per-core from DDR even if the work-division pattern
would theoretically benefit from sharing — there's no place to
stage them.

The two ring-direction-independent concepts to keep separate are:

- **Pure ring-share cost** (operand fits in LX, broadcast across
  cores). Measured at ~88 GB/s per direction.
- **HMI streaming cost** (operand doesn't fit, streamed per-core
  from DDR). Bounded by LPDDR5 bandwidth and HMI contention with
  cross-core sharing on the same ring.

The default planner doesn't model these separately — it sees both
as "ring traffic." In practice operand size is the first-order
determinant of which regime you're in.

### Partial-sum reduction (PSUM)

When the K dimension is split across cores (`k > 1`), each core
produces a partial sum of the same output element. These partial
sums must be reduced across the k cores per output before the
output can be written.

The AIU has a **dedicated 32 B SFP ring** for this reduction. Cores
participating in a single PSUM chain are placed adjacently on the
ring (the planner does this). Each core's SFP unit accumulates the
partial sum from its neighbor and forwards to the next neighbor.
The "psum owner" (one core at the end of the chain) writes the
final value to LX.

The PSUM cost scales as approximately **(chain length) × (per-core
partial size)** — both factors matter. This is why pure K-split
`(1, 1, 32)` is empirically slower than mixed `(2, 1, 16)` on
shapes where K-split is competitive: the mixed split halves both
the chain length (16 cores per PSUM chain instead of 32) and the
per-core partial size (output is split across two m-bands), so the
PSUM cost is roughly quartered.

See [psum_split_findings.md](../../tests/psum_split_findings.md)
for the measurement that established this.

### The HMI bottleneck

For any production matmul where weights come from DDR (which is
all of them for a model larger than ~64 MB), HMI bandwidth is the
chip-level bottleneck. The HMI is a node on the data ring, and it
shares ring bandwidth with cross-core sharing.

This explains a recurring pattern in our investigations: many
levers that look promising on paper turn out to be neutralized
because they don't change the total bytes that must traverse HMI.
For example, reordering core IDs (changing which cores share what)
doesn't help if the same total weight bytes must come through HMI
to each core. The element-priority heuristic helps precisely
because it changes WHAT bytes go through HMI (32× redundancy on
the small input A vs. the large weight B), reducing total HMI
traffic.

## Where performance lives — a decomposition

For a single kernel call, wall time decomposes (approximately) as:

```
T_wall ≈ max(
    T_launch_floor,                                # ~3 ms minimum
    max(T_per_core_compute, T_per_core_data_transit)  # pipelined
        + T_psum                                     # if k > 1
)
```

with each term scaling differently with the shape:

- **T_launch_floor** ≈ 3 ms. Per-call fixed overhead. For shapes
  smaller than ~10 GFLOPs total compute, this dominates.
- **T_per_core_compute** = `(M·N·K / num_cores) / per_core_throughput`.
  Per-core throughput depends on dataflow utilization (block-load
  cost, interleaving fill, sub-chunk efficiency). Empirically peaks
  around 0.5-1 TOPS/core for fp16.
- **T_per_core_data_transit** = `per_core_total_bytes /
  effective_per_core_bandwidth`. Effective bandwidth depends on
  whether HMI is contended, whether sharing fires, whether ring is
  saturated.
- **T_psum** = `(chain_length - 1) · per_core_partial_size /
  sfp_ring_bandwidth`. Zero when k=1.

For a typical Llama-70B q-projection prefill `(128, 8192, 8192)`,
with the element-priority pick `(1, 32, 1)`:

| component | value |
|---|---|
| Launch floor | 3 ms |
| Per-core compute | ~1.5 ms (134 MFLOPs/core at ~0.1 TFLOPs/core) |
| Per-core data transit | ~1.5 ms (4 MB B per core via HMI) |
| Wall time (measured) | 4.05 ms |

So this shape is roughly compute-and-data-bound at parity, well
above the launch floor. The element-priority fix moved this from
6.54 ms (with `(32, 1, 1)` split, where per-core data was 32× B =
massive) to 4.05 ms by reducing per-core data transit dramatically.

For a shape like L3-8B q_proj decode `(1, 4096, 4096)`, both
compute and data are tiny, so wall time bottoms out at the launch
floor regardless of split.

## The work-division planner

The torch_spyre Inductor backend's planner
([core_division.py](../../torch_spyre/_inductor/core_division.py))
takes the shape and produces an `(m, n, k)` split. The default
algorithm has three steps:

1. **Span-required splits** (`must_split_vars`). For each tensor,
   compute the per-core memory span under no split. If it exceeds
   the 256 MB hardware limit, split along an output dimension
   enough to bring it under. These splits are committed first.
2. **Priority ordering** (`prioritize_dimensions`). Among the
   remaining dimensions, rank variables for core assignment.
   Output dims rank first by decreasing stick-adjusted size,
   reduction dims rank last.
3. **Greedy core assignment**
   (`multi_dim_iteration_space_split`). Walk the priority list,
   give each variable the largest divisor of its size that fits
   within the remaining core budget.

The default ranking by stick-adjusted size has a unit-mismatch bug
that we fixed via the `output_element_priority` heuristic — see
[element_priority_theory.md](../../tests/element_priority_theory.md).
The fix ranks output dims by element count, which correctly places
N (a stick dimension, large in elements but small in stick count)
ahead of M (non-stick) for typical prefill matmul shapes.

The planner does not currently model:

- **PSUM reduction cost** for K-split. The greedy ranks K last, so
  K-split is rarely picked. When it is competitive (e.g., L3-8B MLP
  down), it requires explicit override.
- **HMI bandwidth contention** across ops in a sequence. Each op is
  planned independently.
- **Per-corelet placement** within a core. The planner only sees
  cores; the unused second corelet is invisible.

These are the three biggest open opportunities for planner-level
improvements based on what we've explored.

## Performance pitfalls observed in practice

These are recurring failure modes from our investigations.

### 1. Stick-vs-element unit mismatch in priority ranking

The bug `output_element_priority` fixes. Default ranking
compares M (in elements) against N (in sticks = elements / 64),
producing an unfair contest where stick dimensions almost always
lose. Result: pure-M splits picked for shapes where pure-N is
clearly better.

### 2. Pure K-split saturates the SFP ring

A K-split of `(1, 1, 32)` produces a 32-core PSUM chain. The chain
length and the per-core partial size both contribute to PSUM cost.
Empirically pure K-split is **slower** than `output_element_priority`'s
pure-N pick on every shape we've measured. K-split only wins when
combined with a slight m-split that halves both chain factors.
See [psum_split_findings.md](../../tests/psum_split_findings.md).

### 3. Non-power-of-2 stick counts surface as outliers

Several puzzles in our data trace back to non-power-of-2 stick
counts (typically a factor of 7 — N=14336 → 7 sticks per core,
K=28672 → factor of 7 in K). The AIU stack handles power-of-2
stick counts efficiently and degrades on non-power-of-2 values in
ways we haven't fully characterized. Manifests as:

- L3-8B MLP gate/up regression at high `DXP_LX_FRAC_AVAIL` (N=14336)
- L3-70B MLP down's "balanced wins" anomaly (K=28672)
- Various measurement outliers in the catalog sweep

If your shape has a stick count divisible only by primes other
than 2, expect surprises.

### 4. LX scratchpad budget is under-tuned by default

`DXP_LX_FRAC_AVAIL` defaults to 0.2. We measured 1.20× peak speedup
on L3-70B q_proj prefill at 0.8. But the lever is shape-dependent —
some shapes regress at high frac (the non-power-of-2 issue above),
so a global default change isn't safe.
See [lx_scratchpad_budget_findings.md](../../tests/lx_scratchpad_budget_findings.md).

### 5. Cross-call weight preload doesn't fire for `torch.compile`

The AIU stack supports a documented preload mechanism (slide 86 of
the architecture doc) that keeps weights LX-resident across
inference calls. It uses a separate `loadmodel_to_spad` dsengraph,
populated by DSM when the input graph has tensors marked
`_OUT_IS_STATIC=1`. **Torch_spyre's Inductor backend never marks
anything as static**, so DSM has no preload nodes to generate, and
the preload phase runs empty. Result: every kernel call re-streams
weights from DDR.

This is currently the largest known unexploited lever. See
[per_corelet_findings.md](../../tests/per_corelet_findings.md) and
the in-progress [preload_investigation_plan.md](../../tests/preload_investigation_plan.md).

### 6. Span pre-split forces sub-optimal mixed splits on huge weights

Shapes where per-core B exceeds the 256 MB hardware span limit
require N-pre-splitting (`n ≥ 2`) before the planner runs. For
L3-70B MLP down `(128, 8192, 28672)`, this forces `(16, 2, 1)`
which happens to be empirically near-best — but it's a happy
coincidence, not because the planner reasoned about it.

### 7. The launch floor is high

Per-call fixed overhead is ~3 ms. For shapes whose total compute is
< 9 GFLOPs (~3 ms × peak chip throughput), wall time is launch-
floor-bound regardless of any split or heuristic. Decode shapes
(M=1) are perpetually here.

## Comparison with NVIDIA Hopper and Blackwell

If you've worked with Hopper SM90 or Blackwell SM100 matmul, the
following mappings will help you transfer intuition.

### Compute structure

| concept | NVIDIA Hopper / Blackwell | IBM AIU |
|---|---|---|
| Smallest compute unit | one CUDA core / one thread of an SM | one PT unit (8-way SIMD) |
| Compute group | warp (32 threads) | PT row (8 cols × 8 SIMD = 64 MAC) |
| Tensor core | TC unit per warp / WGMMA op | the entire 8×8×8 PT array |
| Per-clock matmul throughput | warpgroup MMA: e.g., 64×128×16 in 1 inst | 64 outputs × 8 inputs FMA per cycle |
| Per-SM peak (ish) | ~1 TFLOP at fp16 | ~1 TFLOP at fp16 (per corelet) |
| Per-chip peak | ~990 TFLOPs (H100, with sparsity) | ~150 TFLOPs (AIU, dense) |

The PT array is conceptually similar to a tensor core but at a
more visible level. Tensor cores execute as opaque WGMMA
instructions; the PT array's spatial flow (W-to-E inputs, N-to-S
outputs) is exposed to the dataflow template author.

### Memory hierarchy

| concept | NVIDIA Hopper | IBM AIU |
|---|---|---|
| L1 / SMEM | 256 KB per SM, software-managed | LX scratchpad: 2 MB per core, software-managed |
| L2 cache | ~50 MB on-die | None |
| HBM | ~80 GB | LPDDR5: chip-dependent, GB scale |
| Register file | ~256 KB per SM, divided among warps | XRF (64 entries × per-PT-unit) + LRF + L0 |

Two big differences:

1. **No L2.** GPUs have a chip-wide L2 cache that absorbs cross-SM
   data reuse. The AIU has nothing equivalent — every cross-core
   reuse must be explicitly orchestrated via the data ring.
2. **Larger SMEM-equivalent.** 2 MB LX vs 256 KB SMEM means more
   working set fits on-chip. Particularly relevant for kernel-style
   templates that want to keep weights resident.

### Cross-block / cross-core communication

| concept | NVIDIA Hopper | IBM AIU |
|---|---|---|
| Within block / within core | shared memory + warp shuffle | LX + L0 + PE/SFP RFs |
| Across blocks / cores | cluster mode (SM90+): TMA shared SMEM, asyncbarrier | data rings (CW + CCW), L3-LU/L3-SU |
| Reduction across blocks | global memory + async barriers | dedicated SFP ring for psum |

Two notable differences:

1. **The AIU has a dedicated PSUM ring**. There's no GPU equivalent
   — partial-sum reduction across SMs goes through global memory or
   the L2 cache. The SFP ring is purely for this purpose, doesn't
   compete with operand traffic, and is the basis for K-split being
   architecturally cheap. (Whether software exploits this is
   another question — see pitfall #2 above.)
2. **The AIU's "cluster" is the whole chip.** All 32 cores are on
   the data ring with reasonable latency to each other. Hopper's
   cluster mode covers up to 16 SMs; Blackwell extended this. AIU
   doesn't need cluster mode because the chip's natural unit is
   already inter-core.

### Async copy / DMA

| concept | NVIDIA Hopper | IBM AIU |
|---|---|---|
| Async copy primitive | TMA (Tensor Memory Accelerator) | L3-LU / L3-SU programmable units |
| Granularity | tensor-shaped descriptors | stick-aligned transfers |
| Completion mechanism | mbarrier / async pipeline | explicit forward/back syncs between named units |

Both architectures have async DMA from off-chip → on-chip, both
support overlap with compute, both require explicit programming.
TMA is more recent and more declarative; L3-LU has been the
mechanism since the AIU's inception.

### Launch overhead

This is the biggest practical difference for application code:

- **GPU kernel launch**: ~1-10 µs. Tolerates many small launches.
- **AIU kernel launch**: ~3 ms. Severely penalizes small launches.

For decode-time inference (M=1 per token, lots of small matmul),
the AIU's launch floor is a real ceiling that fusion or batching
needs to amortize. For prefill, it's a non-issue.

### Persistent kernels and weight preload

| concept | NVIDIA Hopper / Blackwell | IBM AIU |
|---|---|---|
| Keep state across launches | persistent kernel (one launch, many tokens processed) | preload mechanism (separate `loadmodel_to_spad` phase) |
| Mechanism | typically streaming work via mbarriers | dsengraph separation between load-phase and execute-phase |
| Status in our stack | well-supported via Triton / CUTLASS | exists in DSM + dxp but not wired to torch_spyre |

The AIU's preload mechanism is conceptually similar to a persistent
kernel pattern, but at a higher level — the "kernel" is the entire
inference pipeline, with weights pre-staged once. Today's torch_spyre
doesn't reach this mechanism, leaving the per-call weight-fetch
cost on the table.

### When AIU and GPU naturally agree

For compute-bound matmul with weights small enough to fit cache or
scratchpad, both architectures look similar: tile the work, keep the
hot operand resident, stream the cold operand, pipeline async copy
with compute. The kernel templates on AIU and the WGMMA-driven inner
loops on Hopper differ in detail but agree on principle.

### When they diverge

For DRAM-bound matmul (large weights), the AIU's lack of L2 makes
cross-core operand reuse explicit. The data ring + LX scratchpad
combo plays the role of L2 + SMEM, but the boundary is sharp and
software-managed.

For tiny matmul (decode), the AIU's high launch floor makes per-op
latency much worse than GPU's. The right answer is op fusion to
amortize, the same answer as on GPU but more urgent.

## A mental model for reasoning about new shapes

When you see a new matmul shape and want to predict its performance
characteristics on the AIU:

1. **Compute the per-core compute time.** `(M · N · K / 32) /
   ~0.1 TFLOPs/core`. If << 3 ms, expect launch-floor-bound
   behavior.
2. **Compute the per-core data transit.** The dominant cost is
   moving B (the largest tensor). If per-core B can fit in 2 MB,
   sharing might fire. If not, per-core HMI streaming dominates.
3. **Identify the work-division regime.** What does
   `output_element_priority` pick? Is N >> M (pure-N wins)? Is K
   huge with small M·N (K-split candidate)? Is span pre-split
   forcing the choice?
4. **Predict the bottleneck.** Compute? HMI? Launch floor? PSUM
   chain (if K-split)?
5. **Pick the right lever**:
   - Bug fix: `output_element_priority` (already shipped)
   - Staging: `LX_PLANNING=1` + `DXP_LX_FRAC_AVAIL` tuning
   - Reduction-axis: K-split with small m (where applicable)
   - Persistent: cross-call preload (when implemented)

## Glossary and references

### Glossary

- **AIU** — IBM AI Unit, the family of accelerators including Spyre.
- **CL0 / CL1** — Corelet 0 / 1, the two compute engines per RaPiD
  core sharing one LX scratchpad.
- **Corelet** — Independent compute unit; contains a PT array, PE
  array, SFP array, and L0 scratchpad.
- **DSM** — Deep Sentient Model compiler in deeptools — the layer
  between torch_spyre and dxp.
- **dxp / dxp_standalone** — The C++ backend compiler that produces
  binaries for execution.
- **DDR / LPDDR5** — Off-chip memory.
- **HMI** — Host Memory Interface; the on-chip block that talks to
  DDR. Sits as a node on the data ring.
- **LX scratchpad** — 2 MB SRAM per core, software-managed.
- **L0 scratchpad** — 8 × 1 KB per corelet; streaming buffer between
  LX and PT.
- **LRF** — Local register file in PT units; small, holds output-
  stationary partial sums.
- **OS dataflow** — Output stationary; partial outputs sit in LRF.
  Used for first-layer-style ops with very small input channels.
- **PE** — Primitive element units; 1D array of 8 doing post-PT
  accumulation.
- **PSUM** — Partial sum; cross-core reduction needed when K-split.
- **PT array** — The 8×8 systolic-style multiply-accumulate array
  (each PT unit is 8-way SIMD).
- **QGI** — Quad-Global Interface; chip-to-chip link.
- **RaPiD core** — One of the 32 compute cores on the chip; contains
  2 corelets sharing an LX scratchpad.
- **RCU** — Reconfigurable Compute Unit; the AIU chip itself.
- **RIU** — Ring Interface Unit; per-core ring access mediator.
- **SDSC** — Spyre data structure schema; the bundle format passed
  from torch_spyre to dxp.
- **SFP** — Special function units; 1D array of 8. Also: the
  dedicated 32 B PSUM ring.
- **Stick** — 128-byte aligned memory chunk; the atomic transfer
  unit between LX, ring, and HMI. 64 elements at fp16.
- **STCDP** — Sub-Tensor Copy with Dimension Permute; the operation
  used for both tensor relayout and weight preload.
- **WS dataflow** — Weight stationary; weights sit in XRF. Default
  for matmul.
- **XRF** — Extended register file in PT units; large, holds the
  weight-stationary kernel block.

### References

The investigation that produced this document built on:

- `tests/element_priority_theory.md` — the planner bug and fix
- `tests/broadcast_topology_findings.md` — ring topology and bandwidth
- `tests/lx_scratchpad_budget_findings.md` — `DXP_LX_FRAC_AVAIL`
- `tests/psum_split_findings.md` — K-split / PSUM characterization
- `tests/per_corelet_findings.md` — corelet utilization gap
- `tests/bidirectional_ring_findings.md` — dual-ring lever
- `tests/preload_investigation_plan.md` — cross-call preload gap
- `tests/session_summary.md` — meta-pattern across all projects

The IBM AIU architecture documentation describes the hardware in
detail; the kernel template literature (slides 7-30 of the
"Dataflows in Sentient Architecture" deck) walks through how the
WS and OS templates are constructed.

For a deeper look at the existing repo documentation:

- `docs/source/architecture/dataflow_architecture.md` — the AIU
  dataflow model overview
- `docs/source/architecture/spyre_accelerator.md` — Spyre device
  characteristics
- `docs/source/compiler/work_division_planning.md` — the planner
  algorithm (note: stale in places, use this document for the
  authoritative view of what the planner does today)
- `docs/source/compiler/work_division_codegen.md` — how plans
  become code (note: also marked stale)

For the GPU side, [Triton's L2 super grouping
documentation](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
covers the matmul tile-traversal optimizations that AIU
core-ordering is conceptually parallel to (though, per the
core-ordering investigation, the lever is different on AIU because
of the explicit dataflow structure).
