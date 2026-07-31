# Activation-stationary decode matmul (Design A)

This directory contains the implementation evidence and standalone validation
tools for the opt-in Torch-Spyre decode matmul dataflow.

For an eligible FP16 linear:

```text
C = A[M,K] @ W[N,K].T
```

Design A pads logical M to a physical M64 boundary and lowers:

```text
C = (W @ padded_A.T).T[:M]
```

This changes the PT tensor roles without adding a new backend operation:

- padded `A.T` is the XRF-stationary kernel;
- `W` is the West-to-East streamed input;
- the existing BMM backend and ordinary planner perform the work.

The existing weight-stationary decomposition remains the default. Enable the
candidate before importing Torch-Spyre:

```bash
export SPYRE_MATMUL_DATAFLOW=activation_stationary
```

It can also be selected for a scoped compile:

```python
from torch_spyre._inductor import config

with config.patch({"matmul_dataflow": "activation_stationary"}):
    compiled = torch.compile(module, fullgraph=True)
    output = compiled(*inputs)
```

Eligibility is deliberately narrow:

- FP16 activation and weight;
- activation rank at least two;
- 2D weight;
- positive static flattened logical M;
- matching K;
- K and N divisible by 64.

Other shapes use the existing decomposition. Selected Linear weights are
preloaded in the matching K-stick layout. The same streamed-weight schedule is
used at prefill and decode so a selected weight is not repacked inside either
graph.

For staged full-model attribution, restrict the candidate to exact KxN pairs:

```bash
export SPYRE_ACTIVATION_STATIONARY_SHAPES=4096x12800,12800x4096
```

An empty allowlist selects every otherwise eligible shape. The allowlist is a
rollout/attribution control; it does not change either matmul algorithm.

## Focused validation

Host-side decomposition tests:

```bash
python -m pytest -q tests/inductor/test_activation_stationary_linear.py
```

Computed-pad device regression:

```bash
python -m pytest -q \
  tests/inductor/test_restickify.py::test_matmul_with_padded_computed_kernel_restickify
```

Host-only resource model:

```bash
python ring_compute_prototypes/activation_stationary_decode/test_design_a_model.py
python ring_compute_prototypes/activation_stationary_decode/design_a_model.py \
  --logical-m 1 --physical-m 64 --k 4096 --n 4096
```

## Matched device timing

`benchmark_abba.py` compiles stock and Design A in one process, requires
correctness against a CPU reference, inventories the emitted bundles, and
captures serialized incumbent-candidate-candidate-incumbent Kineto events.

Example:

```bash
python ring_compute_prototypes/activation_stationary_decode/benchmark_abba.py \
  --m 1 \
  --k 4096 \
  --n 4096 \
  --warmups 10 \
  --blocks 30 \
  --candidate-source selector \
  --work-division auto \
  --run-dir /tmp/design_a_m1k4096n4096
```

Only Kineto `cat == "kernel"` duration is accepted as device timing. The
candidate event includes padding, identities, restickify, BMM, and slicing.

### Compute-only aligned-M oracle

Use `--boundary compute-only` to isolate the tensor-role/PT schedule from all
layout conversion:

```bash
python ring_compute_prototypes/activation_stationary_decode/benchmark_abba.py \
  --boundary compute-only \
  --m 64 \
  --k 12800 \
  --n 4096 \
  --warmups 10 \
  --blocks 30 \
  --candidate-source manual \
  --work-division auto \
  --weight-layout per-arm \
  --run-dir /tmp/design-a-compute-oracle-m64k12800n4096
```

The oracle preplaces:

- stock A in K-stick layout and W in N-stick layout;
- Design A A in M-stick layout and W in K-stick layout;
- each output in the BMM's direct native layout.

It fails unless both bundles contain exactly one `batchmatmul` root, no
conversion roots, and an emitted ideal-cycle record.

The accepted 30-block device sweeps with automatic work division are:

| Linear | M64 stock / Design A | M512 stock / Design A |
|---|---:|---:|
| K4096 N1024 | 0.8842x | 1.0353x |
| K4096 N4096 | 1.0380x | 0.8810x |
| K4096 N12800 | 0.9525x | 0.9406x |
| K12800 N4096 | 1.0491x | 1.1441x |

Work division must be co-designed with the reversed tensor roles. Use
`--candidate-m-split`, `--candidate-n-split`, and `--candidate-k-split` for an
explicit 32-core candidate. `--candidate-core-order auto` is the default and
keeps placement out of the work-division ablation; `row_major` is a separate
experiment. Use `--incumbent-m-split`, `--incumbent-n-split`, and
`--incumbent-k-split` to test an explicit stock control in the same matched
process. Both triples must multiply to 32.

The best measured schedules are:

| Linear | Best M64 stock / Design A | Best M512 stock / Design A |
|---|---:|---:|
| K4096 N1024 | 1.0480x (`M1 N16 K2`) | 1.1068x (`M8 N4 K1`) |
| K4096 N4096 | 1.0380x (auto) | 0.8810x (auto) |
| K4096 N12800 | 0.9525x (auto) | 0.9549x (`M2 N16 K1`) |
| K12800 N4096 | 1.0491x (auto) | 1.1454x (`M4 N8 K1`) |

The explicit split changes `M64 K4096 N1024` from an automatic-planner loss to
a 1.0480x win. At `M512 K4096 N1024`, it improves the full result from 1.0353x
to 1.1068x. K splitting is useful only for the M64 narrow-output case in this
sweep; at M512 it adds reduction cost and loses.

Design A is not a universal replacement. The winning policy depends on M, K,
and N. The strongest result is the down projection at M512: 1.1454x, or 12.69%
lower latency. The utilization values divide the compiler's ideal PT cycles by
the one-BMM device event; they are not hardware PT-active counters.

Full precision results and trace hashes are in:

- `compute_oracle_m64_results.json`;
- `compute_oracle_m512_results.json`.
- `compute_oracle_work_division_results.json`.

### Why M64 decode gains are modest: matched SMC study

The within-core premise of Design A is valid: make the small activation the
XRF-stationary tensor and stream the large weight through the PT West input.
The chip-level premise is more restrictive. Multicasting a weight shard is
useful only when several cores compute distinct M rows using that same shard.

Fresh `DXP_DEBUG=1` recompilation of the exact timed M64 bundles exposes the
actual L3 programs:

- stock uses `LDGMU` for both operands. Its M split lets four cores share each
  weight shard, while its N split lets four or eight cores share activation;
- Design A uses `LDMU` for disjoint W shards and `LDGMU` for A. A is shared by
  32 cores without K splitting and by 16 cores for the winning K2 schedule;
- the loop-expanded PT compute-FMA count is identical in every matched pair.

`Recipient bytes` below sum bytes delivered to all L3 requestors. They are not
hop-weighted ring-link bytes. `Unique HBM` divides each multicast request by
the active GTR sharer count, recovering the estimated memory response volume.

| M64 shape | Stock / Design A | Unique HBM stock / A | Recipient stock / A | Stock/A XRF loads | Stock/A LX syncs |
|---|---:|---:|---:|---:|---:|
| K4096 N1024 | 1.0480x | 8.5 / 8.5 MiB | 34 / 16 MiB | 4.0x | 4.87x |
| K4096 N4096 | 1.0380x | 32.5 / 32.5 MiB | 132 / 48 MiB | 4.0x | 5.98x |
| K4096 N12800 | 0.9525x | 102.5 / 100.5 MiB | 420 / 116 MiB | 25.0x | 8.79x |
| K12800 N4096 | 1.0491x | 101.5625 / 101.5625 MiB | 412.5 / 150 MiB | 4.0x | 7.50x |

This explains the ceiling. Design A greatly reduces replication after HBM and
reduces block-load/synchronization work, but it does not reduce the mandatory
weight bytes from DRAM. Three rows improve only 3.7%-4.7%. The wide projection
is the decisive contrast: Design A reads 2 MiB fewer unique HBM bytes, delivers
3.62x fewer recipient bytes, performs 25x fewer XRF-load FMAs, and still loses
4.98%. Its PT pointer-control count also rises to 827,904 `XRFACCESS` issues
versus stock's 512,512. Without hardware stall counters, the exact split
between independent-unicast service and PT control is not measured, but ring
replication and XRF block loading are demonstrably not that row's critical
path.

A direct hybrid test tried transposed `M4 N8 K1`: retain stationary A while
restoring four-way multicast of W. It fails structurally:

```text
work_division_hint: buf0 dim d1 size=1 is not evenly divisible by split=4
```

For Design A, physical M64 is the output/stick axis and contains exactly one
stick, so it cannot be divided among four cores. Adding sub-stick replication
would duplicate the same work; for B1 decode, 63 of the physical rows are
padding rather than independent useful tokens. This is not a compiler feature
worth adding for this algorithm.

The resulting decision is:

- keep the measured shape-selective M64 wins as microkernel evidence, with an
  E2E ceiling of roughly 1.0%-1.4% before integration costs;
- do not expect a large B1 decode win from Design A alone;
- revisit activation-stationary execution for genuinely batched decode or
  prefill, where multiple useful M sticks allow cores to share W. The M512
  `M4 N8 K1` down-projection result (1.1454x) is consistent with this condition;
- for B1 decode, pursue a different source of reuse, such as fusing projections
  around their shared activation or batching independent tokens, rather than
  replicating one M stick.

Durable evidence:

- `analyze_smc.py`: loop/GTR-aware SMC accounting;
- `smc_m64_results.json`: four matched timing and SMC pairs with hashes;
- `smc_m64_hybrid_probe.json`: the exact hybrid feasibility result;
- device SMC root: `/tmp/design_a_smc_m64_20260731_v2` on
  `adnan-cdx-spyre-dev-pf`.

### What changes at M512: matched SMC study

The same audit was repeated on the best accepted M512 schedule for each
Granite linear shape. This is the regime where physical M contains eight real
sticks rather than one padded decode stick, so both tensor-role mappings can
divide useful M work among cores.

Every pair passes the original correctness, one-BMM structural, Kineto-order,
and equal compute-FMA gates. The logical grids translate Design A's transposed
BMM coordinates back to the original `C[M,N]` problem.

| M512 shape | Logical grid stock / A | Stock / A | Unique HBM stock / A | Recipient stock / A | Stock/A XRF loads | Stock/A LX syncs | Residual service proxy stock / A |
|---|---|---:|---:|---:|---:|---:|---:|
| K4096 N1024 | M8N4 / M8N4 | 1.1068x | 12 / 12 MiB | 80 / 80 MiB | 2.0x | 1.64x | 47,623 / 36,707 cycles |
| K4096 N4096 | M4N8 / M4N8 | 0.8810x | 40 / 40 MiB | 192 / 192 MiB | 2.0x | 1.65x | 92,242 / 140,126 cycles |
| K4096 N12800 | M4N8 / M2N16 | 0.9549x | 120 / 120 MiB | 560 / 520 MiB | 2.5x | 2.18x | 321,913 / 375,754 cycles |
| K12800 N4096 | M8N4 / M4N8 | 1.1454x | 150 / 125 MiB | 1000 / 600 MiB | 4.0x | 1.65x | 521,140 / 351,028 cycles |

Unlike M64, all eight M512 programs use `LDGMU` multicast for both operands
and emit zero unicast load bytes. The sharer count follows the logical split:
W is shared across M owners and A across N owners. Stock therefore already has
the multicast capability that Design A needs. This traffic is L3/GTR fan-out
of HBM responses; it is not an explicit `STCDPOpLx` core-to-core broadcast.

The first two rows are especially clean fixed-grid controls. Both dataflows
perform the same mathematical PT work, have the same maximum loop-expanded L0
issue count, and move exactly the same HBM and recipient bytes. Design A still
wins the narrow output by 9.65% and loses the square output by 13.51%. In both
rows it halves XRF-load FMAs and reduces LX synchronization, so neither
multicast volume nor those instruction counts explain the crossover.

The residual-service column subtracts the maximum emitted per-core L0 issue
count from device service expressed at the benchmark's 1.1 GHz clock. It
localizes the difference to service not explained by the common L0 issue
floor: Design A removes about 10.9K cycles for N1024 but adds about 47.9K for
N4096 and 53.8K for N12800. This is a schedule-pressure proxy, not a hardware
stall counter. It is consistent with the streamed-weight cadence becoming
less favorable as N grows, but the SMC alone cannot assign those cycles to a
specific hardware queue or dependency.

The down projection is different. Design A changes the grid and cuts repeated
activation response volume: estimated unique HBM bytes fall 150 -> 125 MiB,
recipient delivery falls 1000 -> 600 MiB, and residual service falls by about
170K cycles. That is the strongest mechanistic explanation for its measured
12.69% latency reduction.

The M512 decision is therefore shape-selective:

- retain K4096 N1024 and K12800 N4096 as real conversion-free microkernel
  wins;
- leave K4096 N4096 and K4096 N12800 on stock;
- do not add another broadcast primitive for Design A: the emitted programs
  already multicast both operands;
- if this path is revisited, measure hardware stalls or queue occupancy on the
  same-grid K4096 N4096 crossover before changing the schedule. The current
  evidence does not justify speculative compiler complexity.

Durable evidence:

- `smc_m512_results.json`: four timing/SMC pairs, logical-grid attribution,
  per-unit issue pressure, GTR fan-out, and exact hashes;
- device SMC root: `/tmp/design_a_smc_m512_20260731.fCyy0i` on
  `adnan-cdx-spyre-dev-pf`.

### M512 down projection versus SenDNN

The strongest Design A cell was compared directly with SenDNN FP16 matmul on
the same `adnan-cdx-spyre-dev-pf` device:

```text
[512, 12800] @ [12800, 4096] -> FP16 [512, 4096]
```

The serialized launch order was SenDNN, Design A, SenDNN, Design A, SenDNN.
Every run passed its CPU-reference correctness gate. Each SenDNN run contains
60 `cat == kernel`, `fp16_bmm` events. Each Design A run contains 60 candidate
BMM events from 30 ICCI blocks and passes the one-BMM structural gate.

| Comparison | SenDNN median | Design A median | SenDNN / A throughput | SenDNN latency advantage |
|---|---:|---:|---:|---:|
| Primary, equal 10-warmup bracket | 1006.0895 us | 1051.2275 us | **1.0449x** | **4.29%** |
| Design A 30-warmup sensitivity | 1006.1955 us | 1043.9160 us | **1.0375x** | **3.61%** |
| All events pooled | 1006.1395 us | 1047.0805 us | **1.0407x** | **3.91%** |

The primary throughput is 53.362 TFLOP/s for SenDNN versus 51.071 TFLOP/s
for Design A. The event ranges are disjoint in both controlled comparisons:
the fastest Design A event is still slower than the slowest bracketing SenDNN
event. The earlier archived, cross-stack measurements also agree in direction
(`1003.4225 us` SenDNN versus `1063.847 us` Design A), but are not used as the
primary result.

This is a device-kernel comparison, not host or Granite E2E timing. SenDNN's
static weight preparation is outside the measured Predict loop, and its
`gpu_memcpy`/`gpu_memset` events are excluded. Design A preplaces A and W in
its native device layouts and requests native C, so it likewise excludes
conversion and host transfer. The two frameworks use different frontends
(SenDNN with Torch 2.10 and Design A with Torch 2.11), so this is a serialized
same-device bracket rather than one-process ABBA.

The fresh Design A run still improves over its paired Torch-Spyre incumbent
(`1216.0395 -> 1051.2275 us`, **1.1568x throughput**, or **13.55% lower
latency**) and closes about 78.5% of the incumbent-to-SenDNN latency gap. In
the usual throughput wording, Design A is **15.68% faster than stock** for this
exact down-projection microbenchmark. It does not surpass SenDNN. The remaining
target is approximately 38-45 us, depending on the Design A warmup protocol.

Durable evidence:

- `compare_sendnn_design_a.py`: validates both framework boundaries and
  computes the bracketed comparisons;
- `sendnn_down_m512_comparison.json`: all run statistics, correctness,
  provenance, hashes, and explicit non-claims;
- device root:
  `/tmp/design_a_vs_sendnn_down_m512_20260731.xqyO63`.

#### Final emitted-program comparison

The exact accepted Design A bundle and a fresh final SenDNN SDSC/InitPacket
were decoded with loop expansion. Design A's packet counts reproduce its
independent textual DXP SMC dump for every selected PT and L3 quantity. The
analysis is emitted-program accounting, not a hardware instruction trace or
cycle counter.

| Final-program quantity | SenDNN | Design A | Interpretation |
|---|---:|---:|---|
| Mathematical PT FMA work | 419,430,400 | 419,430,400 | identical |
| Estimated unique HBM response | 125 MiB | 125 MiB | identical |
| L3 recipient delivery | 900 MiB | 600 MiB | A moves 33.3% less |
| Loop-expanded PT issues | 117,361,536 | 112,645,632 | SenDNN has 4.19% more |
| Maximum PT unit-entry issues | 238,761 | 221,404 | SenDNN has 7.84% more |
| XRF-load FMA work | 13,107,200 | 1,638,400 | SenDNN has 8x more |
| Other FMA work | 45,875,200 | 5,734,400 | SenDNN has 8x more |
| `XRFACCESS` issues | 2,950,144 | 6,657,024 | A has 2.26x more |

The static L0 program skeleton is also exactly equal. Consequently, the
remaining SenDNN win is not explained by mathematical work, unique HBM bytes,
recipient fan-out, aggregate PT issue volume, maximum PT entry length, or the
L0 skeleton alone. The leading hypothesis is schedule realization: Design A
uses many more explicit `XRFACCESS` instructions, while SenDNN expresses more
XRF loading through an FMA-based sequence that may pipeline or overlap better.
That is not yet a causal claim. Hardware active/stall counters or a controlled
code-generation ablation are required to prove it.

Durable evidence:

- `analyze_final_initpacket.py`: non-strict InitPacket decoder and loop-aware
  PT/L3 accounting. Mixed LX code/directory records are intentionally excluded
  from dynamic claims;
- `compare_sendnn_design_a_instructions.py`: exact-shape, bundle-hash,
  correctness, SDSC, and packet-versus-SMC gates;
- `sendnn_design_a_instruction_comparison.json`: selected counts, ratios,
  hashes, supported interpretation, and explicit non-claims.

### Large-M result

The same conversion-free oracle was repeated at M4096 and M16384. The
automatic stock planner initially made Design A look as much as 1.82x faster,
so the incumbent was then swept over nearby legal 32-core grids. Against the
best measured incumbent setting, Design A wins only one row:

| Shape | Best stock | Design A | Stock / Design A |
|---|---:|---:|---:|
| M4096 K4096 N1024 | 599.467 us | 662.134 us | 0.9054x |
| M4096 K4096 N4096 | 2754.214 us | 2758.189 us | 0.9986x |
| M4096 K4096 N12800 | 8513.031 us | 7611.254 us | **1.1185x** |
| M4096 K12800 N4096 | 8702.179 us | 8717.557 us | 0.9982x |
| M16384 K4096 N1024 | 2418.206 us | 2759.663 us | 0.8763x |
| M16384 K4096 N4096 | 11142.393 us | 11358.352 us | 0.9810x |
| M16384 K4096 N12800 | 34115.084 us | 36000.008 us | 0.9476x |
| M16384 K12800 N4096 | 51215.819 us | 54194.767 us | 0.9450x |

The surviving M4096 wide-output cell is 10.59% lower latency. At M16384,
tuned stock is faster in every row. The provisional stock-auto ratios of
1.5138x and 1.8214x for the two largest cells were real matched measurements,
but they were planner/decomposition misses rather than wins over the
incumbent's best measured settings.

Exact grids, trace hashes, repeated-trace aggregation, and infeasible-grid
controls are in `large_m_compute_oracle_results.json`.

## Granite E2E

The matched Granite/SenDNN harness and authoritative workload are in:

```text
repository:
  https://github.ibm.com/Adnan-Hoque1/spyre-granite-e2e-bench

branch:
  adnan/sendnn-granite-antoni-repro-20260725

runbook:
  runbooks/full_model_sendnn_vs_torch_spyre.md
```

Use three arms with identical model, prompt, dtype, layer count, warmup, and
measurement boundaries:

1. SenDNN reference;
2. Torch-Spyre with `SPYRE_MATMUL_DATAFLOW=weight_stationary`;
3. Torch-Spyre with `SPYRE_MATMUL_DATAFLOW=activation_stationary`.

Require exact generated-token equality before comparing trace-derived device
program time. Do not compare the profiled Python wall timings.

The accepted isolated results and pinned artifacts are recorded in
`../RING_NATIVE_MATMUL_HANDOFF.md`. They are not Granite E2E claims.

The first E2E-safe scope is now the MLP down projection:

```bash
export SPYRE_MATMUL_DATAFLOW=activation_stationary
export SPYRE_ACTIVATION_STATIONARY_SHAPES=12800x4096
```

At revision `b5d2d4650691deb0e9516678096e8efb023f8405`, this scope uses
K-stick weights, the measured M512 split, and the collapsed-M K-fast fix. It
completed the full 40-layer B1/S512 generation and matched stock output
byte-for-byte, but still regressed the one-generation device screen:

| Device phase | Stock | Down-only Design A | Change |
|---|---:|---:|---:|
| Prefill | 380.755 ms | 442.945 ms | +16.33% |
| Decode average | 162.947 ms | 229.132 ms | +40.62% |

Adding the compute-only-winning `4096x1024` shape reaches first-decode
compilation and fails layout feasibility on an unsupported stick expression.
Therefore the best demonstrated E2E policy remains `weight_stationary`; no
five-generation Design A timing run was justified. Exact roots, trace hashes,
and phase structure are in `e2e_best_policy_results.json` and the dated
handoff section. Do not present the compute-only results as Granite E2E
speedups.

If the winning BMMs can be integrated without incremental conversion work,
their microbenchmark deltas project to about 1.00%-1.43% steady-decode gain,
1.89% prefill gain, and 1.37%-1.61% speedup over the measured
one-prefill/three-decode bracket. This projection is the integration target,
not a measured E2E result.
