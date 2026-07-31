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
