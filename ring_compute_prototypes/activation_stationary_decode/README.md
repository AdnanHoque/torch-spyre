# Activation-stationary decode matmul (Design A)

This directory contains the implementation evidence and standalone validation
tools for the opt-in Torch-Spyre decode matmul dataflow.

For an eligible FP16 linear:

```text
C = A[M,K] @ W[N,K].T
```

Design A explicitly pads logical `M <= 64` to physical M64 and lowers:

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
- static flattened logical `M` in `[1, 64]`;
- matching K;
- K and N divisible by 64.

Other shapes use the existing decomposition.

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

At exact revision `2c1ab140d23f7e200ef45ef26b057229cb393727`, that scope
completed the full 40-layer B1/S512 generation and matched the stock decoded
output byte-for-byte. Its one-generation device trace was about 39.73% slower
on average decode, however. An empty shape allowlist selects every eligible
linear and currently fails first-decode compilation in Deeptools output reuse.
See the dated E2E section in `../RING_NATIVE_MATMUL_HANDOFF.md`; do not present
the isolated 2.36x-4.12x timings as Granite E2E speedups.
