# Stage 3B Work-Distribution Results

This note records the first default-off Stage 3B experiment for the Restickify
Locality RFC. Stage 3B is still an experiment, not a default-on optimization.
The goal was to test whether restickify work distribution can preserve producer
ownership when Stage 2 physical remapping is insufficient.

## Implementation

Stage 3B adds:

- `SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1`
- a producer-dominant split-dimension priority for compiler-inserted
  restickify ops
- no change when the flag is disabled

The rule is intentionally narrow. During `work_distribution`, a restickify op is
eligible only when:

- the source has exactly one in-graph producer
- stride-based symbol correspondence is unambiguous
- the producer has one dominant mapped split dimension
- the corresponding restickify output dimension can split with remaining cores

When eligible, the restickify output dimensions are reordered so the
producer-corresponding dimension is considered first by the existing
`multi_dim_iteration_space_split` logic. The implementation does not change
restickify placement, restickify count, tensor layouts, or graph semantics.

## Validation

The following checks passed on the Spyre pod:

| Check | Result |
|---|---:|
| `python -m pytest tests/inductor/test_restickify_mapping_alignment.py -q` | 9 passed |
| `python -m pytest tests/inductor/test_restickify.py -q` with flags off | 97 passed |
| `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 python -m pytest tests/inductor/test_restickify.py -k "opt_adds_then_matmul_x or opt_adds_then_matmul_y_long_chain or opt_matmul_then_adds" -q` | 3 passed |

## Byte-Hop Result

The targeted synthetic pattern was:

```python
(a + b.t() + c.t()) @ d
```

Telemetry was collected with Stage 3A enabled. Baseline used both alignment
flags off. Stage 3B used both Stage 2 and Stage 3B flags on:

```sh
SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1 \
SPYRE_ALIGN_RESTICKIFY_WORK_DISTRIBUTION=1 \
python tools/restickify_scenario_probe.py \
  --case adds_then_matmul \
  --size 128 \
  --size 512 \
  --size 2048 \
  --ring-telemetry \
  --skip-correctness \
  --output-dir /tmp/restickify-stage3b/stage3b
```

| Size | Baseline byte-hops | Stage 3B byte-hops | Reduction | Baseline restickify split | Stage 3B restickify split |
|---:|---:|---:|---:|---|---|
| 128 | 286,720 | 286,720 | 0.0% | `d0:2,d1:2` | `d0:2,d1:2` |
| 512 | 1,376,256 | 655,360 | 52.4% | `d0:8,d1:4` | `d0:4,d1:8` |
| 2048 | 67,108,864 | 0 | 100.0% | `d0:32` | `d1:32` |

The 2048 case is the clearest win: the producer split is `d1:32`, and Stage 3B
makes the restickify split `d1:32` as well. This preserves logical ownership and
the byte-hop estimate drops to zero for the measurable in-graph restickify.

## Runtime Smoke

A small timing smoke was run with `--warmup 5 --iters 30`. These numbers should
be treated as directional only; they are not a full benchmark.

| Size | Baseline median ms | Stage 3B median ms | Median speedup | Byte-hop reduction |
|---:|---:|---:|---:|---:|
| 512 | 0.128563 | 0.129964 | 0.989x | 52.4% |
| 2048 | 1.541122 | 1.492189 | 1.033x | 100.0% |

The 512 case is effectively noise. The 2048 case shows a modest local-kernel
speedup consistent with eliminating a larger byte-hop cost.

## Interpretation

Stage 3B proves the compiler can reduce exact restickify byte-hop cost by
steering work distribution. It also reinforces the project framing:

- the optimization is shape-dependent
- restickify count and bytes moved can remain unchanged while byte-hops fall
- small shapes may not expose runtime benefit
- model-level impact still requires workload-share evidence

The next step should be model-slice telemetry with Stage 3B enabled, not a
default-on change. If model slices show eligible restickifies with meaningful
runtime share, promote the same cases to a repeated timing benchmark.
