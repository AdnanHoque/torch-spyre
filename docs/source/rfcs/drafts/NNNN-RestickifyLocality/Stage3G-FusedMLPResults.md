# Stage 3G Fused-MLP Results

This note records the fused MLP/SwiGLU-style experiment for the Restickify
Locality RFC. The goal was to test whether Stage 3B's high-signal projection
result survives when embedded inside a more model-block-like graph.

## Probe Cases

Two forward-looking cases were added to `tools/restickify_scenario_probe.py`:

```python
def mlp_gated_projection(x, w_up, w_gate, w_down, residual):
    up = x @ w_up
    gate = x @ w_gate
    activated = up * silu(gate)
    return activated @ w_down + residual
```

```python
def mlp_gated_projection_join(x, y, z, w_up, w_gate, w_down, residual):
    joined = x + y.t() + z.t()
    up = joined @ w_up
    gate = joined @ w_gate
    activated = up * silu(gate)
    return activated @ w_down + residual
```

`SPYRE_PROBE_INTERMEDIATE` controls the intermediate dimension and defaults to
`SPYRE_PROBE_HIDDEN`.

## Validation

The following checks passed:

| Check | Result |
|---|---:|
| `python3 -m py_compile tools/restickify_scenario_probe.py` | passed |
| `python tools/restickify_scenario_probe.py --list` | MLP cases listed |
| smoke, `tokens=128`, `hidden=512` | 2 rows, 0 errors |

Smoke result:

| Case | Restickifies | Bytes moved | Byte-hops |
|---|---:|---:|---:|
| `mlp_gated_projection` | 0 | 0 | 0 |
| `mlp_gated_projection_join` | 2 | 262,144 | 0 |

## Telemetry Sweep

The sweep used `tokens={512,2048}`, `hidden={1024,2048}`, and
`intermediate=hidden`. Baseline had Stage 3B flags off; candidate had both
Stage 3B flags on.

| Hidden | Tokens | Case | Baseline restickifies | Baseline bytes | Baseline byte-hops | Stage 3B byte-hops |
|---:|---:|---|---:|---:|---:|---:|
| 1024 | 512 | `mlp_gated_projection` | 0 | 0 | 0 | 0 |
| 1024 | 512 | `mlp_gated_projection_join` | 2 | 2,097,152 | 0 | 0 |
| 1024 | 2048 | `mlp_gated_projection` | 0 | 0 | 0 | 0 |
| 1024 | 2048 | `mlp_gated_projection_join` | 2 | 8,388,608 | 0 | 0 |
| 2048 | 512 | `mlp_gated_projection` | 0 | 0 | 0 | 0 |
| 2048 | 512 | `mlp_gated_projection_join` | 2 | 4,194,304 | 0 | 0 |
| 2048 | 2048 | `mlp_gated_projection` | 0 | 0 | 0 | 0 |
| 2048 | 2048 | `mlp_gated_projection_join` | 2 | 16,777,216 | 0 | 0 |

No timing run was performed because no case met the telemetry acceptance rule:
nonzero eligible baseline byte-hops with at least 50% Stage 3B reduction.

## Why The Stress MLP Did Not Reproduce The Projection Win

The stress case did create restickifies, but telemetry attributed them to graph
inputs rather than to an in-graph producer. For the high-signal shape
`tokens=2048`, `hidden=2048`, the restickify plan entries were graph inputs
`arg0_1` and `arg2_1` consumed by pointwise buffers `buf0` and `buf1`.

Ring telemetry reported:

```text
skip_reason = graph-input-or-missing-producer
```

for both restickifies. That makes the case outside Stage 3B's scope. Stage 3B
aligns restickify work distribution to a known in-graph producer; it does not
optimize graph-input layout mismatches.

## Interpretation

This is a useful negative result:

- A natural fused MLP can compile with no restickifies at these shapes.
- Adding the known transposed join before the MLP creates restickifies, but they
  are graph-input sourced.
- The high-signal bare projection case does not automatically survive inside a
  gated MLP block.
- This strengthens the RFC framing: producer-aligned ring locality is only one
  piece of restickify locality; graph-input and weight layout mismatches need a
  different optimization path.

## Recommended Next Step

Stop expanding Stage 3B with more ad hoc synthetic cases for now. The next
high-value work should be one of:

- package Stage 3A/3B as a guarded upstreamable compiler-locality change set
  with this evidence,
- add telemetry attribution for graph-input/weight restickifies so input-layout
  and prepacking opportunities can be measured separately,
- design a decode-specific fused slice that compiles far enough to expose real
  `q_seq=1` plus KV/state layout boundaries.
