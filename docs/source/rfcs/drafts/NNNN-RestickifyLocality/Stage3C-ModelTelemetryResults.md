# Stage 3C Model-Telemetry Results

This note records the first attempt to answer the model-impact question for the
Restickify Locality RFC:

> Do real model slices contain enough eligible, high-byte-hop, in-graph
> restickifies for this to matter end-to-end?

The short answer is: not proven yet. The compiler mechanism is real and the
synthetic signal is strong, but the current Granite 4 H-Tiny model-op tests do
not yet provide evidence of high-impact eligible restickifies.

## LX Planning 2x2

Before looking at model slices, the proven `adds_then_matmul` pattern was rerun
as a 2x2:

| Mode | Stage 3B | LX Planning |
|---|---|---|
| A | off | off |
| B | off | on |
| C | on | off |
| D | on | on |

Telemetry runs before scratchpad planning, so the expected result was that
`LX_PLANNING` should not change restickify byte-hop telemetry. That is what was
observed.

| Size | A byte-hops | B byte-hops | C byte-hops | D byte-hops |
|---:|---:|---:|---:|---:|
| 512 | 1,376,256 | 1,376,256 | 655,360 | 655,360 |
| 2048 | 67,108,864 | 67,108,864 | 0 | 0 |

Runtime moved modestly:

| Size | A median ms | B median ms | C median ms | D median ms |
|---:|---:|---:|---:|---:|
| 512 | 0.133958 | 0.127943 | 0.128963 | 0.128871 |
| 2048 | 1.553949 | 1.540489 | 1.514945 | 1.505780 |

This supports two conclusions:

- Stage 3B and LX planning compose; LX does not invalidate the ring telemetry.
- The local runtime effect is directional and modest, roughly 2.6% for Stage 3B
  alone at size 2048 in this run.

## Granite 4 H-Tiny Model-Op Probe

The next probe targeted the Granite 4 H-Tiny YAML-driven model-op tests. These
tests are useful for checking op coverage, but they are not true fused model
slices: each test replays one captured op. That means they are weak evidence for
Stage 3B, which needs an in-graph producer-to-restickify edge.

The following passing nodes were run with telemetry in baseline and Stage 3B
modes:

- `torch.Tensor.contiguous`
- `torch.Tensor.view`
- `torch.add`
- `torch.mul`
- `torch.transpose` case 1
- `torch.transpose` case 2

Both modes passed:

| Mode | Result | Telemetry rows | Bytes moved | Byte-hops |
|---|---:|---:|---:|---:|
| Baseline | 6 passed | 0 | 0 | 0 |
| Stage 3B | 6 passed | 0 | 0 | 0 |

The currently failing Granite 4 H-Tiny nodes were also checked in baseline mode:

| Node | Result |
|---|---|
| `torch.nn.functional.linear` | unsupported layout path: `batchmatmul: cannot restickify y to generated_coord=d1` |
| `torch.softmax` | unsupported codegen path: `max on DataFormats.IEEE_FP32` |
| `torch.topk` | unsupported config: `Topk is not supported for this config` |

Those failing nodes emitted zero telemetry rows because they fail before the
restickify telemetry point.

## Interpretation

The model-op run does not show high-impact eligible restickifies. It also does
not disprove the opportunity. It mostly shows that this is the wrong level of
abstraction for the question:

- Stage 3B optimizes restickifies between an in-graph producer and a restickify
  op.
- YAML model-op tests replay mostly single ops, so many possible layout
  boundaries are graph inputs, graph outputs, or absent.
- Some architecturally interesting Granite 4 ops are currently blocked by
  unrelated compiler support gaps before telemetry can observe restickification.

So the current evidence should be framed carefully:

- Proven: exact ring byte-hop telemetry works.
- Proven: Stage 3B can reduce byte-hops by 52.4% at size 512 and 100% at size
  2048 for a synthetic in-graph pattern.
- Observed: local runtime can improve modestly when byte-hops are eliminated.
- Not proven: Granite/model-level end-to-end impact.

## Recommended Next Experiment

The next model-impact experiment should use fused synthetic model slices rather
than op-level YAML replay. Good candidates are small compiled modules that keep
multiple operations in one FX/Inductor graph:

- attention score/value slice: reshape or transpose, `q @ k.T`, mask/add,
  softmax-like substitute if needed, `attn @ v`
- MLP slice: linear or matmul, activation/gating, projection, residual add
- Mamba-style slice: chunk/view/transpose, convolution or projection surrogate,
  gating, scan/state-update surrogate, projection
- MoE-style slice: router/top-k surrogate where supported, token dispatch
  reshape, expert matmul surrogate, gather/combine

The acceptance criterion should remain telemetry-first: show nonzero eligible
in-graph byte-hops in baseline, then show Stage 3B reduces them, and only then
promote the case to repeated runtime timing.
