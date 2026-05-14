# Restickify Locality: First-Principles Scenarios and Ring-Aware Direction

## Summary

This RFC reframes ring-aware restickify as a forward-looking compiler-locality
project. The goal is not to claim an end-to-end speedup on any single current
model. The goal is to make restickification observable, classify where it can
arise, and create a staged path toward ring-aware layout and work-distribution
decisions when telemetry shows that the affected traffic is material.

Restickification is layout boundary materialization: it is needed when a tensor
edge crosses from one legal Spyre stick layout to another required stick layout.
The amount of data moved matters, but on a ring-connected device the physical
distance between the source core and destination core can matter too. A
restickify that preserves logical ownership on the same or nearby cores should
be cheaper than one that moves the same bytes across distant cores.

The first implementation step should be measurement, not a default-on optimizer.
This RFC proposes a scenario taxonomy, telemetry, synthetic probes, and a staged
optimization plan.

## Background

Spyre tensors use device-specific layouts built around sticks. Most view
operations should be free: the compiler tries to express them as different
coordinate mappings over the same storage rather than materializing copies.
Restickification is the escape hatch for cases where a downstream operation
requires a different stick dimension than the producer currently provides.

The current compiler structure already encodes this model:

| Mechanism | Current role |
|---|---|
| `compute_restickify_needed` | Compares input and target device stick expressions and computes a target layout when the stick must move. |
| `EdgeCostMap` | Scores a restickify edge as `0`, element count, or infeasible. |
| `AllSameNode` | Models pointwise-style joins where inputs and output must be stick-compatible. |
| `FixedInOutNode` | Models ops such as matmul with fixed input and output stick requirements. |
| `finalize_layouts` | Commits selected layouts and records restickify requirements. |
| `insert_restickify` | Synthesizes explicit `spyre.restickify` operations before consumers. |

Today, the optimization target is logical placement of restickifies, measured by
element count. It does not model physical core ownership, ring distance, ring
contention, or repeated graph-input and weight restickification.

## Forward-Looking Model Pressure

Current Granite 3 style experiments are useful but should not be the only lens.
Future architectures may create more layout pressure through mixed token mixers,
expert routing, longer context, and more view-heavy dataflow.

Granite 4 is a useful public proxy for this direction. IBM describes Granite 4.0
as a hybrid Mamba-2/Transformer family, with MoE in select models. IBM also
describes the H-Tiny and H-Small variants as passing the output of each Mamba-2
and transformer block to a fine-grained MoE block. The public H-Tiny config
contains 40 layers, 36 Mamba2 layers, 4 attention layers, 64 local experts, top-6
experts per token, `mamba_chunk_size=256`, and `max_position_embeddings=131072`.

These facts do not prove that restickification is currently a bottleneck on
Granite 4. They motivate synthetic probes that resemble likely future layout
pressure:

- hybrid attention plus state-space dataflow
- token-to-expert dispatch and gather
- shared expert plus routed expert joins
- long-context chunking and decode-state updates
- projection-heavy blocks with transposes, reshapes, and matmuls

## Scenario Taxonomy

| Scenario | Why restickify can occur | Example probe shape |
|---|---|---|
| Graph input or weight layout mismatch | A graph input arrives in a valid layout that the first consumer cannot use directly. | `x @ w`, `x @ w.t()`, `bmm(q, k.transpose(-2, -1))` |
| Pointwise join across mixed layouts | Pointwise ops require compatible sticks across all inputs and output. | `a.t() + b`, `a.t() + b.t() + c` |
| Matmul fixed-stick requirements | Matmul requires lhs, rhs, and output sticks on specific logical dimensions. | `a.t() @ b`, `a @ b.t()`, `(a + b.t()) @ c` |
| View-heavy chains | Views are free until a later op forces a stick dimension that differs from the view-derived layout. | `(a.t() + b).t() + c`, `x.transpose(2, 3).contiguous()` |
| Fanout and diamond graphs | One producer may feed consumers that prefer incompatible layouts. | `u = a + b.t(); u + c; u.t() + d` |
| Attention score path | Q/K/V layout transforms, `k.T`, mask expand, softmax, and `attn @ v` combine different layout needs. | `q @ k.transpose(-2, -1)`, mask `view().expand()` |
| Mamba-style block | Chunking, convolution/state projection, scan/state update, gating, and hidden projection may prefer different split and stick dimensions. | chunked `[batch, seq, hidden]` projections with gate/state joins |
| MoE routing and experts | Router top-k, token grouping, expert matmuls, shared experts, and gather/combine create layout joins. | top-k dispatch to `E` experts, expert output combine |
| Long-context and decode state | Cache/state updates and session packing may create repeated graph-input or persistent-state layout mismatches. | decode step with persistent state read/update |

The taxonomy intentionally includes both scenarios that current Torch-Spyre can
compile today and scenarios that may require more operator enablement before
they become executable probes.

## Current Repo Anchors

The first probes should reuse current compiler and test coverage before adding
new model-shaped tests:

| Scenario class | Existing anchors |
|---|---|
| Layout compatibility and target layout calculation | `torch_spyre/_inductor/pass_utils.py`, especially `compute_restickify_needed` and `compute_restickify_target_layout` |
| Cost model and layout selection | `torch_spyre/_inductor/optimize_restickify.py`, especially `EdgeCostMap`, `AllSameNode`, and `FixedInOutNode` |
| Restickify scheduling and insertion | `torch_spyre/_inductor/insert_restickify.py`, especially `finalize_layouts` and `insert_restickify` |
| Pointwise mixed-stick joins | `tests/inductor/test_restickify.py` tests such as `test_2arg_at_plus_x`, `test_3arg_at_bt_x`, and `test_4arg_at_x_bt_y` |
| Matmul wrong-stick inputs | `tests/inductor/test_restickify.py` tests such as `test_matmul_xt_y`, `test_matmul_x_yt`, and `test_opt_adds_then_matmul_x` |
| View chains and fanout | `tests/inductor/test_restickify.py` tests such as `test_opt_chain_transposed_intermediate`, `test_opt_fanout_intermediate`, and `test_opt_diamond` |
| Attention-style layout pressure | `tests/inductor/test_building_blocks.py::test__simple_attn` and SDPA-shaped tests in `tests/inductor/test_inductor_scalar.py` |
| MoE and Mamba-style probes | No complete restickify-targeted coverage yet; start with synthetic probes rather than claiming current model evidence |

## Ring-Locality Hypotheses

The compiler should distinguish three traffic classes before optimizing:

| Traffic class | Available source ownership | Likely optimization |
|---|---|---|
| In-graph producer to restickify | Producer work distribution is known after `work_distribution`. | Align restickify `coreIdToWkSlice_` with producer ownership. |
| Graph input or weight to restickify | No in-graph producer owns the source. | Choose or precompute input/weight layout for the first high-value consumer. |
| Multi-consumer producer | One producer has several consumers with different layout needs. | Model whether to restickify once near the producer or separately near consumers. |

This split matters because the physical mapping fix for in-graph producers does
not solve graph-input or weight restickifies. Those require layout selection,
prepacking, or persistent-cache strategies.

## Proposed Stages

### Stage 0: Scenario Telemetry and Attribution

Add default-off telemetry that records every restickify with enough context to
classify it into the taxonomy:

- producer, restickify op, and consumer names when available
- whether the source is graph input, parameter, in-graph producer, mutation, or
  fallback-derived
- input and target device layouts
- bytes moved, element count, dtype, rank, shape, and stick dimension change
- producer and restickify work splits when available
- estimated byte-hops when exact producer ownership can be derived
- skip reason when exact ring cost cannot be computed

The first success criterion is coverage: each emitted restickify should have an
actionable class label.

### Stage 1: Synthetic Scenario Probes

Build small probes for each taxonomy row. The probes should run baseline and
telemetry modes in fresh Python processes and produce JSONL plus a summary CSV.
The first win condition is not latency; it is identifying which scenarios
produce nonzero restickifies and which ones are eligible for exact ring-cost
estimation.

Required probes:

- pointwise mixed-stick joins
- matmul wrong-stick inputs
- transpose/view chains
- fanout and diamond graphs
- attention score and value paths
- graph-input and weight restickification
- MoE-style token dispatch, expert matmul, shared expert join, and combine
- Mamba-style chunked projection, state/gate join, and output projection

An initial implementation lives in `tools/restickify_scenario_probe.py`. It uses
the existing `SPYRE_CAPTURE_RESTICKIFY_PLAN` hook, emits JSONL and CSV summaries,
and continues through unsupported forward-looking cases so operator gaps are
recorded instead of hidden. A typical first run is:

```sh
python tools/restickify_scenario_probe.py \
  --include-forward-looking \
  --size 128 \
  --output-dir /tmp/restickify-stage1
```

Latency measurements should be promoted only for probes with nontrivial byte
movement and stable correctness.

### Stage 2: Conservative Physical Mapping Alignment

When a restickify has an in-graph producer and compatible logical ownership,
align the restickify physical core mapping to the producer mapping. This stage
is gated by `SPYRE_ALIGN_RESTICKIFY_CORE_MAPPING=1`, should stay default-off,
and should preserve:

- restickify count
- restickify placement
- selected `SpyreTensorLayout`
- tensor values
- generated behavior when the flag is disabled

If producer and restickify splits cannot be mapped exactly, the optimizer should
skip and report the reason through telemetry.

### Stage 3: Ring-Aware Layout and Work Distribution

Only after telemetry shows high-impact scenarios, extend the optimizer beyond
physical mapping:

- add ring-cost terms to layout/restickify decisions
- prefer work splits that preserve producer-consumer logical ownership
- account for multi-consumer fanout and repeated graph-input/weight restickifies
- consider ring-link contention, not only total byte-hops
- evaluate whether layout selection, work distribution, and core mapping need a
  joint optimizer for high-value patterns

## Measurement Principles

Ring-aware restickify should not be justified by a local kernel speedup alone.
The end-to-end upper bound depends on what fraction of runtime is spent in
eligible restickify-heavy regions. Reports should include:

- affected-kernel speedup
- percentage of model runtime affected
- total bytes moved
- total byte-hops
- eligible versus skipped restickifies
- skip reason distribution
- correctness status

A 5% speedup in a kernel that is 10% of runtime is only about a 0.5% end-to-end
speedup. This should guide how aggressively the compiler changes are framed.

## Public API

This RFC does not propose user-facing API changes. All initial telemetry and
optimization controls should be internal compiler configuration flags and should
default off until validated.

## Test and Validation Plan

Docs-only validation for this RFC:

- build the docs and fix MyST/Sphinx warnings
- check links to source and public architecture references
- confirm the RFC does not claim an unmeasured end-to-end speedup

Compiler validation for future implementation stages:

- unit tests for scenario classification and telemetry schema
- hardware-free tests for ring distance, byte-hop estimation, split mapping, and
  mapping-alignment skips
- existing restickify tests for optimal element-count behavior
- synthetic probe correctness against CPU
- on-device telemetry survey for each taxonomy scenario
- runtime benchmarking only after telemetry shows nontrivial eligible traffic

## Open Questions

- Which graph-input and weight restickifies are repeated enough to justify
  prelayout or persistent-cache work?
- Can work distribution expose enough stable ownership metadata for exact
  byte-hop modeling without coupling too tightly to SuperDSC generation?
- Which MoE and Mamba probes can be compiled today, and which require additional
  operator enablement first?
- Is ring distance sufficient, or do high-value cases require ring-link
  contention modeling?

## References

- [IBM Granite model documentation](https://www.ibm.com/granite/docs/models/granite)
- [IBM Granite 4.0 release blog](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models)
- [Granite 4.0 H-Tiny Base model card](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base)
- [Granite 4.0 H-Tiny Base config](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-base/blob/main/config.json)
- [Torch-Spyre compiler front-end documentation](../../../compiler/inductor_frontend.md)
