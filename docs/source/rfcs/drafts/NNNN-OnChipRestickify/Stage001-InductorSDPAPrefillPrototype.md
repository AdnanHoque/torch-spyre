# Stage 001: Inductor SDPA Prefill Prototype

Date: 2026-05-24

## Purpose

This stage records the first Flash Attention prefill experiment built on top of
the core-to-core primitive work.

The question was whether the newly proven mixed SuperDSC/LX handoff machinery is
enough to start expressing Flash Attention at the Inductor layer.  We tried this
because Spyre already has split-K matmul reductions through the existing BMM/SFP
path, so the next useful test was not a new hardware primitive.  It was whether
Inductor could emit a blockwise online-softmax SDPA shape that uses existing
batch matmul and reductions, then allow the on-chip realization pass to find a
safe same-shard LX handoff inside that graph.

This is intentionally an opt-in prototype.  The stock SDPA decomposition remains
the default.

## Branch And Environment

Implementation worktree:

```text
/home/adnan-cdx/dt-inductor-mixed/torch-spyre-core-to-core-primitive
```

Branch:

```text
AdnanHoque/core-to-core-primitive-20260524
```

Pod:

```text
adnan-cdx-spyre-dev-pf
```

## Code Shape

New default-off config:

```sh
SPYRE_FLASH_ATTENTION_PREFILL=1
SPYRE_FLASH_ATTENTION_PREFILL_BLOCK_SIZE=128
```

The config lives in:

```text
torch_spyre/_inductor/config.py
```

The SDPA overrideable decomposition now has an opt-in prefill branch in:

```text
torch_spyre/_inductor/decompositions.py
```

Eligibility is deliberately narrow:

- 4D query/key/value tensors;
- no dropout;
- no attention bias;
- no causal mask;
- default stock decomposition for everything else.

The prefill branch implements the blockwise online-softmax recurrence using
existing Inductor-visible ops:

```text
QK block matmul
amax over the current KV block
maximum with the running max
exp/rescale correction
sum for the running denominator
BMM for score-block times V-block
final denominator divide
```

That means the prototype exercises existing BMM and reduction lowering instead
of introducing a fused attention op too early.

The on-chip realization pass was also generalized in:

```text
torch_spyre/_inductor/onchip_realize.py
```

The original realization proof matched a hardcoded `add -> add` same-shard edge.
This stage keeps the same fail-closed contract but allows a small pointwise set:

```text
add, exp, identity, maximum, mul, realdiv, sub
```

The detector still requires:

- one producer output;
- exactly one future consumer of that HBM address;
- same work-shard layout;
- a pointwise producer and pointwise consumer;
- size information from the consumer DL op;
- successful LX allocation before mutating the SDSCs.

This lets the SDPA prefill graph expose a production-shaped same-shard LX handoff
without claiming support for arbitrary attention fusion.

## Why This Was Tried

The KB attention design says Flash Attention on Spyre needs more than local data
movement: it needs Q residency, K/V streaming, online softmax state, and in some
shapes cross-program combine.  The core-to-core recipe only proves the lifetime
and movement part.

The useful next experiment was therefore a constrained prefill decomposition that
stays inside the existing Inductor graph model.  If this emits normal BMM and SFP
reductions, then the compiler can start proving pieces of attention without first
requiring a fully fused custom op or Schedule IR implementation.

## Evidence Observed During The Iteration

Focused tests observed passing:

