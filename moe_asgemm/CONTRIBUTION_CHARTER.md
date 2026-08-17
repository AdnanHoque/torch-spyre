# Contribution charter

## The claim

Our defensible claim is the compiler realization and measurement framework for
activation-stationary all-expert MoE execution on Spyre.

The claim consists of:

- a real compiler-generated AIU kernel with one static expert loop;
- shared-LHS expert matmul contracts and lowering;
- invariant activation residency and fixed accumulator residency in LX;
- direct affine expert-weight and runtime-alpha streams;
- exact C32 work ownership and physical core-map preservation;
- fail-closed structure and correctness gates;
- clean-source device reproduction;
- a measured expert-scaling curve;
- matched component and matmul controls; and
- a preregistered native-DDL prediction and falsification rule.

This is broader than authorship of one kernel and narrower than ownership of the
team's complete model path.

## What we built on

Plain source pointers:

```text
Torch-Spyre base commit 65508a025f557663c5694e3596c49b814d87517a
torch_spyre/_inductor/scratchpad/lx_relayout.py
torch_spyre/_inductor/scratchpad/allocator.py
torch_spyre/_inductor/wsr/coarse_tile.py
torch_spyre/_inductor/spyre_kernel.py
torch_spyre/_inductor/codegen/superdsc.py
torch_spyre/_inductor/cost_model.py
```

We reused the existing coarse-tiling, LX-planning, relayout, core-mapping, and
SuperDSC mechanisms instead of creating an unrelated hero-kernel stack.

## What we added

The source delta is organized on branch `moe-asgemm-review-series`:

```text
3ba559f9  activation-stationary expert matmul contracts
6b9d4654  flat expert loop and invariant activation state
9b4c78e3  expert-loop LX residency and accumulator placement
fb6cc77a  expert affine addresses and physical core-map codegen
441db9c6  compiler regression suite
9516d4be  self-contained validation probes
```

The evidence and decision instruments live on branch `moe-asgemm`.

## Matmul plus core-to-core framing

The work legitimately belongs under a matmul-plus-ownership umbrella:

1. Matmul program shape

   Shared-LHS gate/up projections and the down projection are expressed as
   expert-loop matmul leaves rather than materializing `[E,T,H]` activations.

2. Ownership

   The shared activation, gate, up, pointwise chain, down output, and
   accumulator are assigned compatible core views. The compiler preserves the
   chosen physical map through SuperDSC emission.

3. Transport

   The accepted representative schedule uses transport-free M32 row ownership;
   no explicit core-to-core shuffle is needed at that point. The contribution
   is not a claim that a shuffle happened. It is the ownership analysis that
   proves when direct reuse is legal and fails closed when a relayout would be
   required.

4. Future schedules

   If a faster M/N/K split requires redistribution, the existing relayout and
   core-to-core infrastructure is the correct place to express and measure it.
   That becomes a planner choice inside the same framework.

## Attribution boundary

We should not claim:

- invention of dense all-expert MoE as an algorithm;
- ownership of other engineers' model adapter or host integration;
- ownership of per-token selected-expert execution;
- a native custom DDL that has not been implemented; or
- universal dense superiority across token counts and hardware generations.

We can claim:

- the first retained compiler-generated, activation-resident all-expert
  realization in this workstream;
- the compiler mechanisms required to make it real;
- the structural contracts that distinguish it from a spilling proxy;
- the clean correctness and timing evidence; and
- the framework that decides when dense, grouped, or per-route execution wins.

## Review and collaboration posture

The implementation series stays on the frontend/compiler side. Model owners can
invoke the contract without transferring ownership of their integration. A
backend-native DDL, if pursued, should be co-designed as a falsification control
against the retained prediction rather than treated as a replacement for the
decision framework.
