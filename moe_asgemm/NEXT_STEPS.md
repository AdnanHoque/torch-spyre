# Next steps

## 1. Split the prototype into reviewable changes

The branch is an evidence branch containing the integrated prototype. Prepare
a clean upstream series with explicit dependencies:

1. Shared-LHS and prepacked expert-matmul schemas and lowering.
2. Compact invariant read copies and flat expert LoopSpec formation.
3. Loop-carried invariant liveness and fixed accumulator placement.
4. Unit expert-sum normalization and affine expert arguments.
5. C32 ownership alignment and SuperDSC core-map preservation.
6. Full-shape structural and correctness tests.

Each change should retain ordinary matmul, reduction, and scratchpad negative
controls. Do not submit the integrated branch as one review unit.

## 2. Reproduce from a clean build

- Build Torch-Spyre from this branch rather than staging individual files.
- Pin the exact DeepTools installation and record loaded library hashes.
- Run the reduced C1 compile and two-alpha correctness gates.
- Run the full `E=128,T=512,H=2816,F=704,C=32` structural gate.
- Repeat the four-AIU timing protocol.

This closes the present overlay provenance boundary.

## 3. Integrate with the team Step 2 model path

Keep ownership explicit. Antoni and Swagath's model integration can invoke the
activation-stationary compiler contract without replacing their host and model
work.

Required integration inputs are:

```text
X       [512,2816]
Wg      [2816,128,704]
Wu      [2816,128,704]
Wd      [704,128,2816]
alpha   [128,512,1]
```

Validate weight packing and router semantics against the real checkpoint.

## 4. Measure end-to-end and energy

- Add router-logit computation.
- Measure the complete MoE layer and full model.
- Measure or estimate AIU energy for dense versus grouped execution.
- Separate first-token compile effects from steady execution.

## 5. Establish the execution phase boundary

Sweep:

```text
tokens              1 through at least 8192
active experts      1 through 128
top-k               representative model values
routing distribution balanced, skewed, and hot-expert
hardware            AIU 1.0 and AIU 1.5 when available
```

The product outcome should be a selector among per-route,
activation-stationary dense, active-dense, and grouped execution rather than a
universal kernel.

## 6. Optional native DDL follow-on

A monolithic custom DDL is not required for the accepted result. Consider it
only after the compiler-generated program is upstreamed and profiled.

A native DDL would still require a composite frontend ABI and a full PT/SFP
schedule for three matmuls, GELU, two multiplies, and accumulation. It should be
judged against the measured `42.506 ms` block baseline, not assumed faster.
