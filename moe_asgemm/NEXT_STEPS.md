# Next steps

## 0. Freeze prototype-v0 — completed

The prototype source and evidence are protected by signed tags.  Exact
identities, artifact hashes, and oracle rules are recorded in
`moe_asgemm/PROTOTYPE_V0_FREEZE.md` and
`moe_asgemm/prototype_v0_manifest.json`.

No further production-oriented refactoring belongs on the prototype source.

## 1. Cross-sign the architecture contracts — in progress

The docs-only Phase 1 proposal is:

```text
moe_asgemm/architecture/EXPERT_EXECUTION_CONTRACTS_V1.md
moe_asgemm/architecture/PHASE1_REVIEW_AND_SIGNOFF.md
moe_asgemm/architecture/IMPLEMENTATION_BASE_DECISION.md
```

Implementation remains unauthorized until the semantic, work-division,
allocator, codegen/runtime, backend, and model-path roles approve the same
contract revision and select an implementation base.

## 2. Preserve the old review split as prototype history

The frozen prototype used the following dependency split:

1. Shared-LHS and prepacked expert-matmul schemas and lowering.
2. Compact invariant read copies and flat expert LoopSpec formation.
3. Loop-carried invariant liveness and fixed accumulator placement.
4. Unit expert-sum normalization and affine expert arguments.
5. C32 ownership alignment and SuperDSC core-map preservation.
6. Full-shape structural and correctness tests.

This list documents how the proof was assembled.  It is not the production
review plan.  The contract-driven implementation will retain sound general
mechanisms and rewrite the condemned policy paths after Phase 1 sign-off.

## 3. Clean-source reproduction — completed

Completed from a clean checkout of the exact branch with an exact compatible
base native extension:

- affected compiler suites passed;
- reduced C1 compile and two-alpha correctness passed;
- full `E=128,T=512,H=2816,F=704,C=32` structure and correctness passed; and
- two AIUs completed the fixed timing protocol before the user closed the
  replication scope.

The clean full-shape bundle is byte-identical to the earlier accepted bundle.
See `moe_asgemm/CLEAN_REPRODUCTION.md`.

This closes the present overlay provenance boundary.

## 4. Integrate with the team Step 2 model path — completed

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

Completed against retained hf-adapters revision
`672b2fc8b5f017a08c6b43b928deb3ccd0560761`:

- the existing dense router math is preserved;
- `[T,E]` router weights are exposed as load-bearing `[E,T,1]` alpha;
- expert weights are prepacked into the three directly streamed banks;
- the eager model orchestrator registers semantic tensor dimensions and the
  validated LX compiler policy before first compilation;
- the patched model helper emits the byte-identical accepted AS-GEMM bundle;
- exact PR293 and integrated AS-GEMM expert paths passed matched one-AIU timing.

The measured amortized improvement was `372.887 / 42.444 = 8.785x`.  See
`moe_asgemm/MODEL_PATH_INTEGRATION.md`.

## 5. Measure end-to-end and energy

- Add router-logit computation.
- Measure the complete MoE layer and full model.
- Measure or estimate AIU energy for dense versus grouped execution.
- Separate first-token compile effects from steady execution.

## 6. Establish the execution phase boundary

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

## 7. Decomposition and native-DDL prediction — completed

A monolithic custom DDL is not required for the accepted result. Consider it
only after the compiler-generated program is upstreamed and profiled.

A native DDL would still require a composite frontend ABI and a full PT/SFP
schedule for three matmuls, GELU, two multiplies, and accumulation. It should be
judged against the clean measured `42.408-42.592 ms` identity block range, not
assumed faster.

Completed before implementing it:

1. measured a five-point expert sweep with `R²=0.999949`;
2. measured full-graph pointwise substitutions without changing matmul count or
   residency;
3. measured M32 gate/down matmul proxies and recorded their extra HBM traffic;
4. preregistered an expected native-DDL gain of at most five percent and a
   falsification threshold of at least ten percent.

AIUPTI exposes one device job rather than internal SDSC timestamps, so the
decomposition is explicitly experimental rather than a fabricated per-SDSC
trace. See `moe_asgemm/DECOMPOSITION.md` and
`moe_asgemm/NATIVE_DDL_PREDICTION.md`.
