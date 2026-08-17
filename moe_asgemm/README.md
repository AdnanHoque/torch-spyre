# Activation-stationary dense MoE matmul

This branch preserves the implementation and evidence for the executable
activation-stationary dense MoE path developed during the grouped-versus-dense
investigation.

The branch name is `moe-asgemm`. The implementation is called `D-AS-X` in the
retained evidence. It evaluates all 128 experts for a 512-token prefill chunk,
keeps the shared activation and weighted output accumulator in LX, streams the
three expert-weight banks, applies runtime top-8 router weights after each down
projection, and writes one final output.

This is a real generated AIU program, not a roofline model and not merely a
Python reference. It is implemented as one Torch-Spyre bundle containing one
static expert loop and twelve SDSCs. It is not one monolithic native DDL.

## Status

- Torch-Spyre base: `65508a025f557663c5694e3596c49b814d87517a`
- Full shape: `E=128,T=512,H=2816,F=704,C=32`, FP16
- Structure: accepted on four AIUs
- Correctness: accepted on four AIUs and three routing profiles
- Timing: accepted on four AIUs
- Identity single-call median: `46.416 ms`
- Identity five-call-block median: `42.506 ms` per call
- Representative bundle SHA-256:
  `976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727`
- Exact PR293 model-path baseline on cdx: `372.887 ms` block median
- Integrated model-path AS-GEMM on cdx: `42.444 ms` block median
- Matched model-path improvement: `8.785x`

## Contents

- `moe_asgemm/PROTOTYPE_V0_FREEZE.md`: signed prototype source/evidence freeze,
  oracle identities, and immutability rules.
- `moe_asgemm/prototype_v0_manifest.json`: machine-readable prototype freeze.
- `moe_asgemm/PHASE0_PHASE1_SHA256SUMS`: integrity manifest for the freeze and
  proposed contract package.
- `moe_asgemm/architecture/EXPERT_EXECUTION_CONTRACTS_V1.md`: proposed staged
  semantic, planning, placement, binding, backend, failure, and provenance
  contracts.
- `moe_asgemm/architecture/PHASE1_REVIEW_AND_SIGNOFF.md`: named review roles,
  pending decisions, and the human sign-off gate.
- `moe_asgemm/architecture/IMPLEMENTATION_BASE_DECISION.md`: priced main,
  beta-fork, and prototype-refactor options.
- `moe_asgemm/IMPLEMENTATION.md`: compiler and execution mechanisms.
- `moe_asgemm/RESULTS.md`: correctness, structure, and timing.
- `moe_asgemm/CLEAN_REPRODUCTION.md`: clean-checkout compiler, correctness,
  structure, and two-AIU timing confirmation.
- `moe_asgemm/DECOMPOSITION.md`: measured expert slope, component controls,
  explicit SDSC map, and matmul proxies.
- `moe_asgemm/MODEL_PATH_INTEGRATION.md`: exact PR293 integration and matched
  one-AIU baseline-versus-AS-GEMM measurement.
- `moe_asgemm/patches`: portable exact model-path integration patch.
- `moe_asgemm/NATIVE_DDL_PREDICTION.md`: preregistered native-kernel prediction
  and falsification gate.
- `moe_asgemm/CONTRIBUTION_CHARTER.md`: attribution, reuse, and ownership
  boundary.
- `moe_asgemm/EXECUTION_PHASE_MODEL.md`: measured dense/grouped/per-route
  decision framework.
- `moe_asgemm/ARTIFACTS.md`: retained files and validation identities.
- `moe_asgemm/LINEAGE_AND_SCOPE.md`: prior work, attribution, and our delta.
- `moe_asgemm/NEXT_STEPS.md`: upstreaming and product work.
- `moe_asgemm/VALIDATION.md`: repeatable checks and evidence rules.
- `moe_asgemm/artifacts`: compact C1 and full-bank evidence.
- `moe_asgemm/tools`: fail-closed structural and timing analyzers.
- `experiments/dasx_flat_e2_t64_compile_probe.py`: reduced mechanism probe.
- `experiments/dasx_shared_lhs_c32_schedule_probe.py`: accepted full-shape
  correctness and timing probe.
- `experiments/check_c1_bundle_affine.py`: exact expert-address checker.

## Evidence boundary

The measurements are kernel-level. They exclude router-logit computation and
do not establish end-to-end Gemma latency or energy. The dense program includes
post-down top-8 weighting and accumulation. The retained grouped comparison
does not include its weighting and combine, making the measured dense win a
valid one-sided rejection of that grouped implementation at this prefill point.

No claim is made that dense always beats grouping at other token counts,
routing distributions, or hardware generations.
