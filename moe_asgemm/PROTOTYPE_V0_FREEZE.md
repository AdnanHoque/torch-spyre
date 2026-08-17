# AS-GEMM prototype-v0 freeze

## Status

Prototype-v0 is frozen as of 2026-08-17.  It is an experimental proof and an
oracle for the architecture work that follows.  It is not production code and
must not be incrementally converted into the production design.

The freeze has four purposes:

1. preserve the exact compiler mechanism that produced the accepted program;
2. preserve the structure, correctness, and measurement evidence;
3. preserve the model-path adapter used for the matched comparison; and
4. give the replacement architecture a stable burden-of-proof target.

The two Torch-Spyre identities are protected by signed annotated tags.  Tags
must never be moved.  Any later experimental prototype must receive a new
version and new tags.

## Exact identities

### Compiler source and compact validation

```text
branch                 moe-asgemm-review-series
signed tag             moe-asgemm-prototype-v0-source
tag object             1f5ef138d0349c6237622e4720cd5cf5ebda31c2
validation commit      31bc00d4ca74f7e74ded656fb9c6432399690194
validation tree        1060877a31dbfb04985b1112f57d423f577cbce4
production commit      dffd639fdcbb8e69db29a380d1644eb95ebda3a0
production tree        0063ed3c0c824d0c264c7c64b0f8f298f6cbdc45
upstream base          3fd7a0f954a84817a417b6b45639b0d5f3499575
```

### Evidence, decomposition, and matched model comparison

```text
branch                 moe-asgemm
signed tag             moe-asgemm-prototype-v0-evidence
tag object             7d83fd8ec0f8ad751e4fd8a8074a4f2e889a38b0
evidence commit        06c9389dab1ccb462bc422831db142a8f53d1e7a
evidence tree          2587e9cebd5cdbfd040bb21745815ea84432b09c
root manifest SHA-256  87c7d30f8bb8cd4bb52704822e317117a049e01a0812c865e5a49482f1720d12
```

### Model adapter

```text
base revision          672b2fc8b5f017a08c6b43b928deb3ccd0560761
signed adapter commit  fc5198f9f0ffd1a9e458526ba7e9e9010b37092d
adapter tree           0ffb2609969637c03dce373e4925165fe92dc7e4
portable patch         moe_asgemm/patches/0001-Integrate-activation-stationary-expert-compiler-path.patch
patch SHA-256          a86fee8d96d9fd238e524def21431ae072fb9eae4646f3feecebf4fcd63a1f30
```

The portable patch in the evidence tag is the durable copy of the adapter.
The adapter repository itself is not part of the Torch-Spyre remote.

## Oracle artifacts

```text
reduced C1 bundle
  moe_asgemm/artifacts/c1_compile/bundle.mlir
  SHA-256 977d62e74ca5f4a3ff28a89be32e4531ecb858828b77cedbf21e007c05c0c720

reduced C1 correctness
  moe_asgemm/artifacts/c1_correctness/result.json
  SHA-256 06e9ed0206a22707e73adb0a0177fc711bd4813179913ec86d987a342e3a7bfd

representative E128 C32 bundle
  moe_asgemm/artifacts/fullbank_timing/cdx_bundle/bundle.mlir
  SHA-256 976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727

four-AIU aggregate
  moe_asgemm/artifacts/fullbank_timing/comparison.json
  SHA-256 140da8c60971f22033d6a4858f72a3f672a444e39e9244fd4057e19a118d5ec4

decomposition
  moe_asgemm/artifacts/decomposition/analysis.json
  SHA-256 e5aed198d250f5827b2aa84c2e1075d8f65c0750a9d46914653e3f32b26e939e

exact model-path baseline bundle
  moe_asgemm/artifacts/model_integration/pr293_baseline_cdx_02/bundle.mlir
  SHA-256 6bcf466e6a9ac74f7eef3265750a55545dbeba220df4d1828bbfdfa4bfa12f43

integrated model-path AS-GEMM bundle
  moe_asgemm/artifacts/model_integration/pr293_asgemm_cdx_05/bundle.mlir
  SHA-256 976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727

matched model-path comparison
  moe_asgemm/artifacts/model_integration/comparison.json
  SHA-256 d2fdc28d15f5238df7cb04c94e8df8e5b26c1e2e27c7bf4f301bb4595fc4d028
```

## What the oracle proves

At `E=128,T=512,H=2816,F=704,C=32`, the prototype demonstrated:

- one wrapper call, one bundle, and one static expert loop;
- one activation HBM-to-LX preheader shared across all experts;
- direct affine-advanced Wg, Wu, Wd, and alpha operands;
- runtime `[E,T,1]` weighting after down projection;
- all internal activations and the carried accumulator in LX;
- zero HBM-pool intermediates and zero HBM restickify operations;
- one final LX-to-HBM output drain;
- correct response to distinct routing-weight payloads; and
- a matched one-AIU expert-callable comparison of 372.887 ms versus 42.444 ms.

The historical 42.444 ms value is not a permanent regression threshold.  A
replacement must remeasure the prototype form and the replacement on the same
Torch-Spyre SHA, native extension, DeepTools stack, AIU, tensors, and protocol.
Structure, correctness, and the matched relative comparison are the oracle.

## What the oracle does not prove

- The prototype architecture is suitable for production.
- The adapter boundary is strategy-neutral.
- The flat-reduction changes preserve every existing topology.
- The post-divider ownership rewrite is the correct scheduling abstraction.
- Grouped, active-dense, or indexed binding is implemented.
- Router-logit computation or complete model latency is included.
- A literal 42.444 ms result must survive unrelated compiler evolution.

## Freeze rules

1. Do not add commits to the source tag or move either tag.
2. Do not amend an artifact covered by the evidence tag.
3. Put architecture contracts and future analysis on the evidence branch after
   the frozen tag or on a docs-only branch.
4. Start production implementation only after the Phase 1 sign-off gate.
5. If the oracle must change, create prototype-v0.1 with new identities and an
   explicit reason; never rewrite prototype-v0.

## Verification

```text
git verify-tag moe-asgemm-prototype-v0-source
git verify-tag moe-asgemm-prototype-v0-evidence
git rev-parse moe-asgemm-prototype-v0-source^{}
git rev-parse moe-asgemm-prototype-v0-evidence^{}
shasum -a 256 -c moe_asgemm/SHA256SUMS
```

The machine-readable companion is
`moe_asgemm/prototype_v0_manifest.json`.
