# Artifacts

The branch retains a compact evidence set. Large generated Spyre binaries and
tensor payloads remain outside Git; their original run directories are listed
as plain provenance pointers.

## Reduced C1 compile evidence

Path:

```text
moe_asgemm/artifacts/c1_compile
```

Contents include the generated wrapper, accepted bundle, identity contribution
SDSC, accumulator-add SDSC, and compile result for `E=2,T=64,H=64,F=64,C=1`.

Key bundle property:

```text
one affine map s0 + 128*d0
arg2 gate weight -> gate BMM
arg3 up weight   -> up BMM
arg4 down weight -> down BMM
arg5 alpha       -> post-down weighting
```

## Reduced C1 device correctness

Path:

```text
moe_asgemm/artifacts/c1_correctness
```

The same callable and bundle passed two distinct nonbinary alpha payloads.
This evidence closes the earlier defect where the expert loop reused expert 0
on both iterations.

## Full-bank four-AIU timing

Path:

```text
moe_asgemm/artifacts/fullbank_timing
```

Files:

```text
comparison.json                validated aggregate comparison
cdx_result.json                PCI 0000:ac:00.0
clc_result.json                PCI 0000:ba:00.0
current_result.json            PCI 0000:bb:00.0
dev_result.json                PCI 0000:ab:00.0
tested_probe.py                exact measured probe
cdx_bundle/bundle.mlir         representative bundle
cdx_bundle/sdsc_0.json ...     representative SDSCs
```

The representative bundle SHA-256 is:

```text
976e5c8101370a6f482247652b31ec81c5be55c2419011b06746000693fd1727
```

The exact probe is byte-identical to:

```text
experiments/dasx_shared_lhs_c32_schedule_probe.py
SHA-256 c634b030197f27cbd75bc7a37b9fd12b0fbce5e66de42b99dded6a7695a33a7a
```

## Original retained run roots

These are plain paths, not hyperlinks:

```text
moe-execution-proof/local_runs/dense_fair_activation_stationary_timing_20260816_01
moe-unit-reduction-collapse/artifacts/flat-e2-t64-unit-expert-affine-backend-compile-accepted-19
moe-unit-reduction-collapse/artifacts/flat-e2-t64-unit-expert-affine-device-correctness-accepted-20
moe-execution-proof/local_runs/dense_pr293_step2_realshape_20260816_02
moe-execution-proof/local_runs/dense_shared_lhs_ddl_overlay_c1_20260816_01
```

## Integrity

`moe_asgemm/SHA256SUMS` covers the compact branch-owned artifact set and the
three probes. Run from the repository root:

```text
shasum -a 256 -c moe_asgemm/SHA256SUMS
```
