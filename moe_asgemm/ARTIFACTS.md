# Artifacts

The branch retains a compact evidence set. Large generated Spyre binaries and
tensor payloads remain outside Git; their original run directories are listed
as plain provenance pointers.

## Reduced C1 compile evidence

Path:

```text
moe_asgemm/artifacts/c1_compile
```

Contents include the generated wrapper, accepted bundle, complete `sdsc_0.json`
through `sdsc_14.json` set, and compile result for
`E=2,T=64,H=64,F=64,C=1`. The complete SDSC set lets the strict structural
gate independently verify every loop operation.

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

## Clean-source reproduction

Paths:

```text
moe_asgemm/artifacts/clean_reproduction/cdx
moe_asgemm/artifacts/clean_reproduction/clc
```

Each directory contains:

```text
compile_result.json
generated_module.py
bundle.mlir
sdsc_0.json through sdsc_11.json
```

These were generated from a clean checkout of branch head `6dd7132b`. Both
AIUs emitted the representative bundle hash above, completed 540 timing
records and 900 measured calls, and passed all structure and correctness
checks. The tensor payloads and backend binaries are intentionally not stored
in Git.

## Decomposition controls

Path:

```text
moe_asgemm/artifacts/decomposition
```

Contents include the five-point expert sweep, matched E2/E32 full-graph
component substitutions, standalone gate/down matmul proxies, generated source
for every retained control, and `analysis.json`.

The controls are deliberately compact. Correctness tensors, backend binaries,
and failed HBM-spilling leaf attempts remain outside Git.

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
three probes, including the clean-source reproduction. Run from the repository
root:

```text
shasum -a 256 -c moe_asgemm/SHA256SUMS
```
