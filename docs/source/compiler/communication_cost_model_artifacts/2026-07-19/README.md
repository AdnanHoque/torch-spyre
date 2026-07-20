# Communication cost-model evidence, 2026-07-19

This directory is the compact reviewer-facing evidence package for the LX ring
measurement, Flash no-SHUFFLE oracle, and joint-placement conclusions.
It is an evidence snapshot and index, not a self-contained reproduction archive:
the original compiler closures, bundles, traces, and acquisition scripts remain
in the historical experiment roots named inside the copied result records.

The top-level compiler documents are:

- `lx_ring_measurement_methodology.md`: results, first-principles procedure,
  clock treatment, Flash oracle definition, and remaining experiments;
- `joint_ring_placement.md`: what joint placement is and a production design;
- `communication_cost_model.md`: the requested cost-model draft.

## Revision scope

The promoted measurements use:

```text
Torch base       2a20cf3b7ac8aadf629314e40e5059ad82471911
Torch measured   24adc85c04da91d61b13b295d6092438cf2029b4
Deeptools        19280fd7c6bbd91000c63c2a6719a0253e513f4a
LLVM             22.1.3
```

The documentation branch itself starts from the later Torch feature head
`8e8324febe7bb6b266652b9aeda3c778e3b22935`. No claim of performance equivalence
between that rewrite and the measured tree is implied.

## Contents

### `direct_lx/`

- Exact 512 KiB one-way and balanced-duplex results from the primary CDX run.
- Matching 512 KiB replications from the current and CLC pods.
- Exact 128 and 256 KiB one-way results and the three-size linear fit.
- The correctness-gated eight-hop 512 KiB result.

The result files preserve timing scope, device-event inventory checks,
correctness gates, byte scopes, bundle/runtime hashes, and the original
historical acquisition paths.

### `flash_oracle/`

- The five-triple primary Flash report and gate summary.
- A compact package summary with immutable toolchain pins.
- The independent three-triple cross-node replay and its compatibility report.

The cross-node replay is descriptive only and is not pooled with the primary
campaign.

### `joint_placement/`

- Offline permutation and route-load audit.
- Timing and structural preregistrations.
- Full timing and structural reports plus terminal success records.
- An independent audit of the paired timing claim.

The experimental placement patches are intentionally not included as production
code. They were test controls; the production design requires a typed automatic
planner and fail-closed backend contract.

### `provisional_hbm/`

- A salvaged all-32-core dependent memory-roundtrip result.

This datapoint uses the exact Torch/Deeptools revisions but an LLVM 20 helper and
a different all-core scope. It is not a matched denominator for the direct
single-stream LX measurement.

## Clock provenance

The original 7.38 MB hardware console log is excluded because it contains card
serial, wafer, and ECID data. The preserved log has SHA-256
`994e254f9ef4e74d1c6e3e19dc3224fbbd2fb49df5f1751a431f28db4e81b594`.
Its clock readback reported RPD 1100 MHz, RNG 1100 MHz, SOC 1000 MHz, and DDR
6400 MHz on the same historical CDX PF. `CLOCK_READBACK_SANITIZED.txt` preserves
the exact relevant readback and source-line range without device identifiers.

That readback was not simultaneous with the promoted timing process, and the
SPad-ring cycle has not been formally bound to the RNG domain. Raw GB/s is
therefore canonical; the 1.1 GHz efficiencies are explicit conditional
calculations.

## Deliberate exclusions

The package excludes:

- Kineto and PyTorch traces;
- output tensors, binaries, shared libraries, compiler caches, and bundles;
- raw device event inventories already summarized by result JSONs;
- build, console, dataop, and generation logs;
- tar archives and duplicate frozen runtime closures;
- the identifier-bearing raw clock log; and
- historical direct-LX results affected by an incorrect byte numerator or
  incomplete/overlapping correctness windows.

`SHA256SUMS` covers the byte-for-byte copied immutable evidence. `SUMMARY.json`,
`PROVENANCE.json`, and `CLOCK_READBACK_SANITIZED.txt` are curated indexes or
extracts and are reviewed as source text.
