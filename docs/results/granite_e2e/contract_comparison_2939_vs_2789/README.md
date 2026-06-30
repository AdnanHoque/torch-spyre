# Frontend/Backend Contract Comparison: PR #2939 vs PR #2789

This directory contains compact SuperDSC contract excerpts for the same logical
resident-scatter edge:

```text
producer: first SwiGLU projection `batchmatmul`
producer split: {mb:4, out:8, in:1}
consumer: `neg`
consumer split: {mb:32, out:1}
tensor edge: producer output `buf0` -> consumer input `buf0`
communication class: scatter / ownership permutation
```

The raw PR #2789 source artifact is from commit:

```text
db130e5b62e1c3e709216134cf4e59f6f9e2b7e4
docs/source/compiler/lx_relayout_stcdp_swiglu_artifacts_2026_06_22/latest_profile_fixed_branch/raw_sdsc/stcdp_enabled/branch-baseline/inductor-cache/inductor-spyre/sdsc_fused_linear_mul_silu_split_with_sizes_0_u7fv9_a0/sdsc_2.json
```

The full raw file is about 1.1 MB for this one consumer SDSC, so this directory
stores a focused excerpt instead of vendoring the entire generated SDSC.

## Files

| file | approach | what to inspect |
|---|---|---|
| `pr2789_explicit_stcdp_contract_excerpt.json` | PR #2789 explicit movement | `datadscs_`, `STCDPOpLx`, `rangedLxRemap`, `movementRanges`, `coreIdToDscSchedule` |
| `pr2939_dldsc_metadata_contract_excerpt.json` | PR #2939 dl-dsc metadata | `scheduleTree_[0].coordinates_.coreIdToWkSlice_`, no explicit `datadscs_` movement rows |

## What The Difference Proves

PR #2789 makes Torch hand Deeptools a concrete movement plan:

```text
move this source core/address/logical slice
to this destination core/address/logical slice
schedule these STCDPOpLx rows before the consumer compute row
```

That is visible in:

- `datadscs_`
- `op.name = STCDPOpLx`
- `op.rangedLxRemap.movementRanges`
- `coreIdToDscSchedule`

PR #2939 makes Torch hand Deeptools a tensor-distribution contract:

```text
the input tensor is already resident in LX
its producer ownership is this coreIdToWkSlice map
the consumer compute split is {mb:32,out:1}
backend should derive the needed relayout from the mismatch
```

That is visible in:

- consumer `scheduleTree_` input allocation;
- `component_ = lx`;
- `coordinates_.coreIdToWkSlice_` populated with the producer's `{mb:4,out:8}`
  ownership;
- no `datadscs_` / no frontend-authored physical transfer rows.

## Scaling Implication

The PR #2789 raw SDSC for this single `batchmatmul -> neg` edge contains:

```text
datadsc rows: 5
total ranged movement groups: 524
raw sdsc_2.json size: ~1.1 MB
```

That is fine for a narrow proof. It is the scaling concern for larger
communications: all-gather, multicast, and reductions can grow with the number
of source shards, destination shards, and chunks if the frontend emits physical
movement rows.

The PR #2939 contract is compact: it records the producer tensor distribution
and the consumer compute distribution. Backend expansion can still become large
if implemented naively, but the frontend/backend interface itself stays small
and backend-owned.

## Bottom Line

For this simple scatter edge, both contracts can express the same intent.

PR #2789 expresses it as an explicit transfer schedule.

PR #2939 expresses it as a dl-dsc coordinate mismatch and leaves synthesis to
Deeptools.

That is why PR #2789 is useful for forcing experiments, while PR #2939 is the
cleaner long-term production contract.
